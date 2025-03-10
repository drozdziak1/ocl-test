mod nseq_tokenizer;

use clap::Parser;
use dialoguer::Confirm;
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use log::{info, LevelFilter};
use polars::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use std::{
    env,
    fs::File,
    io::ErrorKind,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

use nseq_tokenizer::NSeqTokenizer;

pub type ErrBox = Box<dyn std::error::Error>;

pub const NSEQ_SIZE: usize = 2;

#[derive(Parser)]
#[command(version, about)]
pub struct Cli {
    #[arg(
        long = "fineweb",
        default_value = "/home/drozdziak1/Documents/datasets/fineweb/000_00000.parquet"
    )]
    /// Path to a Fineweb parquet file
    fineweb_path: PathBuf,
    #[arg(short, long, default_value = "./tokenizer.json")]
    /// Where to write the tokenizer after training
    output: PathBuf,
    #[arg(short = 's', long, default_value_t = 100_000)]
    /// At how many unique tokens should we stop training the tokenizer
    vocab_size: u32,
}

fn main() -> Result<(), ErrBox> {
    init_log();

    let cli = Cli::parse();

    match File::open(&cli.output) {
        Ok(_f) => {
            let overwrite_ok = Confirm::new()
                .with_prompt(format!(
                    "{} appears to exist. Do you want to overwrite?",
                    cli.output.display()
                ))
                .interact()?;
            if !overwrite_ok {
                return Ok(());
            }
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(other) => {
            return Err(other.into());
        }
    }

    let df = LazyFrame::scan_parquet(cli.fineweb_path, Default::default())?
        .select([col("text")])
        .collect()?;

    let vocab_size = cli.vocab_size;

    let t = Arc::new(NSeqTokenizer::<2>::new(vocab_size));

    let idx = Arc::new(Mutex::new(0u32));

    let n_rows = df.column("text")?.len();

    let start_time = Instant::now();
    let duration = Arc::new(Mutex::new(None));

    let pb_tokenizer_capacity = ProgressBar::new(vocab_size as u64);
    let pb_dataset_rows = ProgressBar::new(n_rows as u64);

    pb_tokenizer_capacity.set_style(ProgressStyle::with_template(
        "\n[{elapsed_precise}] {msg:>24} {wide_bar} {human_pos:>10}/{human_len}",
    )?);
    pb_dataset_rows.set_style(ProgressStyle::with_template(
        "{msg:>35} {wide_bar} {human_pos:>10}/{human_len}",
    )?);

    pb_tokenizer_capacity.set_message("Tokens created");
    pb_dataset_rows.set_message("Dataset rows processed");

    let mpb = MultiProgress::new();
    mpb.add(pb_tokenizer_capacity.clone());
    mpb.add(pb_dataset_rows.clone());

    df.column("text")?
        .str()?
        .par_iter()
        .progress_with(pb_dataset_rows)
        .for_each_with(
            (
                t.clone(),
                idx.clone(),
                duration.clone(),
                pb_tokenizer_capacity.clone(),
            ),
            |(t, idx, duration, pb), txt| {
                if let Some(txt) = txt {
                    if t.is_full() {
                        return;
                    }

                    let my_idx: u32;

                    {
                        // Temporarily lock
                        let mut idx = idx.lock().expect("could not lock idx!");

                        // assign a value to this thread
                        my_idx = *idx;

                        // increment for next lock caller
                        *idx += 1;

                        // Drop lock at scope end
                    }

                    let (n_new_toks, full) = t.ingest(txt);

                    if full {
                        let mut duration = duration.lock().expect("duration lock");
                        if duration.is_none() {
                            *duration = Some(start_time.elapsed());
                            info!("Tokenizer is full after {} iterations", my_idx);
                        }
                    } else {
                        if my_idx % 100 == 0 {
                            pb.println(format!(
                            "Row {:7}/{}: ingested {:5} new tokens from {:6} chars at {:.7} toks/char",
                            my_idx,
                            n_rows,
                            n_new_toks,
                            txt.len(),
                            n_new_toks as f64 / txt.len() as f64,
                        ));
                        }

                        pb.inc(n_new_toks as u64);
                    }
                }
            },
        );

    pb_tokenizer_capacity.finish();

    let duration = duration
        .lock()
        .expect("duration lock")
        .expect("duration still None after processing");

    println!("Processing took {}s", duration.as_secs());

    let export = t.export();

    let f = File::create(&cli.output)?;

    println!("Exporting tokenizer to {}...", cli.output.display());
    serde_json::to_writer_pretty(&f, &export)?;

    Ok(())
}

/// Init logging at info level by default
fn init_log() {
    match env::var("RUST_LOG") {
        Ok(_value) => env_logger::init(),
        Err(_e) => env_logger::Builder::new()
            .filter_level(LevelFilter::Info)
            .init(),
    }
}
