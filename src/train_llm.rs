mod nseq_tokenizer;
mod transformer;
mod util;

use log::{debug, info, trace, warn};
use ndarray_rand::rand::seq::IteratorRandom;
use nseq_tokenizer::{NSeqTokenizer, TokenizerExport};
use transformer::{Transformer, TransformerTrainData};
use util::{init_log, ErrBox};

use clap::Parser;
use polars::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use std::{fs::File, path::PathBuf};

#[derive(Parser)]
#[command(version, about)]
pub struct Cli {
    /// Path to a pre-trained n-sequence tokenizer (see "train_nseq_tokenizer" for details
    #[arg(short, long, default_value = "./tokenizer.json")]
    tokenizer_path: PathBuf,

    /// Context size
    #[arg(short, long, default_value_t = 10_000)]
    ctx_width: u32,

    /// Embedding table entry length
    #[arg(short, long, default_value_t = 64)]
    embed_dim: u32,

    /// How many heads per block (must divide embed_dim equally)
    #[arg(short = 'H', long, default_value_t = 8)]
    head_count: u32,

    /// how many transformer blocks to put in the model
    #[arg(short, long, default_value_t = 16)]
    block_count: u32,

    /// How many neurons to put into feedforward
    #[arg(short, long, default_value_t = 128)]
    ff_dim: u32,

    /// Learning rate
    #[arg(short, long, default_value_t = 3e-5)]
    lr: f32,

    /// How many samples to train on
    #[arg(short, long, default_value_t = 1_000_000)]
    n_samples: u32,

    /// Up to how many tokens to use for each forward pass
    #[arg(short, long, default_value_t = 1_000)]
    sample_size: u32,

    /// Path to fineweb parquet file
    #[arg(
        short = 'd',
        long = "fineweb",
        default_value = "/home/drozdziak1/Documents/datasets/fineweb/000_00000.parquet"
    )]
    fineweb_path: PathBuf,

    /// Where to write model checkpoints during training
    #[arg(short, long, default_value = "./target/llm-checkpoints")]
    output_dir: PathBuf,
}

fn main() -> Result<(), ErrBox> {
    init_log();

    let cli = Cli::parse();

    let tokenizer_file = File::open(cli.tokenizer_path)?;

    let tokenizer_export: TokenizerExport = serde_json::from_reader(tokenizer_file)?;

    let tokenizer = NSeqTokenizer::<2>::import(&tokenizer_export)?;

    let vocab_size = tokenizer
        .next_free_id
        .read()
        .expect("Could not lock tokenizer next_free_id")
        .clone() as usize;

    let mut t = Transformer::new(
        cli.ctx_width as usize,
        vocab_size,
        cli.embed_dim as usize,
        cli.block_count as usize,
        cli.head_count as usize,
        cli.ff_dim as usize,
    )?;

    let df = LazyFrame::scan_parquet(cli.fineweb_path, Default::default())?
        .select([col("text")])
        .collect()?;

    let mut rng = rand::thread_rng();

    let mut texts_iter = df.column("text")?.str()?.iter().enumerate();

    let sample_size = cli.sample_size as usize;

    let mut i = 0;

    // used for averaging
    let mut losses = vec![];

    'texts_loop: while let Some((txt_idx, Some(txt))) = texts_iter.next() {
        let tokenized: Vec<_> = tokenizer
            .tokenize_str(txt)
            .par_iter()
            .map(|comp| comp.as_u32())
            .collect();

        let tokenized_len = tokenized.len();

        // Go to next text when we've used more than current one's amount of tokens
        let mut tokens_sampled = 0;

        while tokens_sampled < tokenized_len {
            if i >= cli.n_samples {
                break 'texts_loop;
            }
            // let sample_len = (2..sample_size)
            //     .choose(&mut rng)
            //     .expect("could not choose 2..sample_size")
            //     .min(tokenized_len);

            let sample_len = sample_size.min(tokenized_len);

            let start_idx = (0..(tokenized_len - sample_len))
                .choose(&mut rng)
                .unwrap_or(0);

            let sample = &tokenized[start_idx..(start_idx + sample_len)];

            let (x, _last) = sample.split_at(sample_len - 1);

            let (_first, y_gt) = sample.split_at(1);

            let mut loss = 0.0;
            // for _ in 0..2 {
                let mut train_data = TransformerTrainData::default();

                t.naive_fwd(x, Some(&mut train_data))?;

                loss = t.naive_bwd(cli.lr, &train_data, y_gt)?;

                debug!(
		    "txt No.: {:10} | Sample No.: {:6} | idxs: {:5} - {:5} | len: {}| loss: {:.12}",
		    txt_idx,
		    i,
		    start_idx,
		    start_idx + sample_len,
		    sample_len,
		    loss
		);
            // }

            losses.push(loss);

            if i % 10 == 0 {
                let avg = losses.iter().fold(0.0, |acc, x| acc + x) / losses.len() as f32;
                let min = losses
                    .iter()
                    .fold(std::f32::MAX, |acc, x| if *x < acc { *x } else { acc });
                let max = losses
                    .iter()
                    .fold(std::f32::MIN, |acc, x| if *x > acc { *x } else { acc });

                info!(
                    "txt No.: {:10} | Sample No.: {:6} | avg: {:11.9} | min: {:11.9} | max: {:11.9}",
                    txt_idx, i, avg, min, max
                );

                losses = vec![];
            }

            tokens_sampled += sample_len;
            i += 1;
        }
    }

    if i < cli.n_samples {
        warn!("ran out of training data before requested number of training examples!");
    }

    Ok(())
}
