mod nseq_tokenizer;

use std::{
    sync::{Mutex, RwLock},
    time::Instant,
};

use ocl::{Buffer, Platform, ProQue};

use nseq_tokenizer::NSeqTokenizer;

use polars::prelude::*;
use rayon::iter::ParallelIterator;

pub type ErrBox = Box<dyn std::error::Error>;

pub const NSEQ_SIZE: usize = 2;

fn main() -> Result<(), ErrBox> {
    let df = LazyFrame::scan_parquet(
        "/home/drozdziak1/Documents/datasets/fineweb/000_00000.parquet",
        Default::default(),
    )?
    .select([col("text")])
    .collect()?;

    let vocab_size = 100_000;

    let t = Arc::new(NSeqTokenizer::<2>::new(vocab_size));

    let idx = Arc::new(Mutex::new(0u32));

    let n_rows = df.column("text")?.len();

    let start_time = Instant::now();
    let duration = Arc::new(Mutex::new(None));

    df.column("text")?.str()?.par_iter().for_each_with(
        (t.clone(), idx.clone(), duration.clone()),
        |(t, idx, duration), txt| {
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

                let old_vocab_len = t.tokens.read().expect("tokens read").len();

                let (n_new_toks, full) = t.ingest(txt);

                if full {
                    let mut duration = duration.lock().expect("duration lock");
                    if duration.is_none() {
                        *duration = Some(start_time.elapsed());
                        println!("Tokenizer is full after {} iterations", my_idx);
                    }
                } else {
                    let pct_full = (old_vocab_len + n_new_toks) as f64 / vocab_size as f64 * 100.0;
                    println!(
			"Elapsed: {:5}s Row{:10}/{}: {:4.5}% full, {:.5} toks/char ingested {:5} new tokens from {:8} chars",
			start_time.elapsed().as_secs(),
			my_idx,
			n_rows,
			pct_full,
			n_new_toks as f64 / txt.len() as f64,
			n_new_toks,
			txt.len()
		    );
                }
            }
        },
    );

    let duration = duration
        .lock()
        .expect("duration lock")
        .expect("duration still None after processind");

    println!("Processing took {}s", duration.as_secs());

    let kupsko_tokenized = t.tokenize_str("I like big butts and I cannot lie");

    dbg!(kupsko_tokenized);

    Ok(())
}
