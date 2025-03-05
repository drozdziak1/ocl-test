mod nseq_tokenizer;
mod most_common_nseq_kernel;

use ocl::{Buffer, Platform, ProQue};

use nseq_tokenizer::NSeqTokenizer;

use polars::prelude::*;

pub type ErrBox = Box<dyn std::error::Error>;

pub const NSEQ_SIZE: usize = 2;

fn main() -> Result<(), ErrBox> {
    let mut df = LazyFrame::scan_parquet(
        "/home/drozdziak1/Documents/datasets/fineweb/000_00000.parquet",
        Default::default(),
    )?
    .select([col("text")])
    .collect()?;

    let mut t = NSeqTokenizer::<2>::new(1_000_000);

    let n_rows = df.column("text")?.len();

    for (idx, row) in df.column("text")?.str()?.into_no_null_iter().enumerate() {
        let (n_new_toks, full) = t.ingest(row);
        println!(
            "Row {}/{}: +{} tokens from {} chars ({:.05} tokens/char)",
            idx + 1,
            n_rows,
            n_new_toks,
            row.len(),
	    n_new_toks as f64 / row.len() as f64
        );

        if full {
            println!("Tokenizer is full after {} rows", idx + 1);
            break;
        }
    }

    let kupsko_tokenized = t.tokenize_str("I like big butts and I cannot lie");

    dbg!(kupsko_tokenized);

    Ok(())
}
