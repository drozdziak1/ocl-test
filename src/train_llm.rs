mod nseq_tokenizer;
mod transformer;
mod util;

use nseq_tokenizer::{NSeqTokenizer, TokenizerExport};
use transformer::Transformer;
use util::{init_log, ErrBox};

use clap::Parser;

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
    #[arg(short, long, default_value_t = 0.001)]
    lr: f32,

    /// How many samples to train on
    #[arg(short, long, default_value_t = 1_000_000)]
    n_samples: u32,

    /// Path to fineweb parquet file
    #[arg(
        short,
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

    for i in 0..cli.n_samples {
	

    }



    Ok(())
}
