mod nseq_tokenizer;
mod transformer;
mod util;

use clap::Parser;
use log::{debug, info, trace, warn};
use ndarray_rand::rand::seq::IteratorRandom;
use nseq_tokenizer::{NSeqTokenizer, TokenizerExport};
use polars::prelude::*;
use rand::SeedableRng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tokenizers::tokenizer::Tokenizer;

use std::{fs::File, path::PathBuf};

use transformer::{Transformer, TransformerGradients, TransformerTrainData};
use util::{init_log, ErrBox};

#[derive(Parser)]
#[command(version, about)]
pub struct Cli {
    // /// Path to a pre-trained n-sequence tokenizer (see "train_nseq_tokenizer" for details
    // #[arg(short, long, default_value = "./tokenizer.json")]
    // tokenizer_path: PathBuf,
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

    /// How many samples between gradient applications
    #[arg(short = 'B', long, default_value_t = 10)]
    batch_size: u32,

    /// How many samples to train on in total
    #[arg(short, long, default_value_t = 1_000_000)]
    n_samples: u32,

    /// Up to how many tokens to use for each forward pass
    #[arg(short, long, default_value_t = 1_000)]
    sample_size: u32,

    #[arg(short, long, default_value_t = 0xdeadbeef)]
    rng_seed: u64,

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

    let tokenizer =
        Tokenizer::from_pretrained("bert-base-cased", None).expect("Could not load tokenizer");

    let vocab_size = tokenizer.get_vocab_size(true);
    let sample_size = cli.sample_size as usize;
    let batch_size = cli.batch_size as usize;

    let df = LazyFrame::scan_parquet(cli.fineweb_path, Default::default())?
        .select([col("text")])
        .collect()?;

    let col = df.column("text")?.str()?;

    let val_samples_size = if batch_size / 10 > 0 {
        batch_size / 10
    } else {
        1
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(cli.rng_seed);

    let col_sampled = col
        .sample_n(val_samples_size, false, false, Some(cli.rng_seed))
        .expect("Could not sample text column");

    let validation_samples: Vec<Vec<u32>> = (0..val_samples_size)
        .map(|offset| -> Result<Vec<u32>, ErrBox> {
            let txt = col_sampled
                .get(offset)
                .ok_or_else(|| "Could not get record for validation sample")?;

            let encoding = tokenizer
                .encode(txt, false)
                .expect("Could not encode validation txt");

            let sample = encoding
                .get_ids()
                .get(0..sample_size)
                .expect("Could not slice sample_size tokens from validation_encoding")
                .to_owned();

            Ok(sample)
        })
        .collect::<Result<Vec<Vec<u32>>, ErrBox>>()?;

    let mut texts_iter = col.iter().enumerate();

    let mut i = 0;

    // used for averaging
    let mut losses = vec![];

    // Used for batching of gradient applications
    let mut grads_sum: Option<TransformerGradients> = None;

    let mut model = Transformer::new(
        cli.ctx_width as usize,
        vocab_size,
        cli.embed_dim as usize,
        cli.block_count as usize,
        cli.head_count as usize,
        cli.ff_dim as usize,
        &mut rng,
    )?;

    'texts_loop: while let Some((txt_idx, Some(txt))) = texts_iter.next() {
        let encoding = tokenizer.encode(txt, false).expect("Could not encode");
        let tokenized: Vec<_> = encoding.get_ids().to_owned();

        let tokenized_len = tokenized.len();

        // Go to next text when we've used more than current one's amount of tokens
        let mut tokens_sampled = 0;

        while tokens_sampled < tokenized_len {
            if i >= cli.n_samples {
                break 'texts_loop;
            }

            let sample_len = sample_size.min(tokenized_len);

            let start_idx = (0..(tokenized_len - sample_len))
                .choose(&mut rng)
                .unwrap_or(0);

            let sample = &tokenized[start_idx..(start_idx + sample_len)];

            let (x, _last) = sample.split_at(sample_len - 1);

            let (_first, y_gt) = sample.split_at(1);

            let mut train_data = TransformerTrainData::default();

            model.naive_fwd(x, Some(&mut train_data))?;

            let new_grads = model.naive_bwd(&train_data, y_gt)?;

            debug!(
                "txt No.: {:10} | Sample No.: {:6} | idxs: {:5} - {:5} | len: {}| loss: {:.12}",
                txt_idx,
                i,
                start_idx,
                start_idx + sample_len,
                sample_len,
                new_grads.loss,
            );

            losses.push(new_grads.loss);

            if let Some(grads_accum) = grads_sum.as_mut() {
                grads_accum.accum(&new_grads);
            } else {
                grads_sum = Some(new_grads);
            }

            if i % cli.batch_size == 0 {
                let mut val_losses = Vec::with_capacity(batch_size);
                for val_sample in validation_samples.iter() {
                    let (val_x, _last) = val_sample.split_at(sample_size - 1);

                    let (_first, val_y_gt) = val_sample.split_at(1);

                    let mut val_train_data = TransformerTrainData::default();

                    model.naive_fwd(val_x, Some(&mut val_train_data))?;

                    let val_grads = model.naive_bwd(&val_train_data, val_y_gt)?;

                    val_losses.push(val_grads.loss);
                }

                let avg_val_loss =
                    val_losses.iter().fold(0.0, |acc, x| acc + x) / val_losses.len() as f32;

                let avg_t_loss = losses.iter().fold(0.0, |acc, x| acc + x) / losses.len() as f32;
                let min_t_loss = losses
                    .iter()
                    .fold(std::f32::MAX, |acc, x| if *x < acc { *x } else { acc });
                let max_t_loss = losses
                    .iter()
                    .fold(std::f32::MIN, |acc, x| if *x > acc { *x } else { acc });

                info!(
                    "txt No.: {:10} | Sample No.: {:6} | V loss: {:11.9} | avg T loss: {:11.9} | min: {:11.9} | max: {:11.9}",
                    txt_idx, i, avg_val_loss, avg_t_loss, min_t_loss, max_t_loss
                );

                // Apply gradients
                if let Some(grads_sum) = grads_sum.as_mut() {
                    grads_sum.to_mean(batch_size);
                    model.apply_grads(&grads_sum, cli.lr * batch_size as f32)?;
                } else {
                    warn!("grads_accum still empty after batch evaluated");
                }

                losses = vec![];
                grads_sum = None;
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
