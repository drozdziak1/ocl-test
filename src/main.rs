mod nseq_tokenizer;

use ocl::{Buffer, Platform, ProQue};

use nseq_tokenizer::NSeqTokenizer;

pub static COUNT_UNIQUE_NSEQS_KERNEL: &'static str = include_str!("./kernel.cl");

pub type ErrBox = Box<dyn std::error::Error>;

pub const NSEQ_SIZE: usize = 2;

fn main() -> Result<(), ErrBox> {
    let s = "rabarbar jest kurwa tak bardzo najsmaczniejszym czymś że o ja pierdolę";
    Ok(())
}
