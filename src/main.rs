
mod nseq_tokenizer;

use ocl::{Buffer, ProQue};

use nseq_tokenizer::ByteTokenizer;

pub static MOST_COMMON_NSEQ_KERNEL: &'static str = include_str!("./kernel.cl");

pub type ErrBox = Box<dyn std::error::Error>;

pub const NSEQ_SIZE: usize = 2;

fn main() -> Result<(), ErrBox> {

    let s = "rabarbar";

    let s_preprocessed: Vec<u32> = s.chars().map(|c| c as u32).collect();

    let pro_que = ProQue::builder().src(MOST_COMMON_NSEQ_KERNEL).dims(s_preprocessed.len()).build()?;

    let txt_buf = pro_que.buffer_builder().copy_host_slice(s_preprocessed.as_slice()).build()?;
    let nseq_counts_buf = pro_que.create_buffer::<u32>()?;
    let unique_nseqs_buf: Buffer<u32> = pro_que.buffer_builder().len(NSEQ_SIZE * s_preprocessed.len()).build()?;

    let kernel = pro_que
        .kernel_builder("most_common_nseq")
        .arg(NSEQ_SIZE as u32)
        .arg(&txt_buf)
        .arg(&nseq_counts_buf)
        .arg(&unique_nseqs_buf)
        .build()?;

    unsafe {
        kernel.enq()?;
    }


    Ok(())
}
