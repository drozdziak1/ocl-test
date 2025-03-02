mod nseq_tokenizer;

use ocl::{Buffer, Platform, ProQue};

use nseq_tokenizer::ByteTokenizer;

pub static COUNT_UNIQUE_NSEQS_KERNEL: &'static str = include_str!("./kernel.cl");

pub type ErrBox = Box<dyn std::error::Error>;

pub const NSEQ_SIZE: usize = 2;

fn main() -> Result<(), ErrBox> {
    let s = "rabarbar jest kurwa tak bardzo najsmaczniejszym czymś że o ja pierdolę";

    println!("host: work size is {}", s.len());

    let s_preprocessed: Vec<u32> = s.chars().map(|c| c as u32).collect();

    let pro_que = ProQue::builder()
        .src(COUNT_UNIQUE_NSEQS_KERNEL)
        .dims(s_preprocessed.len())
        .build()?;

    let txt_buf = pro_que
        .buffer_builder()
        .copy_host_slice(s_preprocessed.as_slice())
        .build()?;
    let nseq_counts_buf = pro_que.create_buffer::<u32>()?;
    let unique_nseqs_buf: Buffer<u32> = pro_que
        .buffer_builder()
        .len(NSEQ_SIZE * s_preprocessed.len())
        .build()?;
    let nseq_slot_locks: Buffer<i32> = pro_que
        .buffer_builder()
        .len(s_preprocessed.len())
        .fill_val(-1)
        .build()?;

    let kernel = pro_que
        .kernel_builder("count_unique_nseqs")
        .arg(NSEQ_SIZE as u32)
        .arg(&txt_buf)
        .arg(&nseq_counts_buf)
        .arg(&unique_nseqs_buf)
        .arg(&nseq_slot_locks)
        .build()?;

    println!("before enq");
    unsafe {
        kernel.enq()?;
    }
    println!("after enq");

    let mut counts: Vec<u32> = vec![0u32; s_preprocessed.len()];

    println!("before nseq_counts read");
    nseq_counts_buf.read(&mut counts).enq()?;
    println!("after nseq_counts read");

    println!("counts: {:?}", counts);

    let mut unique_nseqs: Vec<u32> = vec![0u32; NSEQ_SIZE * s_preprocessed.len()];

    unique_nseqs_buf.read(&mut unique_nseqs).enq()?;

    let nseq_strings: Vec<String> = unique_nseqs
        .chunks(NSEQ_SIZE)
        .filter_map(|chunk| {
            if chunk.contains(&0) {
                None
            } else {
                let s: String = chunk
                    .iter()
                    .map(|num| char::from_u32(*num).expect("Could not convert u32 to char"))
                    .collect();
                Some(s)
            }
        })
        .collect();

    println!("unique_nseqs: {:?}", nseq_strings);

    Ok(())
}
