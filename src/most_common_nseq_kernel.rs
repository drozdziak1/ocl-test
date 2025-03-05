use lazy_static::lazy_static;
use ocl::ProQue;

use std::sync::{Arc, Mutex};

use crate::ErrBox;

pub const MAX_WORK_SIZE: usize = 32;

pub static MOST_COMMON_NSEQ_KERNEL: &'static str = include_str!("./kernels/most_common_nseq.cl");

lazy_static! {
    static ref PROQUE: Arc<Mutex<ProQue>> = {
        let pq = ProQue::builder()
            .src(MOST_COMMON_NSEQ_KERNEL)
            .build()
            .expect("FATAL: Failed to initialize ProQue");
        Arc::new(Mutex::new(pq))
    };
}

pub fn most_common_nseqs<const N: usize>(txts: Vec<String>) -> Result<Vec<Option<String>>, ErrBox> {
    let mut pq = PROQUE.lock().expect("FATAL: could not lock the ProQue");

    let mut ret = vec![];

    for chunk in txts.chunks(MAX_WORK_SIZE) {
        let work_size = chunk.len();

        let txt_size = chunk.iter().fold(0, |acc, txt| {
            let txt_len = txt.chars().count();

            if txt_len > acc {
                txt_len
            } else {
                acc
            }
        });

        let mut txts_u32: Vec<u32> = vec![0u32; work_size * txt_size];

        for (i, txt) in chunk.iter().enumerate() {
            let mut txt_iter = txt.chars();
            for j in 0..txt_size {
                if let Some(c) = txt_iter.next() {
                    txts_u32[i * txt_size + j] = c as u32;
                } else {
                    break;
                }
            }
        }

	dbg!(&txts_u32);

	pq.set_dims(txt_size);

        let txts_buf = pq.buffer_builder().len(txt_size * work_size).copy_host_slice(&txts_u32).build()?;

        let nseq_counts_buf = pq
            .buffer_builder()
            .len(work_size * txt_size)
            .fill_val(0u32)
            .build()?;

        let unique_nseqs_buf = pq
            .buffer_builder()
            .len(work_size * txt_size * N)
            .fill_val(0u32)
            .build()?;

        let most_common_nseqs_buf = pq
            .buffer_builder()
            .len(work_size * N)
            .fill_val(0u32)
            .build()?;

        let kernel = pq
            .kernel_builder("most_common_nseq")
            .arg(N as u32)
            .arg(&txts_buf)
            .arg(work_size as u32)
            .arg(txt_size as u32)
            .arg(&nseq_counts_buf)
            .arg(&unique_nseqs_buf)
            .arg(&most_common_nseqs_buf)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        let mut most_common_nseqs = vec![0u32; most_common_nseqs_buf.len()];

        most_common_nseqs_buf.read(&mut most_common_nseqs).enq()?;

	dbg!(&most_common_nseqs);

        for nseq in most_common_nseqs.chunks(N) {
            let elem = if nseq.contains(&0) {
                None
            } else {
                let mut s = String::with_capacity(N);
                for c in nseq {
                    s.push(
                        char::from_u32(*c).ok_or(format!("Could not convert {} to unicode", c))?,
                    );
                }
                Some(s)
            };
            ret.push(elem);
        }
    }

    Ok(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_most_common_nseqs_trivial() -> Result<(), ErrBox> {

	let input = vec![
	    "peepee poopoo".to_owned(),
	    "poopoo peepee".to_owned(),
	    "menace".to_owned(),
	    "ppppepepep".to_owned(),
	    ];

	let result = most_common_nseqs::<2>(input)?;

	assert_eq!(result, vec![
	    Some("pe".to_owned()),
	    Some("po".to_owned()),
	    None,
	    Some("pp".to_owned()),
	]);
        
	Ok(())
    }
}
