use std::collections::HashMap;

use ocl::ProQue;

pub static KERNEL: &'static str = include_str!("./kernel.cl");

pub type ErrBox = Box<dyn std::error::Error>;

/// Tokens consist of N components. Each of them is either another
/// token with its own components, or a literal character
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum TokenComponent<const N: usize> {
    Tok(Token<N>),
    Char(char),
}

/// Token representation parametrized by n-sequence size for merging
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Token<const N: usize> {
    id: usize,
    components: [Box<TokenComponent<N>>; N],
}

pub struct ByteTokenizer<const N: usize = 2> {
    tokens: HashMap<[TokenComponent<N>; N], Token<N>>,
    next_free_id: usize,
}

pub fn most_common_nsequence<const N: usize>(tok_comps: Vec<TokenComponent<N>>) -> Option<[TokenComponent<N>; N]> {
    let mut acc = HashMap::new();

    let mut max_nseq = None;
    let mut max_nseq_appearances = 1;

    for nseq in tok_comps.windows(N) {
	let entry = acc.entry(nseq).or_insert(0);
	*entry += 1;

	if *entry > max_nseq_appearances {
	    max_nseq = Some(nseq.try_into().expect("dupsko"));
	    max_nseq_appearances = *entry;
	}
    }

    max_nseq
}

impl<const N: usize> ByteTokenizer<N> {
    pub fn new() -> Self {
        ByteTokenizer {
            tokens: Default::default(),
            next_free_id: 0,
        }
    }

    pub fn tokenize(&self, s: String) -> Vec<TokenComponent<N>> {
        let mut s_tok_comps: Vec<_> = s.chars().map(|c| TokenComponent::Char(c)).collect();

        loop {
            let mut n_replacements = 0;
            let mut i: usize = 0;

            let mut pass_result: Vec<TokenComponent<N>> = Vec::new();

            while let Some(nseq) = s_tok_comps.get(i..i + N) {
                if let Some(tok) = self.tokens.get(nseq) {
                    pass_result.push(TokenComponent::Tok(tok.clone()));
                    n_replacements += 1;
                    i += N;
                } else {
                    pass_result.push(nseq[0].clone());
                    i += 1;
                }
            }

            if i < s_tok_comps.len() {
                pass_result.extend_from_slice(&s_tok_comps[i..])
            }

            if n_replacements == 0 {
                return pass_result;
            }

            s_tok_comps = pass_result;
        }
    }

    /// Constructs new tokens from most common n-sequences in the string.
    pub fn ingest(&mut self, mut s: Vec<char>) -> usize {
        let loop_bounds = s.len() - N + 1;

        let mut i: usize = 0;

        let mut after_pass_accum = Vec::new();

        loop {}
    }
}

fn main() -> Result<(), ErrBox> {
    let pro_que = ProQue::builder().src(KERNEL).dims(1 << 6).build()?;

    let buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("add")
        .arg(&buffer)
        .arg(10.0f32)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    println!("buffer len is {}", buffer.len());

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq()?;

    println!("The value at index [{}] is now '{}'!", 13, vec[13]);
    Ok(())
}
