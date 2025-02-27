use std::{collections::HashMap, fmt::Debug};

use ocl::ProQue;

pub static KERNEL: &'static str = include_str!("./kernel.cl");

pub type ErrBox = Box<dyn std::error::Error>;

/// Tokens consist of N components. Each of them is either another
/// token with its own components, or a literal character
#[derive(Clone, Eq, PartialEq, Hash)]
pub enum TokenComponent<const N: usize> {
    Tok(Box<Token<N>>),
    Char(char),
}

impl<const N: usize> Debug for TokenComponent<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TokenComponent::*;
        match self {
            Tok(tok) => tok.fmt(f),
            Char(c) => write!(f, "{}", c),
        }
    }
}

/// Token representation parametrized by n-sequence size for merging
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Token<const N: usize> {
    id: usize,
    components: [TokenComponent<N>; N],
}

impl<const N: usize> Debug for Token<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}=(", self.id)?;
        for i in 0..self.components.len() {
            Debug::fmt(&self.components[i], f)?;
            if i < self.components.len() - 1 {
                write!(f, "|")?;
            }
        }
        write!(f, ")")
    }
}

pub struct ByteTokenizer<const N: usize = 2> {
    tokens: HashMap<[TokenComponent<N>; N], Token<N>>,
    next_free_id: usize,
}

pub fn most_common_nsequence<const N: usize>(
    tok_comps: &[TokenComponent<N>],
) -> Option<[TokenComponent<N>; N]> {
    let mut acc = HashMap::new();

    let mut max_nseq = None;
    let mut max_nseq_appearances = 1;

    for nseq in tok_comps.windows(N) {
        let entry = acc.entry(nseq).or_insert(0);
        *entry += 1;

        if *entry > max_nseq_appearances {
            max_nseq = Some(nseq.to_owned().try_into().expect("kurwoooooo"));
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

    pub fn tokenize_string(&self, s: &str) -> Vec<TokenComponent<N>> {
        let components = s.chars().map(|c| TokenComponent::Char(c)).collect();

        self.tokenize(components)
    }

    pub fn tokenize(&self, tok_comps: Vec<TokenComponent<N>>) -> Vec<TokenComponent<N>> {
        let mut pass_input: Vec<_> = tok_comps;

        loop {
            let mut n_replacements = 0;
            let mut i: usize = 0;

            let mut pass_result: Vec<TokenComponent<N>> = Vec::new();

            while let Some(nseq) = pass_input.get(i..i + N) {
                if let Some(tok) = self.tokens.get(nseq) {
                    pass_result.push(TokenComponent::Tok(Box::new(tok.clone())));
                    n_replacements += 1;
                    i += N;
                } else {
                    pass_result.push(nseq[0].clone());
                    i += 1;
                }
            }

            if i < pass_input.len() {
                pass_result.extend_from_slice(&pass_input[i..])
            }

            if n_replacements == 0 {
                return pass_result;
            }

            pass_input = pass_result;
        }
    }

    /// Constructs new tokens from most common n-sequences in the
    /// string, returning the number of new tokens created.
    pub fn ingest(&mut self, s: &str) -> usize {
        let mut s_tok_comps = self.tokenize_string(s);

        let mut n_new_tokens = 0;

        while let Some(nseq) = most_common_nsequence(&s_tok_comps) {
            self.tokens.insert(
                nseq.clone(),
                Token {
                    components: nseq,
                    id: self.next_free_id,
                },
            );

            self.next_free_id += 1;
            n_new_tokens += 1;

            s_tok_comps = self.tokenize(s_tok_comps);
        }

        n_new_tokens
    }
}

fn main() -> Result<(), ErrBox> {
    // let pro_que = ProQue::builder().src(KERNEL).dims(1 << 6).build()?;

    // let buffer = pro_que.create_buffer::<f32>()?;

    // let kernel = pro_que
    //     .kernel_builder("add")
    //     .arg(&buffer)
    //     .arg(10.0f32)
    //     .build()?;

    // unsafe {
    //     kernel.enq()?;
    // }

    // println!("buffer len is {}", buffer.len());

    // let mut vec = vec![0.0f32; buffer.len()];
    // buffer.read(&mut vec).enq()?;

    // println!("The value at index [{}] is now '{}'!", 13, vec[13]);

    let mut t: ByteTokenizer<2> = ByteTokenizer::new();

    let input = "Litwo, ojczyzno moja, Ty jesteś jak zdrowie. Ile Cię trzeba cenić, ten tylko się dowie, kto Cię stracił.";

    let n_new = t.ingest(input);

    println!("Added {} new tokens", n_new);

    let output = t.tokenize_string(input);

    println!("Before: {}, After: {}", input.len(), output.len());

    println!("Tokenized: {:?}", output);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_most_common_nseq() {
        let input: Vec<TokenComponent<3>> = "rabarbar"
            .chars()
            .map(|c| TokenComponent::Char(c))
            .collect();

        let most_common = most_common_nsequence(&input);

        let expected: [TokenComponent<3>; 3] = [
            TokenComponent::Char('b'),
            TokenComponent::Char('a'),
            TokenComponent::Char('r'),
        ];

        assert_eq!(most_common, Some(expected));
    }
}
