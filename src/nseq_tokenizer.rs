//! Crude, generalized byte-pair encoding implementation

use std::{collections::HashMap, fmt::Debug};

/// Used for differentiating UTF32 codepoints from token ids
pub const TOK_ID_OFFSET: u32 = 2_000_000;

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

impl<const N: usize> TokenComponent<N> {
    pub fn unroll(&self) -> String {
        match self {
            Self::Char(c) => format!("{}", c),
            Self::Tok(t) => t.unroll(),
        }
    }

    pub fn as_u32(&self) -> u32 {
	match self {
	    Self::Char(c) => *c as u32,
	    Self::Tok(t) => TOK_ID_OFFSET + t.id as u32,
	}
    }
}

/// Token representation parametrized by how many TokenComponent's are
/// merged during tokenization.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Token<const N: usize> {
    id: usize,
    components: [TokenComponent<N>; N],
}

impl<const N: usize> Token<N> {
    pub fn unroll(&self) -> String {
        let mut ret_accum = String::new();
        for comp in &self.components {
            ret_accum.push_str(&comp.unroll());
        }

        ret_accum
    }

    pub fn as_u32s(&self) -> Vec<u32> {
	let mut ret = Vec::with_capacity(N);

	for comp in self.components.iter() {
	    ret.push(comp.as_u32())
	}

	ret
    }
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

/// Finds the most frequent n-sequence of token components.
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

/// Holds n-sequence to token lookups and tracks token id assignment
pub struct NSeqTokenizer<const N: usize = 2> {
    pub tokens: HashMap<[TokenComponent<N>; N], Token<N>>,
    pub vocab_size: usize,
    next_free_id: usize,
}

impl<const N: usize> NSeqTokenizer<N> {
    pub fn new(vocab_size: usize) -> Self {
        NSeqTokenizer {
            tokens: Default::default(),
            vocab_size,
            next_free_id: 0,
        }
    }

    pub fn tokenize_str(&self, s: &str) -> Vec<TokenComponent<N>> {
        let components = s.chars().map(|c| TokenComponent::Char(c)).collect();

        self.tokenize(components)
    }

    /// Performs generic byte n-sequence tokenization basing off
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

    /// Decodes a tokenized input back into a string
    pub fn untokenize(&self, tok_comps: &[TokenComponent<N>]) -> String {
        let mut ret_accum = String::new();

        for comp in tok_comps {
            ret_accum.push_str(&comp.unroll());
        }

        ret_accum
    }

    /// Constructs new tokens from most common n-sequences in the
    /// string, returning the number of new tokens created.
    pub fn ingest(&mut self, s: &str) -> (usize, bool) {
        let mut s_tok_comps = self.tokenize_str(s);

        let mut n_new_tokens = 0;
        let mut vocab_full = false;

        while let Some(nseq) = most_common_nsequence(&s_tok_comps) {
            if self.tokens.len() >= self.vocab_size {
                vocab_full = true;
                break;
            }

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

        (n_new_tokens, vocab_full)
    }
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

    #[test]
    fn test_tokenize_untokenize_equiv() {
        let mut tokenizer = NSeqTokenizer::<2>::new(50);
        let input = "Litwo, ojczyzno moja, Ty jesteś jak zdrowie. Ile Cię trzeba cenić, ten tylko się dowie, kto Cię stracił.";

        tokenizer.ingest(&input);

        let tokenized = tokenizer.tokenize_str(&input);

        let untokenized = tokenizer.untokenize(&tokenized);

        assert_eq!(input, untokenized);
    }

    #[test]
    fn test_tokenizer_respects_vocab_size() {
        let mut tokenizer = NSeqTokenizer::<2>::new(2);
        let input = "blablabla";

        assert_eq!(tokenizer.ingest(&input), (2, true));

        let bl_comps: Vec<_> = "bl".chars().map(|c| TokenComponent::Char(c)).collect();
        let bla_comps = [
            TokenComponent::Tok(Box::new(tokenizer.tokens[bl_comps.as_slice()].clone())),
            TokenComponent::Char('a'),
        ];

        assert!(tokenizer.tokens.contains_key(bl_comps.as_slice()));
        assert!(tokenizer.tokens.contains_key(bla_comps.as_slice()));
    }
}
