//! Crude, generalized byte-pair encoding implementation

use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};

use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::{Arc, RwLock},
};

use crate::ErrBox;

/// Used for differentiating UTF32 codepoints from token ids
pub const TOK_ID_OFFSET: u32 = 2_000_000;

lazy_static! {
    static ref REGEX_ALLOWED: Regex = Regex::new(r"^( ?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+)$")
        .expect("Failed to compile REGEX_ALLOWED");
}

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
            Self::Tok(t) => t.as_u32(),
        }
    }
}

/// Token representation parametrized by how many TokenComponent's are
/// merged during tokenization.
#[derive(Clone, Hash)]
pub struct Token<const N: usize> {
    id: u32,
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

    /// This token's ID + the token ID offset
    pub fn as_u32(&self) -> u32 {
        self.id + TOK_ID_OFFSET
    }

    /// Represent this token's components as u32s
    pub fn as_u32s(&self) -> Vec<u32> {
        let mut ret = Vec::with_capacity(N);

        for comp in self.components.iter() {
            ret.push(comp.as_u32())
        }

        ret
    }
}

impl<const N: usize> PartialEq for Token<N> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<const N: usize> Eq for Token<N> {}

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
pub fn most_common_nsequences<const N: usize>(
    tok_comps: &[TokenComponent<N>],
) -> Vec<([TokenComponent<N>; N], usize)> {
    let mut nseq_acc: HashMap<[TokenComponent<N>; N], usize> = HashMap::new();

    // We turn the whole n-sequence back to a string to check against
    // REGEX_ALLOWED. Once we learn those that don't fit, they are
    // cached here.
    let mut bad_nseqs = HashSet::new();

    for nseq in tok_comps.windows(N) {
        if !bad_nseqs.contains(nseq) {
            let unrolled: String = nseq.iter().map(|comp| comp.unroll()).collect();
            if REGEX_ALLOWED.is_match(&unrolled) {
                let entry = nseq_acc
                    .entry(
                        nseq.to_owned()
                            .try_into()
                            .expect("Somehow the window is not N-sized"),
                    )
                    .or_insert(0);
                *entry += 1;
            } else {
                bad_nseqs.insert(nseq.to_vec());
            }
        }
    }

    let mut nseqs_vec: Vec<([TokenComponent<N>; N], usize)> =
        nseq_acc.into_iter().filter(|(_nseq, n)| *n > 1).collect();

    nseqs_vec.sort_by_key(|(_nseq, n)| *n);

    nseqs_vec
}

/// Holds n-sequence to token lookups and tracks token id assignment
#[derive(Debug)]
pub struct NSeqTokenizer<const N: usize = 2> {
    pub tokens: Arc<RwLock<HashMap<[TokenComponent<N>; N], Token<N>>>>,
    /// Helps elliminate distinct groups of token components that
    /// unroll to the same string during training
    pub tokens_unrolled: Arc<RwLock<HashSet<String>>>,
    pub vocab_size: u32,
    next_free_id: Arc<RwLock<u32>>,
}

impl<const N: usize> NSeqTokenizer<N> {
    pub fn new(vocab_size: u32) -> Self {
        NSeqTokenizer {
            tokens: Default::default(),
            tokens_unrolled: Default::default(),
            vocab_size,
            next_free_id: Arc::new(RwLock::new(0)),
        }
    }

    pub fn import(export: &TokenizerExport) -> Result<Self, ErrBox> {
        let mut tokens = HashMap::new();
        let mut tokens_unrolled = HashSet::new();

        let mut id2token: HashMap<u32, Token<N>> = HashMap::new();

        let mut max_id = 0;

        for (tok_id, tok_comps) in export.tokens.iter() {
            if *tok_id > max_id {
                max_id = *tok_id;
            }

            // Recursively translate this token's components from u32
            // Work item schema:
            // (token ID, token component u32s, components resolved so far)
            let mut work_stack = vec![(tok_id, tok_comps, Vec::with_capacity(N))];

            while let Some((tok_id, tok_comps, mut comps_reconstructed)) = work_stack.pop() {
                for u32_comp in tok_comps
                    .get(comps_reconstructed.len()..)
                    .unwrap_or_default()
                {
                    if *u32_comp >= TOK_ID_OFFSET {
                        if let Some(tok) = id2token.get(u32_comp) {
                            comps_reconstructed.push(TokenComponent::Tok(Box::new(tok.clone())));
                        } else {
                            let comp_comps = export
                                .tokens
                                .get(u32_comp)
                                .ok_or_else(|| format!("Token {} is undefined!", u32_comp))?;

                            // Reinsert the current work since it is not done
                            work_stack.push((tok_id, tok_comps, comps_reconstructed.clone()));

                            // Insert the work for the unknown component token
                            work_stack.push((u32_comp, comp_comps, Vec::with_capacity(N)));
                        }
                    } else {
                        // Leaf case, this one is a char
                        comps_reconstructed.push(TokenComponent::Char(
                            char::from_u32(*u32_comp).ok_or_else(|| {
                                format!("Could not parse {} as unicode", *u32_comp)
                            })?,
                        ));
                    }
                }

                if tok_comps.len() == comps_reconstructed.len() {
                    let comps_array: [TokenComponent<N>; N] =
                        comps_reconstructed.try_into().map_err(|e| {
                            format!(
                                "Could not fit resolved token components into array of {}: {:?}",
                                N, e
                            )
                        })?;

                    let tok = Token {
                        id: tok_id - TOK_ID_OFFSET,
                        components: comps_array.clone(),
                    };

                    id2token.insert(*tok_id, tok.clone());

                    tokens_unrolled.insert(tok.unroll());

                    tokens.insert(comps_array.clone(), tok);
                }
            }
        }

        let next_free_id = max_id + 1;

        Ok(Self {
            tokens: Arc::new(RwLock::new(tokens)),
            tokens_unrolled: Arc::new(RwLock::new(tokens_unrolled)),
            vocab_size: export.vocab_size,
            next_free_id: Arc::new(RwLock::new(next_free_id)),
        })
    }

    pub fn export(&self) -> TokenizerExport {
        let tokens = self.tokens.read().expect("tokens read");

        let export_hashmap: HashMap<_, _> = tokens
            .values()
            .map(|tok| (tok.as_u32(), tok.as_u32s()))
            .collect();

        TokenizerExport {
            n: N as u32,
            tokens: export_hashmap,
            vocab_size: self.vocab_size,
        }
    }

    pub fn is_full(&self) -> bool {
        self.tokens.read().expect("tokens len read").len() as u32 >= self.vocab_size
    }

    pub fn tokenize_str(&self, s: &str) -> Vec<TokenComponent<N>> {
        let components = s.chars().map(|c| TokenComponent::Char(c)).collect();

        self.tokenize(components)
    }

    /// Performs generic byte n-sequence tokenization basing off
    pub fn tokenize(&self, tok_comps: Vec<TokenComponent<N>>) -> Vec<TokenComponent<N>> {
        let mut pass_input: Vec<_> = tok_comps;

        let tokens = self.tokens.read().expect("tokens read");
        loop {
            let mut n_replacements = 0;
            let mut i: usize = 0;

            let mut pass_result: Vec<TokenComponent<N>> = Vec::new();

            while let Some(nseq) = pass_input.get(i..i + N) {
                if let Some(tok) = tokens.get(nseq) {
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
    pub fn ingest(&self, s: &str) -> (usize, bool) {
        let mut s_tok_comps = self.tokenize_str(s);

        let mut total_new_tokens = 0;

        loop {
            let nseqs = most_common_nsequences(&s_tok_comps);

	    // Prevents write-locking self members without anything to do
            if nseqs.is_empty() {
                break;
            }

            // How many times we added a token in this round
            let mut new_tokens_this_pass = 0;

            {
                let mut next_free_id = self.next_free_id.write().expect("next_free_id write");
                let mut tokens = self.tokens.write().expect("tokens write");
                let mut tokens_unrolled =
                    self.tokens_unrolled.write().expect("tokens_unrolled write");

                for (nseq, _n) in nseqs.iter().rev() {
                    if (tokens.len() as u32) >= self.vocab_size {
                        break;
                    }
                    if !tokens.contains_key(nseq) {
                        let unrolled: String = self.untokenize(nseq.as_slice());

                        if !tokens_unrolled.contains(&unrolled) {
                            tokens.insert(
                                nseq.clone(),
                                Token {
                                    components: nseq.clone(),
                                    id: *next_free_id,
                                },
                            );

                            tokens_unrolled.insert(unrolled);

                            *next_free_id += 1;
                            new_tokens_this_pass += 1;

                        }
                    }
                }
            }

            if new_tokens_this_pass == 0 {
                // No work was done, we've added everything
                break;
            }

            total_new_tokens += new_tokens_this_pass;
            s_tok_comps = self.tokenize(s_tok_comps);
        }

        (total_new_tokens, self.is_full())
    }
}

/// Helper struct for serializing NSeqTokenizer
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerExport {
    n: u32,
    vocab_size: u32,
    tokens: HashMap<u32, Vec<u32>>,
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

        let most_common = most_common_nsequences(&input);

        let expected: [TokenComponent<3>; 3] = [
            TokenComponent::Char('b'),
            TokenComponent::Char('a'),
            TokenComponent::Char('r'),
        ];

        assert_eq!(most_common.last(), Some(&(expected, 2)));
    }

    #[test]
    fn test_tokenize_untokenize_equiv() {
        let tokenizer = NSeqTokenizer::<2>::new(50);
        let input = "Litwo, ojczyzno moja, Ty jesteś jak zdrowie. Ile Cię trzeba cenić, ten tylko się dowie, kto Cię stracił.";

        tokenizer.ingest(&input);

        let tokenized = tokenizer.tokenize_str(&input);

        let untokenized = tokenizer.untokenize(&tokenized);

        assert_eq!(input, untokenized);
    }

    #[test]
    fn test_import_export_equiv() -> Result<(), ErrBox> {
        let exported = NSeqTokenizer::<2>::new(50);

        let input = "Litwo, ojczyzno moja, Ty jesteś jak zdrowie. Ile Cię trzeba cenić, ten tylko się dowie, kto Cię stracił.";

        exported.ingest(&input);

        let export = exported.export();

        let imported = NSeqTokenizer::<2>::import(&export)?;

        assert_eq!(
            &*exported.tokens.read().expect("tokens read e"),
            &*imported.tokens.read().expect("tokens read i")
        );

        Ok(())
    }

    #[test]
    fn test_tokenizer_respects_vocab_size() {
        let tokenizer = NSeqTokenizer::<2>::new(2);
        let input = "blablabl";

        assert_eq!(tokenizer.ingest(&input), (2, true));

        let tokens_read = tokenizer.tokens.read().expect("tokens read");

        dbg!(&tokens_read);

        let bl_comps: Vec<_> = "bl".chars().map(|c| TokenComponent::Char(c)).collect();
        let bla_comps = [
            TokenComponent::Tok(Box::new(tokens_read[bl_comps.as_slice()].clone())),
            TokenComponent::Char('a'),
        ];

        assert!(tokens_read.contains_key(bl_comps.as_slice()));
        assert!(tokens_read.contains_key(bla_comps.as_slice()));
    }
}
