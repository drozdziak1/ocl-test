use crate::util::ErrBox;
use ndarray::{Array1, Array2};

pub struct Transformer {
    embed: Array2<f32>,
    blocks: Vec<TransformerBlock>,
}

impl Transformer {
    pub fn new(
        ctx_width: usize,
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        ff_dim: usize,
        dropout: f32,
    ) -> Result<Self, ErrBox> {
        if embed_dim % num_heads != 0 {
            return Err(format!(
                "num_heads ({}) must evenly divide embed_dim ({})",
                num_heads, embed_dim
            )
            .into());
        }

        let embed = Array2::zeros((ctx_width, embed_dim));

        let blocks = (0..num_layers)
            .map(|_i| TransformerBlock::new(embed_dim, num_heads, ff_dim, dropout))
            .collect();

        Ok(Self { embed, blocks })
    }
}

pub struct TransformerBlock {
    query: Array2<f32>,
    key: Array2<f32>,
    value: Array2<f32>,
    ln: Array1<f32>,
    out: Array2<f32>,
}

impl TransformerBlock {
    pub fn new(embed_dim: usize, num_heads: usize, ff_dim: usize, dropout: f32) -> Self {
        let query = Array2::zeros((embed_dim, embed_dim));
        let key = Array2::zeros((embed_dim, embed_dim));
        let value = Array2::zeros((embed_dim, embed_dim));

        let ln = Array1::zeros(embed_dim);

        let out = Array2::zeros((embed_dim, embed_dim));
        Self {
            query,
            key,
            value,
            ln,
            out,
        }
    }
}
