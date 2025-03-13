use crate::util::ErrBox;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct Transformer {
    embed: Array2<f32>,
    blocks: Vec<TransformerBlock>,
}

/// Computes the value of a min/max tensor randomization value,
/// e.g. if X = dims2rand_boundary(), you use it for range (-X, X) for
/// the randomization sampling range.
fn dims2rand_boundary(x: usize, y: usize) -> f32 {
    (x as f32 * y as f32).sqrt()
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

        let rand_boundary = dims2rand_boundary(ctx_width, embed_dim);

        let embed = Array2::random(
            (ctx_width, embed_dim),
            Uniform::new(-rand_boundary, rand_boundary),
        );

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
        let rand_boundary = dims2rand_boundary(embed_dim, embed_dim);
        let rand_dist = Uniform::new(-rand_boundary, rand_boundary);

        let query = Array2::random((embed_dim, embed_dim), rand_dist);
        let key = Array2::random((embed_dim, embed_dim), rand_dist);
        let value = Array2::random((embed_dim, embed_dim), rand_dist);

	let ln_rand_boundary = dims2rand_boundary(1, embed_dim);
        let ln = Array1::random(embed_dim, Uniform::new(-ln_rand_boundary, ln_rand_boundary));

        let out = Array2::random((embed_dim, embed_dim), rand_dist);
        Self {
            query,
            key,
            value,
            ln,
            out,
        }
    }
}
