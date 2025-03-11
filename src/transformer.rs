use ndarray::Array2;

pub struct Transformer {
    embed: Array2<f32>,
}

impl Transformer {
    pub fn new(ctx_width: usize, embed_dim: usize, num_heads: u32, ff_dim: u32, dropout: f32) -> Self {
	let embed = Array2::zeros((ctx_width, embed_dim));
	
	Self {
	    embed,
	}
    }
    
}

pub struct TransformerBlock {
    
}
