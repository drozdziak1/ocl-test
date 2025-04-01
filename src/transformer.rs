use crate::util::ErrBox;

use log::error;
use ndarray::{
    Array, Array1, Array2, Array3, Axis, AxisDescription, Dimension, Order, RemoveAxis, Zip,
};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::Rng;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use std::cmp::PartialOrd;

pub const SMOL_NUMBER: f32 = f32::EPSILON * 10.0;
pub const BIGE_NUMBER: f32 = f32::MAX / 10.0;

fn smol_if_needed(n: f32) -> f32 {
    if n.abs() > SMOL_NUMBER {
        n
    } else {
        SMOL_NUMBER.copysign(n)
    }
}

fn bige_if_needed(n: f32) -> f32 {
    if n.abs() < BIGE_NUMBER {
        n
    } else {
        BIGE_NUMBER.copysign(n)
    }
}

fn clip_if_needed(n: f32) -> f32 {
    bige_if_needed(smol_if_needed(n))
}

// Thin wrapper to capture variable name
macro_rules! check_floats_normal {
    ($val:expr $(,)?) => {
        do_check_floats_normal($val, stringify!($val), false, false)
    };
}

macro_rules! check_floats_not_nan {
    ($val:expr $(,)?) => {
        do_check_floats_normal($val, stringify!($val), false, true)
    };
}

macro_rules! check_floats_not_inf {
    ($val:expr $(,)?) => {
        do_check_floats_normal($val, stringify!($val), true, false)
    };
}

#[cfg(not(feature = "check_floats"))]
fn do_check_floats_normal<D: Dimension>(
    _a: &Array<f32, D>,
    _name: &str,
    _skip_nan: bool,
    _skip_inf: bool,
) -> Result<(), ErrBox> { Ok(())}

#[cfg(feature = "check_floats")]
fn do_check_floats_normal<D: Dimension>(
    a: &Array<f32, D>,
    name: &str,
    skip_nan: bool,
    skip_inf: bool,
) -> Result<(), ErrBox> {
    match (
        a.is_any_nan() && !skip_nan,
        a.is_any_infinite() && !skip_inf,
    ) {
        (true, true) => {
            error!("{}: {:#?}", name, a);
            Err(format!("{}: NaNs and inf values detected", name).into())
        }
        (true, false) => {
            error!("{}: {:#?}", name, a);
            Err(format!("{}: NaNs detected", name).into())
        }
        (false, true) => {
            error!("{}: {:#?}", name, a);
            Err(format!("{}: inf values detected", name).into())
        }
        (false, false) => Ok(()),
    }
}

/// Computes the range for tensor randomization. Randomization is
/// uniform in range (-l, l) where l = 1/sqrt(product(dim0, dim1, ...))
fn dims2rand_dist(dims: &[usize]) -> Uniform<f32> {
    let prod_sqrt = dims
        .iter()
        .fold(1.0, |acc, elem| acc * (*elem) as f32)
        .sqrt()
        .powi(-1);
    Uniform::new(-prod_sqrt, prod_sqrt)
}

/// Performs softmax on the selected axis
fn softmax_axis<D: Dimension + RemoveAxis>(a: &Array<f32, D>, ax: Axis) -> Array<f32, D> {
    let mut out = Array::zeros(a.dim());

    Zip::from(out.axis_iter_mut(ax))
        .and(a.axis_iter(ax))
        .par_for_each(|mut out_slice, a_slice| {
            let mut a_exp = a_slice.exp();
            let sum = a_exp.sum();

            Zip::from(&mut a_exp).par_for_each(|a| {
                *a = clip_if_needed(*a);
            });

            out_slice.assign(&(a_exp / clip_if_needed(sum)));
        });

    return out;
}

/// Naive GELU approximation
fn approx_gelu<D: Dimension>(a: &Array<f32, D>) -> Array<f32, D> {
    let mut out = Array::zeros(a.dim());

    Zip::from(&mut out).and(a).par_for_each(|out_elem, a_elem| {
        // sigmoid(x * 1.702)
        let sigmoid1702 = 1.0 / clip_if_needed(1.0 + (-1.702 * a_elem).exp());

        // GELU(x) ~= x * sigmoid(x * 1.702)
        let approx_gelu = a_elem * sigmoid1702;

        *out_elem = approx_gelu;
    });
    out
}

fn approx_gelu_grad<D: Dimension>(a: &Array<f32, D>) -> Array<f32, D> {
    let mut out = Array::zeros(a.dim());

    Zip::from(&mut out).and(a).par_for_each(|out_elem, a_elem| {
        let x1702 = 1.702 * a_elem;

        // sigmoid(x * 1.702)
        let sigmoid1702 = 1.0 / clip_if_needed(1.0 + (x1702).exp());

        let approx_gelu_grad = sigmoid1702 * (1.0 + x1702 * (1.0 - sigmoid1702));

        *out_elem = approx_gelu_grad;
    });
    out
}

pub struct Transformer {
    vocab_size: usize,
    embed_dim: usize,
    embed: Array2<f32>,
    unembed: Array2<f32>,
    blocks: Vec<TransformerBlock>,
}

#[derive(Default)]
pub struct TransformerTrainData {
    x: Vec<u32>,
    /// X @ embed
    x_embedded: Array2<f32>,
    /// Train data for each attention head
    tb_train_data: Vec<TBTrainData>,
    /// X after each attention block's contribution is added
    x_refined: Array2<f32>,
    /// last_refined_embedding @ unembed
    predictions_raw: Array2<f32>,
    /// softmax(pred_raw)
    predictions: Array2<f32>,
}

#[derive(Clone, Default)]
pub struct TransformerGradients {
    pub loss: f32,
    grad_l_embed: Array2<f32>,
    grad_l_unembed: Array2<f32>,
    grad_blocks: Vec<TBGradients>,
}

impl TransformerGradients {
    pub fn accum(&mut self, other: &Self) {
        self.grad_l_embed += &other.grad_l_embed;
        self.grad_l_unembed += &other.grad_l_unembed;

        self.grad_blocks
            .par_iter_mut()
            .zip(other.grad_blocks.par_iter())
            .for_each(|(block_grads, other_block_grads)| block_grads.accum(other_block_grads));
    }

    /// Sums the squares of all model gradients
    pub fn sum_squares(&self) -> f32 {
        let embed_unembed_iter = vec![&self.grad_l_embed, &self.grad_l_unembed]
            .into_par_iter()
            .map(|array| array.pow2().sum());

        let tb_iter = self.grad_blocks.par_iter().map(|tb| tb.sum_squares());

        let sum = embed_unembed_iter.chain(tb_iter).sum();

        return sum;
    }

    pub fn l2_norm_clip(&mut self, max_norm: f32) {
        let l2_norm = self.sum_squares().sqrt();

        let coeff = max_norm / clip_if_needed(l2_norm);

        if coeff < 1.0 {
            self.multiply(coeff);
        }
    }

    /// Multiply all parameters by coeff
    pub fn multiply(&mut self, coeff: f32) {
        self.grad_l_embed *= coeff;
        self.grad_l_unembed *= coeff;

        self.grad_blocks.par_iter_mut().for_each(|grad_block| {
            grad_block.multiply(coeff);
        });
    }
}

impl Transformer {
    pub fn new<R: Rng + ?Sized>(
        ctx_width: usize,
        vocab_size: usize, // Note: corresponds with the last token ID in the tokenizer
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        ff_dim: usize,
        rng: &mut R,
    ) -> Result<Self, ErrBox> {
        if embed_dim % num_heads != 0 {
            return Err(format!(
                "num_heads ({}) must evenly divide embed_dim ({})",
                num_heads, embed_dim
            )
            .into());
        }

        let rand_dist = dims2rand_dist(vec![vocab_size, embed_dim].as_slice());

        let embed = Array2::random_using((vocab_size, embed_dim), rand_dist, rng);

        let unembed = Array2::random_using((embed_dim, vocab_size), rand_dist, rng);

        let blocks = (0..num_layers)
            .map(|_i| TransformerBlock::new(embed_dim, num_heads, ff_dim, rng))
            .collect();

        Ok(Self {
            vocab_size,
            embed_dim,
            unembed,
            embed,
            blocks,
        })
    }

    pub fn naive_fwd(
        &self,
        x: &[u32],
        train_data: Option<&mut TransformerTrainData>,
    ) -> Result<u32, ErrBox> {
        let mut x_embedded: Array2<f32> = Array2::zeros((x.len(), self.embed_dim));

        Zip::from(x_embedded.outer_iter_mut())
            .and(x)
            .par_for_each(|mut embedded_row, tok_id| {
                let embedding = self.embed.index_axis(Axis(0), *tok_id as usize); // NOTE: panics if index out of bounds

                embedded_row.assign(&embedding);
            });

        check_floats_normal!(&x_embedded)?;

        let mut tb_train_data = if train_data.is_some() {
            vec![Default::default(); self.blocks.len()]
        } else {
            vec![]
        };

        let mut tb_train_data_iter = tb_train_data.iter_mut();

        let mut x_refined = x_embedded.clone();

        for (idx, block) in self.blocks.iter().enumerate() {
            block
                .naive_fwd(&mut x_refined, tb_train_data_iter.next())
                .map_err(|e| format!("Could not forward-pass block {}: {}", idx, e.to_string()))?;
        }

        // pred_raw = x_embedded @ self.unembed
        let predictions_raw = x_embedded.dot(&self.unembed);
        check_floats_normal!(&predictions_raw)?;

        // predictions = softmax(pred_raw)
        let predictions = softmax_axis(&predictions_raw, Axis(0));
        check_floats_normal!(&predictions)?;

        if let Some(train_data) = train_data {
            train_data.x = x.to_owned();
            train_data.x_embedded = x_embedded;
            train_data.tb_train_data = tb_train_data;
            train_data.x_refined = x_refined;
            train_data.predictions_raw = predictions_raw.clone();
            train_data.predictions = predictions.clone();
        }

        let last_prediction = predictions
            .outer_iter()
            .last()
            .ok_or_else(|| "Could not get last prediction")?;

        let (max_token_id, _max_token_prob) = last_prediction.indexed_iter().fold(
            (0, std::f32::MIN),
            |(acc_idx, acc_val), (idx, val)| {
                if *val > acc_val {
                    (idx, *val)
                } else {
                    (acc_idx, acc_val)
                }
            },
        );

        Ok(max_token_id as u32)
    }

    pub fn naive_bwd(
        &mut self,
        train_data: &TransformerTrainData,
        y_gt: &[u32],
    ) -> Result<TransformerGradients, ErrBox> {
        let mut y_gt_onehot: Array2<f32> = Array2::zeros((y_gt.len(), self.vocab_size));

        Zip::from(y_gt_onehot.outer_iter_mut())
            .and(y_gt)
            .par_for_each(|mut y_gt_onehot_row, y_gt_value| {
                y_gt_onehot_row[*y_gt_value as usize] = 1.0;
            });

        let z = &train_data.predictions_raw;
        let p = &train_data.predictions;
        check_floats_normal!(&p)?;

        let mut ln_p = p.clone();

        ln_p.par_iter_mut()
            .for_each(|elem| *elem = fastapprox::faster::ln((*elem).max(SMOL_NUMBER)));
        let loss = -(&y_gt_onehot * ln_p).sum();

        let grad_l_z = p - &y_gt_onehot;
        check_floats_normal!(&grad_l_z)?;

        let grad_l_unembed = train_data.x_refined.t().dot(&grad_l_z);
        check_floats_normal!(&grad_l_unembed)?;

        let grad_l_h = grad_l_z.dot(&self.unembed.t());
        check_floats_normal!(&grad_l_h)?;

        let mut grad_blocks = vec![Default::default(); self.blocks.len()];

        let tb_iter = self
            .blocks
            .iter_mut()
            .zip(train_data.tb_train_data.iter())
            .zip(grad_blocks.iter_mut())
            .enumerate()
            .rev();

        let mut grad_l_block_input = grad_l_h;
        for (idx, ((tb, tb_train_data), tb_grads)) in tb_iter {
            let (new_grad_l_block_input, this_tb_grads) = tb
                .naive_bwd(tb_train_data, grad_l_block_input, idx)
                .map_err(|e| format!("Could not backpropagate block {}: {}", idx, e.to_string()))?;

            *tb_grads = this_tb_grads;

            grad_l_block_input = new_grad_l_block_input;
        }

        let grad_l_x_embedded = grad_l_block_input;

        let mut grad_l_embed: Array2<f32> = Array2::zeros(self.embed.dim());

        for (grad_l_x_embedded_slice, tok_id) in
            grad_l_x_embedded.outer_iter().zip(train_data.x.iter())
        {
            let mut grad_l_embed_slice = grad_l_embed.index_axis_mut(Axis(0), *tok_id as usize);

            grad_l_embed_slice.assign(&(&grad_l_embed_slice + &grad_l_x_embedded_slice));
        }

        check_floats_normal!(&grad_l_embed)?;

        Ok(TransformerGradients {
            loss,
            grad_l_embed,
            grad_l_unembed,
            grad_blocks,
        })
    }

    pub fn apply_grads(&mut self, grads: &TransformerGradients, lr: f32) -> Result<(), ErrBox> {
        self.embed -= &(&grads.grad_l_embed * lr);
        self.unembed -= &(&grads.grad_l_unembed * lr);

        self.blocks
            .par_iter_mut()
            .zip(grads.grad_blocks.par_iter())
            .try_for_each(|(block, block_grads)| -> Result<(), String> {
                block
                    .apply_grads(block_grads, lr)
                    .map_err(|e| e.to_string())?;
                Ok(())
            })?;

        Ok(())
    }
}

pub struct TransformerBlock {
    num_heads: usize,
    head_size: usize,
    ff_dim: usize,
    embed_dim: usize,
    ln_scale: Array1<f32>,
    ln_bias: Array1<f32>,
    query: Array3<f32>,
    key: Array3<f32>,
    value: Array3<f32>,
    atn_out: Array2<f32>,
    ff1: Array2<f32>,
    ff1_b: Array2<f32>,
    ff2: Array2<f32>,
    ff2_b: Array2<f32>,
}

/// Intermediate state of X as it goes through the TransformerBlock; used for training
#[derive(Clone, Default)]
pub struct TBTrainData {
    /// X
    xs: Array2<f32>,
    /// Mean of each embedding
    mu: Array1<f32>,
    /// Variance of each embedding
    var: Array1<f32>,
    /// X normalized w.r.t. mu and variance
    x_hat: Array2<f32>,
    /// X after completing layernorm
    x_ln: Array2<f32>,
    /// X @ Q per attention head
    x_qs: Array3<f32>,
    /// X @ K per attention head
    x_ks: Array3<f32>,
    /// X @ V per attention head
    x_vs: Array3<f32>,
    /// Xq @ Xk per attention head
    xqxk: Array3<f32>,
    /// softmax(xqxk / sqrt(head_size))
    xqxk_act: Array3<f32>,
    /// XqXk_act @ Xv, all attention heads
    atn_scores_reshaped: Array2<f32>,
    /// atn_scores_reshaped @ atn_out
    atn_scores_atn_out: Array2<f32>,
    /// X + atn_out_reshaped
    x_with_atn_out: Array2<f32>,
    /// (X + atn_out_reshaped) @ ff1
    x_ff1: Array2<f32>,
    /// Xff1 + ff1_b
    x_ff1_b: Array2<f32>,
    /// GELU(Xff1_b)
    x_ff1_act: Array2<f32>,
    /// Xff1_act @ ff2
    x_ff2: Array2<f32>,
    /// Xff2 + ff2_b
    x_ff2_b: Array2<f32>,
    /// X + atn_out_reshape + X_ff2_b
    x_with_ff: Array2<f32>,
}

#[derive(Clone, Default)]
pub struct TBGradients {
    grad_l_ln_scale: Array1<f32>,
    grad_l_ln_bias: Array1<f32>,
    grad_l_query: Array3<f32>,
    grad_l_key: Array3<f32>,
    grad_l_value: Array3<f32>,
    grad_l_atn_out: Array2<f32>,
    grad_l_ff1: Array2<f32>,
    grad_l_ff1_b: Array2<f32>,
    grad_l_ff2: Array2<f32>,
    grad_l_ff2_b: Array2<f32>,
}

impl TBGradients {
    pub fn accum(&mut self, other: &Self) {
        self.grad_l_ln_scale += &other.grad_l_ln_scale;
        self.grad_l_ln_bias += &other.grad_l_ln_bias;

        self.grad_l_query += &other.grad_l_query;
        self.grad_l_key += &other.grad_l_key;
        self.grad_l_value += &other.grad_l_value;

        self.grad_l_atn_out += &other.grad_l_atn_out;

        self.grad_l_ff1 += &other.grad_l_ff1;
        self.grad_l_ff1_b += &other.grad_l_ff1_b;

        self.grad_l_ff2 += &other.grad_l_ff2;
        self.grad_l_ff2_b += &other.grad_l_ff2_b;
    }

    pub fn sum_squares(&self) -> f32 {
        let array1_iter = vec![&self.grad_l_ln_scale, &self.grad_l_ln_bias]
            .into_par_iter()
            .map(|arr| arr.pow2().sum());

        let array2_iter = vec![
            &self.grad_l_atn_out,
            &self.grad_l_ff1,
            &self.grad_l_ff1_b,
            &self.grad_l_ff2,
            &self.grad_l_ff2_b,
        ]
        .into_par_iter()
        .map(|arr| arr.pow2().sum());

        let array3_iter = vec![&self.grad_l_query, &self.grad_l_key, &self.grad_l_value]
            .into_par_iter()
            .map(|arr| arr.pow2().sum());

        // Rust is not very happy about Array1, Array2, ...,
        // technically being distinct types, so we apply a map to the
        // square sum independently for all three, ending up with a
        // plain f32 partial sums iterator.
        let sum = array1_iter.chain(array2_iter).chain(array3_iter).sum();

        return sum;
    }

    pub fn multiply(&mut self, coeff: f32) {
        // Array1
        vec![&mut self.grad_l_ln_scale, &mut self.grad_l_ln_bias]
            .into_par_iter()
            .for_each(|arr| *arr *= coeff);

        // Array2
        vec![
            &mut self.grad_l_atn_out,
            &mut self.grad_l_ff1,
            &mut self.grad_l_ff1_b,
            &mut self.grad_l_ff2,
            &mut self.grad_l_ff2_b,
        ]
        .into_par_iter()
        .for_each(|arr| *arr *= coeff);

        // Array3
        vec![
            &mut self.grad_l_query,
            &mut self.grad_l_key,
            &mut self.grad_l_value,
        ]
        .into_par_iter()
        .for_each(|arr| *arr *= coeff);
    }
}

impl TransformerBlock {
    pub fn new<R: Rng + ?Sized>(
        embed_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        rng: &mut R,
    ) -> Self {
        let head_size = embed_dim / num_heads;

        let qkv_rand_dist = dims2rand_dist(vec![num_heads, head_size, embed_dim].as_slice());

        let qkv_dims = (num_heads, embed_dim, head_size);

        let ln_scale = Array1::ones(embed_dim);
        let ln_bias = Array1::zeros(embed_dim);

        let query = Array3::random_using(qkv_dims, qkv_rand_dist, rng);

        let key = Array3::random_using(qkv_dims, qkv_rand_dist, rng);

        let value = Array3::random_using(qkv_dims, qkv_rand_dist, rng);

        let atn_out_rand_dist = dims2rand_dist(vec![embed_dim, embed_dim].as_slice());

        let atn_out = Array2::random_using((embed_dim, embed_dim), atn_out_rand_dist, rng);

        let ff_rand_dist = dims2rand_dist(vec![embed_dim, ff_dim].as_slice());

        let ff1 = Array2::random_using((embed_dim, ff_dim), ff_rand_dist, rng);
        let ff1_b = Array2::random_using((1, ff_dim), dims2rand_dist(vec![ff_dim].as_slice()), rng);

        let ff2 = Array2::random_using((ff_dim, embed_dim), ff_rand_dist, rng);
        let ff2_b = Array2::random_using(
            (1, embed_dim),
            dims2rand_dist(vec![embed_dim].as_slice()),
            rng,
        );

        Self {
            num_heads,
            head_size,
            ff_dim,
            embed_dim,

            ln_scale,
            ln_bias,

            query,
            key,
            value,

            atn_out,

            ff1,
            ff1_b,
            ff2,
            ff2_b,
        }
    }

    pub fn naive_fwd(
        &self,
        x: &mut Array2<f32>,
        train_data: Option<&mut TBTrainData>,
    ) -> Result<(), ErrBox> {
        // T = token count, C = embed dim

        check_floats_normal!(&x)?;
        // (T, C)
        let x_shp: Vec<AxisDescription> = x.axes().collect();

        // Save X for backprop
        let xs = x.clone();

        let mu = xs.mean_axis(Axis(1)).expect("failed to calculate mean");
        check_floats_normal!(&mu)?;

        let var = xs.var_axis(Axis(1), 0.0);
        check_floats_normal!(&var)?;

        // Normalized X preserved for backprop
        let mut x_hat = Array2::zeros(x.dim());

        // X after going through all of layernorm
        let mut x_ln = Array2::zeros(x.dim());

        Zip::from(x_ln.outer_iter_mut())
            .and(x_hat.outer_iter_mut())
            .and(xs.outer_iter())
            .and(&mu)
            .and(&var)
            .par_for_each(
                |mut x_ln_slice, mut x_hat_slice, xs_slice, mu_elem, var_elem| {
                    let x_hat = &xs_slice - *mu_elem / (var_elem + SMOL_NUMBER).sqrt();

                    let new_x_ln_row = &self.ln_scale * &x_hat + &self.ln_bias;

                    x_hat_slice.assign(&x_hat);
                    x_ln_slice.assign(&new_x_ln_row);
                },
            );

        check_floats_normal!(&x_hat)?;
        check_floats_normal!(&x_ln)?;

        let tril: Array2<u8> = Array2::ones((x_shp[0].len, x_shp[0].len)).tril(0);

        // ATTENTION

        let x_qkv_shp = (self.num_heads, x_shp[0].len, self.head_size);

        // X @ Q
        let mut x_qs: Array3<f32> = Array3::zeros(x_qkv_shp);

        Zip::from(x_qs.outer_iter_mut())
            .and(self.query.outer_iter())
            .par_for_each(|mut x_qs_slice, q_slice| {
                x_qs_slice.assign(&x_ln.dot(&q_slice));
            });

        check_floats_normal!(&x_qs)?;

        // X @ K
        let mut x_ks: Array3<f32> = Array3::zeros(x_qkv_shp);

        Zip::from(x_ks.outer_iter_mut())
            .and(self.key.outer_iter())
            .par_for_each(|mut x_ks_slice, k_slice| {
                x_ks_slice.assign(&x_ln.dot(&k_slice));
            });

        check_floats_normal!(&x_ks)?;

        // X @ V
        let mut x_vs: Array3<f32> = Array3::zeros(x_qkv_shp);

        Zip::from(x_vs.outer_iter_mut())
            .and(self.value.outer_iter())
            .par_for_each(|mut x_vs_slice, v_slice| {
                x_vs_slice.assign(&x_ln.dot(&v_slice));
            });

        check_floats_normal!(&x_vs)?;

        // softmax(mask(Q @ K) / sqrt(head_size))
        let mut xqxk: Array3<f32> = Array3::zeros((self.num_heads, x_shp[0].len, x_shp[0].len));
        let mut xqxk_act: Array3<f32> = Array3::zeros(xqxk.dim());

        Zip::from(xqxk.outer_iter_mut())
            .and(xqxk_act.outer_iter_mut())
            .and(x_qs.outer_iter())
            .and(x_ks.outer_iter())
            .par_for_each(
                |mut xqxk_slice, mut xqxk_act_slice, x_qs_slice, x_ks_slice| {
                    let mut matmul_prod = x_qs_slice.dot(&x_ks_slice.t());

                    // Mask off future tokens
                    Zip::from(&mut matmul_prod)
                        .and(&tril)
                        .par_for_each(|prod_elem, tril_elem| {
                            if *tril_elem == 0 {
                                *prod_elem = f32::NEG_INFINITY;
                            }
                        });

                    xqxk_slice.assign(&matmul_prod);

                    let divided = matmul_prod / clip_if_needed((self.head_size as f32).sqrt());
                    check_floats_not_nan!(&divided).unwrap();

                    // Softmax
                    let normalized = softmax_axis(&divided, Axis(0));
                    check_floats_normal!(&normalized).unwrap();
                    xqxk_act_slice.assign(&normalized);
                },
            );

        // check_floats_normal!(&xqxk)?; // Pointless for -inf masking
        check_floats_normal!(&xqxk_act)?;

        let mut atn_scores: Array3<f32> =
            Array3::zeros((self.num_heads, x_shp[0].len, self.head_size));

        Zip::from(atn_scores.outer_iter_mut())
            .and(xqxk_act.outer_iter())
            .and(x_vs.outer_iter())
            .par_for_each(|mut atn_out_slice, xqxk_act_slice, x_vs_slice| {
                atn_out_slice.assign(&xqxk_act_slice.dot(&x_vs_slice));
            });

        // (num_heads, T, head_size) -> (T, C)
        let mut atn_scores_reshaped: Array2<f32> = Array2::zeros((x_shp[0].len, x_shp[1].len));

        Zip::from(atn_scores_reshaped.rows_mut())
            .and(atn_scores.axis_iter(Axis(1)))
            .par_for_each(|mut reshaped_row, atn_scores_slice| {
                let new_row = atn_scores_slice
                    .to_shape(((x_shp[1].len), Order::RowMajor))
                    .expect(&format!(
                        "Could not reshape {:?} to {:?}",
                        atn_scores_slice.dim(),
                        (1, x_shp[1].len)
                    ));

                reshaped_row.assign(&new_row);
            });

        check_floats_normal!(&atn_scores_reshaped)?;

        let atn_scores_atn_out: Array2<f32> = atn_scores_reshaped.dot(&self.atn_out);

        check_floats_normal!(&atn_scores_atn_out)?;

        let x_with_atn_out = &*x + &atn_scores_atn_out;

        check_floats_normal!(&x_with_atn_out)?;

        *x = x_with_atn_out.clone();

        // MLP

        // Xff1 = GELU(X @ ff1 + ff1_b)

        let x_ff1: Array2<f32> = x.dot(&self.ff1);
        check_floats_normal!(&x_ff1)?;

        let mut x_ff1_b: Array2<f32> = x_ff1.clone();

        let ff1_b_row = self.ff1_b.index_axis(Axis(0), 0);

        Zip::from(x_ff1_b.axis_iter_mut(Axis(0))).par_for_each(|mut x_ff1_b_row| {
            x_ff1_b_row.assign(&(&x_ff1_b_row + &ff1_b_row));
        });

        check_floats_normal!(&x_ff1_b)?;

        let x_ff1_act: Array2<f32> = approx_gelu(&x_ff1_b);

        check_floats_normal!(&x_ff1_act)?;

        // Xff2 = Xff1 @ ff2 + ff2_b

        let x_ff2: Array2<f32> = x_ff1_act.dot(&self.ff2);

        let mut x_ff2_b: Array2<f32> = x_ff2.clone();

        let ff2_b_row = self.ff2_b.index_axis(Axis(0), 0);

        Zip::from(x_ff2_b.axis_iter_mut(Axis(0))).par_for_each(|mut x_ff2_b_row| {
            x_ff2_b_row.assign(&(&x_ff2_b_row + &ff2_b_row));
        });

        let x_with_ff = &*x + &x_ff2_b;

        *x = x_with_ff.clone();

        if let Some(train_data) = train_data {
            *train_data = TBTrainData {
                xs,
                mu,
                var,
                x_hat,
                x_ln,
                x_qs,
                x_ks,
                x_vs,
                xqxk,
                xqxk_act,
                atn_scores_reshaped,
                atn_scores_atn_out,
                x_with_atn_out,
                x_ff1,
                x_ff1_b,
                x_ff1_act,
                x_ff2,
                x_ff2_b,
                x_with_ff,
            }
        }

        Ok(())
    }

    pub fn naive_bwd(
        &mut self,
        train_data: &TBTrainData,
        // Next block input's gradient
        grad_l_x_next_block: Array2<f32>,
        // Which block am I?
        idx: usize,
    ) -> Result<(Array2<f32>, TBGradients), ErrBox> {
        // FEEDFORWARD

        // a.k.a. dL/dx_with_ff
        let grad_l_hffn = grad_l_x_next_block;
        check_floats_normal!(&grad_l_hffn)?;

        let grad_l_ff2 = train_data.x_ff1_act.t().dot(&grad_l_hffn);

        let grad_l_ff2_b = grad_l_hffn.sum_axis(Axis(0));

        let grad_l_ff1_act = grad_l_hffn.dot(&self.ff2.t());

        // aka dL/dx_ff1_b
        let grad_l_hprime = grad_l_ff1_act * approx_gelu_grad(&train_data.x_ff1_b);

        let grad_l_ff1 = train_data.x_with_atn_out.t().dot(&grad_l_hprime);

        let grad_l_ff1_b = grad_l_hprime.sum_axis(Axis(0));

        // aka dL/dx_with_atn_out
        let grad_l_hatn = grad_l_hprime.dot(&self.ff1.t());

        // FINAL atn_out MATRIX

        let grad_l_atn_out = train_data.atn_scores_reshaped.t().dot(&grad_l_hatn);

        // aka dL/datn_scores_atn_out
        let grad_l_o = grad_l_hatn.dot(&self.atn_out.t());

        let grad_l_o_shp: Vec<_> = grad_l_o.axes().collect();

        // MULTI-HEAD ATTENTION

        // Split for individual heads, i.e. (T, C) -> (num_heads, T, head_size)
        let mut grad_l_o_reshaped: Array3<f32> =
            Array3::zeros((self.num_heads, grad_l_o_shp[0].len, self.head_size));

        Zip::from(grad_l_o_reshaped.axis_iter_mut(Axis(1)))
            .and(grad_l_o.rows())
            .par_for_each(|mut reshaped_row, grad_l_o_row| {
                let new_row = grad_l_o_row
                    .to_shape(((self.num_heads, self.head_size), Order::RowMajor))
                    .expect(&format!(
                        "Could not reshape {:?} to {:?}",
                        grad_l_o_row.dim(),
                        (self.num_heads, self.head_size)
                    ));
                reshaped_row.assign(&new_row);
            });

        // aka dL/d(softmax(xqxk) / sqrt(head_size))
        let mut grad_l_a: Array3<f32> = Array3::zeros(train_data.xqxk_act.dim());
        let mut grad_l_x_vs: Array3<f32> = Array3::zeros(train_data.x_vs.dim());

        let mut grad_l_xqxk: Array3<f32> = Array3::zeros(train_data.xqxk.dim());

        Zip::from(grad_l_a.outer_iter_mut())
            .and(grad_l_x_vs.outer_iter_mut())
            .and(grad_l_xqxk.outer_iter_mut())
            .and(grad_l_o_reshaped.outer_iter())
            .and(train_data.x_vs.outer_iter())
            .and(train_data.xqxk_act.outer_iter())
            .par_for_each(
                |mut grad_l_a_slice,
                 mut grad_l_x_vs_slice,
                 mut grad_l_xqxk_slice,
                 grad_l_o_slice,
                 x_vs_slice,
                 xqxk_act_slice| {
                    let new_grad_l_a_slice = grad_l_o_slice.dot(&x_vs_slice.t());
                    grad_l_a_slice.assign(&new_grad_l_a_slice);

                    let new_grad_l_x_vs_slice = xqxk_act_slice.dot(&grad_l_o_slice);
                    grad_l_x_vs_slice.assign(&new_grad_l_x_vs_slice);

                    let new_grad_l_xqxk_slice = xqxk_act_slice.to_owned()
                        * (&new_grad_l_a_slice
                            - xqxk_act_slice
                                .t()
                                .dot(&new_grad_l_a_slice)
                                .dot(&xqxk_act_slice));

                    grad_l_xqxk_slice.assign(&new_grad_l_xqxk_slice);
                },
            );

        let mut grad_l_x_qs: Array3<f32> = Array3::zeros(train_data.x_qs.dim());
        let mut grad_l_x_ks: Array3<f32> = Array3::zeros(train_data.x_ks.dim());

        Zip::from(grad_l_x_qs.outer_iter_mut())
            .and(grad_l_x_ks.outer_iter_mut())
            .and(grad_l_xqxk.outer_iter())
            .and(train_data.x_qs.outer_iter())
            .and(train_data.x_ks.outer_iter())
            .par_for_each(
                |mut grad_l_x_qs_slice,
                 mut grad_l_x_ks_slice,
                 grad_l_xqxk_slice,
                 x_qs_slice,
                 x_ks_slice| {
                    let new_grad_l_x_qs_slice = grad_l_xqxk_slice.dot(&x_ks_slice);
                    grad_l_x_qs_slice.assign(&new_grad_l_x_qs_slice);

                    let new_grad_l_x_ks_slice = grad_l_xqxk_slice.t().dot(&x_qs_slice);
                    grad_l_x_ks_slice.assign(&new_grad_l_x_ks_slice);
                },
            );

        let mut grad_l_query: Array3<f32> = Array3::zeros(self.query.dim());
        let mut grad_l_key: Array3<f32> = Array3::zeros(self.key.dim());
        let mut grad_l_value: Array3<f32> = Array3::zeros(self.query.dim());

        Zip::from(grad_l_query.outer_iter_mut())
            .and(grad_l_key.outer_iter_mut())
            .and(grad_l_value.outer_iter_mut())
            .and(grad_l_x_qs.outer_iter())
            .and(grad_l_x_ks.outer_iter())
            .and(grad_l_x_vs.outer_iter())
            .par_for_each(
                |mut grad_l_query_slice,
                 mut grad_l_key_slice,
                 mut grad_l_value_slice,
                 grad_l_x_qs_slice,
                 grad_l_x_ks_slice,
                 grad_l_x_vs_slice| {
                    grad_l_query_slice.assign(&train_data.xs.t().dot(&grad_l_x_qs_slice));
                    grad_l_key_slice.assign(&train_data.xs.t().dot(&grad_l_x_ks_slice));
                    grad_l_value_slice.assign(&train_data.xs.t().dot(&grad_l_x_vs_slice));
                },
            );

        let xs_dim = train_data.xs.dim();

        let mut grad_l_x_per_head: Array3<f32> =
            Array3::zeros((self.num_heads, xs_dim.0, xs_dim.1));

        Zip::from(grad_l_x_per_head.outer_iter_mut())
            .and(grad_l_x_qs.outer_iter())
            .and(self.query.outer_iter())
            .and(grad_l_x_ks.outer_iter())
            .and(self.key.outer_iter())
            .par_for_each(
                |mut x_per_head_slice,
                 grad_l_x_qs_slice,
                 query_slice,
                 grad_l_x_ks_slice,
                 key_slice| {
                    x_per_head_slice.assign(
                        &(&x_per_head_slice
                            + &grad_l_x_qs_slice.dot(&query_slice.t())
                            + &grad_l_x_ks_slice.dot(&key_slice.t())),
                    );
                },
            );

        // Zip takes up to 6 arguments, there would be 7 if we added all of Q, K and V, so we do V separately
        Zip::from(grad_l_x_per_head.outer_iter_mut())
            .and(grad_l_x_vs.outer_iter())
            .and(self.value.outer_iter())
            .par_for_each(|mut x_per_head_slice, grad_l_x_vs_slice, value_slice| {
                x_per_head_slice
                    .assign(&(&x_per_head_slice + &grad_l_x_vs_slice.dot(&value_slice.t())));
            });

        // Sum across heads
        let grad_l_x_ln = grad_l_x_per_head.sum_axis(Axis(0));

        let grad_l_ln_scale = (&grad_l_x_ln * &train_data.x_hat).sum_axis(Axis(0));

        let grad_l_ln_bias = grad_l_x_ln.sum_axis(Axis(0));

        let mut grad_l_x_hat = Array2::zeros(train_data.xs.dim());

        Zip::from(grad_l_x_hat.outer_iter_mut())
            .and(grad_l_x_ln.outer_iter())
            .par_for_each(|mut grad_l_x_hat_slice, grad_l_x_ln_slice| {
                grad_l_x_hat_slice.assign(&(&grad_l_x_ln_slice * &self.ln_scale));
            });

        // Helper precomputed value of X[i,j] - mu[i]
        let mut xs_minus_mu = Array2::zeros(train_data.xs.dim());

        Zip::from(xs_minus_mu.outer_iter_mut())
            .and(train_data.xs.outer_iter())
            .and(&train_data.mu)
            .par_for_each(|mut xs_minus_mu_slice, xs_slice, mu_elem| {
                xs_minus_mu_slice.assign(&(&xs_slice - *mu_elem));
            });

        // NOTE: equations for remaining layernorm gradients are
        // rather large, we split them into arbitrary terms to cope.
        let grad_l_var_term1 = &grad_l_x_hat * &xs_minus_mu;
        let grad_l_var_term2 = -0.5 * (&train_data.var + SMOL_NUMBER).powf(-1.5);

        let mut grad_l_var_term1_times_term2 = Array2::zeros(train_data.xs.dim());

        Zip::from(grad_l_var_term1_times_term2.outer_iter_mut())
            .and(grad_l_var_term1.outer_iter())
            .and(&grad_l_var_term2)
            .par_for_each(
                |mut grad_l_var_term1_times_term2_slice,
                 grad_l_var_term1_slice,
                 grad_l_var_term2_elem| {
                    grad_l_var_term1_times_term2_slice
                        .assign(&(&grad_l_var_term1_slice * *grad_l_var_term2_elem));
                },
            );

        let grad_l_var = grad_l_var_term1_times_term2.sum_axis(Axis(1));

        let mut grad_l_mu_term1 = Array2::zeros(train_data.xs.dim());

        Zip::from(grad_l_mu_term1.outer_iter_mut())
            .and(grad_l_x_hat.outer_iter())
            .and(&train_data.var)
            .par_for_each(|mut grad_l_mu_term1_slice, grad_l_x_hat_slice, var_elem| {
                let new_slice = &grad_l_x_hat_slice * (-1.0 / (*var_elem + SMOL_NUMBER).sqrt());

                grad_l_mu_term1_slice.assign(&new_slice);
            });

        let grad_l_mu_term2 = (-2.0 * &xs_minus_mu / (self.embed_dim as f32)).sum_axis(Axis(1));

        let grad_l_mu = grad_l_mu_term1.sum_axis(Axis(1)) + &grad_l_var * grad_l_mu_term2;

        let mut grad_l_x_term1 = Array2::zeros(train_data.xs.dim());

        Zip::from(grad_l_x_term1.outer_iter_mut())
            .and(grad_l_x_hat.outer_iter())
            .and(&train_data.var)
            .par_for_each(|mut grad_l_x_term1_slice, grad_l_x_hat_slice, var_elem| {
                let new_slice = &grad_l_x_hat_slice / (*var_elem + SMOL_NUMBER).sqrt();

                grad_l_x_term1_slice.assign(&new_slice);
            });

        let mut grad_l_x_term2 = Array2::zeros(train_data.xs.dim());

        Zip::from(grad_l_x_term2.outer_iter_mut())
            .and(xs_minus_mu.outer_iter())
            .and(&grad_l_var)
            .par_for_each(
                |mut grad_l_x_term2_slice, xs_minus_mu_slice, grad_l_var_elem| {
                    let new_slice =
                        *grad_l_var_elem * 2.0 * &xs_minus_mu_slice / (self.embed_dim as f32);

                    grad_l_x_term2_slice.assign(&new_slice);
                },
            );

        let grad_l_x_term3 = grad_l_mu / (self.embed_dim as f32);

        let mut grad_l_x = Array2::zeros(train_data.xs.dim());

        Zip::from(grad_l_x.outer_iter_mut())
            .and(grad_l_x_term1.outer_iter())
            .and(grad_l_x_term2.outer_iter())
            .and(&grad_l_x_term3)
            .par_for_each(
                |mut grad_l_x_slice,
                 grad_l_x_term1_slice,
                 grad_l_x_term2_slice,
                 grad_l_x_term3_elem| {
                    let new_slice =
                        &grad_l_x_term1_slice + &grad_l_x_term2_slice + *grad_l_x_term3_elem;

                    grad_l_x_slice.assign(&new_slice);
                },
            );

        // VERIFY GRADIENTS
        check_floats_normal!(&grad_l_ff2_b)?;
        check_floats_normal!(&grad_l_ff2)?;

        check_floats_normal!(&grad_l_ff1_b)?;
        check_floats_normal!(&grad_l_ff1)?;

        check_floats_normal!(&grad_l_atn_out)?;

        check_floats_normal!(&grad_l_query)?;
        check_floats_normal!(&grad_l_key)?;
        check_floats_normal!(&grad_l_value)?;

        check_floats_normal!(&grad_l_ln_scale)?;
        check_floats_normal!(&grad_l_ln_bias)?;

        Ok((
            grad_l_x,
            TBGradients {
                grad_l_ln_scale,
                grad_l_ln_bias,

                grad_l_query,
                grad_l_key,
                grad_l_value,

                grad_l_atn_out,

                grad_l_ff1,
                grad_l_ff1_b: grad_l_ff1_b.insert_axis(Axis(0)),

                grad_l_ff2,
                grad_l_ff2_b: grad_l_ff2_b.insert_axis(Axis(0)),
            },
        ))
    }

    pub fn apply_grads(&mut self, grads: &TBGradients, lr: f32) -> Result<(), ErrBox> {
        self.ln_scale -= &(&grads.grad_l_ln_scale * lr);
        self.ln_bias -= &(&grads.grad_l_ln_bias * lr);

        self.query -= &(&grads.grad_l_query * lr);
        self.key -= &(&grads.grad_l_key * lr);
        self.value -= &(&grads.grad_l_value * lr);

        self.atn_out -= &(&grads.grad_l_atn_out * lr);

        self.ff1 -= &(&grads.grad_l_ff1 * lr);
        self.ff1_b -= &(&grads.grad_l_ff1_b * lr);

        self.ff2 -= &(&grads.grad_l_ff2 * lr);
        self.ff2_b -= &(&grads.grad_l_ff2_b * lr);

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use ndarray::arr2;

    use crate::util;

    use super::*;

    #[test]
    fn test_tb_naive_fwd_has_correct_output_dims() -> Result<(), ErrBox> {
        let embed_dim = 8;
        let tb = TransformerBlock::new(embed_dim, 2, 15);

        let mut x = Array2::random((42, embed_dim), Uniform::new(-5.0, 5.0));

        let axes_before: Vec<_> = x.axes().collect();

        tb.naive_fwd(&mut x, None)?;

        let axes_after: Vec<_> = x.axes().collect();

        for (before, after) in axes_before.iter().zip(axes_after.iter()) {
            assert_eq!(before.len, after.len);
            assert_eq!(before.stride, after.stride);
        }

        Ok(())
    }

    /// Does this thing even learn?
    #[test]
    fn test_transformer_happy_path() -> Result<(), ErrBox> {
        let ctx_size = 32_000;
        let vocab_size = 15;
        let embed_dim = 32;
        let num_layers = 32;
        let num_heads = 16;
        let ff_dim = 18;
        let lr = 0.01;

        let mut t = Transformer::new(
            ctx_size, vocab_size, embed_dim, num_layers, num_heads, ff_dim,
        )?;

        let n_iter = 20;

        let mut loss = std::f32::MAX;

        for _i in 0..n_iter {
            let mut train_data = TransformerTrainData::default();

            let x = vec![1, 2, 3, 4, 5];

            let _pred = t.naive_fwd(x.as_slice(), Some(&mut train_data))?;

            // Reuse X as Y_gt for training on the tokens leading up to the next token prediction
            let mut y_gt = x;

            // Final token desired prediction
            let expected = 9u32;

            y_gt.push(expected);
            y_gt = (&y_gt[1..]).to_owned();

            let new_loss = t.naive_bwd(lr, &train_data, &y_gt)?;

            assert!(dbg!(new_loss) < dbg!(loss));
        }

        Ok(())
    }

    #[test]
    fn test_softmax_axis_large_neg_number() -> Result<(), ErrBox> {
        util::init_log();

        let a = arr2(&[[2.449346e24, std::f32::NEG_INFINITY]]);

        let normalized = softmax_axis(&a, Axis(0));

        dbg!(&normalized);

        check_floats_normal!(&normalized)?;

        Ok(())
    }
}
