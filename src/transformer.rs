use std::iter::repeat;

use crate::util::ErrBox;
use ndarray::{
    Array, Array1, Array2, Array3, ArrayBase, Axis, AxisDescription, Dimension, Order, Zip,
};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
fn softmax_axis(a: &Array2<f32>, ax: Axis) -> Array2<f32> {
    let mut out = Array2::zeros(a.dim());

    Zip::from(out.axis_iter_mut(ax))
        .and(a.axis_iter(ax))
        .par_for_each(|mut out_slice, a_slice| {
            let a_exp = a_slice.exp();
            let sum = a_exp.sum();

            out_slice.assign(&(a_exp / sum));
        });

    return out;
}

/// Naive GELU approximation
fn approx_gelu<D: Dimension>(a: &Array<f32, D>) -> Array<f32, D> {
    let mut out = Array::zeros(a.dim());

    Zip::from(&mut out).and(a).par_for_each(|out_elem, a_elem| {
        // sigmoid(x * 1.702)
        let sigmoid1702 = 1.0 / (1.0 + (-1.702 * a_elem).exp());

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
        let sigmoid1702 = 1.0 / (1.0 + (x1702).exp());

        let approx_gelu_grad = sigmoid1702 * (1.0 + x1702 * (1.0 - sigmoid1702));

        *out_elem = approx_gelu_grad;
    });
    out
}

pub struct Transformer {
    embed_dim: usize,
    unembed: Array2<f32>,
    embed: Array2<f32>,
    blocks: Vec<TransformerBlock>,
}

#[derive(Default)]
pub struct TransformerTrainData {
    /// X @ embed
    x_embedded: Array2<f32>,
    /// Train data for each attention head
    tb_train_data: Vec<TBTrainData>,
    /// last row of X after all transformer blocks add their output to it
    last_refined_embedding: Array2<f32>,
    /// last_refined_embedding @ unembed
    pred_raw: Array2<f32>,
    /// softmax(pred_raw)
    prediction: Array2<f32>,
}

impl Transformer {
    pub fn new(
        ctx_width: usize,
        vocab_size: usize, // Note: corresponds with the last token ID in the tokenizer
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        ff_dim: usize,
    ) -> Result<Self, ErrBox> {
        if embed_dim % num_heads != 0 {
            return Err(format!(
                "num_heads ({}) must evenly divide embed_dim ({})",
                num_heads, embed_dim
            )
            .into());
        }

        let rand_dist = dims2rand_dist(vec![vocab_size, embed_dim].as_slice());

        let embed = Array2::random((vocab_size, embed_dim), rand_dist);

        let unembed = Array2::random((embed_dim, vocab_size), rand_dist);

        let blocks = (0..num_layers)
            .map(|_i| TransformerBlock::new(embed_dim, num_heads, ff_dim))
            .collect();

        Ok(Self {
            embed_dim,
            unembed,
            embed,
            blocks,
        })
    }

    pub fn naive_fwd(
        &self,
        x: &[u32],
        mut train_data: Option<&mut TransformerTrainData>,
    ) -> Result<Array1<f32>, ErrBox> {
        let mut x_embedded: Array2<f32> = Array2::zeros((x.len(), self.embed_dim));

        Zip::from(x_embedded.outer_iter_mut())
            .and(x)
            .par_for_each(|mut embedded_row, tok_id| {
                let embedding = self.embed.index_axis(Axis(0), *tok_id as usize); // NOTE: panics if index out of bounds

                embedded_row.assign(&embedding);
            });

        let mut tb_train_data = if train_data.is_some() {
            vec![Default::default(); self.blocks.len()]
        } else {
            vec![]
        };

        let mut tb_train_data_iter = tb_train_data.iter_mut();

        for block in self.blocks.iter() {
            block.naive_fwd(&mut x_embedded, tb_train_data_iter.next())?;
        }

        let last_refined_embedding = x_embedded
            .axis_iter(Axis(0))
            .next_back()
            .ok_or_else(|| "Could not get last element of X!")?
            .to_owned();

        let last_refined_embedding = last_refined_embedding.insert_axis(Axis(0));

        // pred_raw = x[last] @ self.unembed
        let pred_raw = &last_refined_embedding.dot(&self.unembed);

        // prediction = softmax(pred_raw)
        let prediction = softmax_axis(&pred_raw, Axis(0));

        if let Some(train_data) = train_data {
            train_data.x_embedded = x_embedded;
            train_data.tb_train_data = tb_train_data;
            train_data.last_refined_embedding = last_refined_embedding;
            train_data.pred_raw = pred_raw.clone();
            train_data.prediction = prediction.clone();
        }

        Ok(prediction.remove_axis(Axis(0)))
    }

    pub fn naive_bwd(
        &mut self,
        lr: f32,
        train_data: &TransformerTrainData,
        y_gt: &Array1<f32>,
    ) -> Result<(), ErrBox> {
        let z = &train_data.pred_raw;
        let p = &train_data.prediction;
        let loss = -(y_gt * p.log10()).sum();

        // dbg!(&loss);

        let grad_l_z = p - y_gt;

        // dbg!(&grad_l_z);

        let grad_l_unembed = train_data.last_refined_embedding.t().dot(&grad_l_z);

        // dbg!(&grad_l_unembed);

        let grad_l_h = grad_l_z.dot(&self.unembed.t());

        dbg!(train_data.tb_train_data.len());

        let tb_iter = self
            .blocks
            .iter_mut()
            .zip(train_data.tb_train_data.iter())
            .rev();

        let mut grad_block_output = grad_l_h;
        for (tb, tb_train_data) in tb_iter {
            grad_block_output = tb.naive_bwd(tb_train_data, grad_block_output)?;
        }

        panic!("kupak");

        Ok(())
    }
}

pub struct TransformerBlock {
    num_heads: usize,
    head_size: usize,
    ff_dim: usize,
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
    /// X @ Q per attention head
    x_qs: Array3<f32>,
    /// X @ K per attention head
    x_ks: Array3<f32>,
    /// X @ V per attention head
    x_vs: Array3<f32>,
    /// Xq @ Xk per attention head
    xqxk: Array3<f32>,
    /// XqXk @ Xv, all attention heads
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

impl TransformerBlock {
    pub fn new(embed_dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        let head_size = embed_dim / num_heads;

        let qkv_rand_dist = dims2rand_dist(vec![num_heads, head_size, embed_dim].as_slice());

        let qkv_dims = (num_heads, embed_dim, head_size);

        let query = Array3::random(qkv_dims, qkv_rand_dist);

        let key = Array3::random(qkv_dims, qkv_rand_dist);

        let value = Array3::random(qkv_dims, qkv_rand_dist);

        let atn_out_rand_dist = dims2rand_dist(vec![embed_dim, embed_dim].as_slice());

        let atn_out = Array2::random((embed_dim, embed_dim), atn_out_rand_dist);

        let ff_rand_dist = dims2rand_dist(vec![embed_dim, ff_dim].as_slice());

        let ff1 = Array2::random((embed_dim, ff_dim), ff_rand_dist);
        let ff1_b = Array2::random((1, ff_dim), dims2rand_dist(vec![ff_dim].as_slice()));

        let ff2 = Array2::random((ff_dim, embed_dim), ff_rand_dist);
        let ff2_b = Array2::random((1, embed_dim), dims2rand_dist(vec![embed_dim].as_slice()));

        Self {
            num_heads,
            head_size,
            ff_dim,

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

        // (T, C)
        let x_shp: Vec<AxisDescription> = x.axes().collect();

        let tril: Array2<u8> = Array2::ones((x_shp[0].len, x_shp[0].len)).tril(0);

        // ATTENTION

        let x_qkv_shp = (self.num_heads, x_shp[0].len, self.head_size);

        // X @ Q
        let mut x_qs: Array3<f32> = Array3::zeros(x_qkv_shp);

        Zip::from(x_qs.outer_iter_mut())
            .and(self.query.outer_iter())
            .par_for_each(|mut x_qs_slice, q_slice| {
                x_qs_slice.assign(&x.dot(&q_slice));
            });

        // X @ K
        let mut x_ks: Array3<f32> = Array3::zeros(x_qkv_shp);

        Zip::from(x_ks.outer_iter_mut())
            .and(self.key.outer_iter())
            .par_for_each(|mut x_ks_slice, k_slice| {
                x_ks_slice.assign(&x.dot(&k_slice));
            });

        // X @ V
        let mut x_vs: Array3<f32> = Array3::zeros(x_qkv_shp);

        Zip::from(x_vs.outer_iter_mut())
            .and(self.value.outer_iter())
            .par_for_each(|mut x_vs_slice, v_slice| {
                x_vs_slice.assign(&x.dot(&v_slice));
            });

        // softmax(mask(Q @ K)) / sqrt(head_size)
        let mut xqxk: Array3<f32> = Array3::zeros((self.num_heads, x_shp[0].len, x_shp[0].len));

        Zip::from(xqxk.outer_iter_mut())
            .and(x_qs.outer_iter())
            .and(x_ks.outer_iter())
            .par_for_each(|mut xqxk_slice, x_qs_slice, x_ks_slice| {
                let mut matmul_prod = x_qs_slice.dot(&x_ks_slice.t());

                // Mask off future tokens
                Zip::from(&mut matmul_prod)
                    .and(&tril)
                    .par_for_each(|prod_elem, tril_elem| {
                        if *tril_elem == 0 {
                            *prod_elem = f32::NEG_INFINITY;
                        }
                    });

                // Softmax
                let normalized =
                    softmax_axis(&matmul_prod, Axis(0)) / (self.head_size as f32).sqrt();

                xqxk_slice.assign(&normalized);
            });

        let mut atn_scores: Array3<f32> =
            Array3::zeros((self.num_heads, x_shp[0].len, self.head_size));

        Zip::from(atn_scores.outer_iter_mut())
            .and(xqxk.outer_iter())
            .and(x_vs.outer_iter())
            .par_for_each(|mut atn_out_slice, xqxk_slice, x_vs_slice| {
                atn_out_slice.assign(&xqxk_slice.dot(&x_vs_slice));
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

        let atn_scores_atn_out: Array2<f32> = atn_scores_reshaped.dot(&self.atn_out);

        let x_with_atn_out = &*x + &atn_scores_atn_out;

        *x = x_with_atn_out.clone();

        // MLP

        // Xff1 = GELU(X @ ff1 + ff1_b)
        let x_ff1_dims = (x_shp[0].len, self.ff_dim);
        let ff1_b_broadcast = self.ff1_b.broadcast(x_ff1_dims).expect(&format!(
            "Could not broadcast ff1_b to dimensions {:?}",
            x_ff1_dims
        ));

        let x_ff1: Array2<f32> = x.dot(&self.ff1);

        let x_ff1_b: Array2<f32> = &x_ff1 + &ff1_b_broadcast;

        let x_ff1_act: Array2<f32> = approx_gelu(&x_ff1_b);

        // Xff2 = Xff1 @ ff2 + ff2_b
        let x_ff2_dims = (x_shp[0].len, x_shp[1].len);
        let ff2_b_broadcast = self
            .ff2_b
            .broadcast((x_shp[0].len, x_shp[1].len))
            .expect(&format!(
                "Could not broadcast ff2_b to dimensions {:?}",
                x_ff2_dims
            ));

        let x_ff2: Array2<f32> = x_ff1_act.dot(&self.ff2);

        let x_ff2_b: Array2<f32> = &x_ff2 + &ff2_b_broadcast;

        let x_with_ff = &*x + &x_ff2_b;

        *x = x_with_ff.clone();

        if let Some(train_data) = train_data {
            *train_data = TBTrainData {
                x_qs,
                x_ks,
                x_vs,
                xqxk,
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
        grad_l_block_out: Array2<f32>,
    ) -> Result<Array2<f32>, ErrBox> {
        // FEEDFORWARD

        // a.k.a. dL/dx_with_ff
        let grad_l_hffn = grad_l_block_out;

        // Final token's ff1 activated output (from chain rule: d(a*b) / d_b = a, d(a*b) / d_a = b)
        let last_x_ff1_act = train_data
            .x_ff1_act
            .axis_iter(Axis(0))
            .next_back()
            .ok_or_else(|| "Could not get last element of x_ff1_act")?
            .to_owned()
            .insert_axis(Axis(0));

        let grad_l_ff2 = last_x_ff1_act.t().dot(&grad_l_hffn);

        let grad_l_ff2_b = grad_l_hffn.clone();

        dbg!(&grad_l_ff2);

        let grad_l_ff1_act = grad_l_hffn.dot(&self.ff2.t());

        dbg!(&grad_l_ff1_act);

        let last_x_ff1_b = train_data
            .x_ff1_b
            .axis_iter(Axis(0))
            .next_back()
            .ok_or_else(|| "Could not get last element of x_ff1_b")?
            .to_owned()
            .insert_axis(Axis(0));

        // aka dL/dx_ff1_b
        let grad_l_hprime = grad_l_ff1_act * approx_gelu_grad(&last_x_ff1_b);

        let last_x_with_atn_out = train_data
            .x_with_atn_out
            .axis_iter(Axis(0))
            .next_back()
            .ok_or_else(|| "Could not get last element of x_with_atn_out")?
            .to_owned()
            .insert_axis(Axis(0));

        let grad_l_ff1 = last_x_with_atn_out.t().dot(&grad_l_hprime);

        dbg!(&grad_l_ff1);

        let grad_l_ff1_b = grad_l_hprime.clone();

        dbg!(&grad_l_ff1_b);

        // aka dL/dx_with_atn_out
        let grad_l_hatn = grad_l_hprime.dot(&self.ff1.t());

        dbg!(&grad_l_hatn);

        // FINAL atn_out MATRIX

        let last_atn_score = train_data
            .atn_scores_reshaped
            .axis_iter(Axis(0))
            .next_back()
            .ok_or_else(|| "Could not get last element of atn_scores_atn_out")?
            .to_owned()
            .insert_axis(Axis(0));

        let grad_l_atn_out = last_atn_score.t().dot(&grad_l_hatn);

        dbg!(&grad_l_atn_out);

        // aka dL/datn_scores_atn_out
        let grad_l_o = grad_l_hatn.dot(&self.atn_out.t());

        let grad_l_o_shp: Vec<_> = grad_l_o.axes().collect();

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

        // aka dL/dxqxk
        let mut grad_l_a: Array3<f32> = Array3::zeros(train_data.xqxk.dim());
        let mut grad_l_xvs: Array3<f32> = Array3::zeros(train_data.x_vs.dim());

        Ok(grad_l_hffn)
    }
}

#[cfg(test)]
mod tests {

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

    #[test]
    fn test_transformer_naive_fwd_happy_path() -> Result<(), ErrBox> {
        let ctx_size = 32_000;
        let vocab_size = 15;
        let embed_dim = 32;
        let num_layers = 32;
        let num_heads = 16;
        let ff_dim = 18;

        let mut t = Transformer::new(
            ctx_size, vocab_size, embed_dim, num_layers, num_heads, ff_dim,
        )?;

        let mut train_data = TransformerTrainData::default();

        let pred = t.naive_fwd(vec![1, 2, 3, 4, 5].as_slice(), Some(&mut train_data))?;

        // one-hot for token number 6
        let mut expected = Array1::zeros(vocab_size);
        expected[5] = 1.0;

        t.naive_bwd(0.001, &train_data, &expected)?;

        Ok(())
    }
}
