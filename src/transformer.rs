
use crate::util::ErrBox;
use ndarray::{
    Array, Array1, Array2, Array3, Axis, AxisDescription, Dimension, Order, Zip,
};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rayon::iter::ParallelIterator;

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
        train_data: Option<&mut TransformerTrainData>,
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

        let mut x_refined = x_embedded.clone();

        for block in self.blocks.iter() {
            block.naive_fwd(&mut x_refined, tb_train_data_iter.next())?;
        }

        // pred_raw = x_embedded @ self.unembed
        let predictions_raw = x_embedded.dot(&self.unembed);

        // predictions = softmax(pred_raw)
        let predictions = softmax_axis(&predictions_raw, Axis(0));

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

        Ok(last_prediction.to_owned())
    }

    pub fn naive_bwd(
        &mut self,
        lr: f32,
        train_data: &TransformerTrainData,
        y_gt_onehot: &Array2<f32>,
    ) -> Result<(), ErrBox> {
        let z = &train_data.predictions_raw;
        let p = &train_data.predictions;
        let loss = -(y_gt_onehot * p.log10()).sum();

        let grad_l_z = p - y_gt_onehot;

        let grad_l_unembed = train_data.x_refined.t().dot(&grad_l_z);

        let grad_l_h = grad_l_z.dot(&self.unembed.t());

        let tb_iter = self
            .blocks
            .iter_mut()
            .zip(train_data.tb_train_data.iter())
            .rev();

        let mut grad_block_output = grad_l_h;
        for (tb, tb_train_data) in tb_iter {
            grad_block_output = tb.naive_bwd(lr, tb_train_data, grad_block_output)?;
        }

        let grad_l_x_embedded = grad_block_output;

        let mut grad_l_embed: Array2<f32> = Array2::zeros(self.embed.dim());

        for (grad_l_x_embedded_slice, tok_id) in
            grad_l_x_embedded.outer_iter().zip(train_data.x.iter())
        {
            let mut grad_l_embed_slice = grad_l_embed.index_axis_mut(Axis(0), *tok_id as usize);

            grad_l_embed_slice.assign(&(&grad_l_embed_slice + &grad_l_x_embedded_slice));
        }

        self.unembed = &self.unembed - grad_l_unembed * lr;
        self.embed = &self.embed - grad_l_embed * lr;

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
    /// X
    xs: Array2<f32>,
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

        // Save X for backprop
        let xs = x.clone();

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

                    // Softmax
                    let normalized =
                        softmax_axis(&(matmul_prod / (self.head_size as f32).sqrt()), Axis(0));

                    xqxk_act_slice.assign(&normalized);
                },
            );

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
                xs,
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
        lr: f32,
        train_data: &TBTrainData,
        grad_l_block_out: Array2<f32>,
    ) -> Result<Array2<f32>, ErrBox> {
        // FEEDFORWARD

        // a.k.a. dL/dx_with_ff
        let grad_l_hffn = grad_l_block_out;

        let grad_l_ff2 = train_data.x_ff1_act.t().dot(&grad_l_hffn);

        let grad_l_ff2_b = grad_l_hffn.clone();

        let grad_l_ff1_act = grad_l_hffn.dot(&self.ff2.t());

        // aka dL/dx_ff1_b
        let grad_l_hprime = grad_l_ff1_act * approx_gelu_grad(&train_data.x_ff1_b);

        let grad_l_ff1 = train_data.x_with_atn_out.t().dot(&grad_l_hprime);

        let grad_l_ff1_b = grad_l_hprime.clone();

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
        let grad_l_x = grad_l_x_per_head.sum_axis(Axis(0));

        // APPLY GRADIENTS
        self.ff2_b = &self.ff2_b - grad_l_ff2_b * lr;
        self.ff2 = &self.ff2 - grad_l_ff2 * lr;

        self.ff1_b = &self.ff1_b - grad_l_ff1_b * lr;
        self.ff1 = &self.ff1 - grad_l_ff1 * lr;

        self.atn_out = &self.atn_out - grad_l_atn_out * lr;

        self.query = &self.query - grad_l_query * lr;
        self.key = &self.key - grad_l_key * lr;
        self.value = &self.value - grad_l_value * lr;

        Ok(grad_l_x)
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

	let n_iter = 100;

        for _i in 0..n_iter {
            let mut train_data = TransformerTrainData::default();

            let x = vec![1, 2, 3, 4, 5];

            let pred = t.naive_fwd(x.as_slice(), Some(&mut train_data))?;

            let pred_tok_id = pred.indexed_iter().fold((0, 0.0), |cur_max, (idx, val)| {
                if *val > cur_max.1 {
                    (idx, *val)
                } else {
                    cur_max
                }
            });

            // Reuse X as Y_gt for training on the tokens leading up to the next token prediction
            let mut y_gt = x;

            // Final token desired prediction
            let expected = 9u32;

            if pred_tok_id.0 == expected as usize {
                return Ok(());
            }

            y_gt.push(expected);
            y_gt = (&y_gt[1..]).to_owned();

            let mut y_gt_onehot: Array2<f32> = Array2::zeros((y_gt.len(), vocab_size));

            Zip::from(y_gt_onehot.outer_iter_mut())
                .and(&y_gt)
                .par_for_each(|mut y_gt_onehot_row, y_gt_value| {
                    y_gt_onehot_row[*y_gt_value as usize] = 1.0;
                });

            t.naive_bwd(lr, &train_data, &y_gt_onehot)?;
        }

        Err(format!("Transformer would not predict 1 2 3 4 5 9 after {} iterations", n_iter).into())
    }
}
