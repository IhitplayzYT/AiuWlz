use std::marker::PhantomData;
use Tensors::{dtype::Float, tensor::Tensor};
use super::kernel::Kernel;
use crate::{SVM::{common::SvmConfig, kernel::valid_kernel}, common::{Configured, Trained}, errors::ModelError};

pub struct SvmClassifier<T: Float, State> {
    x: Option<Tensor<T>>,
    y: Option<Tensor<i64>>,
    config: SvmConfig,
    model: Option<SvmClassifierModel<T>>,
    _state: PhantomData<State>,
}

#[derive(Clone)]
pub struct SvmClassifierModel<T: Float> {
    support_vectors: Vec<T>,
    n_features: usize,
    classes: Vec<i64>,
    coefficients: Vec<T>, // class-major: classes x training rows
    intercepts: Vec<T>,
    kernel: Kernel,
}

impl<T: Float> SvmClassifier<T, Configured> {
    pub fn new(config: SvmConfig) -> Result<Self, ModelError> {
        if !config.c.is_finite() || config.c <= 0.0 || !config.lr.is_finite() || config.lr <= 0.0 || config.lr > 1.0 || config.epochs == 0 || !valid_kernel(&config.kernel){
            return Err(ModelError::ConfigError( "SVM C and lr must be positive,epochs must be non-zero and kernel has o be valid".into(), ));
        }
        Ok(Self {x: None,y: None,config,model: None,_state: PhantomData})
    }

    pub fn init(config: SvmConfig) -> Result<Self, ModelError> { Self::new(config) }

    pub fn set_data(mut self, x: Tensor<T>, y: Tensor<i64>) -> Result<Self, ModelError> {
        if x.ndim() != 2 || x.shape()[0] == 0 || x.shape()[1] == 0 { return Err(ModelError::InvalidData( "X must be a nonempty (samples, features) tensor".into())); }
        if !((y.ndim() == 1 && y.shape()[0] == x.shape()[0]) || (y.ndim() == 2 && y.shape() == [x.shape()[0], 1])){ return Err(ModelError::InvalidData( "Y must contain one label per row of X".into())); }
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }

    /// SMO for SVM
    /// f(x) = sum(alpha_i * y_i * K(x_i,x)) + b
    // Made with major AI assistance
    pub fn fit(self) -> Result<SvmClassifier<T, Trained>, ModelError> {
        let x = self.x.ok_or_else(|| ModelError::MissingData("X".into()))?;
        let y = self.y.ok_or_else(|| ModelError::MissingData("Y".into()))?;
        let n = x.shape()[0];
        let p = x.shape()[1];
        if n == 0 || p == 0 { return Err(ModelError::InvalidData("X has to be non-empty".into())); }
        let mut classes = y.to_vec()?;
        if classes.len() < 2 { return Err(ModelError::InvalidData("Classification requires at least two class".into())); }
        classes.sort_unstable();
        classes.dedup();

        let x = x.to_vec()?;
        let y = y.to_vec()?;

        let tolerance = T::from_f64(1e-3);
        let alpha_eps = T::from_f64(1e-8);
        let c = T::from_f64(self.config.c);

        let zero = T::zero();
        let one = T::one();
        let two = T::from_f64(2.0);

        let mut kmat = vec![zero; n * n]; // Precomputed kernel mat
        for i in 0..n {
            let xis = i * p; 
            let xi = &x[xis..xis+p]; // Get data sample i
            for j in 0..=i{
                let xjs = j * p;
                let xj = &x[xjs..xjs+p]; // Get data sample j
                let v = self.config.kernel.eval(xi, xj);
                kmat[i * n + j] = v;
                kmat[j * n + i] = v;
            }
        }
        let kernel = |i: usize, j: usize| -> T { kmat[i * n + j] };
        
        let mut coefficients = vec![zero; classes.len() * n];
        let mut intercepts = vec![zero; classes.len()];
        
        
        // Optimising
        // f(x) = sum(alpha_i * y_i * K(x_i,x)) + b
        for (class_idx, class) in classes.iter().enumerate() {
            let trgt:Vec<T> = y.iter().map(|l| {if l == class{one}else{-one}}).collect();

            let mut alpha = vec![zero; n];
            let mut bias = zero;
            let mut errors = vec![zero; n];

            // E_i = f(xi) - yi, f(xi)=0 for init
            for i in 0..n { errors[i] = -trgt[i]; }

            let mut no_change_pass = 0;
            let max_passes = self.config.epochs;

            while no_change_pass < max_passes{
                let mut changed = 0;
                for i in 0..n{
                    let yi = trgt[i];
                    let ei = errors[i];
                    let ye = yi * ei;

                    // KKT Violation
                    if !(ye < -tolerance && alpha[i] < c) || (ye > tolerance && alpha[i] > zero) { continue; }

                    // Find max |e_i - e_j| to have optimal optimisation of the second alpha
                    let mut best_j = None;
                    let mut best_diff = zero;
                    for j in 0..n {
                        if i == j { continue;}
                        let delta = errors[i] - errors[j];
                        let magnitude = if delta < zero {-delta} else {delta};
                        if best_j.is_none() || magnitude > best_diff {
                            best_diff = magnitude;
                            best_j = Some(j);
                        }
                    }
                
                    // if n = 0
                    let Some(j) = best_j else {continue;};

                    let o_ai = alpha[i];
                    let o_aj = alpha[j];
                    let yj = trgt[j];
                    let ej = errors[j];

                    // For different labels:
                    // L=max(0,alpha_j-alpha_i)
                    // H=min(C,C+alpha_j-alpha_i)

                    // For equal labels:
                    // L=max(0,alpha_i+alpha_j-C)
                    // H=min(C,alpha_i+alpha_j)

                    let (l,h) = if yi != yj {
                        let l = if o_aj > o_ai {o_aj - o_ai}else{zero};
                        let h = c + (if c + o_aj - o_ai < zero {zero} else{o_aj - o_ai});
                        (l,h)
                    } else{
                        let tot_a = o_ai + o_aj;
                        let l = if tot_a > c { tot_a - c } else { zero };
                        let h = if tot_a < c { tot_a } else { c };
                        (l, h)   
                    };
                    // Very small updates get skipped
                    if h - l <= alpha_eps{ continue }
                    // eta = 2kij - kii - kjj

                    let eta = two * kernel(i, j) - kernel(i, i) - kernel(j, j);
                    if eta >= zero { continue; }

                    // Update alpha_j, then clip it to [L, H].
                    let mut n_aj = o_aj - yj * (ei - ej) / eta;
                    n_aj = n_aj.clamp(l,h);

                    if (n_aj - o_aj).abs() < alpha_eps { continue; }

                    let n_ai = o_ai + yi * yj * (o_aj - n_aj);

                    alpha[i] = n_ai;
                    alpha[j] = n_aj;
                    let b1 = bias - ei - yi * (n_ai - o_ai) * kernel(i, i) - yj * (n_aj - o_aj) * kernel(i, j);
                    let b2 = bias - ej - yi * (n_ai - o_ai) * kernel(i, j) - yj * (n_aj - o_aj) * kernel(j, j);
                    bias = if n_ai > zero && n_ai < c { b1 } else if n_aj > zero && n_aj < c { b2 } else { (b1 + b2) / two};
                    for k in 0..n {
                        let diff_i = yi * (n_ai - o_ai);
                        let diff_j = yj * (n_aj - o_aj);
                        errors[k] += diff_i * kernel(i, k) + diff_j * kernel(j, k) + (bias - (b1 + b2) / two);
                    }
                    changed += 1;
                }
                if changed == 0 { no_change_pass += 1; } else { no_change_pass = 0;}
            }
            let offset = class_idx * n;
            for i in 0..n { coefficients[offset + i] = alpha[i] * trgt[i]; }
            intercepts[class_idx] = bias;
        }
        let model = SvmClassifierModel {support_vectors: x,n_features: p,classes,coefficients,intercepts,kernel: self.config.kernel.clone()};
        Ok(SvmClassifier {x: None,y: None,config: self.config,model: Some(model),_state: PhantomData})
    }
}



impl<T: Float> SvmClassifier<T, Trained> {
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<i64>, ModelError> { self.model().predict(x) }
    pub fn decision_function(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> { self.model().decision_function(x) }
    pub fn classes(&self) -> &[i64] { &self.model().classes }
    pub fn model(&self) -> &SvmClassifierModel<T> { self.model.as_ref().expect("Trained model state ref failed") }
}

impl<T: Float> SvmClassifierModel<T> {
    pub fn classes(&self) -> &[i64] { &self.classes }
    pub fn n_features(&self) -> usize { self.n_features }

    pub fn decision_function(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
        validate_features(x, self.n_features)?;
        let values = x.to_vec()?;
        let n = x.shape()[0];
        let support_count = self.support_vectors.len() / self.n_features;
        
        let mut scores = vec![T::zero(); n * self.classes.len()];
        for i in 0..n {
            let r_st = self.n_features + i;
            let row = &values[r_st..r_st + self.n_features];
            for c in 0..self.classes.len() {
                let mut score = self.intercepts[c];
                for j in 0..support_count {
                    let s_st = j * self.n_features;
                    let support = &self.support_vectors[s_st..s_st + self.n_features];
                    score += self.coefficients[c * support_count + j] * self.kernel.eval(row, support);
                }
                scores[i * self.classes.len() + c] = score;
            }
        }
        Ok(Tensor::from_vec(scores, &[n, self.classes.len()])?)
    }

    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<i64>, ModelError> {
        let scores = self.decision_function(x)?.to_vec()?;
        let k = self.classes.len();
        let predictions = scores.chunks(k).map(|row| {
                let best = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0);
                self.classes[best]
            }).collect();
        Ok(Tensor::from_vec(predictions, &[x.shape()[0]])?)
    }
}

fn validate_features<T: Float>(x: &Tensor<T>, expected: usize) -> Result<(), ModelError> {
    if x.ndim() != 2 || x.shape()[1] != expected { return Err(ModelError::InvalidData(format!("Expected X with shape (samples, {expected}), got {:?}",x.shape()))); }
    Ok(())
}
