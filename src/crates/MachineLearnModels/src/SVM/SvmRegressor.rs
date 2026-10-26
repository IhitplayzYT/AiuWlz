use std::marker::PhantomData;
use Tensors::{dtype::Float, tensor::Tensor};
use super::kernel::Kernel;
use crate::{SVM::{common::SvmConfig, kernel::valid_kernel}, common::{Configured, Trained}, errors::ModelError};

pub struct SvmRegressor<T: Float, State> {
    x: Option<Tensor<T>>,
    y: Option<Tensor<T>>,
    conf: SvmConfig,
    model: Option<SvmRegressorModel<T>>,
    _state: PhantomData<State>,
}

#[derive(Clone)]
pub struct SvmRegressorModel<T: Float> {
    supports: Vec<T>,
    n_features: usize,
    coeff: Vec<T>,
    intercept: T,
    kernel: Kernel,
}

fn is_invalid_svm_conf(config: &SvmConfig) -> bool{
    !config.c.is_finite() || config.c <= 0.0 || !config.lr.is_finite() || config.lr <= 0.0 || config.lr > 1.0 || !config.epsilon.is_finite() || config.epsilon < 0.0 || config.epochs == 0 || !valid_kernel(&config.kernel)
}


impl<T: Float> SvmRegressor<T, Configured> {
    pub fn new(conf: SvmConfig) -> Result<Self, ModelError> {
        if is_invalid_svm_conf(&conf){ return Err(ModelError::ConfigError("SVM C and lr must be positive, epsilon non-negative, and epochs to be non-zero and positive".into())); }
        Ok(Self {x: None,y: None,conf,model: None,_state: PhantomData})
    }

    pub fn init(config: SvmConfig) -> Result<Self, ModelError> { Self::new(config) }

    pub fn set_data(mut self, x: Tensor<T>, y: Tensor<T>) -> Result<Self, ModelError> {
        validate_xy(&x, &y)?;
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }


    pub fn fit(self) -> Result<SvmRegressor<T, Trained>, ModelError> {
        let x = self.x.ok_or_else(|| ModelError::MissingData("X".into()))?;
        let y = self.y.ok_or_else(|| ModelError::MissingData("Y".into()))?;
        let n = x.shape()[0];
        let p = x.shape()[1];

        if n == 0 || p == 0 { return Err(ModelError::InvalidData("X has to be non-empty".into())); }
        if y.len(0).map_err(ModelError::TensorError)? != n { return Err(ModelError::InvalidData( "X and y must have the same number of samples".into())); }

        let x = x.to_vec()?;
        let y = y.to_vec()?;

        let c = T::from_f64(self.conf.c);
        let epsilon = T::from_f64(self.conf.epsilon);
        let alpha_eps = T::from_f64(1e-8);

        let zero = T::zero();
        let one = T::one();
        let two = T::from_f64(2.0);

        // Precompute kernel mat
        let mut kmat = vec![zero; n * n];
        for i in 0..n {
            let xi = &x[i * p..(i + 1) * p];
            for j in 0..=i {
                let xj = &x[j * p..(j + 1) * p];
                let v = self.conf.kernel.eval(xi, xj);
                kmat[i * n + j] = v;
                kmat[j * n + i] = v;
            }
        }

        let kernel = |i: usize, j: usize| -> T {kmat[i * n + j]};

        // Signed dual coefficients:
        // a[i] violation above epsilon
        // a*[i] violation below epsilon
        // beta[i] = a[i] - a*[i]
        // 
        // -C <= beta[i] <= C
        // sum(beta[i]) = 0

        let mut beta = vec![zero; n];

        // f[i] = sum_j beta[j] * K(x_j, x_i)
        let mut f = vec![zero; n];
        let mut no_change_pass = 0;
        let max_passes = self.conf.epochs;

        // Pairwise optimization.
        // Each update:
        // beta[i] += delta
        // beta[j] -= delta
        // hence sum(beta)=0

        while no_change_pass < max_passes {
            let mut changed = 0;

            for i in 0..n {
                for j in (i + 1)..n {
                    let bi = beta[i];
                    let bj = beta[j];

                    // -C <= bi + delta <= C
                    // -C <= bj - delta <= C
                    let lo = {
                        let a = -c - bi;
                        let b = bj - c;
                        if a > b { a } else { b }
                    };

                    let hi = {
                        let a = c - bi;
                        let b = bj + c;
                        if a < b { a } else { b }
                    };

                    if hi - lo <= alpha_eps { continue; }

                    // Smooth pairwise objective
                    //
                    // 0.5 * q * delta^2 + g * delta
                    //
                    // The remaining term is:
                    // epsilon * (|bi + delta| + |bj - delta|)

                    let q = kernel(i, i) + kernel(j, j) - two * kernel(i, j);

                    // q has to have non negative eigen values for kernel entries to be valid
                    if q < -alpha_eps { return Err(ModelError::InvalidData("Kernel matrix is not positive semidefinite".into())); }

                    let q = if q < zero { zero } else { q };
                    
                    let g = f[i] - f[j] - y[i] + y[j];

                    // Absolute value terms change slope at
                    // delta = -bi and delta = bj.
                    //
                    // Interval boundaries at stationary point to find its min.
                    let mut points = vec![lo, hi];

                    let break_i = -bi;
                    let break_j = bj;

                    if break_i > lo && break_i < hi { points.push(break_i); }

                    if break_j > lo && break_j < hi { points.push(break_j); }

                    points.sort_by(|a, b| {a.partial_cmp(b).unwrap()});

                    // Stationary points
                    let initial_len = points.len();

                    for k in 0..(initial_len - 1) {
                        let left = points[k];
                        let right = points[k + 1];

                        if right - left <= alpha_eps {continue;}
                        let mid = (left + right) / two;

                        let si = if bi + mid > zero {one} else if bi + mid < zero {-one} else {zero};

                        let sj = if bj - mid > zero {one} else if bj - mid < zero {-one} else {zero};

                        // q * delta + g + epsilon * (si - sj) = 0
                        if q > alpha_eps {
                            let candidate = -(g + epsilon * (si - sj)) / q;

                            if candidate > left && candidate < right { points.push(candidate);}
                        }
                    }

                    // If q is zero, the objective is piecewise linear;
                    // its minimum occurs at an interval boundary.
                    let mut best_delta = zero;
                    let mut best_objective: Option<T> = None;

                    // Smooth pairwise objective
                    // 0.5 * q * delta^2 + g * delta + epsilon * (|bi + delta| + |bj - delta|)

                    for delta in points {
                        if delta < lo || delta > hi {continue;}
                        let new_bi = bi + delta;
                        let new_bj = bj - delta;
                        let objective = (q * (delta * delta) / two) + g * delta + (epsilon * (new_bi.abs() + new_bj.abs() - bi.abs() - bj.abs()));
                        if best_objective.is_none() || objective < best_objective.unwrap(){
                            best_objective = Some(objective);
                            best_delta = delta;
                        }
                    }

                    if best_delta.abs() <= alpha_eps {continue;}

                    beta[i] = bi + best_delta;
                    beta[j] = bj - best_delta;

                    // Incrementally update predictions without bias.
                    for k in 0..n {
                        f[k] += best_delta * (kernel(i, k) - kernel(j, k));
                    }
                    changed += 1;
                }
            }

            if changed == 0 {
                no_change_pass += 1;
            } else {
                no_change_pass = 0;
            }
        }

        // Recover intercept from supports
        // For beta[i] != 0 and |beta[i]| < C:
        // b = y[i] - f[i] - epsilon * sign(beta[i])

        let mut bias_sum = zero;
        let mut bias_count = 0usize;

        for i in 0..n {
            if beta[i].abs() > alpha_eps && beta[i].abs() < c - alpha_eps{
                let sign = if beta[i] > zero {one} else {-one};
                bias_sum += y[i] - f[i] - epsilon * sign;
                bias_count += 1;
            }
        }

        // Fallback when no free support vectors exist usignn approximation

        let bias = if bias_count > 0 { bias_sum / T::from_f64(bias_count as f64)} else {
            let mut sum = zero;
            for i in 0..n {
                let sign = if beta[i] > zero {one} else if beta[i] < zero {-one} else {zero};
                sum += y[i] - f[i] - epsilon * sign;
            }
            sum / T::from_f64(n as f64)
        };

        let model = SvmRegressorModel {supports: x,n_features: p,coeff: beta,intercept: bias,kernel: self.conf.kernel.clone()};

        Ok(SvmRegressor {x: None,y: None,conf: self.conf,model: Some(model),_state: PhantomData})
    }
}


impl<T: Float> SvmRegressor<T, Trained> {
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> { self.model().predict(x) }
    pub fn model(&self) -> &SvmRegressorModel<T> { self.model.as_ref().expect("Failed to return refernce to trained model state") }
}

impl<T: Float> SvmRegressorModel<T> {
    pub fn n_features(&self) -> usize { self.n_features }

    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
        if x.ndim() != 2 || x.shape()[1] != self.n_features { return Err(ModelError::InvalidData(format!("Expected X with shape (samples, {}), got {:?}",self.n_features,x.shape()))); }
        let values = x.to_vec()?;
        let n = x.shape()[0];
        let count = self.supports.len() / self.n_features;
        let mut predictions = vec![T::zero(); n];
        for i in 0..n {
            let row = &values[i * self.n_features..(i + 1) * self.n_features];
            let mut value = self.intercept;
            for j in 0..count {
                let support = &self.supports[j * self.n_features..(j + 1) * self.n_features];
                value += self.coeff[j] * self.kernel.eval(row, support);
            }
            predictions[i] = value;
        }
        Ok(Tensor::from_vec(predictions, &[n, 1])?)
    }
}

fn validate_xy<T: Float>(x: &Tensor<T>, y: &Tensor<T>) -> Result<(), ModelError> {
    // x -> [n,p]
    if x.ndim() != 2 || x.shape()[0] == 0 || x.shape()[1] == 0 {
        return Err(ModelError::InvalidData("X must be a nonempty tensor of dim (N,P) where N is no of data points,P is no of Independent variables".into()));
    }
    // y -> [n,1] ,[n,]
    if !((y.ndim() == 1 && y.shape()[0] == x.shape()[0]) || (y.ndim() == 2 && y.shape() == [x.shape()[0], 1])){
        return Err(ModelError::InvalidData("Y must have shape (N,) or (N, 1) where N is no of target points and must be equal to no of data points in X".into(),));
    }
    Ok(())
}
