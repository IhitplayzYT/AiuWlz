use Tensors::{dtype::Float, tensor::Tensor};

use crate::{Trees::{RegressorTreeConfig, common::{mean, predict_regr, regressor_decision_fn, validate_predict_x, validate_x, variance_sum}}, errors::ModelError};

#[derive(Clone)]
pub enum RegrNode<T: Float> {
    Leaf(T),
    Split {
        feature: usize,
        threshold: T,
        left: Box<RegrNode<T>>,
        right: Box<RegrNode<T>>,
    },
}

pub struct DecisionTreeRegressor<T: Float> {
    x: Option<Tensor<T>>,
    y: Option<Tensor<T>>,
    config: RegressorTreeConfig<T>,
}

pub struct DecisionTreeRegressorModel<T: Float> {
    root: RegrNode<T>,
    n_features: usize,
}

impl<T: Float> DecisionTreeRegressor<T> {
    pub fn new(config: RegressorTreeConfig<T>) -> Result<Self, ModelError> {
        validate_config(&config)?;
        Ok(Self {x: None,y: None,config})
    }

    pub fn set_data(mut self, x: Tensor<T>, y: Tensor<T>) -> Result<Self, ModelError> {
        validate_x(&x)?;
        if !((y.ndim() == 1 && y.shape()[0] == x.shape()[0]) || (y.ndim() == 2 && y.shape() == [x.shape()[0], 1])){
            return Err(ModelError::InvalidData( "Y must contain one numeric target per row of X".into()));
        }
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }

    pub fn fit(self) -> Result<DecisionTreeRegressorModel<T>, ModelError> {
        let x = self.x.ok_or_else(|| ModelError::MissingData("X".into()))?;
        let y = self.y.ok_or_else(|| ModelError::MissingData("Y".into()))?;
        let n = x.shape()[0];
        let p = x.shape()[1];
        let values = x.to_vec()?;
        let targets = y.to_vec()?;
        let rows = (0..n).map(|i| values[i * p..(i + 1) * p].to_vec()).collect::<Vec<_>>();
        let indices = (0..n).collect::<Vec<_>>();
        let root = build_regr_tree(&rows, &targets, &indices, 0, &self.config);
        Ok(DecisionTreeRegressorModel {root,n_features: p})
    }
}

impl<T: Float> DecisionTreeRegressorModel<T> {
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
        validate_predict_x(x, self.n_features)?;
        let values = x.to_vec()?;
        let p = self.n_features;
        Ok(Tensor::from_vec((0..x.shape()[0]).map(|i| predict_regr(&self.root, &values[i * p..(i + 1) * p])).collect(), &[x.shape()[0], 1])?)
    }
}


fn build_regr_tree<T: Float>(rows: &[Vec<T>],targets: &[T],indices: &[usize],depth: usize,cfg: &RegressorTreeConfig<T>) -> RegrNode<T> {
    let leaf = RegrNode::Leaf(mean(targets, indices));
    if depth >= cfg.max_depth || indices.len() < cfg.min_samples_split { return leaf; }
    let parent = variance_sum(targets, indices);
    let (mut best_gain, mut best_feature, mut best_thresh) = (0.0, 0usize, T::zero());
    let mut best_parts = None;
    for feature in 0..rows[0].len() {
        let mut candidates = indices.iter().map(|&i| rows[i][feature]).collect::<Vec<_>>();
        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        candidates.dedup_by(|a, b| *a == *b);
        for pair in candidates.windows(2) {
            let thresh = (pair[0] + pair[1]) / T::from_f64(2.0);
            let (left, right): (Vec<_>, Vec<_>) = indices.iter().copied().partition(|&i| rows[i][feature] <= thresh);
            if left.is_empty() || right.is_empty() { continue; }
            let gain = parent - regressor_decision_fn(targets, &left,cfg) - regressor_decision_fn(targets, &right,cfg);
            if gain > best_gain {
                best_gain = gain;
                best_feature = feature;
                best_thresh = thresh;
                best_parts = Some((left, right));
            }
        }
    }
    match best_parts {
        Some((left, right)) if best_gain > 0.0 => RegrNode::Split {feature: best_feature,threshold: best_thresh,left: Box::new(build_regr_tree(rows, targets, &left, depth + 1, cfg)),right: Box::new(build_regr_tree(rows, targets, &right, depth + 1, cfg))},
        _ => leaf,
    }
}

pub fn validate_config<T:Float>(config: &RegressorTreeConfig<T>) -> Result<(), ModelError> {
    if config.min_samples_split < 2 {return Err(ModelError::ConfigError("min_samples_split must be at least 2".into()));}
    Ok(())
}