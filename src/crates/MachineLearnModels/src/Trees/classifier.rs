use Tensors::{dtype::Float, tensor::Tensor};

use crate::{Trees::{common::{ ClassifierTreeConfig, classifier_decision_fn, majority, predict_class, validate_predict_x, validate_x}}, errors::ModelError};

#[derive(Clone)]
pub enum ClassNode<T: Float> {
    Leaf(i64),
    Split {
        feature: usize,
        threshold: T,
        left: Box<ClassNode<T>>,
        right: Box<ClassNode<T>>,
    },
}

pub struct DecisionTreeClassifier<T: Float> {
   pub x: Option<Tensor<T>>,
   pub y: Option<Tensor<i64>>,
   pub config: ClassifierTreeConfig,
}

pub struct DecisionTreeClassifierModel<T: Float> {
   pub root: ClassNode<T>,
   pub n_features: usize,
   pub classes: Vec<i64>,
}

impl<T: Float> DecisionTreeClassifier<T> {
    pub fn new(config: ClassifierTreeConfig) -> Result<Self, ModelError> {
        validate_config(&config)?;
        Ok(Self {x: None,y: None,config})
    }

    pub fn set_data(mut self, x: Tensor<T>, y: Tensor<i64>) -> Result<Self, ModelError> {
        validate_x(&x)?;
        if !((y.ndim() == 1 && y.shape()[0] == x.shape()[0]) || (y.ndim() == 2 && y.shape() == [x.shape()[0], 1])) { return Err(ModelError::InvalidData("Y must contain one class label per row of X".into())); }
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }

    pub fn fit(self) -> Result<DecisionTreeClassifierModel<T>, ModelError> {
        let x = self.x.ok_or_else(|| ModelError::MissingData("X".into()))?;
        let y = self.y.ok_or_else(|| ModelError::MissingData("Y".into()))?;
        let n = x.shape()[0];
        let p = x.shape()[1];
        let values = x.to_vec()?;
        let labels = y.to_vec()?;
        let rows = (0..n).map(|i| values[i * p..(i + 1) * p].to_vec()).collect::<Vec<_>>();
        let indices = (0..n).collect::<Vec<_>>();
        let mut classes = labels.clone();
        classes.sort_unstable();
        classes.dedup();
        Ok(DecisionTreeClassifierModel {root: build_class_tree(&rows, &labels, &indices, 0, &self.config),n_features: p,classes})
    }
}

impl<T: Float> DecisionTreeClassifierModel<T> {
    pub fn classes(&self) -> &[i64] { &self.classes }
    
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<i64>, ModelError> {
        validate_predict_x(x, self.n_features)?;
        let values = x.to_vec()?;
        let p = self.n_features;
        let preds = (0..x.shape()[0]).map(|i| predict_class(&self.root, &values[i * p..(i + 1) * p])).collect();
        Ok(Tensor::from_vec(preds, &[x.shape()[0]])?)
    }
}

fn build_class_tree<T: Float>(rows: &[Vec<T>],labels: &[i64],indices: &[usize],depth: usize,cfg: &ClassifierTreeConfig) -> ClassNode<T> {
    let leaf = ClassNode::Leaf(majority(labels, indices));
    if depth >= cfg.max_depth || indices.len() < cfg.min_samples_split || indices.iter().all(|&i| labels[i] == labels[indices[0]]) { return leaf; }
    let parent = classifier_decision_fn(labels, indices,cfg);
    let (mut best_gain, mut best_feature, mut best_thresh) = (0.0, 0usize, T::zero());
    let mut best_parts = None;
    for feature in 0..rows[0].len() {
        let mut candidates = indices.iter().map(|&i| rows[i][feature]).collect::<Vec<_>>();
        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        candidates.dedup_by(|a, b| *a == *b);
        for pair in candidates.windows(2) {
            let thresh = (pair[0] + pair[1]) / T::from_f64(2.0);
            let (left, right): (Vec<_>, Vec<_>) = indices.iter().copied().partition(|&i| rows[i][feature] <= thresh);
            if left.is_empty() || right.is_empty() { continue;}
            let weighted = left.len() as f64 * classifier_decision_fn(labels, &left,cfg) + right.len() as f64 * classifier_decision_fn(labels, &right,cfg);
            let gain = parent - weighted / indices.len() as f64;
            if gain > best_gain {
                best_gain = gain;
                best_feature = feature;
                best_thresh = thresh;
                best_parts = Some((left,right));
            }
        }
    }
    match best_parts {
        Some((left, right)) if best_gain > 0.0 => ClassNode::Split {feature: best_feature,threshold: best_thresh,left: Box::new(build_class_tree(rows, labels, &left, depth + 1, cfg)),right: Box::new(build_class_tree(rows, labels, &right, depth + 1, cfg))},
        _ => leaf,
    }
}

pub fn validate_config(config: &ClassifierTreeConfig) -> Result<(), ModelError> {
    if config.min_samples_split < 2 {return Err(ModelError::ConfigError("min_samples_split must be at least 2".into()));}
    Ok(())
}