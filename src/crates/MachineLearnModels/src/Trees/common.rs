use std::{cmp::Reverse, collections::{BTreeMap, HashMap}, fmt::Debug};

use Tensors::{dtype::Float, tensor::Tensor};

use crate::{Trees::{classifier::ClassNode, regressor::RegrNode}, errors::ModelError};


#[derive(Clone, Debug)]

pub enum ClassifierDecisionFn{
    Gini,
    Entropy,
    Custom(fn(&[i64],&[usize]) -> f64),
}

#[derive(Clone, Debug)]
pub enum RegressorDecisionFn<T:Float>{
    VarianceSum,
    Custom(fn(&[T],&[usize]) -> f64),
}

#[derive(Clone, Debug)]
pub enum Splitter {
    Best,
    Random,
}

#[derive(Clone, Debug)]
pub struct RegressorTreeConfig <T:Float>{
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub max_leaf_nodes: Option<usize>,
    pub min_impurity_decrease: f64,
    pub splitter: Splitter,
    pub decision_fn:RegressorDecisionFn<T> 
}

impl <T:Float> Default for  RegressorTreeConfig<T> {
    fn default() -> Self { 
        Self{max_depth:8,min_samples_split:2,min_samples_leaf:1,max_features:None,max_leaf_nodes:None,min_impurity_decrease:0.0,splitter:Splitter::Best,decision_fn:RegressorDecisionFn::VarianceSum} 
    }
}

impl <T:Float> RegressorTreeConfig<T>{
    pub fn new(max_depth: Option<usize>,min_samples_split: Option<usize>,min_samples_leaf: Option<usize>,max_features: Option<usize>,max_leaf_nodes: Option<usize>,min_impurity_decrease: Option<f64>,splitter: Option<Splitter>,decision_fn: Option<RegressorDecisionFn<T>>) -> Self{
        Self { max_depth: max_depth.unwrap_or(8), min_samples_split: min_samples_split.unwrap_or(2),min_samples_leaf: min_samples_leaf.unwrap_or(1),max_features,max_leaf_nodes,min_impurity_decrease: min_impurity_decrease.unwrap_or(0.0),splitter: splitter.unwrap_or(Splitter::Best),decision_fn: decision_fn.unwrap_or(RegressorDecisionFn::VarianceSum) }
    }
}


#[derive(Clone, Debug)]
pub struct ClassifierTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub max_leaf_nodes: Option<usize>,
    pub min_impurity_decrease: f64,
    pub splitter: Splitter,
    pub decision_fn: ClassifierDecisionFn
}

impl Default for ClassifierTreeConfig {
    fn default() -> Self { 
        Self {max_depth: 8,min_samples_split: 2,min_samples_leaf: 1,max_features: None,max_leaf_nodes: None,min_impurity_decrease: 0.0,splitter: Splitter::Best,decision_fn: ClassifierDecisionFn::Gini } 
    }
}

impl ClassifierTreeConfig{
    pub fn new(max_depth: Option<usize>,min_samples_split: Option<usize>,min_samples_leaf: Option<usize>,max_features: Option<usize>,max_leaf_nodes: Option<usize>,min_impurity_decrease: Option<f64>,splitter: Option<Splitter>,decision_fn: Option<ClassifierDecisionFn>) -> Self{
        Self { max_depth: max_depth.unwrap_or(8), min_samples_split: min_samples_split.unwrap_or(2),min_samples_leaf: min_samples_leaf.unwrap_or(1),max_features,max_leaf_nodes,min_impurity_decrease: min_impurity_decrease.unwrap_or(0.0),splitter: splitter.unwrap_or(Splitter::Best),decision_fn: decision_fn.unwrap_or(ClassifierDecisionFn::Gini) }
    }
}

pub fn validate_x<T: Float>(x: &Tensor<T>) -> Result<(), ModelError> {
    if x.ndim() != 2 || x.shape()[0] == 0 || x.shape()[1] == 0 { return Err(ModelError::InvalidData("X must be a nonempty (samples, features) tensor".into())); }
    Ok(())
}

pub fn validate_predict_x<T: Float>(x: &Tensor<T>, p: usize) -> Result<(), ModelError> {
    if x.ndim() != 2 || x.shape()[1] != p { return Err(ModelError::InvalidData(format!("Expected X with shape (samples, {p}), got {:?}",x.shape()))); }
    Ok(())
}


/// Recursive fxn to predict y during inference
pub fn predict_regr<T: Float>(node: &RegrNode<T>, row: &[T]) -> T {
    match node {
        RegrNode::Leaf(v) => *v,
        RegrNode::Split {feature,threshold,left,right} => predict_regr(if row[*feature] <= *threshold {left} else {right},row),
    }
}

/// Recursive fxn to classify y during inference
pub fn predict_class<T: Float>(node: &ClassNode<T>, row: &[T]) -> i64 {
    match node {
        ClassNode::Leaf(v) => *v,
        ClassNode::Split {feature,threshold,left,right} => predict_class(if row[*feature] <= *threshold {left} else {right},row),
    }
}

pub fn mean<T: Float>(targets: &[T], indices: &[usize]) -> T { indices.iter().fold(T::zero(), |s, &i| s + targets[i]) / T::from_f64(indices.len() as f64) }

/// (x - x_bar)^2
pub fn variance_sum<T: Float>(targets: &[T], indices: &[usize]) -> f64 {
    if indices.is_empty() { return 0.0; }
    let m = mean(targets, indices);
    indices.iter().map(|&i| { (targets[i] - m).to_f64().powi(2)}).sum()
}

// Max freq using btreemap 
pub fn majority(labels: &[i64], indices: &[usize]) -> i64 {
    let mut counts = BTreeMap::<i64, usize>::new();
    for &i in indices { *counts.entry(labels[i]).or_default() += 1; }
    counts.into_iter().max_by_key(|&(label, count)| (count, Reverse(label))).map(|(label, _)| label).unwrap_or_default()
}

// Gini impurity
pub fn gini(labels: &[i64], indices: &[usize]) -> f64 {
    if indices.is_empty() { return 0.0; }
    let mut counts = HashMap::<i64, usize>::new();
    for &i in indices { *counts.entry(labels[i]).or_default() += 1; }
    1.0 - counts.values().map(|&n| {(n as f64 / indices.len() as f64).powi(2)}).sum::<f64>()
}

// Entropy
pub fn entropy(labels:&[i64],indices:&[usize]) -> f64{
    if indices.is_empty() { return 0.0; }
    let mut counts = HashMap::<i64, usize>::new();
    for &i in indices { *counts.entry(labels[i]).or_default() += 1; }
    let tot_cnt = indices.len();
    counts.iter().fold(0.0, |acc,(_,&freq)| {
        let v = freq as f64 / tot_cnt as f64;
        acc + (-v * v.log2())
    })
}
    
pub fn classifier_decision_fn(labels:&[i64],indices:&[usize],cfg:&ClassifierTreeConfig) -> f64{
    match cfg.decision_fn{
        ClassifierDecisionFn::Gini => gini(labels,indices),
        ClassifierDecisionFn::Entropy => entropy(labels,indices),
        ClassifierDecisionFn::Custom(f) => f(labels,indices)
    }
}


pub fn regressor_decision_fn<T:Float>(targets:&[T],indices:&[usize],cfg:&RegressorTreeConfig<T>) -> f64{
    match cfg.decision_fn{
        RegressorDecisionFn::Custom(f) => f(targets,indices),
        RegressorDecisionFn::VarianceSum => variance_sum(targets,indices)
    }
}