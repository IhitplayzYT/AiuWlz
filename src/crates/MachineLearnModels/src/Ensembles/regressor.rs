use std::marker::PhantomData;

use Tensors::{dtype::Float, tensor::Tensor};
use rand::{seq::SliceRandom, Rng};

use crate::{
    Ensembles::common::{Configured, Sampling, Trained},
    errors::ModelError,
    Trees::{DecisionTreeRegressor, DecisionTreeRegressorModel, RegressorTreeConfig},
};

pub struct RandomForest<T: Float, S> {
    n: usize,
    split: Sampling,
    x: Option<Tensor<T>>,
    y: Option<Tensor<T>>,
    config: RegressorTreeConfig<T>,
    trees: Vec<DecisionTreeRegressorModel<T>>,
    _phantom: PhantomData<S>,
}

impl<T: Float> Default for RandomForest<T, Configured> {
    fn default() -> Self { Self {n: 10,split: Sampling::Bagging,x: None,y: None,config: RegressorTreeConfig::default(),trees: Vec::new(),_phantom: PhantomData} }
}

impl<T: Float> RandomForest<T, Configured> {
    pub fn new(n: usize, split: Option<Sampling>) -> Self {
        Self {n,split: split.unwrap_or(Sampling::Bagging),..Self::default()}
    }

    pub fn with_tree_config(mut self, config: RegressorTreeConfig<T>) -> Self {
        self.config = config;
        self
    }

    pub fn set_data(mut self, x: Tensor<T>, y: Tensor<T>) -> Result<Self, ModelError> {
        if x.ndim() != 2 || x.shape()[0] == 0 || x.shape()[1] == 0 { return Err(ModelError::InvalidData("X must be a nonempty (samples, features) tensor".into())); }
        if !((y.ndim() == 1 && y.shape()[0] == x.shape()[0]) || (y.ndim() == 2 && y.shape() == [x.shape()[0], 1])){ return Err(ModelError::InvalidData("Y must contain one numeric target per row of X".into())); }
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }

    pub fn fit(self) -> Result<RandomForest<T, Trained>, ModelError> {
        if self.n == 0 { return Err(ModelError::ConfigError("A random forest must contain at least one tree".into())); }
        let x = self.x.ok_or_else(|| ModelError::MissingData("X".into()))?;
        let y = self.y.ok_or_else(|| ModelError::MissingData("Y".into()))?;
        let n = x.shape()[0];
        let p = x.shape()[1];
        let x_values = x.to_vec()?;
        let y_values = y.to_vec()?;
        let mut tree_config = self.config.clone();
        if tree_config.max_features.is_none() { tree_config.max_features = Some((p as f64).sqrt().ceil() as usize); }
        let mut rng = rand::thread_rng();
        let mut trees = Vec::with_capacity(self.n);
        for _ in 0..self.n {
            let sample_indices = match &self.split {
                Sampling::Bagging => (0..n).map(|_| rng.gen_range(0..n)).collect::<Vec<_>>(),
                Sampling::Pasting => {
                    let mut indices = (0..n).collect::<Vec<_>>();
                    indices.shuffle(&mut rng);
                    indices
                }
            };

            let mut sampled_x = Vec::with_capacity(sample_indices.len() * p);
            let mut sampled_y = Vec::with_capacity(sample_indices.len());
            for index in sample_indices {
                sampled_x.extend_from_slice(&x_values[index * p..(index + 1) * p]);
                sampled_y.push(y_values[index]);
            }
            let tree_x = Tensor::from_vec(sampled_x, &[sampled_y.len(), p])?;
            let cnt = sampled_y.len();
            let tree_y = Tensor::from_vec(sampled_y, &[cnt, 1])?;
            let tree = DecisionTreeRegressor::new(tree_config.clone())?.set_data(tree_x, tree_y)?.fit()?;
            trees.push(tree);
        }
        Ok(RandomForest {n: self.n,split: self.split,x: None,y: None,config: self.config,trees,_phantom: PhantomData})
    }
}

impl<T: Float> RandomForest<T, Trained> {
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
        if self.trees.is_empty() { return Err(ModelError::ConceptError("Cannot predict with a forest that has no trained trees".into())); }
        let mut predictions = self.trees[0].predict(x)?.to_vec()?;
        for tree in &self.trees[1..] {
            for (sum, prediction) in predictions.iter_mut().zip(tree.predict(x)?.to_vec()?) { *sum += prediction; }
        }
        let divisor = T::from_f64(self.trees.len() as f64);
        for prediction in &mut predictions { *prediction /= divisor; }
        Ok(Tensor::from_vec(predictions, &[x.shape()[0], 1])?)
    }

    pub fn n_trees(&self) -> usize { self.trees.len() }
}
