use std::marker::PhantomData;

use Tensors::{dtype::Float, error::TensorError, tensor::Tensor};

use crate::{Ensembles::common::{Configured, Sampling, Trained}, errors::ModelError};

pub struct RandomForest<T:Float,S>{
    n: usize,
    split: Sampling,
    x: Option<Tensor<T>>,
    y: Option<Tensor<T>>,
    _phantom: PhantomData<S>
}

impl <T:Float> Default for RandomForest<T,Configured>{
    fn default() -> Self {Self { n: 10, split: Sampling::Bagging,_phantom:PhantomData,x:None,y:None}}
}

impl<T:Float> RandomForest<T,Configured>{
    pub fn new(n:usize,split:Option<Sampling>) -> Self { Self { n, split: split.unwrap_or(Sampling::Bagging),_phantom:PhantomData,x:None,y:None} }

    pub fn set_data(mut self,x:Tensor<T>,y:Tensor<T>) -> Result<RandomForest<T, Configured>, ModelError>{
        if y.ndim() != 2 || x.ndim() != 2{ return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y has to be a 2d tensor".to_string()))); }        
        if y.shape()[0] != x.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y Tensors are of unequal length".to_string()))); } 
        if y.shape()[1] != 1{ return Err(ModelError::TensorError(TensorError::InvalidShape("Y has to be 2d tensor with 2nd dim being 1".to_string()))); } 
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }

    pub fn fit() -> Result<RandomForest<T,Trained>>{
        
    }

}





