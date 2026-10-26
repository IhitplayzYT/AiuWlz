pub mod Classifier{
use std::marker::PhantomData;
use Tensors::{dtype::Float, tensor::Tensor};
use Tensors::error::TensorError;

use crate::common::{Batch, Configured, Regularisation, Trained};
use crate::errors::ModelError::{self};

// =============================================LOGISTIC CLASSIFIER===============================================  

#[derive(Debug)]
pub struct LogisticClassifier<T: Float, S> {
    pub lr: f64,
    pub x: Option<Tensor<T>>,
    pub y: Option<Tensor<T>>,
    pub w: Option<Tensor<T>>,
    pub batch_size: usize,
    pub regularise: Option<Regularisation>,
    pub include_bias: bool,
    _state: PhantomData<S>,
}


// Configured state
impl<T: Float> LogisticClassifier<T, Configured> {

    pub fn init(lr: f64, include_bias: bool,batch: Batch,regularise: Option<Regularisation>) -> Self {
        Self {lr,x: None,y: None,w: None,include_bias,_state: PhantomData,batch_size: match batch{Batch::MiniBatch(v) => v,Batch::NoBatch => usize::MAX,Batch::Stochastic => 1 },regularise}
    }

    pub fn set_data(mut self,x: Tensor<T>,y: Tensor<T>) -> Result<Self, ModelError> {
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        if y.ndim() != 2 || x.ndim() != 2{ return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y has to be a 2d tensor".to_string()))); }        
        if y.shape()[0] != x.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y Tensors are of unequal length".to_string()))); } 
        if y.shape()[1] != 1{ return Err(ModelError::TensorError(TensorError::InvalidShape("Y has to be 2d tensor with 2nd dim being 1".to_string()))); } 
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }

    pub fn fit(self, epochs: usize) -> Result<LogisticClassifier<T, Trained>, ModelError> {
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        let x_data = self.x.as_ref().ok_or(ModelError::MissingData("X".to_string()))?;
        let y_data = self.y.as_ref().ok_or(ModelError::MissingData("Y".to_string()))?;
        let lr = T::from_f64(self.lr);
        let n = x_data.shape()[0];
        let p = x_data.shape()[1];
        let batch_size = self.batch_size.min(n);

        if n == 0 { return Err(ModelError::InvalidData("Empty training data".into()));}
        if self.batch_size == 0 {return Err(ModelError::InvalidData("Batch size must be positive".into()));}

        // Initialize weights
        let mut w = if self.include_bias { Tensor::<T>::zeros(&[p + 1, 1]) } else { Tensor::<T>::zeros(&[p, 1]) };

        let x_train = if self.include_bias {
            let b0 = Tensor::<T>::ones(&[n, 1]);
            Tensor::<T>::cat(&[&b0,x_data], 1)?
        } else { x_data.clone() };

        for _ in 0..epochs {
            for st in (0..n).step_by(batch_size){
                let ed = (st + batch_size).min(n);
                let bs = ed - st;
                let x_batch = x_train.narrow(0, st, bs)?;
                let y_batch = y_data.narrow(0, st, bs)?;
                let logits = x_batch.matmul(&w)?;
                let pred = logits.sigmoid()?;
                let errs = pred.sub(&y_batch)?;
                let mut grad = x_batch.t()?.matmul(&errs)?.div_scalar(T::from_f64(bs as f64))?;
                if let Some(ref z) = self.regularise{
                    match z{
                        Regularisation::L1(v) | Regularisation::Lasso(v) => {
                            let grad_reg = w.sign()?.mul_scalar(T::from_f64(*v))?;
                            let grad_reg = if self.include_bias{
                                // Zero out the first row (bias term) of grad_reg
                                let zero_row = Tensor::<T>::zeros(&[1, 1]);
                                let other_rows = grad_reg.narrow(0, 1, grad_reg.shape()[0] - 1)?;
                                Tensor::<T>::cat(&[&zero_row, &other_rows], 0)?
                            } else {
                                grad_reg
                            };
                            grad = grad.add(&grad_reg)?;
                        },
                        Regularisation::L2(v) | Regularisation::Ridge(v) => {
                            let grad_reg = w.mul_scalar(T::from_f64(*v))?;
                            let grad_reg = if self.include_bias{
                                // Zero out the first row (bias term) of grad_reg
                                let zero_row = Tensor::<T>::zeros(&[1, 1]);
                                let other_rows = grad_reg.narrow(0, 1, grad_reg.shape()[0] - 1)?;
                                Tensor::<T>::cat(&[&zero_row, &other_rows], 0)?
                            } else {
                                grad_reg
                            };
                            grad = grad.add(&grad_reg)?;

                        },
                        Regularisation::ElastiNet(l1,l2 ) => {
                            let grad_l1 = w.sign()?.mul_scalar(T::from_f64(*l1))?;
                            let grad_l1 = if self.include_bias{
                                // Zero out the first row (bias term) of grad_l1
                                let zero_row = Tensor::<T>::zeros(&[1, 1]);
                                let other_rows = grad_l1.narrow(0, 1, grad_l1.shape()[0] - 1)?;
                                Tensor::<T>::cat(&[&zero_row, &other_rows], 0)?
                            } else {
                                grad_l1
                            };

                            let grad_l2 = w.mul_scalar(T::from_f64(*l2))?;
                            let grad_l2 = if self.include_bias{
                                // Zero out the first row (bias term) of grad_l2
                                let zero_row = Tensor::<T>::zeros(&[1, 1]);
                                let other_rows = grad_l2.narrow(0, 1, grad_l2.shape()[0] - 1)?;
                                Tensor::<T>::cat(&[&zero_row, &other_rows], 0)?
                            } else {
                                grad_l2
                            };
                            grad = grad.add(&grad_l1.add(&grad_l2)?)?;
                        }
                    }
                }
                w = w.sub(&grad.mul_scalar(lr)?)?;
            }
        }
        let LogisticClassifier { lr, x, y, include_bias, batch_size,regularise, .. } = self;
        Ok(LogisticClassifier {lr,x,y,w: Some(w),include_bias,batch_size,_state: PhantomData,regularise})
    }
}

// Trained state
impl<T: Float> LogisticClassifier<T, Trained> {
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
        let w = self.w.as_ref().ok_or(ModelError::MissingData("w".to_string()))?;
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        if x.ndim() != 2 || x.shape()[1] != w.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape(format!("Expected shape with last dim {}, got {:?}", w.shape()[0], x.shape())))); }

        x.matmul(w).map_err(ModelError::TensorError)?.sigmoid().map_err(ModelError::TensorError)
    }

    pub fn weights(&self) -> &Tensor<T> { self.w.as_ref().expect("trained model must have weights") }
}



// =============================================PERCEPTRON CLASSIFIER===============================================  

pub struct Perceptron<T: Float, S> {
    pub lr: f64,
    pub x: Option<Tensor<T>>,
    pub y: Option<Tensor<T>>,
    pub w: Option<Tensor<T>>,
    pub batch_size: usize,
    pub include_bias: bool,
    _state: PhantomData<S>,
}

// Configured state
impl<T: Float> Perceptron<T, Configured> {

    pub fn init(lr: f64, include_bias: bool,batch: Batch) -> Self {
        Self {lr,x: None,y: None,w: None,include_bias,_state: PhantomData,batch_size: match batch{Batch::MiniBatch(v) => v,Batch::NoBatch => usize::MAX,Batch::Stochastic => 1 }}
    }

    pub fn set_data(mut self,x: Tensor<T>,y: Tensor<T>) -> Result<Self, ModelError> {
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        if y.ndim() != 2 || x.ndim() != 2{ return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y has to be a 2d tensor".to_string()))); }        
        if y.shape()[0] != x.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y Tensors are of unequal length".to_string()))); } 
        if y.shape()[1] != 1{ return Err(ModelError::TensorError(TensorError::InvalidShape("Y has to be 2d tensor with 2nd dim being 1".to_string()))); } 
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }

    pub fn fit(self, epochs: usize) -> Result<Perceptron<T, Trained>, ModelError> {
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        let x_data = self.x.as_ref().ok_or(ModelError::MissingData("X".to_string()))?;
        let y_data = self.y.as_ref().ok_or(ModelError::MissingData("Y".to_string()))?;
        let lr = T::from_f64(self.lr);
        let n = x_data.shape()[0];
        let p = x_data.shape()[1];
        let batch_size = self.batch_size.min(n);
        if n == 0 { return Err(ModelError::InvalidData("Empty training data".into()));}
        if self.batch_size == 0 {return Err(ModelError::InvalidData("Batch size must be positive".into()));}

        // Initialize weights
        let mut w = if self.include_bias { Tensor::<T>::zeros(&[p + 1, 1]) } else { Tensor::<T>::zeros(&[p, 1]) };

        let x_train = if self.include_bias {
            let b0 = Tensor::<T>::ones(&[n, 1]);
            Tensor::<T>::cat(&[&b0,x_data], 1)?
        } else { x_data.clone() };

        for _ in 0..epochs {
            for st in (0..n).step_by(batch_size){
                let ed = (st + batch_size).min(n);
                let bs = ed - st;
                let x_batch = x_train.narrow(0, st, bs)?;
                let y_batch = y_data.narrow(0, st, bs)?;
                let logits = x_batch.matmul(&w)?;
                let pred = logits.heavside()?;
                let errs = pred.sub(&y_batch)?;
                let grad = x_batch.t()?.matmul(&errs)?.div_scalar(T::from_f64(bs as f64))?;
                w = w.sub(&grad.mul_scalar(lr)?)?;
            }
        }
        let Perceptron { lr, x, y, include_bias, batch_size, .. } = self;
        Ok( Perceptron{lr,x,y,w: Some(w),include_bias,batch_size,_state: PhantomData})
    }
}

// Trained state
impl<T: Float> Perceptron<T, Trained> {
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
        let w = self.w.as_ref().ok_or(ModelError::MissingData("w".to_string()))?;
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        if x.ndim() != 2 || x.shape()[1] != w.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape(format!("Expected shape with last dim {}, got {:?}", w.shape()[0], x.shape())))); }
        x.matmul(w).map_err(ModelError::TensorError)?.heavside().map_err(ModelError::TensorError)
    }

    pub fn weights(&self) -> &Tensor<T> { self.w.as_ref().expect("trained model must have weights") }
    
}

// =============================================RIDGE CLASSIFIER===============================================  

#[derive(Debug)]
pub struct RidgeClassifier<T: Float, S> {
    pub lr: f64,
    pub x: Option<Tensor<T>>,
    pub y: Option<Tensor<T>>,
    pub w: Option<Tensor<T>>,
    pub lambda: f64, 
    pub batch_size: usize,
    pub include_bias: bool,
    _state: PhantomData<S>,
}


// Configured state
impl<T: Float> RidgeClassifier<T, Configured> {

    pub fn init(lr: f64,lambda:f64, include_bias: bool,batch: Batch) -> Self {
        Self {lr,x: None,y: None,w: None,include_bias,_state: PhantomData,batch_size: match batch{Batch::MiniBatch(v) => v,Batch::NoBatch => usize::MAX,Batch::Stochastic => 1 },lambda}
    }

    pub fn set_data(mut self,x: Tensor<T>,y: Tensor<T>) -> Result<Self, ModelError> {
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        if y.ndim() != 2 || x.ndim() != 2{ return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y has to be a 2d tensor".to_string()))); }        
        if y.shape()[0] != x.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y Tensors are of unequal length".to_string()))); } 
        if y.shape()[1] != 1{ return Err(ModelError::TensorError(TensorError::InvalidShape("Y has to be 2d tensor with 2nd dim being 1".to_string()))); } 
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }

    pub fn fit(self, epochs: usize) -> Result<RidgeClassifier<T, Trained>, ModelError> {
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        let x_data = self.x.as_ref().ok_or(ModelError::MissingData("X".to_string()))?;
        let y_data = self.y.as_ref().ok_or(ModelError::MissingData("Y".to_string()))?;
        let lr = T::from_f64(self.lr);
        let n = x_data.shape()[0];
        let p = x_data.shape()[1];
        let batch_size = self.batch_size.min(n);

        if n == 0 { return Err(ModelError::InvalidData("Empty training data".into()));}
        if self.batch_size == 0 {return Err(ModelError::InvalidData("Batch size must be positive".into()));}

        // Initialize weights
        let mut w = if self.include_bias { Tensor::<T>::zeros(&[p + 1, 1]) } else { Tensor::<T>::zeros(&[p, 1]) };

        let x_train = if self.include_bias {
            let b0 = Tensor::<T>::ones(&[n, 1]);
            Tensor::<T>::cat(&[&b0,x_data], 1)?
        } else { x_data.clone() };

        for _ in 0..epochs {
            for st in (0..n).step_by(batch_size){
                let ed = (st + batch_size).min(n);
                let bs = ed - st;
                let x_batch = x_train.narrow(0, st, bs)?;
                let y_batch = y_data.narrow(0, st, bs)?;
                let logits = x_batch.matmul(&w)?;
                let pred = logits.sigmoid()?;
                let errs = pred.sub(&y_batch)?;

                let grad_data = x_batch.t()?.matmul(&errs)?.div_scalar(T::from_f64(bs as f64))?;
                let grad_reg = w.mul_scalar(T::from_f64(self.lambda))?;

                let grad_reg = if self.include_bias{
                    // Zero out the first row (bias term) of grad_reg
                    let zero_row = Tensor::<T>::zeros(&[1, 1]);
                    let other_rows = grad_reg.narrow(0, 1, grad_reg.shape()[0] - 1)?;
                    Tensor::<T>::cat(&[&zero_row, &other_rows], 0)?
                } else {
                    grad_reg
                };
                w = w.sub(&grad_data.add(&grad_reg)?.mul_scalar(lr)?)?;
            }
        }
        let RidgeClassifier { lr, x, y, include_bias, batch_size, lambda,..} = self;
        Ok(RidgeClassifier {lr,x,y,w: Some(w),include_bias,batch_size,_state: PhantomData,lambda})
    }
}

// Trained state
impl<T: Float> RidgeClassifier<T, Trained> {
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
        let w = self.w.as_ref().ok_or(ModelError::MissingData("w".to_string()))?;
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        if x.ndim() != 2 || x.shape()[1] != w.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape(format!("Expected shape with last dim {}, got {:?}", w.shape()[0], x.shape())))); }
        x.matmul(w).map_err(ModelError::TensorError)?.sigmoid().map_err(ModelError::TensorError)
    }

    pub fn weights(&self) -> &Tensor<T> { self.w.as_ref().expect("trained model must have weights") }
}


// =============================================LASSO CLASSIFIER===============================================  

#[derive(Debug)]
pub struct LassoClassifier<T: Float, S> {
    pub lr: f64,
    pub x: Option<Tensor<T>>,
    pub y: Option<Tensor<T>>,
    pub w: Option<Tensor<T>>,
    pub lambda: f64, 
    pub batch_size: usize,
    pub include_bias: bool,
    _state: PhantomData<S>,
}


// Configured state
impl<T: Float> LassoClassifier<T, Configured> {

    pub fn init(lr: f64,lambda:f64, include_bias: bool,batch: Batch) -> Self {
        Self {lr,x: None,y: None,w: None,include_bias,_state: PhantomData,batch_size: match batch{Batch::MiniBatch(v) => v,Batch::NoBatch => usize::MAX,Batch::Stochastic => 1 },lambda}
    }

    pub fn set_data(mut self,x: Tensor<T>,y: Tensor<T>) -> Result<Self, ModelError> {
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        if y.ndim() != 2 || x.ndim() != 2{ return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y has to be a 2d tensor".to_string()))); }        
        if y.shape()[0] != x.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y Tensors are of unequal length".to_string()))); } 
        if y.shape()[1] != 1{ return Err(ModelError::TensorError(TensorError::InvalidShape("Y has to be 2d tensor with 2nd dim being 1".to_string()))); } 
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }

    pub fn fit(self, epochs: usize) -> Result<LassoClassifier<T, Trained>, ModelError> {
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        let x_data = self.x.as_ref().ok_or(ModelError::MissingData("X".to_string()))?;
        let y_data = self.y.as_ref().ok_or(ModelError::MissingData("Y".to_string()))?;
        let lr = T::from_f64(self.lr);
        let n = x_data.shape()[0];
        let p = x_data.shape()[1];
        let batch_size = self.batch_size.min(n);

        if n == 0 { return Err(ModelError::InvalidData("Empty training data".into()));}
        if self.batch_size == 0 {return Err(ModelError::InvalidData("Batch size must be positive".into()));}

        // Initialize weights
        let mut w = if self.include_bias { Tensor::<T>::zeros(&[p + 1, 1]) } else { Tensor::<T>::zeros(&[p, 1]) };

        let x_train = if self.include_bias {
            let b0 = Tensor::<T>::ones(&[n, 1]);
            Tensor::<T>::cat(&[&b0,x_data], 1)?
        } else { x_data.clone() };

        for _ in 0..epochs {
            for st in (0..n).step_by(batch_size){
                let ed = (st + batch_size).min(n);
                let bs = ed - st;
                let x_batch = x_train.narrow(0, st, bs)?;
                let y_batch = y_data.narrow(0, st, bs)?;
                let logits = x_batch.matmul(&w)?;
                let pred = logits.sigmoid()?;
                let errs = pred.sub(&y_batch)?;

                let grad_data = x_batch.t()?.matmul(&errs)?.div_scalar(T::from_f64(bs as f64))?;
                let grad_reg = w.sign()?.mul_scalar(T::from_f64(self.lambda))?;

                let grad_reg = if self.include_bias{
                    // Zero out the first row (bias term) of grad_reg
                    let zero_row = Tensor::<T>::zeros(&[1, 1]);
                    let other_rows = grad_reg.narrow(0, 1, grad_reg.shape()[0] - 1)?;
                    Tensor::<T>::cat(&[&zero_row, &other_rows], 0)?
                } else {
                    grad_reg
                };
                w = w.sub(&grad_data.add(&grad_reg)?.mul_scalar(lr)?)?;
            }
        }
        let LassoClassifier { lr, x, y, include_bias, batch_size, lambda,..} = self;
        Ok(LassoClassifier {lr,x,y,w: Some(w),include_bias,batch_size,_state: PhantomData,lambda})
    }
}

// Trained state
impl<T: Float> LassoClassifier<T, Trained> {
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
        let w = self.w.as_ref().ok_or(ModelError::MissingData("w".to_string()))?;
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        if x.ndim() != 2 || x.shape()[1] != w.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape(format!("Expected shape with last dim {}, got {:?}", w.shape()[0], x.shape())))); }
        x.matmul(w).map_err(ModelError::TensorError)?.sigmoid().map_err(ModelError::TensorError)
    }

    pub fn weights(&self) -> &Tensor<T> { self.w.as_ref().expect("trained model must have weights") }
}


// =============================================ELASTICNET CLASSIFIER===============================================  

#[derive(Debug)]
pub struct ElasticNetClassifier<T: Float, S> {
    pub lr: f64,
    pub x: Option<Tensor<T>>,
    pub y: Option<Tensor<T>>,
    pub w: Option<Tensor<T>>,
    pub l1: f64, 
    pub l2: f64,
    pub batch_size: usize,
    pub include_bias: bool,
    _state: PhantomData<S>,
}


// Configured state
impl<T: Float> ElasticNetClassifier<T, Configured> {

    pub fn init(lr: f64,l1:f64,l2:f64,include_bias: bool,batch: Batch) -> Self {
        Self {lr,x: None,y: None,w: None,include_bias,_state: PhantomData,batch_size: match batch{Batch::MiniBatch(v) => v,Batch::NoBatch => usize::MAX,Batch::Stochastic => 1 },l1,l2}
    }

    pub fn set_data(mut self,x: Tensor<T>,y: Tensor<T>) -> Result<Self, ModelError> {
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        if y.ndim() != 2 || x.ndim() != 2{ return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y has to be a 2d tensor".to_string()))); }        
        if y.shape()[0] != x.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape("X and Y Tensors are of unequal length".to_string()))); } 
        if y.shape()[1] != 1{ return Err(ModelError::TensorError(TensorError::InvalidShape("Y has to be 2d tensor with 2nd dim being 1".to_string()))); } 
        self.x = Some(x);
        self.y = Some(y);
        Ok(self)
    }

    pub fn fit(self, epochs: usize) -> Result<ElasticNetClassifier<T, Trained>, ModelError> {
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        let x_data = self.x.as_ref().ok_or(ModelError::MissingData("X".to_string()))?;
        let y_data = self.y.as_ref().ok_or(ModelError::MissingData("Y".to_string()))?;
        let lr = T::from_f64(self.lr);
        let n = x_data.shape()[0];
        let p = x_data.shape()[1];
        let batch_size = self.batch_size.min(n);

        if n == 0 { return Err(ModelError::InvalidData("Empty training data".into()));}
        if self.batch_size == 0 {return Err(ModelError::InvalidData("Batch size must be positive".into()));}

        // Initialize weights
        let mut w = if self.include_bias { Tensor::<T>::zeros(&[p + 1, 1]) } else { Tensor::<T>::zeros(&[p, 1]) };

        let x_train = if self.include_bias {
            let b0 = Tensor::<T>::ones(&[n, 1]);
            Tensor::<T>::cat(&[&b0,x_data], 1)?
        } else { x_data.clone() };

        for _ in 0..epochs {
            for st in (0..n).step_by(batch_size){
                let ed = (st + batch_size).min(n);
                let bs = ed - st;
                let x_batch = x_train.narrow(0, st, bs)?;
                let y_batch = y_data.narrow(0, st, bs)?;
                let logits = x_batch.matmul(&w)?;
                let pred = logits.sigmoid()?;
                let errs = pred.sub(&y_batch)?;

                let grad_data = x_batch.t()?.matmul(&errs)?.div_scalar(T::from_f64(bs as f64))?;
                let grad_l1 = w.sign()?.mul_scalar(T::from_f64(self.l1))?;
                let grad_l2= w.mul_scalar(T::from_f64(self.l2))?;

                let grad_l1 = if self.include_bias{
                    // Zero out the first row (bias term) of grad_reg
                    let zero_row = Tensor::<T>::zeros(&[1, 1]);
                    let other_rows = grad_l1.narrow(0, 1, grad_l1.shape()[0] - 1)?;
                    Tensor::<T>::cat(&[&zero_row, &other_rows], 0)?
                } else {
                    grad_l1
                };

                let grad_l2 = if self.include_bias{
                    // Zero out the first row (bias term) of grad_reg
                    let zero_row = Tensor::<T>::zeros(&[1, 1]);
                    let other_rows = grad_l2.narrow(0, 1, grad_l2.shape()[0] - 1)?;
                    Tensor::<T>::cat(&[&zero_row, &other_rows], 0)?
                } else {
                    grad_l2
                };

                w = w.sub(&grad_data.add(&grad_l1.add(&grad_l2)?)?.mul_scalar(lr)?)?;
            }
        }
        let ElasticNetClassifier { lr, x, y, include_bias, batch_size, l1,l2,..} = self;
        Ok(ElasticNetClassifier {lr,x,y,w: Some(w),include_bias,batch_size,_state: PhantomData,l1,l2})
    }
}

// Trained state
impl<T: Float> ElasticNetClassifier<T, Trained> {
    pub fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
        let w = self.w.as_ref().ok_or(ModelError::MissingData("w".to_string()))?;
        // x -> (n,p)
        // y -> (n,1)
        // w -> (p,1)
        // b -> (n,1)
        if x.ndim() != 2 || x.shape()[1] != w.shape()[0] { return Err(ModelError::TensorError(TensorError::InvalidShape(format!("Expected shape with last dim {}, got {:?}", w.shape()[0], x.shape())))); }
        x.matmul(w).map_err(ModelError::TensorError)?.sigmoid().map_err(ModelError::TensorError)
    }

    pub fn weights(&self) -> &Tensor<T> { self.w.as_ref().expect("trained model must have weights") }
}


}