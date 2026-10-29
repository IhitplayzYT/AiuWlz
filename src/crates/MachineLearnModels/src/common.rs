use Tensors::dtype::Float;

pub struct Configured;
pub struct Trained;

#[derive(Debug)]
pub enum Batch{
    Stochastic,
    MiniBatch(usize),
    NoBatch,
}

#[derive(Debug)]
pub enum Regularisation{
    L1(f64),
    L2(f64),
    Ridge(f64),
    Lasso(f64),
    ElastiNet(f64,f64)
}


pub fn euclid_dist<T:Float>(v1: &[T], v2: &[T]) -> T { v1.iter().zip(v2.iter()).fold(T::zero(), |acc, (a, b)| acc + (*a - *b) * (*a - *b)).sqrt() }