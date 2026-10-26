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