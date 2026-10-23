use crate::SVM::Kernel;

#[derive(Clone, Debug)]
pub struct SvmConfig {
    pub c: f64,
    pub epsilon: f64,
    pub lr: f64,
    pub epochs: usize,
    pub kernel: Kernel,
}

impl Default for SvmConfig {
    fn default() -> Self { Self {c: 1.0,epsilon: 0.1,lr: 0.01,epochs: 100,kernel: Kernel::Linear}}
}