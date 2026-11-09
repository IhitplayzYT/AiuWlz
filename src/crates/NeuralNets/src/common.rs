use Tensors::{dtype::Float, error::TensorError, tensor::Tensor};
pub type Result<T> = std::result::Result<T, TensorError>;

pub(crate) fn invalid(message: impl Into<String>) -> TensorError {
    TensorError::InvalidShape(message.into())
}

/// Elementwise activation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    None,
    Linear,
    Relu,
    LeakyRelu(f64),
    Sigmoid,
    Tanh,
    Gelu,
    Elu(f64),
    Selu,
    SiLu,
    Softplus,
    Mish,
    Softsign,
    HardSigmoid,
    HardTanh {min: f64,max: f64},
    Softmax,
}

/// Weight init strats
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Initializer {
    He,
    Kaimang,

    Xavier,
    Glorot,

    Normal,
    Uniform { bound: f64 },

    KaimingUniform,
    HeUniform,

    KaimingNormal,
    HeNormal,

    XavierUniform,
    GlorotUniform,

    LeCunNormal,
    LeCunUniform,

    Zeros,
    Ones,
    Constant(f64),
}


#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    MeanSquaredError,
    MeanAbsoluteError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    BinaryCrossEntropyWithLogits,
    CategoricalCrossEntropyWithLogits,
    Huber {delta: f64},
    SmoothL1,
    KullbackLeibler,
    Hinge,
    PoissonNll,
}

#[derive(Clone, Debug)]
pub struct FitOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub shuffle: bool,
    pub seed: u64,
}

impl Default for FitOptions {
    fn default() -> Self { Self {epochs: 100,batch_size: 32,shuffle: true,seed: 42} }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub loss: f64,
    pub samples: usize,
}


pub fn valid_unit(x: f64) -> bool {x.is_finite() && (0.0..1.0).contains(&x)}
pub fn valid_positive(x: f64) -> bool {x.is_finite() && x > 0.0}
pub fn valid_nonnegative(x: f64) -> bool {x.is_finite() && x >= 0.0}

pub fn validate_feature_input<T: Float>(input: &Tensor<T>, features: usize, name: &str) -> Result<()> {
    if input.ndim() < 2 || input.shape().last() != Some(&features) { return Err(invalid(format!("{name} expects final dimension {features}, got {:?}",input.shape()))); }
    Ok(())
}

pub fn flatten<T: Float>(input: &Tensor<T>) -> Result<Tensor<T>> {
    if input.ndim() < 2 {return Err(invalid("Flatten layer requires batch dim and at least 1 feature dim"));}
    Ok(input.reshape(&[input.shape()[0] as isize, -1])?)
}

pub fn validate_activation(activation: Activation) -> Result<()> {
    match activation {
        Activation::LeakyRelu(a) | Activation::Elu(a) if !a.is_finite() || a < 0.0 => Err(invalid("activation coeff must be finite and non negative")),
        Activation::HardTanh { min, max } if !min.is_finite() || !max.is_finite() || min >= max => {Err(invalid("hardtanh requires finite min & max where min < max"))}
        _ => Ok(()),
    }
}
