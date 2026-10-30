use Tensors::error::TensorError;
pub type Result<T> = std::result::Result<T, TensorError>;

pub(crate) fn invalid(message: impl Into<String>) -> TensorError {
    TensorError::InvalidShape(message.into())
}

/// Elementwise activation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
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

    Zeros,
    Ones,
    Constant(f64),
}

/// Objective used by [`crate::nn::Sequential::fit`].
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
