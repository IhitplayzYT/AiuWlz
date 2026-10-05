use std::{error::Error, fmt};
use Utilities::device::Device::Device;

#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch { op: &'static str, lhs: Vec<usize>, rhs: Vec<usize> },
    InvalidShape(String),
    InvalidDim { dim: isize, ndim: usize },
    DeviceMismatch { lhs: Device, rhs: Device },
    Unsupported(String),
    Cuda(String),
    Custom(String)
}

pub type TensorResult<T> = std::result::Result<T, TensorError>;
pub type Result<T> = std::result::Result<T, TensorError>;

impl Error for TensorError {}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { op, lhs, rhs } => {write!(f, "Shape mismatch for `{op}`, {lhs:?} != {rhs:?}")}
            TensorError::InvalidShape(s) => write!(f, "Invalid shape: {s}"),
            TensorError::InvalidDim { dim, ndim } => {write!(f, "Dim {dim} not present for tensor with {ndim} dims")}
            TensorError::DeviceMismatch { lhs, rhs } => {write!(f, "Device mismatch: {lhs} != {rhs}")}
            TensorError::Unsupported(s) => write!(f, "Unsupported: {s}"),
            TensorError::Cuda(s) => write!(f, "Cuda error: {s}"),
            TensorError::Custom(s) => write!(f, "Error: {s}"),
        }
    }
}

