use Tensors::dtype::Float;

// SVM kernels
#[derive(Clone, Debug, PartialEq)]
pub enum Kernel {
    Linear, // aT * b
    Polynomial { degree: u32, gamma: f64, coef0: f64 }, // (gamma * aT * b + coef0) ^ degree
    Rbf { gamma: f64 }, // e ^ (-gamma * ||a-b||^2)
    Sigmoid { gamma: f64,coef0:f64 } // tanh(gamma * aT * b + coef0)
}

impl Default for Kernel {
    fn default() -> Self { Self::Linear }
}

impl Kernel {
    pub(crate) fn eval<T: Float>(&self, lhs: &[T], rhs: &[T]) -> T {
        let dot = lhs.iter().zip(rhs).fold(T::zero(), |sum, (&a, &b)| sum + a * b);
        match *self {
            Self::Linear => dot,
            Self::Polynomial {degree,gamma,coef0} => (dot * T::from_f64(gamma) + T::from_f64(coef0)).powf(T::from_f64(degree as f64)),
            Self::Rbf { gamma } => {
                let dist = lhs.iter().zip(rhs).fold(T::zero(), |acc, (&a, &b)| {
                    let d = a - b;
                    acc + d * d
                });
                (-T::from_f64(gamma) * dist).exp()
            },
            Self::Sigmoid { gamma, coef0 } => { (dot * T::from_f64(gamma) + T::from_f64(coef0)).tanh() }
        }
    }
}

pub fn valid_kernel(kernel: &Kernel) -> bool {
    match kernel {
        Kernel::Linear => true,
        Kernel::Polynomial { degree, gamma, coef0 } => *degree > 0 && gamma.is_finite() && *gamma > 0.0 && coef0.is_finite(),
        Kernel::Rbf { gamma } => gamma.is_finite() && *gamma > 0.0,
        Kernel::Sigmoid { gamma, coef0 } =>  gamma.is_finite() && *gamma > 0.0 && coef0.is_finite()
    }
}