use crate::dtype::{Element, Float};
use crate::error::{TensorResult, TensorError};
use crate::layout::broadcast_shapes;
use crate::par;
use crate::storage::Storage;
use crate::tensor::Tensor;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// OP codes in CUDA kernels
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp { Neg, Abs, Exp, Ln, Sqrt, Sin, Cos, Tanh, Sigmoid, Relu, Recip, Square, Gelu, Floor, Ceil }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp { Add, Sub, Mul, Div, Max, Min, Eq, Gt, Lt, Ge, Le, Pow }


pub trait IntLike{}
pub trait FloatLike{}
pub trait SignedLike{}

impl IntLike for i8{}
impl IntLike for i16{}
impl IntLike for i32{}
impl IntLike for i64{}
impl IntLike for isize{}
impl IntLike for u8{}
impl IntLike for u16{}
impl IntLike for u32{}
impl IntLike for u64{}
impl IntLike for usize{}

impl FloatLike for f32{}
impl FloatLike for f64{}

impl SignedLike for i8{}
impl SignedLike for i16{}
impl SignedLike for i32{}
impl SignedLike for i64{}
impl SignedLike for isize{}
impl SignedLike for f32{}
impl SignedLike for f64{}


impl<T: Element> Tensor<T> {

    /// Apply Unary op on tensors
    pub fn unary_with(&self, _op: UnaryOp, f: impl Fn(T) -> T + Sync + Send) -> TensorResult<Self> {
        match &*self.storage {
            Storage::Cpu(_) => self.map(f),
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => crate::cuda::unary(self, _op),
        }
    }

    /// Apply binary op using zipped tensors
    pub fn zip_with(&self, rhs: &Self, _op: BinaryOp, f: impl Fn(T, T) -> T + Sync + Send) -> TensorResult<Self> {
        self.same_device(rhs)?;
        let out_shape = broadcast_shapes(self.shape(), rhs.shape())?;
        match (&*self.storage, &*rhs.storage) {
            (Storage::Cpu(a), Storage::Cpu(b)) => {
                let data: Vec<T> = if self.shape() == rhs.shape() && self.is_contiguous() && rhs.is_contiguous() {
                    let (n, oa, ob) = (self.numel(), self.offset(), rhs.offset());
                    par::zip_slices(&a[oa..oa + n], &b[ob..ob + n], |&x, &y| f(x, y))
                } else {
                    let la = self.layout.broadcast_to(&out_shape)?;
                    let lb = rhs.layout.broadcast_to(&out_shape)?;
                    la.offsets().zip(lb.offsets()).map(|(i, j)| f(a[i], b[j])).collect()
                };
                Tensor::from_vec(data, &out_shape)
            }
            #[cfg(feature = "cuda")]
            (Storage::Cuda(_), Storage::Cuda(_)) => crate::cuda::binary(self, rhs, _op, &out_shape),
            #[cfg(feature = "cuda")]
            _ => Err(TensorError::DeviceMismatch { lhs: self.device(), rhs: rhs.device() }),
        }
    }

    /// Apply binary Scalar Op
    pub fn scalar_op(&self, v: T, op: BinaryOp, f: impl Fn(T, T) -> T + Sync + Send) -> TensorResult<Self> {
        if self.device().is_cpu() {
            self.map(move |x| f(x, v))
        } else {
            self.zip_with(&self.scalar_like(v)?, op, f)
        }
    }

    /// No of elems
    pub fn count(&self) -> usize{ self.numel()}

    /// Element wise Negation 
    pub fn neg(&self) -> TensorResult<Self> { 
        self.unary_with(UnaryOp::Neg, |x| -x) 
    }

    /// Element wise Absolute
    pub fn abs(&self) -> TensorResult<Self> { self.unary_with(
        UnaryOp::Abs, |x| Element::abs(x)) 
    }

    /// Element wise Square 
    pub fn square(&self) -> TensorResult<Self> { 
        self.unary_with(UnaryOp::Square, |x| x * x) 
    }

    /// Element wise clamping 
    pub fn clamp(&self, lo: T, hi: T) -> TensorResult<Self> {
        self.via_host(|t| t.map(|x| if x < lo { lo } else if x > hi { hi } else { x }))
    }

    /// Element wise Addition 
    pub fn add(&self, rhs: &Self) -> TensorResult<Self> { 
        self.zip_with(rhs, BinaryOp::Add, |a, b| a + b) 
    }

    /// Element wise Subtraction 
    pub fn sub(&self, rhs: &Self) -> TensorResult<Self> { 
        self.zip_with(rhs, BinaryOp::Sub, |a, b| a - b) 
    }

    /// Element wise multiplication 
    pub fn mul(&self, rhs: &Self) -> TensorResult<Self> { 
        self.zip_with(rhs, BinaryOp::Mul, |a, b| a * b) 
    }

    /// Element wise division
    pub fn div(&self, rhs: &Self) -> TensorResult<Self> { 
        self.zip_with(rhs, BinaryOp::Div, |a, b| a / b) 
    }

    /// Element wise max
    pub fn maximum(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip_with(rhs, BinaryOp::Max, |a, b| if a > b { a } else { b })
    }

    /// Element wise min
    pub fn minimum(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip_with(rhs, BinaryOp::Min, |a, b| if a < b { a } else { b })
    }

    /// Element wise relu 
    pub fn relu(&self) -> TensorResult<Self> {
        self.unary_with(UnaryOp::Relu, |x| if x > T::zero() { x } else { T::zero() })
    }

    /// Thresholding for floating point eq comparisons
    pub fn thresh_eq_t(&self,rhs: &Self,thresh: Option<T>) -> TensorResult<Self>{
        self.zip_with(rhs, BinaryOp::Eq, |a, b| if (a-b).abs() <= thresh.unwrap_or(T::from_f64(0.01)) { T::one() } else { T::zero() })
    }

    /// Thresholding for floating point neq comparisons
    pub fn thresh_neq_t(&self,rhs: &Self,thresh: Option<T>) -> TensorResult<Self>{
        self.zip_with(rhs, BinaryOp::Eq, |a, b| if (a-b).abs() > thresh.unwrap_or(T::from_f64(0.01)) { T::one() } else { T::zero() })
    }

    /// Equals to 
    pub fn eq_t(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip_with(rhs, BinaryOp::Eq, |a, b| if a == b { T::one() } else { T::zero() })
    }

    /// Not Equals to 
    pub fn neq_t(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip_with(rhs, BinaryOp::Eq, |a, b| if a != b { T::one() } else { T::zero() })
    }

    /// Greater than 
    pub fn gt(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip_with(rhs, BinaryOp::Gt, |a, b| if a > b { T::one() } else { T::zero() })
    }
    
    /// Less than 
    pub fn lt(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip_with(rhs, BinaryOp::Lt, |a, b| if a < b { T::one() } else { T::zero() })
    }

    /// Greater than equal to  
    pub fn ge(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip_with(rhs, BinaryOp::Ge, |a, b| if a >= b { T::one() } else { T::zero() })
    }

    /// Less than equal to 
    pub fn le(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip_with(rhs, BinaryOp::Le, |a, b| if a <= b { T::one() } else { T::zero() })
    }

    /// Scaler Addition 
    pub fn add_scalar(&self, v: T) -> TensorResult<Self> {
        self.scalar_op(v, BinaryOp::Add, |a, b| a + b) 
    }

    /// Scaler Subtraction 
    pub fn sub_scalar(&self, v: T) -> TensorResult<Self> {
        self.scalar_op(v, BinaryOp::Sub, |a, b| a - b) 
    }

    /// Scaler Multiplication 
    pub fn mul_scalar(&self, v: T) -> TensorResult<Self> {
        self.scalar_op(v, BinaryOp::Mul, |a, b| a * b) 
    }

    /// Scaler division
    pub fn div_scalar(&self, v: T) -> TensorResult<Self> {
        self.scalar_op(v, BinaryOp::Div, |a, b| a / b) 
    }
    
    /// Map a and b to same shape and based on elem of self if not like zero use elem from z,else elem from y
    pub fn where_cond(&self, a: &Self, b: &Self) -> TensorResult<Self> {
        self.same_device(a)?;
        self.same_device(b)?;
        let shape = broadcast_shapes(&broadcast_shapes(self.shape(), a.shape())?, b.shape())?;
        let m = self.broadcast_to(&shape)?.to_vec()?;
        let x = a.broadcast_to(&shape)?.to_vec()?;
        let y = b.broadcast_to(&shape)?.to_vec()?;
        let out: Vec<T> = (0..m.len()).map(|i| if m[i] != T::zero() { x[i] } else { y[i] }).collect();
        Tensor::from_vec(out, &shape)?.to_device(self.device())
    }

    fn _tri(&self, diagonal: isize, lower: bool) -> TensorResult<Self> {
        let l = self.ndim();
        if l < 2 { return Err(TensorError::InvalidShape("Tri need atleast 2 dims".into()));}
        let (r, c) = (self.shape()[l - 2], self.shape()[l - 1]);
        let mask = Tensor::from_fn(&[r, c], |ix| {
            let (i, j) = (ix[0] as isize, ix[1] as isize);
            if (lower && j - i <= diagonal) || (!lower && j - i >= diagonal) { T::one() } else { T::zero() }
        }).to_device(self.device())?;
        mask.where_cond(self, &self.zeros_like())
    }

    /// Lower triangle below shifted diagonal [0 to get normal]
    pub fn tril(&self, diagonal: isize) -> TensorResult<Self> { self._tri(diagonal, true) }

    /// Lower trinagle below diagonal
    pub fn lower_triangle(&self) -> TensorResult<Self> { self._tri(0, true) }

    /// Upper triangle above shifted diagonal [0 to get normal]
    pub fn triu(&self, diagonal: isize) -> TensorResult<Self> { self._tri(diagonal, false) }

    /// Upper trinagle above diagonal
    pub fn upper_triangle(&self) -> TensorResult<Self> { self._tri(0, false) }

    fn _tri_alt(&self, diagonal: isize, lower: bool) -> TensorResult<Self> {
        let l = self.ndim();
        if l < 2 { return Err(TensorError::InvalidShape("Tri need atleast 2 dims".into()));}
        let (r, c) = (self.shape()[l - 2], self.shape()[l - 1]);
        let mask = Tensor::from_fn(&[r, c], |ix| {
            let (i, j) = (ix[0] as isize, ix[1] as isize);
            let j_mask = (c as isize - 1) + diagonal - i;
            if (lower && j >= j_mask) || (!lower && j <= j_mask) {T::one()} else {T::zero()}
        }).to_device(self.device())?;
        mask.where_cond(self, &self.zeros_like())
    }

    /// Lower triangle below shifted alternate diagonal [0 to get normal]
    pub fn alt_tril(&self, diagonal: isize) -> TensorResult<Self> { self._tri_alt(diagonal, true) }

    /// Lower triangle below alternate diagonal
    pub fn alt_lower_triangle(&self) -> TensorResult<Self> { self._tri_alt(0, true) }

    /// Upper triangle above shifted alternate diagonal [0 to get normal]
    pub fn alt_triu(&self, diagonal: isize) -> TensorResult<Self> { self._tri_alt(diagonal, false) }

    /// Upper triangle above alternate diagonal
    pub fn alt_upper_triangle(&self) -> TensorResult<Self> { self._tri_alt(0, false) }

}

impl<T: Float> Tensor<T> {

    /// Element wise exponential (e ^ elem)
    pub fn exp(&self) -> TensorResult<Self> { 
        self.unary_with(UnaryOp::Exp, |x| x.exp()) 
    }

    /// Element wise natural log (ln(elem))
    pub fn ln(&self) -> TensorResult<Self> { 
        self.unary_with(UnaryOp::Ln, |x| x.ln()) 
    }

    /// Element wise square root (sqrt(elem))
    pub fn sqrt(&self) -> TensorResult<Self> { 
        self.unary_with(UnaryOp::Sqrt, |x| x.sqrt()) 
    }

    /// Element wise sin (sin(elem))
    pub fn sin(&self) -> TensorResult<Self> {
        self.unary_with(UnaryOp::Sin, |x| x.sin()) 
    }

    /// Element wise cos (cos(elem))
    pub fn cos(&self) -> TensorResult<Self> {
        self.unary_with(UnaryOp::Cos, |x| x.cos()) 
    }

    /// Element wise tanh (tanh(elem))
    pub fn tanh(&self) -> TensorResult<Self> {
        self.unary_with(UnaryOp::Tanh, |x| x.tanh()) 
    }

    /// Element wise floor
    pub fn floor(&self) -> TensorResult<Self> {
        self.unary_with(UnaryOp::Floor, |x| x.floor()) 
    }

    /// Element wise ceil
    pub fn ceil(&self) -> TensorResult<Self> {
        self.unary_with(UnaryOp::Ceil, |x| x.ceil()) 
    }
    
    /// Element wise reciprocal
    pub fn recip(&self) -> TensorResult<Self> {
        self.unary_with(UnaryOp::Recip, |x| T::one() / x) 
    }

    /// Element wise sigmoid 
    pub fn sigmoid(&self) -> TensorResult<Self> {
        self.unary_with(UnaryOp::Sigmoid, |x| T::one() / (T::one() + (-x).exp()))
    }

    /// GELU using fast Tanh approximation
    pub fn gelu(&self) -> TensorResult<Self> {
        self.unary_with(UnaryOp::Gelu, |x| {
            let c = T::from_f64(0.7978845608028654);
            T::from_f64(0.5) * x * (T::one() + (c * (x + (T::from_f64(0.044715) * x * x * x))).tanh())
        })
    }

    /// Silu/Swish     
    pub fn silu(&self) -> TensorResult<Self> {self.mul(&self.sigmoid()?)}
    pub fn swish(&self) -> TensorResult<Self> {self.silu()}

    /// Leaky Relu
    pub fn lrelu(&self, alpha: T) -> TensorResult<Self> {
        self.via_host(|t| t.map(|x| if x > T::zero() { x } else { alpha * x }))
    }

    /// Element wise power using two arrays A(a1,a2,..an) and B(b1,b2,..bn)  -> C(a1^b1,a2^b2,..an^bn)
    pub fn pow(&self, rhs: &Self) -> TensorResult<Self> { self.zip_with(rhs, BinaryOp::Pow, |a, b| a.powf(b)) }

    /// Element wise power using floating point exponent
    pub fn powf(&self, e: T) -> TensorResult<Self> { self.scalar_op(e, BinaryOp::Pow, |a, b| a.powf(b)) }

    /// Element wise power using integer exponent
    pub fn powi32(&self, n: i32) -> TensorResult<Self> { self.powf(T::from_f64(n as f64)) }
    pub fn powi64(&self, n: i64) -> TensorResult<Self> { self.powf(T::from_f64(n as f64)) }
}

macro_rules! impl_bin_ops {
    ($tr:ident, $m:ident) => {
        impl<T: Element> $tr<&Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;
            fn $m(self, rhs: &Tensor<T>) -> Tensor<T> { Tensor::$m(self, rhs).expect(concat!("Tensor ", stringify!($m), " failed")) }
        }
        impl<T: Element> $tr<Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;
            fn $m(self, rhs: Tensor<T>) -> Tensor<T> { Tensor::$m(&self, &rhs).expect(concat!("Tensor ", stringify!($m), " failed")) }
        }
        impl<T: Element> $tr<&Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;
            fn $m(self, rhs: &Tensor<T>) -> Tensor<T> { Tensor::$m(&self, rhs).expect(concat!("Tensor ", stringify!($m), " failed")) }
        }
        impl<T: Element> $tr<Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;
            fn $m(self, rhs: Tensor<T>) -> Tensor<T> { Tensor::$m(self, &rhs).expect(concat!("Tensor ", stringify!($m), " failed")) }
        }
    };
}
impl_bin_ops!(Add, add);
impl_bin_ops!(Sub, sub);
impl_bin_ops!(Mul, mul);
impl_bin_ops!(Div, div);

macro_rules! impl_scalar_ops {
    ($t:ty) => {
        impl Add<$t> for &Tensor<$t> { type Output = Tensor<$t>; fn add(self, v: $t) -> Tensor<$t> { self.add_scalar(v).unwrap() } }
        impl Sub<$t> for &Tensor<$t> { type Output = Tensor<$t>; fn sub(self, v: $t) -> Tensor<$t> { self.sub_scalar(v).unwrap() } }
        impl Mul<$t> for &Tensor<$t> { type Output = Tensor<$t>; fn mul(self, v: $t) -> Tensor<$t> { self.mul_scalar(v).unwrap() } }
        impl Div<$t> for &Tensor<$t> { type Output = Tensor<$t>; fn div(self, v: $t) -> Tensor<$t> { self.div_scalar(v).unwrap() } }
        impl Add<$t> for Tensor<$t> { type Output = Tensor<$t>; fn add(self, v: $t) -> Tensor<$t> { self.add_scalar(v).unwrap() } }
        impl Sub<$t> for Tensor<$t> { type Output = Tensor<$t>; fn sub(self, v: $t) -> Tensor<$t> { self.sub_scalar(v).unwrap() } }
        impl Mul<$t> for Tensor<$t> { type Output = Tensor<$t>; fn mul(self, v: $t) -> Tensor<$t> { self.mul_scalar(v).unwrap() } }
        impl Div<$t> for Tensor<$t> { type Output = Tensor<$t>; fn div(self, v: $t) -> Tensor<$t> { self.div_scalar(v).unwrap() } }
    };
}
impl_scalar_ops!(f32);
impl_scalar_ops!(f64);
impl_scalar_ops!(i32);
impl_scalar_ops!(i64);

impl<T: Element> Neg for &Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Tensor<T> { Tensor::neg(self).expect("Tensor neg failed") }
}

impl<T: Element> Neg for Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Tensor<T> { Tensor::neg(&self).expect("Tensor neg failed") }
}
