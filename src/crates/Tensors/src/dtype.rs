use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Extra bounds an element needs to live in CUDA memory. Empty without the `cuda` feature.
#[cfg(feature = "cuda")]
pub trait DeviceElem: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Unpin {}
#[cfg(feature = "cuda")]
impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Unpin> DeviceElem for T {}

#[cfg(not(feature = "cuda"))]
pub trait DeviceElem {}
#[cfg(not(feature = "cuda"))]
impl<T> DeviceElem for T {}
// Done to support uint
pub trait _Element: Copy + Send + Sync + 'static + PartialOrd + Debug + Display + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + DeviceElem + AddAssign<Self> + SubAssign<Self> + MulAssign<Self> + DivAssign<Self> {
    const NAME: &'static str;
    const CUDA_TYPE: Option<&'static str>;
    fn zero() -> Self;
    fn one() -> Self;
    fn lowest() -> Self;
    fn highest() -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

pub trait Element: _Element + Neg<Output = Self>{
    fn abs(self) -> Self {
        if self < Self::zero() { -self } else { self }
    }
}


pub trait Float: Element {
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    fn powf(self, e: Self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn is_nan(self) -> bool;
}

macro_rules! impl_float {
    ($t:ty, $name:expr, $cuda:expr) => {
        impl _Element for $t{
            const NAME: &'static str = $name;
            const CUDA_TYPE: Option<&'static str> = Some($cuda);
            fn zero() -> Self { 0.0 }
            fn one() -> Self { 1.0 }
            fn lowest() -> Self { <$t>::NEG_INFINITY }
            fn highest() -> Self { <$t>::INFINITY }
            fn from_f64(v: f64) -> Self { v as $t }
            fn to_f64(self) -> f64 { self as f64 }
        }

        impl Element for $t {}

        impl Float for $t {
            fn exp(self) -> Self { <$t>::exp(self) }
            fn ln(self) -> Self { <$t>::ln(self) }
            fn sqrt(self) -> Self { <$t>::sqrt(self) }
            fn sin(self) -> Self { <$t>::sin(self) }
            fn cos(self) -> Self { <$t>::cos(self) }
            fn tanh(self) -> Self { <$t>::tanh(self) }
            fn powf(self, e: Self) -> Self { <$t>::powf(self, e) }
            fn floor(self) -> Self { <$t>::floor(self) }
            fn ceil(self) -> Self { <$t>::ceil(self) }
            fn is_nan(self) -> bool { <$t>::is_nan(self) }
        }
    };
}

macro_rules! impl_int {
    ($t:ty, $name:expr) => {
        impl _Element for $t{
            const NAME: &'static str = $name;
            const CUDA_TYPE: Option<&'static str> = None;
            fn zero() -> Self { 0 }
            fn one() -> Self { 1 }
            fn lowest() -> Self { <$t>::MIN }
            fn highest() -> Self { <$t>::MAX }
            fn from_f64(v: f64) -> Self { v as $t }
            fn to_f64(self) -> f64 { self as f64 }
        }

        impl Element for $t {}
    };
}


macro_rules! impl_uint {
    ($t:ty, $name:expr) => {
        impl _Element for $t {
            const NAME: &'static str = $name;
            const CUDA_TYPE: Option<&'static str> = None;
            fn zero() -> Self { 0 }
            fn one() -> Self { 1 }
            fn lowest() -> Self { <$t>::MIN }
            fn highest() -> Self { <$t>::MAX }
            fn from_f64(v: f64) -> Self { v as $t }
            fn to_f64(self) -> f64 { self as f64 }
        }
    };
}

// Generative macro calls to finihs the impl of the traits for rust types 

impl_float!(f32, "f32", "float"); 
impl_float!(f64, "f64", "double");

impl_uint!(u8,"u8");
impl_uint!(u16,"u16");
impl_uint!(u32,"u32");
impl_uint!(u64,"u64");
impl_uint!(usize,"usize");

impl_int!(i8, "i8");
impl_int!(i16, "i16");
impl_int!(i32, "i32");
impl_int!(i64, "i64");
impl_int!(isize, "isize");
