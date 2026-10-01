pub mod random;
pub mod device;

pub mod Utilities{
    // Some trait definations for the numpy based impl of scalars,vectors and matrices
    
    use std::ops::{Add, Div, Mul, Sub};

use crate::device::Device::Device;

    pub trait Shape{
        fn shape(&self) -> &[usize]; // Shape

        // No of dims
        fn ndim(&self) -> usize{
            self.shape().len()
        }

        // No of elems
        fn numel(&self) -> usize{
            self.shape().iter().product()
        }  
    }

    pub trait Storage {
        type Elem;
        fn len(&self) -> usize;
        fn device(&self) -> &Device;
        fn as_ptr(&self) -> *const Self::Elem;
        fn as_mut_ptr(&mut self) -> *mut Self::Elem;
    }

    pub trait Layout {
        fn shape(&self) -> &[usize];
        // How much to move from onen dim to another,eg:  (2,3) so return = [3,1] i.e to mmove to next value in resp deim we need to move foerard by this many elem chunks
        fn strides(&self) -> &[isize];
        fn offset(&self) -> isize; // Offset in memory/storage
        fn is_contiguous(&self) -> bool; // Are the elems contigious in memory/storage
    }

    pub trait Tensor<T>: Shape + Layout + Storage + Sized + Add + Sub + Mul + Div + MinMathsContract<Self,T> + VectorOpsContract<Self,T>{
        fn unpack(self) -> dyn Iterator<Item = T>;
        fn reshape(&self,shape: &[usize]) -> Self; // Get a new reshaped Tensor
        fn view(&self,shape: &[usize],strides: &[isize]) -> Self; // Get a slice
        fn On(&mut self,dev: &Device ); // Move Tensor to a Device
    }
    

    // EG: Tensor<_Tensor,f32>
    pub trait MinMathsContract<M: Tensor<T>,T>{
        fn Min(&self) -> T;
        fn Max(&self) -> T;
        fn Count(&self) -> usize;
        fn Mag(&self) -> T;
        fn Percentile(&self,ptile: f64) -> f64;

        fn Mean(&self) -> f64;
        fn Median(&self) -> f64;
        fn Mode(&self) -> f64;

        fn SD(&self) -> f64;
        fn Var(&self) -> f64;

        fn Cov(&self,other: & impl MinMathsContract<M,T>) -> f64;
        fn Pearson_coeff(&self,other: &M) -> f64;
    }

    pub trait VectorOpsContract<M: Tensor<T>,T>{
        fn Dot(&self,other: &M) -> Self;
        fn Cross(&self,other: &M) -> Self;
        fn Norm(&self,other: &M) -> Self;
        fn Mag(&self,other: &M) -> Self;
        fn Empty(&self) -> Self;
        fn Eye(&self) -> Self;
        fn Arange(&self,range: (T,T,T)) -> Self;
    }








}
