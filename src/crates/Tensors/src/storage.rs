use Utilities::device::Device::Device;
use crate::dtype::Element;

#[derive(Debug)]
pub enum Storage<T: Element> {
    Cpu(Vec<T>),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::CudaStorage<T>),
}

impl<T: Element> Storage<T> {
    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(v) => v.len(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(c) => c.len(),
        }
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "cuda")]
            Storage::Cuda(c) => Device::Cuda(c.ordinal),
        }
    }

    pub fn as_cpu(&self) -> Option<&[T]> {
        match self {
            Storage::Cpu(v) => Some(v),
            #[cfg(feature = "cuda")]
            _ => None,
        }
    }

    pub fn as_ptr(&self) -> *const T {
        match self {
            Storage::Cpu(v) => v.as_ptr(),
            #[cfg(feature = "cuda")]
            _ => std::ptr::null(),
        }
    }





}

