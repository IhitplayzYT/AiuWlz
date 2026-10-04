use Utilities::device::Device::Device;
use crate::dtype::{Element, Float};
use crate::error::{Result, TensorError};
use crate::layout::{normalize_dim, normalize_dim_inclusive, Layout};
use crate::par;
use Utilities::random::Random::Rng;
use crate::storage::Storage;
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor<T: Element> {
    pub storage: Arc<Storage<T>>,
    pub layout: Layout,
}

impl<T: Element> Tensor<T> {
    pub fn from_comp(storage: Storage<T>, layout: Layout) -> Self {
        Tensor { storage: Arc::new(storage), layout }
    }

    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(TensorError::InvalidShape(format!("Insufficient elems: {} can't fill shape {:?}",data.len(), shape)));
        }
        Ok(Self::from_comp(Storage::Cpu(data), Layout::contiguous(shape)))
    }

    pub fn scalar(v: T) -> Self {
        Self::from_comp(Storage::Cpu(vec![v]), Layout::contiguous(&[]))
    }

    pub fn fill(shape: &[usize], v: T) -> Self {
        let n = shape.iter().product();
        Self::from_comp(Storage::Cpu(vec![v; n]), Layout::contiguous(shape))
    }
    pub fn zeros(shape: &[usize]) -> Self { Self::fill(shape, T::zero())}
    pub fn ones(shape: &[usize]) -> Self { Self::fill(shape, T::one()) }

    pub fn zeros_like(&self) -> Self { Self::zeros(self.shape()).on_device_unchecked(self.device()) }
    pub fn ones_like(&self) -> Self { Self::ones(self.shape()).on_device_unchecked(self.device()) }
    pub fn fill_like(&self, v: T) -> Self { Self::fill(self.shape(), v).on_device_unchecked(self.device()) }
    pub fn on_device_unchecked(self, dev: Device) -> Self { self.to_device(dev).expect(&format!("Failed to move tensor to {}",dev))}

    pub fn arange(start: T, end: T, step: T) -> Result<Self> {
        if step == T::zero() { return Err(TensorError::InvalidShape("Step canot be 0".into()));}
        let mut v = Vec::new();
        let mut x = start;
        // Support for negative step suggested by AI
        while (step > T::zero() && x < end) || (step < T::zero() && x > end) {
            v.push(x);
            x += step;
        }
        let l = v.len();
        Self::from_vec(v, &[l])
    }

    pub fn eye(n: usize) -> Self {
        let mut v = vec![T::zero(); n * n];
        for i in 0..n { v[i * n + i] = T::one(); }
        Self::from_vec(v, &[n, n]).unwrap()
    }

    /// Build a tensor by calling `f` on every multi-index made using AI
    pub fn from_fn(shape: &[usize], mut f: impl FnMut(&[usize]) -> T) -> Self {
        let n: usize = shape.iter().product();
        let mut idx = vec![0usize; shape.len()];
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(f(&idx));
            for d in (0..shape.len()).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] { break; }
                idx[d] = 0;
            }
        }
        Self::from_vec(v, shape).unwrap()
    }
}

impl<T: Float> Tensor<T> {
    pub fn linspace(start: T, end: T, steps: usize) -> Self {
        let (st, ed) = (start.to_f64(), end.to_f64());
        let v: Vec<T> = (0..steps).map(|i| T::from_f64(st + (ed - st) *  if steps > 1 { i as f64 / (steps - 1) as f64 } else { 0.0 })).collect();
        Self::from_vec(v, &[steps]).unwrap()
    }
    
    pub fn rand(shape: &[usize], rng: &mut Rng) -> Self {
        Self::from_vec((0..shape.iter().product()).map(|_| T::from_f64(rng.uniform())).collect(), shape).unwrap()
    }

    pub fn randn(shape: &[usize], rng: &mut Rng) -> Self {
        Self::from_vec((0..shape.iter().product()).map(|_| T::from_f64(rng.normal())).collect(), shape).unwrap()
    }
}

impl<T: Element> Tensor<T> {
    pub fn device(&self) -> Device { self.storage.device() }
    pub fn device_str(&self) -> Option<&'static str> { T::CUDA_TYPE }
    pub fn dtype(&self) -> &'static str { T::NAME }
    pub fn shape(&self) -> &[usize] { self.layout.shape() }
    pub fn strides(&self) -> &[isize] { self.layout.strides() }
    pub fn offset(&self) -> usize { self.layout.offset() }
    pub fn ndim(&self) -> usize { self.layout.ndim() }
    pub fn numel(&self) -> usize { self.layout.numel() }
    pub fn is_contiguous(&self) -> bool { self.layout.is_contiguous() }
    pub fn layout(&self) -> &Layout { &self.layout }
    pub fn storage(&self) -> &Storage<T> { &self.storage }
    pub fn storage_ptr(&self) -> *const T { self.storage.as_ptr() }
    pub fn storage_as_cpu(&self) -> Option<&[T]> { self.storage.as_cpu() }

    // Get data on a host array
    pub fn host_data(&self) -> Result<Cow<'_, [T]>> {
        match &*self.storage {
            Storage::Cpu(v) => {
                if self.layout.is_contiguous() {
                    let o = self.layout.offset();
                    Ok(Cow::Borrowed(&v[o..o + self.numel()]))
                } else {
                    Ok(Cow::Owned(self.layout.offsets().map(|o| v[o]).collect()))
                }
            }
            #[cfg(feature = "cuda")]
            Storage::Cuda(c) => {
                let v = c.to_host()?;
                Ok(Cow::Owned(self.layout.offsets().map(|o| v[o]).collect()))
            }
        }
    }

    pub fn to_vec(&self) -> Result<Vec<T>> { Ok(self.host_data()?.to_vec()) }

    pub fn item(&self) -> Result<T> {
        if self.numel() != 1 { return Err(TensorError::InvalidShape(format!("Item needs 1 elem, tensor has {:?}", self.shape())));}
        Ok(self.host_data()?[0])
    }

    pub fn get(&self, index: &[usize]) -> Result<T> {
        if index.len() != self.ndim() || index.iter().zip(self.shape()).any(|(i, s)| i >= s) {
            return Err(TensorError::InvalidShape(format!("Index {:?} out of bounds for {:?}", index, self.shape())));
        }
        let mut t = self.clone();
        for (d, &i) in index.iter().enumerate().rev() {
            t = Tensor { storage: t.storage.clone(), layout: t.layout.select(d, i)? }; 
        }
        t.item()
    }

    pub fn to_device(&self, dev: Device) -> Result<Self> {
        if self.device() == dev { return Ok(self.clone()); }
        let host = self.host_data()?.into_owned();
        match dev {
            Device::Cpu => Self::from_vec(host, self.shape()),
            Device::Cuda(_ordinal) => {
                #[cfg(feature = "cuda")]
                {
                    let c = crate::cuda::CudaStorage::from_host(&host, _ordinal)?;
                    Ok(Self::from_comp(Storage::Cuda(c), Layout::contiguous(self.shape())))
                }
                #[cfg(not(feature = "cuda"))]
                { Err(TensorError::Unsupported("Please enable cuda feature to move Tensor to Cuda device".into())) }
            }
        }
    }
    pub fn cpu(&self) -> Result<Self> { self.to_device(Device::Cpu) }
    pub fn cuda(&self, ordinal: usize) -> Result<Self> { self.to_device(Device::Cuda(ordinal)) }
    pub fn scalar_like(&self, v: T) -> Result<Self> {Tensor::scalar(v).to_device(self.device())} 

    pub fn same_device(&self, other: &Self) -> Result<()> {
        if self.device() != other.device() {
            Err(TensorError::DeviceMismatch { lhs: self.device(), rhs: other.device() })
        } else { Ok(()) }
    }

    /// Do a op on Tensor in CPU
    pub fn via_host<R: Element>(&self, f: impl FnOnce(&Tensor<T>) -> Result<Tensor<R>>) -> Result<Tensor<R>> {
        if self.device().is_cpu() { return f(self); }
        let out = f(&self.cpu()?)?;
        if R::CUDA_TYPE.is_some() { out.to_device(self.device()) } else { Ok(out) }
    }


    pub fn compact(&self) -> Result<Self> {
        if self.layout.is_contiguous() && self.layout.offset() == 0 && self.storage.len() == self.numel() {
            return Ok(self.clone());
        }

        match &*self.storage {
            Storage::Cpu(_) => Self::from_vec(self.host_data()?.into_owned(), self.shape()),
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => crate::cuda::compact(self),
        }
    }

    pub fn to_contiguous(&self) -> Result<Self> {
        if self.layout.is_contiguous() { Ok(self.clone()) } else { self.compact() }
    }

    pub fn view_as(&self, layout: Layout) -> Self { Tensor { storage: self.storage.clone(), layout } }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        if shape.iter().product::<usize>() != self.numel() { return Err(TensorError::InvalidShape(format!("Can't reshape {:?} into {:?}", self.shape(), shape)));}
        let base = self.to_contiguous()?;
        Ok(base.view_as(base.layout.reshape(shape)?))
    }

    /// Reshape supporting a -1 entry made by AI
    pub fn reshape_infer(&self, shape: &[isize]) -> Result<Self> {
        let known: usize = shape.iter().filter_map(|s| if *s >= 0 {Some(*s as usize)} else {None}).product();
        let n_neg = shape.iter().filter(|&&s| s < 0).count();
        if n_neg > 1 || (n_neg == 1 && (known == 0 || self.numel() % known != 0 /* This is since unknown dim is a whole num multiple cant be fractional */)) {
            return Err(TensorError::InvalidShape(format!("Cannot infer shape {:?}", shape)));
        }
        let dims: Vec<usize> = shape.iter().map(|&s| if s < 0 { self.numel() / known } else { s as usize }).collect();
        self.reshape(&dims)
    }

    pub fn flatten(&self) -> Result<Self> { self.reshape(&[self.numel()]) }

    pub fn transpose(&self, d0: isize, d1: isize) -> Result<Self> {
        let (a, b) = (normalize_dim(d0, self.ndim())?, normalize_dim(d1, self.ndim())?);
        Ok(self.view_as(self.layout.transpose(a, b)?))
    }
    
    pub fn t(&self) -> Result<Self> { self.transpose(-2, -1) }
    pub fn permute(&self, dims: &[usize]) -> Result<Self> { Ok(self.view_as(self.layout.permute(dims)?)) }
    pub fn unsqueeze(&self, dim: isize) -> Result<Self> { Ok(self.view_as(self.layout.unsqueeze(normalize_dim_inclusive(dim, self.ndim())?)))}
    pub fn squeeze(&self, dim: isize) -> Result<Self> { Ok(self.view_as(self.layout.squeeze(normalize_dim(dim, self.ndim())?)?))}

    pub fn squeeze_all(&self) -> Result<Self> {
        let shape: Vec<usize> = self.shape().iter().copied().filter(|&s| s != 1).collect();
        self.reshape(&shape)
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> { Ok(self.view_as(self.layout.broadcast_to(shape)?)) }    
    pub fn expand(&self, shape: &[usize]) -> Result<Self> { self.broadcast_to(shape) }
    pub fn narrow(&self, dim: isize, start: usize, len: usize) -> Result<Self> { Ok(self.view_as(self.layout.narrow(normalize_dim(dim, self.ndim())?, start, len)?))}
    pub fn select(&self, dim: isize, index: usize) -> Result<Self> { Ok(self.view_as(self.layout.select(normalize_dim(dim, self.ndim())?, index)?))}
    pub fn flip(&self, dim: isize) -> Result<Self> { Ok(self.view_as(self.layout.flip(normalize_dim(dim, self.ndim())?)?))}

    /// Move `dim` to the last position (used by reductions) suggested and made by AI
    pub fn movedim_last(&self, dim: usize) -> Result<Self> {
        let mut perm: Vec<usize> = (0..self.ndim()).filter(|&d| d != dim).collect();
        perm.push(dim);
        self.permute(&perm)
    }

    pub fn map<R: Element>(&self, f: impl Fn(T) -> R + Sync + Send) -> Result<Tensor<R>> {
        let data = self.host_data()?;
        let out = par::map_slice(&data, |&x| f(x));
        let t = Tensor::from_vec(out, self.shape())?;
        if R::CUDA_TYPE.is_some() { t.to_device(self.device()) } else { Ok(t) }
    }

    pub fn cast<U: Element>(&self) -> Result<Tensor<U>> { self.map(|x| U::from_f64(x.to_f64())) }

    // Check if two tensors are equal using absolute and relative tolerance formula
    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> bool {
        if self.shape() != other.shape() { return false; }
        match (self.host_data(), other.host_data()) {
            (Ok(a), Ok(b)) => a.iter().zip(b.iter()).all(|(x, y)| {
                let (x, y) = (x.to_f64(), y.to_f64());
                (x - y).abs() <= atol + rtol * y.abs()
            }),
            _ => false,
        }
    }

    //  Used AI to generate this
    pub fn cat(tensors: &[&Tensor<T>], dim: isize) -> Result<Self> {
        let first = *tensors.first().ok_or_else(|| TensorError::InvalidShape("No tensors provided to Concat".into()))?;
        let d = normalize_dim(dim, first.ndim())?;
        let mut out_shape = first.shape().to_vec();
        out_shape[d] = 0;
        for t in tensors {
            first.same_device(t)?;
            let ok = t.ndim() == first.ndim() && (0..t.ndim()).all(|i| i == d || t.shape()[i] == first.shape()[i]);
            if !ok { return Err(TensorError::ShapeMismatch { op: "Concat", lhs: first.shape().to_vec(), rhs: t.shape().to_vec() });}
            out_shape[d] += t.shape()[d];
        }
        let outer: usize = first.shape()[..d].iter().product();
        let inner: usize = first.shape()[d + 1..].iter().product();
        let datas: Vec<Cow<[T]>> = tensors.iter().map(|t| t.host_data()).collect::<Result<_>>()?;
        let mut out = Vec::with_capacity(out_shape.iter().product());
        for o in 0..outer {
            for (t, data) in tensors.iter().zip(&datas) {
                let chunk = t.shape()[d] * inner;
                out.extend_from_slice(&data[o * chunk..(o + 1) * chunk]);
            }
        }
        Tensor::from_vec(out, &out_shape)?.to_device(first.device())
    }

    pub fn stack(tensors: &[&Tensor<T>], dim: isize) -> Result<Self> {
        let first = *tensors.first().ok_or_else(|| TensorError::InvalidShape("No Tensors provided to stack".into()))?;
        let d = normalize_dim_inclusive(dim, first.ndim())? as isize;
        let expanded: Vec<Tensor<T>> = tensors.iter().map(|t| t.unsqueeze(d)).collect::<Result<_>>()?;
        let refs: Vec<&Tensor<T>> = expanded.iter().collect();
        Self::cat(&refs, d)
    }

    /// slice of indexes along dims
    pub fn index_select(&self, dim: isize, indices: &[usize]) -> Result<Self> {
        let parts: Vec<Tensor<T>> = indices.iter().map(|&i| self.narrow(dim, i, 1)).collect::<Result<_>>()?;
        let refs: Vec<&Tensor<T>> = parts.iter().collect();
        Self::cat(&refs, dim)
    }

    /// Split into size chunks along dim
    pub fn split(&self, dim: isize, size: usize) -> Result<Vec<Self>> {
        let d = normalize_dim(dim, self.ndim())?;
        let n = self.shape()[d];
        (0..n).step_by(size.max(1)).map(|s| self.narrow(d as isize, s, size.min(n - s))).collect()
    }
}
