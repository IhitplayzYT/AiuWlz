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
    /// Build from own handled and created memeory and layout
    /// Warning! User needs to guarentee valid memory and layout is provided
    pub fn from_comp(storage: Storage<T>, layout: Layout) -> Self {
        Tensor { storage: Arc::new(storage), layout }
    }

    /// Build a Tensor from 1d Vec
    pub fn from_1d(data: Vec<T>) -> Result<Self>{
        let l = data.len();
        Ok(Self::from_comp(Storage::Cpu(data), Layout::contiguous(&[l])))
    }

    /// Build a Tensor from 2d Vec
    pub fn from_2d(data: Vec<Vec<T>>) -> Result<Self>{
        let m = data.len();
        if m == 0 {
            return Err(TensorError::InvalidShape("Cannot build Tensor from empty array".to_string()))
        }
        let n = data[0].len();
        let data = data.into_iter().flatten().collect::<Vec<T>>();
        Ok(Self::from_comp(Storage::Cpu(data), Layout::contiguous(&[m,n])))
    }

    /// Build a Tensor from 3d Vec
    pub fn from_3d(data: Vec<Vec<Vec<T>>>) -> Result<Self>{
        let x = data.len();
        if x == 0 {
            return Err(TensorError::InvalidShape("Cannot build Tensor from empty array".to_string()))
        }
        let y = data[0].len();
        if y == 0 {
            return Err(TensorError::InvalidShape("Cannot build Tensor from empty array".to_string()))
        }
        let z = data[0][0].len();
        let data = data.into_iter().flatten().collect::<Vec<Vec<T>>>().into_iter().flatten().collect::<Vec<T>>();
        Ok(Self::from_comp(Storage::Cpu(data), Layout::contiguous(&[x,y,z])))
    }


    /// Build a Tensor from vec and shape
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(TensorError::InvalidShape(format!("Insufficient elems: {} can't fill shape {:?}",data.len(), shape)));
        }
        Ok(Self::from_comp(Storage::Cpu(data), Layout::contiguous(shape)))
    }

    /// Create a scalar Tensor on Cpu
    pub fn scalar(v: T) -> Self {
        Self::from_comp(Storage::Cpu(vec![v]), Layout::contiguous(&[]))
    }

    /// Create a new array filled with same elem on Cpu 
    pub fn fill(shape: &[usize], v: T) -> Self {
        let n = shape.iter().product();
        #[cfg(feature = "cuda")]
        if T::CUDA_TYPE.is_some() {
            use crate::cuda::CudaStorage;
 
            return Self::from_comp(
                Storage::Cuda(CudaStorage::from_host(&crate::cuda::fill(shape, v, 0).unwrap().to_vec().unwrap(),0).unwrap()),
                Layout::contiguous(shape)
            );
        }
        Self::from_comp(Storage::Cpu(vec![v; n]), Layout::contiguous(shape))
    }

    /// Create a new array filled with 0/0.0 on Cpu 
    pub fn zeros(shape: &[usize]) -> Self {Self::fill(shape, T::zero())}

    /// Create a new array filled with 1/1.0 on Cpu 
    pub fn ones(shape: &[usize]) -> Self { Self::fill(shape, T::one()) }

    /// Create a new array filled with 0/0.0 on the current tensor's device
    pub fn zeros_like(&self) -> Self { Self::zeros(self.shape()).on_device_unchecked(self.device()) }

    /// Create a new array filled with 1/1.0 on the current tensor's device
    pub fn ones_like(&self) -> Self { Self::ones(self.shape()).on_device_unchecked(self.device()) }

    /// Create a new array filled with same elem on the current tensor's device
    pub fn fill_like(&self, v: T) -> Self { Self::fill(self.shape(), v).on_device_unchecked(self.device()) }

    /// Program panics on failure, use only when guarenteed valid
    pub fn on_device_unchecked(self, dev: Device) -> Self { self.on(dev).expect(&format!("Failed to move tensor to {}",dev))}

    /// Create a Tensor from a range
    pub fn arange(start: T, end: T, step: T) -> Result<Self> {
        if step == T::zero() { return Err(TensorError::InvalidShape("Step canot be 0".into()));}
        #[cfg(feature = "cuda")]
        if T::CUDA_TYPE.is_some() { return crate::cuda::arange(start, end, step, 0);}
        let mut v = Vec::new();
        let mut x = start;
        while (step > T::zero() && x < end) || (step < T::zero() && x > end) {
            v.push(x);
            x += step;
        }
        let l = v.len();
        Self::from_vec(v, &[l])
    }

    /// Identity matrix
    pub fn eye(n: usize) -> Self {
        let mut v = vec![T::zero(); n * n];
        for i in 0..n { v[i * n + i] = T::one(); }
        Self::from_vec(v, &[n, n]).unwrap()
    }

    /// Build a tensor by calling fn on every multi-index
    // Made using AI
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

    /// Create a n elem tensor on a range
    pub fn linspace(start: T, end: T, steps: usize) -> Self {
        let step = if steps > 1 { (end - start) / T::from_f64((steps - 1) as f64) } else { T::zero() };
        #[cfg(feature = "cuda")]
        if T::CUDA_TYPE.is_some() {
            return crate::cuda::arange(start, end + step, step, 0).unwrap();
        }
        let (st, ed) = (start.to_f64(), end.to_f64());
        let v: Vec<T> = (0..steps).map(|i| T::from_f64(st + (ed - st) *  if steps > 1 { i as f64 / (steps - 1) as f64 } else { 0.0 })).collect();
        Self::from_vec(v, &[steps]).unwrap()
    }
    
    /// Create a tensor sampled from unit Uniform distrib
    pub fn rand(shape: &[usize], rng: &mut Rng) -> Self {
        Self::from_vec((0..shape.iter().product()).map(|_| T::from_f64(rng.uniform())).collect(), shape).unwrap()
    }

    /// Create a tensor sampled from normal distrib
    pub fn randn(shape: &[usize], rng: &mut Rng) -> Self {
        Self::from_vec((0..shape.iter().product()).map(|_| T::from_f64(rng.normal())).collect(), shape).unwrap()
    }    
}

impl<T: Element> Tensor<T> {
    /// Get device
    pub fn device(&self) -> Device { self.storage.device() }
    /// Get device str
    pub fn device_str(&self) -> Option<&'static str> { T::CUDA_TYPE }
    /// Get dtype of elem the Tensor is holding
    pub fn dtype(&self) -> &'static str { T::NAME }
    /// Get shape of Tensor
    pub fn shape(&self) -> &[usize] { self.layout.shape() }
    /// Get strides of iterating over Tensor
    pub fn strides(&self) -> &[isize] { self.layout.strides() }
    /// Get memory offset of where Tensor is storred
    pub fn offset(&self) -> usize { self.layout.offset() }
    /// No of dimension of Tensor
    pub fn ndim(&self) -> usize { self.layout.ndim() }
    /// No of elems of Tensor
    pub fn numel(&self) -> usize { self.layout.numel() }
    /// Is tensor contigious
    pub fn is_contiguous(&self) -> bool { self.layout.is_contiguous() }
    /// Get layout of Tensor
    pub fn layout(&self) -> &Layout { &self.layout }
    /// Get layout of Storage 
    pub fn storage(&self) -> &Storage<T> { &self.storage }
    
    /// Get data on a host
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

    /// Get vector of the data
    pub fn to_vec(&self) -> Result<Vec<T>> { 
        Ok(self.host_data()?.to_vec()) 
    }

    /// Returns the item in a Scalar or unit Tensor
    pub fn item(&self) -> Result<T> {
        if self.numel() != 1 { return Err(TensorError::InvalidShape(format!("Item needs 1 elem, tensor has {:?}", self.shape())));}
        Ok(self.host_data()?[0])
    }

    /// Index on an elem
    pub fn get(&self, index: &[usize]) -> Result<T> {
        if index.len() != self.ndim() || index.iter().zip(self.shape()).any(|(i, s)| i >= s) { return Err(TensorError::InvalidShape(format!("Index {:?} out of bounds for {:?}", index, self.shape())));}
        let mut t = self.clone();
        for (d, &i) in index.iter().enumerate().rev() {
            t = Tensor { storage: t.storage.clone(), layout: t.layout.select(d, i)? }; 
        }
        t.item()
    }

    /// Move Tensor to a device
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

    /// Same as to_device, moves Tensor to specified device
    pub fn on(&self,dev: Device) -> Result<Self>{
        self.to_device(dev)
    }
    
    /// Move Tensor to CPU
    pub fn cpu(&self) -> Result<Self> { 
        self.to_device(Device::Cpu) 
    }
    
    /// Move Tensor to a Cuda device
    pub fn cuda(&self, ordinal: usize) -> Result<Self> { 
        self.to_device(Device::Cuda(ordinal)) 
    }

    /// Create a scalar Tensor on the Tensor's device
    pub fn scalar_like(&self, v: T) -> Result<Self> {
        Tensor::scalar(v).on(self.device())
    } 

    /// Create a scalar Tensor on provided Device
    pub fn scalar_on(v:T,dev: Device) -> Result<Self>{
        Tensor::scalar(v).on(dev)
    }

    /// Check if two tensors are on same device
    pub fn same_device(&self, other: &Self) -> Result<()> {
        if self.device() != other.device() {
            Err(TensorError::DeviceMismatch { lhs: self.device(), rhs: other.device() })
        } else { Ok(()) }
    }

    /// Do a TensorOp on self in CPU and return
    pub fn via_host<R: Element>(&self, f: impl FnOnce(&Tensor<T>) -> Result<Tensor<R>>) -> Result<Tensor<R>> {
        if self.device().is_cpu() { return f(self); }
        let out = f(&self.cpu()?)?;
        if R::CUDA_TYPE.is_some() { out.to_device(self.device()) } else { Ok(out) }
    }

    /// Compact a tensor
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

    /// Return a Contigous tensor
    pub fn to_contiguous(&self) -> Result<Self> {
        if self.layout.is_contiguous() { Ok(self.clone()) } else { self.compact() }
    }

    /// Get the Tensor according to layout view provided
    pub fn view_as(&self, layout: Layout) -> Self { Tensor { storage: self.storage.clone(), layout } }
    
    /// Reshape tensor, supporting one unknown dim
    pub fn reshape(&self, shape: &[isize]) -> Result<Self> {
        // Handle the -1 shape if present
        let known: usize = shape.iter().filter_map(|s| if *s >= 0 {Some(*s as usize)} else {None}).product();
        let n_neg = shape.iter().filter(|&&s| s < 0).count();
        if n_neg > 1 || (n_neg == 1 && (known == 0 || self.numel() % known != 0 /* This is since unknown dim is a whole num multiple cant be fractional */)) {
            return Err(TensorError::InvalidShape(format!("Cannot infer shape {:?}", shape)));
        }
        // Updated shape conisisting of no negatives
        let shape: Vec<usize> = shape.iter().map(|&s| if s < 0 { self.numel() / known } else { s as usize }).collect();
        if shape.iter().product::<usize>() != self.numel() { return Err(TensorError::InvalidShape(format!("Can't reshape {:?} into {:?}", self.shape(), shape)));}
        let base = self.to_contiguous()?;
        Ok(base.view_as(base.layout.reshape(&shape)?))
    }


    /// Reshape to a flat 1D tensor
    pub fn flatten(&self) -> Result<Self> { self.reshape(&[self.numel() as isize]) }

    /// Transpose along two dims
    pub fn transpose(&self, d0: isize, d1: isize) -> Result<Self> {
        let (a, b) = (normalize_dim(d0, self.ndim())?, normalize_dim(d1, self.ndim())?);
        Ok(self.view_as(self.layout.transpose(a, b)?))
    }

    /// Transpose along last two dims    
    pub fn t(&self) -> Result<Self> { self.transpose(-2, -1) }

    /// Takes a 0 indexed array containing all elems till len-1
    /// And then based on the value at that idx reorders the Tensor
    /// Eg: Convert a tensor like 6,7,8,9,10 -> 10,9,7,8,6 using dim=[4,3,1,2,0] 
    pub fn permute(&self, dims: &[usize]) -> Result<Self> { Ok(self.view_as(self.layout.permute(dims)?)) }

    /// Append a new dimension of 1 element
    pub fn unsqueeze(&self, dim: isize) -> Result<Self> { Ok(self.view_as(self.layout.unsqueeze(normalize_dim_inclusive(dim, self.ndim())?)))}

    /// Remove a 1 element dimension
    pub fn squeeze(&self, dim: isize) -> Result<Self> { Ok(self.view_as(self.layout.squeeze(normalize_dim(dim, self.ndim())?)?))}

    /// Remove all 1 element dims
    pub fn squeeze_all(&self) -> Result<Self> {
        let shape: Vec<isize> = self.shape().iter().copied().filter_map(|s| if s != 1{Some(s as isize)}else{None}).collect();
        self.reshape(&shape)
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> { Ok(self.view_as(self.layout.broadcast_to(shape)?)) }    
    pub fn expand(&self, shape: &[usize]) -> Result<Self> { self.broadcast_to(shape) }
    pub fn narrow(&self, dim: isize, start: usize, len: usize) -> Result<Self> { Ok(self.view_as(self.layout.narrow(normalize_dim(dim, self.ndim())?, start, len)?))}
    pub fn select(&self, dim: isize, index: usize) -> Result<Self> { Ok(self.view_as(self.layout.select(normalize_dim(dim, self.ndim())?, index)?))}
    pub fn flip(&self, dim: isize) -> Result<Self> { Ok(self.view_as(self.layout.flip(normalize_dim(dim, self.ndim())?)?))}

    /// Move dim to the last position suggested and made by AI for reductions
    pub fn movedim_last(&self, dim: usize) -> Result<Self> {
        let mut perm: Vec<usize> = (0..self.ndim()).filter(|&d| d != dim).collect();
        perm.push(dim);
        self.permute(&perm)
    }

    /// Parallelised Rayon Map function
    pub fn map<R: Element>(&self, f: impl Fn(T) -> R + Sync + Send) -> Result<Tensor<R>> {
        let data = self.host_data()?;
        let out = par::map_slice(&data, |&x| f(x));
        let t = Tensor::from_vec(out, self.shape())?;
        if R::CUDA_TYPE.is_some() { t.to_device(self.device()) } else { Ok(t) }
    }

    pub fn cast<U: Element>(&self) -> Result<Tensor<U>> { self.map(|x| U::from_f64(x.to_f64())) }

    /// Check if two tensors are equal using absolute and relative tolerance formula
    // To deal with FP
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

    /// Concats Tensors across an existing dim
    // Used AI to make this funcion
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

    /// Stacks tensor along a dim new dim
    pub fn stack(tensors: &[&Tensor<T>], dim: isize) -> Result<Self> {
        let first = *tensors.first().ok_or_else(|| TensorError::InvalidShape("No Tensors provided to stack".into()))?;
        let d = normalize_dim_inclusive(dim, first.ndim())? as isize;
        let expanded: Vec<Tensor<T>> = tensors.iter().map(|t| t.unsqueeze(d)).collect::<Result<_>>()?; // new dim created for each tensor , dim [d0,d1,...dn+1]
        let refs: Vec<&Tensor<T>> = expanded.iter().collect();
        Self::cat(&refs, d) // Cat across th new dim
    }

    /// Slice of indexes along dims
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
