**17. Add fill kernel for tensor initialization**
```cuda
extern "C" __global__ void fill_k(T* out, size_t n, T value) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = value;
}
```
**Reason:** Enables GPU-side tensor filling instead of CPU allocation then transfer.

**18. Add arange kernel for range generation**
```cuda
extern "C" __global__ void arange_k(T* out, size_t n, T start, T step) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = start + (T)i * step;
}
```
**Reason:** Generates ranges directly on GPU, avoiding CPU allocation and transfer overhead.

**19. Add gather/index_select kernel**
```cuda
extern "C" __global__ void gather_k(T* out, const T* inp, const size_t* indices, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = inp[indices[i]];
}
```
**Reason:** Parallelizes index_select operation which currently uses sequential narrow+cat on CPU.

**20. Add scatter kernel for index assignment**
```cuda
extern "C" __global__ void scatter_k(T* out, const T* values, const size_t* indices, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[indices[i]] = values[i];
}
```
**Reason:** Enables parallel index assignment operations for advanced indexing.

**21. Add transpose kernel for 2D transpose**
```cuda
extern "C" __global__ void transpose_2d_k(T* out, const T* inp, size_t m, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    out[j * m + i] = inp[i * n + j];
}
```
**Reason:** Accelerates 2D transpose operations on GPU instead of relying on stride manipulation which may not be optimal.

**22. Add concat kernel for tensor concatenation**
```cuda
extern "C" __global__ void concat_k(T* out, const T* const* inputs, const size_t* offsets, const size_t* sizes, size_t num_inputs, size_t total_elems) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= total_elems) return;
    size_t input_idx = 0;
    while (i >= offsets[input_idx + 1] && input_idx < num_inputs - 1) input_idx++;
    size_t local_idx = i - offsets[input_idx];
    out[i] = inputs[input_idx][local_idx];
}
```
**Reason:** Parallelizes concatenation which currently does sequential memory copying on CPU.

**23. Add softmax kernel with numerical stability**
```cuda
extern "C" __global__ void softmax_k(T* out, const T* inp, size_t rows, size_t cols) {
    size_t row = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;
    
    extern __shared__ __align__(8) unsigned char smem_raw[];
    T* smax = (T*)smem_raw;
    
    // Find max in row (parallel reduction)
    if (threadIdx.x == 0) {
        T max_val = inp[row * cols];
        for (size_t i = 1; i < cols; i++) {
            if (inp[row * cols + i] > max_val) max_val = inp[row * cols + i];
        }
        smax[0] = max_val;
    }
    __syncthreads();
    
    // Compute exp and sum
    T exp_val = exp{S}(inp[row * cols + col] - smax[0]);
    extern __shared__ __align__(8) unsigned char smem_raw2[];
    T* ssum = (T*)smem_raw2;
    if (threadIdx.x == 0) ssum[0] = (T)0;
    __syncthreads();
    
    atomicAdd(&ssum[0], exp_val);
    __syncthreads();
    
    out[row * cols + col] = exp_val / ssum[0];
}
```
**Reason:** Dedicated softmax kernel with shared memory for max reduction and sum, much faster than current exp-sub-div sequence.

**24. Add layer_norm kernel**
```cuda
extern "C" __global__ void layer_norm_k(T* out, const T* inp, size_t rows, size_t cols, T eps) {
    size_t row = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;
    
    extern __shared__ __align__(8) unsigned char smem_raw[];
    T* smean = (T*)smem_raw;
    T* svar = (T*)(smem_raw + sizeof(T));
    
    if (threadIdx.x == 0) {
        // Compute mean
        T sum = (T)0;
        for (size_t i = 0; i < cols; i++) sum += inp[row * cols + i];
        smean[0] = sum / (T)cols;
        
        // Compute variance
        T var = (T)0;
        for (size_t i = 0; i < cols; i++) {
            T diff = inp[row * cols + i] - smean[0];
            var += diff * diff;
        }
        svar[0] = var / (T)cols;
    }
    __syncthreads();
    
    out[row * cols + col] = (inp[row * cols + col] - smean[0]) / sqrt{S}(svar[0] + eps);
}
```
**Reason:** Single-pass layer norm on GPU with shared memory for mean/variance computation, faster than current sub-mean-div-std sequence.

**25. Improve matmul kernel with tiling and vectorized loads**
```cuda
#define TILE 32
#define VECTOR_WIDTH 4

extern "C" __global__ void matmul_optimized_k(T* out, const T* a, const T* b, size_t m, size_t k, size_t n) {
    size_t bz = blockIdx.z;
    const T* A = a + bz * m * k;
    const T* B = b + bz * k * n;
    T* C = out + bz * m * n;
    
    __shared__ T As[TILE][TILE];
    __shared__ T Bs[TILE][TILE];
    
    size_t row = blockIdx.y * TILE + threadIdx.y;
    size_t col = blockIdx.x * TILE + threadIdx.x;
    
    T acc[VECTOR_WIDTH] = {(T)0, (T)0, (T)0, (T)0};
    
    for (size_t t = 0; t < (k + TILE - 1) / TILE; ++t) {
        // Vectorized loads for better memory bandwidth
        if (row < m && threadIdx.x * VECTOR_WIDTH < TILE) {
            #pragma unroll
            for (int v = 0; v < VECTOR_WIDTH; v++) {
                size_t ac = t * TILE + threadIdx.x * VECTOR_WIDTH + v;
                As[threadIdx.y][threadIdx.x * VECTOR_WIDTH + v] = (ac < k) ? A[row * k + ac] : (T)0;
            }
        }
        if (col < n && threadIdx.y * VECTOR_WIDTH < TILE) {
            #pragma unroll
            for (int v = 0; v < VECTOR_WIDTH; v++) {
                size_t br = t * TILE + threadIdx.y * VECTOR_WIDTH + v;
                Bs[threadIdx.y * VECTOR_WIDTH + v][threadIdx.x] = (br < k) ? B[br * n + col] : (T)0;
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            #pragma unroll
            for (int v = 0; v < VECTOR_WIDTH; v++) {
                acc[v] += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            }
        }
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = acc[0];
    }
}
```
**Reason:** Larger tiles (32x32), vectorized memory loads, and unrolled loops for better memory bandwidth utilization and register reuse.

**26. Improve reduce kernel with warp shuffle**
```cuda
extern "C" __global__ void reduce_optimized_k(T* out, const T* a, size_t cols, int op, T init) {
    extern __shared__ __align__(8) unsigned char smem_raw[];
    T* sh = (T*)smem_raw;
    const T* p = a + (size_t)blockIdx.x * cols;
    
    T v = init;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        v = comb(v, p[j], op);
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    
    // Warp-level shuffle for faster intra-warp reduction
    for (unsigned s = 16; s > 0; s >>= 1) {
        if (threadIdx.x < 32) {
            T other = __shfl_down_sync(0xFFFFFFFF, sh[threadIdx.x], s);
            sh[threadIdx.x] = comb(sh[threadIdx.x], other, op);
        }
        __syncthreads();
    }
    
    // Inter-warp reduction
    if (threadIdx.x < 32) {
        for (unsigned s = blockDim.x / 64; s > 0; s >>= 1) {
            if (threadIdx.x < s * 32) {
                T other = sh[threadIdx.x + s * 32];
                sh[threadIdx.x] = comb(sh[threadIdx.x], other, op);
            }
        }
    }
    
    if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}
```
**Reason:** Uses warp shuffle instructions for faster intra-warp reduction, reducing shared memory access and synchronization overhead.

### cuda/mod.rs - Add New Functions

**27. Add clamp CUDA function**
```rust
pub fn clamp<T: Element>(t: &Tensor<T>, lo: T, hi: T) -> Result<Tensor<T>> {
    let t = t.compact()?;
    let c = cstore(&t)?;
    let n = t.numel();
    let mut out = alloc(c, n)?;
    if n > 0 {
        let f = func::<T>(&c.dev, "clamp_k")?;
        let mut b = c.dev.stream.launch_builder(&f);
        b.arg(&mut out).arg(&c.data).arg(&n).arg(&lo).arg(&hi);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.map_err(cu)?;
    }
    Ok(wrap(out, c, t.shape()))
}
```
**Reason:** Exposes clamp kernel to Rust API for GPU acceleration.

**28. Add where_cond CUDA function**
```rust
pub fn where_cond<T: Element>(cond: &Tensor<T>, a: &Tensor<T>, b: &Tensor<T>, out_shape: &[usize]) -> Result<Tensor<T>> {
    let cond = cond.broadcast_to(out_shape)?.compact()?;
    let a = a.broadcast_to(out_shape)?.compact()?;
    let b = b.broadcast_to(out_shape)?.compact()?;
    let (cc, ca, cb) = (cstore(&cond)?, cstore(&a)?, cstore(&b)?);
    let n = cond.numel();
    let mut out = alloc(cc, n)?;
    if n > 0 {
        let f = func::<T>(&cc.dev, "where_k")?;
        let mut bl = cc.dev.stream.launch_builder(&f);
        bl.arg(&mut out).arg(&cc.data).arg(&ca.data).arg(&cb.data).arg(&n);
        unsafe { bl.launch(LaunchConfig::for_num_elems(n as u32)) }.map_err(cu)?;
    }
    Ok(wrap(out, cc, out_shape))
}
```
**Reason:** Exposes where_cond kernel for GPU acceleration of conditional selection.

**29. Add scalar_op CUDA function**
```rust
pub fn scalar_op<T: Element>(t: &Tensor<T>, scalar: T, op: i32) -> Result<Tensor<T>> {
    let t = t.compact()?;
    let c = cstore(&t)?;
    let n = t.numel();
    let mut out = alloc(c, n)?;
    if n > 0 {
        let f = func::<T>(&c.dev, "scalar_k")?;
        let mut b = c.dev.stream.launch_builder(&f);
        b.arg(&mut out).arg(&c.data).arg(&n).arg(&scalar).arg(&op);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.map_err(cu)?;
    }
    Ok(wrap(out, c, t.shape()))
}
```
**Reason:** Exposes scalar_op kernel to avoid scalar tensor creation overhead.

**30. Add fill CUDA function**
```rust
pub fn fill<T: Element>(shape: &[usize], v: T, ordinal: usize) -> Result<Tensor<T>> {
    let dev = device(ordinal)?;
    let n = shape.iter().product::<usize>();
    let mut out = if n > 0 {
        dev.stream.alloc::<T>(n).map_err(cu)?
    } else {
        dev.stream.alloc_zeros::<T>(1).map_err(cu)?
    };
    if n > 0 {
        let f = func::<T>(&dev, "fill_k")?;
        let mut b = dev.stream.launch_builder(&f);
        b.arg(&mut out).arg(&n).arg(&v);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.map_err(cu)?;
    }
    let c = CudaStorage { data: out, dev: dev.clone(), ordinal };
    Ok(Tensor::from_comp(Storage::Cuda(c), Layout::contiguous(shape)))
}
```
**Reason:** Enables GPU-side tensor filling for zeros/ones/fill operations.

**31. Add arange CUDA function**
```rust
pub fn arange<T: Element>(start: T, end: T, step: T, ordinal: usize) -> Result<Tensor<T>> {
    let dev = device(ordinal)?;
    let mut v = Vec::new();
    let mut x = start;
    while (step > T::zero() && x < end) || (step < T::zero() && x > end) {
        v.push(x);
        x += step;
    }
    let n = v.len();
    let mut out = dev.stream.alloc::<T>(n).map_err(cu)?;
    if n > 0 {
        let f = func::<T>(&dev, "arange_k")?;
        let mut b = dev.stream.launch_builder(&f);
        b.arg(&mut out).arg(&n).arg(&start).arg(&step);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.map_err(cu)?;
    }
    let c = CudaStorage { data: out, dev, ordinal };
    Ok(Tensor::from_comp(Storage::Cuda(c), Layout::contiguous(&[n])))
}
```
**Reason:** Generates ranges directly on GPU avoiding CPU allocation.

**32. Add softmax CUDA function**
```rust
pub fn softmax<T: Float>(t: &Tensor<T>, dim: usize) -> Result<Tensor<T>> {
    let cols = t.shape()[dim];
    if cols == 0 || t.numel() == 0 {
        return t.via_host(|h| h.softmax(dim as isize));
    }
    let moved = t.movedim_last(dim)?.compact()?;
    let c = cstore(&moved)?;
    let rows = moved.numel() / cols;
    let mut out = alloc(c, rows * cols)?;
    const BLOCK: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (((cols + BLOCK - 1) / BLOCK) as u32, rows as u32, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: BLOCK * std::mem::size_of::<T>() as u32 * 2,
    };
    let f = func::<T>(&c.dev, "softmax_k")?;
    let mut b = c.dev.stream.launch_builder(&f);
    b.arg(&mut out).arg(&c.data).arg(&rows).arg(&cols);
    unsafe { b.launch(cfg) }.map_err(cu)?;
    let r = wrap(out, c, moved.shape());
    r.movedim_last(dim)
}
```
**Reason:** Exposes optimized softmax kernel with shared memory reduction.

**33. Add layer_norm CUDA function**
```rust
pub fn layer_norm<T: Float>(t: &Tensor<T>, eps: T) -> Result<Tensor<T>> {
    let cols = t.shape()[t.ndim() - 1];
    if cols == 0 || t.numel() == 0 {
        return t.via_host(|h| h.layer_norm(eps));
    }
    let c = cstore(&t)?;
    let rows = t.numel() / cols;
    let mut out = alloc(c, rows * cols)?;
    const BLOCK: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (((cols + BLOCK - 1) / BLOCK) as u32, rows as u32, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: BLOCK * std::mem::size_of::<T>() as u32 * 2,
    };
    let f = func::<T>(&c.dev, "layer_norm_k")?;
    let mut b = c.dev.stream.launch_builder(&f);
    b.arg(&mut out).arg(&c.data).arg(&rows).arg(&cols).arg(&eps);
    unsafe { b.launch(cfg) }.map_err(cu)?;
    Ok(wrap(out, c, t.shape()))
}
```
**Reason:** Exposes layer_norm kernel for single-pass GPU normalization.

### ops_elementwise.rs - Update to Use CUDA

**34. Update clamp to use CUDA (Line 82-84)**
```rust
pub fn clamp(&self, lo: T, hi: T) -> TensorResult<Self> {
    match &*self.storage {
        Storage::Cpu(_) => self.map(|t| t.map(|x| if x < lo { lo } else if x > hi { hi } else { x })),
        #[cfg(feature = "cuda")]
        Storage::Cuda(_) => crate::cuda::clamp(self, lo, hi),
    }
}
```
**Reason:** Enables GPU acceleration for clamp operation instead of forcing CPU via via_host.

**35. Update where_cond to use CUDA (Line 182-191)**
```rust
pub fn where_cond(&self, a: &Self, b: &Self) -> TensorResult<Self> {
    self.same_device(a)?;
    self.same_device(b)?;
    let shape = broadcast_shapes(&broadcast_shapes(self.shape(), a.shape())?, b.shape())?;
    match (&*self.storage, &*a.storage, &*b.storage) {
        (Storage::Cpu(_), Storage::Cpu(_), Storage::Cpu(_)) => {
            let m = self.broadcast_to(&shape)?.to_vec()?;
            let x = a.broadcast_to(&shape)?.to_vec()?;
            let y = b.broadcast_to(&shape)?.to_vec()?;
            let out: Vec<T> = (0..m.len()).map(|i| if m[i] != T::zero() { x[i] } else { y[i] }).collect();
            Tensor::from_vec(out, &shape)?.to_device(self.device())
        }
        #[cfg(feature = "cuda")]
        (Storage::Cuda(_), Storage::Cuda(_), Storage::Cuda(_)) => crate::cuda::where_cond(self, a, b, &shape),
        #[cfg(feature = "cuda")]
        _ => Err(TensorError::DeviceMismatch { lhs: self.device(), rhs: a.device() }),
    }
}
```
**Reason:** Enables GPU acceleration for where_cond with proper device checking.

**36. Update scalar_op to use CUDA (Line 55-61)**
```rust
pub fn scalar_op(&self, v: T, op: BinaryOp, f: impl Fn(T, T) -> T + Sync + Send) -> TensorResult<Self> {
    match &*self.storage {
        Storage::Cpu(_) => self.map(move |x| f(x, v)),
        #[cfg(feature = "cuda")]
        Storage::Cuda(_) => crate::cuda::scalar_op(self, v, op as i32),
    }
}
```
**Reason:** Uses dedicated scalar kernel on CUDA instead of creating scalar tensor and broadcasting.

### tensor.rs - Update to Use CUDA

**37. Update zeros to use CUDA (Line 78)**
```rust
pub fn zeros(shape: &[usize]) -> Self { 
    Self::fill(shape, T::zero()) 
}
```
**Reason:** Delegates to fill which can use CUDA fill kernel.

**38. Update ones to use CUDA (Line 81)**
```rust
pub fn ones(shape: &[usize]) -> Self { 
    Self::fill(shape, T::one()) 
}
```
**Reason:** Delegates to fill which can use CUDA fill kernel.

**39. Update fill to use CUDA (Line 72-75)**
```rust
pub fn fill(shape: &[usize], v: T) -> Self {
    let n = shape.iter().product();
    #[cfg(feature = "cuda")]
    if T::CUDA_TYPE.is_some() {
        return Self::from_comp(Storage::Cuda(crate::cuda::fill(shape, v, 0).unwrap()), Layout::contiguous(shape));
    }
    Self::from_comp(Storage::Cpu(vec![v; n]), Layout::contiguous(shape))
}
```
**Reason:** Uses CUDA fill kernel when available for GPU tensor creation.

**40. Update arange to use CUDA (Line 96-106)**
```rust
pub fn arange(start: T, end: T, step: T) -> Result<Self> {
    if step == T::zero() { return Err(TensorError::InvalidShape("Step canot be 0".into()));}
    #[cfg(feature = "cuda")]
    if T::CUDA_TYPE.is_some() {
        return crate::cuda::arange(start, end, step, 0);
    }
    let mut v = Vec::new();
    let mut x = start;
    while (step > T::zero() && x < end) || (step < T::zero() && x > end) {
        v.push(x);
        x += step;
    }
    let l = v.len();
    Self::from_vec(v, &[l])
}
```
**Reason:** Uses CUDA arange kernel when available for GPU range generation.

**41. Update linspace to use CUDA (Line 135-139)**
```rust
pub fn linspace(start: T, end: T, steps: usize) -> Self {
    let step = if steps > 1 { (end - start) / T::from_f64((steps - 1) as f64) } else { T::zero() };
    #[cfg(feature = "cuda")]
    if T::CUDA_TYPE.is_some() {
        return crate::cuda::arange(start, end + step, step, 0).unwrap();
    }
    let (st, ed) = (start.to_f64(), end.to_f64());
    let v: Vec<T> = (0..steps).map(|i| T::from_f64(st + (ed - st) * if steps > 1 { i as f64 / (steps - 1) as f64 } else { 0.0 })).collect();
    Self::from_vec(v, &[steps]).unwrap()
}
```
**Reason:** Reuses arange kernel for linspace on CUDA.

### ops_linalg.rs - Update to Use CUDA

**42. Update softmax to use CUDA (Line 254-257)**
```rust
pub fn softmax(&self, dim: isize) -> TensorResult<Self> {
    match &*self.storage {
        Storage::Cpu(_) => {
            let e = self.sub(&self.max(dim, true)?)?.exp()?;
            e.div(&e.sum(dim, true)?)
        }
        #[cfg(feature = "cuda")]
        Storage::Cuda(_) => crate::cuda::softmax(self, normalize_dim(dim, self.ndim())?),
    }
}
```
**Reason:** Uses optimized CUDA softmax kernel instead of sequential operations.

**43. Update layer_norm to use CUDA (Line 274-278)**
```rust
pub fn layer_norm(&self, eps: T) -> TensorResult<Self> {
    match &*self.storage {
        Storage::Cpu(_) => {
            let mean = self.mean(-1, true)?;
            let var = self.var(-1, true, false)?;
            self.sub(&mean)?.div(&var.add_scalar(eps)?.sqrt()?)
        }
        #[cfg(feature = "cuda")]
        Storage::Cuda(_) => crate::cuda::layer_norm(self, eps),
    }
}
```
**Reason:** Uses single-pass CUDA layer_norm kernel instead of multi-step CPU operations.

### par.rs - Improve CPU Parallelization

**44. Add parallel reduce function (New function)**
```rust
pub fn reduce<T: Sync + Send + Copy, R: Send>(data: &[T], init: R, f: impl Fn(R, T) -> R + Sync + Send) -> R {
    #[cfg(feature = "parallel")]
    if data.len() >= MX_PARALLEL {
        return data.par_iter().fold(|| init, |acc, &x| f(acc, x)).reduce(|| init, |a, b| f(a, b));
    }
    data.iter().fold(init, |acc, &x| f(acc, x))
}
```
**Reason:** Adds parallel reduction for CPU operations like sum_all, max_all, min_all to utilize multiple cores.

**45. Add parallel sort function (New function)**
```rust
pub fn sort<T: Send + Ord>(data: &mut [T]) {
    #[cfg(feature = "parallel")]
    if data.len() >= MX_PARALLEL {
        data.par_sort();
        return;
    }
    data.sort();
}
```
**Reason:** Enables parallel sorting for operations like median, percentile, mode that require sorted data.

**46. Update sorted to use parallel sort (ops_stats.rs Line 15-19)**
```rust
fn sorted<T: Element>(s: &[T]) -> Vec<T> {
    let mut v = s.to_vec();
    crate::par::sort(&mut v);
    v
}
```
**Reason:** Uses parallel sort for better CPU performance on large tensors.

**47. Add parallel histogram function (New function in par.rs)**
```rust
pub fn histogram<T: Sync + Send + Copy + Hash>(data: &[T]) -> HashMap<T, usize> {
    #[cfg(feature = "parallel")]
    if data.len() >= MX_PARALLEL {
        return data.par_iter().fold(HashMap::new, |mut acc, &x| {
            *acc.entry(x).or_insert(0) += 1;
            acc
        }).reduce(HashMap::new, |mut acc, mut other| {
            for (k, v) in other.drain() {
                *acc.entry(k).or_insert(0) += v;
            }
            acc
        });
    }
    let mut map = HashMap::new();
    for &x in data {
        *map.entry(x).or_insert(0) += 1;
    }
    map
}
```
**Reason:** Parallel histogram computation for mode operation on large tensors.

**48. Update mode_of to use parallel histogram (ops_stats.rs Line 32-41)**
```rust
fn mode_of<T: Element>(s: &[T]) -> T {
    if s.is_empty() { return T::zero(); }
    let hist = crate::par::histogram(s);
    hist.into_iter().max_by_key(|&(_, count)| count).map(|(val, _)| val).unwrap_or(T::zero())
}
```
**Reason:** Uses parallel histogram for O(n) mode computation instead of O(n log n) sort-based approach.

**49. Add parallel argmin/argmax function (New function in par.rs)**
```rust
pub fn argmin<T: Sync + Send + Copy + PartialOrd>(data: &[T]) -> usize {
    #[cfg(feature = "parallel")]
    if data.len() >= MX_PARALLEL {
        return data.par_chunks(MX_PARALLEL)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let (local_min_idx, _) = chunk.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
                chunk_idx * MX_PARALLEL + local_min_idx
            })
            .min_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap())
            .unwrap();
    }
    data.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0
}

pub fn argmax<T: Sync + Send + Copy + PartialOrd>(data: &[T]) -> usize {
    #[cfg(feature = "parallel")]
    if data.len() >= MX_PARALLEL {
        return data.par_chunks(MX_PARALLEL)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let (local_max_idx, _) = chunk.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
                chunk_idx * MX_PARALLEL + local_max_idx
            })
            .max_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap())
            .unwrap();
    }
    data.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0
}
```
**Reason:** Parallel argmin/argmax for better CPU performance on large tensors.

**50. Update argmax/argmin to use parallel versions (ops_stats.rs Lines 112-127)**
```rust
pub fn argmax(&self, dim: isize, keepdim: bool) -> TensorResult<Tensor<i64>> {
    self.reduce_with(dim, keepdim, |s| crate::par::argmax(s) as i64)
}

pub fn argmin(&self, dim: isize, keepdim: bool) -> TensorResult<Tensor<i64>> {
    self.reduce_with(dim, keepdim, |s| crate::par::argmin(s) as i64)
}
```
**Reason:** Uses parallel argmin/argmax for better CPU performance.
