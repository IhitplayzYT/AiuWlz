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

### ops_elementwise.rs - Update to Use CUDA

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
