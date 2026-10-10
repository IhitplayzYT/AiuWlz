//! CUDA C source, compiled at runtime with NVRTC once per (device, dtype).

const SRC: &str = r#"
typedef {T} T;

// Apply a element Op -> Sum = 0, Max = 1, Min = 0
__device__ __forceinline__ T comb(T a, T b, int op) { return op == 0 ? a + b : (op == 1 ? (a > b ? a : b) : (a < b ? a : b)); }

// Strided multi-thread iter into a dense matrix 
extern "C" __global__ void strided_copy(T* out, const T* inp, size_t n, size_t ndim,const long long* shape, const long long* strides, size_t offset) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t rem = i;
    long long off = (long long)offset;
    // Iterate from last dim to first dim
    for (int d = (int)ndim - 1; d >= 0; --d) {
        size_t s = (size_t)shape[d]; // Dim len
        off += (long long)(rem % s) * strides[d];
        rem /= s;
    }
    out[i] = inp[off];
}

// Parallel Clamping(Used for clipping and exploding gradients)
extern "C" __global__ void clamp_k(T* out, const T* a, size_t n, T lo, T hi) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    T x = a[i];
    out[i] = x < lo ? lo : (x > hi ? hi : x);
}

// Parallel where_cond
extern "C" __global__ void where_k(T* out, const T* cond, const T* a, const T* b, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = cond[i] != (T)0 ? a[i] : b[i];
}

//  opcode=>
//  0  => negation
//  1  => abs
//  2  => exp
//  3  => log
//  4  => sqrt 
//  5  => sin 
//  6  => cos
//  7  => tanh 
//  8  => sigmoid 
//  9  => relu
//  10 => inv 
//  11 => square 
//  12 => gelu 
//  13 => floor 
//  other => ceil 
extern "C" __global__ void unary_k(T* out, const T* a, size_t n, int op) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    T x = a[i];
    switch (op) {
        case 0:  out[i] = -x; break;
        case 1:  out[i] = fabs{S}(x); break;
        case 2:  out[i] = exp{S}(x); break;
        case 3:  out[i] = log{S}(x); break;
        case 4:  out[i] = sqrt{S}(x); break;
        case 5:  out[i] = sin{S}(x); break;
        case 6:  out[i] = cos{S}(x); break;
        case 7:  out[i] = tanh{S}(x); break;
        case 8:  out[i] = (T)1 / ((T)1 + exp{S}(-x)); break;
        case 9:  out[i] = x > (T)0 ? x : (T)0; break;
        case 10: out[i] = (T)1 / x; break;
        case 11: out[i] = x * x; break;
        case 12: out[i] = (T)0.5 * x * ((T)1 + tanh{S}((T)0.7978845608028654 * (x + (T)0.044715 * x * x * x))); break;
        case 13: out[i] = floor{S}(x); break;
        default: out[i] = ceil{S}(x); break;
    }
}

//  opcode=>
//  0  => add 
//  1  => sub
//  2  => mul 
//  3  => div 
//  4  => max 
//  5  => min 
//  6  => eq
//  7  => neq 
//  8  => gt 
//  9  => lt 
//  10 => ge 
//  11 => le 
//  other => pow 
extern "C" __global__ void binary_k(T* out, const T* a, const T* b, size_t n, int op) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    T x = a[i], y = b[i], r;
    switch (op) {
        case 0:  r = x + y; break;
        case 1:  r = x - y; break;
        case 2:  r = x * y; break;
        case 3:  r = x / y; break;
        case 4:  r = x > y ? x : y; break;
        case 5:  r = x < y ? x : y; break;
        case 6:  r = x == y ? (T)1 : (T)0; break;
        case 7:  r = x != y ? (T)1 : (T)0; break;
        case 8:  r = x > y ? (T)1 : (T)0; break;
        case 9:  r = x < y ? (T)1 : (T)0; break;
        case 10:  r = x >= y ? (T)1 : (T)0; break;
        case 11: r = x <= y ? (T)1 : (T)0; break;
        default: r = pow{S}(x, y); break;
    }
    out[i] = r;
}

// Batched [m,k] x [k,n] with 16x16 shared-memory tiles. blockIdx.z = batch index.
// Matrix_mul
#define TILE 16
extern "C" __global__ void matmul_k(T* out, const T* a, const T* b, size_t m, size_t k, size_t n) {
    size_t bz = blockIdx.z;
    const T* A = a + bz * m * k;
    const T* B = b + bz * k * n;
    T* C = out + bz * m * n;
    __shared__ T As[TILE][TILE];
    __shared__ T Bs[TILE][TILE];
    size_t row = blockIdx.y * TILE + threadIdx.y;
    size_t col = blockIdx.x * TILE + threadIdx.x;
    T acc = (T)0;
    for (size_t t = 0; t < (k + TILE - 1) / TILE; ++t) {
        size_t ac = t * TILE + threadIdx.x;
        size_t br = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < m && ac < k) ? A[row * k + ac] : (T)0;
        Bs[threadIdx.y][threadIdx.x] = (br < k && col < n) ? B[br * n + col] : (T)0;
        // Ensure batch buffer filled 
        __syncthreads();
        // Aggrgating multip between elem of the row of buffa and col of buffb
        for (int i = 0; i < TILE; ++i) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) C[row * n + col] = acc;
}

// One block per row; reduces `cols` contiguous elements (sum / max / min).
// Suggested by AI
extern "C" __global__ void reduce_k(T* out, const T* a, size_t cols, int op, T init) {
    extern __shared__ __align__(8) unsigned char smem_raw[];
    T* sh = (T*)smem_raw;
    const T* p = a + (size_t)blockIdx.x * cols;
    T v = init;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) v = comb(v, p[j], op);
    sh[threadIdx.x] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = comb(sh[threadIdx.x], sh[threadIdx.x + s], op);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}

// Parallel Scalar Op
extern "C" __global__ void scalar_k(T* out, const T* a, size_t n, T scalar, int op) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    T x = a[i];
    switch (op) {
        case 0:  out[i] = x + scalar; break;
        case 1:  out[i] = x - scalar; break;
        case 2:  out[i] = x * scalar; break;
        case 3:  out[i] = x / scalar; break;
        default: out[i] = pow{S}(x, scalar); break;
    }
}
"#;

pub fn source(ctype: &str) -> String {
    let suffix = if ctype == "float" { "f" } else { "" };
    SRC.replace("{T}", ctype).replace("{S}", suffix)
}
