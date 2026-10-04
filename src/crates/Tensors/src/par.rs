// File made using AI for faster tensor ops using rayon for parallel iterables
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const MX_PARALLEL: usize = 1 << 15;

pub fn map_slice<A: Sync, R: Send>(src: &[A], f: impl Fn(&A) -> R + Sync + Send) -> Vec<R> {
    #[cfg(feature = "parallel")]
    if src.len() >= MX_PARALLEL {
        return src.par_iter().map(f).collect();
    }
    src.iter().map(f).collect()
}

pub fn zip_slices<A: Sync, B: Sync, R: Send>(a: &[A], b: &[B], f: impl Fn(&A, &B) -> R + Sync + Send) -> Vec<R> {
    #[cfg(feature = "parallel")]
    if a.len() >= MX_PARALLEL {
        return a.par_iter().zip(b.par_iter()).map(|(x, y)| f(x, y)).collect();
    }
    a.iter().zip(b.iter()).map(|(x, y)| f(x, y)).collect()
}

/// Run `f(row_index, row)` over `out.chunks_mut(row_len)`.
pub fn for_each_row<R: Send>(out: &mut [R], row_len: usize, work_per_row: usize, f: impl Fn(usize, &mut [R]) + Sync + Send) {
    if row_len == 0 {
        return;
    }
    #[cfg(feature = "parallel")]
    if (out.len() / row_len) * work_per_row >= MX_PARALLEL {
        out.par_chunks_mut(row_len).enumerate().for_each(|(i, r)| f(i, r));
        return;
    }
    out.chunks_mut(row_len).enumerate().for_each(|(i,r)| f(i, r));
}
