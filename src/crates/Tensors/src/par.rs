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

/// Map integer positions without allocating a temporary index vector.
pub fn map_range<R: Send>(len: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    #[cfg(feature = "parallel")]
    if len >= MX_PARALLEL {
        return (0..len).into_par_iter().map(f).collect();
    }
    (0..len).map(f).collect()
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

pub fn reduce<T: Sync + Send + Copy, R: Send + Copy>(data: &[T], init: R, f: impl Fn(R, T) -> R + Sync + Send) -> R {
    data.iter().fold(init, |acc, &x| f(acc, x))
}

pub fn sort<T: Send + Ord>(data: &mut [T]) {
    #[cfg(feature = "parallel")]
    if data.len() >= MX_PARALLEL {
        data.par_sort();
        return;
    }
    data.sort();
}

pub fn reverse<T: Send>(data: &mut [T]) {
    #[cfg(feature = "parallel")]
    if data.len() >= MX_PARALLEL {
        data.reverse();
        return;
    }
    data.reverse();
}

pub fn sort_by<T: Send>(data: &mut [T], compare: impl Fn(&T, &T) -> std::cmp::Ordering + Sync + Send) {
    #[cfg(feature = "parallel")]
    if data.len() >= MX_PARALLEL {
        data.par_sort_by(compare);
        return;
    }
    data.sort_by(compare);
}


pub fn argmin<T: Sync + Send + Copy + PartialOrd>(data: &[T]) -> usize {
    #[cfg(feature = "parallel")]
    if data.len() >= MX_PARALLEL {
        return data.par_chunks(MX_PARALLEL).enumerate().map(|(chunk_idx, chunk)| {
                let (local_min_idx, _) = chunk.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
                chunk_idx * MX_PARALLEL + local_min_idx
            }).min_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap()).unwrap();
    }
    data.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0
}

pub fn argmax<T: Sync + Send + Copy + PartialOrd>(data: &[T]) -> usize {
    #[cfg(feature = "parallel")]
    if data.len() >= MX_PARALLEL {
        return data.par_chunks(MX_PARALLEL).enumerate().map(|(chunk_idx, chunk)| {
                let (local_max_idx, _) = chunk.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
                chunk_idx * MX_PARALLEL + local_max_idx
            }).max_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap()).unwrap();
    }
    data.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0
}
