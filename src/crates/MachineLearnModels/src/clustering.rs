pub mod Clustering{
    use std::{error::Error, f64::INFINITY, num::FpCategory::Zero};

use Tensors::{dtype::Float, tensor::{Tensor, deref_as_f64}};

#[derive(Debug,Clone)]
pub struct KNN{
    cfg: KnnConfig

}


#[derive(Debug,Clone)]
pub struct KnnConfig{
   pub k: usize,
   pub metric: Metric,
   pub init: InitMethod
}


#[derive(Debug,Clone)]
pub enum Metric{
    Manhatten,
    Euclidien,
    Minkowski(f64),
    Chebyshev,
    Pdim(f64),
}


#[derive(Debug,Clone)]
pub enum InitMethod{
    Starting,
    Ending,
    Random,
    PartitionedMedian
}

pub fn ClusteringInitMethod<T:Float>(x:InitMethod,v:Vec<T>,k:usize) -> Result<Vec<T>,Box<dyn Error>>{
    match x{
        InitMethod::Starting => { Ok(v.windows(k).next().unwrap().to_vec())},
        InitMethod::Ending => { Ok(v.windows(k).last().unwrap().to_vec())},
        InitMethod::Random => {
            while k


        },
        InitMethod::PartitionedMedian => {},
    }
}


pub fn dist<T:Float>(v1: &Vec<T>,v2: &Vec<T>,cfg: &KnnConfig) -> f64{
    let (v1,v2) = (deref_as_f64(v1),deref_as_f64(v2));
    match cfg.metric{
        Metric::Minkowski(v) | Metric::Pdim(v) => pdim(&v1, &v2, v),
        Metric::Euclidien => pdim(&v1, &v2, 2.0),
        Metric::Chebyshev => pdim(&v1, &v2, f64::INFINITY),
        Metric::Manhatten => pdim(&v1, &v2, 1.0)
    }
}

pub fn pdim<T:Float>(v1: &Vec<T>,v2: &Vec<T>,p:f64) -> T{
    match p{
        1.0 => v1.iter().zip(v2.iter()).fold(T::zero(),|acc, (v1,v2)| acc + (*v1-*v2).abs()),
        2.0 => v1.iter().zip(v2.iter()).fold(T::zero(),|acc, (v1,v2)| acc + (*v1-*v2).powf(T::from_f64(2.0))),
        f64::INFINITY => v1.iter().zip(v2.iter()).map(|(a,b)| (*a-*b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(T::zero()),
        _ => v1.iter().zip(v2.iter()).fold(T::zero(),|acc,(a,b)| { acc + (*a- *b).abs().powf(T::from_f64(p))}).powf(-T::from_f64(p))
    }
}


}