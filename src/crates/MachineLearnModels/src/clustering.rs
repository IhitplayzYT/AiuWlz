pub mod Clustering{
    use std::{error::Error, marker::PhantomData};
    use Tensors::{dtype::Float, tensor::{Tensor, deref_as_f64}};
    use rand::seq::SliceRandom;
    use crate::{common::euclid_dist, errors::ModelError};

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

#[derive(Debug,Clone,Copy)]
pub enum InitMethod{
    Starting,
    Ending,
    Random,
    PartitionedMedian
}

pub fn ClusteringInitMethod<T:Float>(x:InitMethod, data: &Tensor<T>, k:usize) -> Result<Vec<Vec<T>>,ModelError>{
    if data.ndim() != 2 {return Err(ModelError::InvalidData("Data must be 2-dimensional".into()));}
    let n = data.shape()[0];
    let p = data.shape()[1];
    if n < k {return Err(ModelError::InvalidData("Number of samples must be at least k".into()));}
    let data_vec = data.to_vec().map_err(|e| ModelError::ConceptError(e.to_string()))?;
    match x{
        InitMethod::Starting => {
            let mut centroids = Vec::with_capacity(k);
            for i in 0..k {
                let start = i * p;
                centroids.push(data_vec[start..start+p].to_vec());
            }
            Ok(centroids)
        },
        InitMethod::Ending => {
            let mut centroids = Vec::with_capacity(k);
            for i in 0..k {
                let idx = n - k + i;
                let start = idx * p;
                centroids.push(data_vec[start..start+p].to_vec());
            }
            Ok(centroids)
        },
        InitMethod::Random => {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            let mut centroids = Vec::with_capacity(k);
            for i in 0..k {
                let idx = indices[i];
                let start = idx * p;
                centroids.push(data_vec[start..start+p].to_vec());
            }
            Ok(centroids)
        },
        InitMethod::PartitionedMedian => {
            let step = n / k;
            let mut centroids = Vec::with_capacity(k);
            for i in 0..k {
                let idx = i * step;
                let start = idx * p;
                centroids.push(data_vec[start..start+p].to_vec());
            }
            Ok(centroids)
        },
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
        _ => v1.iter().zip(v2.iter()).fold(T::zero(),|acc,(a,b)| { acc + (*a- *b).abs().powf(T::from_f64(p))}).powf(T::from_f64(1.0/p))
    }
}

// ============================================== KMEANS ========================================================

#[derive(Debug,Clone)]
pub struct KMeans<T: Float>{
    cfg: KMeansConfig<T>,
    centroids: Option<Vec<Vec<T>>>,
}

#[derive(Debug,Clone)]
pub struct KMeansConfig<T: Float>{
    pub k: usize,
    pub max_iter: usize,
    pub tolerance: T,
    pub init: InitMethod,
}

impl<T: Float> Default for KMeansConfig<T> {
    fn default() -> Self { Self {k: 3,max_iter: 100,tolerance: T::from_f64(1e-4),init: InitMethod::Random} }
}

impl<T: Float + std::iter::Sum> KMeans<T> {
    pub fn new(config: KMeansConfig<T>) -> Self {Self { cfg: config, centroids: None }}

    pub fn fit(&mut self, x: &Tensor<T>) -> Result<(), ModelError> {
        if x.ndim() != 2 { return Err(ModelError::InvalidData("X must be 2-dimensional".into())); }
        let n = x.shape()[0];
        let p = x.shape()[1];
        let data = x.to_vec().map_err(|e| ModelError::ConceptError(e.to_string()))?;
        let rows: Vec<Vec<T>> = (0..n).map(|i| data[i * p..(i + 1) * p].to_vec()).collect();
        let mut centroids = ClusteringInitMethod(self.cfg.init, x, self.cfg.k)?;
        for _ in 0..self.cfg.max_iter {
            let mut clusters: Vec<Vec<usize>> = vec![vec![]; self.cfg.k];
            for (idx, row) in rows.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_dist = T::from_f64(f64::MAX);
                for (i, centroid) in centroids.iter().enumerate() {
                    let d = euclid_dist(row, centroid);
                    if d < best_dist {
                        best_dist = d;
                        best_cluster = i;
                    }
                }
                clusters[best_cluster].push(idx);
            }

            let mut new_centroids = centroids.clone();
            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                if cluster.is_empty() { continue; }
                // Mean of the values in each cluster to decide new centroids
                for feat in 0..p { new_centroids[cluster_idx][feat] =  cluster.iter().map(|&idx| rows[idx][feat]).sum::<T>() / T::from_f64(cluster.len() as f64); }
            }
            // Check converg 
            let max_shift = centroids.iter().zip(new_centroids.iter()).map(|(old, new)| euclid_dist(old, new)).fold(T::zero(), |acc, d| if d > acc { d } else { acc });
            centroids = new_centroids;
            if max_shift < self.cfg.tolerance { break; }
        }
        self.centroids = Some(centroids);
        Ok(())
    }

    pub fn predict(&self, x: &Tensor<T>) -> Result<Vec<usize>,ModelError> {
        let centroids = self.centroids.as_ref().ok_or("Model not fitted").map_err(|x| ModelError::ConceptError(x.into()))?;
        let n = x.shape()[0];
        let p = x.shape()[1];
        let data = x.to_vec().map_err(|e| ModelError::ConceptError(e.to_string()))?;
        let rows: Vec<Vec<T>> = (0..n).map(|i| data[i * p..(i + 1) * p].to_vec()).collect();
        let mut ret = Vec::with_capacity(n);
        for row in rows.iter() {
            let mut best_cluster = 0;
            let mut best_dist = T::from_f64(f64::MAX);
            for (i, centroid) in centroids.iter().enumerate() {
                let d = euclid_dist(row, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_cluster = i;
                }
            }
            ret.push(best_cluster);
        }
        Ok(ret)
    }

    pub fn fit_predict(&mut self, x: &Tensor<T>) -> Result<Vec<usize>,ModelError> {
        self.fit(x)?;
        self.predict(x)
    }

    pub fn centroids(&self) -> Option<&Vec<Vec<T>>> { self.centroids.as_ref() }

}

// ============================================== DBSCAN ========================================================
// Core,border and noise pts
// Mark in a set 
// Then density connection

#[derive(Debug,Clone)]
pub struct DBSCAN<T: Float> {
    eps: T,
    min: usize,
    labels: Option<Vec<i32>>,
}

#[derive(Debug,Clone)]
pub struct DBSCANConfig<T: Float> {
    pub eps: T,
    pub min: usize,
}

impl<T: Float> Default for DBSCANConfig<T> {
    fn default() -> Self { Self {eps: T::from_f64(0.5),min: 5} }
}

impl<T: Float> DBSCAN<T> {
    pub fn new(config: DBSCANConfig<T>) -> Self { Self { eps: config.eps, min: config.min, labels: None} }

    pub fn fit(&mut self, x: &Tensor<T>) -> Result<(), ModelError> {
        if x.ndim() != 2 { return Err(ModelError::InvalidData("X must be 2-dimensional".into())); }
        let n = x.shape()[0];
        let p = x.shape()[1];
        let data = x.to_vec()?;
        let rows: Vec<Vec<T>> = (0..n).map(|i| data[i * p..(i + 1) * p].to_vec()).collect();
        let mut labels = vec![0i32; n];
        let mut cluster_id = 0;
        for i in 0..n {
            if labels[i] != 0 { continue; }
            let neighbors = self.region_query(&rows, i);
            if neighbors.len() < self.min { labels[i] = -1;} // Noise 
            else { self.expand_cluster(&rows, &mut labels, i, &neighbors, &mut cluster_id); }
        }
        self.labels = Some(labels);
        Ok(())
    }

    pub fn fit_predict(&mut self, x: &Tensor<T>) -> Result<Vec<i32>,ModelError> {
        self.fit(x)?;
        Ok(self.labels.clone().unwrap())
    }

    pub fn labels(&self) -> Option<&Vec<i32>> { self.labels.as_ref() }

    fn region_query(&self, rows: &[Vec<T>], point_idx: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            if euclid_dist(&rows[point_idx], row) <= self.eps { neighbors.push(i); }
        }
        neighbors
    }

    // Made using AI
    fn expand_cluster(&self,rows: &[Vec<T>],labels: &mut [i32],point_idx: usize,neighbors: &[usize],cluster_id: &mut i32) {
        *cluster_id += 1;
        labels[point_idx] = *cluster_id;
        let mut seed_queue: Vec<usize> = neighbors.to_vec();
        while let Some(curr_idx) = seed_queue.pop() {
            if labels[curr_idx] == -1 { labels[curr_idx] = *cluster_id; }
            if labels[curr_idx] != 0 { continue; }
            labels[curr_idx] = *cluster_id;
            let current_neighbors = self.region_query(rows, curr_idx);
            if current_neighbors.len() >= self.min { seed_queue.extend(current_neighbors); }
        }
    }

}

// ========================================== Hierarchical Clustering ========================================================


#[derive(Debug)]
pub struct HierarchicalClustering<T: Float> {
    n_clusters: usize,
    linkage: Linkage,
    labels: Option<Vec<usize>>,
    _phantom: PhantomData<T>
}

#[derive(Debug,Clone)]
pub enum Linkage {
    Single,
    Complete,
    Average,
    Ward,
}

#[derive(Debug)]
pub struct HierarchicalConfig<T: Float> {
    pub n_clusters: usize,
    pub linkage: Linkage,
    _phantom: PhantomData<T>
}

impl<T: Float> Default for HierarchicalConfig<T> {
    fn default() -> Self { Self {n_clusters: 2,linkage: Linkage::Average,_phantom:PhantomData} }
}

impl<T: Float> HierarchicalClustering<T> {
    pub fn new(config: HierarchicalConfig<T>) -> Self { Self {n_clusters: config.n_clusters,linkage: config.linkage,labels: None,_phantom:PhantomData} }

    pub fn fit(&mut self, x: &Tensor<T>) -> Result<(), ModelError> {
        if x.ndim() != 2 { return Err(ModelError::InvalidData("X must be 2-dimensional".into())); }
        let n = x.shape()[0];
        let p = x.shape()[1];
        let data = x.to_vec()?;
        let rows: Vec<Vec<T>> = (0..n).map(|i| data[i * p..(i + 1) * p].to_vec()).collect();
        let mut labels = vec![0usize; n];
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        while clusters.len() > self.n_clusters {
            let (best_i, best_j) = self.find_closest_clusters(&rows, &clusters)?;
            let merged = clusters[best_i].iter().chain(clusters[best_j].iter()).copied().collect();
            clusters.remove(best_j.max(best_i));
            clusters.remove(best_i.min(best_j));
            clusters.push(merged);
        }

        for (cluster_id, cluster) in clusters.iter().enumerate() {
            for &idx in cluster {labels[idx] = cluster_id;}
        }

        self.labels = Some(labels);
        Ok(())
    }

    pub fn fit_predict(&mut self, x: &Tensor<T>) -> Result<Vec<usize>,ModelError> {
        self.fit(x)?;
        Ok(self.labels.clone().unwrap())
    }

    pub fn labels(&self) -> Option<&Vec<usize>> { self.labels.as_ref() }

    fn find_closest_clusters(&self, rows: &[Vec<T>], clusters: &[Vec<usize>]) -> Result<(usize, usize), ModelError> {
        let mut best_dist = T::from_f64(f64::MAX);
        let mut best_pair = (0, 0);
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let dist = self.cluster_dist(rows, &clusters[i], &clusters[j]);
                if dist < best_dist {
                    best_dist = dist;
                    best_pair = (i, j);
                }
            }
        }
        Ok(best_pair)
    }

    fn cluster_dist(&self, rows: &[Vec<T>], cluster1: &[usize], cluster2: &[usize]) -> T {
        match self.linkage {
            Linkage::Single => {
                let mut min_dist = T::from_f64(f64::MAX);
                for &i in cluster1 {
                    for &j in cluster2 {
                        let d = euclid_dist(&rows[i], &rows[j]);
                        if d < min_dist { min_dist = d; }
                    }
                }
                min_dist
            }
            Linkage::Complete => {
                let mut max_dist = T::zero();
                for &i in cluster1 {
                    for &j in cluster2 {
                        let d = euclid_dist(&rows[i], &rows[j]);
                        if d > max_dist  {max_dist = d; }
                    }
                }
                max_dist
            }
            Linkage::Average => {
                let mut sum = T::zero();
                let mut count = 0usize;
                for &i in cluster1 {
                    for &j in cluster2 {
                        sum += euclid_dist(&rows[i], &rows[j]);
                        count += 1;
                    }
                }
                sum / T::from_f64(count as f64)
            }
            Linkage::Ward => {
            // dist(c1,c2)^2 * (l1*l2)/(l1+l2)
                let c1 = self.centroid(rows, cluster1);
                let c2 = self.centroid(rows, cluster2);
                let n1 = cluster1.len() as f64;
                let n2 = cluster2.len() as f64;
                let dist = euclid_dist(&c1, &c2);
                dist * dist * T::from_f64((n1 * n2) / (n1 + n2))
            }
        }
    }

    fn centroid(&self, rows: &[Vec<T>], cluster: &[usize]) -> Vec<T> {
        let p = rows[0].len();
        let mut centroid = vec![T::zero(); p];        
        for &idx in cluster {
            for (i, &val) in rows[idx].iter().enumerate() { centroid[i] += val;}
        }        
        for k in 0..p { centroid[k] /= T::from_f64(cluster.len() as f64); }
        centroid
    }

}




}