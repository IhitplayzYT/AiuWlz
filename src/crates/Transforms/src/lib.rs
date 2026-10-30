pub mod dimensionality_reduction {
    use MachineLearnModels::errors::ModelError;
    use Tensors::{dtype::Float, tensor::Tensor};

    #[derive(Debug, Clone)]
    pub struct PCA<T: Float> {
        n_components: usize,
        components: Option<Vec<Vec<T>>>,
        mean: Option<Vec<T>>,
        explained_variance: Option<Vec<T>>,
    }

    #[derive(Debug, Clone)]
    pub struct PCAConfig {
        pub n_components: usize,
    }

    impl Default for PCAConfig {
        fn default() -> Self {
            Self { n_components: 2 }
        }
    }

    impl<T: Float + std::iter::Sum> PCA<T> {
        pub fn new(config: PCAConfig) -> Self {
            Self {n_components: config.n_components,components: None,mean: None,explained_variance: None}
        }

        pub fn fit(&mut self, x: &Tensor<T>) -> Result<(),ModelError> {
            if x.ndim() != 2 { return Err(ModelError::InvalidData("X must be 2-dimensional".into()));}
            let n = x.shape()[0];
            let p = x.shape()[1];
            let data = x.to_vec()?;
            let rows: Vec<Vec<T>> = (0..n).map(|i| data[i * p..(i + 1) * p].to_vec()).collect();
            let mean = self.compute_mean(&rows);
            let centered = self.center_data(&rows, &mean);
            let cov = self.compute_covariance(&centered);
            let components = self.compute_eigenvectors(&cov, self.n_components);

            // Compute explained variance
            let explained_variance = self.compute_explained_variance(&centered, &components);

            self.components = Some(components);
            self.mean = Some(mean);
            self.explained_variance = Some(explained_variance);
            Ok(())
        }

        pub fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
            let components = self.components.as_ref().ok_or("Model not fitted").map_err(|x| ModelError::ConceptError(x.into()))?;
            let mean = self.mean.as_ref().ok_or("Model not fitted").map_err(|x| ModelError::ConceptError(x.into()))?;
            let n = x.shape()[0];
            let p = x.shape()[1];
            let data = x.to_vec()?;
            let rows: Vec<Vec<T>> = (0..n).map(|i| data[i * p..(i + 1) * p].to_vec()).collect();
            let centered = self.center_data(&rows, mean);
            let transformed = self.project(&centered, components);
            let mut result = Vec::with_capacity(n * self.n_components);
            for row in &transformed { result.extend(row); }
            Ok(Tensor::from_vec(result, &[n, self.n_components])?)
        }

        pub fn fit_transform(&mut self, x: &Tensor<T>) -> Result<Tensor<T>,ModelError> {
            self.fit(x)?;
            self.transform(x)
        }

        pub fn components(&self) -> Option<&Vec<Vec<T>>> { self.components.as_ref() }

        pub fn explained_variance(&self) -> Option<&Vec<T>> { self.explained_variance.as_ref() }

        fn compute_mean(&self, rows: &[Vec<T>]) -> Vec<T> {
            let n_features = rows[0].len();
            let mut mean = vec![T::zero(); n_features];
            for row in rows {
                for (i, &val) in row.iter().enumerate() { mean[i] += val; }
            }
            for i in 0..n_features { mean[i] /= T::from_f64(rows.len() as f64);}
            mean
        }

        fn center_data(&self, rows: &[Vec<T>], mean: &[T]) -> Vec<Vec<T>> { rows.iter().map(|row| {row.iter().zip(mean.iter()).map(|(&x, &m)| x - m).collect()}).collect() }

        fn compute_covariance(&self, centered: &[Vec<T>]) -> Vec<Vec<T>> {
            let n = centered[0].len();
            let p = centered.len();
            let mut cov = vec![vec![T::zero(); n]; n];
            for i in 0..n {
                for j in i..n {
                    let mut sum = T::zero();
                    for row in centered { sum += row[i] * row[j]; }
                    cov[i][j] = sum / T::from_f64((p - 1) as f64);
                    cov[j][i] = cov[i][j];
                }
            }
            cov
        }

        // Made using AI
        fn compute_eigenvectors(&self,cov: &[Vec<T>],n_components: usize) -> Vec<Vec<T>> {
            let n = cov.len();
            let mut components = Vec::with_capacity(n_components.min(n));
            let mut matrix = cov.to_vec();
            for _ in 0..n_components.min(n) {
                let mut eigenvector = vec![T::one(); n];
                let norm: T = eigenvector.iter().map(|&x| x * x).sum::<T>().sqrt();
                for v in eigenvector.iter_mut() { *v /= norm;}
                for _ in 0..100 {
                    let mut new_vec = vec![T::zero(); n];
                    for i in 0..n {
                        for j in 0..n {
                            new_vec[i] += matrix[i][j] * eigenvector[j];
                        }
                    }

                    let norm: T = new_vec.iter().map(|&x| x * x).sum::<T>().sqrt();
                    if norm < T::from_f64(1e-10) { break; }
                    for v in new_vec.iter_mut() { *v /= norm; }
                    let diff: T = eigenvector.iter().zip(new_vec.iter()).map(|(&a, &b)| (a - b).abs()).sum();
                    eigenvector = new_vec;
                    if diff < T::from_f64(1e-6) { break; }
                }

                let mut eigenvalue = T::zero();
                for i in 0..n {
                    for j in 0..n {
                        eigenvalue += eigenvector[i] * matrix[i][j] * eigenvector[j];
                    }
                }

                // Deflation: C = C - lambda * v * v^T
                for i in 0..n {
                    for j in 0..n {
                        matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
                    }
                }
                components.push(eigenvector);
            }
            components
        }

        fn compute_explained_variance(&self, centered: &[Vec<T>], components: &[Vec<T>]) -> Vec<T> {
            let n = centered.len() as f64;
            let mut variance = Vec::with_capacity(components.len());
            for component in components {
                let mut sum = T::zero();
                for row in centered {
                    let proj:T = row.iter().zip(component.iter()).map(|(&x, &c)| x * c).sum();
                    sum += proj * proj;
                }
                variance.push(sum / T::from_f64(n));
            }
            variance
        }

        fn project(&self, data: &[Vec<T>], components: &[Vec<T>]) -> Vec<Vec<T>> {
            data.iter().map(|row| {components.iter().map(|comp| row.iter().zip(comp.iter()).map(|(&x, &c)| x * c).sum()).collect()}).collect()
        }
    }

    // t-SNE (t-Distributed Stochastic Neighbor Embedding)
    #[derive(Debug, Clone)]
    pub struct TSNE<T: Float> {
        n_components: usize,
        learning_rate: T,
        n_iter: usize,
        perplexity:f64,
        embedding: Option<Vec<Vec<T>>>,
    }

    #[derive(Debug, Clone)]
    pub struct TSNEConfig<T: Float> {
        pub n_components: usize,
        pub perplexity: f64,
        pub learning_rate: T,
        pub n_iter: usize,
    }

    impl<T: Float> Default for TSNEConfig<T> {
        fn default() -> Self { Self {n_components: 2,perplexity: 30.0,learning_rate: T::from_f64(200.0),n_iter: 1000} }
    }

    impl<T: Float + std::iter::Sum> TSNE<T> {
        pub fn new(config: TSNEConfig<T>) -> Self { Self {n_components: config.n_components,learning_rate: config.learning_rate,n_iter: config.n_iter,embedding: None,perplexity:30.0} }

        pub fn fit_transform(&mut self, x: &Tensor<T>) -> Result<Tensor<T>, ModelError> {
            if x.ndim() != 2 { return Err(ModelError::InvalidData("X must be 2-dimensional".into())); }
            let n_samples = x.shape()[0];
            let n_features = x.shape()[1];
            let data = x.to_vec()?;
            let rows: Vec<Vec<T>> = (0..n_samples).map(|i| data[i * n_features..(i + 1) * n_features].to_vec()).collect();

            // random init
            let _rng = rand::thread_rng();
            let mut embedding: Vec<Vec<T>> = (0..n_samples).map(|_| {(0..self.n_components).map(|_| T::from_f64(rand::random::<f64>() * 0.0001)).collect()}).collect();

            let p = self.p_affine(&rows)?;

            // Gradient descent optimization
            for _ in 0..self.n_iter {
                let q = self.q_affine(&embedding)?;
                let gradient = self.compute_gradient(&p, &q, &embedding);

                for (i, grad) in gradient.iter().enumerate() {
                    for (j, &g) in grad.iter().enumerate() {
                        embedding[i][j] -= self.learning_rate * g;
                    }
                }
            }
            self.embedding = Some(embedding.clone());

            let mut result = Vec::with_capacity(n_samples * self.n_components);
            for row in &embedding { result.extend(row); }
            Ok(Tensor::from_vec(result, &[n_samples, self.n_components])?)
        }

        pub fn embedding(&self) -> Option<&Vec<Vec<T>>> { self.embedding.as_ref() }

        // Made using AI assistance
        fn p_affine(&self, rows: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
            let n_samples = rows.len();
            if n_samples < 2 { return Err(ModelError::InvalidData("t-SNE requires at least 2 samples".into())); }
            let n_features = rows[0].len();
            if n_features == 0 || rows.iter().any(|r| r.len() != n_features) { return Err(ModelError::InvalidData("Invalid or inconsistent feature dimensions".into())); }
            let zero = T::zero();
            let one = T::one();
            let two = T::from_f64(2.0);
            let tolerance = T::from_f64(1e-5);

            let mut dist = vec![vec![zero; n_samples]; n_samples];
            for i in 0..n_samples {
                for j in (i + 1)..n_samples {
                    let d: T = rows[i].iter().zip(rows[j].iter()).map(|(&a, &b)| {(a - b) * (a - b)}).sum();
                    dist[i][j] = d;
                    dist[j][i] = d;
                }
            }

            let mut p = vec![vec![zero; n_samples]; n_samples];
            let mut tgt_p = T::from_f64(self.perplexity);
            let mx_p = T::from_f64((n_samples - 1) as f64);
            if tgt_p < one { tgt_p = one; }
            if tgt_p > mx_p { tgt_p = mx_p; }
            let tgt_e = tgt_p.ln();
            for i in 0..n_samples {

                let mut beta = one;
                let mut beta_min = zero;
                let mut beta_max: Option<T> = None;
                for _ in 0..50 {
                    let mut sum = zero;
                    let mut w_dist = zero;
                    for j in 0..n_samples {
                        if i == j { continue;}
                        let v = (-beta * dist[i][j]).exp();
                        sum += v;
                        w_dist += dist[i][j] * v;
                    }
                    if sum <= zero { break; }

                    // H(P_i) = ln(sum) + beta * weighted_dist / sum
                    let entropy = sum.ln() + (beta * w_dist / sum);
                    let diff = entropy - tgt_e;
                    if diff.abs() < tolerance {break;}

                    if diff > zero {
                        beta_min = beta;
                        // Binary search to adj beta
                        beta = match beta_max {
                            Some(max) => (beta + max) / two,
                            None => beta * two,
                        };
                    } else {
                        beta_max = Some(beta);
                        beta = (beta_min + beta) / two;
                    }
                }

                // Final conditional probs
                let mut sum = zero;

                for j in 0..n_samples {
                    if i == j {continue;}
                    p[i][j] = (-beta * dist[i][j]).exp();
                    sum += p[i][j];
                }

                if sum > zero {
                    for j in 0..n_samples { p[i][j] /= sum; }
                }
            }

            let mut p_sym = vec![vec![zero; n_samples]; n_samples];
            let normalization = T::from_f64(2.0 * n_samples as f64);
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j { p_sym[i][j] = (p[i][j] + p[j][i]) / normalization; }
                }
            }
            Ok(p_sym)
        }

        fn q_affine(&self, embedding: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
            let n_samples = embedding.len();
            let mut q = vec![vec![T::zero(); n_samples]; n_samples];
            for i in 0..n_samples {
                for j in (i + 1)..n_samples {
                    let dist: T = embedding[i].iter().zip(embedding[j].iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
                    q[i][j] = (T::one() + dist).powf(T::from_f64(-1.0));
                    q[j][i] = q[i][j];
                }
            }
            let sum: T = q.iter().flat_map(|row| row.iter().copied()).sum();
            if sum > T::zero() {
                for i in 0..n_samples {
                    for j in 0..n_samples { q[i][j] /= sum; }
                }
            }
            Ok(q)
        }

        fn compute_gradient(&self, p: &[Vec<T>], q: &[Vec<T>], embedding: &[Vec<T>]) -> Vec<Vec<T>> {
            let n_samples = embedding.len();
            let mut gradient = vec![vec![T::zero(); self.n_components]; n_samples];
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j {
                        for k in 0..self.n_components {
                            gradient[i][k] += ((p[i][j] - q[i][j]) * (T::one() + q[i][j])) * (embedding[i][k] - embedding[j][k]);
                        }
                    }
                }
            }
            gradient
        }

    }

}
