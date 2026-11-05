use Tensors::{dtype::Float, tensor::Tensor};
use crate::common::{Result, invalid, valid_nonnegative, valid_positive, valid_unit};

/// Epoch based lr sched
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum  LrSched{
    /// lr * gamma for every step_size 
    Step { step_size: usize, gamma: f64 },
    /// lr * gamma every epoch
    Exponential { gamma: f64 },
    /// Go from lr -> lb_lr in steps, slow lr reduction at start then fast descent
    /// lr = lb_lr + 0.5 * (ub_lr - lb_lr) * (1 + cos(pi * t /T)) , T is total steps,t is current step
    CosineAnnealing {steps: usize,lb_lr: f64},
}

#[derive(Clone, Debug)]
pub struct LrScheduler<T: Float> {
    schedule: LrSched,
    base_lr: T,
    last_epoch: Option<usize>,
}

impl<T: Float> LrScheduler<T> {
    pub fn new(schedule: LrSched, base_lr: T) -> Result<Self> {
        if !base_lr.to_f64().is_finite() || base_lr <= T::zero() { return Err(invalid("base learning rate must be finite and positive")); }
        let valid = match schedule {
            LrSched::Step { step_size, gamma } => { step_size > 0 && valid_positive(gamma) }
            LrSched::Exponential { gamma } => valid_positive(gamma),
            LrSched::CosineAnnealing {steps,lb_lr} => { steps > 0 && lb_lr.is_finite() && lb_lr >= 0.0 && lb_lr <= base_lr.to_f64() }
        };
        if !valid { return Err(invalid("invalid LR sched params")); }
        Ok(Self {schedule,base_lr,last_epoch: None})
    }
    pub fn step_decay(base_lr: T, step_size: usize, gamma: f64) -> Result<Self> { Self::new(LrSched::Step { step_size, gamma },base_lr) }
    pub fn exponential(base_lr: T, gamma: f64) -> Result<Self> { Self::new(LrSched::Exponential { gamma },base_lr) }
    pub fn cosine_annealing(lr: T,steps: usize,lb_lr: T) -> Result<Self> { Self::new(LrSched::CosineAnnealing {steps,lb_lr:lb_lr.to_f64()},lr) }
    pub fn schedule(&self) -> LrSched { self.schedule }
    pub fn last_epoch(&self) -> Option<usize> { self.last_epoch}

    pub fn step(&mut self, epoch: usize,optimizer: &mut Optimizer<T>) -> Result<T> {
        if self.last_epoch.is_some_and(|prev| epoch < prev) { return Err(invalid("learning-rate scheduler epochs must be monotonic")); }
        let base = self.base_lr.to_f64();
        let rate = T::from_f64(match self.schedule {
            LrSched::Step { step_size, gamma } => { base * gamma.powf((epoch / step_size) as f64)}
            LrSched::Exponential { gamma } => base * gamma.powf(epoch as f64),
            LrSched::CosineAnnealing {steps,lb_lr} => {
            // lr = lb_lr + 0.5 * (ub_lr - lb_lr) * (1 + cos(pi * t /T)) , T is total steps,t is curr step
            lb_lr + (base - lb_lr) * (1.0 + (std::f64::consts::PI * epoch.min(steps) as f64 / steps as f64).cos()) * 0.5
            }
        });
        optimizer.set_lr(rate)?;
        self.last_epoch = Some(epoch);
        Ok(rate)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum OptimizerKind {
    Sgd,
    Momentum {momentum: f64},
    Nesterov {momentum: f64},
    RmsProp {rho: f64,epsilon: f64},
    Adagrad {epsilon: f64,lr_decay: f64,weight_decay: f64,initial_accumulator: f64},
    Adadelta {rho: f64,epsilon: f64},
    Adam {beta1: f64,beta2: f64,epsilon: f64},
    AdamW {beta1: f64,beta2: f64,epsilon: f64,weight_decay: f64},
}

pub struct Optimizer<T: Float> {
    pub lr: T,
    pub kind: OptimizerKind,
    pub max_grad_norm: Option<T>,
    dx: Vec<Tensor<T>>,
    dx2: Vec<Tensor<T>>,
    step: usize,
}

impl<T: Float> Optimizer<T> {

    // Used AI suggested defualts for params
    pub fn sgd(lr: T) -> Result<Self> { Self::new(lr, OptimizerKind::Sgd) }
    pub fn momentum(lr: T, momentum: f64) -> Result<Self> { Self::new(lr, OptimizerKind::Momentum { momentum }) }
    pub fn nesterov(lr: T, momentum: f64) -> Result<Self> { Self::new(lr, OptimizerKind::Nesterov { momentum }) }
    pub fn rmsprop(lr: T, rho: f64, epsilon: f64) -> Result<Self> { Self::new(lr, OptimizerKind::RmsProp { rho, epsilon }) }
    pub fn adam(lr: T) -> Result<Self> { Self::new(lr,OptimizerKind::Adam {beta1: 0.9,beta2: 0.999,epsilon: 1e-8}) }
    pub fn adamw(lr: T, weight_decay: f64) -> Result<Self> { Self::new(lr,OptimizerKind::AdamW {beta1: 0.9,beta2: 0.999,epsilon: 1e-8,weight_decay}) }
    pub fn adagrad(lr: T) -> Result<Self> { Self::new(lr,OptimizerKind::Adagrad {epsilon: 1e-10,lr_decay: 0.0,weight_decay: 0.0,initial_accumulator: 0.0}) }
    pub fn adadelta(lr: T) -> Result<Self> { Self::new(lr,OptimizerKind::Adadelta {rho: 0.9,epsilon: 1e-6}) }

    pub fn new(lr: T, kind: OptimizerKind) -> Result<Self> {
        if !lr.to_f64().is_finite() || lr <= T::zero() { return Err(invalid("Lr must be finite and positive")); }
        let valid = match kind {
            OptimizerKind::Sgd => true,
            OptimizerKind::Momentum { momentum } | OptimizerKind::Nesterov { momentum } => { valid_unit(momentum) }
            OptimizerKind::RmsProp { rho, epsilon } | OptimizerKind::Adadelta { rho, epsilon } => { valid_unit(rho) && valid_positive(epsilon) }
            OptimizerKind::Adam {beta1,beta2,epsilon} => valid_unit(beta1) && valid_unit(beta2) && valid_positive(epsilon),
            OptimizerKind::AdamW {beta1,beta2,epsilon,weight_decay} => {valid_unit(beta1) && valid_unit(beta2) && valid_positive(epsilon) && valid_nonnegative(weight_decay)} 
            OptimizerKind::Adagrad {epsilon,lr_decay,weight_decay,initial_accumulator,} => { valid_positive(epsilon) && valid_nonnegative(lr_decay) && valid_nonnegative(weight_decay) && valid_nonnegative(initial_accumulator) }
        };
        if !valid { return Err(invalid("optimizer coeffs must be in [0,1) and epsilon must be positive")); }
        Ok(Self {lr,kind,max_grad_norm: None,dx: Vec::new(),dx2: Vec::new(),step: 0})
    }

    pub fn reset(&mut self) {
        self.dx.clear();
        self.dx2.clear();
        self.step = 0;
    }
    
    pub fn step_count(&self) -> usize { self.step}
    
    pub fn set_lr(&mut self, lr: T) -> Result<()> {
        if !lr.to_f64().is_finite() || lr <= T::zero() { return Err(invalid("Lr must be finite and positive")); }
        self.lr = lr;
        Ok(())
    }

    pub fn with_grad_clip(mut self, max_norm: T) -> Result<Self> {
        if !max_norm.to_f64().is_finite() || max_norm <= T::zero() { return Err(invalid("Grad clip norm must be positive and finite")); }
        self.max_grad_norm = Some(max_norm);
        Ok(self)
    }

    pub fn update(&mut self, params: &mut [Tensor<T>], grads: &[Tensor<T>]) -> Result<()> {
        if params.len() != grads.len() { return Err(invalid("Diff no of param and grads recieved")); }
        for (p, g) in params.iter().zip(grads) {
            if p.shape() != g.shape() { return Err(invalid(format!("Grad shape {:?} does not match param shape {:?}",g.shape(),p.shape()))); }
            p.same_device(g)?;
        }

        let grads = if let Some(max_norm) = self.max_grad_norm {
            let t = grads.iter().map(Tensor::flatten).collect::<std::result::Result<Vec<Tensor<T>>, _>>()?;
            let refs: Vec<&Tensor<T>> = t.iter().collect();
            let norm = if refs.is_empty() { T::zero() } else { Tensor::cat(&refs, 0)?.square()?.sum_all()?.sqrt() };
            // Scale down the vecs to max of max_norm
            if norm > max_norm { grads.iter().map(|g| g.mul_scalar(max_norm / norm)).collect::<std::result::Result<Vec<_>, _>>()?} 
            else { grads.to_vec() }
        } else {
            grads.to_vec()
        };
        if self.dx.len() != params.len() || self.dx2.len() != params.len() || self.dx.iter().zip(params.iter()).any(|(m, p)| m.shape() != p.shape()) || self.dx2.iter().zip(params.iter()).any(|(v, p)| v.shape() != p.shape()) {
            self.dx = params.iter().map(Tensor::zeros_like).collect();
            self.dx2 = params.iter().map(|p| match self.kind {
                    OptimizerKind::Adagrad {initial_accumulator,..} => p.fill_like(T::from_f64(initial_accumulator)),
                    _ => p.zeros_like(),
                }).collect();
            self.step = 0;
        }
        
        self.step += 1;
        let lr = self.lr;
        for i in 0..params.len() {
            let update = match self.kind {
                // grad = grad
                OptimizerKind::Sgd => grads[i].clone(),
                // acc_grad = gamma * acc_grad+ cur_grad
                // update = acc_grad
                OptimizerKind::Momentum { momentum } => {
                    self.dx[i] = self.dx[i].mul_scalar(T::from_f64(momentum))?.add(&grads[i])?;
                    self.dx[i].clone()
                },
                // acc_grad = acc_grad * gamma + cur_grad
                // update = acc_grad * gamma + grad
                OptimizerKind::Nesterov { momentum } => {
                    self.dx[i] = self.dx[i].mul_scalar(T::from_f64(momentum))?.add(&grads[i])?;
                    grads[i].add(&self.dx[i].mul_scalar(T::from_f64(momentum))?)?
                },
                // acc_momentum = acc_momentum * rho + grad^2 * (1 - rho)\
                // update = grad / (sqrt(acc_momentum) + eps)
                OptimizerKind::RmsProp { rho, epsilon } => {
                    let rho = T::from_f64(rho);
                    self.dx2[i] = self.dx2[i].mul_scalar(rho)?.add(&grads[i].square()?.mul_scalar(T::one() - rho)?)?;
                    grads[i].div(&self.dx2[i].sqrt()?.add_scalar(T::from_f64(epsilon))?)?
                },
                // acc_grad = acc_grad * beta1 + grad * (1 - beta1)
                // acc_momentum = acc_momentum * beta2 + grad^2 * (1 - beta2)
                // Zero init bias during earlys timesteps
                // acc_grad = acc_grad / (1 - beta1^t)
                // acc_momentum = acc_momentum / (1 - beta2^t)
                // update = acc_momentum / (sqrt(acc_grad) + eps)
                OptimizerKind::Adam {beta1,beta2,epsilon} => {
                    let (b1, b2) = (T::from_f64(beta1), T::from_f64(beta2));
                    self.dx[i] = self.dx[i].mul_scalar(b1)?.add(&grads[i].mul_scalar(T::one() - b1)?)?;
                    self.dx2[i] = self.dx2[i].mul_scalar(b2)?.add(&grads[i].square()?.mul_scalar(T::one() - b2)?)?;
                    let m = self.dx[i].div_scalar(T::from_f64(1.0 - beta1.powf(self.step as f64)))?;
                    let v = self.dx2[i].div_scalar(T::from_f64(1.0 - beta2.powf(self.step as f64)))?;
                    m.div(&v.sqrt()?.add_scalar(T::from_f64(epsilon))?)?
                },
                // acc_grad = acc_grad * beta1 + grad * (1 - beta1)
                // acc_momentum = acc_momentum * beta2 + grad^2 * (1 - beta2)
                // Zero init bias during earlys timesteps
                // acc_grad = acc_grad / (1 - beta1^t)
                // acc_momentum = acc_momentum / (1 - beta2^t)
                // update = acc_momentum / (sqrt(acc_grad) + eps)
                // param = param - param * lr * decay - update * lr
                // param = param(1 - lr*decay) - update*lr
                OptimizerKind::AdamW {beta1,beta2,epsilon,..} => {
                    let (b1, b2) = (T::from_f64(beta1), T::from_f64(beta2));
                    self.dx[i] = self.dx[i].mul_scalar(b1)?.add(&grads[i].mul_scalar(T::one() - b1)?)?;
                    self.dx2[i] = self.dx2[i].mul_scalar(b2)?.add(&grads[i].square()?.mul_scalar(T::one() - b2)?)?;
                    let m = self.dx[i].div_scalar(T::from_f64(1.0 - beta1.powf(self.step as f64)))?;
                    let v = self.dx2[i].div_scalar(T::from_f64(1.0 - beta2.powf(self.step as f64)))?;
                    m.div(&v.sqrt()?.add_scalar(T::from_f64(epsilon))?)?
                },
                // Used AI to make adagrad sicne a bigt confused on formula
                OptimizerKind::Adagrad {epsilon,lr_decay,weight_decay,..} => {
                    let grad = if weight_decay != 0.0 {grads[i].add(&params[i].mul_scalar(T::from_f64(weight_decay))?)?} else {grads[i].clone()};
                    self.dx2[i] = self.dx2[i].add(&grad.square()?)?;
                    grad.div(&self.dx2[i].sqrt()?.add_scalar(T::from_f64(epsilon))?)?.mul_scalar(T::from_f64(1.0 / (1.0 + lr_decay * (self.step - 1) as f64)))?
                },
                // acc_grad = acc_grad * rho + (1 - rho) * grad^2
                // update = sqrt(acc_momentum + eps)/ sqrt(acc_grad + eps) * grad
                // acc_momentum = rho * acc_momentum + (1 - rho) * update^2
                OptimizerKind::Adadelta { rho, epsilon } => {
                    let rho = T::from_f64(rho);
                    self.dx[i] = self.dx[i].mul_scalar(rho)?.add(&grads[i].square()?.mul_scalar(T::one() - rho)?)?;
                    let update = self.dx2[i].add_scalar(T::from_f64(epsilon))?.sqrt()?.div(&self.dx[i].add_scalar(T::from_f64(epsilon))?.sqrt()?)?.mul(&grads[i])?;
                    self.dx2[i] = self.dx2[i].mul_scalar(rho)?.add(&update.square()?.mul_scalar(T::one() - rho)?)?;
                    update
                }
            };

            match self.kind {
                OptimizerKind::AdamW { weight_decay, .. } if weight_decay != 0.0 => { params[i] = params[i].mul_scalar(T::one() - lr * T::from_f64(weight_decay))?.sub(&update.mul_scalar(lr)?)?; }
                _ => params[i] = params[i].sub(&update.mul_scalar(lr)?)?,
            }
        }
        Ok(())
    }
}


