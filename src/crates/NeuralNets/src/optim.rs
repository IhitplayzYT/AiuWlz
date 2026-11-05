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


}


