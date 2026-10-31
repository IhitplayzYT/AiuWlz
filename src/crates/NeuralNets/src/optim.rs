//! Tensor-based parameter optimizers.

use Tensors::{dtype::Float, tensor::Tensor};

use crate::common::{Result, invalid, valid_positive};



/// Epoch based lr
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
    base_learning_rate: T,
    last_epoch: Option<usize>,
}

impl<T: Float> LrScheduler<T> {
    pub fn new(schedule: LrSched, base_learning_rate: T) -> Result<Self> {
        if !base_learning_rate.to_f64().is_finite() || base_learning_rate <= T::zero() { return Err(invalid("base learning rate must be finite and positive")); }
        let valid = match schedule {
            LrSched::Step { step_size, gamma } => { step_size > 0 && valid_positive(gamma) }
            LrSched::Exponential { gamma } => valid_positive(gamma),
            LrSched::CosineAnnealing {steps,lb_lr} => { steps > 0 && lb_lr.is_finite() && lb_lr >= 0.0 && lb_lr <= base_learning_rate.to_f64() }
        };
        if !valid { return Err(invalid("invalid LR sched params")); }
        Ok(Self {schedule,base_learning_rate,last_epoch: None})
    }

    pub fn step_decay(base_learning_rate: T, step_size: usize, gamma: f64) -> Result<Self> {
        Self::new(LrSched::Step { step_size, gamma },base_learning_rate)
    }

    pub fn exponential(base_learning_rate: T, gamma: f64) -> Result<Self> {
        Self::new(LrSched::Exponential { gamma },base_learning_rate)
    }

    pub fn cosine_annealing(lr: T,steps: usize,lb_lr: T) -> Result<Self> {
        Self::new(LrSched::CosineAnnealing {steps,lb_lr:lb_lr.to_f64()},lr)
    }

    pub fn schedule(&self) -> LrSched { self.schedule }

    pub fn last_epoch(&self) -> Option<usize> { self.last_epoch}

    /// Apply the rate for a zero-based epoch. Repeating an epoch is supported;
    /// stepping backwards is rejected to avoid silently rewinding training state.
    pub fn step(&mut self, epoch: usize) -> Result<T> {
        if self.last_epoch.is_some_and(|prev| epoch < prev) { return Err(invalid("learning-rate scheduler epochs must be monotonic")); }
        let base = self.base_learning_rate.to_f64();
        let rate = T::from_f64(match self.schedule {
            LrSched::Step { step_size, gamma } => { base * gamma.powf((epoch / step_size) as f64)}
            LrSched::Exponential { gamma } => base * gamma.powf(epoch as f64),
            LrSched::CosineAnnealing {steps,lb_lr} => {
            // lr = lb_lr + 0.5 * (ub_lr - lb_lr) * (1 + cos(pi * t /T)) , T is total steps,t is current step
            lb_lr + (base - lb_lr) * (1.0 + (std::f64::consts::PI * epoch.min(steps) as f64 / steps as f64).cos()) * 0.5
            }
        });
        self.last_epoch = Some(epoch);
        Ok(rate)
    }
}
