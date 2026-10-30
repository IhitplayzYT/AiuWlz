use Tensors::{autograd::Comp, dtype::Float, tensor::Tensor};
use crate::common::{Loss, Result, invalid};
use Metrics::metrics::Loss as MetricLoss;

pub fn loss_comp<T: Float>(prediction: &Comp<T>,target: &Tensor<T>,loss: Loss) -> Result<Comp<T>> {
    if prediction.shape() != target.shape() { return Err(invalid(format!("loss target shape {:?} does not match prediction shape {:?}",target.shape(),prediction.shape()))); }
    if prediction.numel() == 0 { return Err(invalid("loss cannot be computed for an empty tensor")); }
    prediction.data().same_device(target)?;
    let zero = Comp::constant(prediction.data().zeros_like());
    let value = match loss {
        Loss::MeanSquaredError => prediction.sub(&Comp::constant(target.clone()))?.square()?.mean_all()?,
        Loss::MeanAbsoluteError => prediction.sub(&Comp::constant(target.clone()))?.abs()?.mean_all()?,
        Loss::BinaryCrossEntropy => {
            validate_unit_interval(target, "binary cross entropy targets")?;
            let p = prediction.clamp(T::from_f64(1e-7), T::one() - T::from_f64(1e-7))?;
            let y = Comp::constant(target.clone());
            let one = Comp::constant(p.data().ones_like());
            y.mul(&p.ln()?)?.add(&one.sub(&y)?.mul(&one.sub(&p)?.ln()?)?)?.neg()?.mean_all()?
        }
        Loss::CategoricalCrossEntropy => {
            validate_categorical(prediction, target)?;
            let p = prediction.clamp(T::from_f64(1e-7), T::one())?;
            Comp::constant(target.clone()).mul(&p.ln()?.neg()?)?.sum_all()?.div_scalar(T::from_f64(target.shape()[0] as f64))?
        }
        Loss::BinaryCrossEntropyWithLogits => {
            validate_unit_interval(target, "binary cross entropy targets")?;
            let y = Comp::constant(target.clone());
            let stable_softplus = prediction.maximum(&zero)?.add(&prediction.abs()?.neg()?.exp()?.add_scalar(T::one())?.ln()?)?;
            stable_softplus.sub(&prediction.mul(&y)?)?.mean_all()?
        }
        Loss::CategoricalCrossEntropyWithLogits => {
            validate_categorical(prediction, target)?;
            Comp::constant(target.clone()).mul(&prediction.log_softmax(-1)?.neg()?)?.sum_all()?.div_scalar(T::from_f64(target.shape()[0] as f64))?
        }
        Loss::Huber { delta } => {
            if !delta.is_finite() || delta <= 0.0 { return Err(invalid("Huber delta must be finite and positive")); }
            let diff = prediction.sub(&Comp::constant(target.clone()))?;
            let abs = diff.abs()?;
            let mask = abs.data().le(&Tensor::scalar(T::from_f64(delta)).to_device(abs.device())?)?;
            let quadratic = diff.square()?.mul_scalar(T::from_f64(0.5))?;
            let linear = abs.mul_scalar(T::from_f64(delta))?.sub_scalar(T::from_f64(0.5 * delta * delta))?;
            Comp::where_cond(&mask, &quadratic, &linear)?.mean_all()?
        }
        Loss::SmoothL1 => {
            let diff = prediction.sub(&Comp::constant(target.clone()))?;
            let abs = diff.abs()?;
            let mask = abs.data().le(&Tensor::scalar(T::one()).to_device(abs.device())?)?;
            let quadratic = diff.square()?.mul_scalar(T::from_f64(0.5))?;
            let linear = abs.sub_scalar(T::from_f64(0.5))?;
            Comp::where_cond(&mask, &quadratic, &linear)?.mean_all()?
        }
        Loss::KullbackLeibler => {
            if target.min_all()? < T::zero() { return Err(invalid("KL targets cannot be negative")); }
            let eps = T::from_f64(1e-7);
            let y = Comp::constant(target.clamp(eps, T::highest())?);
            let p = prediction.clamp(eps, T::highest())?;
            y.mul(&y.div(&p)?.ln()?)?.mean_all()?
        }
        Loss::Hinge => {
            let y = Comp::constant(target.clone());
            Comp::constant(prediction.data().ones_like()).sub(&prediction.mul(&y)?)?.maximum(&zero)?.mean_all()?
        }
        Loss::PoissonNll => {
            let p = prediction.clamp(T::from_f64(1e-7), T::highest())?;
            p.sub(&Comp::constant(target.clone()).mul(&p.ln()?)?)?.mean_all()?
        }
    };
    Ok(value)
}

/// Compute a scalar loss and predict grad
pub fn loss_and_gradient<T: Float>(prediction: &Tensor<T>,target: &Tensor<T>,loss: Loss) -> Result<(Tensor<T>, Tensor<T>)> {
    let pred = Comp::leaf(prediction.clone());
    let value = loss_comp(&pred, target, loss)?;
    let scalar = value.data();
    value.backward()?;
    let gradient = pred.grad().ok_or_else(|| invalid("autograd did not produce a prediction gradient"))?;
    Ok((scalar, gradient))
}

/// Use the workspace Metrics crate for the losses it already defines.
pub fn metrics_loss_value<T: Float>(prediction: &Tensor<T>,target: &Tensor<T>,loss: Loss) -> Result<T> {
    Ok(match loss {
        Loss::MeanSquaredError => MetricLoss::MSE(prediction, target)?,
        Loss::MeanAbsoluteError => MetricLoss::MAE(prediction, target)?,
        Loss::BinaryCrossEntropy => MetricLoss::BCE(prediction, target)?,
        Loss::CategoricalCrossEntropy => MetricLoss::CCE(prediction, target)?,
        Loss::Huber { delta } => MetricLoss::Huber(prediction, target, T::from_f64(delta))?,
        Loss::SmoothL1 => MetricLoss::SmoothL1(prediction, target)?,
        Loss::KullbackLeibler => {MetricLoss::KLdiv(prediction, target)? / T::from_f64(prediction.numel() as f64)}
        Loss::Hinge => MetricLoss::Hinge(prediction, target)?,
        Loss::BinaryCrossEntropyWithLogits | Loss::CategoricalCrossEntropyWithLogits | Loss::PoissonNll => {loss_comp(&Comp::constant(prediction.clone()), target, loss)?.item()?}
    })
}

fn validate_unit_interval<T: Float>(target: &Tensor<T>, name: &str) -> Result<()> {
    if target.min_all()? < T::zero() || target.max_all()? > T::one() { return Err(invalid(format!("{name} must be in [0, 1]"))); }
    Ok(())
}

fn validate_categorical<T: Float>(prediction: &Comp<T>, target: &Tensor<T>) -> Result<()> {
    if prediction.ndim() != 2 || prediction.shape()[0] == 0 { return Err(invalid("categorical cross entropy expects [batch, classes]"));}
    if target.min_all()? < T::zero() { return Err(invalid("categorical targets cannot be negative"));}
    Ok(())
}

pub fn one_hot<T: Float>(labels: &Tensor<i64>,classes: usize,device: Utilities::device::Device::Device) -> Result<Tensor<T>> {
    if classes == 0 { return Err(invalid("one-hot class count must be positive"));}
    if labels.numel() > 0 && (labels.min_all()? < 0 || labels.max_all()? >= classes as i64) { return Err(invalid("class label is outside the configured class range"));}
    let count = labels.numel();
    let indices = labels.reshape(&[count as isize, 1])?.to_device(device)?;
    let base = Tensor::<T>::zeros(&[count, classes]).to_device(device)?;
    let values = Tensor::<T>::ones(&[count, 1]).to_device(device)?;
    let mut output_shape: Vec<isize> = labels.shape().iter().map(|&d| d as isize).collect();
    output_shape.push(classes as isize);
    Ok(base.scatter(1, &indices, &values)?.reshape(&output_shape)?)
}
