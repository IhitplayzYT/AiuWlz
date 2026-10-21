use Tensors::tensor::Tensor;
use Tensors::error::{TensorError,TensorResult};
use Tensors::dtype::{Element,Float};

pub mod Loss {
    use super::*;
    /// Mean squared error
    /// MSE = mean((pred - target)^2)
    pub fn MSE<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<T> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch { op: "mse", lhs: pred.shape().to_vec(), rhs: target.shape().to_vec()});}
        pred.sub(target)?.square()?.mean_all()
    }

    /// Mean absolute error
    /// MAE = mean(|pred - target|)
    pub fn MAE<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<T> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "mae",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        pred.sub(target)?.abs()?.mean_all()
    }

    /// Root mean squared error
    /// RMSE = sqrt(mean((pred - target)^2))
    pub fn RMSE<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<T> { Ok(MSE(pred, target)?.sqrt()) }

    /// Huber Loss
    /// delta: thresh where loss transitions from quadratic to linear
    /// Loss = {|diff| < delta -> 0.5 * diff^2,delta * (diff - 0.5 * diff)}
    pub fn Huber<T: Float>(pred: &Tensor<T>, target: &Tensor<T>, delta: T) -> TensorResult<T> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "huber",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        let diff = pred.sub(target)?;
        let abs_diff = diff.abs()?;
        let quadratic = abs_diff.le(&Tensor::scalar(delta))?;
        let ret = quadratic.where_cond(&diff.square()?.mul_scalar(T::from_f64(0.5))?,&abs_diff.mul_scalar(delta)?.sub(&Tensor::scalar(delta * delta * T::from_f64(0.5)))?)?;
        ret.mean_all()
    }

    /// Smooth L1 (Huber with delta=1.0)
    pub fn SmoothL1<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<T> { Huber(pred, target, T::one()) }

    /// Binary Cross Entropy 
    /// BCE = -mean(target * log( pred)+ (1 - target)* log(1- pred))
    pub fn BCE<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<T> {
        if pred.shape() != target.shape() { return Err( TensorError::ShapeMismatch {op: "binary_cross_entropy",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()} ); }
        let eps = T::epsilon();
        let clamped = pred.clamp(eps, T::one() - eps)?;
        target.mul(&clamped.ln()?)?.add(& target.sub_scalar(T::one())?.mul(&clamped.sub_scalar(T::one())?.ln()?)?)?.neg()?.mean_all()
    }

    /// Binary Cross Entropy with Logits as input
    pub fn BCElogits<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<T> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "binary_cross_entropy_with_logits",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        let mx = pred.neg()?.max_all()?;
        let term1 = pred.neg()?.mul_scalar(T::one())?.exp()?.ln()?.add_scalar(mx)?;
        let term2 = pred.sub_scalar(mx)?.exp()?.ln()?.add_scalar(mx)?;
        pred.mul(target)?.sub(&term1)?.add(&term2)?.mean_all()
    }

    /// Categorical cross entropy 
    /// CE = -sum(target * log(pred))
    /// pred: predicted probabilities, shape [batch, num_classes]
    /// target: one-hot encoded labels, shape [batch, num_classes]
    pub fn CCE<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<T> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "cross_entropy",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        let eps = T::epsilon();
        let clamped = pred.clamp(eps, T::one())?;
        Ok(target.mul(&clamped.ln()?)?.neg()?.sum_all()? / T::from_f64(pred.shape()[0] as f64))
    }

    /// Cross Entropy with Logits (combines LogSoftmax + NLLLoss)
    /// pred: raw logits, shape [batch, num_classes]
    /// target: class indices, shape [batch]
    pub fn CCElogits<T: Float>(pred: &Tensor<T>, target: &Tensor<i64>) -> TensorResult<T> {
        if pred.ndim() != 2 { return Err(TensorError::InvalidShape("CCElogits expects 2D logits".into())); }
        let (batch_sz, n_class) = (pred.shape()[0], pred.shape()[1]);
        if target.numel() != batch_sz { return Err(TensorError::ShapeMismatch {op: "CCElogits",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        // Log softmax = z - sum(log(e^z))
        let logits = pred.sub(& pred.max(1, true)?)?;
        let log_softmax = logits.exp()?.sum(1, true)?.ln()?.neg()?.add(&logits)?;

        let target = target.to_vec()?;
        let log_softmax = log_softmax.to_vec()?;
        let mut loss = T::zero();
        for i in 0..batch_sz {
            let idx = target[i] as usize;
            if idx < n_class { loss += log_softmax[i * n_class + idx]; }
        }
        Ok(loss.neg() / T::from_f64(batch_sz as f64))
    }

    /// Negative Log Likelihood
    /// NLL = -mean(log(pred[target_class]))
    /// pred: log probabilities, shape [batch, num_classes]
    /// target: class indices, shape [batch]
    pub fn NLL<T: Float>(pred: &Tensor<T>, target: &Tensor<i64>) -> TensorResult<T> {
        if pred.ndim() != 2 { return Err(TensorError::InvalidShape("nll_loss expects 2D log probabilities".into())); }
        let (batch_sz, n_class) = (pred.shape()[0], pred.shape()[1]);
        if target.numel() != batch_sz { return Err(TensorError::ShapeMismatch {op: "nll_loss",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        
        let pred_data = pred.to_vec()?;
        let target = target.to_vec()?;
        let mut loss = T::zero();
        for i in 0..batch_sz {
            let idx = target[i] as usize;
            if idx < n_class { loss -= pred_data[i * n_class + idx]; }
        }
        Ok(loss / T::from_f64(batch_sz as f64))
    }

    /// KL Divergence Loss
    /// KL(t || p) = sum(t * log(t/p))
    pub fn KLdiv<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<T> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "kl_div",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        let eps = T::epsilon();
        let p_clamped = pred.clamp(eps, T::one())?;
        let t_clamped = target.clamp(eps, T::one())?;
        t_clamped.mul(&t_clamped.div(&p_clamped)?.ln()?)?.sum_all()
    }

    /// Hinge Loss
    /// Hinge = mean(relu(1 - t*p))
    pub fn Hinge<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<T> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "hinge",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        Tensor::scalar(T::one()).sub(&pred.mul(target)?)?.relu()?.mean_all()
    }

    /// Cosine Loss
    pub fn CosineLoss<T: Float>(input1: &Tensor<T>,input2: &Tensor<T>,target: &Tensor<T>,margin: T) -> TensorResult<T> {
        if input1.shape() != input2.shape() || input1.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "cosine_embedding_loss",lhs: input1.shape().to_vec(),rhs: input2.shape().to_vec()}); }
        
        let n1 = input1.square()?.sum_all()?.sqrt();
        let n2 = input2.square()?.sum_all()?.sqrt();

        let cosine_sim = input1.mul(input2)?.sum_all()? / (n1 * n2);

        let sim = T::one() - cosine_sim;
        let dissim = cosine_sim.sub(margin).max(T::zero());

        let target_data = target.to_vec()?;
        let mut total_loss = T::zero();
        for &t in &target_data {
            if t > T::zero() { total_loss += sim;}
            else { total_loss += dissim;}
        }
        Ok(total_loss / T::from_f64(target_data.len() as f64))
    }

    /// Poisson Negative Log Likelihood Loss
    /// loss = mean(pred - target * log(pred) + log(target!))
    pub fn PoissonNNL<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<T> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "poisson_nll",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        let eps = T::epsilon();
        let clamped = pred.clamp(eps, T::highest())?;
        clamped.sub(&target.mul(&clamped.ln()?)?)?.mean_all()
    }
}

pub mod Classific {
    use super::*;

    /// Accuracy
    /// acc = tp + tn / tot_pred
    pub fn Accuracy<T: Float>(pred: &Tensor<T>, target: &Tensor<i64>) -> TensorResult<f64> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "accuracy",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}) }
        let pred = pred.to_vec()?;
        let target = target.to_vec()?;
        Ok( pred.iter().zip(target.iter()).filter(|(p, t)| p.to_f64().round() as i64 == **t).count() as f64 / pred.len() as f64)
    }

    /// Confusion Matrix
    pub fn ConfusionMat<T: Float>(pred: &Tensor<T>,target: &Tensor<i64>,num_classes: usize) -> TensorResult<Tensor<i64>> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "confusion_matrix",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }

        let pred = pred.to_vec()?;
        let target = target.to_vec()?;
        let mut mat = vec![0i64; num_classes * num_classes];
        for (p, t) in pred.iter().zip(target.iter()) {
            let pred_idx = p.to_f64().round() as usize;
            let target_idx = *t as usize;
            if pred_idx < num_classes && target_idx < num_classes {
                mat[target_idx * num_classes + pred_idx] += 1;
            }
        }
        Tensor::from_vec(mat, &[num_classes, num_classes])
    }

    /// Precision
    /// prec = TP / (TP + FP)
    pub fn Precision<T: Float>(pred: &Tensor<T>,target: &Tensor<i64>,num_classes: usize) -> TensorResult<Vec<f64>> {
        let cm = ConfusionMat(pred, target, num_classes)?;
        let cm = cm.to_vec()?;
        let mut ret = Vec::with_capacity(num_classes);
        for class in 0..num_classes {
            let tp = cm[class * num_classes + class];
            let fp: i64 = (0..num_classes).filter(|&i| i != class).map(|i| cm[i * num_classes + class]).sum();
            ret.push( if tp + fp > 0 {tp as f64 / (tp + fp) as f64} else {0.0});
        }
        Ok(ret)
    }

    /// Recall (Sensitivity)
    /// recall = TP / (TP + FN)
    pub fn Recall<T: Float>(pred: &Tensor<T>,target: &Tensor<i64>,num_classes: usize) -> TensorResult<Vec<f64>> {
        let cm = ConfusionMat(pred, target, num_classes)?;
        let cm = cm.to_vec()?;
        let mut ret = Vec::with_capacity(num_classes);
        for class in 0..num_classes {
            let tp = cm[class * num_classes + class];
            let fn_count: i64 = (0..num_classes).filter(|&i| i != class).map(|i| cm[class * num_classes + i]).sum();
            ret.push( if tp + fn_count > 0 {tp as f64 / (tp + fn_count) as f64} else {0.0});
        }
        Ok(ret)
    }

    /// F1 Score
    /// F1 = 2 * (p * r) / (p + r)
    pub fn F1<T: Float>(pred: &Tensor<T>,target: &Tensor<i64>,num_classes: usize) -> TensorResult<Vec<f64>> {
        let pr = Precision(pred, target, num_classes)?;
        let rr = Recall(pred, target, num_classes)?;
        let mut ret = Vec::with_capacity(num_classes);
        for (p, r) in pr.iter().zip(rr.iter()) {
            ret.push( if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 });
        }
        Ok(ret)
    }

    /// Macro F1 Score (avg of f1 scores across all classes)
    pub fn MacroF1<T: Float>(pred: &Tensor<T>,target: &Tensor<i64>,num_classes: usize) -> TensorResult<f64> {
        let x = F1(pred, target, num_classes)?;
        Ok(x.iter().sum::<f64>() / x.len() as f64)
    }

    /// Micro F1 Score (global f1 using total TP, FP, FN)
    pub fn MicroF1<T: Float>(pred: &Tensor<T>,target: &Tensor<i64>,num_classes: usize) -> TensorResult<f64> {
        let cm = ConfusionMat(pred, target, num_classes)?;
        let cm = cm.to_vec()?;
        let mut t_tp = 0i64;
        let mut t_fp = 0i64;
        let mut t_fn = 0i64;
        for class in 0..num_classes {
            t_tp += cm[class * num_classes + class];
            t_fp += (0..num_classes).filter(|&i| i != class).map(|i| cm[i * num_classes + class]).sum::<i64>();
            t_fn += (0..num_classes).filter(|&i| i != class).map(|i| cm[class * num_classes + i]).sum::<i64>();
        }
        let p = if t_tp + t_fp > 0 { t_tp as f64 / (t_tp + t_fp) as f64 } else { 0.0 };
        let r = if t_tp + t_fn > 0 { t_tp as f64 / (t_tp + t_fn) as f64 } else { 0.0};
        if p + r > 0.0 { Ok(2.0 * p * r / (p + r)) } else { Ok(0.0) }
    }

    /// Specificity (True Negative Rate)
    /// specificity = TN / (TN + FP)
    pub fn Specificity<T: Float>(pred: &Tensor<T>,target: &Tensor<i64>,num_classes: usize) -> TensorResult<Vec<f64>> {
        let cm =  ConfusionMat(pred, target, num_classes)?;
        let cm = cm.to_vec()?;
        let mut ret = Vec::with_capacity(num_classes);
        for class in 0..num_classes {
            let tn: i64 = (0..num_classes).flat_map(|i| (0..num_classes).map(move |j| (i, j))).filter(|&(i, j)| i != class && j != class).map(|(i, j)| cm[i * num_classes + j]).sum();
            let fp: i64 = (0..num_classes).filter(|&i| i != class).map(|i| cm[i * num_classes + class]).sum();
            ret.push( if tn + fp > 0 { tn as f64 / (tn + fp) as f64 } else { 0.0 });
        }
        Ok(ret)
    }

    /// Area Under ROC Curve (AUC) - simplified implementation
    // Used ai for this
    pub fn Auc_Roc<T: Float>(pred: &Tensor<T>,target: &Tensor<i64>) -> TensorResult<f64> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "auc_roc",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        let pred = pred.to_vec()?;
        let target = target.to_vec()?;

        // Sort by predicted scores
        let mut indexed: Vec<(f64, i64)> = pred.iter().zip(target.iter()).map(|(p, t)| (p.to_f64(), *t)).collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut tpr = 0.0;
        let mut fpr = 0.0;
        let mut auc = 0.0;
        let mut prev_fpr = 0.0;
        let mut prev_tpr = 0.0;
        let total_pos = target.iter().filter(|&t| *t == 1).count() as f64;
        let total_neg = target.iter().filter(|&t| *t == 0).count() as f64;

        if total_pos == 0.0 || total_neg == 0.0 { return Ok(0.0); }

        for (_, label) in indexed {
            if label == 1 {
                tpr += 1.0 / total_pos;
            } else {
                fpr += 1.0 / total_neg;
                // Trapezoidal rule
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
                prev_fpr = fpr;
                prev_tpr = tpr;
            }
        }
        Ok(auc)
    }
}

/// Regression Metrics Module
pub mod regression {
    use super::*;

    /// R squared
    /// R2 = 1 - (SS_res / SS_tot)
    /// SS_res = sum((y_true - y_pred)^2)
    /// SS_tot = sum((y_true - y_mean)^2)
    pub fn R2_score<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<f64> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch { op: "r2_score", lhs: pred.shape().to_vec(), rhs: target.shape().to_vec()}); }
        let ss_res = pred.sub(target)?.square()?.sum_all()?.to_f64();
        let t_mean = target.mean_all()?.to_f64();
        let ss_tot = target.sub_scalar(T::from_f64(t_mean))?.square()?.sum_all()?.to_f64();
        if ss_tot == 0.0 {
            if ss_res == 0.0 { return Ok(1.0); }
            return Ok(0.0);
        }
        Ok(1.0 - ss_res / ss_tot)
    }

    /// Adjusted R squared
    /// AdjR2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    /// n: No of samples, p: No of independent varibs 
    pub fn AdjR2_score<T: Float>(pred: &Tensor<T>,target: &Tensor<T>,n_iv: usize) -> TensorResult<f64> {
        let n = pred.numel();
        if n < n_iv + 1 { return Ok(0.0) }
        Ok(1.0 - (1.0 - R2_score(pred, target)?) * ((n - 1) as f64 / (n - n_iv - 1) as f64))
    }

    /// Mean Absolute Percentage Error
    /// MAPE = mean(|(target - pred) / target|) * 100
    pub fn MAPE<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<f64> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "mape",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        let target = target.to_vec()?;
        let pred = pred.to_vec()?;
        
        let mut sm = 0.0;
        let mut  cnt = 0;
        for (t, p) in target.iter().zip(pred.iter()) {
            if t.to_f64().abs() > f64::EPSILON {
                sm += ((t.to_f64() - p.to_f64()) / t.to_f64()).abs();
                cnt += 1;
            }
        }
        if cnt == 0 { return Ok(0.0); }else{Ok((sm / cnt as f64) * 100.0)}
    }

    /// Mean Squared Logarithmic Error
    /// MSLE = mean((log(target + 1) - log(pred + 1))^2)
    pub fn MSLE<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<f64> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "msle",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        let log_pred = pred.add_scalar(T::one())?.ln()?;
        let log_target = target.add_scalar(T::one())?.ln()?;
        log_pred.sub(&log_target)?.square()?.mean_all().map(|t| t.to_f64())
    }

    pub fn MSE<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<f64> { super::Loss::MSE(pred, target).map(|t| t.to_f64()) }

    /// Mean Absolute Error (for reference)
    pub fn MAE<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<f64> { super::Loss::MAE(pred, target).map(|t| t.to_f64()) }

    /// Root Mean Squared Error (for reference)
    pub fn RMSE<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<f64> { super::Loss::RMSE(pred, target).map(|t| t.to_f64()) }

    /// Explained Variance 
    ///  1 - Var(y - y_pred) / Var(y)
    pub fn explained_variance<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<f64> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "explained_variance",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        let res = target.sub(pred)?;
        let var_res = res.var_all(true)?.to_f64().max(0.0);
        let var_target = target.var_all(true)?.to_f64();
        if var_target == 0.0 { return Ok(1.0); } else{Ok(1.0 - var_res / var_target)}
    }

    /// Max Absolute Error
    pub fn MaxAbsErr<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<f64> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "max_error",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        pred.sub(target)?.abs()?.max_all().map(|t| t.to_f64())
    }

    /// Median Absolute Error
    pub fn MedianAbsErr<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> TensorResult<f64> {
        if pred.shape() != target.shape() { return Err(TensorError::ShapeMismatch {op: "median_absolute_error",lhs: pred.shape().to_vec(),rhs: target.shape().to_vec()}); }
        pred.sub(target)?.abs()?.median_all().map(|t| t.to_f64())
    }
}
