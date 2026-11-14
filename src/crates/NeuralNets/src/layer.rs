use Tensors::{autograd::Comp, dtype::Float, tensor::Tensor};
use Utilities::random::Random::Rng;
use crate::{common::{Activation, Initializer, Result, flatten, invalid, validate_activation, validate_feature_input}, layers::{basic::{BatchNorm1d, Dense, Dropout, Identity, LayerNorm, PRelu, RMSNorm, ResidualBlock, Unflatten}, cnn::{AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, AvgPool1d, AvgPool2d, Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, MaxPool1d, MaxPool2d, validate_conv1d_input}, rnn::{Gru, Lstm}}};

/// Layer Wrapper using for NN
#[derive(Clone,Debug)]
pub enum Layer<T: Float> {
    Dense(Dense<T>),
    Conv1d(Conv1d<T>),
    Conv2d(Conv2d<T>),
    ConvTranspose1d(ConvTranspose1d<T>),
    ConvTranspose2d(ConvTranspose2d<T>),
    Dropout(Dropout),
    Identity(Identity),
    Unflatten(Unflatten),
    LayerNorm(LayerNorm<T>),
    RMSNorm(RMSNorm<T>),
    BatchNorm1d(BatchNorm1d<T>),
    PRelu(PRelu<T>),
    Residual(Box<ResidualBlock<T>>),
    Activation(Activation),
    Flatten,
    MaxPool2d(MaxPool2d),
    AvgPool2d(AvgPool2d),
    MaxPool1d(MaxPool1d),
    AvgPool1d(AvgPool1d),
    AdaptiveMaxPool1d(AdaptiveMaxPool1d),
    AdaptiveAvgPool1d(AdaptiveAvgPool1d),
    AdaptiveMaxPool2d(AdaptiveMaxPool2d),
    AdaptiveAvgPool2d(AdaptiveAvgPool2d),
    GlobalAvgPool1d,
    GlobalMaxPool1d,
    GlobalAvgPool2d,
    GlobalMaxPool2d,
    Lstm(Lstm<T>),
    Gru(Gru<T>),
}

impl<T: Float> Layer<T> {

    /// Get the tensor params from the Layer
    pub fn params(&self) -> Vec<Tensor<T>> {
        match self {
            Self::Dense(x) => vec![x.weights.clone(), x.bias.clone()],
            Self::Conv1d(x) => vec![x.weights.clone(), x.bias.clone()],
            Self::Conv2d(x) => vec![x.weights.clone(), x.bias.clone()],
            Self::ConvTranspose1d(x) => vec![x.weights.clone(), x.bias.clone()],
            Self::ConvTranspose2d(x) => vec![x.weights.clone(), x.bias.clone()],
            Self::LayerNorm(x) => vec![x.weight.clone(), x.bias.clone()],
            Self::RMSNorm(x) => vec![x.weight.clone()],
            Self::BatchNorm1d(x) => vec![x.weight.clone(), x.bias.clone()],
            Self::PRelu(x) => vec![x.weight.clone()],
            Self::Residual(x) => x.layers.iter().flat_map(Layer::params).collect(),
            Self::Lstm(x) => vec![x.w_ih.clone(), x.w_hh.clone(), x.b_ih.clone(), x.b_hh.clone()],
            Self::Gru(x) => vec![x.w_ih.clone(), x.w_hh.clone(), x.b_ih.clone(), x.b_hh.clone()],
            _ => Vec::new(),
        }
    }

    /// Get the total no of trainable params in the entire Layered Network
    pub fn param_count(&self) -> usize{
        let mut cnt = 0;
        self._param_cnt_helper(&mut cnt); 
        cnt
    }

    fn _param_cnt_helper(&self,cnt:&mut usize) {
        match self{
            Layer::AvgPool1d(x) => *cnt += x.param_count(),
            Layer::AvgPool2d(x) => *cnt += x.param_count(),
            Layer::BatchNorm1d(x) => *cnt += x.param_count(),
            Layer::Conv1d(x) => *cnt += x.param_count(),
            Layer::Conv2d(x) => *cnt += x.param_count(),
            Layer::Dense(x) => *cnt += x.param_count(),
            Layer::Dropout(x) => *cnt += x.param_count(),
            Layer::Activation(_) | Layer::Flatten | Layer::Identity(_) | Layer::Unflatten(_) | Layer::AdaptiveAvgPool1d(_) | Layer::AdaptiveAvgPool2d(_) | Layer::AdaptiveMaxPool1d(_) | Layer::AdaptiveMaxPool2d(_) | Layer::GlobalAvgPool1d | Layer::GlobalAvgPool2d | Layer::GlobalMaxPool1d | Layer::GlobalMaxPool2d | Layer::MaxPool1d(_) | Layer::MaxPool2d(_)=> *cnt += 0,
            Layer::LayerNorm(x) => *cnt += x.param_count(),
            Layer::PRelu(x) => *cnt += x.param_count(),
            Layer::Residual(x) => x.layers.iter().for_each(|x| x._param_cnt_helper(cnt)),
            Layer::ConvTranspose1d(x) => *cnt += x.param_count(),
            Layer::ConvTranspose2d(x) => *cnt += x.param_count(),
            Layer::Lstm(x) => *cnt += x.param_count(),
            Layer::Gru(x) => *cnt += x.param_count(),
            Layer::RMSNorm(x) => *cnt += x.param_count(),
        };
    }

    /// Set the params in the Layer Network
    pub fn set_params(&mut self, params: &[Tensor<T>]) -> Result<()> {
        match self {
            Self::Dense(layer) => {
                if params.len() != 2 { return Err(invalid("Dense layer requires weight and bias tensors")); }
                layer.set_parameters(params)
            }
            Self::Conv1d(layer) => layer.set_parameters(params),
            Self::Conv2d(layer) => layer.set_parameters(params),
            Self::ConvTranspose1d(layer) => layer.set_parameters(params),
            Self::ConvTranspose2d(layer) => layer.set_parameters(params),
            Self::Lstm(layer) => layer.set_parameters(params),
            Self::Gru(layer) => layer.set_parameters(params),
            Self::LayerNorm(layer) => set_wt_b(&mut layer.weight, &mut layer.bias, params),
            Self::RMSNorm(layer) => {
                if params.len() != 1 || params[0].shape() != layer.weight.shape() { return Err(invalid("RMS norm params shape mismatch")); }
                if params[0].device() != layer.weight.device() { return Err(invalid("RMSnorm layer requires weight on same device")); }
                layer.weight = params[0].clone();
                Ok(())
            },
            Self::BatchNorm1d(layer) => set_wt_b(&mut layer.weight, &mut layer.bias, params),
            Self::PRelu(layer) => {
                if params.len() != 1 || params[0].shape() != layer.weight.shape() { return Err(invalid("Prelu shape mismatch")); }
                if params[0].device() != layer.weight.device() { return Err(invalid("prelu layer requires weight on same device")); }
                layer.weight = params[0].clone();
                Ok(())
            },
            Self::Residual(block) => {
                // Get the total size of tensor coming into the layer from all children
                let expected: usize = block.layers.iter().map(|child| child.params().len()).sum();
                if params.len() != expected { return Err(invalid("Residual param count mismatch"));}
                let mut st = 0;
                for child in &mut block.layers {
                    let l = child.params().len();
                    if l > 0 {
                        child.set_params(&params[st..st + l])?;
                        st += l;
                    }
                }
                Ok(())
            },
            _ if params.is_empty() => Ok(()),
            _ => Err(invalid("stateless layer cannot accept parameters")),
        }
    }

    /// The output size of Network 
    pub fn output_size(&self, input_features: Option<usize>) -> Option<usize> {
        match self {
            Self::Dense(x) => Some(x.output_size()),
            Self::Conv1d(x) => Some(x.out_channels),
            Self::Conv2d(x) => Some(x.out_channels),
            Self::ConvTranspose1d(x) => Some(x.out_channels),
            Self::ConvTranspose2d(x) => Some(x.out_channels),
            Self::Lstm(x) => Some(x.hidden_sz),
            Self::Gru(x) => Some(x.hidden_sz),
            Self::LayerNorm(x) => Some(x.features),
            Self::RMSNorm(x) => Some(x.features),
            Self::BatchNorm1d(x) => Some(x.features),
            Self::PRelu(x) => Some(x.features),
            Self::Flatten => None,
            _ => input_features,
        }
    }

    pub fn input_size(&self) -> Option<usize>{
        match self {
            Self::Dense(x) => Some(x.input_size()),
            Self::Conv1d(x) => Some(x.in_channels),
            Self::Conv2d(x) => Some(x.in_channels),
            Self::ConvTranspose1d(x) => Some(x.in_channels),
            Self::ConvTranspose2d(x) => Some(x.in_channels),
            Self::Lstm(x) => Some(x.input_sz),
            Self::Gru(x) => Some(x.input_sz),
            Self::LayerNorm(x) => Some(x.input_size()),
            Self::RMSNorm(x) => Some(x.features),
            Self::BatchNorm1d(x) => Some(x.input_size()),
            Self::PRelu(x) => Some(x.features),
            Self::Flatten | Self::Dropout(_) | Self::Identity(_) | Self::Activation(_) | Self::MaxPool1d(_) | Self::MaxPool2d(_) | Self::AvgPool1d(_) | Self::AvgPool2d(_) | Self::AdaptiveAvgPool1d(_) | Self::AdaptiveAvgPool2d(_) | Self::AdaptiveMaxPool1d(_) | Self::AdaptiveMaxPool2d(_) | Self::GlobalAvgPool1d | Self::GlobalAvgPool2d | Self::GlobalMaxPool1d | Self::GlobalMaxPool2d | Self::Unflatten(_) => None,
            Self::Residual(x) => x.layers.iter().find_map(Layer::input_size)
        }
    }

    /// Fwd pass
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        match self {
            Self::Dense(layer) => layer.forward(input),
            Self::Conv1d(layer) => layer.forward(input),
            Self::Conv2d(layer) => layer.forward(input),
            Self::ConvTranspose1d(layer) => layer.forward(input),
            Self::ConvTranspose2d(layer) => layer.forward(input),
            Self::Lstm(layer) => layer.forward(input),
            Self::Gru(layer) => layer.forward(input),
            Self::Identity(_) => Ok(input.clone()),
            Self::Unflatten(layer) => unflatten(input,layer),
            Self::Dropout(_) => Ok(input.clone()),
            Self::BatchNorm1d(layer) => {
                if input.ndim() != 2 { return Err(invalid("Batch norm expects [batch, features] input")); }
                validate_feature_input(input, layer.features, "batch norm")?;
                // ((x - mu)/sigma) * w + b
                input.sub(&layer.running_mean)?.div(&layer.running_var.add_scalar(layer.epsilon)?.sqrt()?)?.mul(&layer.weight)?.add(&layer.bias).map_err(Into::into)
            },
            Self::LayerNorm(layer) => {
                validate_feature_input(input, layer.features, "layer norm")?;
                // ((x - mu)/sigma) * w + b
                input.layer_norm(layer.epsilon)?.mul(&layer.weight)?.add(&layer.bias).map_err(Into::into)
            },
            Self::RMSNorm(layer) => {
                validate_feature_input(input,layer.features,"RMS norm")?;
                //  w * (x / ( sqrt(mean(x^2) + eps)))
                input.div(&input.square()?.mean(-1,true)?.add_scalar(layer.epsilon)?.sqrt()?)?.mul(&layer.weight).map_err(Into::into)
            },
            Self::Activation(activation) => activate_tensor(input, *activation),
            Self::PRelu(layer) => {
                validate_feature_input(input, layer.features, "PReLU")?;
                // input > 0: input else w * input
                input.gt(&input.zeros_like())?.where_cond(input, &input.mul(&layer.weight)?)
            },
            Self::Residual(block) => {
                let mut output = input.clone();
                for child in &block.layers { output = child.forward(&output)?; }
                if output.shape() != input.shape() { return Err(invalid("residual block must preserve the input shape")); }
                input.add(&output).map_err(Into::into)
            }
            Self::Flatten => flatten(input),
            Self::MaxPool2d(layer) => Ok(input.max_pool2d_with_indices(layer.kernel, layer.stride, layer.padding)?.0),
            Self::AvgPool2d(layer) => input.avg_pool2d(layer.kernel,layer.stride,layer.padding,layer.count_pad).map_err(Into::into),
            Self::MaxPool1d(layer) => {
                if input.ndim() != 3 { return Err(invalid("MaxPool1d expects [batch, channels, length] input")); }
                Ok(input.unsqueeze(2)?.max_pool2d_with_indices((1, layer.kernel_size),(1, layer.stride),(0, layer.padding),)?.0.squeeze(2)?)
            }
            Self::AvgPool1d(layer) => {
                if input.ndim() != 3 { return Err(invalid("AvgPool1d expects [batch, channels, length] input")); }
                input.unsqueeze(2)?.avg_pool2d((1, layer.kernel_size),(1, layer.stride),(0, layer.padding),layer.count_pad,)?.squeeze(2).map_err(Into::into)
            }
            Self::AdaptiveMaxPool1d(layer) => adaptive_pool1d(input,layer.output_size,true),
            Self::AdaptiveAvgPool1d(layer) => adaptive_pool1d(input,layer.output_size,false),
            Self::AdaptiveMaxPool2d(layer) => adaptive_pool2d(input,layer.output_size,true),
            Self::AdaptiveAvgPool2d(layer) => adaptive_pool2d(input,layer.output_size,false),
            Self::GlobalAvgPool2d => {
                if input.ndim() != 4 {return Err(invalid("GlobalAvgPool2D expects NCHW 4 rank input"));}
                input.mean(3, true)?.mean(2, true).map_err(Into::into)
            }
            Self::GlobalMaxPool2d => {
                if input.ndim() != 4 {
                    return Err(invalid("GlobalMaxPool2D expects NCHW 4 rank input"));
                }
                input.max(3, true)?.max(2, true).map_err(Into::into)
            }
            Self::GlobalAvgPool1d => {
                if input.ndim() != 3 {
                    return Err(invalid("GlobalAvgPool1D expects [batch, channels, length] input"));
                }
                input.mean(2, true).map_err(Into::into)
            }
            Self::GlobalMaxPool1d => {
                if input.ndim() != 3 {
                    return Err(invalid("GlobalMaxPool1D expects [batch, channels, length] input"));
                }
                input.max(2, true).map_err(Into::into)
            }
        }
    }

    /// Build the autograd graph 
    pub(crate) fn forward_graph(&mut self,input: &Comp<T>,training: bool,rng: &mut Rng) -> Result<(Comp<T>, Vec<Comp<T>>)> {
        match self {
            Self::Dense(layer) => {
                let w = Comp::leaf(layer.weights.clone());
                let b = Comp::leaf(layer.bias.clone());
                Ok(( apply_activation(&input.matmul(&w)?.add(&b)?, layer.activation)?, vec![w, b]))
            },
            Self::Conv1d(layer) => {
                let x = input.data();
                validate_conv1d_input(&x, layer.in_channels)?;
                let x2d = input.unsqueeze(2)?;
                let w = Comp::leaf(layer.weights.clone());
                let b = Comp::leaf(layer.bias.clone());
                let w2d = w.reshape(&[layer.out_channels as isize,layer.in_channels as isize,1,layer.kernel_size as isize])?;
                let b2d = b.reshape(&[1, layer.out_channels as isize, 1, 1])?;
                let x_data = x2d.data();
                let w_data = w2d.data();
                let (stride, padding) = (layer.stride, layer.padding);
                let output = x_data.conv2d(&w_data,Some(&b2d.data()),(1,stride),(0, padding))?;
                let conv = Comp::from_op(output, vec![x2d.clone(), w2d, b2d], move |g| {
                    let (dx, dw, db) = x_data.conv2d_backward(&w_data, g, (1, stride), (0, padding))?;
                    Ok(vec![Some(dx), Some(dw), Some(db)])
                });
                Ok((apply_activation(& conv.squeeze(2)?, layer.activation)?, vec![w, b]))
            },
            Self::Conv2d(layer) => {
                let x = input.data();
                if x.ndim() != 4 || x.shape()[1] != layer.in_channels { return Err(invalid(format!("Conv2d expects [batch, {}, height, width], got {:?}",layer.in_channels,x.shape()))); }
                let w = Comp::leaf(layer.weights.clone());
                let b = Comp::leaf(layer.bias.clone());
                let w_data = w.data();
                let output = x.conv2d(&w_data, Some(&b.data()), layer.stride, layer.padding)?;
                let x = x;
                let weights_saved = w_data.clone();
                let (stride,padding) = (layer.stride,layer.padding);
                let conv = Comp::from_op(output,vec![input.clone(), w.clone(), b.clone()],move |g| {
                        let (dx, dw, db) = x.conv2d_backward(&weights_saved, g, stride, padding)?;
                        Ok(vec![Some(dx), Some(dw), Some(db)])
                    },
                );
                Ok((apply_activation(&conv, layer.activation)?, vec![w, b]))
            },
            Self::ConvTranspose2d(layer) => {
                let x = input.data();
                if x.ndim() != 4 || x.shape()[1] != layer.in_channels { return Err(invalid(format!("ConvTranspose2d expects [batch, {}, height, width], got {:?}",layer.in_channels,x.shape()))); }
                let w = Comp::leaf(layer.weights.clone());
                let b = Comp::leaf(layer.bias.clone());
                let x_data = x.clone();
                let w_data = w.data();
                let grad = x_data.conv_transpose2d(&w_data,layer.stride,layer.padding)?.add(&b.data())?;
                let (stride,padding) = (layer.stride,layer.padding);
                let conv = Comp::from_op(grad,vec![input.clone(),w.clone(),b.clone()],move |g| {
                    let (dx,dw,db) = x_data.conv_transpose2d_backward(&w_data,g,stride,padding)?;
                    Ok(vec![Some(dx),Some(dw),Some(db)])
                });
                Ok((apply_activation(&conv,layer.activation)?,vec![w,b]))
            },
            Self::ConvTranspose1d(layer) => {
                let x = input.data();
                validate_conv1d_input(&x,layer.in_channels)?;
                let x2d = input.unsqueeze(2)?;
                let w = Comp::leaf(layer.weights.clone());
                let b = Comp::leaf(layer.bias.clone());
                let w2d = w.unsqueeze(2)?;
                let b2d = b.reshape(&[1,layer.out_channels as isize,1,1])?;
                let x_data = x2d.data();
                let w_data = w2d.data();
                let grad = x_data.conv_transpose2d(&w_data,(1,layer.stride),(0,layer.padding))?.add(&b2d.data())?;
                let (stride,padding) = ((1,layer.stride),(0,layer.padding));
                let conv = Comp::from_op(grad,vec![x2d.clone(),w2d.clone(),b2d.clone()],move |g| {
                    let (dx,dw,db) = x_data.conv_transpose2d_backward(&w_data,g,stride,padding)?;
                    Ok(vec![Some(dx),Some(dw),Some(db)])
                });
                Ok((apply_activation(&conv.squeeze(2)?,layer.activation)?,vec![w,b]))
            },
            Self::Lstm(layer) => layer.forward_graph(input),
            Self::Gru(layer) => layer.forward_graph(input),
            Self::Dropout(layer) => {
                if !training || layer.probability == 0.0 { return Ok((input.clone(), Vec::new())); }
                let data = input.data();
                let random = Tensor::<T>::rand(data.shape(), rng).to_device(data.device())?;
                // We multiply wiht 1/(1-p) to preeserve the mean, suggested by AI  
                let mask = random.gt(&Tensor::scalar(T::from_f64(layer.probability)).to_device(data.device())?)?.mul_scalar(T::from_f64(1.0 / (1.0 - layer.probability)))?;
                Ok((input.mul(&Comp::constant(mask))?, Vec::new()))
            },
            Self::LayerNorm(layer) => {
                let x = input.data();
                validate_feature_input(&x, layer.features, "layer norm")?;
                let w = Comp::leaf(layer.weight.clone());
                let b = Comp::leaf(layer.bias.clone());
                Ok(( input.layer_norm(layer.epsilon)?.mul(&w)?.add(&b)?, vec![w, b]))
            },
            Self::RMSNorm(layer) => {
                validate_feature_input(&input.data(),layer.features,"RMS norm")?;
                let w = Comp::leaf(layer.weight.clone());
                Ok((input.div(&input.square()?.mean(-1,true)?.add_scalar(layer.epsilon)?.sqrt()?)?.mul(&w)?,vec![w]))
            },
            Self::BatchNorm1d(layer) => {
                let x = input.data();
                if x.ndim() != 2 { return Err(invalid("Batch norm expects [batch, features] input")); }
                validate_feature_input(&x, layer.features, "batch norm")?;
                let w = Comp::leaf(layer.weight.clone());
                let b = Comp::leaf(layer.bias.clone());
                let z= if training {
                    let mean = input.mean(0, true)?;
                    let var = input.var(0, true, false)?;
                    // running = (1 - momentum)* running + momrntum * cur
                    layer.running_mean = layer.running_mean.mul_scalar(T::one() - layer.momentum)?.add(&mean.data().mul_scalar(layer.momentum)?)?;
                    layer.running_var = layer.running_var.mul_scalar(T::one() - layer.momentum)?.add(&var.data().mul_scalar(layer.momentum)?)?;
                    input.sub(&mean)?.div(&var.add_scalar(layer.epsilon)?.sqrt()?)?
                } else {
                    input.sub(&Comp::constant(layer.running_mean.clone()))?.div(&Comp::constant(layer.running_var.add_scalar(layer.epsilon)?.sqrt()?))?
                };
                Ok(( z.mul(&w)?.add(&b)?, vec![w, b]))
            },
            Self::PRelu(layer) => {
                validate_feature_input(&input.data(), layer.features, "PReLU")?;
                let w = Comp::leaf(layer.weight.clone());
                Ok((input.prelu(&w)?, vec![w]))
            },
            Self::Residual(block) => {
                let mut out = input.clone();
                let mut params = Vec::new();
                for c in &mut block.layers {
                    let (grad, child_params) = c.forward_graph(&out, training, rng)?;
                    out = grad;
                    params.extend(child_params);
                }
                if out.shape() != input.shape() { return Err(invalid("residual block has to have correct input shape[FATAL]")); }
                Ok((input.add(&out)?, params))
            },
            Self::MaxPool2d(layer) => {
                let x = input.data();
                let (grad, idxs) = x.max_pool2d_with_indices(layer.kernel, layer.stride, layer.padding)?;
                let x_data = x.clone();
                let idxs_data = idxs.clone();
                let (kernel, stride, padding) = (layer.kernel, layer.stride, layer.padding);
                let grad = Comp::from_op(grad, vec![input.clone()], move |g| {
                    Ok(vec![Some(x_data.max_pool2d_backward(&idxs_data,g,kernel,stride,padding)?)])
                });
                Ok((grad, Vec::new()))
            },
            Self::AvgPool2d(layer) => {
                let x = input.data();
                let grad = x.avg_pool2d(layer.kernel,layer.stride,layer.padding,layer.count_pad)?;
                let x_data = x.clone();
                let (kernel, stride, padding, count_include_pad) = (layer.kernel,layer.stride,layer.padding,layer.count_pad);
                let grad = Comp::from_op(grad, vec![input.clone()], move |g| {
                    Ok(vec![Some(x_data.avg_pool2d_backward(g,kernel,stride,padding,count_include_pad)?)])
                });
                Ok((grad, Vec::new()))
            },
            Self::MaxPool1d(layer) => {
                let x = input.data();
                if x.ndim() != 3 { return Err(invalid("MaxPool1d expects [batch, channels, length]")); }
                let x2d = input.unsqueeze(2)?;
                let x_data = x2d.data();
                let (grad, idxs) = x_data.max_pool2d_with_indices((1, layer.kernel_size),(1, layer.stride),(0, layer.padding))?;
                let idxs_data = idxs.clone();
                let (kernel, stride, padding) = ((1, layer.kernel_size),(1, layer.stride),(0, layer.padding));
                let ret = Comp::from_op(grad, vec![x2d], move |g| {
                    Ok(vec![Some(x_data.max_pool2d_backward(&idxs_data,g,kernel,stride,padding)?)])
                });
                Ok((ret.squeeze(2)?, Vec::new()))
            },
            Self::AvgPool1d(layer) => {
                let x = input.data();
                if x.ndim() != 3 { return Err(invalid("AvgPool1d expects [batch, channels, length]")); }
                let x2d = input.unsqueeze(2)?;
                let x_saved = x2d.data();
                let (kernel, stride, padding, include_pad) = ((1, layer.kernel_size),(1, layer.stride),(0, layer.padding),layer.count_pad);
                let grad = x_saved.avg_pool2d(kernel, stride, padding, include_pad)?;
                let grad = Comp::from_op(grad, vec![x2d], move |g| {
                    Ok(vec![Some(x_saved.avg_pool2d_backward(g,kernel,stride,padding,include_pad)?)])
                });
                Ok((grad.squeeze(2)?, Vec::new()))
            },
            Self::Activation(activation) => Ok((apply_activation(input, *activation)?, Vec::new())),
            Self::Flatten => Ok((input.reshape(&[input.shape()[0] as isize, -1])?, Vec::new())),
            Self::Identity(_) => Ok((input.clone(),Vec::new())),
            Self::Unflatten(layer) => Ok((unflatten_comp(input,layer)?,Vec::new())),
            Self::GlobalAvgPool2d => {
                if input.ndim() != 4 { return Err(invalid("GlobalAvgPool2D expects NCHW 4 rank input")); }
                Ok((input.mean(3, true)?.mean(2, true)?, Vec::new()))
            },
            Self::GlobalMaxPool2d => {
                if input.ndim() != 4 { return Err(invalid("GlobalMaxPool2D expects NCHW 4 rank input")); }
                Ok((input.max(3, true)?.max(2, true)?, Vec::new()))
            },
            Self::GlobalAvgPool1d => {
                if input.ndim() != 3 { return Err(invalid("GlobalAvgPool1D expects [batch, channels, length]")); }
                Ok((input.mean(2, true)?, Vec::new()))
            },
            Self::AdaptiveMaxPool1d(layer) => Ok((adaptive_pool1d_comp(input,layer.output_size,true)?,Vec::new())),
            Self::AdaptiveAvgPool1d(layer) => Ok((adaptive_pool1d_comp(input,layer.output_size,false)?,Vec::new())),
            Self::AdaptiveMaxPool2d(layer) => Ok((adaptive_pool2d_comp(input,layer.output_size,true)?,Vec::new())),
            Self::AdaptiveAvgPool2d(layer) => Ok((adaptive_pool2d_comp(input,layer.output_size,false)?,Vec::new())),
            Self::GlobalMaxPool1d => {
                if input.ndim() != 3 { return Err(invalid("GlobalMaxPoold1D expects [batch, channels, length]")) }
                Ok((input.max(2, true)?, Vec::new()))
            },
        }
    }
}

/// Initialise a tensor with the type of init or fin/fout params
pub fn initialize<T: Float>(shape: &[usize],fan_in: usize,fan_out: usize,initializer: Initializer,rng: &mut Rng) -> Result<Tensor<T>> {
    if shape.iter().any(|&d| d == 0) { return Err(invalid("Initializer shape dims be positive")); }
    let fi = fan_in.max(1) as f64;
    let fo = fan_out.max(1) as f64;
    let (mut value, scale, shift) = match initializer {
        Initializer::Zeros => (Tensor::zeros(shape), 0.0, 0.0),
        Initializer::Ones => (Tensor::ones(shape), 0.0, 0.0),
        Initializer::Constant(c) => {
            if !c.is_finite() { return Err(invalid("Const initializer must be finite")); }
            (Tensor::zeros(shape), 0.0, c)
        },
        Initializer::Uniform { bound } => {
            if !bound.is_finite() || bound < 0.0 { return Err(invalid("Uniform initializer must be finite and positive")); }
            (Tensor::rand(shape, rng), 2.0 * bound, -bound)
        },
        Initializer::Normal => (Tensor::randn(shape, rng), 1.0 / fi.sqrt(), 0.0), // 1/sqrt(fan_in) Suggested by AI for stable initialisation
        Initializer::HeNormal | Initializer::KaimingNormal => (Tensor::randn(shape,rng), (2.0 / fi).sqrt(), 0.0),
        Initializer::He | Initializer::KaimingUniform | Initializer::Kaimang | Initializer::HeUniform => (Tensor::rand(shape, rng),2.0 * (6.0 / fi).sqrt(),-(6.0 / fi).sqrt()),
        Initializer::Xavier | Initializer::Glorot=> (Tensor::randn(shape, rng), (2.0 / (fi + fo)).sqrt(), 0.0),
        Initializer::XavierUniform | Initializer::GlorotUniform => (Tensor::rand(shape,rng),2.0 * (6.0 / (fi + fo)).sqrt(),-(6.0 / (fi + fo)).sqrt()),
        Initializer::LeCunNormal => (Tensor::randn(shape,rng), 1.0 / fi.sqrt(), 0.0),
        Initializer::LeCunUniform => (Tensor::rand(shape, rng), 2.0 * (3.0 / fi).sqrt(), -(3.0/fi).sqrt()),
    };
    if scale != 0.0 { value = value.mul_scalar(T::from_f64(scale))?; }
    if shift != 0.0 { value = value.add_scalar(T::from_f64(shift))?; }
    Ok(value)
}

pub fn set_wt_b<T: Float>(weight: &mut Tensor<T>,bias: &mut Tensor<T>,params: &[Tensor<T>]) -> Result<()> {
    if params.len() != 2 { return Err(invalid("Norm layer requires weight and bias tensors"));}
    if params[0].shape() != weight.shape() || params[1].shape() != bias.shape() {return Err(invalid("Norm param shape aren't same"));}
    params[0].same_device(&params[1])?;
    if params[0].device() != weight.device() { return Err(invalid("Norm params are on diff device compared to layer"));}
    *weight = params[0].clone();
    *bias = params[1].clone();
    Ok(())
}

fn unflatten<T: Float>(input: &Tensor<T>,layer: &Unflatten) -> Result<Tensor<T>> { Ok(input.reshape(&unflatten_shape(input.shape(),layer)?)?) }
fn unflatten_comp<T: Float>(input: &Comp<T>,layer: &Unflatten) -> Result<Comp<T>> { Ok(input.reshape(&unflatten_shape(&input.shape(),layer)?)?) }

fn unflatten_shape(shape: &[usize],layer: &Unflatten) -> Result<Vec<isize>> {
    let dim = if layer.dim < 0 { shape.len() as isize + layer.dim } else { layer.dim }; // Negative indexing
    if dim < 0 || dim >= shape.len() as isize { return Err(invalid("Unflatten dims is out of range")); }
    let size = layer.unflattened_size.iter().fold(1usize,|acc,&dim| acc * dim);
    if shape[dim as usize] != size { return Err(invalid("Unflatten sizes doesnt multiply to selected dim")); }
    let mut output = shape.iter().map(|&x| x as isize).collect::<Vec<_>>();
    output.splice(dim as usize..=dim as usize,layer.unflattened_size.iter().map(|&x| x as isize));
    Ok(output)
}

fn adaptive_pool1d<T: Float>(input: &Tensor<T>,output_size: usize,max: bool) -> Result<Tensor<T>> {
    if input.ndim() != 3 { return Err(invalid("AdaptivePool1d expects [batch, channels, length]")); }
    if output_size == 0 { return Err(invalid("AdaptivePool1d output size must be positive")); }
    let len = input.shape()[2];
    let mut outputs = Vec::with_capacity(output_size);
    for i in 0..output_size {
        let st = i * len / output_size;
        let ed = ((i + 1) * len + output_size - 1) / output_size;
        if ed <= st { return Err(invalid("Adaptive1d Poolling window is empty")); }
        let bin = input.narrow(2,st,ed-st)?;
        outputs.push(if max { bin.max(2,true)? } else { bin.mean(2,true)? });
    }
    Tensor::cat(& outputs.iter().collect::<Vec<_>>(),2).map_err(Into::into)
}

fn adaptive_pool2d<T: Float>(input: &Tensor<T>,output_size: (usize,usize),max: bool) -> Result<Tensor<T>> {
    if input.ndim() != 4 { return Err(invalid("AdaptivePool2d expects [batch, channels, height, width]")); }
    if output_size.0 == 0 || output_size.1 == 0 { return Err(invalid("AdaptivePool2d output dims must be positive")); }
    let (h,w) = (input.shape()[2],input.shape()[3]);
    let mut rows = Vec::with_capacity(output_size.0);
    for y in 0..output_size.0 {
        let y0 = y * h / output_size.0;
        let y1 = ((y + 1) * h + output_size.0 - 1) / output_size.0;
        if y1 <= y0 { return Err(invalid("Adaptive pool output size exceeds input size")); }
        let mut cols = Vec::with_capacity(output_size.1);
        for x in 0..output_size.1 {
            let x0 = (x * w) / output_size.1;
            let x1 = ((x + 1) * w + output_size.1 - 1) / output_size.1;
            if x1 <= x0 { return Err(invalid("Adaptive pool output size exceeds input size")); }
            let bin = input.narrow(2,y0,y1-y0)?.narrow(3,x0,x1-x0)?;
            cols.push(if max { bin.max(3,true)?.max(2,true)? } else { bin.mean(3,true)?.mean(2,true)? });
        }
        rows.push(Tensor::cat(&cols.iter().collect::<Vec<_>>(),3)?);
    }
    Tensor::cat(& rows.iter().collect::<Vec<_>>(),2).map_err(Into::into)
}

fn adaptive_pool1d_comp<T: Float>(input: &Comp<T>,output_size: usize,max: bool) -> Result<Comp<T>> {
    let shape = input.shape();
    if shape.len() != 3 { return Err(invalid("AdaptivePool1d expects [batch, channels, length]")); }
    if output_size == 0 { return Err(invalid("AdaptivePool1d output size must be positive")); }
    let mut outputs = Vec::with_capacity(output_size);
    for i in 0..output_size {
        let start = i * shape[2] / output_size;
        let end = ((i + 1) * shape[2] + output_size - 1) / output_size;
        if end <= start { return Err(invalid("Adaptive pool output size exceeds input size")); }
        let bin = input.narrow(2,start,end-start)?;
        outputs.push(if max { bin.max(2,true)? } else { bin.mean(2,true)? });
    }
    let refs = outputs.iter().collect::<Vec<_>>();
    Comp::cat(&refs,2).map_err(Into::into)
}

fn adaptive_pool2d_comp<T: Float>(input: &Comp<T>,output_size: (usize,usize),max: bool) -> Result<Comp<T>> {
    let shape = input.shape();
    if shape.len() != 4 { return Err(invalid("AdaptivePool2d expects [batch, channels, height, width]")); }
    if output_size.0 == 0 || output_size.1 == 0 { return Err(invalid("Adaptive pool output dims must be positive")); }
    let mut rows = Vec::with_capacity(output_size.0);
    for y in 0..output_size.0 {
        let y0 = y * shape[2] / output_size.0;
        let y1 = ((y + 1) * shape[2] + output_size.0 - 1) / output_size.0;
        if y1 <= y0 { return Err(invalid("Adaptive pool output size exceeds input size")); }
        let mut cols = Vec::with_capacity(output_size.1);
        for x in 0..output_size.1 {
            let x0 = x * shape[3] / output_size.1;
            let x1 = ((x + 1) * shape[3] + output_size.1 - 1) / output_size.1;
            if x1 <= x0 { return Err(invalid("Adaptive pool output size exceeds input size")); }
            let bin = input.narrow(2,y0,y1-y0)?.narrow(3,x0,x1-x0)?;
            cols.push(if max { bin.max(3,true)?.max(2,true)? } else { bin.mean(3,true)?.mean(2,true)? });
        }
        rows.push(Comp::cat(&cols.iter().collect::<Vec<_>>(),3)?);
    }
    Comp::cat(&rows.iter().collect::<Vec<_>>(),2).map_err(Into::into)
}



// Do forward pass
pub(crate) fn activate_tensor<T: Float>(x: &Tensor<T>,activation: Activation) -> Result<Tensor<T>> {
    validate_activation(activation)?;
    Ok(match activation {
        Activation::None | Activation::Linear => x.clone(),
        Activation::Relu => x.relu()?,
        Activation::LeakyRelu(a) => x.lrelu(T::from_f64(a))?,
        Activation::Sigmoid => x.sigmoid()?,
        Activation::Tanh => x.tanh()?,
        Activation::Gelu => x.gelu()?,
        Activation::Elu(a) => x.elu(T::from_f64(a))?,
        Activation::Selu => x.selu(T::from_f64(1.0507009873554805),T::from_f64(1.6732632423543772))?,
        Activation::SiLu => x.silu()?,
        Activation::Softplus => x.softplus()?,
        Activation::Mish => x.mish()?,
        Activation::Softsign => x.softsign()?,
        Activation::HardSigmoid => x.hardsigmoid()?,
        Activation::HardTanh { min, max } => x.hardtanh(T::from_f64(min), T::from_f64(max))?,
        Activation::Softmax => x.softmax(-1)?,
    })
}

// Do backward pass
pub(crate) fn apply_activation<T: Float>(x: &Comp<T>, activation: Activation) -> Result<Comp<T>> {
    validate_activation(activation)?;
    Ok(match activation {
        Activation::None | Activation::Linear => x.clone(),
        Activation::Relu => x.relu()?,
        Activation::LeakyRelu(a) => x.lrelu(T::from_f64(a))?,
        Activation::Sigmoid => x.sigmoid()?,
        Activation::Tanh => x.tanh()?,
        Activation::Gelu => x.gelu()?,
        Activation::Elu(a) => x.elu(T::from_f64(a))?,
        Activation::Selu => x.selu(T::from_f64(1.0507009873554805),T::from_f64(1.6732632423543772))?,
        Activation::SiLu => x.silu()?,
        Activation::Softplus => x.softplus()?,
        Activation::Mish => x.mish()?,
        Activation::Softsign => x.softsign()?,
        Activation::HardSigmoid => x.hardsigmoid()?,
        Activation::HardTanh { min, max } => x.hardtanh(T::from_f64(min), T::from_f64(max))?,
        Activation::Softmax => x.softmax(-1)?,
    })
}
