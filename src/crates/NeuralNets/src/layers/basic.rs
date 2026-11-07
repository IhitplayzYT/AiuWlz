use Tensors::{dtype::Float, tensor::Tensor};
use Utilities::{device::Device::Device, random::Random::Rng};

use crate::{Activation, Initializer, Result, common::{invalid, validate_activation, validate_feature_input}, layer::{Layer, activate_tensor, initialize}};

/// Dense layer
#[derive(Debug,Clone)]
pub struct Dense<T: Float> {
    pub weights: Tensor<T>,
    pub bias: Tensor<T>,
    pub activation: Activation,
}
pub type Linear<T> = Dense<T>;

impl<T: Float> Dense<T> {

    pub fn new(input_size: usize,output_size: usize,activation: Activation,initializer: Initializer,rng: &mut Rng) -> Result<Self> { Self::new_on(input_size,output_size,activation,initializer,rng,Device::Cpu) }
    pub fn new_on(input_size: usize,output_size: usize,activation: Activation,initializer: Initializer,rng: &mut Rng,device: Device) -> Result<Self> { Self::new_with_initializers(input_size,output_size,activation,initializer,Initializer::Zeros,rng,device) }
    pub fn input_size(&self) -> usize { self.weights.shape()[0]}
    pub fn output_size(&self) -> usize { self.weights.shape()[1]}
    pub fn param_count(&self) -> usize{ (1 + self.input_size()) * self.output_size() }
    pub fn new_with_initializers(input_size: usize,output_size: usize,activation: Activation,weight_init: Initializer,bias_init: Initializer,rng: &mut Rng,device: Device) -> Result<Self> {
        if input_size == 0 || output_size == 0 { return Err(invalid("Dense layer dims have be positive")); }
        validate_activation(activation)?;
        let weights = initialize(&[input_size, output_size],input_size,output_size,weight_init,rng,)?.to_device(device)?;
        let bias = initialize(&[1, output_size], input_size, output_size, bias_init, rng)?.to_device(device)?;
        Ok(Self {weights,bias,activation})
    }

    pub fn to_device(&mut self,dev: Device) -> Result<()>{
        self.weights = self.weights.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        Ok(())
    }

    pub fn set_params(&mut self, weights: Tensor<T>, bias: Tensor<T>) -> Result<()> {
        if weights.shape() != self.weights.shape() { return Err(invalid(format!("Expected weight shape {:?}, got {:?}",self.weights.shape(),weights.shape()))); }
        if bias.shape() != self.bias.shape() { return Err(invalid(format!("Expected bias shape {:?}, got {:?}",self.bias.shape(),bias.shape()))); }
        weights.same_device(&bias)?;
        if weights.device() != self.weights.device() { return Err(invalid("New params aren't on same device as layer")); }
        self.weights = weights;
        self.bias = bias;
        Ok(())
    }

    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        validate_feature_input(input, self.input_size(), "dense")?;
        input.same_device(&self.weights)?;
        activate_tensor(&input.matmul(&self.weights)?.add(&self.bias)?,self.activation)
    }
}

/// Layer norm
#[derive(Clone,Debug)]
pub struct LayerNorm<T: Float> {
    pub features: usize,
    pub weight: Tensor<T>,
    pub bias: Tensor<T>,
    pub epsilon: T,
}

impl<T: Float> LayerNorm<T> {
    pub fn new(features: usize, epsilon: T, device: Device) -> Result<Self> {
        if features == 0 || epsilon <= T::zero() { return Err(invalid("Layer norm features and epsilon must be positive")); }
        Ok(Self {features,weight: Tensor::ones(&[1, features]).to_device(device)?,bias: Tensor::zeros(&[1, features]).to_device(device)?,epsilon})
    }

    pub fn set_params(&mut self,features: usize,epsilon: T) {
        self.epsilon = epsilon;
        self.features = features;
        self.weight = Tensor::ones(&[1,features]);
        self.bias = Tensor::zeros(&[1,features]);
    }

    pub fn to_device(&mut self,dev:Device) -> Result<()>{
        self.weight = self.weight.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        Ok(())
    }

    pub fn input_size(&self) -> usize {self.features}
    pub fn output_size(&self) -> usize {self.features}
    pub fn param_count(&self) -> usize {self.features * 2}
}

/// Batch norm for [batch, features]
#[derive(Clone,Debug)]
pub struct BatchNorm1d<T: Float> {
    pub features: usize,
    pub weight: Tensor<T>,
    pub bias: Tensor<T>,
    pub running_mean: Tensor<T>,
    pub running_var: Tensor<T>,
    pub epsilon: T,
    pub momentum: T,
}

impl<T: Float> BatchNorm1d<T> {
    pub fn new(features: usize, epsilon: T, momentum: T, device: Device) -> Result<Self> {
        if features == 0 || epsilon <= T::zero() || momentum < T::zero() || momentum > T::one() { return Err(invalid("Batch norm requires positive features,epsilon and momentum in [0, 1]")); }
        Ok(Self {features,weight: Tensor::ones(&[1, features]).to_device(device)?,bias: Tensor::zeros(&[1, features]).to_device(device)?,running_mean: Tensor::zeros(&[1, features]).to_device(device)?,running_var: Tensor::ones(&[1, features]).to_device(device)?,epsilon,momentum})
    }

    pub fn to_device(&mut self,dev:Device) -> Result<()>{
        self.weight = self.weight.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        self.running_mean = self.running_mean.to_device(dev)?;
        self.running_var = self.running_var.to_device(dev)?;
        Ok(())
    }
    pub fn input_size(&self) -> usize {self.features}
    pub fn output_size(&self) -> usize {self.features}
    pub fn param_count(&self) -> usize { self.features * 2}
}

/// Dropout layer
#[derive(Clone, Copy, Debug)]
pub struct Dropout {
    pub probability: f64,
}

impl Dropout {
    pub fn new(probability: f64) -> Result<Self> {
        if !probability.is_finite() || !(0.0..1.0).contains(&probability) {return Err(invalid("Dropout prob must be in [0, 1)"));}
        Ok(Self { probability })
    }

    pub fn param_count(&self) -> usize { 0 }
}

/// Skip connection
#[derive(Clone,Debug)]
pub struct ResidualBlock<T: Float> {
    pub layers: Vec<Layer<T>>,
}
pub type Skip<T> = ResidualBlock<T>;

impl<T: Float> ResidualBlock<T> {
    pub fn new(layers: Vec<Layer<T>>) -> Result<Self> {
        if layers.is_empty() { return Err(invalid("Residual block must contain at least 1 layer")); }
        Ok(Self { layers })
    }

    pub fn param_count(&self) -> usize { 0 }
}

/// PReLU with learned negative slope per feature
#[derive(Clone,Debug)]
pub struct PRelu<T: Float> {
    pub features: usize,
    pub weight: Tensor<T>,
}

impl<T: Float> PRelu<T> {
    pub fn new(features: usize, device: Device) -> Result<Self> {
        if features == 0 { return Err(invalid("PReLU feature count must be positive"));}
        Ok(Self {features,weight: Tensor::fill(&[1, features], T::from_f64(0.25)).to_device(device)?})
    }

    pub fn to_device(&mut self,dev: Device) -> Result<()>{
        self.weight = self.weight.to_device(dev)?;
        Ok(())
    }

    pub fn param_count(&self) -> usize{ self.features }
}