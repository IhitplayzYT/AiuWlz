use Tensors::{dtype::Float, tensor::Tensor};
use Utilities::{device::Device::Device, random::Random::Rng};

use crate::{Activation, Initializer, Result, common::{invalid, validate_activation}, layer::{activate_tensor, initialize}};

/// 2D convolution layer
/// Input & Output = [batch_sz,channels,height,width]
/// Kernel = [output_channels,input_channels,height,width]
#[derive(Clone,Debug)]
pub struct Conv2d<T: Float> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub weights: Tensor<T>,
    pub bias: Tensor<T>, // [1, output_channels, 1, 1]
    pub activation: Activation,
}
pub type Conv2D<T> = Conv2d<T>;

impl<T: Float> Conv2d<T> {
    /// Create a new Conv2D layer
    pub fn new(in_channels: usize,out_channels: usize,kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize),activation: Activation,initializer: Initializer,rng: &mut Rng,device: Device) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 || kernel.0 == 0 || kernel.1 == 0 || stride.0 == 0 || stride.1 == 0{ return Err(invalid("Conv channels, kernel, and strides must be positive")); }
        validate_activation(activation)?;
        let fan_in = in_channels * kernel.0 * kernel.1;
        let fan_out = out_channels * kernel.0 * kernel.1;
        let weights = initialize(&[out_channels, in_channels, kernel.0, kernel.1],fan_in,fan_out,initializer,rng,)?.to_device(device)?;
        let bias = Tensor::zeros(&[1, out_channels, 1, 1]).to_device(device)?;
        Ok(Self {in_channels,out_channels,kernel,stride,padding,weights,bias,activation})
    }

    /// Conv2D forward pass
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if input.ndim() != 4 || input.shape()[1] != self.in_channels { return Err(invalid(format!("Conv2d expects [batch, {}, height, width], got {:?}",self.in_channels,input.shape()))); }
        activate_tensor(&input.conv2d(&self.weights, Some(&self.bias), self.stride, self.padding)?,self.activation)
    }

    /// Set the weight and bias([w,b])
    pub fn set_parameters(&mut self, params: &[Tensor<T>]) -> Result<()> {
        if params.len() != 2 || params[0].shape() != self.weights.shape() || params[1].shape() != self.bias.shape() { return Err(invalid("Conv2d param shape is diff")); }
        params[0].same_device(&params[1])?;
        if params[0].device() != self.weights.device() { return Err(invalid("Conv2d parameters aren't on same device as the layer")); }
        self.weights = params[0].clone();
        self.bias = params[1].clone();
        Ok(())
    }

    /// Set the weigth and bias
    pub fn set_params(&mut self, weight:Tensor<T>,bias:Tensor<T>) -> Result<()> {
        if weight.shape() != self.weights.shape() || bias.shape() != self.bias.shape() { return Err(invalid("Conv2d param shape is diff")); }
        weight.same_device(&bias)?;
        if weight.device() != self.weights.device() { return Err(invalid("Conv2d parameters aren't on same device as the layer")); }
        self.weights = weight.clone();
        self.bias = bias.clone();
        Ok(())
    }

    /// Sets a symetrical stride
    pub fn sym_stride(&mut self,stride:usize) {
        self.stride = (stride,stride);
    }

    /// Sets a symetrical kernel 
    pub fn sym_kernel(&mut self,kernel:usize) {
        self.kernel = (kernel,kernel);
    }

    /// Sets a symetrical padding 
    pub fn sym_pad(&mut self,padding:usize) {
        self.padding = (padding,padding);
    }

    /// Chaining op with a symetrical stride
    pub fn with_sym_stride(mut self,stride:usize) -> Self{
        self.stride = (stride,stride);
        self
    }

    /// Chaining op with a symetrical kernel 
    pub fn with_sym_kernel(mut self,kernel:usize) -> Self{
        self.kernel = (kernel,kernel);
        self
    }

    /// Chaining op with a symetrical padding 
    pub fn with_sym_pad(mut self,padding:usize) -> Self{
        self.padding = (padding,padding);
        self
    }

    /// Move to the specified device
    pub fn to_device(&mut self,dev:Device) -> Result<()>{
        self.weights = self.weights.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        Ok(())
    }

    /// Chaining op with specific device 
    pub fn with_device(mut self,dev: Device) -> Result<Self>{
        self.weights = self.weights.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        Ok(self)
    }

    /// Chaining with a specific stride
    pub fn with_stride(mut self,stride: (usize, usize)) -> Self{
        self.stride = stride;
        self 
    }

    /// No of trainable params
    pub fn param_count(&self) -> usize{ self.out_channels * (self.in_channels * self.kernel.0 * self.kernel.1 + 1) }
}

/// 1D convolution
/// Input & Output:[batch, channels, length]
/// weight:[out, in, kernel]
#[derive(Clone,Debug)]
pub struct Conv1d<T: Float> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weights: Tensor<T>,
    pub bias: Tensor<T>,
    pub activation: Activation,
}
pub type Conv1D<T> = Conv1d<T>;

impl<T: Float> Conv1d<T> {

    /// Create a new Conv1D layer
    pub fn new(in_channels: usize,out_channels: usize,kernel_size: usize,stride: usize,padding: usize,activation: Activation,initializer: Initializer,rng: &mut Rng,device: Device) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 || kernel_size == 0 || stride == 0 { return Err(invalid("Conv1d channels, kernel, and stride must be positive")); }
        validate_activation(activation)?;
        let weights = initialize(&[out_channels, in_channels, kernel_size],in_channels * kernel_size,out_channels * kernel_size,initializer,rng)?.to_device(device)?;
        let bias = Tensor::zeros(&[1, out_channels, 1]).to_device(device)?;
        Ok(Self {in_channels,out_channels,kernel_size,stride,padding,weights,bias,activation})
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        validate_conv1d_input(input, self.in_channels)?;
        let x = input.unsqueeze(2)?;
        let w = self.weights.reshape(&[self.out_channels as isize,self.in_channels as isize,1,self.kernel_size as isize])?;
        let b = self.bias.reshape(&[1, self.out_channels as isize, 1, 1])?;
        activate_tensor(&x.conv2d(&w, Some(&b), (1, self.stride), (0, self.padding))?.squeeze(2)?,self.activation)
    }

    /// Set the params([w,b])
    pub fn set_parameters(&mut self, params: &[Tensor<T>]) -> Result<()> {
        if params.len() != 2 || params[0].shape() != self.weights.shape() || params[1].shape() != self.bias.shape(){ return Err(invalid("Conv1d parameter shape mismatch")); }
        params[0].same_device(&params[1])?;
        if params[0].device() != self.weights.device() { return Err(invalid("Conv1d params should be on same device as the layer")); }
        self.weights = params[0].clone();
        self.bias = params[1].clone();
        Ok(())
    }

    /// Set the params
    pub fn set_params(&mut self, weight:Tensor<T>,bias: Tensor<T>) -> Result<()> {
        if weight.shape() != self.weights.shape() || bias.shape() != self.bias.shape(){ return Err(invalid("Conv1d parameter shape mismatch")); }
        weight.same_device(&bias)?;
        if weight.device() != self.weights.device() { return Err(invalid("Conv1d params should be on same device as the layer")); }
        self.weights = weight.clone();
        self.bias = bias.clone();
        Ok(())
    }

    /// Chaining with specific stride
    pub fn with_stride(mut self,stride: usize) -> Self{
        self.stride = stride;
        self
    }

    /// Move to specific device
    pub fn to_device(&mut self,dev: Device) -> Result<()> {
        self.weights = self.weights.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        Ok(())
    }

    /// Chaining op with specific device 
    pub fn with_device(mut self,dev: Device) -> Result<Self>{
        self.weights = self.weights.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        Ok(self)
    }

    /// No of trainable params
    pub fn param_count(&self) -> usize{ self.out_channels * (self.in_channels * self.kernel_size + 1) }

}

/// 2D Max pooling
#[derive(Clone, Copy, Debug)]
pub struct MaxPool2d {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}
pub type MaxPool2D = MaxPool2d;

impl MaxPool2d {

    /// Create a new Maxpool2D
    pub fn new(kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize)) -> Result<Self> {
        validate_pool(kernel, stride)?;
        Ok(Self {kernel,stride,padding})
    }

    /// Chaining op with a specific kernel
    pub fn with_kernel(mut self,kernel: (usize, usize)) -> Self{
        self.kernel = kernel;
        self
    }

    /// Chaining op with a specific stride
    pub fn with_stride(mut self,stride: (usize, usize)) -> Self{
        self.stride = stride;
        self
    }
    

    /// Chaining op with a specific padding 
    pub fn with_padding(mut self,padding: (usize, usize)) -> Self{
        self.padding = padding;
        self
    }

    /// Sets a symetrical stride
    pub fn sym_stride(&mut self,stride:usize) {
        self.stride = (stride,stride);
    }

    /// Sets a symetrical kernel 
    pub fn sym_kernel(&mut self,kernel:usize) {
        self.kernel = (kernel,kernel);
    }

    /// Sets a symetrical padding 
    pub fn sym_pad(&mut self,padding:usize) {
        self.padding = (padding,padding);
    }

    /// Chaining op with a symetrical stride
    pub fn with_sym_stride(mut self,stride:usize) -> Self{
        self.stride = (stride,stride);
        self
    }

    /// Chaining op with a symetrical kernel 
    pub fn with_sym_kernel(mut self,kernel:usize) -> Self{
        self.kernel = (kernel,kernel);
        self
    }

    /// Chaining op with a symetrical padding 
    pub fn with_sym_pad(mut self,padding:usize) -> Self{
        self.padding = (padding,padding);
        self
    }

    /// Set the params
    pub fn set_params(&mut self,kernel: Option<(usize, usize)>, stride: Option<(usize, usize)>,padding: Option<(usize, usize)>){
        if let Some(padding) = padding{
            self.padding = padding;
        }

        if let Some(kernel) = kernel{
            if kernel.0 != 0 && kernel.1 != 0 {self.kernel = kernel;}
        }

        if let Some(stride) = stride{
            if stride.0 != 0 && stride.1 != 0 {self.stride = stride;}
        }

    }

    /// No of trainable params
    pub fn param_count(&self) -> usize { 0 }
}

/// Avg Pooling
#[derive(Clone, Copy, Debug)]
pub struct AvgPool2d {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub count_pad: bool,
}
pub type AvgPool2D = AvgPool2d;

impl AvgPool2d {
    /// Create a new Avgpool2D layer
    pub fn new(kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize),count_pad: bool) -> Result<Self> {
        validate_pool(kernel, stride)?;
        Ok(Self {kernel,stride,padding,count_pad})
    }

    /// Chaining op with a specific kernel 
    pub fn with_kernel(mut self,kernel: (usize, usize)) -> Self{
        self.kernel = kernel;
        self
    }

    /// Chaining op with a specific stride 
    pub fn with_stride(mut self,stride: (usize, usize)) -> Self{
        self.stride = stride;
        self
    }
    
    /// Chaining op with a specific padding 
    pub fn with_padding(mut self,padding: (usize, usize)) -> Self{
        self.padding = padding;
        self
    }

    /// Chaining op with count_pad set to true 
    pub fn with_pad_count(mut self) -> Self{
        self.count_pad = true;
        self
    }

    /// Sets a symetrical stride
    pub fn sym_stride(&mut self,stride:usize) {
        self.stride = (stride,stride);
    }

    /// Sets a symetrical kernel 
    pub fn sym_kernel(&mut self,kernel:usize) {
        self.kernel = (kernel,kernel);
    }

    /// Sets a symetrical padding 
    pub fn sym_pad(&mut self,padding:usize) {
        self.padding = (padding,padding);
    }

    /// Chaining op with a symetrical stride
    pub fn with_sym_stride(mut self,stride:usize) -> Self{
        self.stride = (stride,stride);
        self
    }

    /// Chaining op with a symetrical kernel 
    pub fn with_sym_kernel(mut self,kernel:usize) -> Self{
        self.kernel = (kernel,kernel);
        self
    }

    /// Chaining op with a symetrical padding 
    pub fn with_sym_pad(mut self,padding:usize) -> Self{
        self.padding = (padding,padding);
        self
    }

    /// Set the params
    pub fn set_params(&mut self,kernel: Option<(usize, usize)>, stride: Option<(usize, usize)>,padding: Option<(usize, usize)>,count_pad:Option<bool>){
        if let Some(padding) = padding{
            self.padding = padding;
        }

        if let Some(kernel) = kernel{
            if kernel.0 != 0 && kernel.1 != 0 {self.kernel = kernel;}
        }

        if let Some(stride) = stride{
            if stride.0 != 0 && stride.1 != 0 {self.stride = stride;}
        }

        if let Some(count_pad) = count_pad{
            self.count_pad = count_pad;
        }

    }

    /// No of trainable params
    pub fn param_count(&self) -> usize { 0 }
}


/// 1D max pool
#[derive(Clone, Copy, Debug)]
pub struct MaxPool1d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}
pub type MaxPool1D = MaxPool1d;

impl MaxPool1d {

    /// Create a new Maxpool1D layer
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Result<Self> {
        validate_pool((1, kernel_size), (1, stride))?;
        Ok(Self {kernel_size,stride,padding})
    }

    /// Chaining op with a specific kernel 
    pub fn with_kernel(mut self,kernel: usize) -> Self{
        self.kernel_size = kernel;
        self
    }

    /// Chaining op with a specific stride 
    pub fn with_stride(mut self,stride: usize) -> Self{
        self.stride = stride;
        self
    }
    
    /// Chaining op with a specific padding 
    pub fn with_padding(mut self,padding: usize) -> Self{
        self.padding = padding;
        self
    }

    /// Set the params
    pub fn set_params(&mut self,kernel: Option<usize>, stride: Option<usize>,padding: Option<usize>){
        if let Some(padding) = padding{
            self.padding = padding;
        }

        if let Some(kernel_size) = kernel{
            if kernel_size != 0 {self.kernel_size = kernel_size;}
        }

        if let Some(stride) = stride{
            if stride != 0 {self.stride = stride;}
        }

    }

    /// No of trainable params
    pub fn param_count(&self) -> usize { 0 }
}

#[derive(Clone, Copy, Debug)]
pub struct AvgPool1d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_pad: bool,
}
pub type AvgPool1D = AvgPool1d; 

impl AvgPool1d {
    /// Create a new Avgpool1D layer
    pub fn new(kernel_size: usize,stride: usize,padding: usize,count_pad: bool) -> Result<Self> {
        validate_pool((1, kernel_size), (1, stride))?;
        Ok(Self {kernel_size,stride,padding,count_pad})
    }

    /// Chaining op with a specific kernel 
    pub fn with_kernel(mut self,kernel: usize) -> Self{
        self.kernel_size = kernel;
        self
    }

    /// Chaining op with a specific stride 
    pub fn with_stride(mut self,stride: usize) -> Self{
        self.stride = stride;
        self
    }
    
    /// Chaining op with a specific padding
    pub fn with_padding(mut self,padding: usize) -> Self{
        self.padding = padding;
        self
    }

    /// Chaining op with count_pad set to true 
    pub fn with_pad_count(mut self) -> Self{
        self.count_pad = true;
        self
    }

    /// Set the params
    pub fn set_params(&mut self,kernel: Option<usize>, stride: Option<usize>,padding: Option<usize>,pad_count: Option<bool>){
        if let Some(padding) = padding{
            self.padding = padding;
        }

        if let Some(kernel_size) = kernel{
            if kernel_size != 0 {self.kernel_size = kernel_size;}
        }

        if let Some(stride) = stride{
            if stride != 0 {self.stride = stride;}
        }

        if let Some(pad_count) = pad_count{
            self.count_pad = pad_count;
        }

    }

    /// No of trainable params 
    pub fn param_count(&self) -> usize { 0 }
}



#[derive(Clone,Debug)]
pub struct ConvTranspose2d<T: Float> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub weights: Tensor<T>,
    pub bias: Tensor<T>,
    pub activation: Activation,
}
pub type ConvTranspose2D<T> = ConvTranspose2d<T>;

impl<T: Float> ConvTranspose2d<T> {

    /// Create a new ConvTranspose2d layer
    pub fn new(in_channels: usize,out_channels: usize,kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize),activation: Activation,initializer: Initializer,rng: &mut Rng,device: Device) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 || kernel.0 == 0 || kernel.1 == 0 || stride.0 == 0 || stride.1 == 0 { return Err(invalid("ConvTranspose2d channels, kernel, and strides must be positive")); }
        validate_activation(activation)?;
        let fan_in = out_channels * kernel.0 * kernel.1;
        let fan_out = in_channels * kernel.0 * kernel.1;
        let weights = initialize(&[in_channels, out_channels, kernel.0, kernel.1],fan_in,fan_out,initializer,rng)?.to_device(device)?;
        let bias = Tensor::zeros(&[1, out_channels, 1, 1]).to_device(device)?;
        Ok(Self {in_channels,out_channels,kernel,stride,padding,weights,bias,activation})
    }

    /// Fwd pass
    pub fn forward(&self,input: &Tensor<T>) -> Result<Tensor<T>> {
        if input.ndim() != 4 || input.shape()[1] != self.in_channels { return Err(invalid(format!("ConvTranspose2d expects [batch, {}, height, width], got {:?}",self.in_channels,input.shape()))); }
        let output = input.conv_transpose2d(&self.weights,self.stride,self.padding)?;
        activate_tensor(&output.add(&self.bias)?,self.activation)
    }

    /// Set the params [weight,bias]
    pub fn set_parameters(&mut self,params: &[Tensor<T>]) -> Result<()> {
        if params.len() != 2 || params[0].shape() != self.weights.shape() || params[1].shape() != self.bias.shape() { return Err(invalid("ConvTranspose2d parameter shape mismatch")); }
        params[0].same_device(&params[1])?;
        if params[0].device() != self.weights.device() { return Err(invalid("ConvTranspose2d parameters aren't on same device as the layer")); }
        self.weights = params[0].clone();
        self.bias = params[1].clone();
        Ok(())
    }

    /// Set the params
    pub fn set_params(&mut self,weight: Tensor<T>,bias: Tensor<T>) -> Result<()> { self.set_parameters(&[weight,bias]) }

    /// Move layer to dev
    pub fn to_device(&mut self,dev: Device) -> Result<()> {
        self.weights = self.weights.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        Ok(())
    }

    /// Chaining op for move to dev
    pub fn with_device(mut self,dev: Device) -> Result<Self> {
        self.weights = self.weights.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        Ok(self)
    }

    /// Chaining op for with stride
    pub fn with_stride(mut self,stride: (usize,usize)) -> Self { self.stride = stride; self }

    /// Sets a symetrical stride
    pub fn sym_stride(&mut self,stride:usize) {
        self.stride = (stride,stride);
    }

    /// Sets a symetrical kernel 
    pub fn sym_kernel(&mut self,kernel:usize) {
        self.kernel = (kernel,kernel);
    }

    /// Sets a symetrical padding 
    pub fn sym_pad(&mut self,padding:usize) {
        self.padding = (padding,padding);
    }

    /// Chaining op with a symetrical stride
    pub fn with_sym_stride(mut self,stride:usize) -> Self{
        self.stride = (stride,stride);
        self
    }

    /// Chaining op with a symetrical kernel 
    pub fn with_sym_kernel(mut self,kernel:usize) -> Self{
        self.kernel = (kernel,kernel);
        self
    }

    /// Chaining op with a symetrical padding 
    pub fn with_sym_pad(mut self,padding:usize) -> Self{
        self.padding = (padding,padding);
        self
    }

    /// No of trainable params
    pub fn param_count(&self) -> usize { self.out_channels * (self.in_channels * self.kernel.0 * self.kernel.1 + 1) }
}

#[derive(Clone,Debug)]
pub struct ConvTranspose1d<T: Float> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weights: Tensor<T>,
    pub bias: Tensor<T>,
    pub activation: Activation,
}
pub type ConvTranspose1D<T> = ConvTranspose1d<T>;

impl<T: Float> ConvTranspose1d<T> {

    /// Create a new ConvTranspose1d layer
    pub fn new(in_channels: usize,out_channels: usize,kernel_size: usize,stride: usize,padding: usize,activation: Activation,initializer: Initializer,rng: &mut Rng,device: Device) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 || kernel_size == 0 || stride == 0 { return Err(invalid("ConvTranspose1d channels, kernel, and stride must be positive")); }
        validate_activation(activation)?;
        let weights = initialize(&[in_channels, out_channels, kernel_size],out_channels * kernel_size,in_channels * kernel_size,initializer,rng)?.to_device(device)?;
        let bias = Tensor::zeros(&[1, out_channels, 1]).to_device(device)?;
        Ok(Self {in_channels,out_channels,kernel_size,stride,padding,weights,bias,activation})
    }

    ///  Fwd pass
    pub fn forward(&self,input: &Tensor<T>) -> Result<Tensor<T>> {
        validate_conv1d_input(input,self.in_channels)?;
        let x = input.unsqueeze(2)?;
        let weights = self.weights.unsqueeze(2)?;
        let output = x.conv_transpose2d(&weights,(1,self.stride),(0,self.padding))?.squeeze(2)?;
        activate_tensor(&output.add(&self.bias)?,self.activation)
    }

    /// Set the params[weight,bias]
    pub fn set_parameters(&mut self,params: &[Tensor<T>]) -> Result<()> {
        if params.len() != 2 || params[0].shape() != self.weights.shape() || params[1].shape() != self.bias.shape() { return Err(invalid("ConvTranspose1d parameter shape mismatch")); }
        params[0].same_device(&params[1])?;
        if params[0].device() != self.weights.device() { return Err(invalid("ConvTranspose1d parameters aren't on same device as the layer")); }
        self.weights = params[0].clone();
        self.bias = params[1].clone();
        Ok(())
    }

    /// Set the params
    pub fn set_params(&mut self,weight: Tensor<T>,bias: Tensor<T>) -> Result<()> { self.set_parameters(&[weight,bias]) }

    /// Move to the dev
    pub fn to_device(&mut self,dev: Device) -> Result<()> {
        self.weights = self.weights.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        Ok(())
    }

    /// Chaining op for move to dev
    pub fn with_device(mut self,dev: Device) -> Result<Self> {
        self.weights = self.weights.to_device(dev)?;
        self.bias = self.bias.to_device(dev)?;
        Ok(self)
    }

    /// Chaining op for with stride
    pub fn with_stride(mut self,stride: usize) -> Self { self.stride = stride; self }

    /// No of trainable params
    pub fn param_count(&self) -> usize { self.out_channels * (self.in_channels * self.kernel_size + 1) }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveMaxPool1d { pub output_size: usize }
pub type AdaptiveMaxPool1D = AdaptiveMaxPool1d;
impl AdaptiveMaxPool1d {

    /// Creates a new AdaptiveMaxPool1d layer
    pub fn new(output_size: usize) -> Result<Self> { if output_size == 0 { return Err(invalid("Adaptive pool output size must be positive")); } Ok(Self {output_size}) }

    /// No of trainable params
    pub fn param_count(&self) -> usize { 0 }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveAvgPool1d { pub output_size: usize }
pub type AdaptiveAvgPool1D = AdaptiveAvgPool1d;
impl AdaptiveAvgPool1d {

    /// Creates a new AdaptiveAvgPool1d layer
    pub fn new(output_size: usize) -> Result<Self> { if output_size == 0 { return Err(invalid("Adaptive pool output size must be positive")); } Ok(Self {output_size}) }

    /// No of trainable params
    pub fn param_count(&self) -> usize { 0 }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveMaxPool2d { pub output_size: (usize, usize) }
pub type AdaptiveMaxPool2D = AdaptiveMaxPool2d;
impl AdaptiveMaxPool2d {

    /// Creates a new AdaptiveMaxPool2d layer
    pub fn new(output_size: (usize, usize)) -> Result<Self> { if output_size.0 == 0 || output_size.1 == 0 { return Err(invalid("Adaptive pool output dimensions must be positive")); } Ok(Self {output_size}) }

    /// No of trainable params
    pub fn param_count(&self) -> usize { 0 }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveAvgPool2d { pub output_size: (usize, usize) }
pub type AdaptiveAvgPool2D = AdaptiveAvgPool2d;
impl AdaptiveAvgPool2d {

    /// Creates a new AdaptiveAvgPool2d layer
    pub fn new(output_size: (usize, usize)) -> Result<Self> { if output_size.0 == 0 || output_size.1 == 0 { return Err(invalid("Adaptive pool output dimensions must be positive")); } Ok(Self {output_size}) }

    /// No of trainable params
    pub fn param_count(&self) -> usize { 0 }
}


pub fn validate_pool(kernel: (usize, usize), stride: (usize, usize)) -> Result<()> { if kernel.0 == 0 || kernel.1 == 0 || stride.0 == 0 || stride.1 == 0 {Err(invalid("Pooling kernel and stride dims must be positive"))} else {Ok(())} }
pub fn validate_conv1d_input<T: Float>(input: &Tensor<T>, channels: usize) -> Result<()> { if input.ndim() != 3 || input.shape()[1] != channels { return Err(invalid(format!("Conv1d expected [batch, {channels}, len], got {:?}",input.shape()))); } else {Ok(())} }
