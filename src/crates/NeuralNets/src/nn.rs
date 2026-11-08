use Tensors::{autograd::Comp, dtype::Float, tensor::Tensor};
use Utilities::{device::Device::Device, random::Random::Rng};
use crate::{common::{Activation, EpochMetrics, FitOptions, Initializer, Loss, Result, invalid}, layer::Layer,layers::{basic::*,cnn::*,spec::*}, loss::{loss_comp, metrics_loss_value, one_hot}, optim::{LrScheduler, Optimizer}};

/// Similar to Pytorch Sequential api
pub struct Sequential<T: Float> {
    layers: Vec<Layer<T>>,
    embedding: Option<TokenEmbedding<T>>,
    device: Device,
    rng: Rng,
}

pub type NeuralNetwork<T> = Sequential<T>;
pub type NN<T> = Sequential<T>;

impl<T: Float> Default for Sequential<T> {
    fn default() -> Self { Self::new() }
}

impl<T: Float> Sequential<T> {

    /// Make a new Sequential object 
    pub fn new() -> Self { Self::new_on(Device::Cpu) }

    /// Make a new Sequential object  on device 
    pub fn new_on(device: Device) -> Self { Self {layers: Vec::new(),embedding: None,device,rng: Rng::seed(42)} }

    /// Update the seed
    pub fn set_seed(&mut self, seed: u64) {self.rng = Rng::seed(seed);}

    /// Count of the total no of Layers
    pub fn layer_count(&self) -> usize {self.layers.len() + usize::from(self.embedding.is_some())}
    
    pub fn is_empty(&self) -> bool {self.layers.is_empty() && self.embedding.is_none()}

    /// Ref to the layers
    pub fn layers(&self) -> &[Layer<T>] {&self.layers}

    /// Mutable Ref to the layers
    pub fn layers_mut(&mut self) -> &mut [Layer<T>] {&mut self.layers}

    /// Ref of the embedding layer
    pub fn embedding(&self) -> Option<&TokenEmbedding<T>> {self.embedding.as_ref()}

    /// Get the current device of obj 
    pub fn device(&self) -> Device {self.device}

    /// The input size to the Sequential layers
    pub fn input_size(&self) -> Option<usize> { self.layers.iter().find_map(|l| {if let Layer::Dense(d) = l { Some(d.input_size()) } else { None }}) }

    /// Output size of the Sequential layers:
    pub fn output_size(&self) -> Option<usize> {
        for layer in self.layers.iter().rev() {
            match layer {
                Layer::Dense(x) => return Some(x.output_size()),
                Layer::Conv1d(x) => return Some(x.out_channels),
                Layer::Conv2d(x) => return Some(x.out_channels),
                Layer::GlobalAvgPool1d | Layer::GlobalMaxPool1d | Layer::GlobalAvgPool2d | Layer::GlobalMaxPool2d => {}
                Layer::LayerNorm(x) => return Some(x.features),
                Layer::BatchNorm1d(x) => return Some(x.features),
                Layer::Flatten => return None,
                _ => {}
            }
        }
        if self.layers.is_empty() { self.embedding.as_ref().map(|embedding| embedding.embed_dim) } else { None }
    }

    /// Total no of trainable params in the Sequential obj
    pub fn param_count(&self) -> usize { self.params().iter().map(Tensor::numel).sum()}

    /// Vce containing containing trainable params Tensors
    pub fn params(&self) -> Vec<Tensor<T>> {
        let mut parameters = self.embedding.iter().map(|embedding| embedding.weights.clone()).collect::<Vec<_>>();
        parameters.extend(self.layers.iter().flat_map(Layer::params));
        parameters
    }

    /// Set the params [embedding_params,rest of the params sequentially] 
    pub fn set_parameters(&mut self, parameters: &[Tensor<T>]) -> Result<()> {
        let expected = self.params().len();

        if parameters.len() == expected || parameters.len() == expected-1{} else if parameters.len() != expected { return Err(invalid(format!("expected {expected} parameter tensors, got {}",parameters.len()))); }
        let mut st = 0;
        if parameters.len() == expected{
            if let Some(embedding) = &mut self.embedding {
                if parameters[0].shape() != embedding.weights.shape() || parameters[0].device() != self.device{ return Err(invalid("EMbeeding param shape or device not same")); }
                embedding.weights = parameters[0].clone();
                st = 1;
            }
        }
        for layer in &mut self.layers {
            let x = layer.params().len();
            if x > 0 {
                let t = &parameters[st..st + x];
                for p in t {
                    if p.device() != self.device { return Err(invalid("model parameters must be on the model device")); }
                }
                layer.set_params(t)?;
                st += x;
            }
        }
        Ok(())
    }


    /// Attach an embedding layer before the rest of the layers
    /// Works only when embedding layer doesnt exist of the layers are empty
    pub fn add_embedding(&mut self,vocabulary_size: usize,embedding_dim: usize,initializer: Initializer,rng: &mut Rng) -> Result<&mut Self> {
        if self.embedding.is_some() || !self.layers.is_empty() { return Err(invalid("embedding must be the first and only embedding module in a sequential model")); }
        self.embedding = Some(TokenEmbedding::new(vocabulary_size,embedding_dim,initializer,rng,self.device)?);
        Ok(self)
    }

    /// Adds a layer to the obj
    pub fn add_layer(&mut self, layer: Layer<T>) -> Result<&mut Self> {
        let params = layer.params();
        if let Some(first) = params.first() {
            if params.iter().any(|p| p.device() != first.device()) { return Err(invalid("Layer parameters are on diff device")); }
        }

        let inited = self.embedding.is_some() || !self.layers.is_empty();
        for p in &params {
            if inited && p.device() != self.device { return Err(invalid("Layer params must use the models device")); }
        }
        if !inited && !params.is_empty() { self.device = params[0].device(); }        
        let mut last_features = None;
        for previous in self.layers.iter().rev() {
            match previous {
                Layer::Dense(d) => {last_features = Some(d.output_size());break;},
                Layer::Conv2d(d) => {last_features = Some(d.out_channels);break;},
                Layer::Conv1d(d) => {last_features = Some(d.out_channels);break;},
                Layer::LayerNorm(n) => {last_features = Some(n.features);break;},
                Layer::BatchNorm1d(n) => {last_features = Some(n.features);break;},
                Layer::PRelu(n) => {last_features = Some(n.features);break;},
                Layer::Flatten => break,
                _ => {}
            }
        }

        if let (Some(last), Layer::Dense(next)) = (last_features, &layer) {
            if last != next.input_size() { return Err(invalid(format!("Input size {} does not match previous feature size {last}",next.input_size()))); }
        }

        if let (Some(last), Layer::LayerNorm(next)) = (last_features, &layer) {
            if last != next.features { return Err(invalid(format!("Layer norm features {} does not match previous feature size {last}",next.features))); }
        }

        if let (Some(last), Layer::BatchNorm1d(next)) = (last_features, &layer) {
            if last != next.features { return Err(invalid(format!("Batch norm features {} does not match previous feature size {last}",next.features))); }
        }

        if let (Some(last), Layer::Conv1d(next)) = (last_features, &layer) {
            if last != next.in_channels { return Err(invalid(format!("Convolution channels {} do not match previous feature count {last}",next.in_channels))); }
        }

        if let (Some(last), Layer::Conv2d(next)) = (last_features, &layer) {
            if last != next.in_channels { return Err(invalid(format!("Convolution channels {} do not match previous feature count {last}",next.in_channels))); }
        }

        self.layers.push(layer);
        Ok(self)
    }
    // ---------------------------------------------- Dense -----------------------------------------------

    /// Adds a Dense layer
    pub fn emplace_dense(&mut self,input_size: usize,output_size: usize,activation: Activation,initializer: Initializer,rng: &mut Rng) -> Result<&mut Self> {
        self.add_layer(Layer::Dense(Dense::new_with_initializers(input_size,output_size,activation,initializer,Initializer::Zeros,rng,self.device)?))
    }

    /// Adds a Dense layer with defaults for Relu, KaimingUniform init
    pub fn add_dense(&mut self,input_size: usize,output_size: usize) -> Result<&mut Self> {
        let mut z = self.rng.clone();
        self.add_layer(Layer::Dense(Dense::new_with_initializers(input_size,output_size,Activation::Relu,Initializer::KaimingUniform,Initializer::Zeros,&mut z,self.device)?))
    }

    /// Adds a preinitialised Dense layer
    pub fn append_dense(&mut self,layer: Dense<T>) -> Result<&mut Self>{
        self.add_layer(Layer::Dense(layer))
    }

    // ---------------------------------------------- Conv2D -----------------------------------------------

    /// Adds a Conv2D layer
    pub fn emplace_conv2d(&mut self,in_channels: usize,out_channels: usize,kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize),activation: Activation,initializer: Initializer,rng: &mut Rng) -> Result<&mut Self> {
        self.add_layer(Layer::Conv2d(Conv2d::new(in_channels,out_channels,kernel,stride,padding,activation,initializer,rng,self.device)?))
    }

    /// Adds a Conv2D layer with defaults for Relu, KaimingUniform init
    pub fn add_conv2d(&mut self,in_channels: usize,out_channels: usize,kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize)) -> Result<&mut Self> {
        let mut z = self.rng.clone();
        self.add_layer(Layer::Conv2d(Conv2d::new(in_channels,out_channels,kernel,stride,padding,Activation::Relu,Initializer::KaimingUniform,&mut z,self.device)?))
    }
    
    /// Adds a preinitialised Conv2D layer
    pub fn append_conv2d(&mut self,layer: Conv2d<T>) -> Result<&mut Self> {
        self.add_layer(Layer::Conv2d(layer))
    }


    // ---------------------------------------------- Conv1D -----------------------------------------------

    /// Adds a Conv1D layer
    pub fn emplace_conv1d(&mut self,in_channels: usize,out_channels: usize,kernel_size: usize,stride: usize,padding: usize,activation: Activation,initializer: Initializer,rng: &mut Rng) -> Result<&mut Self> {
        self.add_layer(Layer::Conv1d(Conv1d::new(in_channels,out_channels,kernel_size,stride,padding,activation,initializer,rng,self.device)?))
    }

    /// Adds a Conv1D layer with defaults for Relu, KaimingUniform init
    pub fn add_conv1d(&mut self,in_channels: usize,out_channels: usize,kernel_size: usize,stride: usize,padding: usize) -> Result<&mut Self> {
        let mut z = self.rng.clone();
        self.add_layer(Layer::Conv1d(Conv1d::new(in_channels,out_channels,kernel_size,stride,padding,Activation::Relu,Initializer::KaimingUniform,&mut z,self.device)?))
    }

    /// Adds a preinitialised Conv1D layer
    pub fn append_conv1d(&mut self,layer: Conv1d<T>) -> Result<&mut Self> {
        self.add_layer(Layer::Conv1d(layer))
    }

    // ---------------------------------------------- Dropout -----------------------------------------------
    /// Adds a Dropout layer
    pub fn emplace_dropout(&mut self, probability: f64) -> Result<&mut Self> { self.add_layer(Layer::Dropout(Dropout::new(probability)?)) }
    /// Adds a Dropout layer
    pub fn add_dropout(&mut self, probability: f64) -> Result<&mut Self> { self.add_layer(Layer::Dropout(Dropout::new(probability)?)) }
    /// Adds a preinitialised Dropout layer
    pub fn append_dropout(&mut self, layer: Dropout) -> Result<&mut Self> {self.add_layer(Layer::Dropout(layer)) }

    // ---------------------------------------------- LayerNorm -----------------------------------------------
    /// Adds a LayerNorm layer
    pub fn emplace_layer_norm(&mut self, features: usize, epsilon: T) -> Result<&mut Self> { self.add_layer(Layer::LayerNorm(LayerNorm::new(features,epsilon,self.device)?)) }
    /// Adds a LayerNorm layer with eps = 1e-5
    pub fn add_layer_norm(&mut self, features: usize) -> Result<&mut Self> { self.add_layer(Layer::LayerNorm(LayerNorm::new(features,T::from_f32(1e-5),self.device)?)) }
    /// Adds a preinitialised LayerNorm layer
    pub fn append_layer_norm(&mut self,layer: LayerNorm<T>) -> Result<&mut Self> { self.add_layer(Layer::LayerNorm(layer)) }

    // ---------------------------------------------- LayerNorm -----------------------------------------------
    /// Adds a BatchNorm1D layer
    pub fn emplace_batch_norm1d(&mut self,features: usize,epsilon: T,momentum: T) -> Result<&mut Self> { self.add_layer(Layer::BatchNorm1d(BatchNorm1d::new(features,epsilon,momentum,self.device)?)) }
    /// Adds a BatchNorm1D layer with eps = 1e-5
    pub fn add_batch_norm1d(&mut self,features: usize,momentum: T) -> Result<&mut Self> { self.add_layer(Layer::BatchNorm1d(BatchNorm1d::new(features,T::from_f32(1e-5),momentum,self.device)?)) }
    /// Adds a preinitialised BatchNorm1D layer
    pub fn append_batch_norm1d(&mut self,layer: BatchNorm1d<T>) -> Result<&mut Self> { self.add_layer(Layer::BatchNorm1d(layer)) }

    // ---------------------------------------------- Prelu -----------------------------------------------
    /// Adds a Prelu layer
    pub fn emplace_prelu(&mut self, features: usize) -> Result<&mut Self> { self.add_layer(Layer::PRelu(PRelu::new(features, self.device)?)) } 
    /// Adds a Prelu layer
    pub fn add_prelu(&mut self, features: usize) -> Result<&mut Self> { self.add_layer(Layer::PRelu(PRelu::new(features, self.device)?)) }   
    /// Adds a preinitialised Prelu layer
    pub fn append_prelu(&mut self,layer: PRelu<T>) -> Result<&mut Self> { self.add_layer(Layer::PRelu(layer)) }


    // ---------------------------------------------- Activation -----------------------------------------------
    /// Adds a Activation layer
    pub fn emplace_activation(&mut self, activation: Activation) -> Result<&mut Self> { self.add_layer(Layer::Activation(activation)) }
    /// Adds a Activation layer
    pub fn add_activation(&mut self, activation: Activation) -> Result<&mut Self> { self.add_layer(Layer::Activation(activation)) }
    /// Adds a Activation layer
    pub fn append_activation(&mut self, activation: Activation) -> Result<&mut Self> { self.add_layer(Layer::Activation(activation)) }
    
    // ---------------------------------------------- Flatten -----------------------------------------------
    /// Adds a Flatten layer
    pub fn emplace_flatten(&mut self) -> Result<&mut Self> { self.add_layer(Layer::Flatten) }
    /// Adds a Flatten layer
    pub fn add_flatten(&mut self) -> Result<&mut Self> { self.add_layer(Layer::Flatten) }
    /// Adds a Flatten layer
    pub fn append_flatten(&mut self) -> Result<&mut Self> { self.add_layer(Layer::Flatten) }

    // ---------------------------------------------- GlobalAvgPool2D -----------------------------------------------
    /// Adds a GlobalAvgPool2D layer
    pub fn emplace_global_avg_pool2d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalAvgPool2d)}
    /// Adds a GlobalAvgPool2D layer
    pub fn add_global_avg_pool2d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalAvgPool2d)}
    /// Adds a GlobalAvgPool2D layer
    pub fn append_global_avg_pool2d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalAvgPool2d)}

    // ---------------------------------------------- GlobalMaxPool2D -----------------------------------------------
    /// Adds a GlobalMaxPool2D layer
    pub fn emplace_global_max_pool2d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalMaxPool2d) }
    /// Adds a GlobalMaxPool2D layer
    pub fn add_global_max_pool2d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalMaxPool2d) }
    /// Adds a GlobalMaxPool2D layer
    pub fn append_global_max_pool2d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalMaxPool2d) }

    // ---------------------------------------------- GlobalAvgPool1D -----------------------------------------------
    /// Adds a GlobalAvgPool1D layer
    pub fn emplace_global_avg_pool1d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalAvgPool1d) }
    /// Adds a GlobalAvgPool1D layer
    pub fn add_global_avg_pool1d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalAvgPool1d) }
    /// Adds a GlobalAvgPool1D layer
    pub fn append_global_avg_pool1d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalAvgPool1d) }

    // ---------------------------------------------- GlobalMaxPool1D -----------------------------------------------
    /// Adds a GlobalMaxPool1D layer
    pub fn emplace_global_max_pool1d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalMaxPool1d) }
    /// Adds a GlobalMaxPool1D layer
    pub fn add_global_max_pool1d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalMaxPool1d) }
    /// Adds a GlobalMaxPool1D layer
    pub fn append_global_max_pool1d(&mut self) -> Result<&mut Self> { self.add_layer(Layer::GlobalMaxPool1d) }

    // ---------------------------------------------- MaxPool1D -----------------------------------------------
    /// Adds a MaxPool1D layer
    pub fn emplace_max_pool1d(&mut self,kernel_size: usize,stride: usize,padding: usize) -> Result<&mut Self> { self.add_layer(Layer::MaxPool1d(MaxPool1d::new(kernel_size,stride,padding)?)) }
    /// Adds a MaxPool1D layer
    pub fn add_max_pool1d(&mut self,kernel_size: usize,stride: usize,padding: usize) -> Result<&mut Self> { self.add_layer(Layer::MaxPool1d(MaxPool1d::new(kernel_size,stride,padding)?)) }
    /// Adds a preinitialised MaxPool1D layer
    pub fn append_max_pool1d(&mut self,layer: MaxPool1d) -> Result<&mut Self> { self.add_layer(Layer::MaxPool1d(layer)) }


    // ---------------------------------------------- AvgPool1D -----------------------------------------------
    /// Adds a AvgPool1D layer
    pub fn emplace_avg_pool1d(&mut self,kernel_size: usize,stride: usize,padding: usize,count_pad: bool) -> Result<&mut Self> { self.add_layer(Layer::AvgPool1d(AvgPool1d::new(kernel_size,stride,padding,count_pad)?)) }
    /// Adds a AvgPool1D layer with count_pad=True
    pub fn add_avg_pool1d(&mut self,kernel_size: usize,stride: usize,padding: usize) -> Result<&mut Self> { self.add_layer(Layer::AvgPool1d(AvgPool1d::new(kernel_size,stride,padding,true)?)) }
    /// Adds a preinitiLised AvgPool1D layer
    pub fn append_avg_pool1d(&mut self,layer: AvgPool1d) -> Result<&mut Self> { self.add_layer(Layer::AvgPool1d(layer)) }

    // ---------------------------------------------- MaxPool2D -----------------------------------------------
    /// Adds a MaxPool2D layer
    pub fn emplace_max_pool2d(&mut self,kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize)) -> Result<&mut Self> { self.add_layer(Layer::MaxPool2d(MaxPool2d::new(kernel, stride, padding)?)) }
    /// Adds a MaxPool2D layer
    pub fn add_max_pool2d(&mut self,kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize)) -> Result<&mut Self> { self.add_layer(Layer::MaxPool2d(MaxPool2d::new(kernel, stride, padding)?)) }
    /// Adds a preinitialised MaxPool2D layer
    pub fn append_max_pool2d(&mut self,layer: MaxPool2d) -> Result<&mut Self> { self.add_layer(Layer::MaxPool2d(layer)) }

    // ---------------------------------------------- AvgPool2D -----------------------------------------------
    /// Adds a AvgPool2D layer
    pub fn emplace_avg_pool2d(&mut self,kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize),count_pad: bool) -> Result<&mut Self> { self.add_layer(Layer::AvgPool2d(AvgPool2d::new(kernel,stride,padding,count_pad)?)) }
    /// Adds a AvgPool2D layer with count_pad=True
    pub fn add_avg_pool2d(&mut self,kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize)) -> Result<&mut Self> { self.add_layer(Layer::AvgPool2d(AvgPool2d::new(kernel,stride,padding,true)?)) }
    /// Adds a preinitialised AvgPool2D layer
    pub fn append_avg_pool2d(&mut self,layer: AvgPool2d) -> Result<&mut Self> { self.add_layer(Layer::AvgPool2d(layer)) }


    // ---------------------------------------------- Residual -----------------------------------------------
    /// Adds a Residual layer
    pub fn emplace_residual(&mut self, layers: Vec<Layer<T>>) -> Result<&mut Self> { self.add_layer(Layer::Residual(Box::new(ResidualBlock::new(layers)?))) }
    /// Adds a Residual layer
    pub fn add_residual(&mut self, layers: Vec<Layer<T>>) -> Result<&mut Self> { self.add_layer(Layer::Residual(Box::new(ResidualBlock::new(layers)?))) }
    /// Adds a preinitialised Residual layer
    pub fn append_residual(&mut self, layer: ResidualBlock<T>) -> Result<&mut Self> { self.add_layer(Layer::Residual(Box::new(layer))) }


}
