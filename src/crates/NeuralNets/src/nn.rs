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
                Layer::ConvTranspose1d(x) => return Some(x.out_channels),
                Layer::ConvTranspose2d(x) => return Some(x.out_channels),
                Layer::GlobalAvgPool1d | Layer::GlobalMaxPool1d | Layer::GlobalAvgPool2d | Layer::GlobalMaxPool2d => {}
                Layer::LayerNorm(x) => return Some(x.features),
                Layer::RMSNorm(x) => return Some(x.features),
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
                Layer::ConvTranspose1d(d) => {last_features = Some(d.out_channels);break;},
                Layer::ConvTranspose2d(d) => {last_features = Some(d.out_channels);break;},
                Layer::LayerNorm(n) => {last_features = Some(n.features);break;},
                Layer::RMSNorm(n) => {last_features = Some(n.features);break;},
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

        if let (Some(last), Layer::RMSNorm(next)) = (last_features, &layer) {
            if last != next.features { return Err(invalid(format!("RMS norm features {} does not match previous feature size {last}",next.features))); }
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

        if let (Some(last), Layer::ConvTranspose1d(next)) = (last_features, &layer) {
            if last != next.in_channels { return Err(invalid(format!("Convolution channels {} do not match previous feature count {last}",next.in_channels))); }
        }

        if let (Some(last), Layer::ConvTranspose2d(next)) = (last_features, &layer) {
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

    // ---------------------------------------------- ConvTranspose2D -----------------------------------------------
    /// Adds a ConvTranspose2D layer
    pub fn emplace_conv_transpose2d(&mut self,in_channels: usize,out_channels: usize,kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize),activation: Activation,initializer: Initializer,rng: &mut Rng) -> Result<&mut Self> {
        self.add_layer(Layer::ConvTranspose2d(ConvTranspose2d::new(in_channels,out_channels,kernel,stride,padding,activation,initializer,rng,self.device)?))
    }
    /// Adds a ConvTranspose2D layer
    pub fn add_conv_transpose2d(&mut self,in_channels: usize,out_channels: usize,kernel: (usize, usize),stride: (usize, usize),padding: (usize, usize)) -> Result<&mut Self> {
        let mut z = self.rng.clone();
        self.add_layer(Layer::ConvTranspose2d(ConvTranspose2d::new(in_channels,out_channels,kernel,stride,padding,Activation::Relu,Initializer::KaimingUniform,&mut z,self.device)?))
    }
    /// Adds a preinitialised ConvTranspose2D layer
    pub fn append_conv_transpose2d(&mut self,layer: ConvTranspose2d<T>) -> Result<&mut Self> { self.add_layer(Layer::ConvTranspose2d(layer)) }

    // ---------------------------------------------- ConvTranspose1D -----------------------------------------------
    /// Adds a ConvTranspose1D layer
    pub fn emplace_conv_transpose1d(&mut self,in_channels: usize,out_channels: usize,kernel_size: usize,stride: usize,padding: usize,activation: Activation,initializer: Initializer,rng: &mut Rng) -> Result<&mut Self> {
        self.add_layer(Layer::ConvTranspose1d(ConvTranspose1d::new(in_channels,out_channels,kernel_size,stride,padding,activation,initializer,rng,self.device)?))
    }
    /// Adds a ConvTranspose1D layer
    pub fn add_conv_transpose1d(&mut self,in_channels: usize,out_channels: usize,kernel_size: usize,stride: usize,padding: usize) -> Result<&mut Self> {
        let mut z = self.rng.clone();
        self.add_layer(Layer::ConvTranspose1d(ConvTranspose1d::new(in_channels,out_channels,kernel_size,stride,padding,Activation::Relu,Initializer::KaimingUniform,&mut z,self.device)?))
    }
    /// Adds a preinitialised ConvTranspose1D layer
    pub fn append_conv_transpose1d(&mut self,layer: ConvTranspose1d<T>) -> Result<&mut Self> { self.add_layer(Layer::ConvTranspose1d(layer)) }

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

    // ---------------------------------------------- RMSNorm -----------------------------------------------
    /// Adds a RMSNorm layer
    pub fn emplace_rms_norm(&mut self,features: usize,epsilon: T) -> Result<&mut Self> { self.add_layer(Layer::RMSNorm(RMSNorm::new(features,epsilon,self.device)?)) }
    /// Adds a RMSNorm layer with eps = 1e-5
    pub fn add_rms_norm(&mut self,features: usize) -> Result<&mut Self> { self.add_layer(Layer::RMSNorm(RMSNorm::new(features,T::from_f32(1e-5),self.device)?)) }
    /// Adds a preinitialised RMSNorm layer
    pub fn append_rms_norm(&mut self,layer: RMSNorm<T>) -> Result<&mut Self> { self.add_layer(Layer::RMSNorm(layer)) }

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

    // ---------------------------------------------- Identity -----------------------------------------------
    /// Adds a Identity layer
    pub fn emplace_identity(&mut self) -> Result<&mut Self> { self.add_layer(Layer::Identity(Identity::new())) }
    /// Adds a Identity layer
    pub fn add_identity(&mut self) -> Result<&mut Self> { self.add_layer(Layer::Identity(Identity::new())) }
    /// Adds a preinitilised Identity layer
    pub fn append_identity(&mut self,layer: Identity) -> Result<&mut Self> { self.add_layer(Layer::Identity(layer)) }


    // ---------------------------------------------- Unflatten -----------------------------------------------
    /// Adds a Unflatten layer
    pub fn emplace_unflatten(&mut self,dim: isize,unflattened_size: Vec<usize>) -> Result<&mut Self> { self.add_layer(Layer::Unflatten(Unflatten::new(dim,unflattened_size)?)) }
    /// Adds a Unflatten layer
    pub fn add_unflatten(&mut self,dim: isize,unflattened_size: Vec<usize>) -> Result<&mut Self> { self.add_layer(Layer::Unflatten(Unflatten::new(dim,unflattened_size)?)) }
    /// Adds a preintialised Unflatten layer
    pub fn append_unflatten(&mut self,layer: Unflatten) -> Result<&mut Self> { self.add_layer(Layer::Unflatten(layer)) }

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

    // ---------------------------------------------- AdaptiveMaxPool1D -----------------------------------------------
    /// Adds an AdaptiveMaxPool1D layer
    pub fn emplace_adaptive_max_pool1d(&mut self,output_size: usize) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveMaxPool1d(AdaptiveMaxPool1d::new(output_size)?)) }
    /// Adds an AdaptiveMaxPool1D layer
    pub fn add_adaptive_max_pool1d(&mut self,output_size: usize) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveMaxPool1d(AdaptiveMaxPool1d::new(output_size)?)) }
    /// Adds a preinitialised AdaptiveMaxPool1D layer
    pub fn append_adaptive_max_pool1d(&mut self,layer: AdaptiveMaxPool1d) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveMaxPool1d(layer)) }

    // ---------------------------------------------- AdaptiveAvgPool1D -----------------------------------------------
    /// Adds an AdaptiveAvgPool1D layer
    pub fn emplace_adaptive_avg_pool1d(&mut self,output_size: usize) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveAvgPool1d(AdaptiveAvgPool1d::new(output_size)?)) }
    /// Adds an AdaptiveAvgPool1D layer
    pub fn add_adaptive_avg_pool1d(&mut self,output_size: usize) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveAvgPool1d(AdaptiveAvgPool1d::new(output_size)?)) }
    /// Adds a preinitialised AdaptiveAvgPool1D layer
    pub fn append_adaptive_avg_pool1d(&mut self,layer: AdaptiveAvgPool1d) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveAvgPool1d(layer)) }

    // ---------------------------------------------- AdaptiveMaxPool2D -----------------------------------------------
    /// Adds an AdaptiveMaxPool2D layer
    pub fn emplace_adaptive_max_pool2d(&mut self,output_size: (usize,usize)) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveMaxPool2d(AdaptiveMaxPool2d::new(output_size)?)) }
    /// Adds an AdaptiveMaxPool2D layer
    pub fn add_adaptive_max_pool2d(&mut self,output_size: (usize,usize)) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveMaxPool2d(AdaptiveMaxPool2d::new(output_size)?)) }
    /// Adds a preinitialised AdaptiveMaxPool2D layer
    pub fn append_adaptive_max_pool2d(&mut self,layer: AdaptiveMaxPool2d) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveMaxPool2d(layer)) }

    // ---------------------------------------------- AdaptiveAvgPool2D -----------------------------------------------
    /// Adds an AdaptiveAvgPool2D layer
    pub fn emplace_adaptive_avg_pool2d(&mut self,output_size: (usize,usize)) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveAvgPool2d(AdaptiveAvgPool2d::new(output_size)?)) }
    /// Adds an AdaptiveAvgPool2D layer
    pub fn add_adaptive_avg_pool2d(&mut self,output_size: (usize,usize)) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveAvgPool2d(AdaptiveAvgPool2d::new(output_size)?)) }
    /// Adds a preinitialised AdaptiveAvgPool2D layer
    pub fn append_adaptive_avg_pool2d(&mut self,layer: AdaptiveAvgPool2d) -> Result<&mut Self> { self.add_layer(Layer::AdaptiveAvgPool2d(layer)) }


    // ---------------------------------------------- Residual -----------------------------------------------
    /// Adds a Residual layer
    pub fn emplace_residual(&mut self, layers: Vec<Layer<T>>) -> Result<&mut Self> { self.add_layer(Layer::Residual(Box::new(ResidualBlock::new(layers)?))) }
    /// Adds a Residual layer
    pub fn add_residual(&mut self, layers: Vec<Layer<T>>) -> Result<&mut Self> { self.add_layer(Layer::Residual(Box::new(ResidualBlock::new(layers)?))) }
    /// Adds a preinitialised Residual layer
    pub fn append_residual(&mut self, layer: ResidualBlock<T>) -> Result<&mut Self> { self.add_layer(Layer::Residual(Box::new(layer))) }

    /// Inference fwd
    /// Dropout layer is disabled and layernorm use running stats 
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        self.validate_input(input)?;
        let mut value = input.clone();
        for layer in &self.layers { value = layer.forward(&value)?; }
        Ok(value)
    }

    /// Same as forward
    pub fn predict(&self, input: &Tensor<T>) -> Result<Tensor<T>> { self.forward(input) }

    /// Predict the class
    pub fn predict_classes(&self, input: &Tensor<T>) -> Result<Tensor<i32>> {
        if self.output_size().unwrap_or(0) < 2 {return Err(invalid("predict_classes require at least 2 outputs"));}
        Ok(self.forward(input)?.argmax(-1, false)?)
    }

    /// Returns the accuracy of predictions of the network for the input
    pub fn accuracy(&self, input: &Tensor<T>, target: &Tensor<T>) -> Result<f64> {
        let pred = self.predict_classes(input)?;
        let exp = target.argmax(-1, false)?;
        Ok(pred.eq_t(&exp)?.sum_all()? as f64 / input.shape()[0] as f64)
    }

    /// Run a batch of integer token sequences with embedding.
    pub fn forward_tokens(&self, token_ids: &Tensor<i32>) -> Result<Tensor<T>> {
        let embedding = self.embedding.as_ref().ok_or_else(|| invalid("No embedding layer found"))?;
        if token_ids.ndim() != 2 || token_ids.shape()[0] == 0 || token_ids.shape()[1] == 0 { return Err(invalid("Token input must have shape [batch, sequence] with non zero dims")); }
        if token_ids.device() != self.device { return Err(invalid("Token id and model params must be on same dev")); }
        // each batch can contain a sequence of token ids
        let (batch,seq) = (token_ids.shape()[0],token_ids.shape()[1]);
        let embedded = embedding.forward(token_ids)?.reshape(&[(batch * seq) as isize,embedding.embed_dim as isize])?; // Make proper sized lookup table for each token
        let output = if self.layers.is_empty() { embedded } else {self.forward(&embedded)? };
        let mut shape = vec![batch as isize, seq as isize];
        shape.extend(output.shape()[1..].iter().map(|&d| d as isize));
        // Reshape into the [batch,seq,...] shape
        output.reshape(&shape).map_err(Into::into)
    }

    /// Same as forward_tokens
    pub fn predict_tokens(&self, token_ids: &Tensor<i32>) -> Result<Tensor<T>> { self.forward_tokens(token_ids) }

    /// Forward, autograd backward, then optim update for one mini-batch.
    pub fn train_on_batch(&mut self,input: &Tensor<T>,target: &Tensor<T>,loss: Loss,optim: &mut Optimizer<T>) -> Result<T> {
        self.validate_dataset_pair(input, target)?;
        self.train_graph_on_batch(Comp::constant(input.clone()),target,loss,optim,Vec::new())
    }

    /// Train 1 batch of token id with shape [batch, sequence, ...] or preflattened [batch * sequence, ...]
    pub fn train_tokens_on_batch(&mut self,token_ids: &Tensor<i32>,target: &Tensor<T>,loss: Loss,optim: &mut Optimizer<T>) -> Result<T> {
        let embedding = self.embedding.as_ref().ok_or_else(|| invalid("No embedding layer found"))?;
        if token_ids.ndim() != 2 || token_ids.shape()[0] == 0 || token_ids.shape()[1] == 0 { return Err(invalid("Input must have shape: [batch, sequence] with non zero dims")); }
        if token_ids.device() != self.device { return Err(invalid("Token id and model have to be on same device")); }
        if token_ids.device() != embedding.weights.device() { return Err(invalid("Token ids and embedding weight have to be on same device")); }
        let (batch, seq) = (token_ids.shape()[0], token_ids.shape()[1]);
        let indices = validate_token_indices(token_ids, embedding.vocab_sz)?;
        let weights = embedding.weights.clone();
        let embed_dim = embedding.embed_dim;
        let pfx = [batch, seq];
        let target = if target.shape().starts_with(&pfx) {
            let mut flat_shape = vec![(batch * seq) as isize];
            flat_shape.extend(target.shape()[2..].iter().map(|&d| d as isize));
            target.reshape(&flat_shape)?
        } else if target.shape().first() == Some(&(batch * seq)) { target.clone() } else { return Err(invalid("Targets must start with [batch, seq] or flattened token count [batch * seq,...]")); };
        target.same_device(&weights)?;
        let embedding_node = Comp::leaf(weights);
        let graph = embedding_node.index_select(0, &indices)?.reshape(&[(batch * seq) as isize, embed_dim as isize])?;
        self.train_graph_on_batch(graph, &target, loss, optim, vec![embedding_node])
    }

    pub fn train_graph_on_batch(&mut self,mut graph: Comp<T>,target: &Tensor<T>,loss: Loss,optim: &mut Optimizer<T>,mut param_comps: Vec<Comp<T>>) -> Result<T> {
        for layer in &mut self.layers {
            let (next, params) = layer.forward_graph(&graph, true, &mut self.rng)?;
            graph = next;
            param_comps.extend(params);
        }
        let loss_node = loss_comp(&graph, target, loss)?;
        let loss_data = loss_node.item()?;
        loss_node.backward()?;
        let mut grads = Vec::with_capacity(param_comps.len());
        for node in param_comps{ grads.push(node.grad().ok_or_else(|| invalid("autograd did not produce a parameter gradient"))?); }
        let mut params = self.params();
        optim.update(&mut params, &grads)?;
        self.set_parameters(&params)?;
        Ok(loss_data)
    }


    /// Validates if input is proper to the sequential api
    pub fn validate_input(&self, input: &Tensor<T>) -> Result<()> {
        if self.layers.is_empty() { return Err(invalid("Network is empty")); }
        if input.ndim() < 2 || input.shape()[0] == 0 { return Err(invalid("Network must have batch and one param dim atleast")); }
        if input.device() != self.device { return Err(invalid(format!("Input on {}, but model is on {}",input.device(),self.device))); }
        Ok(())
    }

    /// Validates input and target pair
    pub fn validate_dataset_pair(&self, input: &Tensor<T>, target: &Tensor<T>) -> Result<()> {
        self.validate_input(input)?;
        if target.shape().first() != input.shape().first() { return Err(invalid("No of samples/batches isnt same for input and target")); }
        input.same_device(target)?;
        Ok(())
    }


}
