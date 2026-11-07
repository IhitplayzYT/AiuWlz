use Tensors::{dtype::Float, tensor::Tensor};
use Utilities::{device::Device::Device, random::Random::Rng};

use crate::{Initializer, Result, common::invalid, layer::initialize};

/// Vocab lookup table. 
/// Input(Token Id) -> Output(Embedding)
#[derive(Clone)]
pub struct TokenEmbedding<T: Float> {
    pub vocab_sz: usize,
    pub embed_dim: usize,
    pub weights: Tensor<T>,
}

impl<T: Float> TokenEmbedding<T> {

    /// Create a new Tokenembedding layer
    pub fn new(vocab_sz: usize,embed_dim: usize,initializer: Initializer,rng: &mut Rng,device: Device) -> Result<Self> {
        if vocab_sz == 0 || embed_dim == 0 { return Err(invalid("Embedding vocab and dims must be positive")); }
        let weights = initialize(&[vocab_sz, embed_dim],embed_dim,embed_dim,initializer,rng)?.to_device(device)?;
        Ok(Self {vocab_sz,embed_dim,weights})
    }

    /// Fwd pass
    pub fn forward(&self, token_ids: &Tensor<i32>) -> Result<Tensor<T>> {
        if token_ids.device() != self.weights.device() { return Err(invalid("Token and embeddings are on diff devices ")); }
        let ids = validate_token_indices(token_ids, self.vocab_sz)?;
        let mut shape: Vec<isize> = token_ids.shape().iter().map(|&d| d as isize).collect();
        shape.push(self.embed_dim as isize);
        self.weights.index_select(0, &ids)?.reshape(&shape).map_err(Into::into)
    }

    /// Move layer to dev 
    pub fn to_device(&mut self,dev: Device) -> Result<()>{
        self.weights = self.weights.to_device(dev)?;
        Ok(())
    }

    /// Chaining op with specific device 
    pub fn with_device(mut self,dev: Device) -> Result<Self>{
        self.weights = self.weights.to_device(dev)?;
        Ok(self)
    }

}

pub fn validate_token_indices(token_ids: &Tensor<i32>, vocab_sz: usize) -> Result<Vec<usize>> {
    let raw = token_ids.to_vec()?;
    if let Some(&bad) = raw.iter().find(|&&id| id < 0 || id as u128 >= vocab_sz as u128){
        return Err(invalid(format!("Token ID {bad} is outside vocab range [0,{vocab_sz})"))); 
    }
    Ok(raw.into_iter().map(|id| id as usize).collect())
}