use Tensors::{autograd::Comp, dtype::Float, tensor::Tensor};
use Utilities::{device::Device::Device, random::Random::Rng};

use crate::{Initializer, Result, common::invalid, layer::initialize};

/// LSTM layer 
/// Input: [batch, time, features]
#[derive(Clone,Debug)]
pub struct Lstm<T: Float> {
    pub input_sz: usize,
    pub hidden_sz: usize,
    pub w_ih: Tensor<T>, // Input to hidden w  // STores the weigth matrix for each of the 4 gates
    pub w_hh: Tensor<T>, // Hidden to hidden w 
    pub b_ih: Tensor<T>, // STores the bias matrix for each of the 4 gates
    pub b_hh: Tensor<T>, // Hidden ot hidden b 
}
pub type LSTM<T> = Lstm<T>;

impl<T: Float> Lstm<T> {

    /// Create a new Lstm
    pub fn new(input_sz: usize,hidden_sz: usize,initializer: Initializer,rng: &mut Rng,device: Device) -> Result<Self> {
        if input_sz == 0 || hidden_sz == 0 { return Err(invalid("LSTM input and hidden sizes must be non zero")); }
        let w_ih = initialize(&[input_sz, hidden_sz * 4],input_sz,hidden_sz * 4,initializer,rng)?.to_device(device)?;
        let w_hh = initialize(&[hidden_sz, hidden_sz * 4],hidden_sz,hidden_sz * 4,initializer,rng)?.to_device(device)?;
        let b_ih = Tensor::zeros(&[1, hidden_sz * 4]).to_device(device)?;
        let b_hh = Tensor::zeros(&[1, hidden_sz * 4]).to_device(device)?;
        Ok(Self {input_sz,hidden_sz,w_ih,w_hh,b_ih,b_hh})
    }

    /// Fwd pass
    pub fn forward(&self,input: &Tensor<T>) -> Result<Tensor<T>> { Ok(self.forward_graph(&Comp::constant(input.clone()))?.0.data()) }

    /// Internal use comp graph building
    // z = (x_t * w_ih) + b_ih + (h_t * w_hh) + b_hh
    // (z_i, z_f, z_g, z_o) = split(z,4)
    // ig = sigmoid(z_i)   // input gate
    // fg = sigmoid(z_f)   // forget gate
    // cg = tanh(z_g)      // cell gate
    // og = sigmoid(z_o)   // output gate
    // c_(t+1) = fg * c_t + ig * cg 
    // h_(t+1) = og * tanh(c_(t+1)) 
    pub fn forward_graph(&self,input: &Comp<T>) -> Result<(Comp<T>,Vec<Comp<T>>)> {
        if input.ndim() != 3 || input.shape()[1] == 0 || input.shape()[2] != self.input_sz { return Err(invalid(format!("LSTM expects [batch, positive time, {}], got {:?}",self.input_sz,input.shape()))); }
        input.data().same_device(&self.w_ih)?;
        let w_ih = Comp::leaf(self.w_ih.clone());
        let w_hh = Comp::leaf(self.w_hh.clone());
        let b_ih = Comp::leaf(self.b_ih.clone());
        let b_hh = Comp::leaf(self.b_hh.clone());
        let batch = input.shape()[0];

        let mut hidden = Comp::constant(Tensor::zeros(&[batch,self.hidden_sz]).to_device(self.w_ih.device())?);
        let mut cell = Comp::constant(Tensor::zeros(&[batch,self.hidden_sz]).to_device(self.w_ih.device())?);
        let mut outputs = Vec::with_capacity(input.shape()[1]);
        for step in input.split(1,1)? {
            let x = step.squeeze(1)?;
            let g = x.matmul(&w_ih)?.add(&b_ih)?.add(&hidden.matmul(&w_hh)?)?.add(&b_hh)?;
            let mut gates = g.split(1,self.hidden_sz)?.into_iter(); // since g is [batch_size, 4 * hidden_state] and each gate uses [batch.hidden_state]
            let ig = gates.next().ok_or_else(|| invalid("LSTM input gate missing"))?.sigmoid()?;
            let fg = gates.next().ok_or_else(|| invalid("LSTM forget gate missing"))?.sigmoid()?;
            let cg = gates.next().ok_or_else(|| invalid("LSTM cell gate missing"))?.tanh()?;
            let og = gates.next().ok_or_else(|| invalid("LSTM output gate missing"))?.sigmoid()?;
            cell = fg.mul(&cell)?.add(&ig.mul(&cg)?)?;
            hidden = og.mul(&cell.tanh()?)?;
            outputs.push(hidden.unsqueeze(1)?);
        }
        let outputs = outputs.iter().collect::<Vec<_>>();
        Ok((Comp::cat(&outputs,1)?,vec![w_ih,w_hh,b_ih,b_hh]))
    }

    /// Set params
    pub fn set_parameters(&mut self,params: &[Tensor<T>]) -> Result<()> {
        if params.len() != 4 || params[0].shape() != self.w_ih.shape() || params[1].shape() != self.w_hh.shape() || params[2].shape() != self.b_ih.shape() || params[3].shape() != self.b_hh.shape() { return Err(invalid("LSTM parameter shape mismatch")); }
        for param in params { param.same_device(&self.w_ih)?; }
        if params.iter().any(|param| param.device() != self.w_ih.device()) { return Err(invalid("LSTM parameters must be on the layer device")); }
        self.w_ih = params[0].clone();
        self.w_hh = params[1].clone(); 
        self.b_ih = params[2].clone(); 
        self.b_hh = params[3].clone();
        Ok(())
    }

    /// Move layer to dev
    pub fn to_device(&mut self,device: Device) -> Result<()> {
        self.w_ih = self.w_ih.to_device(device)?; 
        self.w_hh = self.w_hh.to_device(device)?;
        self.b_ih = self.b_ih.to_device(device)?; 
        self.b_hh = self.b_hh.to_device(device)?;
        Ok(())
    }

    /// No of trainable params
    pub fn param_count(&self) -> usize { self.hidden_sz * 4 * (self.input_sz + self.hidden_sz + 2) }
}

/// GRU layer
/// Input: [batch, time, features]
#[derive(Clone,Debug)]
pub struct Gru<T: Float> {
    pub input_sz: usize,
    pub hidden_sz: usize,
    pub w_ih: Tensor<T>,
    pub w_hh: Tensor<T>,
    pub b_ih: Tensor<T>,
    pub b_hh: Tensor<T>,
}
pub type GRU<T> = Gru<T>;

impl<T: Float> Gru<T> {

    /// Create a new GRU
    pub fn new(input_sz: usize,hidden_sz: usize,initializer: Initializer,rng: &mut Rng,device: Device) -> Result<Self> {
        if input_sz == 0 || hidden_sz == 0 { return Err(invalid("GRU input and hidden sizes must be positive")); }
        let w_ih = initialize(&[input_sz, hidden_sz * 3],input_sz,hidden_sz * 3,initializer,rng)?.to_device(device)?;
        let w_hh = initialize(&[hidden_sz, hidden_sz * 3],hidden_sz,hidden_sz * 3,initializer,rng)?.to_device(device)?;
        let b_ih = Tensor::zeros(&[1, hidden_sz * 3]).to_device(device)?;
        let b_hh = Tensor::zeros(&[1, hidden_sz * 3]).to_device(device)?;
        Ok(Self {input_sz,hidden_sz,w_ih,w_hh,b_ih,b_hh})
    }

    /// Fwd pass
    pub fn forward(&self,input: &Tensor<T>) -> Result<Tensor<T>> { Ok(self.forward_graph(&Comp::constant(input.clone()))?.0.data()) }

    /// Internal use comp graph building
    /// z_ih = (x_t * w_ih) + b_ih
    /// z_hh = (h_t * w_hh) + b_hh
    /// (x_r, x_z, x_n) = split(z_ih, 3)
    /// (h_r, h_z, h_n) = split(z_hh, 3)
    /// rg = sigmoid(x_r + h_r)       // reset gate
    /// ug = sigmoid(x_z + h_z)       // update gate
    /// cg = tanh(x_n + rg * h_n)     // Cand hidden state
    /// h_(t+1) = (1 - ug) * cg + ug * h_t 
    pub fn forward_graph(&self,input: &Comp<T>) -> Result<(Comp<T>,Vec<Comp<T>>)> {
        if input.ndim() != 3 || input.shape()[1] == 0 || input.shape()[2] != self.input_sz { return Err(invalid(format!("GRU expects [batch, positive time, {}], got {:?}",self.input_sz,input.shape()))); }
        input.data().same_device(&self.w_ih)?;
        let w_ih = Comp::leaf(self.w_ih.clone());
        let w_hh = Comp::leaf(self.w_hh.clone());
        let b_ih = Comp::leaf(self.b_ih.clone());
        let b_hh = Comp::leaf(self.b_hh.clone());
        let batch = input.shape()[0];
        let mut hidden = Comp::constant(Tensor::zeros(&[batch,self.hidden_sz]).to_device(self.w_ih.device())?);
        let mut outputs = Vec::with_capacity(input.shape()[1]);
        for step in input.split(1,1)? {
            let x = step.squeeze(1)?;
            let input_gates = x.matmul(&w_ih)?.add(&b_ih)?;
            let hidden_gates = hidden.matmul(&w_hh)?.add(&b_hh)?;
            let mut input_gates = input_gates.split(1,self.hidden_sz)?.into_iter();
            let mut hidden_gates = hidden_gates.split(1,self.hidden_sz)?.into_iter();
            let rg = input_gates.next().ok_or_else(|| invalid("GRU reset gate missing"))?.add(&hidden_gates.next().ok_or_else(|| invalid("GRU reset gate missing"))?)?.sigmoid()?;
            let ug = input_gates.next().ok_or_else(|| invalid("GRU update gate missing"))?.add(&hidden_gates.next().ok_or_else(|| invalid("GRU update gate missing"))?)?.sigmoid()?;
            let cg = input_gates.next().ok_or_else(|| invalid("GRU candidate gate missing"))?;
            let cg_input = hidden_gates.next().ok_or_else(|| invalid("GRU candidate gate missing"))?;
            let candidate = cg_input.add(&rg.mul(&cg)?)?.tanh()?;
            hidden = ug.mul(&hidden)?.add(&ug.mul_scalar(-T::one())?.add_scalar(T::one())?.mul(&candidate)?)?;
            outputs.push(hidden.unsqueeze(1)?);
        }
        let outputs = outputs.iter().collect::<Vec<_>>();
        Ok((Comp::cat(&outputs,1)?,vec![w_ih,w_hh,b_ih,b_hh]))
    }

    /// Set params
    pub fn set_parameters(&mut self,params: &[Tensor<T>]) -> Result<()> {
        if params.len() != 4 || params[0].shape() != self.w_ih.shape() || params[1].shape() != self.w_hh.shape() || params[2].shape() != self.b_ih.shape() || params[3].shape() != self.b_hh.shape() { return Err(invalid("GRU parameter shape mismatch")); }
        for param in params { param.same_device(&self.w_ih)?; }
        if params.iter().any(|param| param.device() != self.w_ih.device()) { return Err(invalid("GRU parameters must be on the layer device")); }
        self.w_ih = params[0].clone(); 
        self.w_hh = params[1].clone(); 
        self.b_ih = params[2].clone(); 
        self.b_hh = params[3].clone();
        Ok(())
    }

    /// Move layer to dev
    pub fn to_device(&mut self,device: Device) -> Result<()> {
        self.w_ih = self.w_ih.to_device(device)?; 
        self.w_hh = self.w_hh.to_device(device)?;
        self.b_ih = self.b_ih.to_device(device)?; 
        self.b_hh = self.b_hh.to_device(device)?;
        Ok(())
    }

    /// Total no of params
    pub fn param_count(&self) -> usize { self.hidden_sz * 3 * (self.input_sz + self.hidden_sz + 2) }
}
