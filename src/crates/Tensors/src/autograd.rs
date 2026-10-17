use std::{cell::Cell, sync::{Arc, Mutex, RwLock, atomic::{AtomicUsize, Ordering}}};

use Utilities::device::Device::Device;

use crate::{dtype::Float, error::Result, tensor::Tensor};

#[cfg(feature = "parallel")]
fn join<A: Send, B: Send>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B) { rayon::join(a, b) }
#[cfg(not(feature = "parallel"))]
fn join<A, B>(a: impl FnOnce() -> A, b: impl FnOnce() -> B) -> (A, B) { (a(), b()) }

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
thread_local! { static GRAD_ON: Cell<bool> = Cell::new(true); }

struct GradGuard(bool);

impl Drop for GradGuard { 
    fn drop(&mut self) {
        GRAD_ON.with(|g| g.set(self.0)); 
    } 
}

/// Is computational graph building is on in curr thread
pub fn is_grad_enabled() -> bool { GRAD_ON.with(|g| g.get()) }

/// Run fxn without building computational graph
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _ = GradGuard(GRAD_ON.with(|g| g.replace(false))); // Drop to trigger change
    f()
}

/// Backward pass fxn closure type
/// Takes grad of output and returns grad to parent
/// If no parent no need to autograd them (Doesnt grad on the starting input to the comp graph)
type BackFn<T> = Box<dyn Fn(&Tensor<T>) -> Result<Vec<Option<Tensor<T>>>> + Send + Sync>;


/// Node in computational graph
struct Node<T: Float> {
    id: usize,
    data: RwLock<Tensor<T>>,
    grad: Mutex<Option<Tensor<T>>>,
    requires_grad: bool,
    parents: Vec<Comp<T>>,
    back: Option<BackFn<T>>, // BackFn dependency injection
}

/// Convert to shape by summing along the extra dim until we reach tgt shape
fn unbroadcast<T: Float>(mut g: Tensor<T>, shape: &[usize]) -> Result<Tensor<T>> {
    if g.shape() == shape { return Ok(g); }
    while g.ndim() > shape.len() { g = g.sum(0, false)?; }
    for (d, &s) in shape.iter().enumerate() { if s == 1 && g.shape()[d] != 1 { g = g.sum(d as isize, true)?; } }
    Ok(g)
}

/// Convert to shape
fn expand_to<T: Float>(g: &Tensor<T>, shape: &[usize]) -> Result<Tensor<T>> { g.broadcast_to(shape)?.compact() }


/// Add a dim back
fn keep<T: Float>(g: &Tensor<T>, d: usize, keepdim: bool) -> Result<Tensor<T>> {  if keepdim { Ok(g.clone()) } else { g.unsqueeze(d as isize) } }


/// Moves dim to the last 
fn rows_to_tensor<T: Float>(data: Vec<T>, shape: &[usize], d: usize) -> Result<Tensor<T>> {
    let nd = shape.len();
    // Get all dims apart from the d
    let mut x: Vec<usize> = shape.iter().enumerate().filter(|(i, _)| *i != d).map(|(_, &s)| s).collect();
    x.push(shape[d]);
    let back: Vec<usize> = (0..nd).map(|i| if i < d { i } else if i == d { nd - 1 } else { i - 1 }).collect();
    Tensor::from_vec(data, &x)?.permute(&back)?.compact()
}

/// Even split of grad update taht use min/max btwn 2 vects 
fn split_ratio<T: Float>(a: &Tensor<T>, b: &Tensor<T>, greater: bool) -> Result<Tensor<T>> {
    let h = a.eq_t(b)?.mul_scalar(T::from_f64(0.5))?;
    (if greater { a.gt(b)? } else { a.lt(b)? }).add(&h)
}

/// Ref Counted node handle for a tensor
#[derive(Clone)]
pub struct Comp<T: Float>(Arc<Node<T>>);

impl<T: Float> Comp<T> {

    /// Init a new component from tensor for autograd
    pub fn new(data: Tensor<T>, requires_grad: bool) -> Self {
        Comp(Arc::new(Node { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), data: RwLock::new(data), grad: Mutex::new(None), requires_grad, parents: vec![], back: None }))
    }

    /// Init a new leaf node component from tensor for autograd
    pub fn leaf(data: Tensor<T>) -> Self { Self::new(data, true) }

    /// Init a new component from tensor, that wont autograd
    pub fn constant(data: Tensor<T>) -> Self { Self::new(data, false) }
    pub fn non_leaf(data: Tensor<T>) -> Self { Self::new(data, false) }


    /// Make a comp from op,if no grad then const comp
    pub fn from_op(y: Tensor<T>, parents: Vec<Comp<T>>, back: impl Fn(&Tensor<T>) -> Result<Vec<Option<Tensor<T>>>> + Send + Sync + 'static) -> Self {
        if !is_grad_enabled() || !parents.iter().any(|p| p.0.requires_grad) { return Self::constant(y); }
        Comp(Arc::new(Node { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), data: RwLock::new(y), grad: Mutex::new(None), requires_grad: true, parents, back: Some(Box::new(back)) }))
    }

    /// Single parent op
    pub fn un(&self, y: Tensor<T>, bw: impl Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static) -> Self {
        Self::from_op(y, vec![self.clone()], move |g| Ok(vec![Some(bw(g)?)]))
    }

    /// Two parent ops
    fn bin(&self, other: &Comp<T>, y: Tensor<T>, op_a: impl Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static, op_b: impl Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static) -> Self {
        let (a_is_grad, b_is_grad, shape_a, shape_b) = (self.requires_grad(), other.requires_grad(), self.shape(), other.shape());
        Self::from_op(y, vec![self.clone(), other.clone()], move |g| {
            let (a, b) = match (a_is_grad,b_is_grad) {
                (true, true) => {let (a, b) = join(|| {op_a(g)}, || {op_b(g)}); (Some(a?), Some(b?)) }
                (true, false) => (Some(op_a(g)?), None),
                (false, true) => (None, Some(op_b(g)?)),
                _ => (None, None),
            };
            Ok(vec![a.map(|x| unbroadcast(x, &shape_a)).transpose()?, b.map(|x| unbroadcast(x, &shape_b)).transpose()?])
        })
    }

    /// Accumulated grad of comp 
    pub fn grad(&self) -> Option<Tensor<T>> { self.0.grad.lock().unwrap().clone() }

    /// Clear grad 
    pub fn zero_grad(&self) { *self.0.grad.lock().unwrap() = None; }

    // Checks if autograd comp
    pub fn requires_grad(&self) -> bool { self.0.requires_grad }

    // Check if comp is leaf
    pub fn is_leaf(&self) -> bool { self.0.back.is_none() }

    /// Get const comp from self
    pub fn detach(&self) -> Self { Self::constant(self.data()) }

    /// Get shape
    pub fn shape(&self) -> Vec<usize> { self.0.data.read().unwrap().shape().to_vec() }

    /// Get ndim
    pub fn ndim(&self) -> usize { self.0.data.read().unwrap().ndim() }

    /// Get numel
    pub fn numel(&self) -> usize { self.0.data.read().unwrap().numel() }

    /// Get device
    pub fn device(&self) -> Device { self.0.data.read().unwrap().device() }

    /// Get data
    pub fn data(&self) -> Tensor<T> { self.0.data.read().unwrap().clone() }

    /// Get item
    pub fn item(&self) -> Result<T> { self.data().item() }

    /// Get vec of data
    pub fn to_vec(&self) -> Result<Vec<T>> { self.data().to_vec() }

    /// y=f(x) autograd if f is similar in both directions i.e y = f(x) and x' = f(y')
    pub fn lin(&self, f: impl Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static) -> Result<Self> { Ok(self.un(f(&self.data())?, move |g| f(g))) }

    /// Move to device,return the grad back to og device
    pub fn to_device(&self, dev: Device) -> Result<Self> {
        let (x, src) = (self.data(), self.device());
        let y = x.to_device(dev)?;
        Ok(self.un(y, move |g| g.to_device(src)))
    }

}
