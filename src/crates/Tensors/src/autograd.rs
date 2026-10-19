use std::{cell::Cell, collections::{HashMap, HashSet}, fmt, sync::{Arc, Mutex, RwLock, atomic::{AtomicUsize, Ordering}}};

use Utilities::{Utilities::Quartile, device::Device::Device};

use crate::{dtype::Float, error::{Result, TensorError}, layout::{normalize_dim, normalize_dim_inclusive}, par, tensor::Tensor};

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

pub fn with_grad<R>(f: impl FnOnce() -> R) -> R {
    let _guard = GradGuard(GRAD_ON.with(|g| g.replace(true)));
    f()
}


/// Backward pass fxn closure type
/// Takes grad of output and returns grad to parent
/// If no parent no need to autograd them (Doesnt grad on the starting input to the comp graph)
type BackFn<T> = Box<dyn Fn(&Tensor<T>) -> Result<Vec<Option<Tensor<T>>>> + Send + Sync>;


/// Node in computational graph
pub struct Node<T: Float> {
    pub id: usize,
    pub data: RwLock<Tensor<T>>,
    pub grad: Mutex<Option<Tensor<T>>>,
    pub requires_grad: bool,
    pub parents: Vec<Comp<T>>,
    pub back: Option<BackFn<T>>, // BackFn dependency injection
}

/// Convert to shape by summing along the extra dim until we reach tgt shape
pub fn unbroadcast<T: Float>(mut g: Tensor<T>, shape: &[usize]) -> Result<Tensor<T>> {
    if g.shape() == shape { return Ok(g); }
    while g.ndim() > shape.len() { g = g.sum(0, false)?; }
    for (d, &s) in shape.iter().enumerate() { if s == 1 && g.shape()[d] != 1 { g = g.sum(d as isize, true)?; } }
    Ok(g)
}

/// Convert to shape
pub fn expand_to<T: Float>(g: &Tensor<T>, shape: &[usize]) -> Result<Tensor<T>> { g.broadcast_to(shape)?.compact() }


/// Add a dim back
pub fn keep<T: Float>(g: &Tensor<T>, d: usize, keepdim: bool) -> Result<Tensor<T>> {  if keepdim { Ok(g.clone()) } else { g.unsqueeze(d as isize) } }


/// Moves dim to the last 
pub fn rows_to_tensor<T: Float>(data: Vec<T>, shape: &[usize], d: usize) -> Result<Tensor<T>> {
    let nd = shape.len();
    // Get all dims apart from the d
    let mut x: Vec<usize> = shape.iter().enumerate().filter(|(i, _)| *i != d).map(|(_, &s)| s).collect();
    x.push(shape[d]);
    let back: Vec<usize> = (0..nd).map(|i| if i < d { i } else if i == d { nd - 1 } else { i - 1 }).collect();
    Tensor::from_vec(data, &x)?.permute(&back)?.compact()
}

/// Even split of grad update taht use min/max btwn 2 vects 
pub fn split_ratio<T: Float>(a: &Tensor<T>, b: &Tensor<T>, greater: bool) -> Result<Tensor<T>> {
    let h = a.eq_t(b)?.mul_scalar(T::from_f64(0.5))?;
    (if greater { a.gt(b)? } else { a.lt(b)? }).add(&h)
}

/// Ref Counted node handle for a tensor
#[derive(Clone)]
pub struct Comp<T: Float>(Arc<Node<T>>);

impl<T: Float> fmt::Debug for Comp<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Comp(data={}, requires_grad={}, has_grad={}, leaf={})", self.data(), self.requires_grad(), self.grad().is_some(), self.is_leaf())
    }
}


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

    /// Comp for single parent ops
    pub fn un(&self, y: Tensor<T>, bw: impl Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static) -> Self {
        Self::from_op(y, vec![self.clone()], move |g| Ok(vec![Some(bw(g)?)]))
    }

    /// Comp for two parent ops
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

    // Set data
    pub fn set_data(&self, t: Tensor<T>) -> Result<()> {
        if t.shape() != &self.shape()[..] { return Err(TensorError::ShapeMismatch { op: "set_data", lhs: self.shape(), rhs: t.shape().to_vec() }); }
        *self.0.data.write().unwrap() = t;
        Ok(())
    }

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

    /// SGD: w -= lr * grad
    pub fn sgd(&self, lr: T) -> Result<()> {
        if let Some(g) = self.grad() { self.set_data(self.data().sub(&g.mul_scalar(lr)?)?)?; }
        Ok(())
    }

    /// Topo sort comp graph
    // Used ai to make
    pub fn topo_sort(&self) -> Vec<Comp<T>> {
        let mut ret = Vec::new();
        let mut seen= HashSet::new();
        let mut stack= vec![(self.clone(), 0)];
        seen.insert(self.0.id);
        while let Some((v, i)) = stack.pop() {
            if i < v.0.parents.len() {
                let p = v.0.parents[i].clone();
                stack.push((v,i+1));
                if p.0.requires_grad && seen.insert(p.0.id) { stack.push((p, 0)); }
            } else {
                ret.push(v);     
            }
        }
        ret
    } 

    /// Accumulate grad bhetween slef and another tensor
    pub fn acc_grad(&self, g: Tensor<T>) -> Result<()> {
        let mut s = self.0.grad.lock().unwrap();
        let ret = match s.take() { Some(old) => old.add(&g)?, None => g };
        *s = Some(ret);
        Ok(())
    }

    /// Backprop
    pub fn backward(&self) -> Result<()> {
        if self.numel() != 1 { return Err(TensorError::InvalidShape(format!("backward() needs a single element Comp, got {:?}, use backward_with", self.shape()))); }
        self.backward_with(self.data().ones_like())
    }

    /// Backward pass using a init as b
    pub fn backward_with(&self, b: Tensor<T>) -> Result<()> {
        if !self.0.requires_grad { return Err(TensorError::Custom("Cannot backprop a comp without requires_grad".into())); }
        if b.shape() != self.shape().as_slice() { return Err(TensorError::ShapeMismatch { op: "backward_with", lhs: self.shape(), rhs: b.shape().to_vec() }); }
        let mut grads = HashMap::new();
        grads.insert(self.0.id, b);
        for v in self.topo_sort().into_iter().rev() {
            let Some(g) = grads.remove(&v.0.id) else { continue };
            match &v.0.back {
                None => v.acc_grad(g)?, // If no bakprop fn simply acc_grad
                Some(back) => {
                    let gs = back(&g)?;
                    for (p, pg) in v.0.parents.iter().zip(gs) {
                        // Accumulate the grads into parents
                        let Some(pg) = pg else { continue };
                        if !p.0.requires_grad { continue; }
                        if pg.shape() != &p.shape()[..] { return Err(TensorError::ShapeMismatch { op: "backward_with", lhs: p.shape(), rhs: pg.shape().to_vec() }); }
                        let merged = match grads.remove(&p.0.id) { Some(old) => old.add(&pg)?, None => pg };
                        grads.insert(p.0.id, merged);
                    }
                }
            }
        }
        Ok(())
    }

    /// Custom univariable op with supplied fn and derivative
    pub fn apply(&self, f: impl Fn(T) -> T + Sync + Send, df: impl Fn(T) -> T + Sync + Send + 'static) -> Result<Self> {
        let x = self.data();
        Ok(self.un(x.map(f)?, move |g| g.mul(&x.map(&df)?)))
    }

    /// Make a add comp 
    /// y=a+b
    /// dy/da = 1
    /// dy/db = 1
    pub fn add(&self, o: &Comp<T>) -> Result<Self> {
        Ok(self.bin(o, self.data().add(&o.data())?, |g| Ok(g.clone()), |g| Ok(g.clone()))) 
    }

    /// Make a sub comp 
    /// y=a-b
    /// dy/da = 1
    /// dy/db = -1
    pub fn sub(&self, o: &Comp<T>) -> Result<Self> { 
        Ok(self.bin(o, self.data().sub(&o.data())?, |g| Ok(g.clone()), |g| g.neg())) 
    }

    /// Make a mul comp 
    /// y=a*b
    /// dy/da = b
    /// dy/db = a
    pub fn mul(&self, o: &Comp<T>) -> Result<Self> {
        let (a, b) = (self.data(), o.data());
        Ok(self.bin(o,  a.mul(&b)?, move |g| g.mul(&b), move |g| g.mul(&a)))
    }
    
    /// Make a div comp 
    /// y=a/b
    /// dy/da = 1/b 
    /// dy/db = -a/b^2
    pub fn div(&self, o: &Comp<T>) -> Result<Self> {
        let (a, b) = (self.data(), o.data());
        let y = a.div(&b)?;
        let b2 = b.clone();
        Ok(self.bin(o, y, move |g| g.div(&b), move |g| g.mul(&a)?.div(&b2.square()?)?.neg()))
    }

    /// Element wise max, share the grad equally
    pub fn maximum(&self, o: &Comp<T>) -> Result<Self> {
        let (a, b) = (self.data(), o.data());
        let y = a.maximum(&b)?;
        let (a2, b2) = (a.clone(), b.clone());
        Ok(self.bin(o, y, move |g| g.mul(&split_ratio(&a, &b, true)?), move |g| g.mul(&split_ratio(&b2, &a2, true)?)))
    }

    /// Element wise min, share the grad equally
    pub fn minimum(&self, o: &Comp<T>) -> Result<Self> {
        let (a, b) = (self.data(), o.data());
        let y = a.minimum(&b)?;
        let (a2, b2) = (a.clone(), b.clone());
        Ok(self.bin(o, y, move |g| g.mul(&split_ratio(&a, &b, false)?), move |g| g.mul(&split_ratio(&b2, &a2, false)?)))
    }

    /// Make a pow comp
    /// y = a ^ b
    /// dy/da = b * a ^ (b-1)
    /// dy/db = a^b * ln(a)
    pub fn pow(&self, e: &Comp<T>) -> Result<Self> {
        let (x, p) = (self.data(), e.data());
        let y = x.pow(&p)?;
        let (x2, y2) = (x.clone(), y.clone());
        Ok(self.bin(e, y, move |g| g.mul(&p)?.mul(&x.pow(&p.sub_scalar(T::one())?)?), move |g| g.mul(&y2)?.mul(&x2.ln()?)))
    }

    /// Make a powf comp(constant expo)
    /// y = a ^ c
    /// dy/da = c * a ^ (c-1)
    pub fn powf(&self, e: T) -> Result<Self> {
        let x = self.data();
        let y = x.powf(e)?;
        Ok(self.un(y, move |g| if e == T::zero() { Ok(g.zeros_like()) } else { g.mul(&x.powf(e - T::one())?.mul_scalar(e)?) }))
    }

    pub fn powi32(&self, n: i32) -> Result<Self> { self.powf(T::from_f64(n as f64)) }
    pub fn powi64(&self, n: i64) -> Result<Self> { self.powf(T::from_f64(n as f64)) }   

    /// cond != 0 ? a : b each backfn is a cond mask 
    pub fn where_cond(cond: &Tensor<T>, a: &Comp<T>, b: &Comp<T>) -> Result<Self> {
        let y = cond.where_cond(&a.data(), &b.data())?;
        let (c1, c2) = (cond.clone(), cond.clone());
        Ok(a.bin(b, y, move |g| c1.where_cond(g, &g.zeros_like()), move |g| c2.where_cond(&g.zeros_like(), g)))
    }

    /// Comparisons are const leaf comp 
    pub fn eq_t(&self, o: &Comp<T>) -> Result<Self> { Ok(Self::constant(self.data().eq_t(&o.data())?)) }
    pub fn neq_t(&self, o: &Comp<T>) -> Result<Self> { Ok(Self::constant(self.data().neq_t(&o.data())?)) }
    pub fn gt(&self, o: &Comp<T>) -> Result<Self> { Ok(Self::constant(self.data().gt(&o.data())?)) }
    pub fn lt(&self, o: &Comp<T>) -> Result<Self> { Ok(Self::constant(self.data().lt(&o.data())?)) }
    pub fn ge(&self, o: &Comp<T>) -> Result<Self> { Ok(Self::constant(self.data().ge(&o.data())?)) }
    pub fn le(&self, o: &Comp<T>) -> Result<Self> { Ok(Self::constant(self.data().le(&o.data())?)) }
    pub fn thresh_eq_t(&self, o: &Comp<T>, thresh: Option<T>) -> Result<Self> { Ok(Self::constant(self.data().thresh_eq_t(&o.data(), thresh)?)) }
    pub fn thresh_neq_t(&self, o: &Comp<T>, thresh: Option<T>) -> Result<Self> { Ok(Self::constant(self.data().thresh_neq_t(&o.data(), thresh)?)) }

    /// Make a scalar add comp
    /// y = a+c
    /// dy/da = 1
    pub fn add_scalar(&self, v: T) -> Result<Self> { Ok(self.un(self.data().add_scalar(v)?, |g| Ok(g.clone()))) }

    /// Make a scalar sub comp
    /// y = a-c
    /// dy/da = 1
    pub fn sub_scalar(&self, v: T) -> Result<Self> { Ok(self.un(self.data().sub_scalar(v)?, |g| Ok(g.clone()))) }

    /// Make a scalar mul comp
    /// y = a*c
    /// dy/da = c
    pub fn mul_scalar(&self, v: T) -> Result<Self> { Ok(self.un(self.data().mul_scalar(v)?, move |g| g.mul_scalar(v))) }
    
    /// Make a scalar div comp
    /// y = a/c
    /// dy/da = 1/c   
    pub fn div_scalar(&self, v: T) -> Result<Self> { Ok(self.un(self.data().div_scalar(v)?, move |g| g.div_scalar(v))) }

    /// Make neg comp
    /// y = -a
    /// dy/da = -1
    pub fn neg(&self) -> Result<Self> { let y = self.data().neg()?; Ok(self.un(y, |g| g.neg())) }

    /// Make abs comp
    /// y = |a|
    /// dy/da = {a < 0 -> -1  ,a > 0 -> 1} 
    pub fn abs(&self) -> Result<Self> {
        let x = self.data();
        Ok(self.un( x.abs()?, move |g| { let z = x.zeros_like(); g.mul(&x.gt(&z)?.sub(&x.lt(&z)?)?) }))
    }

    /// Make square comp
    /// y = a ^ 2
    /// dy/da = 2 * a
    pub fn square(&self) -> Result<Self> {
        let x = self.data();
        Ok(self.un( x.square()?, move |g| g.mul(&x.mul_scalar(T::from_f64(2.0))?)))
    }

    /// Make clamp comp
    /// y = a ^ 2
    /// dy/da = {a < mn && x > mx -> 0 ,else -> 1}
    pub fn clamp(&self, lo: T, hi: T) -> Result<Self> {
        let x = self.data();
        let y = x.clamp(lo, hi)?;
        Ok(self.un(y, move |g| g.mul(&x.ge(&x.fill_like(lo))?.mul(&x.le(&x.fill_like(hi))?)?)))
    }

    /// Make exp comp
    /// y = e ^ a
    /// dy/da = e ^ a
    pub fn exp(&self) -> Result<Self> {
        let y = self.data().exp()?;
        let yc = y.clone(); 
        Ok(self.un(y, move |g| g.mul(&yc))) 
    }

    /// Make ln comp
    /// y = ln(a)
    /// dy/da = 1/a
    pub fn ln(&self) -> Result<Self> { 
        let x = self.data(); 
        let y = x.ln()?; 
        Ok(self.un(y, move |g| g.div(&x))) 
    }

    /// make sqrt comp
    /// y = a ^ 0.5
    /// dy/da = 0.5 / a 
    pub fn sqrt(&self) -> Result<Self> { 
        let y = self.data().sqrt()?; 
        let yc = y.clone(); 
        Ok(self.un(y, move |g| g.div(&yc.mul_scalar(T::from_f64(2.0))?))) 
    }

    /// make sin comp
    /// y = sin(a)
    /// dy/da = cos(a)
    pub fn sin(&self) -> Result<Self> { 
        let x = self.data(); 
        let y = x.sin()?; 
        Ok(self.un(y, move |g| g.mul(&x.cos()?))) 
    }

    /// make cos comp
    /// y = cos(a)
    /// dy/da = -sin(a)   
    pub fn cos(&self) -> Result<Self> { 
        let x = self.data(); 
        let y = x.cos()?; 
        Ok(self.un(y, move |g| g.mul(&x.sin()?)?.neg())) 
    }

    /// make tan comp
    /// y = tanh(a)
    /// dy/da = 1 - tanh^2(a)
    pub fn tanh(&self) -> Result<Self> { 
        let y = self.data().tanh()?; 
        let yc = y.clone(); 
        Ok(self.un(y, move |g| g.mul(&yc.square()?.neg()?.add_scalar(T::one())?))) 
    }

    /// make recip comp
    /// y = 1/a
    /// dy/da = -1 / a ^ 2
    pub fn recip(&self) -> Result<Self> { 
        let y = self.data().recip()?; 
        let yc = y.clone(); 
        Ok(self.un(y, move |g| g.mul(&yc.square()?)?.neg())) 
    }

    /// Zero grad
    pub fn floor(&self) -> Result<Self> { let y = self.data().floor()?; Ok(self.un(y, |g| Ok(g.zeros_like()))) }
    pub fn ceil(&self) -> Result<Self> { let y = self.data().ceil()?; Ok(self.un(y, |g| Ok(g.zeros_like()))) }

    
    // ======================Activation fxn=============================


    /// make sigmoid comp
    /// y = 1 / (1 + e ^ -a)
    /// dy/da = y * (1 - y)
    pub fn sigmoid(&self) -> Result<Self> { 
        let y = self.data().sigmoid()?; 
        let yc = y.clone(); 
        Ok(self.un(y, move |g| g.mul(&yc.mul(&yc.neg()?.add_scalar(T::one())?)?))) 
    }

    
    /// make relu comp
    /// y = {a > 0 -> a, else -> 0}
    /// dy/da = {a > 0 -> 1,else -> 0}
    pub fn relu(&self) -> Result<Self> {
        let x = self.data();
        let y = x.relu()?;
        Ok(self.un(y, move |g| g.mul(&x.gt(&x.zeros_like())?)))
    }

    /// make leaky relu comp
    /// y = {a > 0 -> a, else -> a * alpha}
    /// dy/da = {a > 0 -> 1,else -> alpha}
    pub fn lrelu(&self, alpha: T) -> Result<Self> {
        let x = self.data();
        let y = x.lrelu(alpha)?;
        Ok(self.un(y, move |g| g.mul(&x.gt(&x.zeros_like())?.mul_scalar(T::one() - alpha)?.add_scalar(alpha)?)))
    }

    /// make random relu comp
    /// y = {a > 0 -> a, else -> a * alpha}
    /// dy/da = {a > 0 -> 1,else -> alpha}
    pub fn rrelu(&self, alpha: T) -> Result<Self> {
        let x = self.data();
        let y = x.lrelu(alpha)?;
        Ok(self.un(y, move |g| g.mul(&x.gt(&x.zeros_like())?.mul_scalar(T::one() - alpha)?.add_scalar(alpha)?)))
    }

    /// make gelu comp
    // made usign ai
    pub fn gelu(&self) -> Result<Self> {
        let x = self.data();
        let y = x.gelu()?;
        Ok(self.un(y, move |g| {
            let (c, a, one, half) = (T::from_f64(0.7978845608028654), T::from_f64(0.044715), T::one(), T::from_f64(0.5));
            let x2 = x.square()?;
            let t = x.add(&x.mul(&x2)?.mul_scalar(a)?)?.mul_scalar(c)?.tanh()?;
            let sech2 = t.square()?.neg()?.add_scalar(one)?;
            let du = x2.mul_scalar(T::from_f64(3.0) * a)?.add_scalar(one)?.mul_scalar(c)?;
            g.mul(&t.add_scalar(one)?.mul_scalar(half)?.add(&x.mul(&sech2)?.mul(&du)?.mul_scalar(half)?)?)
        }))
    }

    /// Make selu comp
    /// y = {a > 0 -> lambda * a, else -> lambda * alpha * (e^a - 1 ) }
    /// dy/da = {a > o -> lambda,else -> lambda * alpha * e^a}
    // Tricky eqn so used ai to build
    pub fn selu(&self, lambda: T, alpha: T) -> Result<Self> {
        let x = self.data();
        let y = x.selu(lambda, alpha)?;
        Ok(self.un(y, move |g| {
            let m = x.gt(&x.zeros_like())?;
            let neg = m.neg()?.add_scalar(T::one())?.mul(&x.exp()?.mul_scalar(lambda * alpha)?)?;
            g.mul(&m.mul_scalar(lambda)?.add(&neg)?)
        }))
    }

    /// Make elu comp
    /// y = {a > 0 -> a, else ->  alpha * (e^a - 1 ) }
    /// dy/da = {a > 0 -> 1 ,else ->  alpha * e^a}
    pub fn elu(&self, alpha: T) -> Result<Self> {
        let x = self.data();
        let y = x.elu(alpha)?;
        Ok(self.un(y, move |g| {
            let m = x.gt(&x.zeros_like())?;
            let neg = m.neg()?.add_scalar(T::one())?.mul(&x.exp()?.mul_scalar(alpha)?)?;
            g.mul(&m.add(&neg)?)
        }))
    }

    /// make softplus comp
    /// y = ln(1 + e^a)
    /// dy/da = 1 / ( 1 + e^-a) 
    pub fn softplus(&self) -> Result<Self> { 
        let x = self.data(); 
        let y = x.softplus()?; 
        Ok(self.un(y, move |g| g.mul(&x.sigmoid()?))) 
    }

    /// Make hardshrink comp
    /// y = {a < -lambda || a > lambda -> a, else -> 0 }
    /// dy/da = {a < -lambda || a > lambda -> 1, else -> 0 }
    pub fn hardshrink(&self, lambda: T) -> Result<Self> {
        let x = self.data();
        let y = x.hardshrink(lambda)?;
        Ok(self.un(y, move |g| g.mul(&x.abs()?.gt(&x.fill_like(lambda))?)))
    }

    /// Make hardtanh comp
    /// y = {a > mx -> mx,a < mn -> mn,else -> a}
    pub fn hardtanh(&self, min: T, max: T) -> Result<Self> { 
        self.clamp(min, max) // This is autograd clamp
    }

    /// Make hardsigmoid comp
    /// y = {a > mx -> mx,a < mn -> mn,else -> (a + 3) / 6, mx = 1,mn = 0}
    pub fn hardsigmoid(&self) -> Result<Self> { 
        self.add_scalar(T::one().mul(T::from_f64(3.0)))?.div_scalar(T::from_f64(6.0))?.clamp(T::zero(), T::one()) 
    }

    /// Make tanshrink comp
    /// y = a - tanh(a)
    pub fn tanhshrink(&self) -> Result<Self> { 
        self.sub(&self.tanh()?) 
    }

    /// make silu/swish comp
    /// y = a * sigmoid(a)
    pub fn silu(&self) -> Result<Self> { self.mul(&self.sigmoid()?) }
    pub fn swish(&self) -> Result<Self> { self.silu() }

    /// make softsign comp
    /// y = a / (1 + |a|)
    pub fn softsign(&self) -> Result<Self> { 
        self.div(&self.abs()?.add_scalar(T::one())?) 
    }

    /// make mish comp
    /// y = a * tanh(softplus(a))
    pub fn mish(&self) -> Result<Self> { 
        self.mul(&self.softplus()?.tanh()?) 
    }


    // Naive prop impls simply to handle such fxn in comp graph
    // Since these ops will be indirectly a part of the comp graph so we need rules for forawrd andf backward pass
    // i,e x -> reshape -> y then whne going from y -> reshape -> x hence we need comps for such non differntial comps
    // Used ai assistance to quickly add 

    pub fn tril(&self, diagonal: isize) -> Result<Self> { self.lin(move |t| t.tril(diagonal)) }
    pub fn triu(&self, diagonal: isize) -> Result<Self> { self.lin(move |t| t.triu(diagonal)) }
    pub fn lower_triangle(&self) -> Result<Self> { self.tril(0) }
    pub fn upper_triangle(&self) -> Result<Self> { self.triu(0) }
    pub fn alt_tril(&self, diagonal: isize) -> Result<Self> { self.lin(move |t| t.alt_tril(diagonal)) }
    pub fn alt_triu(&self, diagonal: isize) -> Result<Self> { self.lin(move |t| t.alt_triu(diagonal)) }
    pub fn alt_lower_triangle(&self) -> Result<Self> { self.alt_tril(0) }
    pub fn alt_upper_triangle(&self) -> Result<Self> { self.alt_triu(0) }

    pub fn reshape(&self, shape: &[isize]) -> Result<Self> {
        let x = self.data();
        let y = x.reshape(shape)?;
        // Same reshape op applied in both directions
        Ok(self.un(y, move |g| g.reshape(&x.shape().iter().map(|&s| s as isize).collect::<Vec<isize>>())))
    }

    pub fn flatten(&self) -> Result<Self> { 
        self.reshape(&[self.numel() as isize]) 
    }

    pub fn transpose(&self, d0: isize, d1: isize) -> Result<Self> { 
        Ok(self.un(self.data().transpose(d0, d1)?, move |g| g.transpose(d0, d1))) 
    }

    pub fn t(&self) -> Result<Self> { 
        self.transpose(-2, -1) 
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        let y = self.data().permute(dims)?;
        // Revert the permutation
        let mut inv = vec![0usize; dims.len()];
        for (i, &d) in dims.iter().enumerate() { inv[d] = i; }
        Ok(self.un(y, move |g| g.permute(&inv)))
    }

    pub fn unsqueeze(&self, dim: isize) -> Result<Self> {
        let x = self.data();
        let d = normalize_dim_inclusive(dim, x.ndim())?;
        let y = x.unsqueeze(dim)?;
        Ok(self.un(y, move |g| g.squeeze(d as isize)))
    }

    pub fn squeeze(&self, dim: isize) -> Result<Self> {
        let x = self.data();
        let d = normalize_dim(dim, x.ndim())?;
        let y = x.squeeze(dim)?;
        Ok(self.un(y, move |g| g.unsqueeze(d as isize)))
    }

    pub fn squeeze_all(&self) -> Result<Self> {
        self.reshape(& self.shape().iter().copied().filter(|&s| s != 1).map(|s| s as isize).collect::<Vec<isize>>())
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        let x = self.data();
        Ok(self.un(x.broadcast_to(shape)?, move |g| unbroadcast(g.clone(), &x.shape().to_vec())))
    }
    pub fn expand(&self, shape: &[usize]) -> Result<Self> {  self.broadcast_to(shape) }


    pub fn flip(&self, dim: isize) -> Result<Self> { 
        Ok(self.un(self.data().flip(dim)?, move |g| g.flip(dim))) 
    }

    pub fn narrow(&self, dim: isize, start: usize, len: usize) -> Result<Self> {
        let x = self.data();
        let y = x.narrow(dim, start, len)?;
        let d = normalize_dim(dim, x.ndim())?;
        let sh = x.shape().to_vec();
        Ok(self.un(y, move |g| {
            let mk = |l: usize| { let mut s = sh.clone(); s[d] = l; Tensor::<T>::zeros(&s).on(g.device()) };
            let mut parts: Vec<Tensor<T>> = Vec::new();
            if start > 0 { parts.push(mk(start)?); }
            parts.push(g.clone());
            if sh[d] - start - len > 0 { parts.push(mk(sh[d] - start - len)?); }
            Tensor::cat(& parts.iter().collect::<Vec<&Tensor<T>>>(), d as isize)
        }))
    }

    pub fn select(&self, dim: isize, index: usize) -> Result<Self> { 
        self.narrow(dim, index, 1)?.squeeze(dim) 
    }

    pub fn split(&self, dim: isize, size: usize) -> Result<Vec<Self>> {
        let d = normalize_dim(dim, self.ndim())?;
        let n = self.shape()[d];
        (0..n).step_by(size.max(1)).map(|s| self.narrow(d as isize, s, size.min(n - s))).collect()
    }

    pub fn cat(vars: &[&Comp<T>], dim: isize) -> Result<Self> {
        let ts: Vec<Tensor<T>> = vars.iter().map(|v| v.data()).collect();
        let refs: Vec<&Tensor<T>> = ts.iter().collect();
        let y = Tensor::cat(&refs, dim)?;
        let d = normalize_dim(dim, y.ndim())?;
        let sizes: Vec<usize> = ts.iter().map(|t| t.shape()[d]).collect();
        Ok(Self::from_op(y, vars.iter().map(|v| (*v).clone()).collect(), move |g| {
            let (mut off, mut out) = (0, Vec::with_capacity(sizes.len()));
            for &s in &sizes { out.push(Some(g.narrow(d as isize, off, s)?.compact()?)); off += s; }
            Ok(out)
        }))
    }

    pub fn stack(vars: &[&Comp<T>], dim: isize) -> Result<Self> {
        let first = *vars.first().ok_or_else(|| TensorError::InvalidShape("No Comps provided to stack".into()))?;
        let d = normalize_dim_inclusive(dim, first.ndim())? as isize;
        let ex: Vec<Comp<T>> = vars.iter().map(|v| v.unsqueeze(d)).collect::<Result<_>>()?;
        let refs: Vec<&Comp<T>> = ex.iter().collect();
        Self::cat(&refs, d)
    }

    /// Gather slices along dim in forward, idx accumulate in backward
    pub fn index_select(&self, dim: isize, indices: &[usize]) -> Result<Self> {
        let x = self.data();
        let d = normalize_dim(dim, x.ndim())?;
        let y = x.index_select(dim, indices)?;
        let (sh, idx) = (x.shape().to_vec(), indices.to_vec());
        Ok(self.un(y, move |g| {
            let (outer, inner, n, k) = (sh[..d].iter().product::<usize>(), sh[d + 1..].iter().product::<usize>(), sh[d], idx.len());
            let gv = g.to_vec()?;
            let mut out = vec![T::zero(); outer * n * inner];
            par::for_each_row(&mut out, n * inner, k * inner, |o, row| {
                for (j, &i) in idx.iter().enumerate() {
                    let src = &gv[(o * k + j) * inner..(o * k + j + 1) * inner];
                    for q in 0..inner { row[i * inner + q] += src[q]; }
                }});
            Tensor::from_vec(out, &sh)?.to_device(g.device())
        }))
    }

    
    pub fn sum(&self, dim: isize, keepdim: bool) -> Result<Self> {
        let x = self.data();
        let d = normalize_dim(dim, x.ndim())?;
        Ok(self.un(x.sum(dim, keepdim)?, move |g| expand_to(&keep(g, d, keepdim)?, &x.shape().to_vec())))
    }

    pub fn mean(&self, dim: isize, keepdim: bool) -> Result<Self> {
        self.sum(dim, keepdim)?.div_scalar(T::from_f64( self.shape()[normalize_dim(dim, self.ndim())?] as f64))
    }

    pub fn extreme(&self, dim: isize, keepdim: bool, is_max: bool) -> Result<Self> {
        let x = self.data();
        let d = normalize_dim(dim, x.ndim())?;
        let y = if is_max { x.max(dim, keepdim)? } else { x.min(dim, keepdim)? };
        let yk = keep(&y, d, keepdim)?;
        Ok(self.un(y, move |g| { let m = x.eq_t(&yk)?; let n = m.sum(d as isize, true)?; m.mul(&keep(g, d, keepdim)?)?.div(&n) }))
    }

    /// Max over dim, ties share the grad equally
    pub fn max(&self, dim: isize, keepdim: bool) -> Result<Self> { self.extreme(dim, keepdim, true) }

    /// Min over dim, ties share the grad equally
    pub fn min(&self, dim: isize, keepdim: bool) -> Result<Self> { self.extreme(dim, keepdim, false) }

    /// Product over dim
    pub fn prod(&self, dim: isize, keepdim: bool) -> Result<Self> {
        let x = self.data();
        let d = normalize_dim(dim, x.ndim())?;
        let y = x.prod(dim, keepdim)?;
        let sh = x.shape().to_vec();
        Ok(self.un(y, move |g| {
            let n = sh[d];
            let xm = x.movedim_last(d)?.host_data()?.into_owned();
            let gv = g.to_vec()?;
            let mut out = vec![T::zero(); xm.len()];
            par::for_each_row(&mut out, n, n, |i, row| {
                let xs = &xm[i * n..(i + 1) * n];
                let mut pre = T::one();
                for j in 0..n { row[j] = pre; pre = pre * xs[j]; }
                let mut suf = T::one();
                for j in (0..n).rev() { row[j] = row[j] * suf * gv[i]; suf = suf * xs[j]; }
            });
            rows_to_tensor(out, &sh, d)?.to_device(g.device())
        }))
    }

    pub fn sum_all(&self) -> Result<Self> { self.flatten()?.sum(0, false) }
    pub fn mean_all(&self) -> Result<Self> { self.flatten()?.mean(0, false) }
    pub fn max_all(&self) -> Result<Self> { self.flatten()?.max(0, false) }
    pub fn min_all(&self) -> Result<Self> { self.flatten()?.min(0, false) }
    pub fn prod_all(&self) -> Result<Self> { self.flatten()?.prod(0, false) }
    pub fn range(&self, dim: isize) -> Result<Self> { self.max(dim, true)?.sub(&self.min(dim, true)?) }
    pub fn range_all(&self) -> Result<Self> { self.max_all()?.sub(&self.min_all()?) }


    pub fn percentile(&self, p: f64, dim: isize, keepdim: bool) -> Result<Self> {
        let x = self.data();
        let d = normalize_dim(dim, x.ndim())?;
        let y = x.percentile(p, dim, keepdim)?;
        let sh = x.shape().to_vec();
        Ok(self.un(y, move |g| {
            let n = sh[d];
            let xm = x.movedim_last(d)?.host_data()?.into_owned();
            let gv = g.to_vec()?;
            let mut out = vec![T::zero(); xm.len()];
            let rank = (p / 100.0).clamp(0.0, 1.0) * (n.max(1) - 1) as f64;
            let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
            let fr = rank - lo as f64;
            par::for_each_row(&mut out, n, n, |i, row| {
                let xs = &xm[i * n..(i + 1) * n];
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap_or(std::cmp::Ordering::Equal));
                row[idx[lo]] += gv[i] * T::from_f64(1.0 - fr);
                row[idx[hi]] += gv[i] * T::from_f64(fr);
            });
            rows_to_tensor(out, &sh, d)?.to_device(g.device())
        }))
    }

    pub fn median(&self, dim: isize, keepdim: bool) -> Result<Self> { self.percentile(50.0, dim, keepdim) }

    pub fn percentile_all(&self, p: f64) -> Result<Self> { self.flatten()?.percentile(p, 0, false) }

    pub fn median_all(&self) -> Result<Self> { self.percentile_all(50.0) }

    pub fn quartile(&self, dim: isize, qtile: Quartile) -> Result<Self> {
        let pct = match qtile { Quartile::Q1 => 25, Quartile::Q2 => 50, Quartile::Q3 => 75, Quartile::Q(x) => x.clamp(0, 100) } as f64;
        self.percentile(pct, dim, true)
    }

    pub fn quartile_all(&self, qtile: Quartile) -> Result<Self> { self.flatten()?.quartile(0, qtile)?.reshape(&[]) }

    pub fn iqr(&self, dim: isize) -> Result<Self> { self.quartile(dim, Quartile::Q3)?.sub(&self.quartile(dim, Quartile::Q1)?) }

    pub fn iqr_all(&self) -> Result<Self> { self.quartile_all(Quartile::Q3)?.sub(&self.quartile_all(Quartile::Q1)?) }

    pub fn var(&self, dim: isize, keepdim: bool, is_sample: bool) -> Result<Self> {
        let n = self.shape()[normalize_dim(dim, self.ndim())?];
        let denom = if is_sample { n.saturating_sub(1) } else { n }.max(1);
        self.sub(&self.mean(dim, true)?)?.square()?.sum(dim, keepdim)?.div_scalar(T::from_f64(denom as f64))
    }

    pub fn std(&self, dim: isize, keepdim: bool, is_sample: bool) -> Result<Self> { self.var(dim, keepdim, is_sample)?.sqrt() }

    pub fn var_all(&self, is_sample: bool) -> Result<Self> { self.flatten()?.var(0, false, is_sample) }

    pub fn std_all(&self, is_sample: bool) -> Result<Self> { self.var_all(is_sample)?.sqrt() }

    pub fn coeff_of_var(&self, dim: isize, is_sample: bool) -> Result<Self> { self.std(dim, true, is_sample)?.div(&self.mean(dim, true)?)?.mul_scalar(T::from_f64(100.0)) }

    pub fn coeff_of_var_all(&self, is_sample: bool) -> Result<Self> { self.std_all(is_sample)?.div(&self.mean_all()?)?.mul_scalar(T::from_f64(100.0)) }

    pub fn mag(&self) -> Result<Self> { self.square()?.sum_all()?.sqrt() }

    pub fn cov(&self, other: &Self) -> Result<Self> {
        if self.numel() != other.numel() { return Err(TensorError::ShapeMismatch { op: "cov", lhs: self.shape(), rhs: other.shape() }); }
        let (a, b) = (self.flatten()?, other.flatten()?);
        let prod = a.sub(&a.mean_all()?)?.mul(&b.sub(&b.mean_all()?)?)?;
        prod.sum_all()?.div_scalar(T::from_f64((a.numel().max(2) - 1) as f64))
    }

    pub fn pearson(&self, other: &Self) -> Result<Self> { self.cov(other)?.div(&self.std_all(true)?.mul(&other.std_all(true)?)?) }

    pub fn corr(&self, other: &Self) -> Result<Self> { self.pearson(other) }

    pub fn cov_matrix(&self) -> Result<Self> {
        if self.ndim() != 2 { return Err(TensorError::InvalidShape("Covariance matrix works for only 2D tensors ".into())); }
        let n = self.shape()[0];
        let c = self.sub(&self.mean(0, true)?)?;
        c.t()?.matmul(&c)?.div_scalar(T::from_f64((n.max(2) - 1) as f64))
    }

    pub fn corrcoef(&self) -> Result<Self> {
        let c = self.cov_matrix()?;
        let d = c.diag()?.sqrt()?;
        c.div(&d.unsqueeze(1)?.mul(&d.unsqueeze(0)?)?)
    }

    pub fn corr_mat(&self) -> Result<Self> { self.corrcoef() }

    pub fn skewness(&self) -> Result<Self> { self.sub(&self.mean_all()?)?.div(&self.std_all(true)?)?.powi32(3)?.mean_all() }

    pub fn kurtosis(&self) -> Result<Self> { self.sub(&self.mean_all()?)?.div(&self.std_all(true)?)?.powi32(4)?.mean_all() }

    pub fn skewness_all(&self, is_sample: bool, dim: isize) -> Result<Self> { self.sub(&self.mean(dim, true)?)?.div(&self.std(dim, true, is_sample)?)?.powi32(3)?.mean(dim, false) }

    pub fn kurtosis_all(&self, is_sample: bool, dim: isize) -> Result<Self> { self.sub(&self.mean(dim, true)?)?.div(&self.std(dim, true, is_sample)?)?.powi32(4)?.mean(dim, false) }

    pub fn matmul_nd(&self, o: &Comp<T>) -> Result<Self> {
        let (a, b) = (self.data(), o.data());
        let y = a.matmul(&b)?;
        Ok(self.bin(o, y, move |g| g.matmul(&b.t()?), move |g| a.t()?.matmul(g)))
    }

    pub fn matmul(&self, o: &Comp<T>) -> Result<Self> {
        let (sa, sb) = (self.ndim() == 1, o.ndim() == 1);
        let a = if sa { self.unsqueeze(0)? } else { self.clone() };
        let b = if sb { o.unsqueeze(1)? } else { o.clone() };
        let r = a.matmul_nd(&b)?;
        if sa && sb { r.reshape(&[]) } else if sa { r.squeeze(-2) } else if sb { r.squeeze(-1) } else { Ok(r) }
    }

    pub fn dot(&self, o: &Comp<T>) -> Result<Self> {
        if self.shape().last() != o.shape().last() || self.ndim() == 0 { return Err(TensorError::ShapeMismatch { op: "Dot", lhs: self.shape(), rhs: o.shape() }); }
        self.mul(o)?.sum(-1, false)
    }

    pub fn cross(&self, o: &Comp<T>) -> Result<Self> {
        if self.shape().last() != Some(&3) || o.shape().last() != Some(&3) { return Err(TensorError::ShapeMismatch { op: "Cross", lhs: self.shape(), rhs: o.shape() }); }
        let (a0, a1, a2) = (self.select(-1, 0)?, self.select(-1, 1)?, self.select(-1, 2)?);
        let (b0, b1, b2) = (o.select(-1, 0)?, o.select(-1, 1)?, o.select(-1, 2)?);
        let c0 = a1.mul(&b2)?.sub(&a2.mul(&b1)?)?;
        let c1 = a2.mul(&b0)?.sub(&a0.mul(&b2)?)?;
        let c2 = a0.mul(&b1)?.sub(&a1.mul(&b0)?)?;
        Self::stack(&[&c0, &c1, &c2], -1)
    }

    pub fn outer(&self, o: &Comp<T>) -> Result<Self> {
        if self.ndim() != 1 || o.ndim() != 1 { return Err(TensorError::InvalidShape("Outer expects 1D tensors".into())); }
        self.unsqueeze(1)?.mul(&o.unsqueeze(0)?)
    }

    pub fn diag(&self) -> Result<Self> {
        let x = self.data();
        let y = x.diag()?;
        let sh = x.shape().to_vec();
        Ok(self.un(y, move |g| {
            if sh.len() == 1 { g.diag() } else { let gv = g.to_vec()?; Tensor::from_fn(&sh, |ix| if ix[0] == ix[1] { gv[ix[0]] } else { T::zero() }).to_device(g.device()) }
        }))
    }
    
    pub fn trace(&self) -> Result<Self> {
        if self.ndim() != 2 { return Err(TensorError::InvalidShape("Trace expects a 2D tensor".into())); }
        self.diag()?.sum(0, false)
    }
    
    pub fn inv(&self) -> Result<Self> {
        let y = self.data().inv()?;
        let yc = y.clone();
        Ok(self.un(y, move |g| { let yt = yc.t()?; yt.matmul(g)?.matmul(&yt)?.neg() }))
    }
    
    pub fn det(&self) -> Result<Self> {
        let x = self.data();
        let d = x.det()?;
        let y = Tensor::scalar(d).on(x.device())?;
        Ok(self.un(y, move |g| x.inv()?.t()?.mul_scalar(d)?.mul(g)))
    }
    
    pub fn solve(&self, b: &Comp<T>) -> Result<Self> {
        let (a, bt) = (self.data(), b.data());
        let x = a.solve(&bt)?;
        let xc = x.clone();
        let (a2, vec_rhs) = (a.clone(), bt.ndim() == 1);
        Ok(self.bin(b, x, move |g| {
            let gb = a.t()?.solve(g)?;
            let (gb2, x2) = if vec_rhs { (gb.unsqueeze(1)?, xc.unsqueeze(1)?) } else { (gb, xc.clone()) };
            gb2.matmul(&x2.t()?)?.neg()
        }, move |g| a2.t()?.solve(g)))
    }


    pub fn norm(&self, p: f64, dim: isize, keepdim: bool) -> Result<Self> {
        if p.abs() <= 0.01 { return Err(TensorError::Custom(format!("Norm of {} is too close to zero and wil lead to unstable exponentiation", p))); }
        let a = self.abs()?;
        if p.is_infinite() { a.max(dim, keepdim) }
        else if p == 1.0 { a.sum(dim, keepdim) }
        else if p == 2.0 { self.square()?.sum(dim, keepdim)?.sqrt() }
        else { a.powf(T::from_f64(p))?.sum(dim, keepdim)?.powf(T::from_f64(1.0 / p)) }
    }

    pub fn normalize(&self, dim: isize) -> Result<Self> { let n = self.norm(2.0, dim, true)?.add_scalar(T::from_f64(1e-12))?; self.div(&n) }

    pub fn cosine_similarity(&self, o: &Comp<T>, dim: isize) -> Result<Self> { self.normalize(dim)?.mul(&o.normalize(dim)?)?.sum(dim, false) }

    pub fn pdim_dist(&self, o: &Comp<T>, p: f64, dim: isize, keepdim: bool) -> Result<Self> { self.sub(o)?.norm(p, dim, keepdim) }

    pub fn manhatten(&self, o: &Comp<T>, dim: isize) -> Result<Self> { self.sub(o)?.norm(1.0, dim, true) }

    pub fn euclidean(&self, o: &Comp<T>, dim: isize) -> Result<Self> { self.sub(o)?.norm(2.0, dim, true) }

    pub fn softmax(&self, dim: isize) -> Result<Self> {
        let y = self.data().softmax(dim)?;
        let yc = y.clone();
        Ok(self.un(y, move |g| { let s = g.mul(&yc)?.sum(dim, true)?; yc.mul(&g.sub(&s)?) }))
    }

    pub fn logsumexp(&self, dim: isize, keepdim: bool) -> Result<Self> {
        let x = self.data();
        let d = normalize_dim(dim, x.ndim())?;
        let y = x.logsumexp(dim, keepdim)?;
        Ok(self.un(y, move |g| keep(g, d, keepdim)?.mul(&x.softmax(dim)?)))
    }
    
    pub fn log_softmax(&self, dim: isize) -> Result<Self> { self.sub(&self.logsumexp(dim, true)?) }
    
    pub fn layer_norm(&self, eps: T) -> Result<Self> {
        let x = self.data();
        let y = x.layer_norm(eps)?;
        let yc = y.clone();
        Ok(self.un(y, move |g| {
            let sigma = x.var(-1, true, false)?.add_scalar(eps)?.sqrt()?;
            let (mg, mgy) = (g.mean(-1, true)?, g.mul(&yc)?.mean(-1, true)?);
            g.sub(&mg)?.sub(&yc.mul(&mgy)?)?.div(&sigma)
        }))
    }

}
// So no need to explicilty do a.add(b) instead we can do a + b

impl<T: Float> std::ops::Neg for &Comp<T> { type Output = Comp<T>; fn neg(self) -> Comp<T> { Comp::neg(self).expect("Comp neg failed") } }
impl<T: Float> std::ops::Neg for Comp<T> { type Output = Comp<T>; fn neg(self) -> Comp<T> { Comp::neg(&self).expect("Comp neg failed") } }

macro_rules! impl_bin_ops_comp {
    ($tr:ident, $m:ident) => {
        impl<T: Float> std::ops::$tr<&Comp<T>> for &Comp<T> { type Output = Comp<T>; fn $m(self, rhs: &Comp<T>) -> Comp<T> { Comp::$m(self, rhs).expect(concat!("Comp ", stringify!($m), " failed")) } }
        impl<T: Float> std::ops::$tr<Comp<T>> for Comp<T> { type Output = Comp<T>; fn $m(self, rhs: Comp<T>) -> Comp<T> { Comp::$m(&self, &rhs).expect(concat!("Comp ", stringify!($m), " failed")) } }
        impl<T: Float> std::ops::$tr<&Comp<T>> for Comp<T> { type Output = Comp<T>; fn $m(self, rhs: &Comp<T>) -> Comp<T> { Comp::$m(&self, rhs).expect(concat!("Comp ", stringify!($m), " failed")) } }
        impl<T: Float> std::ops::$tr<Comp<T>> for &Comp<T> { type Output = Comp<T>; fn $m(self, rhs: Comp<T>) -> Comp<T> { Comp::$m(self, &rhs).expect(concat!("Comp ", stringify!($m), " failed")) } }
    };
}
impl_bin_ops_comp!(Add, add);
impl_bin_ops_comp!(Sub, sub);
impl_bin_ops_comp!(Mul, mul);
impl_bin_ops_comp!(Div, div);

macro_rules! impl_scalar_ops_comp {
    ($t:ty) => {
        impl std::ops::Add<$t> for &Comp<$t> { type Output = Comp<$t>; fn add(self, v: $t) -> Comp<$t> { self.add_scalar(v).unwrap() } }
        impl std::ops::Sub<$t> for &Comp<$t> { type Output = Comp<$t>; fn sub(self, v: $t) -> Comp<$t> { self.sub_scalar(v).unwrap() } }
        impl std::ops::Mul<$t> for &Comp<$t> { type Output = Comp<$t>; fn mul(self, v: $t) -> Comp<$t> { self.mul_scalar(v).unwrap() } }
        impl std::ops::Div<$t> for &Comp<$t> { type Output = Comp<$t>; fn div(self, v: $t) -> Comp<$t> { self.div_scalar(v).unwrap() } }
        impl std::ops::Add<$t> for Comp<$t> { type Output = Comp<$t>; fn add(self, v: $t) -> Comp<$t> { self.add_scalar(v).unwrap() } }
        impl std::ops::Sub<$t> for Comp<$t> { type Output = Comp<$t>; fn sub(self, v: $t) -> Comp<$t> { self.sub_scalar(v).unwrap() } }
        impl std::ops::Mul<$t> for Comp<$t> { type Output = Comp<$t>; fn mul(self, v: $t) -> Comp<$t> { self.mul_scalar(v).unwrap() } }
        impl std::ops::Div<$t> for Comp<$t> { type Output = Comp<$t>; fn div(self, v: $t) -> Comp<$t> { self.div_scalar(v).unwrap() } }
    };
}

impl_scalar_ops_comp!(f32);
impl_scalar_ops_comp!(f64);