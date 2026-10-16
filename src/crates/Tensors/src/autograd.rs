use std::{cell::Cell, sync::atomic::AtomicUsize};

use crate::{error::Result, tensor::Tensor};

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

