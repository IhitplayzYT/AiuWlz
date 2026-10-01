use crate::dtype::Element;
use crate::tensor::Tensor::Tensor;
use std::fmt::{self, Display};

const EDGE: usize = 3;

// Write the nested tensor data in proper fmt to the formatter
// Made using AI 
fn _fmt_helper<T: Element>(f: &mut fmt::Formatter<'_>, data: &[T], shape: &[usize], depth: usize) -> fmt::Result {
    if shape.is_empty() {
        return write!(f, "{}", data[0]);
    }
    let n = shape[0];
    let inner: usize = shape[1..].iter().product();
    let sep = if shape.len() > 1 { format!(",\n{}", " ".repeat(depth + 8)) } else { ", ".to_string() };
    write!(f, "[")?;
    let show = |f: &mut fmt::Formatter<'_>, i: usize| _fmt_helper(f, &data[i * inner..(i + 1) * inner], &shape[1..], depth + 1);
    if n > 2 * EDGE {
        for i in 0..EDGE { show(f, i)?; write!(f, "{sep}")?; }
        write!(f, "...{sep}")?;
        for i in n - EDGE..n { show(f, i)?; if i + 1 < n { write!(f, "{sep}")?; } }
    } else {
        for i in 0..n { show(f, i)?; if i + 1 < n { write!(f, "{sep}")?; } }
    }
    write!(f, "]")
}

impl<T: Element> Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_vec() {
            Ok(data) => {
                write!(f, "Tensor(")?;
                // Pastes the nested tensor here
                if data.is_empty() {
                    write!(f, "[]")?; 
                } else {
                    _fmt_helper(f, &data, self.shape(), 0)?; 
                }
                write!(f, ", shape={:?}, dtype={}, device={})", self.shape(), T::NAME, self.device())
            }
            Err(e) => write!(f, "Tensor(Corrupt: {e})"),
        }
    }
}

impl<T: Element> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { 
        Display::fmt(self, f) 
    }
}
