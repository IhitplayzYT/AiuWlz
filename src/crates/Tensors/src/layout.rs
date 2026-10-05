use crate::error::{TensorResult, TensorError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

/// Get strides array from shape for contigious block 
// eg : [3,4] -> [4,1] i.e to move in 1st dim we need to move by 4 mem blocks and to move in 2nd dim we need to move 1 mem block
pub fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut acc = 1isize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i].max(1) as isize;
    }
    strides
}

/// Handle the -tve indexing in dims
pub fn normalize_dim(dim: isize, ndim: usize) -> TensorResult<usize> {
    let n = ndim as isize;
    if dim >= -n && dim < n {
        Ok(if dim < 0 { (dim + n) as usize } else { dim as usize })
    } else {
        Err(TensorError::InvalidDim { dim, ndim })
    }
}

/// Like normalize_dim but allows dim = ndim for unsqueeze op
pub fn normalize_dim_inclusive(dim: isize, ndim: usize) -> TensorResult<usize> {
    let n = ndim as isize + 1;
    if dim >= -n && dim < n {
        Ok(if dim < 0 { (dim + n) as usize } else { dim as usize })
    } else {
        Err(TensorError::InvalidDim { dim, ndim })
    }
}

/// Tensor broadcasting during any ops 
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> TensorResult<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    let (la,lb) = (a.len(),b.len());
    for i in 0..n {
        // The extra dims are init to 1
        let da = if i < n - la { 1 } else { a[i - (n - la)] };
        let db = if i < n - lb { 1 } else { b[i - (n - lb)] };
        out[i] = if da == db || db == 1 {da} else if da == 1 {db} else {return Err(TensorError::ShapeMismatch {op: "Tensor broadcats",lhs: a.to_vec(),rhs: b.to_vec()});};
    }
    Ok(out)
}

impl Layout {
    /// Make a contigious layout
    pub fn contiguous(shape: &[usize]) -> Self {
        Layout { shape: shape.to_vec(), strides: contiguous_strides(shape), offset: 0 }
    }

    /// Make a Arbitrary layout
    pub fn new(shape: Vec<usize>, strides: Vec<isize>, offset: usize) -> TensorResult<Self> {
        if shape.len() != strides.len() {
            return Err(TensorError::InvalidShape(format!("Shape {:?} and Strides {:?} of diff dim",shape, strides)));
        }
        Ok(Layout { shape, strides, offset })
    }
    /// Get layout shape
    pub fn shape(&self) -> &[usize] { &self.shape }
    /// Get layout strides 
    pub fn strides(&self) -> &[isize] { &self.strides }
    /// Get layout offset 
    pub fn offset(&self) -> usize { self.offset }
    /// Get layout ndim 
    pub fn ndim(&self) -> usize { self.shape.len() }
    /// Get no of elem in a layout  
    pub fn numel(&self) -> usize { self.shape.iter().product() }

    /// Check if a layout is contigious
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1isize;
        for (&s, &st) in self.shape.iter().zip(&self.strides).rev() {
            if s != 1 && st != expected {
                return false;
            }
            expected *= s as isize;
        }
        true
    }

    /// Reshape a layout
    pub fn reshape(&self, new_shape: &[usize]) -> TensorResult<Layout> {
        if new_shape.iter().product::<usize>() != self.numel() {
            return Err(TensorError::InvalidShape(format!("Inconsistent layout {:?} cannot reshape into {:?}",self.shape, new_shape)));
        }
        if !self.is_contiguous() {
            return Err(TensorError::InvalidShape("Non-Contigious layout, for reshape continous is required".into()));
        }
        Ok(Layout { shape: new_shape.to_vec(), strides: contiguous_strides(new_shape), offset: self.offset })
    }


    /// Takes a 0 indexed array containing all elems till n-1
    /// And then based on the value at that idx convertes the shape[idx] and layout[idx] to shape[dims[idx]] and layout[dims[idx]]
    /// Reorders layout shape and strides based on dims provided
    pub fn permute(&self, dims: &[usize]) -> TensorResult<Layout> {
        let n = self.ndim();
        let mut seen = vec![false; n];
        if dims.len() != n || dims.iter().any(|&d| d >= n || std::mem::replace(&mut seen[d], true)) {
            return Err(TensorError::InvalidShape(format!("{:?} is not a permutation of 0..{}", dims, n)));
        }
        Ok(Layout {shape: dims.iter().map(|&d| self.shape[d]).collect(),strides: dims.iter().map(|&d| self.strides[d]).collect(),offset: self.offset})
    }

    /// Transpose along d0  and d1
    pub fn transpose(&self, d0: usize, d1: usize) -> TensorResult<Layout> {
        let mut dims: Vec<usize> = (0..self.ndim()).collect();
        if d0 >= dims.len() || d1 >= dims.len() {
            return Err(TensorError::InvalidDim { dim: d0.max(d1) as isize, ndim: self.ndim() });
        }
        dims.swap(d0, d1);
        self.permute(&dims)
    }

    /// Creates a new reshaped Layout according to target shape
    pub fn broadcast_to(&self, target: &[usize]) -> TensorResult<Layout> {
        let (n,nd) = (target.len(),self.ndim());
        if n < nd { return Err(TensorError::ShapeMismatch { op: "Broadcast to", lhs: self.shape.clone(), rhs: target.to_vec() });}
        let extra = n - nd;
        let mut strides = vec![0isize; n];
        for i in 0..n {
            if i < extra {
                strides[i] = 0;
            }else{
                let (s, st) = (self.shape[i - extra], self.strides[i - extra]);
                if s == target[i] {
                    strides[i] = st;
                } else if s == 1 {
                    strides[i] = 0; // shape == 1 means no stride
                } else {
                    return Err(TensorError::ShapeMismatch { op: "Broadcast to", lhs: self.shape.clone(), rhs: target.to_vec() });
                }
            }
        }
        Ok(Layout { shape: target.to_vec(), strides, offset: self.offset })
    }

    /// Slice along dim, from st to st+l
    pub fn narrow(&self, dim: usize, st: usize, l: usize) -> TensorResult<Layout> {
        if dim >= self.ndim() || st > self.shape[dim] || l > self.shape[dim] - st { return Err(TensorError::InvalidShape(format!("Narrow(dim={dim}, start={st}, len={l}) out of bounds for {:?}",self.shape))); }
        let mut out = self.clone();
        out.shape[dim] = l;
        if l > 0 {
            out.offset = (self.offset as isize + st as isize * self.strides[dim]) as usize;
        }
        Ok(out)
    }

    /// From the selected dim take the index th elem and remove that dim 
    /// eg : dim= 3 x 2 x 4  we so select(2,1) we are doing is selecting 2nd dim i.e [,,4] and from this array remove the 1st elem in the dim i.e 4th dim willl have 0,2,3 since 1th elem from that dim is removed
    pub fn select(&self, dim: usize, index: usize) -> TensorResult<Layout> {
        let mut l = self.narrow(dim, index, 1)?;
        l.shape.remove(dim);
        l.strides.remove(dim);
        Ok(l)
    }

    /// Adds a new dim with size 1
    pub fn unsqueeze(&self, dim: usize) -> Layout {
        let mut out = self.clone();
        let st = if dim < out.ndim() { out.strides[dim] * out.shape[dim] as isize } else { 1 };
        out.shape.insert(dim, 1);
        out.strides.insert(dim, st);
        out
    }

    /// Removes the dim if dim have a singular elem
    pub fn squeeze(&self, dim: usize) -> TensorResult<Layout> {
        if dim >= self.ndim() || self.shape[dim] != 1 { return Err(TensorError::InvalidShape(format!("Cannot squeeze dim:{dim} of {:?}", self.shape)));}
        let mut out = self.clone();
        out.shape.remove(dim);
        out.strides.remove(dim);
        Ok(out)
    }

    /// Reverse iteration view of dim by altering strides in retunred layout
    pub fn flip(&self, dim: usize) -> TensorResult<Layout> {
        if dim >= self.ndim() { return Err(TensorError::InvalidDim { dim: dim as isize, ndim: self.ndim() });}
        let mut out = self.clone();
        // Offset is last elem of the dim
        if self.shape[dim] > 0 {
            out.offset = (self.offset as isize + (self.shape[dim] as isize - 1) * self.strides[dim]) as usize;
        }
        // negative stride to iter along dim in rev
        out.strides[dim] = -self.strides[dim];
        Ok(out)
    }
    
    pub fn offsets(&self) -> StridedIter {
        StridedIter {shape: self.shape.clone(),strides: self.strides.clone(),idx: vec![0; self.shape.len()],cur: self.offset as isize,rem: self.numel()}
    }
}

pub struct StridedIter {
    shape: Vec<usize>,
    strides: Vec<isize>,
    idx: Vec<usize>,
    cur: isize,
    rem: usize,
}

impl Iterator for StridedIter {
    type Item = usize;
    // Multi dimension iterator logic made using AI
    fn next(&mut self) -> Option<usize> {
        if self.rem == 0 {
            return None;
        }
        let out = self.cur as usize;
        self.rem -= 1;
        for d in (0..self.shape.len()).rev() {
            self.idx[d] += 1;
            self.cur += self.strides[d];
            if self.idx[d] < self.shape[d] {
                break;
            }
            self.cur -= self.strides[d] * self.shape[d] as isize;
            self.idx[d] = 0;
        }
        Some(out)
    }

    fn all<F>(&mut self,mut f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        while let Some(z) = self.next(){
            if !f(z){
                return false
            }
        }
        return true
    }

    fn any<F>(&mut self,mut f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        while let Some(z) = self.next(){
            if f(z){
                return true 
            }
        }
        return false 
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.rem, Some(self.rem))
    }
}
