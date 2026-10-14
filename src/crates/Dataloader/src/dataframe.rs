use polars::prelude::*;


/// Main DataFrame wrapper over Polars 
#[derive(Clone)]
pub struct DataFrame {
    inner: polars::frame::DataFrame,
}

