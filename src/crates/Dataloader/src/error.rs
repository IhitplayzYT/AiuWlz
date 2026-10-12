use thiserror::Error;
pub type Result<T> = std::result::Result<T, DataloaderError>;

#[derive(Error, Debug)]
pub enum DataloaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),
    #[error("Tensor error: {0}")]
    Tensor(String),
    #[error("Conversion error: {0}")]
    Conversion(String),
    #[error("Column not found: {0}")]
    ColumnNotFound(String),
    #[error("Invalid data type for column '{0}': expected {1}, found {2}")]
    InvalidDataType(String, String, String),
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    #[error("Empty data")]
    EmptyData,
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
}
