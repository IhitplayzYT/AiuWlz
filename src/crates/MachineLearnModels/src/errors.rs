use Dataloader::error::DataloaderError;
use Tensors::error::TensorError;

#[derive(Debug)]
pub enum ModelError{
    TensorError(TensorError),
    DataLoaderError(DataloaderError),
    ConfigError(String),
    ConceptError(String),
    InvalidData(String),
    MissingData(String)
}

impl From<TensorError> for ModelError {
    fn from(err: TensorError) -> Self {
        ModelError::TensorError(err)
    }
}

impl From<DataloaderError> for ModelError{
    fn from(err: DataloaderError) -> Self {
        ModelError::DataLoaderError(err)
    }
}
