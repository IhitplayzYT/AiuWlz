
//! One-vs-rest multiclass SVM classifier.
//!
//! The classifier implementation trains one binary margin model per class.
pub use super::svm_classifier::{
 SvmClassifier, SvmClassifierModel,
};

pub use crate::common::{Configured,Trained};
pub use super::common::SvmConfig;