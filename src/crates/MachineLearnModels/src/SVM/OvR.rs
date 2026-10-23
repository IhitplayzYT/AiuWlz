
//! One-vs-rest multiclass SVM classifier.
//!
//! The classifier implementation trains one binary margin model per class.
pub use super::svm_classifier::{
 SvmClassifier, SvmClassifierConfig, SvmClassifierModel,
};

pub use crate::common::{Configured,Trained};