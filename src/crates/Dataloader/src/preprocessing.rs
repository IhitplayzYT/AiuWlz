use crate::error::{DataloaderError, Result};
use crate::types::{DataColumn, DataValue};

/// Null values stay null when transformed. Values not observed during fitting return an error
#[derive(Clone, Debug)]
pub struct LabelEncoder {
    classes: Vec<DataValue>,
}

impl LabelEncoder {

    pub fn fit(column: &DataColumn) -> Self {
        let mut classes = Vec::new();
        for value in &column.values {
            if matches!(value, DataValue::Null) { continue; }
            if !classes.contains(value) { classes.push(value.clone()); }
        }
        Self { classes }
    }

    pub fn classes(&self) -> &[DataValue] { &self.classes }

    /// Convert categories to zero-based integer codes
    pub fn transform(&self, column: &DataColumn, output_name: impl Into<String>) -> Result<DataColumn> {
        let mut encoded = Vec::with_capacity(column.len());
        for value in &column.values {
            if matches!(value, DataValue::Null) {
                encoded.push(DataValue::Null);
            } else if let Some(index) = self.classes.iter().position(|class| class == value) {
                encoded.push(DataValue::Int32(index as i32));
            } else {
                return Err(DataloaderError::Conversion(format!("Unknown category '{value}' for label encoder")));
            }
        }
        Ok(DataColumn::new(output_name, encoded))
    }

    /// Restore values from integer codes
    pub fn inverse_transform(&self, column: &DataColumn, output_name: impl Into<String>) -> Result<DataColumn> {
        let mut decoded = Vec::with_capacity(column.len());
        for value in &column.values {
            if matches!(value, DataValue::Null) {
                decoded.push(DataValue::Null);
                continue;
            }
            let index = value.as_f64().filter(|number| { number.is_finite() && *number >= 0.0 && number.fract() == 0.0 }).map(|number| number as usize).ok_or_else(|| DataloaderError::Conversion("Label codes must be non-negative integers".to_string()))?;
            let class = self.classes.get(index).ok_or_else(|| {DataloaderError::Conversion(format!("Label code {index} is outside the fitted classes"))})?;
            decoded.push(class.clone());
        }
        Ok(DataColumn::new(output_name, decoded))
    }
}


pub fn value_counts(column: &DataColumn) -> Vec<(DataValue, usize)> {
    let mut counts: Vec<(DataValue, usize)> = Vec::new();
    for v in &column.values {
        if let Some((_, count)) = counts.iter_mut().find(|(category, _)| category == v) {
            *count += 1;
        } else {
            counts.push((v.clone(), 1));
        }
    }
    counts
}

