use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use std::fmt::{self, Display};

// Enum for dataframe types
#[derive(Clone, Debug, PartialEq)]
pub enum DataValue {
    Null,
    Bool(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    String(String),
    Date(NaiveDate),
    DateTime(NaiveDateTime),
    Time(NaiveTime),
}

impl DataValue {
    /// Check if numeric for tensor conversion
    pub fn is_numeric(&self) -> bool {
        matches!(self, DataValue::Int8(_) | DataValue::Int16(_) | DataValue::Int32(_) | DataValue::Int64(_) | DataValue::UInt8(_) | DataValue::UInt16(_) | DataValue::UInt32(_) | DataValue::UInt64(_) | DataValue::Float32(_) | DataValue::Float64(_))
    }

    /// Convert to f64 if numeric
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            DataValue::Int8(v) => Some(*v as f64),
            DataValue::Int16(v) => Some(*v as f64),
            DataValue::Int32(v) => Some(*v as f64),
            DataValue::Int64(v) => Some(*v as f64),
            DataValue::UInt8(v) => Some(*v as f64),
            DataValue::UInt16(v) => Some(*v as f64),
            DataValue::UInt32(v) => Some(*v as f64),
            DataValue::UInt64(v) => Some(*v as f64),
            DataValue::Float32(v) => Some(*v as f64),
            DataValue::Float64(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the type as str 
    pub fn type_name(&self) -> &'static str {
        match self {
            DataValue::Null => "null",
            DataValue::Bool(_) => "bool",
            DataValue::Int8(_) => "i8",
            DataValue::Int16(_) => "i16",
            DataValue::Int32(_) => "i32",
            DataValue::Int64(_) => "i64",
            DataValue::UInt8(_) => "u8",
            DataValue::UInt16(_) => "u16",
            DataValue::UInt32(_) => "u32",
            DataValue::UInt64(_) => "u64",
            DataValue::Float32(_) => "f32",
            DataValue::Float64(_) => "f64",
            DataValue::String(_) => "string",
            DataValue::Date(_) => "date",
            DataValue::DateTime(_) => "datetime",
            DataValue::Time(_) => "time",
        }
    }
}

impl Display for DataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataValue::Null => write!(f, "null"),
            DataValue::Bool(v) => write!(f, "{}", v),
            DataValue::Int8(v) => write!(f, "{}", v),
            DataValue::Int16(v) => write!(f, "{}", v),
            DataValue::Int32(v) => write!(f, "{}", v),
            DataValue::Int64(v) => write!(f, "{}", v),
            DataValue::UInt8(v) => write!(f, "{}", v),
            DataValue::UInt16(v) => write!(f, "{}", v),
            DataValue::UInt32(v) => write!(f, "{}", v),
            DataValue::UInt64(v) => write!(f, "{}", v),
            DataValue::Float32(v) => write!(f, "{}", v),
            DataValue::Float64(v) => write!(f, "{}", v),
            DataValue::String(v) => write!(f, "{}", v),
            DataValue::Date(v) => write!(f, "{}", v),
            DataValue::DateTime(v) => write!(f, "{}", v),
            DataValue::Time(v) => write!(f, "{}", v),
        }
    }
}


/// Column with mlti type data
#[derive(Clone, Debug)]
pub struct DataColumn {
    pub name: String,
    pub values: Vec<DataValue>,
}

impl DataColumn {
    /// Create a new DataColumn
    pub fn new(name: impl Into<String>, values: Vec<DataValue>) -> Self { Self {name: name.into(),values} }

    /// Get len of DataColumn
    pub fn len(&self) -> usize { self.values.len() }
    
    /// Check if DataColumn is empty
    pub fn is_empty(&self) -> bool { self.values.is_empty() }

    /// Get the col's dtype
    pub fn dtype(&self) -> &'static str {
        if self.values.is_empty() { return "unknown";}
        // for simplicity we determine col type by 1st elem in col
        // since currently no support for ambigious objects 
        self.values[0].type_name()
    }

    /// Check if all vals in column are numeric
    pub fn is_numeric(&self) -> bool { self.values.iter().all(|v| v.is_numeric() || matches!(v, DataValue::Null))}

    /// Get numeric values as f64, skipping nulls
    pub fn as_numeric_f64(&self) -> Result<Vec<Option<f64>>, crate::error::DataloaderError> {
        if !self.is_numeric() { return Err(crate::error::DataloaderError::Conversion(format!("Column '{}' is not numeric",self.name))); }
        Ok(self.values.iter().map(|v| v.as_f64()).collect())
    }
}

/// Options for loading data from csv and excels
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    pub has_header: bool,
    pub separator: Option<char>,
    pub skip_rows: usize,
    pub null_values: Option<Vec<String>>,
    pub dtype_overrides: Option<Vec<(String, String)>>,
}

impl LoadOptions {
    pub fn new() -> Self { Self::default() }

    /// Set flag of header and return Self
    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Set seperator and return Self
    pub fn with_separator(mut self, sep: char) -> Self {
        self.separator = Some(sep);
        self
    }

    /// Set no of starting rows to skip and return Self
    pub fn with_skip_rows(mut self, skip: usize) -> Self {
        self.skip_rows = skip;
        self
    }

    /// Set following  and return Self
    pub fn with_null_values(mut self, nulls: Vec<String>) -> Self {
        self.null_values = Some(nulls);
        self
    }

    pub fn with_dtype_overrides(mut self, overrides: Vec<(String, String)>) -> Self {
        self.dtype_overrides = Some(overrides);
        self
    }
}
