use polars::prelude::*;
use crate::conversion::{series_to_data_column, series_to_tensor,series_to_tensor_2d};
use crate::error::{DataloaderError, Result};
use crate::io::{load_csv, load_excel, load_file};
use crate::types::LoadOptions;
use crate::types::{DataColumn, DataValue};
use std::path::Path;
use Tensors::dtype::Element;
use Tensors::tensor::Tensor;
use crate::preprocessing::{value_counts as count_values, LabelEncoder};

/// Main DataFrame wrapper over Polars 
#[derive(Clone)]
pub struct DataFrame {
    inner: polars::frame::DataFrame,
}

impl std::fmt::Debug for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DataFrame(shape={:?}, columns={:?})", self.shape(), self.cols())
    }
}

impl DataFrame {

    /// Create a new DataFrame from Polars DataFrame
    pub fn from_polars(df: polars::frame::DataFrame) -> Self { Self { inner: df } }

    /// Get the underlying Polars DataFrame
    pub fn as_polars(&self) -> &polars::frame::DataFrame { &self.inner }

    /// Get mutable reference to underlying Polars DataFrame
    pub fn as_mut_polars(&mut self) -> &mut polars::frame::DataFrame { &mut self.inner }

    /// Load from CSV file
    pub fn from_csv<P: AsRef<Path>>(path: P, options: &LoadOptions) -> Result<Self> { Ok(Self::from_polars(load_csv(path, options)?)) }

    /// Load from Excel file
    pub fn from_excel<P: AsRef<Path>>(path: P, options: &LoadOptions) -> Result<Self> { Ok(Self::from_polars(load_excel(path, options)?)) }

    /// Load from file (auto-detect type)
    pub fn from_file<P: AsRef<Path>>(path: P, options: &LoadOptions) -> Result<Self> { Ok(Self::from_polars(load_file(path, options)?)) }

    /// Create DataFrame from a vector of DataColumns
    pub fn from_columns(columns: Vec<DataColumn>) -> Result<Self> {
        let series_vec: Vec<Series> = columns.iter().map(|col| crate::conversion::data_column_to_series(col)).collect::<Result<_>>()?;
        Ok(Self::from_polars(polars::frame::DataFrame::new(series_vec)?))
    }
    
    /// No of rows
    pub fn height(&self) -> usize { self.inner.height() }

    /// No of cols
    pub fn width(&self) -> usize { self.inner.width() }

    /// Get shape (rows, columns)
    pub fn shape(&self) -> (usize, usize) { (self.height(), self.width()) }

    /// Split into training and test Dataframes. test_size is a fraction in (0, 1).
    pub fn train_test_split(&self, test_size: f64, shuffle: bool, seed: Option<u64>) -> Result<(Self, Self)> {
        if !(0.0 < test_size && test_size < 1.0) || !test_size.is_finite() { return Err(DataloaderError::Conversion("test_size must be between 0 and 1".to_string())); }
        let rows = self.height();
        if rows < 2 { return Err(DataloaderError::EmptyData); }
        let test_count = ((rows as f64 * test_size).ceil() as usize).clamp(1, rows - 1);
        let mut indices: Vec<usize> = (0..rows).collect();
        if shuffle { deterministic_shuffle(&mut indices, seed.unwrap_or(0)); }
        let mut test_mask = vec![false; rows];
        for index in indices.into_iter().take(test_count) { test_mask[index] = true; }
        self.split_by_mask(&test_mask)
    }

    /// Split while preserving the label proportions
    pub fn stratified_train_test_split(&self, test_size: f64, stratify_by: &str, seed: Option<u64>) -> Result<(Self, Self)> {
        if !(0.0 < test_size && test_size < 1.0) || !test_size.is_finite() { return Err(DataloaderError::Conversion("test_size must be between 0 and 1".to_string())); }
        let labels = self.column(stratify_by)?;
        if self.height() < 2 { return Err(DataloaderError::EmptyData); }
        let groups = count_values(&labels);
        if groups.iter().any(|(_, count)| *count < 2) { return Err(DataloaderError::Conversion("Every stratification category must have at least two rows".to_string())); }
        let mut test_mask = vec![false; self.height()];
        let mut rng = SplitMix64(seed.unwrap_or(0));
        for (category, count) in groups {
            let mut indices: Vec<usize> = labels.values.iter().enumerate().filter_map(|(index, value)| if *value == category { Some(index) } else { None }).collect();
            rng.shuffle(&mut indices);
            let test_count = ((count as f64 * test_size).round() as usize).clamp(1, count - 1);
            for index in indices.into_iter().take(test_count) { test_mask[index] = true; }
        }
        self.split_by_mask(&test_mask)
    }

    // Split the dataset by a boolmask
    pub fn split_by_mask(&self, test_mask: &[bool]) -> Result<(Self, Self)> {
        let mask = Series::new("split_mask", test_mask.to_vec());
        let mask = mask.bool().map_err(|error| DataloaderError::Conversion(error.to_string()))?;
        let test = self.inner.filter(mask)?;
        let train_mask: Vec<bool> = test_mask.iter().map(|value| !value).collect();
        let mask = Series::new("split_mask", train_mask);
        let mask = mask.bool().map_err(|error| DataloaderError::Conversion(error.to_string()))?;
        let train = self.inner.filter(mask)?;
        Ok((Self::from_polars(train), Self::from_polars(test)))
    }

    /// Names of categorical col
    pub fn categorical_columns(&self) -> Vec<String> { self.cols().into_iter().filter(|name| { self.column(name).map(|column| column.values.iter().any(|value| matches!(value, DataValue::String(_) | DataValue::Bool(_) | DataValue::Date(_) |DataValue::DateTime(_) | DataValue::Time(_)))).unwrap_or(false) }).collect() }

    /// Unique values in first-seen order, including null if present.
    pub fn unique_values(&self, column: &str) -> Result<Vec<DataValue>> {
        let values = self.column(column)?.values;
        let mut unique = Vec::new();
        for value in values { if !unique.contains(&value) { unique.push(value); } }
        Ok(unique)
    }

    /// Count each value including null if present.
    pub fn value_counts(&self, column: &str) -> Result<Vec<(DataValue, usize)>> {
        Ok(count_values(&self.column(column)?))
    }

    /// Fit a label encoder on one column
    pub fn label_encode(&self, column: &str, output_name: Option<&str>) -> Result<(Self, LabelEncoder)> {
        let src = self.column(column)?;
        let encoder = LabelEncoder::fit(&src);
        let encoded = encoder.transform(&src, output_name.unwrap_or(&format!("{column}_encoded")))?;
        let name = encoded.clone().name;
        Ok((self.with_column(&name, encoded)?, encoder))
    }

    /// Replace a categorical column with boolean dummy columns, like pandas `get_dummies`.
    /// Null values produce false in every dummy column.
    pub fn one_hot_encode(&self, column: &str) -> Result<Self> {
        let source = self.column(column)?;
        let categories: Vec<DataValue> = self.unique_values(column)?.into_iter().filter(|value| !matches!(value, DataValue::Null)).collect();
        let mut result = self.drop(&[column])?;
        for category in categories {
            let name = format!("{column}_{}", category);
            if result.cols().iter().any(|existing| existing == &name) { return Err(DataloaderError::Conversion(format!("One-hot output column '{name}' already exists"))); }
            let values = source.values.iter().map(|value| DataValue::Bool(*value == category)).collect();
            result = result.with_column(&name.clone(), DataColumn::new(name, values))?;
        }
        Ok(result)
    }

    /// Standardize a numeric col
    pub fn Standardize(&self, column: &str) -> Result<Self> {
        let src = self.column(column)?;
        let x = src.as_numeric_f64()?;
        let non_null: Vec<f64> = x.iter().flatten().copied().collect();
        if non_null.is_empty() { return Err(DataloaderError::EmptyData); }
        let mean = non_null.iter().sum::<f64>() / non_null.len() as f64;
        let std_dev = (non_null.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / non_null.len() as f64).sqrt();
        let scaled = x.into_iter().map(|value| match value {
            Some(value) if std_dev > 0.0 => DataValue::Float64((value - mean) / std_dev),
            Some(_) => DataValue::Float64(0.0),
            None => DataValue::Null,
        }).collect();
        self.with_column(column, DataColumn::new(column, scaled))
    }

    /// Scale a numeric column into [0, 1]
    pub fn MinMaxScale(&self, column: &str) -> Result<Self> {
        let source = self.column(column)?;
        let numbers = source.as_numeric_f64()?;
        let present: Vec<f64> = numbers.iter().flatten().copied().collect();
        if present.is_empty() { return Err(DataloaderError::EmptyData); }
        let min = present.iter().copied().fold(f64::INFINITY, f64::min);
        let max = present.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let scaled = numbers.into_iter().map(|value| match value {
            Some(value) if range > 0.0 => DataValue::Float64((value - min) / range),
            Some(_) => DataValue::Float64(0.0),
            None => DataValue::Null,
        }).collect();
        self.with_column(column, DataColumn::new(column, scaled))
    }

    /// Get col names
    pub fn cols(&self) -> Vec<String> { self.inner.get_column_names().iter().map(|x| x.to_string()).collect() }

    /// Select cols by name
    pub fn select(&self, names: &[&str]) -> Result<Self> { Ok(Self::from_polars(self.inner.select(names)?)) }

    /// Select cols by name as Datacolumn
    pub fn column(&self, name: &str) -> Result<DataColumn> {
        let series = self.inner.column(name).map_err(|_| DataloaderError::ColumnNotFound(name.to_string()))?;
        series_to_data_column(series)
    }

    /// Select cols as DataColumns
    pub fn columns(&self, names: &[&str]) -> Result<Vec<DataColumn>> { names.iter().map(|&name| self.column(name)).collect() }

    /// Filter rows
    pub fn filter<F>(&self, predicate: F) -> Result<Self> where F: Fn(&[DataValue]) -> bool{
        let h = self.height();
        let col_names = self.cols();
        let mut mask = Vec::with_capacity(h);
        for row_idx in 0..h {
            let row_values: Vec<DataValue> = col_names.iter().map(|name| { self.column(name).unwrap().values[row_idx].clone()}).collect();
            mask.push(predicate(&row_values));
        }
        let s = Series::new("mask", mask);
        let mask_bool = s.bool().map_err(|_| DataloaderError::Conversion("Failed to convert Series into bool".to_string()))?;
        let ret = self.inner.filter(mask_bool)?;
        Ok(Self::from_polars(ret))
    }

    /// Filter using Polars expression
    pub fn filter_expr(&self, expr: Expr) -> Result<Self> { Ok(Self::from_polars( self.inner.clone().lazy().filter(expr).collect()?)) }
    
    /// Map fxn over col
    pub fn map_column<F>(&self, column: &str, f: F) -> Result<Self> where F: Fn(&DataValue) -> DataValue{
        let mapped = DataColumn::new(column,self.column(column)?.values.iter().map(f).collect());
        let mut ret = self.inner.clone();
        ret.replace(column,crate::conversion::data_column_to_series(&mapped)?)?;
        Ok(Self::from_polars(ret))
    }

    /// Add a new col
    pub fn with_column(&self, _name: &str, column: DataColumn) -> Result<Self> {
        let mut ret = self.inner.clone();
        ret.with_column(crate::conversion::data_column_to_series(&column)?)?;
        Ok(Self::from_polars(ret))
    }
    pub fn add_column(&self,_name:&str,column:DataColumn) -> Result<Self>{ self.with_column(_name, column)  }
    pub fn add_col(&self,_name:&str,column:DataColumn) -> Result<Self>{ self.with_column(_name, column)  }

    /// Drop cols
    pub fn drop(&self, names: &[&str]) -> Result<Self> {
        let mut ret = self.inner.clone();
        for name in names { ret = ret.drop(name)?; }
        Ok(Self::from_polars(ret))
    }

    /// Rename col
    pub fn rename(&self, o: &str, n: &str) -> Result<Self> {
        let mut ret = self.inner.clone();
        ret.rename(o, n)?;
        Ok(Self::from_polars(ret))
    }

    /// Sort col
    pub fn sort(&self, column: &str, descending: bool) -> Result<Self> { Ok(Self::from_polars( self.inner.sort([column], SortMultipleOptions::default().with_order_descending_multi(vec![descending]))?)) }

    /// Group by col
    pub fn group_by(&self, by: &[&str]) -> Result<GroupBy<'_>> { Ok(GroupBy { inner:  self.inner.group_by(by).map_err(|e| DataloaderError::Polars(e))? }) }

    /// Get first n rows
    pub fn head(&self, n: usize) -> Result<Self> { Ok(Self::from_polars( self.inner.head(Some(n)))) }

    /// Get last n rows
    pub fn tail(&self, n: usize) -> Result<Self> { Ok(Self::from_polars( self.inner.tail(Some(n)))) }

    /// Get rows by idx from offset upto offset+length 
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> { Ok(Self::from_polars( self.inner.slice(offset as i64, length))) }

    /// Vertically concat with other df 
    pub fn concat_vertical(&self, other: &DataFrame) -> Result<Self> {
        let mut df = self.inner.clone();
        df.vstack_mut(&other.inner).map_err(|e| DataloaderError::Polars(e))?;
        Ok(Self::from_polars(df))
    }

    // Horizontally concat with other df
    pub fn concat_horizontal(&self, other: &DataFrame) -> Result<Self> {
        let df = polars::functions::concat_df_horizontal(&[self.inner.clone(), other.inner.clone()],true)?;
        Ok(Self::from_polars(df))
    }

    /// Apply polars expr to the df
    pub fn apply_expr(&self, expr: Expr) -> Result<Self> { Ok(Self::from_polars( self.inner.clone().lazy().select([expr]).collect()?)) }

    /// Get col as tensor
    pub fn column_as_tensor<T: Element>(&self, name: &str) -> Result<Tensor<T>> {
        let ret = self.inner.column(name).map_err(|_| DataloaderError::ColumnNotFound(name.to_string()))?;
        series_to_tensor(ret)
    }

    /// Get mutliple cols as 2d tensor
    pub fn columns_as_tensor_2d<T: Element>(&self, names: &[&str]) -> Result<Tensor<T>> {
        let series_vec: Vec<Series> = names.iter().map(|&name| { self.inner.column(name).map_err(|_| DataloaderError::ColumnNotFound(name.to_string())).cloned()}).collect::<Result<_>>()?;
        series_to_tensor_2d(&series_vec)
    }

    /// Get numeric cols only as 2d tensor
    pub fn numeric_as_tensor_2d<T: Element>(&self) -> Result<Tensor<T>> {
        let col_names = self.cols();
        let numeric_names: Vec<&str> = col_names.iter().filter(|&name| {
                if let Ok(col) = self.column(name) {
                    col.is_numeric()
                } else {
                    false
                }
            }).map(|s| s.as_str()).collect();
        if numeric_names.is_empty() { return Err(DataloaderError::Conversion("No numeric columns found".to_string())); }
        self.columns_as_tensor_2d(&numeric_names)
    }

    /// Get all cols as DataColumns
    pub fn as_data_columns(&self) -> Result<Vec<DataColumn>> { self.cols().iter().map(|name| self.column(name)).collect() }

    /// Drop duplicate rows
    // Made using AI
    pub fn unique(&self, subset: Option<&[&str]>) -> Result<Self> {
        let subset_vec = subset.map(|s| s.iter().map(|&x| x.to_string()).collect::<Vec<_>>());
        let df = self.inner.unique(subset_vec.as_deref(), UniqueKeepStrategy::First, None)?;
        Ok(Self::from_polars(df))
    }

    /// Count nulls per col
    pub fn null_count(&self) -> Self { Self::from_polars(self.inner.null_count()) }

    /// Fill null with default
    pub fn fill_null(&self, strategy: FillNullStrategy) -> Result<Self> { Ok(Self::from_polars(self.inner.fill_null(strategy)?)) }

    /// Drop rows with null 
    pub fn drop_nulls(&self) -> Result<Self> { Ok(Self::from_polars(self.inner.drop_nulls::<String>(None)?)) }

    /// Cross join
    pub fn cross_join(&self, other: &DataFrame) -> Result<Self> { Ok(Self::from_polars(self.inner.cross_join(&other.inner,None,None)?)) }

    /// Inner join
    pub fn inner_join(&self, other: &DataFrame, on: &str) -> Result<Self> { Ok(Self::from_polars(self.inner.join(&other.inner, [on], [on], JoinArgs::new(JoinType::Inner))?)) }

    /// Left join
    pub fn left_join(&self, other: &DataFrame, on: &str) -> Result<Self> { Ok(Self::from_polars(self.inner.join(&other.inner, [on], [on], JoinArgs::new(JoinType::Left))?)) }

    /// Right join
    pub fn right_join(&self, other: &DataFrame, on: &str) -> Result<Self> { Ok(Self::from_polars(self.inner.join(&other.inner, [on], [on], JoinArgs::new(JoinType::Right))?)) }

    /// Full join
    pub fn full_join(&self, other: &DataFrame, on: &str) -> Result<Self> { Ok(Self::from_polars(self.inner.join(&other.inner, [on], [on], JoinArgs::new(JoinType::Full))?)) }

    pub fn join(&self,other: &DataFrame,on: Option<&str>,join_t: JoinType) -> Result<Self>{
        match join_t{
            JoinType::Cross => {self.cross_join(other)}
            JoinType::Left => {self.left_join(other,on.ok_or(DataloaderError::UnsupportedOperation("Require the on field for left join".to_string()))?)}
            JoinType::Right => {self.right_join(other,on.ok_or(DataloaderError::UnsupportedOperation("Require the on field for right join".to_string()))?)}
            JoinType::Inner => {self.inner_join(other,on.ok_or(DataloaderError::UnsupportedOperation("Require the on field for inner join".to_string()))?)}
            JoinType::Full => {self.full_join(other,on.ok_or(DataloaderError::UnsupportedOperation("Require the on field for full join".to_string()))?)}
        }
    }

    /// aggregate using polars expr 
    pub fn aggregate(&self, aggs: &[Expr]) -> Result<Self> { Ok(Self::from_polars(self.inner.clone().lazy().group_by([col("*")]).agg(aggs).collect()?)) }
}

/// group by wrapper
pub struct GroupBy<'a> {
    inner: polars::frame::group_by::GroupBy<'a>,
}

impl<'a> GroupBy<'a> {
    
    /// Get first row of each group
    pub fn first(&self) -> Result<DataFrame> { Ok(DataFrame::from_polars(self.inner.first()?)) }

    /// Get last row of each group
    pub fn last(&self) -> Result<DataFrame> { Ok(DataFrame::from_polars(self.inner.last()?)) }

    /// Sum of each group
    pub fn sum(&self) -> Result<DataFrame> { Ok(DataFrame::from_polars(self.inner.sum()?)) }

    /// Count of each group
    pub fn count(&self) -> Result<DataFrame> { Ok(DataFrame::from_polars(self.inner.count()?)) }

    /// Mean of each group
    pub fn mean(&self) -> Result<DataFrame> { Ok(DataFrame::from_polars(self.inner.mean()?)) }

    /// Min of each group
    pub fn min(&self) -> Result<DataFrame> { Ok(DataFrame::from_polars(self.inner.min()?)) }

    /// Max of each group
    pub fn max(&self) -> Result<DataFrame> { Ok(DataFrame::from_polars(self.inner.max()?)) }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let other = (self.next() as usize) % (index + 1);
            values.swap(index, other);
        }
    }
}

fn deterministic_shuffle<T>(values: &mut [T], seed: u64) {
    SplitMix64(seed).shuffle(values);
}

