use polars::prelude::*;
use crate::conversion::{series_to_data_column, series_to_tensor,series_to_tensor_2d};
use crate::error::{DataloaderError, Result};
use crate::io::{load_csv, load_excel, load_file};
use crate::types::LoadOptions;
use crate::types::{DataColumn, DataValue};
use std::path::Path;
use Tensors::dtype::Element;
use Tensors::tensor::Tensor;

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

