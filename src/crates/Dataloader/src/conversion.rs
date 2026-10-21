use std::collections::{HashMap, HashSet};
use std::hash::DefaultHasher;

use crate::error::{DataloaderError, Result};
use crate::types::{DataColumn, DataValue};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use polars::prelude::*;
use Tensors::tensor::Tensor;
use Tensors::dtype::{Element, Float};

/// Convert Polars Series to DataColumn
pub fn series_to_data_column(series: &Series) -> Result<DataColumn> {
    let name = series.name().to_string();
    let values = match series.dtype() {
        DataType::Null => vec![DataValue::Null; series.len()],
        DataType::Boolean => {
            series.bool().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(b) => DataValue::Bool(b),
                None => DataValue::Null,
            }).collect()
        }
        DataType::Int8 => {
            series.i8().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(v) => DataValue::Int8(v),
                None => DataValue::Null,
            }).collect()
        }
        DataType::Int16 => {
            series.i16().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(v) => DataValue::Int16(v),
                None => DataValue::Null,
            }).collect()
        }
        DataType::Int32 => {
            series.i32().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(v) => DataValue::Int32(v),
                None => DataValue::Null,
            }).collect()
        }
        DataType::Int64 => {
            series.i64().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(v) => DataValue::Int64(v),
                None => DataValue::Null,
            }).collect()
        }
        DataType::UInt8 => {
            series.u8().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(v) => DataValue::UInt8(v),
                None => DataValue::Null,
            }).collect()
        }
        DataType::UInt16 => {
            series.u16().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(v) => DataValue::UInt16(v),
                None => DataValue::Null,
            }).collect()
        }
        DataType::UInt32 => {
            series.u32().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(v) => DataValue::UInt32(v),
                None => DataValue::Null,
            }).collect()
        }
        DataType::UInt64 => {
            series.u64().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(v) => DataValue::UInt64(v),
                None => DataValue::Null,
            }).collect()
        }
        DataType::Float32 => {
            series.f32().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(v) => DataValue::Float32(v),
                None => DataValue::Null,
            }).collect()
        }
        DataType::Float64 => {
            series.f64().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(v) => DataValue::Float64(v),
                None => DataValue::Null,
            }).collect()
        }
        DataType::String => {
            series.str().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(s) => DataValue::String(s.to_string()),
                None => DataValue::Null,
            }).collect()
        }
        DataType::Date => {
            series.date().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(d) => DataValue::Date(NaiveDate::from_num_days_from_ce_opt(d).unwrap_or(NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())),
                None => DataValue::Null,
            }).collect()
        }
        DataType::Datetime(_, _) => {
            series.datetime().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(dt) => DataValue::DateTime(NaiveDateTime::from_timestamp_opt(dt, 0).unwrap_or_else(|| NaiveDateTime::from_timestamp_opt(0, 0).unwrap())),
                None => DataValue::Null,
            }).collect()
        }
        DataType::Time => {
            series.time().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| match opt {
                Some(t) => DataValue::Time(NaiveTime::from_num_seconds_from_midnight_opt(t as u32, 0).unwrap_or(NaiveTime::from_hms_opt(0, 0, 0).unwrap())),
                None => DataValue::Null,
            }).collect()
        }
        dt => { return Err(DataloaderError::UnsupportedOperation(format!("Unsupported dtype: {:?}",dt)))}
    };

    Ok(DataColumn::new(name, values))
}

/// Convert DataColumn to a Polars Series
pub fn data_column_to_series(column: &DataColumn) -> Result<Series> {
    if column.is_empty() { return Ok(Series::new(&column.name, &[None as Option<i32>])); }
    match &column.values[0] {
        DataValue::Null => {
            let first_non_null = column.values.iter().find(|v| !matches!(v, DataValue::Null));
            if let Some(DataValue::Bool(_)) = first_non_null {
                let vals: Vec<Option<bool>> = column.values.iter().map(|v| match v {
                        DataValue::Bool(b) => Some(*b),
                        _ => None,
                    }).collect();
                Ok(Series::new(&column.name, vals))
            } else {
                let vals: Vec<Option<i32>> = column.values.iter().map(|v| match v {
                        DataValue::Int32(i) => Some(*i),
                        _ => None,
                    }).collect();
                Ok(Series::new(&column.name, vals))
            }
        }
        DataValue::Bool(_) => {
            let vals: Vec<Option<bool>> = column.values.iter().map(|v| match v {
                    DataValue::Bool(b) => Some(*b),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::Int8(_) => {
            let vals: Vec<Option<i8>> = column.values.iter().map(|v| match v {
                    DataValue::Int8(i) => Some(*i),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::Int16(_) => {
            let vals: Vec<Option<i16>> = column.values.iter().map(|v| match v {
                    DataValue::Int16(i) => Some(*i),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::Int32(_) => {
            let vals: Vec<Option<i32>> = column.values.iter().map(|v| match v {
                    DataValue::Int32(i) => Some(*i),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::Int64(_) => {
            let vals: Vec<Option<i64>> = column.values.iter().map(|v| match v {
                    DataValue::Int64(i) => Some(*i),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::UInt8(_) => {
            let vals: Vec<Option<i32>> = column.values.iter().map(|v| match v {
                    DataValue::UInt8(i) => Some(*i as i32),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::UInt16(_) => {
            let vals: Vec<Option<i32>> = column.values.iter().map(|v| match v {
                    DataValue::UInt16(i) => Some(*i as i32),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::UInt32(_) => {
            let vals: Vec<Option<u32>> = column.values.iter().map(|v| match v {
                    DataValue::UInt32(i) => Some(*i),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::UInt64(_) => {
            let vals: Vec<Option<u64>> = column.values.iter().map(|v| match v {
                    DataValue::UInt64(i) => Some(*i),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::Float32(_) => {
            let vals: Vec<Option<f32>> = column.values.iter().map(|v| match v {
                    DataValue::Float32(f) => Some(*f),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::Float64(_) => {
            let vals: Vec<Option<f64>> = column.values.iter().map(|v| match v {
                    DataValue::Float64(f) => Some(*f),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::String(_) => {
            let vals: Vec<Option<String>> = column.values.iter().map(|v| match v {
                    DataValue::String(s) => Some(s.clone()),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::Date(_) => {
            let vals: Vec<Option<NaiveDate>> = column.values.iter().map(|v| match v {
                    DataValue::Date(d) => Some(*d),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::DateTime(_) => {
            let vals: Vec<Option<NaiveDateTime>> = column.values.iter().map(|v| match v {
                    DataValue::DateTime(dt) => Some(*dt),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        }
        DataValue::Time(_) => {
            let vals: Vec<Option<NaiveTime>> = column.values.iter().map(|v| match v {
                    DataValue::Time(t) => Some(*t),
                    _ => None,
                }).collect();
            Ok(Series::new(&column.name, vals))
        },
        DataValue::Vec(_) => {
        let vals: Vec<Option<Vec<f64>>> = column.values.iter().map(|v| match v {
                DataValue::Vec(t) => Some(t.clone()),
                _ => None,
            }).collect();
            Ok(Series::new(&column.name, vals.iter().map(|v| {v.as_ref().map(|values| {Series::new("".into(), values.as_slice())})
        }).collect::<Vec<Option<Series>>>()))
    }
    }
}

/// Convert a numeric DataColumn to a Tensor
pub fn data_column_to_tensor<T: Element>(column: &DataColumn) -> Result<Tensor<T>> {
    if !column.is_numeric() { return Err(DataloaderError::Conversion(format!("Column '{}' is non-numeric and can't be converted to Tensor",column.name))); }
    let data: Vec<T> = column.values.iter().map(|v| { v.as_f64().map(|f| T::from_f64(f)).ok_or_else(|| DataloaderError::Conversion("Null value in numeric column".to_string()))}).collect::<Result<Vec<_>>>()?;
    Tensor::from_1d(data).map_err(|x| DataloaderError::Tensor(format!("Tensor error: {x}")))
}

/// Convert a Polars Series to a Tensor (numeric only)
pub fn series_to_tensor<T: Element>(series: &Series) -> Result<Tensor<T>> {
    let data: Vec<T> = match series.dtype() {
        DataType::Int8 => series.i8().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| opt.map(|v| T::from_f64(v as f64))).collect::<Option<Vec<_>>>().ok_or_else(|| DataloaderError::Conversion("Null val in series".to_string()))?,
        DataType::Int16 => series.i16().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| opt.map(|v| T::from_f64(v as f64))).collect::<Option<Vec<_>>>().ok_or_else(|| DataloaderError::Conversion("Null val in series".to_string()))?,
        DataType::Int32 => series.i32().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| opt.map(|v| T::from_f64(v as f64))).collect::<Option<Vec<_>>>().ok_or_else(|| DataloaderError::Conversion("Null val in series".to_string()))?,
        DataType::Int64 => series.i64().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| opt.map(|v| T::from_f64(v as f64))).collect::<Option<Vec<_>>>().ok_or_else(|| DataloaderError::Conversion("Null val in series".to_string()))?,
        DataType::UInt8 => series.u8().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| opt.map(|v| T::from_f64(v as f64))).collect::<Option<Vec<_>>>().ok_or_else(|| DataloaderError::Conversion("Null val in series".to_string()))?,
        DataType::UInt16 => series.u16().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| opt.map(|v| T::from_f64(v as f64))).collect::<Option<Vec<_>>>().ok_or_else(|| DataloaderError::Conversion("Null val in series".to_string()))?,
        DataType::UInt32 => series.u32().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| opt.map(|v| T::from_f64(v as f64))).collect::<Option<Vec<_>>>().ok_or_else(|| DataloaderError::Conversion("Null val in series".to_string()))?,
        DataType::UInt64 => series.u64().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| opt.map(|v| T::from_f64(v as f64))).collect::<Option<Vec<_>>>().ok_or_else(|| DataloaderError::Conversion("Null val in series".to_string()))?,
        DataType::Float32 => series.f32().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| opt.map(|v| T::from_f64(v as f64))).collect::<Option<Vec<_>>>().ok_or_else(|| DataloaderError::Conversion("Null val in series".to_string()))?,
        DataType::Float64 => series.f64().map_err(|e| DataloaderError::Conversion(e.to_string()))?.into_iter().map(|opt| opt.map(|v| T::from_f64(v))).collect::<Option<Vec<_>>>().ok_or_else(|| DataloaderError::Conversion("Null val in series".to_string()))?,
        dt => { return Err(DataloaderError::Conversion(format!("Cannot convert non-numeric type {:?} to Tensor",dt)))
        }
    };
    Tensor::from_1d(data).map_err(|x| DataloaderError::Tensor(format!("Tensor error: {x}")))
}

/// Convert multiple numeric DataColumns to a 2D Tensor
pub fn data_columns_to_tensor_2d<T: Element>(columns: &[DataColumn]) -> Result<Tensor<T>> {
    if columns.is_empty() { return Err(DataloaderError::EmptyData); }
    let (m,n) = (columns.len(),columns[0].len());
    if !columns.iter().all(|c| c.len() == n) { return Err(DataloaderError::ShapeMismatch("All col should be same len".to_string())); }
    if !columns.iter().all(|c| c.is_numeric()) {return Err(DataloaderError::Conversion("All cols should be numeric".to_string()));}

    let mut data = Vec::with_capacity(m * n );
    for row in 0..n {
        for col in columns {
            match &col.values[row] {
                DataValue::Null => { return Err(DataloaderError::Conversion("Null val in col".to_string())) }
                v => {
                    if let Some(f) = v.as_f64() {
                        data.push(T::from_f64(f));
                    } else {
                        return Err(DataloaderError::Conversion("Non-numeric val in col".to_string()));
                    }
                }
            }
        }
    }
    Tensor::from_2d(data.chunks(m).map(|chunk| chunk.to_vec()).collect()).map_err(|x| DataloaderError::Tensor(format!("Loader failed due to Tensor error: {x}")))
}

/// Convert multiple Polars Series to a 2D Tensor
pub fn series_to_tensor_2d<T: Element>(series_vec: &[Series]) -> Result<Tensor<T>> {
    if series_vec.is_empty() { return Err(DataloaderError::EmptyData); }
    let  (m,n)= (series_vec.len(),series_vec[0].len());
    if !series_vec.iter().all(|s| s.len() == n) { return Err(DataloaderError::ShapeMismatch("All series should be same len".to_string())); }
    let mut data = Vec::with_capacity(m * n);
    for row in 0..n {
        for series in series_vec {
            let val = match series.dtype() {
                DataType::Int64 => series.i64().map_err(|e| DataloaderError::Conversion(e.to_string()))?.get(row).map(|v| T::from_f64(v as f64)),
                DataType::Float64 => series.f64().map_err(|e| DataloaderError::Conversion(e.to_string()))?.get(row).map(|v| T::from_f64(v)),
                DataType::Float32 => series.f32().map_err(|e| DataloaderError::Conversion(e.to_string()))?.get(row).map(|v| T::from_f64(v as f64)),
                DataType::Int32 => series.i32().map_err(|e| DataloaderError::Conversion(e.to_string()))?.get(row).map(|v| T::from_f64(v as f64)),
                _ => { return Err(DataloaderError::Conversion(format!("Unsupported type {:?} for makign tensor",series.dtype()))) }
            };

            match val {
                Some(v) => data.push(v),
                None => { return Err(DataloaderError::Conversion("Null val in series".to_string())) }
            }
        }
    }
    Tensor::from_2d(data.chunks(m).map(|chunk| chunk.to_vec()).collect()).map_err(|x| DataloaderError::Tensor(format!("Tensor error: {x}")))
}


pub fn Onehot(c:&DataColumn) -> DataColumn{
    let mut hsher = DefaultHasher::new();
    let mut seen = HashSet::new();
    c.values.iter().for_each(|x| { 
        seen.insert(x.id(&mut hsher));
    });
    let l = seen.len();
    let mp = seen.into_iter().enumerate().map(|(i,x)| {
        let mut v = vec![0.0;l];
        v[i] = 1.0;
        (x,v)
    }).collect::<HashMap<u64,Vec<f64>>>();
    let mut k = Vec::with_capacity(c.len());
    c.values.iter().enumerate().for_each(|(i,x)| { 
        k[i] = DataValue::Vec(mp.get(&x.id(&mut hsher)).unwrap().clone());
    });
    
    DataColumn::new(c.name.clone(),k)
}

pub fn Ordinal(c:&DataColumn) -> DataColumn{
    let mut hsher = DefaultHasher::new();
    let mut seen = HashSet::new();
    c.values.iter().for_each(|x| { 
        seen.insert(x.id(&mut hsher));
    });
    let l = seen.len();
    let mp = seen.into_iter().enumerate().map(|(i,x)| {
        (x,vec![(i+1) as f64])
    }).collect::<HashMap<u64,Vec<f64>>>();

    let mut k = Vec::with_capacity(c.len());
    c.values.iter().enumerate().for_each(|(i,x)| { 
        k[i] = DataValue::Vec(mp.get(&x.id(&mut hsher)).unwrap().clone());
    });
    
    DataColumn::new(c.name.clone(),k)
}


pub fn col_to_tensor<T:Float>(c: &DataColumn) -> Result<Tensor<f64>>{
   if c.is_tensorizable(){       
        Ok(Tensor::from_2d( c.values.clone().into_iter().filter_map(|x| if matches!(x,DataValue::Vec(_)){Some(match x {DataValue::Vec(p) => p,_=>{panic!("Not possible")}})}else{None}).collect::<Vec<Vec<f64>>>()).map_err(|x|{ DataloaderError::Tensor(x.to_string())})?)
   }else if c.is_numeric(){
        Ok(Tensor::from_1d( c.values.clone().into_iter().filter_map(|x| if matches!(x,DataValue::Null){None}else{Some(x.as_f64().unwrap())}).collect::<Vec<f64>>()).map_err(|x|{ DataloaderError::Tensor(x.to_string())})?)
   }else{
    return Err(DataloaderError::InvalidDataType(c.name.clone(), "DataValue::Vec or DataValue::is_numeric() type".to_string(), "Non Tensorizable oir Non Numeirc".to_string()));
   }
}
