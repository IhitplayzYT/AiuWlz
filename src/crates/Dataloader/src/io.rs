use crate::error::{DataloaderError, Result};
use crate::types::LoadOptions;
use calamine::{Reader, Xlsx, open_workbook};
use polars::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Load data from csv file
pub fn load_csv<P: AsRef<Path>>(path: P, _options: &LoadOptions) -> Result<DataFrame> {
    let file = File::open(path.as_ref())?;
    let df = CsvReader::new(file).finish().map_err(|e| DataloaderError::Polars(e))?;
    Ok(df)
}

/// Load data from execl file
/// Fxn internally converts excel sheet to csv then parse it using polars 
pub fn load_excel<P: AsRef<Path>>(path: P, options: &LoadOptions) -> Result<DataFrame> {
    let mut sheets: Xlsx<_> = open_workbook(path.as_ref()).map_err(|e| DataloaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    let range = sheets.worksheet_range_at(0).ok_or_else(|| DataloaderError::Conversion("Worksheet not found".to_string()))?.map_err(|e| DataloaderError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    let rows: Vec<Vec<String>> = range.rows().skip(options.skip_rows).map(|row| {row.iter().map(|x| x.to_string()).collect()}).collect();
    if rows.is_empty() { return Err(DataloaderError::EmptyData); }
    let sep = options.separator.unwrap_or(',');
    let csv_content: String = rows.iter().map(|row| row.join(&sep.to_string())).collect::<Vec<_>>().join("\n");
    let df = CsvReader::new(std::io::Cursor::new(csv_content)).finish().map_err(|e| DataloaderError::Polars(e))?;
    Ok(df)
}

/// Load buffered data from a CSV file
pub fn load_csv_sync<P: AsRef<Path>>(path: P, _options: &LoadOptions) -> Result<DataFrame> {
    let file = File::open(path.as_ref())?;
    let mut buf_reader = BufReader::new(file);
    let mut content = String::new();
    buf_reader.read_to_string(&mut content)?;
    let df = CsvReader::new(std::io::Cursor::new(content)).finish().map_err(|e| DataloaderError::Polars(e))?;
    Ok(df)
}

/// Load buffered data from an Excel file
pub fn load_excel_sync<P: AsRef<Path>>(path: P, options: &LoadOptions) -> Result<DataFrame> {
    load_excel(path, options)
}

/// Save polars Dataframe to csv
pub fn save_csv<P: AsRef<Path>>(df: &DataFrame, path: P, options: &LoadOptions) -> Result<()> {
    let file = File::create(path.as_ref())?;
    let mut csv_writer = CsvWriter::new(file).include_header(options.has_header).with_separator(options.separator.unwrap_or(',') as u8);
    let mut df_clone = df.clone();
    csv_writer.finish(&mut df_clone).map_err(|e| DataloaderError::Polars(e))?;
    Ok(())
}

/// File type from ext
pub fn detect_ftype<P: AsRef<Path>>(path: P) -> Result<FileType> {
    let path = path.as_ref();
    let ext = path.extension().and_then(|ext| ext.to_str()).ok_or_else(|| DataloaderError::Conversion("File has no extension".to_string()))?;
    match ext.to_lowercase().as_str() {
        "csv" => Ok(FileType::Csv),
        "xlsx" | "xls" => Ok(FileType::Excel),
        _ => Err(DataloaderError::Conversion(format!("Unsupported file ext: {}\nOnly csv and excel are supported as of now",ext))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Csv,
    Excel,
}

/// Load data from file
pub fn load_file<P: AsRef<Path>>(path: P, options: &LoadOptions) -> Result<DataFrame> {
    match detect_ftype(&path)? {
        FileType::Csv => load_csv(path, options),
        FileType::Excel => load_excel(path, options),
    }
}

/// Load buffered data from file
pub fn load_file_sync<P: AsRef<Path>>(path: P, options: &LoadOptions) -> Result<DataFrame> {
    match detect_ftype(&path)? {
        FileType::Csv => load_csv_sync(path, options),
        FileType::Excel => load_excel_sync(path, options),
    }
}
