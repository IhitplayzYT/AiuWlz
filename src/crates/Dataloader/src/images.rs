// pixel value starts with little endian width and height (four bytes each), then
// contains row-major RGBA8 pixels

use crate::dataframe::DataFrame;
use crate::error::{DataloaderError, Result};
use crate::types::{DataColumn, DataValue};
use Tensors::{dtype::Element, tensor::Tensor};
use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageTensorLayout {
    ChannelsFirst,
    ChannelsLast,
}

#[derive(Clone, Debug)]
pub struct Image {
    image: DynamicImage,
    source: Option<PathBuf>,
}

impl Image {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let image = ImageReader::open(path).map_err(image_error)?.decode().map_err(image_error)?;
        Ok(Self {image,source: Some(path.to_path_buf())})
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let image = image::load_from_memory(bytes).map_err(image_error)?;
        Ok(Self {image,source: None})
    }

    pub fn from_dynamic_image(image: DynamicImage) -> Self { Self {image,source: None} }

    pub fn from_pixel_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {return Err(DataloaderError::Image("Pixel buffer is missing its dimension header".into()));}
        let width = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let height = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let expected = (width as usize).checked_mul(height as usize).and_then(|count| count.checked_mul(4)).and_then(|count| count.checked_add(8)).ok_or_else(|| DataloaderError::Image("Pixel buffer dimensions overflow".into()))?;
        if width == 0 || height == 0 || bytes.len() != expected { return Err(DataloaderError::Image(format!("Invalid RGBA pixel buffer: expected {expected} bytes for {width}x{height}, received {}",bytes.len()))); }
        let rgba = image::RgbaImage::from_raw(width, height, bytes[8..].to_vec()).ok_or_else(|| {DataloaderError::Image("Could not construct RGBA image from pixels".into())})?;
        Ok(Self {image: DynamicImage::ImageRgba8(rgba),source: None})
    }

    pub fn width(&self) -> u32 { self.image.width() }
    pub fn height(&self) -> u32 { self.image.height() }
    pub fn dimensions(&self) -> (u32, u32) { self.image.dimensions() }
    pub fn channels(&self) -> u8 { self.image.color().channel_count() }
    pub fn rgba_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width() || y >= self.height() { return None; }
        Some(self.image.get_pixel(x, y).0)
    }
    pub fn source(&self) -> Option<&Path> { self.source.as_deref() }
    pub fn as_dynamic_image(&self) -> &DynamicImage { &self.image }

    /// Raw row-major RGBA8 pixels
    pub fn raw_pixels(&self) -> Vec<u8> { self.image.to_rgba8().into_raw() }

    /// Pixels with dims, ready for dataframe
    pub fn pixel_bytes(&self) -> Vec<u8> {
        let (width, height) = self.dimensions();
        let mut bytes = Vec::with_capacity(8 + width as usize * height as usize * 4);
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&self.raw_pixels());
        bytes
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> { self.image.save(path.as_ref()).map_err(image_error) }

    pub fn save_with_format(&self, path: impl AsRef<Path>, format: ImageFormat) -> Result<()> { self.image.save_with_format(path.as_ref(), format).map_err(image_error) }

    /// Encode the image into an in-memory file
    pub fn encode(&self, format: ImageFormat) -> Result<Vec<u8>> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        self.image.write_to(&mut bytes, format).map_err(image_error)?;
        Ok(bytes.into_inner())
    }

    pub fn thumbnail(&self, width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 { return Err(DataloaderError::Image("Thumbnail dimensions must be non-zero".into())); }
        Ok(Self {image: self.image.thumbnail(width, height),source: self.source.clone()})
    }

    pub fn transform<F, E>(&self, transform: F) -> Result<Self> where F: FnOnce(&DynamicImage) -> std::result::Result<DynamicImage, E>, E: std::fmt::Display{
        let image = transform(&self.image).map_err(|error| DataloaderError::Image(error.to_string()))?;
        Ok(Self {image,source: self.source.clone()})
    }

    /// Convert to an RGB tensor with values in [0, 1] when `normalize` is true.
    pub fn to_tensor<T: Element>(&self,layout: ImageTensorLayout,normalize: bool) -> Result<Tensor<T>> {
        let rgb = self.image.to_rgb8();
        let (width, height) = rgb.dimensions();
        let values = rgb.into_raw().into_iter().map(|value| { T::from_f64(if normalize { value as f64 / 255.0 } else { value as f64 })}).collect();
        self.tensor_from_hwc(values, width, height, layout)
    }

    /// Convert to an RGB tensor with channel wise normalization.
    pub fn to_normalized_tensor<T: Element>(&self,layout: ImageTensorLayout,mean: [f32; 3],std: [f32; 3],) -> Result<Tensor<T>> {
        if mean.iter().any(|value| !value.is_finite()) || std.iter().any(|value| !value.is_finite() || *value == 0.0){ return Err(DataloaderError::Image("Normalization means must be finite and standard deviations finite and non-zero".into())); }
        let rgb = self.image.to_rgb8();
        let (width, height) = rgb.dimensions();
        let values = rgb.into_raw().chunks_exact(3).flat_map(|pixel| {(0..3).map(move |channel| {T::from_f64((pixel[channel] as f64 / 255.0 - mean[channel] as f64) / std[channel] as f64)})}).collect();
        self.tensor_from_hwc(values, width, height, layout)
    }

    fn tensor_from_hwc<T: Element>(&self,mut values: Vec<T>,width: u32,height: u32,layout: ImageTensorLayout) -> Result<Tensor<T>> {
        let shape = match layout {
            ImageTensorLayout::ChannelsLast => vec![height as usize, width as usize, 3],
            ImageTensorLayout::ChannelsFirst => {
                let mut chw = vec![T::zero(); values.len()];
                for y in 0..height as usize {
                    for x in 0..width as usize {
                        for channel in 0..3 { chw[channel * width as usize * height as usize + y * width as usize + x] = values[(y * width as usize + x) * 3 + channel]; }
                    }
                }
                values = chw;
                vec![3, height as usize, width as usize]
            }
        };
        Tensor::from_vec(values, &shape).map_err(|error| DataloaderError::Tensor(error.to_string()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct ImageDataset {
    images: Vec<Image>,
}

impl ImageDataset {
    pub fn new(images: Vec<Image>) -> Self { Self { images } }

    pub fn load_dataframe(path: impl AsRef<Path>, recursive: bool) -> Result<DataFrame> {
        Self::from_dir(path, recursive)?.to_dataframe()
    }

    pub fn from_paths<I, P>(paths: I) -> Result<Self> where I: IntoIterator<Item = P>, P: AsRef<Path>{
        paths.into_iter().map(Image::open).collect::<Result<Vec<_>>>().map(Self::new)
    }

    /// Load image files in sorted order from a dir
    pub fn from_dir(path: impl AsRef<Path>, recursive: bool) -> Result<Self> {
        let mut paths = Vec::new();
        collect_image_paths(path.as_ref(), recursive, &mut paths)?;
        paths.sort();
        if paths.is_empty() { return Err(DataloaderError::EmptyData); }
        Self::from_paths(paths)
    }

    pub fn from_dataframe(frame: &DataFrame,path_column: &str,pixels_column: &str) -> Result<Self> {
        let paths = frame.column(path_column)?;
        let pixels = frame.column(pixels_column)?;
        if paths.len() != pixels.len() {return Err(DataloaderError::ShapeMismatch("Image path and pixel columns have different lengths".into()));}
        let images = paths.values.iter().zip(pixels.values.iter()).map(|(path, pixels)| {
                let source = match path {
                    DataValue::String(path) => Some(PathBuf::from(path)),
                    DataValue::Null => None,
                    other => { return Err(DataloaderError::InvalidDataType(path_column.into(),"string".into(),other.type_name().into())); }
                };
                let bytes = match pixels {
                    DataValue::Binary(bytes) => bytes.as_slice(),
                    DataValue::Null => {return Err(DataloaderError::Image("Null pixel row in image DataFrame".into(),));}
                    other => { return Err(DataloaderError::InvalidDataType(pixels_column.into(),"binary".into(),other.type_name().into())); }
                };
                let mut image = Image::from_pixel_bytes(bytes)?;
                image.source = source;
                Ok(image)
            }).collect::<Result<Vec<_>>>()?;
        Ok(Self::new(images))
    }

    pub fn len(&self) -> usize { self.images.len() }
    pub fn is_empty(&self) -> bool { self.images.is_empty() }
    pub fn get(&self, index: usize) -> Option<&Image> { self.images.get(index) }
    pub fn images(&self) -> &[Image] { &self.images }

    pub fn to_dataframe(&self) -> Result<DataFrame> {
        if self.is_empty() { return Err(DataloaderError::EmptyData); }
        let paths = self.images.iter().map(|image| match image.source() {
                Some(path) => DataValue::String(path.to_string_lossy().into_owned()),
                None => DataValue::Null,
            }).collect();
        let pixels = self.images.iter().map(|image| DataValue::Binary(image.pixel_bytes())).collect();
        DataFrame::from_columns(vec![DataColumn::new("path", paths),DataColumn::new("pixels", pixels)])
    }

    pub fn transform<F, E>(&self, transform: F) -> Result<Self> where F: Fn(&DynamicImage) -> std::result::Result<DynamicImage, E>, E: std::fmt::Display,{
        self.images.iter().map(|image| image.transform(|pixels| transform(pixels))).collect::<Result<Vec<_>>>().map(Self::new)
    }

    /// Return dataset slices as batches
    pub fn batches(&self, batch_size: usize, drop_last: bool) -> Result<Vec<&[Image]>> {
        if batch_size == 0 { return Err(DataloaderError::Image("Batch size must be non-zero".into())); }
        let mut batches: Vec<&[Image]> = self.images.chunks(batch_size).collect();
        if drop_last && self.len() % batch_size != 0 { batches.pop(); }
        Ok(batches)
    }

    pub fn save_images(&self, directory: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
        fs::create_dir_all(directory.as_ref())?;
        self.images.iter().enumerate().map(|(index, image)| {
            let extension = image.source().and_then(Path::extension).and_then(|value| value.to_str()).filter(|value| !value.is_empty()).unwrap_or("png");
            let path = directory.as_ref().join(format!("{index:06}.{extension}"));
            image.save(&path)?;
            Ok(path)
        }).collect()
    }
}

fn collect_image_paths(path: &Path, recursive: bool, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if entry.file_type()?.is_dir() && recursive {
            collect_image_paths(&entry_path, true, output)?;
        } else if entry.file_type()?.is_file() && is_supported_image(&entry_path) {
            output.push(entry_path);
        }
    }
    Ok(())
}

pub fn is_supported_image(path: &Path) -> bool { matches!(path.extension().and_then(|value| value.to_str()).map(str::to_ascii_lowercase).as_deref(),Some("png" | "jpg" | "jpeg" | "gif" | "bmp" | "tif" | "tiff" | "webp" | "ico" | "pnm" | "qoi"))}

pub fn image_error(error: impl std::fmt::Display) -> DataloaderError { DataloaderError::Image(error.to_string()) }
