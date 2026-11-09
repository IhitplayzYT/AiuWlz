use image::imageops::GaussianBlurParameters;
use ::image::{DynamicImage, GenericImageView, imageops::FilterType};
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImageTransform {
    Resize {width: u32,height: u32,filter: ResizeFilter},
    CenterCrop {width: u32,height: u32},
    Crop {x: u32,y: u32,width: u32,height: u32},
    Rotate90,
    Rotate180,
    Rotate270,
    FlipHorizontal,
    FlipVertical,
    Grayscale,
    HueRotate{degree: i32},
    Invert,
    Unsharpen{sigma:f32,threshold:i32},
    Blur {sigma: f32},
    Brightness {amount: i32},
    Contrast {amount: f32},
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeFilter {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

impl From<ResizeFilter> for FilterType {
    fn from(filter: ResizeFilter) -> Self {
        match filter {
            ResizeFilter::Nearest => FilterType::Nearest,
            ResizeFilter::Triangle => FilterType::Triangle,
            ResizeFilter::CatmullRom => FilterType::CatmullRom,
            ResizeFilter::Gaussian => FilterType::Gaussian,
            ResizeFilter::Lanczos3 => FilterType::Lanczos3,
        }
    }
}

pub fn apply(image: &DynamicImage, transform: &ImageTransform) -> Result<DynamicImage, String> {
    let (width, height) = image.dimensions();
    match *transform {
        ImageTransform::Resize {width,height,filter} => {
            if width == 0 || height == 0 { return Err("Resize dims must be non zero".into()); }
            Ok(image.resize_exact(width, height, filter.into()))
        },
        ImageTransform::Unsharpen { sigma, threshold } => Ok(image.unsharpen(sigma, threshold)),
        ImageTransform::Crop {x,y,width: crop_width,height: crop_height} => {
            validate_crop(x, y, crop_width, crop_height, width, height)?;
            Ok(image.crop_imm(x, y, crop_width, crop_height))
        }
        ImageTransform::CenterCrop {width: crop_width,height: crop_height} => {
            validate_crop(0, 0, crop_width, crop_height, width, height)?;
            let x = (width - crop_width) / 2;
            let y = (height - crop_height) / 2;
            Ok(image.crop_imm(x, y, crop_width, crop_height))
        },
        ImageTransform::HueRotate { degree } => Ok(image.huerotate(degree)),

        ImageTransform::Rotate90 => Ok(image.rotate90()),
        ImageTransform::Rotate180 => Ok(image.rotate180()),
        ImageTransform::Rotate270 => Ok(image.rotate270()),
        ImageTransform::FlipHorizontal => Ok(image.fliph()),
        ImageTransform::FlipVertical => Ok(image.flipv()),
        ImageTransform::Grayscale => Ok(image.grayscale()),
        ImageTransform::Invert => {
            let mut result = image.clone();
            result.invert();
            Ok(result)
        }
        ImageTransform::Blur { sigma } => {
            if !sigma.is_finite() || sigma < 0.0 { return Err("Blur sigma must be finite and non-negative".into()); }
            Ok(image.blur(sigma))
        }
        ImageTransform::Brightness { amount } => Ok(image.brighten(amount)),
        ImageTransform::Contrast { amount } => { 
            if !amount.is_finite() { return Err("Contrast amount must be finite".into()); }
            Ok(image.adjust_contrast(amount))
        }
    }
}

fn validate_crop(x: u32,y: u32,crop_width: u32,crop_height: u32,width: u32,height: u32) -> Result<(), String> {
    if crop_width == 0 || crop_height == 0 { return Err("Crop dims must be non zero".into());}
    if x > width || y > height || crop_width > width.saturating_sub(x) || crop_height > height.saturating_sub(y){ return Err(format!("Crop ({x}, {y}, {crop_width}, {crop_height}) exceeds image bounds ({width}, {height})")); }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct TransformPipeline {
    transforms: Vec<ImageTransform>,
}

impl TransformPipeline {
    pub fn new(transforms: Vec<ImageTransform>) -> Self { Self { transforms } }
    pub fn push(&mut self, transform: ImageTransform) { self.transforms.push(transform); }
    pub fn transforms(&self) -> &[ImageTransform] { &self.transforms }
    pub fn apply(&self, image: &DynamicImage) -> Result<DynamicImage, String> { self.transforms.iter().try_fold(image.clone(), |current, transform| {apply(&current, transform)}) }
    pub fn finalize(&self, image: &DynamicImage) -> Result<DynamicImage, String> { self.transforms.iter().try_fold(image.clone(), |current, transform| {apply(&current, transform)}) }
}
