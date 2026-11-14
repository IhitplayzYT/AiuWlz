pub mod Device{
    #[allow(non_snake_case,non_camel_case_types,dead_code)]
    use std::fmt;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Device {
        Cpu,
        Cuda(usize),
    }

    impl Device {
        pub fn is_cpu(&self) -> bool {
            matches!(self, Device::Cpu)
        }
        pub fn is_cuda(&self) -> bool {
            matches!(self, Device::Cuda(_))
        }

    }

    impl fmt::Display for Device {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Device::Cpu => write!(f, "cpu"),
                Device::Cuda(i) => write!(f, "cuda:{i}"),
            }
        }
    }

    impl From<&str> for Device{
        fn from(value: &str) -> Self {
            match value{
                "cpu" => return Device::Cpu,
                _ => {}
            }
            let k = value.strip_prefix("cuda:").and_then(|x| x.strip_prefix("dev:")).and_then(|x| x.strip_prefix("gpu:")).and_then(|x| x.strip_prefix(":")).unwrap();
            Self::Cuda(k.parse().expect("The device str has to be of the fmt cpu,cuda:x,dev:x,gpu:x or :x"))
        }
    }

}
