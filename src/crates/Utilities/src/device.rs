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

}
