pub mod Tensor{
    use std::error::Error;


    pub struct Tensor<T>{
        buff: T
    }

    impl<T> Tensor<T>{
        
        pub fn to_vec(&self) -> Result<&[isize],Box<dyn Error>>{
            Ok(&[1,2])
        }

        pub fn shape(&self) -> &[usize]{
            &[1,2]
        }

    }


}