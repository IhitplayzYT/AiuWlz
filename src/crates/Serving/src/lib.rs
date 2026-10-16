pub mod Serving{
    use Utilities::logger::Logger;


    pub struct RequestPolicy<T,R>{
        f: Box<dyn FnOnce(T) -> R>
    }

    impl <T,R> RequestPolicy<T,R>{
        pub fn new(f: Box<dyn FnOnce(T) -> R>) -> Self{
            Self { f:f }
        }

        pub fn apply_policy(self,data:T) -> R{
            (self.f)(data)
        }
    }


    pub trait Servable<X> {
        fn train(&mut self,input:X);
        fn predict(&self,input:X) -> X;
    }


//    pub struct Server<X,M:Servable<X>,T,R>{
//        model: M,
//        policy: RequestPolicy<T,R>,
//        logger: Logger
//    }










}