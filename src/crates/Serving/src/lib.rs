pub mod Serving{

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



    pub struct Server<M,P,T,R>{
        model: dyn M,
        policy: RequestPolicy<T,R>,
        logger: Logger

    }










}