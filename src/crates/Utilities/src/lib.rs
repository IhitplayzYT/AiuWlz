pub mod Utilities{

    // Macros:
    //   - Reshape (Using of tuple syntax)
    //   


    // Some trait definations for the numpy based impl of scalars,vectors and matrices

    pub trait Collection<T,M: Collection>{
        fn unpack(self) -> Iterable<T>;
        fn dims(&self) -> usize;
        fn shape(&self) -> Vec<usize>;
    }


    pub trait MinMathsContract<M: Collection + Add + Sub + Mul + Div,T: MinMathsContract>{
        fn min(&self) -> usize;
        fn max(&self) -> usize;
        fn count(&self) -> usize;
        fn mag(&self) -> usize;

        fn Percentile(&self,ptile: f64) -> M;

        fn mean(&self) -> M;
        fn median(&self) -> M;
        fn mode(&self) -> M;

        fn SD(&self) -> f64;
        fn Var(&self) -> f64;

        fn Cov(&self,other: &T) -> f64;
        fn Pearson_coeff(&self,other: &impl T) -> f64;
    }








}
