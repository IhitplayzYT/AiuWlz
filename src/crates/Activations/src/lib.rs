
pub mod Activation{
use Tensors::{dtype::{Element, Float}, tensor::Tensor};
    pub trait _Activation: Element + Float {
        fn prelu(self, alpha: Self) -> Self;
        fn rrelu(self, alpha: Self) -> Self;

        fn elu(self, alpha: Self) -> Self;
        fn selu(self, alpha: Self, lambda: Self) -> Self;

        fn mish(self) -> Self;

        fn htanh(self) -> Self;
        fn hsigmoid(self) -> Self;

        fn softplus(self) -> Self;
        fn softsign(self) -> Self;
        fn tanhshrink(self) -> Self;
        fn hardshrink(self, lambda: Self) -> Self;
    }

    pub trait Activation {
        type Elem: Element + Float;

        fn relu(&self) -> Tensor<Self::Elem>;
        fn lrelu(&self, alpha: Self::Elem) -> Tensor<Self::Elem>;
        fn prelu(&self, alpha: Self::Elem) -> Tensor<Self::Elem>;

        fn elu(&self, alpha: Self::Elem) -> Tensor<Self::Elem>;
        fn selu(&self, alpha: Self::Elem, lambda: Self::Elem) -> Tensor<Self::Elem>;

        fn gelu(&self) -> Tensor<Self::Elem>;
        fn silu(&self) -> Tensor<Self::Elem>;
        fn mish(&self) -> Tensor<Self::Elem>;

        fn tanh(&self) -> Tensor<Self::Elem>;
        fn sigmoid(&self) -> Tensor<Self::Elem>;
        fn htanh(&self) -> Tensor<Self::Elem>;
        fn hsigmoid(&self) -> Tensor<Self::Elem>;

        fn softplus(&self) -> Tensor<Self::Elem>;
        fn softsign(&self) -> Tensor<Self::Elem>;
        fn tanhshrink(&self) -> Tensor<Self::Elem>;
        fn hardshrink(&self, lambda: Self::Elem) -> Tensor<Self::Elem>;

        fn softmax(&self, dim: isize) -> Tensor<Self::Elem>;
        fn logsoftmax(&self, dim: isize) -> Tensor<Self::Elem>;
    }

}