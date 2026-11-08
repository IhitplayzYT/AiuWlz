# NeuralNets

`NeuralNets` provides a tensor-native sequential model API for dense and
feed-forward architectures. Parameter gradients are computed through
`Tensors::autograd::Comp`; minibatch indexing, activations, loss reductions,
normalization, and parameter updates use `Tensor` operations. Evaluation
metrics reuse the workspace `Metrics` crate.

## Example

```rust
use NeuralNets::{Activation, FitOptions, Initializer, Layer, Loss, LrScheduler, Optimizer, Sequential};
use Tensors::tensor::Tensor;
use Utilities::random::Random::Rng;

let mut rng = Rng::seed(7);
let mut model = Sequential::<f32>::new();
model.add_dense(32, 64, Activation::Gelu, Initializer::KaimingNormal, &mut rng)?;
model.add_layer_norm(64, 1e-5)?;
model.add_dropout(0.1)?;
model.add_dense(64, 10, Activation::Linear, Initializer::Xavier, &mut rng)?;

let input = Tensor::zeros(&[128, 32]);
let labels = Tensor::zeros(&[128]); // replace with class IDs in [0, 10)
let mut optimizer = Optimizer::adamw(1e-3, 1e-4)?.with_grad_clip_norm(1.0)?;
let options = FitOptions { epochs: 20, batch_size: 32, shuffle: true, seed: 123 };
let mut schedule = LrScheduler::cosine_annealing(1e-3_f32, 20, 1e-5)?;
let history = model.fit_class_indices_with_scheduler(
    &input,
    &labels,
    Loss::CategoricalCrossEntropyWithLogits,
    &mut optimizer,
    &options,
    &mut schedule,
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Included building blocks

- Layers: dense/linear, Conv1d/Conv2d, 1D/2D max/average and global average/max pooling, dropout, layer
  normalization, batch normalization, PReLU, flatten, activations, and
  identity residual blocks.
- Activations: ReLU, leaky ReLU, sigmoid, tanh, GELU, ELU, SELU, SiLU,
  softplus, Mish, softsign, hard sigmoid, hard tanh, and softmax.
- Initializers: zeros, ones, constants, bounded uniform, He/Kaiming uniform
  and normal, Xavier uniform and normal, and LeCun normal.
- Optimizers: SGD, momentum, Nesterov, RMSProp, Adagrad, Adadelta, Adam, and
  AdamW, with optional global gradient norm clipping.
- Learning-rate schedules: step decay, exponential decay, and cosine annealing;
  `fit_with_scheduler` and class-index variants apply them once per epoch.
- Losses: MSE, MAE, BCE, BCE with logits, categorical cross entropy (probability
  and logits forms), Huber, Smooth L1, KL divergence, hinge, and Poisson NLL.

`Conv1d` accepts `[batch, channels, length]`; `Conv2d` accepts NCHW inputs and
OIHW filters, with pair-valued stride and padding. Conv1d maps to the same
parallel tensor convolution primitive by adding a singleton spatial axis.
The 1D pooling layers use the corresponding 2D tensor pooling operations with
a singleton height axis, and preserve `[batch, channels, length]` layout.
Convolution operations and backward passes use parallel tensor kernels on CPU;
CUDA currently uses the tensor library's host fallback. `Flatten` converts
`[batch, ...]` to `[batch, features]`. `BatchNorm1d` expects
`[batch, features]`; `LayerNorm` works over the final feature axis. `Dropout` is
active during `fit` and disabled for `forward`/`predict`. Categorical loss on
probabilities expects a final softmax layer; the logits form expects raw final
scores.

For token models, add `TokenEmbedding` first with `model.add_embedding(...)`,
then build layers over the embedding dimension. `forward_tokens` and
`fit_tokens` accept integer IDs shaped `[batch, sequence]`; training targets
should preserve those leading dimensions, such as `[batch, sequence, classes]`
for token classification. Embedding lookup uses tensor indexing and its
backward pass accumulates gradients for repeated token IDs.

An image classifier can use `Conv2d -> activation -> global pooling -> Flatten
-> Dense`. The convolution primitive currently supports stride and padding;
dilation and grouped convolution are not implemented yet.
