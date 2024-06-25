## PyTorch
* `torch.tensor` vs ~~`torch.Tensor`~~: infers dtype vs sets to float
* `torch.Tensor.view` returns a tensor that references same underlying data 
http://blog.ezyang.com/2019/05/pytorch-internals/
* [`torch.Tensor.item`](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html): op that returns the value of a single-element tensor as a standard Python number; not differentiable
* [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
* `torch.nn.functional.cross_entropy`: 
  - efficient impl (fused kernel for ops in forward pass, analytical optimization for backward pass)
    ```python
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdims=True)
    loss = -prob[torch.arange(32), Y].log().mean()
    ```
  - numerical stability: ie when logits contain large +ve -> inf count (trick: `logits = logits - logits.max()` normalizes with no impact on loss)
* `torch.no_grad`: context-manager that disables gradient calculation (for eval/test, inference)
* data loader: `torch.stack`, `torch.cat`

### [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
* function overrides: `forward` (called using __call__ to run registered hooks), `generate` 
* `self.training`: bool var for training vs eval mode
* `self.parameters(recurse=True)`: iterator
* `self.register_buffer(name,tensor,persistent=True)`: add a buffer that can be accessed as attr `self.name`

## Implementation Details
* Regularization Techniques
    * penality to encourage model smoothing
    * dropout: effect of an ensemble of networks
* Learning rate decay: ie step decay
    * update to data ratio visualization is a good way to tune (1e-3 is a good target)
* Parameter Initialization: well-behaved pre-activations (unit std, gradients), logits
    * no longer need to be super precise due to more stable optimization w/ residual connections, normalization layers, better optimizers (RMSProp, Adam)
        * `torch.nn.init.kaiming_normal_`: mode (default fine)
    * scaling before softmax important: diffuse distribution results in better neuron activation (wide range of -ve and +ve values sharpens towards one-hot at the max value -> single neuron contribution)

## Issues
* vanishing gradient (flat sections of activation functions) 
    * dead neurons
* exploding gradient
* degredataion problem (optimization issues for deep layers / large number of parameters)

## Insights
* residual/skip connection - gradient highway: improves gradient flow during backprop for more stable/efficient optimization
    * near initialization, residuals are small so optimizer can focus on smaller set of parameters
* normalization layer - vs ad-hoc hand-tuning of linear layer weight initialization gains to balance activations vs gradient flow
    * layer norm: dim=1 -> no more coupling! 
        * IMPL: no difference between training vs eval modes, buffers no longer needed
    * batch norm: hidden states can just be normalized (dim=0) to the well-behaved unit gaussian at initialization
        * solves: pre-activation states too small -> activation function low activity, too large -> activation saturates and no gradient flow 
        * simple building block: common to append a batch norm layer to layers that contain multiplication (ie every linear/conv layer)
            * vs directly finding scaling factors for weight matrices <- intractable
        * side-effect: adds coupling bewteen examples 
            * -> bugs (-ve): GroupNorm or LayerNorm may be better
            * -> regularization (+ve)
* embedding layer: token embedding lookup equivalent to one-hot encoded input @ embedding matrix (ie 1 trivial linear layer)


# Transformers
* transformer block: communication (self-attention) -> computation (ffn), residual connections
* GPT-3 paper hyperparameters: n_parameters n_layers, d_model=n_embd, n_heads, d_head=head_size, batch_size, learning_rate

## Self-Attention
Andrej's Notes:
- Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
- There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
- Each example across batch dimension is of course processed completely independently and never "talk" to each other
- In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
- "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
- "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much.
