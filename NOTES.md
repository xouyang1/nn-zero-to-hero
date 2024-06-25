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

## Implementation Details
* Regularization: penality to encourage model smoothing
* Learning rate decay: ie step decay
    * update to data ratio visualization is a good way to tune (1e-3 is a good target)
* Parameter Initialization: well-behaved pre-activations (unit std, gradients), logits
    * no longer need to be super precise due to more stable optimization w/ residual connections, normalization layers, better optimizers (RMSProp, Adam)
        * `torch.nn.init.kaiming_normal_`: mode (default fine)

## Issues
* vanishing gradient (flat sections of activation functions)
* dead neurons

## Insights
* normalization layer - impact: reliable training of dnn's (before needed to play with linear layer weight initialization gain to balance activations vs gradient flow) 
    * batch norm: hidden states can just be normalized to the well-behaved unit gaussian at initialization
        * solves: pre-activation states too small -> activation function low activity, too large -> activation saturates and no gradient flow 
        * simple building block: common to append a batch norm layer to layers that contain multiplication (ie every linear/conv layer)
            * vs directly finding scaling factors for weight matrices <- intractable
        * side-effect: adds coupling bewteen examples 
            * -> bugs (-ve): GroupNorm or LayerNorm may be better
            * -> regularization (+ve)
* embedding layer: token embedding lookup equivalent to one-hot encoded input @ embedding matrix (ie 1 trivial linear layer)