## pytorch
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

## Implementation Details
* Regularization: penality to encourage model smoothing
* Learning rate decay

## Intuition
* embedding layer: token embedding lookup equivalent to one-hot encoded input @ embedding matrix (ie 1 trivial linear layer)