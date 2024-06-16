## pytorch
* `torch.tensor` vs ~~`torch.Tensor`~~: infers dtype vs sets to float
* [`torch.Tensor.item`](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html): op that returns the value of a single-element tensor as a standard Python number; not differentiable
* [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)

## Implementation Details
* Regularization: penality to encourage model smoothing
