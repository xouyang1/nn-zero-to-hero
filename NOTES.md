## PyTorch
### Basics
* `torch.tensor` vs ~~`torch.Tensor`~~: infers dtype vs sets to float
* `torch.Tensor.view` returns a tensor that references same underlying data 
http://blog.ezyang.com/2019/05/pytorch-internals/
* [`torch.Tensor.item`](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html): op that returns the value of a single-element tensor as a standard Python number; not differentiable
* `torch.Tensor.detach`: returns a new Tensor, detached from the current graph, will never require gradient
* [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
* data loader: `torch.stack`, `torch.cat`

### Model: [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
* function overrides: `forward` (called using __call__ to run registered hooks), `generate` 
* `self.training`: bool var for training vs eval mode
* `self.parameters(recurse=True)`: iterator
* `self.register_buffer(name,tensor,persistent=True)`: add a buffer that can be accessed as attr `self.name`

### Forward Pass
* `torch.nn.functional.scaled_dot_product_attention`: fused kernel 
  ```python
  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
  att = F.softmax(att, dim=-1)
  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
  ```
* `torch.nn.functional.cross_entropy`: 
  - efficient impl (fused kernel for ops in forward pass, analytical optimization for backward pass)
    ```python
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdims=True)
    loss = -prob[torch.arange(32), Y].log().mean()
    ```
  - numerical stability: ie when logits contain large +ve -> inf count (trick: `logits = logits - logits.max()` normalizes with no impact on loss)
* `torch.no_grad`: context-manager that disables gradient calculation (for eval/test, inference)


### Training
* `torch.optim.AdamW`: "bug fix from Adam"
    - `fused=True`: fused kernel for parameter updates
* `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`: hacky solution to prevent exploding gradients; applied after backward pass, before optimizer updates parameters
* `torch.optim.lr_scheduler`: many implementations of schedulers


### CUDA
* `<tensor> = <tensor>.to(device)`, `model.to(device)`
* `torch.cuda.synchronize()`: ie when taking timing statistics
* `model = torch.compile(model)`: JIT compilation to optimize into faster and more efficient kernels, model graph optimization

*Mixed Precision Training*
* `torch.autocast`: context-manager for forward pass only! (backward pass and optimizer should be outside)
    - https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16: ie activations
    - https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float32: ie norms, softmax
* `torch.set_float32_matmul_precision("high")` for TF32

*Multi-GPU Distributed Communication*
* [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html): `python -m torch.distributed.run`
    - `RANK` (worker rank in a worker group) and `WORLD_SIZE` (total num of workers in a worker group) env vars assigned automatically
* [`torch.distributed.all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce): reduces the tensor data across all machines -> bitwise identical in all processes
    - ie `dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)`
* [`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html): container provides [gradient synchronization](https://huggingface.co/docs/accelerate/en/concept_guides/gradient_synchronization) at module level across model replicas
    - user is responsible for defining input chunking/sharding
    - `self.module` returns the raw model
* `torch.distributed.destroy_process_group()`: cleanup

## Implementation Details
* Regularization Techniques
    * penality to encourage model smoothing
    * dropout: effect of an ensemble of networks
    * weight decay in matmuls and embeddings
* Learning rate
    - LR decay/scheduler
        ```python
        for param_group in optimizer.param_groups:
            param_group['lr'] = <new lr>
        ```
        - step decay
        - cosine decay with linear warmup
    - update to data ratio visualization is a good way to tune (1e-3 is a good target)
* Parameter Initialization: well-behaved pre-activations (unit std, gradients), logits
    * no longer need to be super precise due to more stable optimization w/ residual connections, normalization layers, better optimizers (RMSProp, Adam)
        * `torch.nn.init.kaiming_normal_`: mode (default fine)
    * scaling before softmax important: diffuse distribution results in better neuron activation (wide range of -ve and +ve values sharpens towards one-hot at the max value -> single neuron contribution)
* Remember to `optimizer.zero_grad()` before `loss.backward()`!
* Optimize GPU utilization
    * reduced precision
    * JIT compilation with torch compile
    * Flash Attention: fused kernel for the full attention calc that adds FLOPS to reduce HBM reads/writes -> faster
        - online softmax trick: incrementally evaluate with intermediate values
    * scan constants/scaling factor choices for "ugly"/suspicious numbers to change to "nice" numbers (powers of 2)


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
* transformer block
    * communication (self-attention) -> computation (ffn)
    * reduce (self-attention) -> map (ffn)
    * residual connections on residual pathway/stream

## Self-Attention
Andrej's Notes:
- Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
- There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
- Each example across batch dimension is of course processed completely independently and never "talk" to each other
- In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
- "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
- "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much.

## Hyperparameters
GPT-3 paper hyperparameters: n_parameters n_layers, d_model=n_embd, n_heads, d_head=head_size, batch_size (BxT), learning_rate
- bigger model uses: bigger batch size, smaller LR
- IMPL_ **gradient accumulation** over serial microsteps (each microstep fits into GPU memory)

## Datasets 
* DataLoader IMPL_: break up documents in a more random order to avoid spurious correlation across documents

### Training mixtures
CommonCrawl: large but very very noisy
- HF [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) 2024-04 (15T tokens) - detailed methodology
    - data range: 2013 summer - 2024 winter
    - each filtering technique was evaluated by evaluating models trained on the filtered dataset (ablation models) 
        - training 1.8B parameter model on 28B tokens (~5h on 64 H100) vs 350B tokens (~2.5d)
    - finding: CommonCrawl team filtering out adult content from 2022-2023 harmed quality
- Cerebras RedPajama -> [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) 2023-06-09 (627B tokens)

### Eval 
- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness): "a unifying framework that allows any causal language model to be tested on the same exact inputs and codebase."

#### Benchmarks 
- HellaSwag (2019-05): no longer difficult - [96% accuracy]
(https://paperswithcode.com/sota/sentence-completion-on-hellaswag) but provides early signal even on small models
    - methodology: adversarial filtering
    - smooth
    - IMPL_: for small models where LM not skilled in multiple choice, multiple choice questions can be tested by batching the options (B = num choices, T = longest completion, pad all other options) with shared context tokens and identifying the best mean loss option within the batch as the answer
