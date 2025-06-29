# Advanced PyTorch Techniques: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

1. [Introduction](#introduction)
2. [Mixed Precision Training](#mixed-precision-training)
3. [Distributed Training](#distributed-training)
4. [Custom Modules and Layers](#custom-modules-and-layers)
5. [Hooks and Model Introspection](#hooks-and-model-introspection)
6. [Model Export and Deployment](#model-export-and-deployment)
7. [Quantization and Pruning](#quantization-and-pruning)
8. [Performance Optimization](#performance-optimization)
9. [Debugging and Profiling](#debugging-and-profiling)
10. [Best Practices](#best-practices)

---

## Introduction

This guide covers advanced PyTorch techniques for building, training, and deploying state-of-the-art deep learning models efficiently and at scale. These techniques are essential for research, production, and maximizing hardware utilization.

---

## Mixed Precision Training

Mixed precision uses both 16-bit and 32-bit floating point types to speed up training and reduce memory usage, especially on modern GPUs (NVIDIA Volta+).

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = ...  # your model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    with autocast():
        output = model(data)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

- Use `autocast()` for forward and loss computation.
- Use `GradScaler` to prevent underflow in gradients.
- Enable with `torch.backends.cudnn.benchmark = True` for best performance.

---

## Distributed Training

PyTorch supports distributed training across multiple GPUs and nodes using `torch.distributed` and `torch.nn.parallel`.

### Data Parallelism (Single Node)

```python
import torch.nn as nn
model = nn.DataParallel(model)
```

### Distributed Data Parallel (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
# Run with: python -m torch.distributed.launch --nproc_per_node=NUM_GPUS script.py

dist.init_process_group("nccl")
model = DDP(model.cuda())
```

- Use `DistributedSampler` for your DataLoader.
- DDP is faster and more scalable than DataParallel.

---

## Custom Modules and Layers

You can create custom layers by subclassing `nn.Module`.

```python
import torch.nn as nn
import torch

class CustomSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.swish = CustomSwish()
        self.fc2 = nn.Linear(20, 1)
    def forward(self, x):
        x = self.swish(self.fc1(x))
        return self.fc2(x)
```

---

## Hooks and Model Introspection

Hooks allow you to inspect or modify activations and gradients during forward/backward passes.

### Forward Hook Example

```python
def print_activation(module, input, output):
    print(f"Activation from {module}: {output.shape}")

hook = model.layer.register_forward_hook(print_activation)
output = model(input)
hook.remove()
```

### Backward Hook Example

```python
def print_grad(module, grad_input, grad_output):
    print(f"Gradient from {module}: {grad_output[0].shape}")

hook = model.layer.register_backward_hook(print_grad)
output = model(input)
loss = loss_fn(output, target)
loss.backward()
hook.remove()
```

---

## Model Export and Deployment

Export models for deployment using TorchScript or ONNX.

### TorchScript

```python
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

### ONNX Export

```python
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx", 
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
```

- Use ONNX for interoperability with other frameworks and deployment tools.

---

## Quantization and Pruning

Reduce model size and inference latency for deployment on edge devices.

### Quantization

```python
import torch.quantization

model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)
# Calibrate with data
for data, _ in dataloader:
    model_prepared(data)
model_quantized = torch.quantization.convert(model_prepared)
```

### Pruning

```python
import torch.nn.utils.prune as prune

prune.l1_unstructured(model.fc, name='weight', amount=0.5)
prune.remove(model.fc, 'weight')
```

---

## Performance Optimization

- Use `torch.backends.cudnn.benchmark = True` for variable input sizes.
- Use `pin_memory=True` and `num_workers > 0` in DataLoader for faster data transfer.
- Profile with `torch.profiler` or `torch.utils.bottleneck`.
- Use `torch.compile()` (PyTorch 2.0+) for graph-level optimizations:

```python
model = torch.compile(model)
```

---

## Debugging and Profiling

### Debugging
- Use `torch.autograd.set_detect_anomaly(True)` to find NaNs/infs in backward pass.
- Use `pdb` or `ipdb` for interactive debugging.

### Profiling

```python
import torch.profiler

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, (data, target) in enumerate(dataloader):
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        prof.step()
```

- Visualize with TensorBoard: `tensorboard --logdir=./log`

---

## Best Practices

- Use mixed precision and DDP for large models and datasets.
- Profile and optimize data pipelines.
- Use hooks for debugging and visualization.
- Quantize/prune for deployment on edge devices.
- Export with TorchScript/ONNX for production.

---

## References
- [PyTorch Advanced Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Distributed Overview](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [PyTorch Model Optimization](https://pytorch.org/tutorials/recipes/recipes/quantization.html) 