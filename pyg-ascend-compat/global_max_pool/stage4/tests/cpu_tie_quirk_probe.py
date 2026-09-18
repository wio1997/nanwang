#!/usr/bin/env python3
"""Stage 4: pin down the exact PyTorch reduction-backward tie rule behind the CPU oracle.

The oracle showed that a group whose maximum equals the *zero initial value* distributes the
gradient differently (e.g. `[[0.0],[-0.0]]` -> 1/3 each, not 1/2).  PyG calls
`out.new_zeros(size).scatter_reduce_(dim, index, src, reduce="amax", include_self=False)`, so the
question is whether the extra denominator term comes from the *self slot* (independently of its
value) or from the numeric value 0.

Decisive experiments: keep the source data identical and only change the initial value of the
destination buffer (and include_self).
"""

from __future__ import annotations

import torch
from torch_geometric.nn import global_max_pool


def show(tag, x_vals, idx, size, init, include_self):
    x = torch.tensor(x_vals, dtype=torch.float32, requires_grad=True)
    idx_t = torch.tensor(idx, dtype=torch.int64)
    f = x.shape[1]
    dest = torch.full((size, f), init, dtype=torch.float32)
    exp_idx = idx_t.unsqueeze(-1).expand(-1, f)
    out = dest.scatter_reduce_(-2, exp_idx, x, reduce="amax", include_self=include_self)
    out.sum().backward()
    print(f"{tag:44s} init={init:<8g} include_self={str(include_self):5s} "
          f"out={out.detach().tolist()} grad={x.grad.tolist()}")


print("=" * 100)
print("A) source max = 0, varying the destination initial value (include_self=False)")
print("=" * 100)
for init in (0.0, -1000.0, 1000.0, 2.0):
    try:
        show(f"src=[[0],[0]] size=1", [[0.0], [0.0]], [0, 0], 1, init, False)
    except Exception as exc:  # noqa: BLE001
        print(f"init={init} -> {type(exc).__name__}: {exc}")
print()
show("src=[[0]] size=1", [[0.0]], [0], 1, 0.0, False)
show("src=[[0]] size=1", [[0.0]], [0], 1, -1000.0, False)
show("src=[[0],[-0]] size=1", [[0.0], [-0.0]], [0, 0], 1, 0.0, False)
show("src=[[-0]] size=1", [[-0.0]], [0], 1, 0.0, False)
show("src=[[5],[5]] size=1", [[5.0], [5.0]], [0, 0], 1, 0.0, False)
show("src=[[-5],[-5]] size=1", [[-5.0], [-5.0]], [0, 0], 1, 0.0, False)

print()
print("=" * 100)
print("B) include_self=True with a distinct self value (does self always join the count?)")
print("=" * 100)
show("src=[[0],[0]] size=1 include_self", [[0.0], [0.0]], [0, 0], 1, 0.0, True)
show("src=[[5],[5]] size=1 include_self", [[5.0], [5.0]], [0, 0], 1, 5.0, True)
show("src=[[5],[5]] size=1 include_self(self=1)", [[5.0], [5.0]], [0, 0], 1, 1.0, True)

print()
print("=" * 100)
print("C) same data through the real PyG API (PyG always uses new_zeros + include_self=False)")
print("=" * 100)
for tag, xv, idx in (("x=[[0],[0]]", [[0.0], [0.0]], [0, 0]),
                     ("x=[[0],[-0]]", [[0.0], [-0.0]], [0, 0]),
                     ("x=[[0],[-0],[-1]]", [[0.0], [-0.0], [-1.0]], [0, 0, 0]),
                     ("x=[[0]]", [[0.0]], [0]),
                     ("x=[[2],[0]]", [[2.0], [0.0]], [0, 0])):
    x = torch.tensor(xv, dtype=torch.float32, requires_grad=True)
    b = torch.tensor(idx, dtype=torch.int64)
    out = global_max_pool(x, b, 1)
    out.sum().backward()
    print(f"{tag:28s} out={out.detach().tolist()} grad={x.grad.tolist()}")

print()
print("=" * 100)
print("D) group with zero max in a multi-group, explicit-size setting (PyG API)")
print("=" * 100)
x = torch.tensor([[4.0], [4.0], [0.0]], requires_grad=True)
b = torch.tensor([0, 0, 3], dtype=torch.int64)
out = global_max_pool(x, b, 5)
up = torch.tensor([[1.0], [5.0], [7.0], [11.0], [13.0]])
(out * up).sum().backward()
print(f"x=[[4],[4],[0]] batch=[0,0,3] size=5 out={out.detach().tolist()} grad={x.grad.tolist()}")

x = torch.tensor([[4.0], [4.0], [0.0]], requires_grad=True)
out = global_max_pool(x, b, 5)
out.sum().backward()
print(f"same, upstream=1                     out={out.detach().tolist()} grad={x.grad.tolist()}")
