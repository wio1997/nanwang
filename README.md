# nanwang

Ascend compatibility validation artifacts.

## PyTorch Geometric on Ascend 910B3

The [`pyg-ascend-compat`](./pyg-ascend-compat/) directory contains the first-round compatibility screening for:

- Ascend 910B3
- CANN 8.5.1
- PyTorch/torch_npu 2.9.0
- PyTorch Geometric 2.6.1

Start with [`pyg-ascend-compat/final_report.md`](./pyg-ascend-compat/final_report.md). The directory also includes the container recipe, test harnesses, machine-readable results, profiler summaries, and bounded root-cause notes. Raw profiler databases are intentionally excluded because of their size and environment-specific metadata.

