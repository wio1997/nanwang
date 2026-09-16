# PyG 2.8.0.post1 validation environment

- Hardware: 8 × Ascend 910B3; validation device `npu:1`
- Host: aarch64, Ubuntu 22.04 LTS, kernel `5.15.0-25-generic`
- Host Ascend driver: `26.0.rc1` (`ascend910b-driver` package and `version.info`)
- Container: `wio-pyg-cann851-pyg280`
- Image: `local/wio-pyg-cann851:torch2.9-pyg2.8.0.post1`
- CANN: 8.5.1, active path `/usr/local/Ascend/cann-8.5.1`
- Python: 3.11.14
- PyTorch: `2.9.0+cpu`
- torch_npu: 2.9.0
- PyG: 2.8.0.post1
- Eight NPUs are visible. A synchronized fp32 matmul on `npu:1` passed.
- All seven requested API symbols import, including the deprecated `global_sort_pool` wrapper.

The software stack inside the process matches the requested Python/CANN/PyTorch/torch_npu/PyG versions. The host driver does **not** match the requested 25.2.0 row; it remains 26.0.rc1 and was not changed because driver replacement is a host-wide, reboot-sensitive operation outside this isolated container validation.
