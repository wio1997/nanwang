# Stage 1B — ScatterMaxV1 NPU runtime bring-up

Goal: run the custom op built in Stage 1A (`ScatterMaxV1`, FP32 data + INT32 index) on a real
Ascend 910B3 through the generated ACLNN API, without installing `mx_driving` / PyG patches.

## Layout

```
stage1b/
├── runner/
│   ├── scattermaxv1_runner.cpp   # minimal ACLNN runner + CPU golden + compare
│   └── build_runner.sh           # builds against the isolated custom OPP
├── tests/
│   └── run_stage1b_tests.sh      # T1..T5 driver
└── results/                      # dumped src/index/expected/actual per case
```

## Isolated deployment (no system CANN touched)

* custom OPP installed to `/root/zyg/build/scattermax_runtime_opp` via
  `custom_opp_ubuntu_aarch64.run --quiet --install-path=...`
  (`/usr/local/Ascend/cann-8.5.1/opp` is left untouched, its `vendors/` stays empty)
* runtime env, set only inside the test shell:
  * `ASCEND_CUSTOM_OPP_PATH=<install>/vendors/customize`
  * `LD_LIBRARY_PATH=<install>/vendors/customize/op_api/lib:$LD_LIBRARY_PATH`
  (exactly what the vendor `bin/set_env.bash` sets)

## Usage

```bash
bash runner/build_runner.sh          # -> runner/scattermaxv1_runner
bash tests/run_stage1b_tests.sh      # -> /root/zyg/logs/stage1b_*.log
```

Single case:

```bash
source /root/zyg/build/scattermax_runtime_opp/vendors/customize/bin/set_env.bash
runner/scattermaxv1_runner --case t4 --n 4 --f 8 --size 5 --index 0,2,0,1 \
    --tol 0 --dump /tmp/t4.txt
```

The runner always uses device 0 only (`aclrtSetDevice(0)`), no distributed setup.
The output tensor is caller-provided and pre-filled with `-inf`, so the raw op result is what
the CPU golden (also `-inf` initialised) compares against.
