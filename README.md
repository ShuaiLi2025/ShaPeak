# ShapeaK ADMM

This repository contains MATLAB and Python/JAX implementations of the ShapeaK ADMM algorithm for binary integer programming experiments.

The code accompanies the sharp-peak penalty approach described in:

> Shenglong Zhou, Shuai Li, Hui Zhang, and Ziyan Luo. "Sharp-peak functions for exactly penalizing binary integer programming." arXiv preprint arXiv:2509.00895, 2025.

## Repository Layout

```text
.
|-- ShaPeak-MATLAB/
|   |-- demonRecovery.m        # Recovery problem demo
|   |-- demonMIMO.m            # Classical MIMO detection demo
|   |-- demon1bMIMO.m          # One-bit MIMO detection demo
|   |-- demonQUBO.m            # Synthetic QUBO demo
|   |-- solver/                # MATLAB ShapeaK ADMM solver
|   |-- SPFs/                  # Sharp-peak functions and proximal operators
|   `-- examples/              # Problem-specific objective/gradient routines
`-- ShaPeak-Python/
    `-- shapeakadmm/
        |-- main.py            # Standalone Python runner
        |-- run_gset.py        # Batch runner for Max-Cut on Gset
        |-- run_biqbin.py      # Batch runner for UBQP / BiqBin instances
        |-- shapeak_jax.py     # JAX implementation of ShapeaK ADMM
        |-- shapeak_spf.py     # Sharp-peak penalty functions in JAX
        |-- data.py            # Dataset parsers and preprocessing utilities
        `-- instance/          # Gset and Instances_uBQP benchmark data
```

## MATLAB Demos

The MATLAB implementation includes demonstrations for recovery, classical MIMO, one-bit MIMO, and QUBO problems.

From MATLAB, enter the MATLAB folder and run one of the demo scripts:

```matlab
cd ShaPeak-MATLAB
demonRecovery
demonMIMO
demon1bMIMO
demonQUBO
```

Each demo adds the current folder and its subfolders to the MATLAB path using `addpath(genpath(pwd))`.

## Python/JAX Experiments

The Python version is organized as a package named `shapeakadmm`. It currently includes Max-Cut experiments on Gset instances and UBQP experiments on `Instances_uBQP` data.

### Requirements

The Python implementation requires a JAX environment. Install the dependencies appropriate for your platform, for example:

```bash
pip install numpy scipy networkx jax optax
```

For GPU/TPU runs, install the JAX build that matches your accelerator and driver setup.

### Run a Single Gset Max-Cut Instance

```bash
cd ShaPeak-Python
python shapeakadmm/main.py --dataset Gset --Gset_id 22 --overwrite
```

The output is written by default to:

```text
ShaPeak-Python/shapeakadmm/result/mc/Gset/
```

### Batch Run Gset Instances

The batch runner is configured for Gset instances `G22` through `G81` by default:

```bash
cd ShaPeak-Python
python shapeakadmm/run_gset.py --start_id 22 --end_id 81 --overwrite
```

The script writes per-instance result files and summary files:

```text
shapeakadmm/result/mc/Gset/run_gset_summary.txt
shapeakadmm/result/mc/Gset/run_gset_summary.csv
```

### Run a Single UBQP Instance

```bash
cd ShaPeak-Python
python shapeakadmm/main.py \
  --dataset UBQP \
  --instance_file instance/Instances_uBQP/be100.1.sparse.mc \
  --overwrite
```

The output is written by default to:

```text
ShaPeak-Python/shapeakadmm/result/ubqp/Instances_uBQP/
```

### Batch Run UBQP / BiqBin Instances

```bash
cd ShaPeak-Python
python shapeakadmm/run_biqbin.py --overwrite
```

To run only a small subset:

```bash
python shapeakadmm/run_biqbin.py --limit 5 --overwrite
```

## Supported Sharp-Peak Penalties

The Python/JAX implementation currently supports the following penalty families:

```text
haa2205, gaa2205, 111105, 111100, 111101,
gaa2200, haa2200, gaa2201, haa2201
```

They can be selected with the `--shapeak_penf` argument, for example:

```bash
python shapeakadmm/main.py --dataset Gset --Gset_id 22 --shapeak_penf gaa2205 --overwrite
```

## Citation

If you use this code in academic work, please cite:

```bibtex
@article{zhou2025sharppeak,
  title={Sharp-peak functions for exactly penalizing binary integer programming},
  author={Zhou, Shenglong and Li, Shuai and Zhang, Hui and Luo, Ziyan},
  journal={arXiv preprint arXiv:2509.00895},
  year={2025}
}
```

## Notes

- Result and cache files are generated under `ShaPeak-Python/shapeakadmm/result/` and `ShaPeak-Python/shapeakadmm/cache/`.
- The MATLAB MIMO demos use functions such as `pskmod`, which may require the MATLAB Communications Toolbox.
- Please add an appropriate license file before distributing or reusing this repository publicly.
