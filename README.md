# ShapeaK ADMM

This repository contains MATLAB and Python/JAX implementations of the Shapeak algorithm for unconstrained binary integer programming experiments.

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

## Benchmark Data

The Python experiments use benchmark instances from the following public datasets:

- Gset Max-Cut instances: [https://web.stanford.edu/~yyye/yyye/Gset/](https://web.stanford.edu/~yyye/yyye/Gset/)
- BiqBin benchmark instances: [http://www.biqbin.eu/Home/BenchmarkInstances](http://www.biqbin.eu/Home/BenchmarkInstances)

## Citation

If you use this code in academic work, please cite:

```bibtex
@article{zhou2025sharp,
  title={Sharp-Peak Functions for Exactly Penalizing Binary Integer Programming},
  author={Zhou, Shenglong and Li, Shuai and Zhang, Hui and Luo, Ziyan},
  journal={arXiv preprint arXiv:2509.00895},
  year={2025}
}
```

## Notes

- Result files are generated under `ShaPeak-Python/shapeakadmm/result/`.
- The MATLAB MIMO demos use functions such as `pskmod`, which may require the MATLAB Communications Toolbox.
- Please add an appropriate license file before distributing or reusing this repository publicly.
