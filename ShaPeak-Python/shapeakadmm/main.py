import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


if __package__ in {None, ""}:
    PACKAGE_ROOT = Path(__file__).resolve().parent
    REPO_ROOT = PACKAGE_ROOT.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from shapeakadmm.data import (  # noqa: E402
        INSTANCE_ROOT,
        build_preconditioner,
        fix_seed,
        generate_Max_cut,
        generate_uBQP_sparse_mc,
        instance_tag,
        parse_gset,
        resolve_in_package,
    )
else:
    from .data import (
        INSTANCE_ROOT,
        build_preconditioner,
        fix_seed,
        generate_Max_cut,
        generate_uBQP_sparse_mc,
        instance_tag,
        parse_gset,
        resolve_in_package,
    )


PACKAGE_ROOT = Path(__file__).resolve().parent


def _str_to_bool(value: str | int | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_y_init(value: str) -> Optional[float]:
    token = str(value).strip().lower()
    if token == "grad":
        return None
    try:
        return float(token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--shapeak_y_init must be 'grad' or a number.") from exc


def _load_optional_array(path_str: str, dtype=np.float32):
    if not path_str.strip():
        return None
    arr_path = resolve_in_package(path_str.strip())
    return np.load(arr_path, allow_pickle=False).astype(dtype)


def _default_output_dir(dataset: str) -> Path:
    if dataset == "Gset":
        return PACKAGE_ROOT / "result" / "mc" / "Gset"
    return PACKAGE_ROOT / "result" / "ubqp" / "Instances_uBQP"


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir.strip():
        return resolve_in_package(args.output_dir.strip())
    return _default_output_dir(args.dataset)


def _prepare_instance(args: argparse.Namespace) -> tuple[dict[str, Any], Path, str, str]:
    if args.dataset == "Gset":
        graph = parse_gset(args.Gset_id)
        data = generate_Max_cut(graph)
        output_dir = _resolve_output_dir(args)
        output_file = output_dir / f"G{args.Gset_id}.txt"
        cache_name = f"G{args.Gset_id}"
        return data, output_file, "mc_Gset", cache_name

    instance_path = resolve_in_package(args.instance_file.strip())
    if not instance_path.exists():
        raise FileNotFoundError(f"UBQP instance file not found: {instance_path}")
    data = generate_uBQP_sparse_mc(instance_path)
    output_dir = _resolve_output_dir(args)
    tag = instance_tag(instance_path)
    output_file = output_dir / f"{tag}.txt"
    return data, output_file, "ubqp", tag


def _build_solver(args: argparse.Namespace, data: dict[str, Any], dataset_for_cache: str, cache_name: str):
    try:
        if __package__ in {None, ""}:
            from shapeakadmm.shapeak_jax import SHAPEAK
        else:
            from .shapeak_jax import SHAPEAK
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", str(exc))
        raise ModuleNotFoundError(
            f"ShapeaK ADMM requires '{missing}'. Please run this script in your JAX/Optax environment."
        ) from exc

    shapeak_tol = args.shapeak_tol if args.shapeak_tol > 0 else 1e-5 * np.sqrt(data["num_vars"])
    y_init = _parse_y_init(args.shapeak_y_init)

    x0 = _load_optional_array(args.shapeak_x0_file)
    y0 = _load_optional_array(args.shapeak_y0_file)
    h_precond = build_preconditioner(
        preQ=args.shapeak_preQ,
        data=data,
        dataset=dataset_for_cache,
        name=cache_name,
        cache_override=args.shapeak_qp_cache,
        rebuild=args.shapeak_qp_rebuild,
        eps=args.shapeak_qp_eps,
        verbose=args.verbose,
    )

    return SHAPEAK(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        pen=args.shapeak_pen,
        primal_lr=args.shapeak_lr,
        batch_size=args.batch,
        max_iters=args.max_iters,
        seed=args.seed,
        preQ=args.shapeak_preQ,
        sigma=args.shapeak_sigma,
        it0=args.shapeak_it0,
        penrt=args.shapeak_penrt,
        tol=shapeak_tol,
        check=bool(args.shapeak_check),
        penf=args.shapeak_penf,
        a=args.shapeak_a,
        optimizer_type=args.shapeak_optimizer,
        verbose=args.verbose,
        cg_tol=args.shapeak_cg_tol,
        cg_iters=args.shapeak_cg_iters,
        n0=args.shapeak_n0,
        x0=x0,
        y0=y0,
        y_init=y_init,
        y_random_scale=args.shapeak_y_random_scale,
        H_precond=h_precond,
    )


def _write_result(
        output_file: Path,
        args: argparse.Namespace,
        data: dict[str, Any],
        solver,
        solving_time: float,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("params:" + str(args) + "\n")
        if args.dataset == "UBQP":
            incumbents = [float(-v) for v in solver.objVal_record]
            f.write("incumbents:" + str(incumbents) + "\n")
            f.write("timing:" + str([float(v) for v in solver.timing_record]) + "\n")
            f.write("total time:" + str(solving_time) + "\n")
            f.write("best_obj:" + str(incumbents[-1]) + "\n")
            return

        f.write("incumbents:" + str([float(v) for v in solver.objVal_record]) + "\n")
        f.write("timing:" + str([float(v) for v in solver.timing_record]) + "\n")
        f.write("total time:" + str(solving_time) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone ShapeaK ADMM runner.")

    parser.add_argument("--dataset", type=str, default="UBIP", choices=["Gset", "UBQP"])
    parser.add_argument("--Gset_id", type=int, default=22)
    parser.add_argument(
        "--instance_file",
        type=str,
        default="instance/Instances_uBQP/be100.1.sparse.mc",
        help="UBQP .sparse.mc file, resolved relative to shapeakadmm/ when not absolute.",
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max_iters", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=_str_to_bool, default=True)

    parser.add_argument("--shapeak_pen", type=float, default=1e-5)
    parser.add_argument("--shapeak_preQ", type=int, default=0)
    parser.add_argument("--shapeak_sigma", type=float, default=12)
    parser.add_argument("--shapeak_it0", type=int, default=100)
    parser.add_argument("--shapeak_penrt", type=float, default=1.25)
    parser.add_argument(
        "--shapeak_penf",
        type=str,
        default="gaa2205",
        help=(
            "ShapeaK penalty family: haa2205, gaa2205, 111105, 111100, 111101, "
            "gaa2200, haa2200, gaa2201, haa2201"
        ),
    )
    parser.add_argument("--shapeak_a", type=float, default=2.5)
    parser.add_argument("--shapeak_cg_tol", type=float, default=1e-8)
    parser.add_argument("--shapeak_cg_iters", type=int, default=6)
    parser.add_argument("--shapeak_n0", type=int, default=6000)
    parser.add_argument("--shapeak_check", type=int, default=0)
    parser.add_argument("--shapeak_optimizer", type=str, default="adam", choices=["adam", "rmsprop"])
    parser.add_argument("--shapeak_lr", type=float, default=3.5)
    parser.add_argument("--shapeak_tol", type=float, default=-1e-3)
    parser.add_argument("--shapeak_y_init", type=str, default="grad")
    parser.add_argument("--shapeak_y_random_scale", type=float, default=0.0)
    parser.add_argument("--shapeak_x0_file", type=str, default="")
    parser.add_argument("--shapeak_y0_file", type=str, default="")
    parser.add_argument("--shapeak_qp_cache", type=str, default="")
    parser.add_argument("--shapeak_qp_rebuild", type=int, default=0)
    parser.add_argument("--shapeak_qp_eps", type=float, default=0.0)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.verbose = bool(args.verbose)
    fix_seed(args.seed)

    data, output_file, dataset_for_cache, cache_name = _prepare_instance(args)
    if output_file.exists() and not args.overwrite:
        print("PASS", output_file)
        return 0

    solver = _build_solver(args, data, dataset_for_cache, cache_name)
    t0 = time.perf_counter()
    solver.optimize()
    solving_time = time.perf_counter() - t0

    _write_result(output_file, args, data, solver, solving_time)
    print(f"[shapeakadmm] result={output_file}")
    print(f"[shapeakadmm] best={float(solver.objVal_record[-1])} time={solving_time:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
