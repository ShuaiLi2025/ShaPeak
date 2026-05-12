import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
MAIN_PY = PACKAGE_ROOT / "main.py"
GSET_DIR = PACKAGE_ROOT / "instance" / "Gset"
TEXT_TAIL_LIMIT = 2000


def _resolve_in_package(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def _tail_text(text: str, limit: int = TEXT_TAIL_LIMIT) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _cmd_to_text(cmd: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(cmd))


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir.strip():
        return _resolve_in_package(args.output_dir.strip())
    return PACKAGE_ROOT / "result" / "mc" / "Gset"


def _available_gset_ids(start_id: int, end_id: int) -> tuple[list[int], list[int]]:
    existing: list[int] = []
    missing: list[int] = []
    for gset_id in range(start_id, end_id + 1):
        if (GSET_DIR / f"G{gset_id}.txt").exists():
            existing.append(gset_id)
        else:
            missing.append(gset_id)
    return existing, missing


def _sigma_for_gset(gset_id: int) -> float:
    return 1.0 if gset_id <= 59 else 2.0


def _lr_for_gset(gset_id: int) -> float:
    return 4.5 if 22 <= gset_id <= 36 else 3.5


def _build_command(args: argparse.Namespace, gset_id: int) -> list[str]:
    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--dataset",
        "Gset",
        "--Gset_id",
        str(gset_id),
        "--max_iters",
        str(args.max_iters),
        "--batch",
        str(args.batch),
        "--seed",
        str(args.seed),
        "--verbose",
        str(int(args.verbose)),
        "--shapeak_sigma",
        str(_sigma_for_gset(gset_id)),
        "--shapeak_pen",
        str(args.shapeak_pen),
        "--shapeak_lr",
        str(_lr_for_gset(gset_id)),
        "--shapeak_a",
        str(args.shapeak_a),
        "--shapeak_penf",
        args.shapeak_penf,
        "--shapeak_y_init",
        args.shapeak_y_init,
    ]
    if args.output_dir.strip():
        cmd.extend(["--output_dir", args.output_dir.strip()])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _run_one(cmd: Sequence[str], timeout_sec: int) -> tuple[str, str, float, str, str, str]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(PACKAGE_ROOT),
            text=True,
            capture_output=True,
            timeout=None if timeout_sec <= 0 else timeout_sec,
            check=False,
        )
        elapsed = time.perf_counter() - start
        status = "success" if proc.returncode == 0 else "failed"
        return (
            status,
            str(proc.returncode),
            elapsed,
            _tail_text(proc.stdout or ""),
            _tail_text(proc.stderr or ""),
            "",
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return (
            "failed",
            "TIMEOUT",
            elapsed,
            _tail_text(exc.stdout or ""),
            _tail_text(exc.stderr or ""),
            f"Subprocess exceeded timeout ({timeout_sec}s)",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return ("failed", "EXCEPTION", elapsed, "", "", f"{type(exc).__name__}: {exc}")


def _write_summary(
        output_dir: Path,
        rows: list[dict],
        missing_ids: list[int],
        started: str,
        ended: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_txt = output_dir / "run_gset_summary.txt"
    summary_csv = output_dir / "run_gset_summary.csv"

    total = len(rows)
    success = sum(1 for row in rows if row["status"] == "success")
    failed = sum(1 for row in rows if row["status"] == "failed")
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    lines = [
        f"started_at: {started}",
        f"ended_at: {ended}",
        f"total_existing_instances: {total}",
        f"success: {success}",
        f"failed: {failed}",
        f"skipped: {skipped}",
        f"missing_ids: {missing_ids}",
    ]
    if failed:
        lines.append("failed_instances:")
        for row in rows:
            if row["status"] == "failed":
                lines.append(f"- G{row['gset_id']} | {row['error_msg'] or row['stderr_tail']}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = [
        "gset_id",
        "sigma",
        "lr",
        "status",
        "return_code",
        "elapsed_sec",
        "result_file",
        "command",
        "stdout_tail",
        "stderr_tail",
        "error_msg",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch runner for ShapeaK on Gset G22-G81.")
    parser.add_argument("--start_id", type=int, default=22)
    parser.add_argument("--end_id", type=int, default=81)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout_sec", type=int, default=0)

    parser.add_argument("--max_iters", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument("--shapeak_pen", type=float, default=1e-6)
    parser.add_argument("--shapeak_a", type=float, default=3.0)
    parser.add_argument("--shapeak_penf", type=str, default="gaa2205")
    parser.add_argument("--shapeak_y_init", type=str, default="grad")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.verbose = bool(args.verbose)

    output_dir = _output_dir(args)
    gset_ids, missing_ids = _available_gset_ids(args.start_id, args.end_id)
    if not gset_ids:
        print("[run_gset] no existing Gset instances found in requested range.")
        return 1

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict] = []
    total = len(gset_ids)
    print("[run_gset] start to batch run Gset instances with ShapeaK...")
    print(f"[run_gset] output_dir={output_dir}")

    for idx, gset_id in enumerate(gset_ids, start=1):
        result_file = output_dir / f"G{gset_id}.txt"
        cmd = _build_command(args, gset_id)
        command_text = _cmd_to_text(cmd)
        sigma = _sigma_for_gset(gset_id)
        lr = _lr_for_gset(gset_id)

        if result_file.exists() and not args.overwrite:
            print(f"[{idx}/{total}] skip G{gset_id} (exists)")
            rows.append(
                {
                    "gset_id": gset_id,
                    "sigma": sigma,
                    "lr": lr,
                    "status": "skipped",
                    "return_code": "",
                    "elapsed_sec": 0.0,
                    "result_file": str(result_file),
                    "command": command_text,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "error_msg": "",
                }
            )
            continue

        print(f"[{idx}/{total}] run  G{gset_id}")
        status, return_code, elapsed, stdout_tail, stderr_tail, error_msg = _run_one(cmd, args.timeout_sec)
        print(f"[{idx}/{total}] {status} G{gset_id} ({elapsed:.2f}s)")
        rows.append(
            {
                "gset_id": gset_id,
                "sigma": sigma,
                "lr": lr,
                "status": status,
                "return_code": return_code,
                "elapsed_sec": f"{elapsed:.6f}",
                "result_file": str(result_file),
                "command": command_text,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "error_msg": error_msg,
            }
        )

    ended = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_summary(output_dir, rows, missing_ids, started, ended)
    failed = sum(1 for row in rows if row["status"] == "failed")
    print(f"[run_gset] summary={output_dir / 'run_gset_summary.txt'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
