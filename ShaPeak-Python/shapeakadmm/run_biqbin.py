import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
MAIN_PY = PACKAGE_ROOT / "main.py"
TEXT_TAIL_LIMIT = 2000


def _resolve_in_package(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def _instance_tag(path: Path) -> str:
    name = path.name
    if name.endswith(".sparse.mc"):
        return name[:-len(".sparse.mc")]
    return path.stem


def _natural_key(path: Path):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path.name)]


def _tail_text(text: str, limit: int = TEXT_TAIL_LIMIT) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _cmd_to_text(cmd: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(cmd))


def _discover_instances(args: argparse.Namespace) -> list[Path]:
    if args.instance_file.strip():
        path = _resolve_in_package(args.instance_file.strip())
        if not path.exists():
            raise FileNotFoundError(f"Instance file not found: {path}")
        return [path]

    instance_dir = _resolve_in_package(args.instance_dir)
    if not instance_dir.exists():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")

    files = sorted(instance_dir.glob(args.pattern), key=_natural_key)
    if args.limit > 0:
        files = files[:args.limit]
    return files


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir.strip():
        return _resolve_in_package(args.output_dir.strip())
    return PACKAGE_ROOT / "result" / "ubqp" / "Instances_uBQP"


def _build_command(args: argparse.Namespace, instance_path: Path) -> list[str]:
    rel_instance = instance_path
    try:
        rel_instance = instance_path.relative_to(PACKAGE_ROOT)
    except ValueError:
        pass

    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--dataset",
        "UBQP",
        "--instance_file",
        str(rel_instance),
        "--max_iters",
        str(args.max_iters),
        "--batch",
        str(args.batch),
        "--seed",
        str(args.seed),
        "--verbose",
        str(int(args.verbose)),
        "--shapeak_sigma",
        str(args.shapeak_sigma),
        "--shapeak_pen",
        str(args.shapeak_pen),
        "--shapeak_lr",
        str(args.shapeak_lr),
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


def _write_summary(output_dir: Path, rows: list[dict], started: str, ended: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_txt = output_dir / "run_biqbin_summary.txt"
    summary_csv = output_dir / "run_biqbin_summary.csv"

    total = len(rows)
    success = sum(1 for row in rows if row["status"] == "success")
    failed = sum(1 for row in rows if row["status"] == "failed")
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    lines = [
        f"started_at: {started}",
        f"ended_at: {ended}",
        f"total_instances: {total}",
        f"success: {success}",
        f"failed: {failed}",
        f"skipped: {skipped}",
    ]
    if failed:
        lines.append("failed_instances:")
        for row in rows:
            if row["status"] == "failed":
                lines.append(f"- {row['instance']} | {row['error_msg'] or row['stderr_tail']}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = [
        "instance",
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
    parser = argparse.ArgumentParser(description="Batch runner for ShapeaK on Instances_uBQP.")
    parser.add_argument("--instance_file", type=str, default="")
    parser.add_argument("--instance_dir", type=str, default="instance/Instances_uBQP")
    parser.add_argument("--pattern", type=str, default="*.sparse.mc")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout_sec", type=int, default=0)

    parser.add_argument("--max_iters", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--shapeak_sigma", type=float, default=12)
    parser.add_argument("--shapeak_pen", type=float, default=1e-5)
    parser.add_argument("--shapeak_lr", type=float, default=3.5)
    parser.add_argument("--shapeak_a", type=float, default=2.5)
    parser.add_argument("--shapeak_penf", type=str, default="haa2205")
    parser.add_argument("--shapeak_y_init", type=str, default="grad")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.verbose = bool(args.verbose)

    output_dir = _output_dir(args)
    instances = _discover_instances(args)
    if not instances:
        print("[run_biqbin] no instances found.")
        return 1

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict] = []
    total = len(instances)
    print("[run_gset] start to batch run BiqBin instances with ShapeaK...")
    print(f"[run_biqbin] output_dir={output_dir}")

    for idx, instance_path in enumerate(instances, start=1):
        tag = _instance_tag(instance_path)
        result_file = output_dir / f"{tag}.txt"
        cmd = _build_command(args, instance_path)
        command_text = _cmd_to_text(cmd)

        if result_file.exists() and not args.overwrite:
            print(f"[{idx}/{total}] skip {instance_path.name} (exists)")
            rows.append(
                {
                    "instance": instance_path.name,
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

        print(f"[{idx}/{total}] run  {instance_path.name}")
        status, return_code, elapsed, stdout_tail, stderr_tail, error_msg = _run_one(cmd, args.timeout_sec)
        print(f"[{idx}/{total}] {status} {instance_path.name} ({elapsed:.2f}s)")
        rows.append(
            {
                "instance": instance_path.name,
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
    _write_summary(output_dir, rows, started, ended)
    failed = sum(1 for row in rows if row["status"] == "failed")
    print(f"[run_biqbin] summary={output_dir / 'run_biqbin_summary.txt'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
