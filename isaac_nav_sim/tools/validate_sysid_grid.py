"""Validate a saved sysid preset across speed and ground-friction grids."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SUMMARY_RE = re.compile(
    r"cmd_vx=(?P<cmd_vx>[-+0-9.]+)\s+"
    r"mean_vx=(?P<mean_vx>[-+0-9.]+)\s+"
    r"tracking_ratio=(?P<tracking_ratio>[-+0-9.]+)\s+"
    r"mean_abs_vy=(?P<mean_abs_vy>[-+0-9.]+)\s+"
    r"mean_root_z=(?P<mean_root_z>[-+0-9.]+)\s+"
    r"std_root_z=(?P<std_root_z>[-+0-9.]+)\s+"
    r"mean_abs_roll=(?P<mean_abs_roll>[-+0-9.]+)\s+"
    r"mean_abs_pitch=(?P<mean_abs_pitch>[-+0-9.]+)"
)


def parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def parse_summary(output: str) -> dict[str, float]:
    matches = list(SUMMARY_RE.finditer(output))
    if not matches:
        raise RuntimeError("No summary line found in rollout output.")
    return {key: float(value) for key, value in matches[-1].groupdict().items()}


def validate_row(metrics: dict[str, float]) -> tuple[bool, str]:
    reasons = []
    cmd_vx = metrics["cmd_vx"]
    mean_vx = metrics["mean_vx"]
    if cmd_vx > 0.15 and mean_vx < 0.75 * cmd_vx:
        reasons.append("slow")
    if metrics["mean_abs_vy"] > max(0.18, 0.25 * max(cmd_vx, 0.1)):
        reasons.append("side_drift")
    if metrics["std_root_z"] > 0.025:
        reasons.append("height_bounce")
    if metrics["mean_abs_roll"] > 0.20 or metrics["mean_abs_pitch"] > 0.20:
        reasons.append("body_tilt")
    return not reasons, ",".join(reasons) or "ok"


def run_case(args, cmd_vx: float, friction: float) -> dict[str, float | str]:
    script = Path(__file__).with_name("play_mujoco_onnx_go2.py")
    with args.params_json.open() as file:
        params = json.load(file)
    params["ground_friction"] = friction

    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=args.output_dir, delete=False) as file:
        json.dump(params, file)
        case_params_path = Path(file.name)

    try:
        cmd = [
            sys.executable,
            str(script),
            "--headless",
            "--num_envs",
            "1",
            "--cmd_vx",
            str(cmd_vx),
            "--sysid_params_json",
            str(case_params_path),
            "--steps",
            str(args.steps),
            "--warmup_steps",
            str(args.warmup_steps),
        ]
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
    finally:
        case_params_path.unlink(missing_ok=True)
    row: dict[str, float | str] = {
        "requested_cmd_vx": cmd_vx,
        "requested_ground_friction": friction,
        "returncode": float(proc.returncode),
    }
    if proc.returncode != 0:
        row["ok"] = "False"
        row["status"] = "process_error"
        row["error"] = proc.stdout[-2000:]
        return row
    try:
        row.update(parse_summary(proc.stdout))
    except RuntimeError as exc:
        row["ok"] = "False"
        row["status"] = "parse_error"
        row["error"] = f"{exc}\n{proc.stdout[-2000:]}"
        return row
    ok, status = validate_row(row)  # type: ignore[arg-type]
    row["ok"] = str(ok)
    row["status"] = status
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Go2 sysid preset over velocity/friction grids.")
    parser.add_argument(
        "--params_json",
        type=Path,
        default=Path("outputs/sysid_fast_phase/isaac_sysid_fast_stable.json"),
    )
    parser.add_argument("--cmd_vxs", type=parse_float_list, default=parse_float_list("0.3,0.6,1.0"))
    parser.add_argument("--frictions", type=parse_float_list, default=parse_float_list("0.1,0.2,0.4,0.6,1.0"))
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/sysid_validation"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cmd_vx in args.cmd_vxs:
        for friction in args.frictions:
            print(f"validate cmd_vx={cmd_vx:.2f} friction={friction:.2f}", flush=True)
            row = run_case(args, cmd_vx=cmd_vx, friction=friction)
            rows.append(row)
            print(
                f"  ok={row.get('ok')} status={row.get('status')} "
                f"mean_vx={row.get('mean_vx')} vy={row.get('mean_abs_vy')} "
                f"std_z={row.get('std_root_z')}",
                flush=True,
            )

    csv_path = args.output_dir / "grid_validation.csv"
    json_path = args.output_dir / "grid_validation.json"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as file:
        json.dump(rows, file, indent=2, sort_keys=True)

    ok_count = sum(row.get("ok") == "True" for row in rows)
    print(f"saved csv:  {csv_path}")
    print(f"saved json: {json_path}")
    print(f"passed: {ok_count}/{len(rows)}")


if __name__ == "__main__":
    main()
