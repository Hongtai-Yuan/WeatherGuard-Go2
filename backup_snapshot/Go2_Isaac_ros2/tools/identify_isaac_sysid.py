"""Black-box sysid sweep for the MuJoCo ONNX Go2 policy in Isaac.

This is the practical first pass before full real-data inverse-dynamics sysid:
sample a compact physical parameterization, run Isaac, score rollout metrics,
and save the best parameter set for later validation/tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
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


BASE_RANGES = {
    "action_scale": (0.34, 0.44),
    "policy_command_scale": (1.0, 1.5),
    "policy_phase_period": (0.48, 0.70),
    "joint_damping_scale": (0.70, 1.35),
    "rotor_inertia_scale": (0.60, 1.40),
    "joint_friction": (0.0, 0.06),
    "actuator_stiffness_scale": (0.85, 1.25),
    "actuator_effort_scale": (1.0, 1.20),
    "body_inertia_scale": (0.85, 1.20),
    "base_com_shift_x": (-0.020, 0.020),
    "base_com_shift_y": (-0.015, 0.015),
    "base_com_shift_z": (-0.015, 0.015),
    "ground_friction": (0.60, 1.00),
}

FAST_RANGES = {
    "action_scale": (0.34, 0.52),
    "policy_command_scale": (1.0, 2.5),
    "policy_phase_period": (0.30, 0.65),
    "joint_damping_scale": (0.45, 1.10),
    "rotor_inertia_scale": (0.50, 1.30),
    "joint_friction": (0.0, 0.025),
    "actuator_stiffness_scale": (1.00, 1.80),
    "actuator_effort_scale": (1.00, 1.60),
    "body_inertia_scale": (0.85, 1.15),
    "base_com_shift_x": (-0.025, 0.010),
    "base_com_shift_y": (-0.018, 0.018),
    "base_com_shift_z": (-0.012, 0.012),
    "ground_friction": (0.60, 1.20),
}


def ranges_for_profile(profile: str) -> dict[str, tuple[float, float]]:
    if profile == "fast":
        return FAST_RANGES
    return BASE_RANGES


def sample_candidate(rng: random.Random, ranges: dict[str, tuple[float, float]]) -> dict[str, float]:
    return {name: rng.uniform(low, high) for name, (low, high) in ranges.items()}


def default_candidate() -> dict[str, float]:
    return {
        "action_scale": 0.40,
        "policy_command_scale": 1.0,
        "policy_phase_period": 0.6,
        "joint_damping_scale": 1.0,
        "rotor_inertia_scale": 1.0,
        "joint_friction": 0.0,
        "actuator_stiffness_scale": 1.0,
        "actuator_effort_scale": 1.0,
        "body_inertia_scale": 1.0,
        "base_com_shift_x": 0.0,
        "base_com_shift_y": 0.0,
        "base_com_shift_z": 0.0,
        "ground_friction": 0.6,
    }


def parse_summary(output: str) -> dict[str, float]:
    matches = list(SUMMARY_RE.finditer(output))
    if not matches:
        raise RuntimeError("No summary line found in rollout output.")
    return {key: float(value) for key, value in matches[-1].groupdict().items()}


def score(metrics: dict[str, float], target_vx: float, target_z: float) -> float:
    vx_error = metrics["mean_vx"] - target_vx
    height_error = metrics["mean_root_z"] - target_z
    slow_penalty = max(0.0, 0.85 * target_vx - metrics["mean_vx"])
    return (
        6.0 * vx_error * vx_error
        + 8.0 * slow_penalty * slow_penalty
        + 2.0 * metrics["mean_abs_vy"] * metrics["mean_abs_vy"]
        + 10.0 * metrics["std_root_z"] * metrics["std_root_z"]
        + 0.5 * height_error * height_error
        + 0.5 * metrics["mean_abs_roll"] * metrics["mean_abs_roll"]
        + 0.5 * metrics["mean_abs_pitch"] * metrics["mean_abs_pitch"]
    )


def run_candidate(args, candidate: dict[str, float]) -> dict[str, float | str]:
    script = Path(__file__).with_name("play_mujoco_onnx_go2.py")
    cmd = [
        sys.executable,
        str(script),
        "--headless",
        "--num_envs",
        "1",
        "--cmd_vx",
        str(args.cmd_vx),
        "--steps",
        str(args.steps),
        "--warmup_steps",
        str(args.warmup_steps),
    ]
    for name, value in candidate.items():
        cmd.extend([f"--{name}", f"{value:.8f}"])

    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
        check=False,
    )
    result: dict[str, float | str] = dict(candidate)
    result["returncode"] = float(proc.returncode)
    if proc.returncode != 0:
        result["loss"] = float("inf")
        result["error"] = proc.stdout[-2000:]
        return result

    try:
        metrics = parse_summary(proc.stdout)
    except RuntimeError as exc:
        result["loss"] = float("inf")
        result["error"] = f"{exc}\n{proc.stdout[-2000:]}"
        return result

    result.update(metrics)
    result["loss"] = score(metrics, target_vx=args.cmd_vx, target_z=args.target_z)
    return result


def write_results(output_dir: Path, rows: list[dict[str, float | str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "isaac_sysid_sweep.csv"
    json_path = output_dir / "isaac_sysid_best.json"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda row: float(row["loss"]))
    with json_path.open("w") as file:
        json.dump(best, file, indent=2, sort_keys=True)

    print(f"saved sweep: {csv_path}")
    print(f"saved best:  {json_path}")
    print("best:")
    print(json.dumps(best, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run black-box Isaac sysid sweep for Go2 MuJoCo ONNX policy.")
    parser.add_argument("--cmd_vx", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target_z", type=float, default=0.35)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/sysid"))
    parser.add_argument("--profile", choices=("base", "fast"), default="base")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ranges = ranges_for_profile(args.profile)
    candidates = [default_candidate()]
    candidates.extend(sample_candidate(rng, ranges) for _ in range(max(0, args.samples - 1)))

    rows = []
    for idx, candidate in enumerate(candidates):
        print(f"[{idx + 1}/{len(candidates)}] candidate={json.dumps(candidate, sort_keys=True)}", flush=True)
        row = run_candidate(args, candidate)
        rows.append(row)
        print(
            f"  loss={row.get('loss')} mean_vx={row.get('mean_vx')} "
            f"mean_abs_vy={row.get('mean_abs_vy')} std_z={row.get('std_root_z')}",
            flush=True,
        )

    write_results(args.output_dir, rows)


if __name__ == "__main__":
    main()
