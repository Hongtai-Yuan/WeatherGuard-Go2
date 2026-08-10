import json
import sys
from pathlib import Path


import matplotlib.pyplot as plt


def load_eval_log(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute_errors(rows: list[dict]) -> tuple[list[float], list[float]]:
    vx_err = [abs(row["cmd_vx"] - row["real_vx"]) for row in rows]
    wz_err = [abs(row["cmd_wz"] - row["real_wz"]) for row in rows]
    return vx_err, wz_err


def compute_metrics(rows: list[dict]) -> dict:
    vx_err, wz_err = compute_errors(rows)

    half_idx = len(rows) // 2
    steady_vx_err = mean(vx_err[half_idx:])
    steady_wz_err = mean(wz_err[half_idx:])

    return {
        "num_steps": len(rows),
        "mean_abs_vx_error": mean(vx_err),
        "mean_abs_wz_error": mean(wz_err),
        "steady_state_vx_error": steady_vx_err,
        "steady_state_wz_error": steady_wz_err,
    }


def same_command(a: dict, b: dict, tol: float = 1e-9) -> bool:
    return (
        abs(a["cmd_vx"] - b["cmd_vx"]) < tol
        and abs(a["cmd_vy"] - b["cmd_vy"]) < tol
        and abs(a["cmd_wz"] - b["cmd_wz"]) < tol
    )


def split_into_segments(logs: list[dict]) -> list[dict]:
    if not logs:
        return []

    segments = []
    start_idx = 0

    for i in range(1, len(logs)):
        if not same_command(logs[i - 1], logs[i]):
            seg_rows = logs[start_idx:i]
            segments.append(build_segment_info(seg_rows, start_idx, i - 1))
            start_idx = i

    seg_rows = logs[start_idx:]
    segments.append(build_segment_info(seg_rows, start_idx, len(logs) - 1))
    return segments


def build_segment_info(rows: list[dict], start_step: int, end_step: int) -> dict:
    first = rows[0]
    metrics = compute_metrics(rows)

    return {
        "start_step": start_step,
        "end_step": end_step,
        "start_time": first["t"],
        "end_time": rows[-1]["t"],
        "cmd_vx": first["cmd_vx"],
        "cmd_vy": first["cmd_vy"],
        "cmd_wz": first["cmd_wz"],
        "rows": rows,
        "metrics": metrics,
    }


def print_global_metrics(log_file: str, logs: list[dict]) -> None:
    metrics = compute_metrics(logs)

    print(f"log_file: {log_file}")
    print(f"num_steps: {metrics['num_steps']}")
    print(f"mean_abs_vx_error: {metrics['mean_abs_vx_error']:.6f}")
    print(f"mean_abs_wz_error: {metrics['mean_abs_wz_error']:.6f}")
    print(f"steady_state_vx_error: {metrics['steady_state_vx_error']:.6f}")
    print(f"steady_state_wz_error: {metrics['steady_state_wz_error']:.6f}")


def print_segment_metrics(segments: list[dict]) -> None:
    print()
    print(f"num_segments: {len(segments)}")

    for idx, seg in enumerate(segments):
        m = seg["metrics"]
        print()
        print(f"Segment {idx}")
        print(f"  steps: {seg['start_step']} -> {seg['end_step']}")
        print(f"  time : {seg['start_time']:.3f} -> {seg['end_time']:.3f}")
        print(
            "  cmd  : "
            f"vx={seg['cmd_vx']:.3f}, "
            f"vy={seg['cmd_vy']:.3f}, "
            f"wz={seg['cmd_wz']:.3f}"
        )
        print(f"  num_steps              : {m['num_steps']}")
        print(f"  mean_abs_vx_error      : {m['mean_abs_vx_error']:.6f}")
        print(f"  mean_abs_wz_error      : {m['mean_abs_wz_error']:.6f}")
        print(f"  steady_state_vx_error  : {m['steady_state_vx_error']:.6f}")
        print(f"  steady_state_wz_error  : {m['steady_state_wz_error']:.6f}")


def save_plots(log_file: str, logs: list[dict]) -> None:
    t = [row["t"] for row in logs]

    cmd_vx = [row["cmd_vx"] for row in logs]
    real_vx = [row["real_vx"] for row in logs]
    cmd_wz = [row["cmd_wz"] for row in logs]
    real_wz = [row["real_wz"] for row in logs]

    vx_err = [abs(a - b) for a, b in zip(cmd_vx, real_vx)]
    wz_err = [abs(a - b) for a, b in zip(cmd_wz, real_wz)]

    stem = Path(log_file).stem
    parts = stem.split("_")

    try:
        model_idx = parts.index("model")
        model_tag = parts[model_idx - 1]
        terrain_start = 2
        terrain_end = model_idx - 1
        terrain_tag = "_".join(parts[terrain_start:terrain_end])
    except (ValueError, IndexError):
        model_tag = "unknown"
        terrain_tag = "_".join(parts[2:])

    out_dir = Path("tmp/plots") / model_tag / terrain_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(t, cmd_vx, label="cmd_vx")
    plt.plot(t, real_vx, label="real_vx")
    plt.xlabel("time (s)")
    plt.ylabel("vx")
    plt.title(f"{stem} - vx tracking")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_vx_tracking.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t, cmd_wz, label="cmd_wz")
    plt.plot(t, real_wz, label="real_wz")
    plt.xlabel("time (s)")
    plt.ylabel("wz")
    plt.title(f"{stem} - wz tracking")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_wz_tracking.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t, vx_err, label="abs_vx_error")
    plt.plot(t, wz_err, label="abs_wz_error")
    plt.xlabel("time (s)")
    plt.ylabel("absolute error")
    plt.title(f"{stem} - tracking error")
    plt.legend()
    plt.tight_layout()

    mean_vx_err = mean(vx_err)
    mean_wz_err = mean(wz_err)

    plt.text(0.02, 0.98, f'Mean VX Error: {mean_vx_err:.6f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.text(0.02, 0.90, f'Mean WZ Error: {mean_wz_err:.6f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(out_dir / f"{stem}_error.png")
    plt.close()

    half_idx = len(vx_err) // 2
    steady_vx_err = mean(vx_err[half_idx:])
    steady_wz_err = mean(wz_err[half_idx:])

    plt.figure(figsize=(8, 5))
    labels = ['Mean VX Error', 'Mean WZ Error', 'Steady VX Error', 'Steady WZ Error']
    values = [mean_vx_err, mean_wz_err, steady_vx_err, steady_wz_err]

    plt.bar(labels, values)
    plt.ylabel('Error Value')
    plt.title(f'{stem} - Tracking Errors Summary')

    for i, v in enumerate(values):
        plt.text(i, v, f'{v:.6f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_errors_summary.png")
    plt.close()

    print()
    print(f"saved plots to: {out_dir}")

def process_directory(dir_path: str):
    dir_path = Path(dir_path)

    if not dir_path.exists():
        print(f"Directory not found: {dir_path}")
        return

    files = sorted(dir_path.glob("eval_log_*.json"))

    if not files:
        print("No json files found")
        return

    print(f"Found {len(files)} files")

    for file in files:
        print("\n" + "=" * 60)
        print(f"Processing: {file.name}")

        logs = load_eval_log(str(file))

        if not logs:
            print("Empty log, skip")
            continue

        print_global_metrics(str(file), logs)

        segments = split_into_segments(logs)
        print_segment_metrics(segments)

        save_plots(str(file), logs)


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("python analyze_and_plot_eval.py <log.json | directory>")
        sys.exit(1)

    path = sys.argv[1]

    if Path(path).is_dir():
        process_directory(path)
        return

    log_file = path
    logs = load_eval_log(log_file)

    if not logs:
        print(f"Empty log file: {log_file}")
        sys.exit(1)

    print_global_metrics(log_file, logs)

    segments = split_into_segments(logs)
    print_segment_metrics(segments)

    save_plots(log_file, logs)

if __name__ == "__main__":
    main()

#python scripts/analyze_and_plot_eval.py tmp/