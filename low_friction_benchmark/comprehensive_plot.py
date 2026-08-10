import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_eval_log(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def compute_metrics(rows: list[dict]) -> dict:
    vx_err = [abs(row["cmd_vx"] - row["real_vx"]) for row in rows]
    wz_err = [abs(row["cmd_wz"] - row["real_wz"]) for row in rows]
    half_idx = int(len(rows) * 0.7)
    steady_vx_err = mean(vx_err[half_idx:])
    steady_wz_err = mean(wz_err[half_idx:])
    return {
        "mean_abs_vx_error": mean(vx_err),
        "mean_abs_wz_error": mean(wz_err),
        "steady_state_vx_error": steady_vx_err,
        "steady_state_wz_error": steady_wz_err,
    }

def main():
    base_dir = Path("tmp")
    models = ["baseline", "slippery1.0"]
    terrains = ["Unitree_Go2_Flat", "Unitree_Go2_Slippery"]
    episodes = ["hold_zero", "step_vx_03", "step_vx_06", "step_vx_09", "step_wz_03", "step_wz_06","step_wz_09", "step_lwz_rwz","switch_vx_wz"]
    
    data = {}
    for model in models:
        for terrain in terrains:
            for episode in episodes:
                filename = f"eval_log_{terrain}_{model}_model_4800_{episode}.json"
                filepath = base_dir / filename
                if filepath.exists():
                    logs = load_eval_log(str(filepath))
                    metrics = compute_metrics(logs)
                    key = f"{model}_{terrain}_{episode}"
                    data[key] = metrics
    
    # 准备数据
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # 3行3列子图
    axes = axes.flatten()
    
    for i, episode in enumerate(episodes):
        ax = axes[i]
        labels = []
        mean_vx = []
        mean_wz = []
        steady_vx = []
        steady_wz = []
        
        for model in models:
            for terrain in terrains:
                key = f"{model}_{terrain}_{episode}"
                if key in data:
                    labels.append(f"{model}\n{terrain.replace('Unitree_Go2_', '')}")
                    mean_vx.append(data[key]["mean_abs_vx_error"])
                    mean_wz.append(data[key]["mean_abs_wz_error"])
                    steady_vx.append(data[key]["steady_state_vx_error"])
                    steady_wz.append(data[key]["steady_state_wz_error"])
        
        x = np.arange(len(labels))
        width = 0.2
        
        ax.bar(x - 1.5*width, mean_vx, width, label='Mean VX Error', color='blue', alpha=0.7)
        ax.bar(x - 0.5*width, mean_wz, width, label='Mean WZ Error', color='orange', alpha=0.7)
        ax.bar(x + 0.5*width, steady_vx, width, label='Steady VX Error', color='green', alpha=0.7)
        ax.bar(x + 1.5*width, steady_wz, width, label='Steady WZ Error', color='red', alpha=0.7)
        
        ax.set_title(f'{episode} - Error Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("tmp/comprehensive_error_comparison.png", dpi=300)
    plt.close()
    print("Comprehensive error comparison plot saved to tmp/comprehensive_error_comparison.png")

if __name__ == "__main__":
    main()

#python scripts/comprehensive_plot.py
