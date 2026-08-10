"""Quick ONNX metadata/inference check for the MuJoCo Go2 policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx


DEFAULT_POLICY = (
    Path(__file__).resolve().parents[1]
    / "ckpts"
    / "unitree_go2"
    / "mujoco_slippery_policy.onnx"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is not installed. Run: pip install onnxruntime") from exc

    model = onnx.load(str(args.policy))
    meta = {entry.key: entry.value for entry in model.metadata_props}
    print(f"policy: {args.policy}")
    print("inputs:")
    for item in model.graph.input:
        dims = [dim.dim_value or dim.dim_param for dim in item.type.tensor_type.shape.dim]
        print(f"  {item.name}: {dims}")
    print("outputs:")
    for item in model.graph.output:
        dims = [dim.dim_value or dim.dim_param for dim in item.type.tensor_type.shape.dim]
        print(f"  {item.name}: {dims}")
    print("metadata:")
    for key in ("joint_names", "default_joint_pos", "observation_names", "action_scale"):
        print(f"  {key}: {meta.get(key)}")

    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dummy_obs = np.zeros((1, 47), dtype=np.float32)
    actions = session.run([output_name], {input_name: dummy_obs})[0]
    print(f"dummy action shape: {actions.shape}")
    print(f"dummy action finite: {np.isfinite(actions).all()}")
    print(f"dummy action min/max: {actions.min():.4f} / {actions.max():.4f}")


if __name__ == "__main__":
    main()
