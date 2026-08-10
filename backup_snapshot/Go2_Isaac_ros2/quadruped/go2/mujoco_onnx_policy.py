"""MuJoCo-exported ONNX locomotion policy adapter for Isaac Go2.

This module keeps the existing Isaac environment intact and builds the
47-dimensional observation expected by the MuJoCo/mjlab Go2 velocity policy:

  base_ang_vel, projected_gravity, command, phase, joint_pos, joint_vel, actions

The ONNX file already contains the observation normalizer, so this adapter feeds
raw observations in the exported MuJoCo order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import torch

import quadruped.go2.go2_ctrl as go2_ctrl


DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "ckpts"
    / "unitree_go2"
    / "mujoco_slippery_policy.onnx"
)


@dataclass(frozen=True)
class MujocoPolicyMetadata:
    joint_names: tuple[str, ...]
    default_joint_pos: torch.Tensor
    action_scale: float
    observation_names: tuple[str, ...]


def _metadata_dict(model: onnx.ModelProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in model.metadata_props}


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_mujoco_policy_metadata(policy_path: str | Path, device: torch.device | str) -> MujocoPolicyMetadata:
    model = onnx.load(str(policy_path))
    meta = _metadata_dict(model)

    missing = [key for key in ("joint_names", "default_joint_pos", "action_scale", "observation_names") if key not in meta]
    if missing:
        raise ValueError(f"ONNX policy is missing required metadata keys: {missing}")

    return MujocoPolicyMetadata(
        joint_names=_parse_str_tuple(meta["joint_names"]),
        default_joint_pos=torch.tensor(
            _parse_float_tuple(meta["default_joint_pos"]),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        action_scale=float(meta["action_scale"]),
        observation_names=_parse_str_tuple(meta["observation_names"]),
    )


def _resolve_name_order(available_names: Iterable[str], target_names: Iterable[str]) -> list[int]:
    available = list(available_names)
    name_to_idx = {name: idx for idx, name in enumerate(available)}
    missing = [name for name in target_names if name not in name_to_idx]
    if missing:
        raise ValueError(
            "Isaac Go2 joint names do not match the MuJoCo policy metadata. "
            f"Missing joints: {missing}. Available joints: {available}"
        )
    return [name_to_idx[name] for name in target_names]


def _phase(
    num_envs: int,
    step_count: int,
    step_dt: float,
    command: torch.Tensor,
    device: torch.device,
    period: float,
) -> torch.Tensor:
    global_phase = (step_count * step_dt) % period / period
    phase = torch.empty((num_envs, 2), dtype=torch.float32, device=device)
    phase[:, 0] = torch.sin(torch.tensor(global_phase * 2.0 * np.pi, device=device))
    phase[:, 1] = torch.cos(torch.tensor(global_phase * 2.0 * np.pi, device=device))
    stand_mask = torch.linalg.norm(command, dim=1) < 0.1
    return torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)


class MujocoOnnxPolicy:
    """Callable Isaac policy backed by the MuJoCo-exported ONNX actor."""

    def __init__(
        self,
        env,
        policy_path: str | Path = DEFAULT_POLICY_PATH,
        command_scale: float = 1.0,
        phase_period: float = 0.6,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for MuJoCo ONNX policy inference. "
                "Install it in env_isaaclab with: pip install onnxruntime"
            ) from exc

        self.env = env
        self.unwrapped = env.unwrapped
        self.device = torch.device(self.unwrapped.device)
        self.policy_path = Path(policy_path).expanduser().resolve()
        self.meta = load_mujoco_policy_metadata(self.policy_path, self.device)

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if self.device.type == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(self.policy_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        robot = self.unwrapped.scene["unitree_go2"]
        self.robot = robot
        self.joint_order = torch.tensor(
            _resolve_name_order(robot.data.joint_names, self.meta.joint_names),
            dtype=torch.long,
            device=self.device,
        )
        self.last_action_mujoco = torch.zeros((self.unwrapped.num_envs, len(self.meta.joint_names)), dtype=torch.float32, device=self.device)
        self.step_count = 0
        self.step_dt = float(self.unwrapped.step_dt)
        self.command_scale = float(command_scale)
        self.phase_period = float(phase_period)

    def _build_obs(self) -> torch.Tensor:
        data = self.robot.data
        command = go2_ctrl.base_vel_cmd_input.to(self.device)
        if command.shape[0] != self.unwrapped.num_envs:
            command = command[:1].repeat(self.unwrapped.num_envs, 1)
        policy_command = command * self.command_scale

        joint_pos = data.joint_pos[:, self.joint_order]
        joint_vel = data.joint_vel[:, self.joint_order]
        default_pos = self.meta.default_joint_pos.repeat(self.unwrapped.num_envs, 1)

        obs_terms = [
            data.root_ang_vel_b,
            data.projected_gravity_b,
            policy_command,
            _phase(
                self.unwrapped.num_envs,
                self.step_count,
                self.step_dt,
                policy_command,
                self.device,
                self.phase_period,
            ),
            joint_pos - default_pos,
            joint_vel,
            self.last_action_mujoco,
        ]
        obs = torch.cat(obs_terms, dim=-1)
        if obs.shape[-1] != 47:
            raise RuntimeError(f"MuJoCo ONNX policy expects 47 obs values, got {obs.shape[-1]}")
        return obs

    def alignment_report(self) -> str:
        action_term = getattr(getattr(self.unwrapped, "action_manager", None), "_terms", {}).get("joint_pos")
        action_joint_names = getattr(action_term, "_joint_names", None)
        action_joint_ids = getattr(action_term, "_joint_ids", None)
        action_scale = getattr(action_term, "_scale", None)
        if isinstance(action_scale, torch.Tensor):
            action_scale = action_scale[0].detach().cpu().numpy().round(4).tolist()
        joint_pos = self.robot.data.joint_pos[:, self.joint_order]
        isaac_default_pos = self.robot.data.default_joint_pos[:, self.joint_order]
        default_pos = self.meta.default_joint_pos.repeat(self.unwrapped.num_envs, 1)
        default_error = torch.max(torch.abs(joint_pos - default_pos)).item()
        isaac_default_error = torch.max(torch.abs(isaac_default_pos - default_pos)).item()
        obs = self._build_obs()
        lines = [
            "MuJoCo ONNX <-> Isaac alignment report",
            f"  policy_path: {self.policy_path}",
            f"  policy_joint_order: {list(self.meta.joint_names)}",
            f"  isaac_robot_joint_names: {self.robot.data.joint_names}",
            f"  obs_dim: {obs.shape[-1]}",
            f"  default_joint_pos_max_abs_error_after_reset: {default_error:.6f}",
            f"  isaac_default_vs_onnx_max_abs_error: {isaac_default_error:.6f}",
            f"  reset_joint_pos_mujoco_order: {joint_pos[0].detach().cpu().numpy().round(4).tolist()}",
            f"  isaac_default_joint_pos_mujoco_order: {isaac_default_pos[0].detach().cpu().numpy().round(4).tolist()}",
            f"  onnx_default_joint_pos: {default_pos[0].detach().cpu().numpy().round(4).tolist()}",
            f"  action_manager_joint_names: {action_joint_names}",
            f"  action_manager_joint_ids: {action_joint_ids}",
            f"  action_scale_from_onnx_metadata: {self.meta.action_scale}",
            f"  action_scale_used_by_isaac: {action_scale}",
            f"  policy_command_scale: {self.command_scale}",
            f"  policy_phase_period: {self.phase_period}",
        ]
        return "\n".join(lines)

    def reset_to_mujoco_default_pose(self) -> None:
        env_ids = torch.arange(self.unwrapped.num_envs, dtype=torch.long, device=self.device)
        data = self.robot.data

        root_state = data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.unwrapped.scene.env_origins[env_ids]
        joint_pos = data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(data.default_joint_vel[env_ids])

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        self.unwrapped.scene.write_data_to_sim()
        self.unwrapped.sim.forward()

        self.last_action_mujoco.zero_()
        self.step_count = 0

    def __call__(self, _obs) -> torch.Tensor:
        obs = self._build_obs()
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        raw_actions = self.session.run([self.output_name], {self.input_name: obs_np})[0]
        action_mujoco = torch.from_numpy(raw_actions).to(self.device, dtype=torch.float32)
        self.last_action_mujoco = action_mujoco

        # Go2MujocoOnnxEnvCfg sets JointPositionActionCfg.joint_names to the
        # ONNX/MuJoCo metadata order with preserve_order=True, so the network
        # output can be sent directly as normalized joint-position actions.
        self.step_count += 1
        return action_mujoco


def get_mujoco_onnx_policy(
    env,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    command_scale: float = 1.0,
    phase_period: float = 0.6,
) -> MujocoOnnxPolicy:
    return MujocoOnnxPolicy(
        env,
        policy_path=policy_path,
        command_scale=command_scale,
        phase_period=phase_period,
    )
