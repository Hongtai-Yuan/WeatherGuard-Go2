"""Visual Isaac test for the MuJoCo-exported Go2 ONNX policy.

Examples:
  python tools/play_mujoco_onnx_go2.py --num_envs 1 --cmd_vx 0.5
  python tools/play_mujoco_onnx_go2.py --num_envs 1 --cmd_wz 0.5

Keyboard control is also active:
  W/S: forward/back, A/D: lateral, Z/C: yaw, key release: zero command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play MuJoCo ONNX Go2 policy in Isaac.")
parser.add_argument("--policy", type=str, default=None, help="Path to MuJoCo-exported policy.onnx.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--freq", type=float, default=50.0)
parser.add_argument("--cmd_vx", type=float, default=0.0)
parser.add_argument("--cmd_vy", type=float, default=0.0)
parser.add_argument("--cmd_wz", type=float, default=0.0)
parser.add_argument("--policy_command_scale", type=float, default=1.0)
parser.add_argument("--policy_phase_period", type=float, default=0.6)
parser.add_argument("--ground_friction", type=float, default=0.6)
parser.add_argument("--action_scale", type=float, default=0.40)
parser.add_argument("--joint_damping_scale", type=float, default=1.0)
parser.add_argument("--rotor_inertia_scale", type=float, default=1.0)
parser.add_argument("--joint_friction", type=float, default=0.0)
parser.add_argument("--actuator_stiffness_scale", type=float, default=1.0)
parser.add_argument("--actuator_effort_scale", type=float, default=1.0)
parser.add_argument("--body_inertia_scale", type=float, default=1.0)
parser.add_argument("--base_com_shift_x", type=float, default=0.0)
parser.add_argument("--base_com_shift_y", type=float, default=0.0)
parser.add_argument("--base_com_shift_z", type=float, default=0.0)
parser.add_argument("--sysid_params_json", type=str, default=None, help="Optional JSON file produced by identify_isaac_sysid.py.")
parser.add_argument("--print_alignment", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--steps", type=int, default=0, help="0 means run until the Isaac app is closed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import omni  # noqa: E402

from quadruped.go2.go2_env import Go2MujocoOnnxEnvCfg, apply_mujoco_onnx_sysid_params, camera_follow  # noqa: E402
import quadruped.go2.go2_ctrl as go2_ctrl  # noqa: E402
from quadruped.go2.mujoco_onnx_policy import DEFAULT_POLICY_PATH, get_mujoco_onnx_policy  # noqa: E402


SYSID_PARAM_NAMES = (
    "action_scale",
    "policy_command_scale",
    "policy_phase_period",
    "ground_friction",
    "joint_damping_scale",
    "rotor_inertia_scale",
    "joint_friction",
    "actuator_stiffness_scale",
    "actuator_effort_scale",
    "body_inertia_scale",
    "base_com_shift_x",
    "base_com_shift_y",
    "base_com_shift_z",
)


def load_sysid_params(path: str) -> None:
    with Path(path).expanduser().open() as file:
        params = json.load(file)
    for name in SYSID_PARAM_NAMES:
        cli_name = f"--{name}"
        cli_was_passed = any(arg == cli_name or arg.startswith(f"{cli_name}=") for arg in sys.argv[1:])
        if name in params and not cli_was_passed:
            setattr(args_cli, name, float(params[name]))
    print(f"loaded sysid params: {path}", flush=True)


def apply_runtime_body_sysid(env, inertia_scale: float, base_com_shift: tuple[float, float, float]) -> None:
    robot = env.unwrapped.scene["unitree_go2"]
    env_ids = torch.arange(env.unwrapped.num_envs, dtype=torch.long, device="cpu")

    if abs(inertia_scale - 1.0) > 1e-6:
        inertias = robot.root_physx_view.get_inertias()
        inertias[:, 0, :] *= inertia_scale
        robot.root_physx_view.set_inertias(inertias, env_ids)

    if any(abs(value) > 1e-9 for value in base_com_shift):
        coms = robot.root_physx_view.get_coms()
        shift = torch.tensor(base_com_shift, dtype=coms.dtype, device=coms.device)
        coms[:, 0, :3] += shift
        robot.root_physx_view.set_coms(coms, env_ids)


def main() -> None:
    if args_cli.sysid_params_json:
        load_sysid_params(args_cli.sysid_params_json)

    policy_path = args_cli.policy or str(DEFAULT_POLICY_PATH)
    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"ONNX policy not found: {policy_path}\n"
            "Copy one to ckpts/unitree_go2/mujoco_slippery_policy.onnx or pass --policy."
        )

    env_cfg = Go2MujocoOnnxEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.decimation = max(1, round(1.0 / env_cfg.sim.dt / args_cli.freq))
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.observations.policy.height_scan = None
    env_cfg.actions.joint_pos.scale = args_cli.action_scale
    apply_mujoco_onnx_sysid_params(
        env_cfg,
        joint_damping_scale=args_cli.joint_damping_scale,
        rotor_inertia_scale=args_cli.rotor_inertia_scale,
        joint_friction=args_cli.joint_friction,
        actuator_stiffness_scale=args_cli.actuator_stiffness_scale,
        actuator_effort_scale=args_cli.actuator_effort_scale,
    )
    env_cfg.scene.ground.spawn.physics_material.static_friction = args_cli.ground_friction
    env_cfg.scene.ground.spawn.physics_material.dynamic_friction = args_cli.ground_friction

    go2_ctrl.init_base_vel_cmd(args_cli.num_envs)
    go2_ctrl.base_vel_cmd_input[:] = torch.tensor(
        [args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_wz], dtype=torch.float32
    )

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    policy = get_mujoco_onnx_policy(
        env,
        policy_path=policy_path,
        command_scale=args_cli.policy_command_scale,
        phase_period=args_cli.policy_phase_period,
    )
    apply_runtime_body_sysid(
        env,
        inertia_scale=args_cli.body_inertia_scale,
        base_com_shift=(args_cli.base_com_shift_x, args_cli.base_com_shift_y, args_cli.base_com_shift_z),
    )

    try:
        system_input = carb.input.acquire_input_interface()
        system_input.subscribe_to_keyboard_events(
            omni.appwindow.get_default_app_window().get_keyboard(),
            go2_ctrl.sub_keyboard_event,
        )
    except RuntimeError:
        print("[WARN] Keyboard input is unavailable; continuing with fixed command.", flush=True)

    obs, _ = env.reset()
    policy.reset_to_mujoco_default_pose()
    if args_cli.print_alignment:
        print(policy.alignment_report(), flush=True)

    sim_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    step = 0
    vx_samples = []
    vy_abs_samples = []
    z_samples = []
    pitch_abs_samples = []
    roll_abs_samples = []
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if not getattr(args_cli, "headless", False):
            camera_follow(env)
        robot_data = env.unwrapped.scene["unitree_go2"].data
        root_vel = robot_data.root_lin_vel_b[0].detach().cpu().numpy()
        root_height = robot_data.root_pos_w[0, 2].detach().cpu().item()
        if step >= args_cli.warmup_steps:
            vx_samples.append(float(root_vel[0]))
            vy_abs_samples.append(abs(float(root_vel[1])))
            z_samples.append(float(root_height))
            projected_gravity = robot_data.projected_gravity_b[0].detach().cpu().numpy()
            base_up = -projected_gravity
            roll_abs_samples.append(abs(float(np.arctan2(base_up[1], base_up[2]))))
            pitch_abs_samples.append(abs(float(np.arctan2(-base_up[0], base_up[2]))))
        if step % 50 == 0:
            print(
                f"step={step:05d} cmd={go2_ctrl.base_vel_cmd_input[0].numpy()} "
                f"root_lin_vel_b={root_vel} root_z={root_height:.3f} "
                f"action_minmax=({actions.min().item():.3f},{actions.max().item():.3f})",
                flush=True,
            )
        step += 1
        if args_cli.steps and step >= args_cli.steps:
            break

        elapsed = time.time() - start_time
        if elapsed < sim_step_dt:
            time.sleep(sim_step_dt - elapsed)

    if vx_samples:
        mean_vx = sum(vx_samples) / len(vx_samples)
        mean_abs_vy = sum(vy_abs_samples) / len(vy_abs_samples)
        mean_z = sum(z_samples) / len(z_samples)
        std_z = float(np.std(z_samples))
        mean_abs_roll = sum(roll_abs_samples) / len(roll_abs_samples)
        mean_abs_pitch = sum(pitch_abs_samples) / len(pitch_abs_samples)
        print(
            f"summary after warmup={args_cli.warmup_steps}: "
            f"cmd_vx={args_cli.cmd_vx:.3f} mean_vx={mean_vx:.3f} "
            f"tracking_ratio={mean_vx / args_cli.cmd_vx if args_cli.cmd_vx else 0.0:.2f} "
            f"mean_abs_vy={mean_abs_vy:.3f} mean_root_z={mean_z:.3f} "
            f"std_root_z={std_z:.3f} mean_abs_roll={mean_abs_roll:.3f} "
            f"mean_abs_pitch={mean_abs_pitch:.3f}",
            flush=True,
        )

    env.close()
    if getattr(args_cli, "headless", False):
        os._exit(0)
    simulation_app.close()


if __name__ == "__main__":
    main()
