"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import json
import mjlab


import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
#from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  command_file: str | None = None   # 固定 command JSON 路径
  episode_name: str | None = None   # 选哪个 episode
  model_tag: str | None = None   # 模型类型标签，如 baseline / slippery / robust
  all_episodes: bool = False  # 是否跑所有 episodes
  video: bool = False
  video_length: int = 500
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
 
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
   # 先读取固定 command JSON，并选中一个 episode
  if cfg.command_file is None:
    raise ValueError("--command-file is required")
  if cfg.episode_name is None:
    raise ValueError("--episode-name is required")

  command_data = load_command_json(cfg.command_file)
  episode_spec = get_episode_spec(command_data, cfg.episode_name)

  print(f"[INFO] command_file: {cfg.command_file}")
  print(f"[INFO] episode_name: {cfg.episode_name}")
  print(f"[INFO] episode_spec keys: {list(episode_spec.keys())}")

    # 按 control_dt 展开成逐 step command
  

  env_cfg = load_env_cfg(task_id, play=True)
  #print(vars(env_cfg.sim))
  agent_cfg = load_rl_cfg(task_id)

  control_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
  step_commands = build_step_commands(episode_spec, control_dt)

  print(f"[INFO] num_step_commands: {len(step_commands)}")
  print(f"[INFO] first_5_step_commands: {step_commands[:5]}")

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  # 🎥 如果用户没传分辨率，给一个默认高清
  if cfg.video_height is None:
      env_cfg.viewer.height = 720
  else:
      env_cfg.viewer.height = cfg.video_height

  if cfg.video_width is None:
      env_cfg.viewer.width = 1280
  else:
      env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else "human"#None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None
    

    # 从 task_id 提取地形名（Flat / Slippery）
    terrain_name = "Flat" if "Flat" in task_id else "Slippery"

    video_folder = (
        log_dir
        / "videos"
        / terrain_name
        / (cfg.episode_name if cfg.episode_name is not None else "unknown_episode")
    )
      
    base_env = VideoRecorder(
      base_env,
      video_folder=video_folder,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
 
  twist_term = base_env.command_manager._terms["twist"]
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)
      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
  
  logs = []
  obs, _ = env.reset()

  for step in range(len(step_commands)):
    # 取当前 step 的 [vx, vy, wz]
    cmd_single = torch.tensor([step_commands[step]], device=base_env.device)

    cmd = cmd_single.repeat(base_env.num_envs, 1)
    twist_term.set_eval_command(cmd)

    with torch.inference_mode():
      action = policy(obs)

    obs, rew, dones, infos = env.step(action)
    base_env.render()

        # 记录当前 step 的原始评测数据
    logs.append(
      {
        "step": step,
        "t": step * control_dt,
        "cmd_vx": float(cmd_single[0, 0].item()),
        "cmd_vy": float(cmd_single[0, 1].item()),
        "cmd_wz": float(cmd_single[0, 2].item()),
        "real_vx": float(base_env.scene["robot"].data.root_link_lin_vel_b[0, 0].item()),
        "real_vy": float(base_env.scene["robot"].data.root_link_lin_vel_b[0, 1].item()),
        "real_wz": float(base_env.scene["robot"].data.root_link_ang_vel_b[0, 2].item()),
        "reward": float(rew[0].item()),
        "done": bool(dones[0].item()),
      }
    )

  
    if step % 50 == 0:
      print(f"step={step}")
      print("cmd:", twist_term.command[0].detach().cpu().numpy())
      print(
        "real_lin_vel_b:",
        base_env.scene["robot"].data.root_link_lin_vel_b[0].detach().cpu().numpy(),
      )
      print(
        "real_ang_vel_b:",
        base_env.scene["robot"].data.root_link_ang_vel_b[0].detach().cpu().numpy(),
      )
  # rollout 结束后，把原始日志写到文件
  ckpt_name = Path(cfg.checkpoint_file).stem if cfg.checkpoint_file is not None else "no_ckpt"

  # 把 task_id 里的横杠改成下划线，方便放进文件名。
  terrain_tag = task_id.replace("-", "_")

  # 模型类型标签：优先用命令行显式传入的 model_tag。
  # 如果没传，就退化成 checkpoint 文件名。
  policy_tag = cfg.model_tag if cfg.model_tag is not None else ckpt_name

  # 文件名同时包含地形、模型类型、checkpoint、episode，避免不同实验结果混淆。
  output_path = Path("tmp") / (
    f"eval_log_{terrain_tag}_{policy_tag}_{ckpt_name}_{cfg.episode_name}.json"
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)

  #print(f"[DEBUG] about to save log, len(logs)={len(logs)}")

  with output_path.open("w", encoding="utf-8") as f:
    json.dump(logs, f, ensure_ascii=False, indent=2)

  #print(f"[DEBUG] file saved exists={output_path.exists()}")
  print(f"[INFO] saved raw log to: {output_path}")
 
  env.close()

def load_command_json(command_file: str) -> dict:
    """读取固定 command 的 JSON 文件。"""
    command_path = Path(command_file)
    with command_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_episode_spec(command_data: dict, episode_name: str) -> dict:
    """从 JSON 里取出指定 episode，兼容 dict / list 两种 episodes 结构。"""
    episodes = command_data["episodes"]

    # 结构1: {"episodes": {"step_vx_2": {...}}}
    if isinstance(episodes, dict):
        if episode_name not in episodes:
            raise KeyError(f"episode '{episode_name}' not found in command file")
        return episodes[episode_name]

    # 结构2: {"episodes": [{"name": "step_vx_2", ...}, ...]}
    if isinstance(episodes, list):
        for ep in episodes:
            if ep.get("name") == episode_name:
                return ep
        raise KeyError(f"episode '{episode_name}' not found in command file")

    raise TypeError("unsupported command file format: 'episodes' must be dict or list")

def build_step_commands(episode_spec: dict, control_dt: float) -> list[list[float]]:
  """把 episode 里的 segments 按 control_dt 展开成逐 step 的 [vx, vy, wz]。"""
  if "segments" not in episode_spec:
    raise KeyError("episode_spec must contain 'segments'")

  segments = episode_spec["segments"]
  if len(segments) == 0:
    raise ValueError("episode_spec['segments'] must not be empty")

  step_commands: list[list[float]] = []

  for seg in segments:
    duration_s = float(seg["duration"])
    num_steps = max(1, int(round(duration_s / control_dt)))

    # 每个 segment 对应一个常量 command
    cmd = [
      float(seg["vx"]),
      float(seg["vy"]),
      float(seg["wz"]),
    ]

    for _ in range(num_steps):
      step_commands.append(cmd)

  return step_commands

def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  if args.all_episodes:
    if args.command_file is None:
      raise ValueError("--command-file is required for --all-episodes")
    command_data = load_command_json(args.command_file)
    episodes = command_data["episodes"]
    if isinstance(episodes, dict):
      episode_names = list(episodes.keys())
    else:
      episode_names = [ep["name"] for ep in episodes]
    for ep_name in episode_names:
      cfg_dict = asdict(args)
      cfg_dict['episode_name'] = ep_name
      cfg_ep = PlayConfig(**cfg_dict)
      run_play(chosen_task, cfg_ep)
  else:
    run_play(chosen_task, args)


if __name__ == "__main__":
  main()

"""
python scripts/eval_fixed_cmd.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/baseline_logs/model_4000.pt \
  --model-tag baseline \
  --command-file tmp/go2_eval_cmd_set_v1.json \
  --episode-name step_vx_06

"""#  baseline模型flat地形

"""
python scripts/eval_fixed_cmd.py Unitree-Go2-Slippery \
  --checkpoint-file logs/rsl_rl/go2_velocity/baseline_logs/model_4000.pt \
  --model-tag baseline \
  --command-file tmp/go2_eval_cmd_set_v1.json \
  --episode-name step_vx_06

""" #baseline 模型，在 slippery 地形上测

"""
python scripts/eval_fixed_cmd.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/slippery/model_4000.pt \
  --model-tag slippery \
  --command-file tmp/go2_eval_cmd_set_v1.json \
  --episode-name step_vx_06

""" #slippery 模型，在 flat 地形上测

"""
python scripts/eval_fixed_cmd.py Unitree-Go2-Slippery \
  --checkpoint-file logs/rsl_rl/go2_velocity/slippery1.0/model_4800.pt \
  --model-tag slippery1.0 \
  --command-file tmp/go2_eval_cmd_set_v1.json \
  --episode-name step_lwz_rwz \
  --video True \
  --no-terminations True

"""#slippery 模型，在 slippery 地形上测

"""
python scripts/eval_fixed_cmd.py Unitree-Go2-Slippery \
  --checkpoint-file logs/rsl_rl/go2_velocity/baseline_logs/model_4800.pt \
  --model-tag baseline \
  --command-file tmp/go2_eval_cmd_set_v1.json \
  --episode-name step_lwz_rwz \
  --video True \
  --no-terminations True

"""

"""
python scripts/eval_fixed_cmd.py Unitree-Go2-Slippery \
  --checkpoint-file logs/rsl_rl/go2_velocity/slippery1.0/model_4800.pt \
  --model-tag slippery1.0 \
  --command-file tmp/go2_eval_cmd_set_v1.json \
  --all-episodes True \
  --video True \
  --no-terminations True
"""

"""
python scripts/eval_fixed_cmd.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/baseline_logs/model_4800.pt \
  --model-tag baseline \
  --command-file tmp/go2_eval_cmd_set_v1.json \
  --all-episodes True \
  --video True \
  --no-terminations True
"""

# 你可以在命令行里指定不同的 episode_name 来跑不同的 episode，或者直接 --all-episodes 跑所有 episode。
"""
python scripts/eval_fixed_cmd.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/slippery1.0/model_4800.pt \
  --model-tag slippery1.0 \
  --command-file tmp/go2_eval_cmd_set_v1.json \
  --all-episodes True \
  --video True \
  --no-terminations True
"""    