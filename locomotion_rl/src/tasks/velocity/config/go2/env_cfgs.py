"""Unitree Go2 velocity environment configurations."""

from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp


#from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg   #库里原始版本
from src.tasks.velocity.mdp import UniformVelocityCommandCfg    #你自己改过的版本

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]

SYSID_FAST_STABLE = {
  "joint_damping_scale": 0.5480519630509263,
  "rotor_inertia_scale": 1.020747578431883,
  "actuator_stiffness_scale": 1.4287056034453514,
  "actuator_effort_scale": 1.2194133501475513,
  "base_com_shift_x": -0.00723974933837029,
  "base_com_shift_y": -0.016650156296088542,
  "base_com_shift_z": -0.0015925035921027387,
  "policy_phase_period": 0.30,
}


def _apply_sysid_centered_go2_actuators(cfg: ManagerBasedRlEnvCfg) -> None:
  """Center Go2 actuator parameters on the Isaac/MuJoCo sysid result."""
  robot_cfg = cfg.scene.entities["robot"]
  stiffness_scale = SYSID_FAST_STABLE["actuator_stiffness_scale"]
  effort_scale = SYSID_FAST_STABLE["actuator_effort_scale"]
  damping_scale = SYSID_FAST_STABLE["joint_damping_scale"]
  armature_scale = SYSID_FAST_STABLE["rotor_inertia_scale"]

  for actuator in robot_cfg.articulation.actuators:
    names = " ".join(actuator.target_names_expr)
    is_calf = "calf" in names
    base_stiffness = 40.0 if is_calf else 20.0
    base_damping = 2.0 if is_calf else 1.0
    base_effort = 45.0 if is_calf else 23.5
    base_armature = 0.02 if is_calf else 0.01

    actuator.stiffness = base_stiffness * stiffness_scale
    actuator.damping = base_damping * damping_scale
    actuator.effort_limit = base_effort * effort_scale
    actuator.armature = base_armature * armature_scale


def _apply_sysid_centered_go2_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  """Use sysid values as DR centers instead of broad blind randomization."""
  cfg.events["foot_friction"].params["ranges"] = (0.1, 1.0)
  cfg.events["base_com"].params["ranges"] = {
    0: (
      SYSID_FAST_STABLE["base_com_shift_x"] - 0.015,
      SYSID_FAST_STABLE["base_com_shift_x"] + 0.015,
    ),
    1: (
      SYSID_FAST_STABLE["base_com_shift_y"] - 0.015,
      SYSID_FAST_STABLE["base_com_shift_y"] + 0.015,
    ),
    2: (
      SYSID_FAST_STABLE["base_com_shift_z"] - 0.012,
      SYSID_FAST_STABLE["base_com_shift_z"] + 0.012,
    ),
  }
  cfg.observations["actor"].terms["phase"].params["period"] = SYSID_FAST_STABLE["policy_phase_period"]
  cfg.observations["critic"].terms["phase"].params["period"] = SYSID_FAST_STABLE["policy_phase_period"]
  cfg.rewards["foot_gait"].params["period"] = SYSID_FAST_STABLE["policy_phase_period"]

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.3, 1.6)
  twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)
  twist_cmd.ranges.ang_vel_z = (-1.2, 1.2)

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.3, 1.2), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
      {"step": 3000 * 24, "lin_vel_x": (-0.5, 1.6), "lin_vel_y": (-0.6, 0.6), "ang_vel_z": (-1.2, 1.2)},
      {"step": 7000 * 24, "lin_vel_x": (-0.8, 2.0), "lin_vel_y": (-0.8, 0.8), "ang_vel_z": (-1.5, 1.5)},
    ]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    
  return cfg

def unitree_go2_slippery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 slippery flat terrain velocity configuration."""
  # 先直接复用 flat 环境，保证任务结构、观测、奖励、动作都不变。
  cfg = unitree_go2_flat_env_cfg(play=play)

  # 这里后面专门放“slippery 地面”的最小修改。
  # 先只建立函数壳子，方便下一步继续加摩擦设置。

  cfg.events["foot_friction"].params["ranges"] = (0.2, 0.2)

   # 调试打印：确认 slippery 环境里实际使用的摩擦范围。
  print("[DEBUG] slippery foot_friction ranges:", cfg.events["foot_friction"].params["ranges"])

  return cfg

def unitree_go2_slippery_train_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 low-friction training configuration."""
  # 先复用 flat 环境，保持任务结构不变。
  cfg = unitree_go2_flat_env_cfg(play=play)

  _apply_sysid_centered_go2_actuators(cfg)
  _apply_sysid_centered_go2_dr(cfg)

  # 调试打印：确认训练环境里实际使用的脚部摩擦范围。
  print("[DEBUG] slippery-train foot_friction ranges:", cfg.events["foot_friction"].params["ranges"])
  print("[DEBUG] slippery-train base_com ranges:", cfg.events["base_com"].params["ranges"])
  print("[DEBUG] slippery-train phase period:", cfg.observations["actor"].terms["phase"].params["period"])

  return cfg
