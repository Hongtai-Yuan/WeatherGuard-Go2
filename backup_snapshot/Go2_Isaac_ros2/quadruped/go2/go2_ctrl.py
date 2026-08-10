import os
import json
import torch
import carb
import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv
from quadruped.go2.go2_ctrl_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner
from isaaclab.actuators import DCMotorCfg
from quadruped.go2.mujoco_onnx_policy import DEFAULT_POLICY_PATH, get_mujoco_onnx_policy

base_vel_cmd_input = None

MUJOCO_ONNX_SYSID_PARAM_NAMES = (
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

MUJOCO_ONNX_DEFAULT_SYSID = {
    "action_scale": 0.25,
    "policy_command_scale": 1.0,
    "policy_phase_period": 0.30,
    "ground_friction": 0.6,
    "joint_damping_scale": 0.5480519630509263,
    "rotor_inertia_scale": 1.020747578431883,
    "joint_friction": 0.0,
    "actuator_stiffness_scale": 1.4287056034453514,
    "actuator_effort_scale": 1.2194133501475513,
    "body_inertia_scale": 0.8672280373732336,
    "base_com_shift_x": -0.00723974933837029,
    "base_com_shift_y": -0.016650156296088542,
    "base_com_shift_z": -0.0015925035921027387,
}


def _load_mujoco_onnx_sysid_params(path=None, overrides=None):
    params = dict(MUJOCO_ONNX_DEFAULT_SYSID)
    if path:
        with open(os.path.abspath(path), "r", encoding="utf-8") as file:
            loaded = json.load(file)
        params.update({name: float(loaded[name]) for name in MUJOCO_ONNX_SYSID_PARAM_NAMES if name in loaded})
    if overrides:
        params.update({name: float(value) for name, value in overrides.items() if value is not None})
    return params


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


def apply_mujoco_onnx_sysid_params(
    cfg,
    *,
    joint_damping_scale: float,
    rotor_inertia_scale: float,
    joint_friction: float,
    actuator_stiffness_scale: float,
    actuator_effort_scale: float,
) -> None:
    cfg.scene.unitree_go2.actuators = {
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=23.5 * actuator_effort_scale,
            saturation_effort=23.5 * actuator_effort_scale,
            velocity_limit=30.0,
            stiffness=20.0 * actuator_stiffness_scale,
            damping=1.0 * joint_damping_scale,
            armature=0.01 * rotor_inertia_scale,
            friction=joint_friction,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=23.5 * actuator_effort_scale,
            saturation_effort=23.5 * actuator_effort_scale,
            velocity_limit=30.0,
            stiffness=20.0 * actuator_stiffness_scale,
            damping=1.0 * joint_damping_scale,
            armature=0.01 * rotor_inertia_scale,
            friction=joint_friction,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=45.0 * actuator_effort_scale,
            saturation_effort=45.0 * actuator_effort_scale,
            velocity_limit=30.0,
            stiffness=40.0 * actuator_stiffness_scale,
            damping=2.0 * joint_damping_scale,
            armature=0.02 * rotor_inertia_scale,
            friction=joint_friction,
        ),
    }

# Initialize base_vel_cmd_input as a tensor when created
def init_base_vel_cmd(num_envs):
    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)

# Modify base_vel_cmd to use the tensor directly
def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    global base_vel_cmd_input
    return base_vel_cmd_input.clone().to(env.device)

# Update sub_keyboard_event to modify specific rows of the tensor based on key inputs
def sub_keyboard_event(event) -> bool:
    global base_vel_cmd_input

    lin_vel=1.0  # stable for 50 freq
    ang_vel=1.0  # stable for 50 freq
    
    if base_vel_cmd_input is not None:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:


#Default Keys----------------------------------------------------------------------------------------------------
            # Update tensor values for environment 0
            if event.input.name == 'W':
                base_vel_cmd_input[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
            elif event.input.name == 'S':
                base_vel_cmd_input[0] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
            elif event.input.name == 'A':
                base_vel_cmd_input[0] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
            elif event.input.name == 'D':
                base_vel_cmd_input[0] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
            elif event.input.name == 'Z':
                base_vel_cmd_input[0] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
            elif event.input.name == 'C':
                base_vel_cmd_input[0] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)

            
            # If there are multiple environments, handle inputs for env 1
            if base_vel_cmd_input.shape[0] > 1:
                if event.input.name == 'I':
                    base_vel_cmd_input[1] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
                elif event.input.name == 'K':
                    base_vel_cmd_input[1] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
                elif event.input.name == 'J':
                    base_vel_cmd_input[1] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
                elif event.input.name == 'L':
                    base_vel_cmd_input[1] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
                elif event.input.name == 'M':
                    base_vel_cmd_input[1] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
                elif event.input.name == '>':
                    base_vel_cmd_input[1] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
        
        # Reset commands to zero on key release
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            base_vel_cmd_input.zero_()
    return True

def get_rsl_flat_policy(cfg):
    cfg.observations.policy.height_scan = None
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy

def get_mujoco_onnx_policy_env(
    cfg,
    policy_path=None,
    sysid_params_json=None,
    sysid_overrides=None,
):
    params = _load_mujoco_onnx_sysid_params(sysid_params_json, sysid_overrides)
    cfg.observations.policy.height_scan = None
    cfg.actions.joint_pos.scale = params["action_scale"]
    cfg.scene.ground.spawn.physics_material.static_friction = params["ground_friction"]
    cfg.scene.ground.spawn.physics_material.dynamic_friction = params["ground_friction"]
    apply_mujoco_onnx_sysid_params(
        cfg,
        joint_damping_scale=params["joint_damping_scale"],
        rotor_inertia_scale=params["rotor_inertia_scale"],
        joint_friction=params["joint_friction"],
        actuator_stiffness_scale=params["actuator_stiffness_scale"],
        actuator_effort_scale=params["actuator_effort_scale"],
    )

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)
    policy = get_mujoco_onnx_policy(
        env,
        policy_path=policy_path or DEFAULT_POLICY_PATH,
        command_scale=params["policy_command_scale"],
        phase_period=params["policy_phase_period"],
    )
    apply_runtime_body_sysid(
        env,
        inertia_scale=params["body_inertia_scale"],
        base_com_shift=(
            params["base_com_shift_x"],
            params["base_com_shift_y"],
            params["base_com_shift_z"],
        ),
    )
    print(
        "Loaded MuJoCo ONNX Go2 locomotion "
        f"policy={policy_path or DEFAULT_POLICY_PATH} "
        f"friction={params['ground_friction']} "
        f"action_scale={params['action_scale']} "
        f"phase_period={params['policy_phase_period']}",
        flush=True,
    )
    return env, policy

def get_rsl_rough_policy(cfg):
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy
