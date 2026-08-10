from isaaclab.scene import InteractiveSceneCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import DCMotorCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg
from isaacsim.core.utils.viewports import set_camera_view
import numpy as np
from scipy.spatial.transform import Rotation as R
import quadruped.go2.go2_ctrl as go2_ctrl


MUJOCO_GO2_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

MUJOCO_GO2_DEFAULT_JOINT_POS = [
    -0.1,
    0.9,
    -1.8,
    0.1,
    0.9,
    -1.8,
    -0.1,
    0.9,
    -1.8,
    0.1,
    0.9,
    -1.8,
]


def apply_mujoco_onnx_sysid_params(
    cfg,
    *,
    joint_damping_scale: float = 1.0,
    rotor_inertia_scale: float = 1.0,
    joint_friction: float = 0.0,
    actuator_stiffness_scale: float = 1.0,
    actuator_effort_scale: float = 1.0,
) -> None:
    """Apply low-dimensional sysid parameters to the MuJoCo-ONNX Go2 config.

    The course notes separate viscous damping b, Coulomb friction fc and rotor
    inertia Ir. In Isaac's DC motor config these map most directly to damping,
    friction and armature respectively.
    """

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


@configclass
class Go2SimCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.1, 0.1, 0.1),
            size=(300.0, 300.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.6,
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="average",
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0, 0, 1e-4)
        )
    )
    
    # lights
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/DomeLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    # )

    # Go2 Robot
    unitree_go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2")
    # Go2 foot contact sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Go2/.*_foot", history_length=3, track_air_time=True)

    # Go2 height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20)), 
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), 
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="unitree_go2",
        joint_names=MUJOCO_GO2_JOINT_NAMES,
        preserve_order=True,
        scale=0.25,
        offset=dict(zip(MUJOCO_GO2_JOINT_NAMES, MUJOCO_GO2_DEFAULT_JOINT_POS)),
        use_default_offset=False,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,
                               params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,
                               params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    params={"asset_cfg": SceneEntityCfg(name="unitree_go2")},
                                    noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
        # velocity command
        base_vel_cmd = ObsTerm(func=go2_ctrl.base_vel_cmd)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                            params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,
                            params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
        actions = ObsTerm(func=mdp.last_action)
        
        # Height scan
        height_scan = ObsTerm(func=mdp.height_scan,
                              params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                              clip=(-1.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_vel_cmd = mdp.UniformVelocityCommandCfg(
        asset_name="unitree_go2",
        resampling_time_range=(0.0, 0.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )

@configclass
class EventCfg:
    """Configuration for events."""
    pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass



@configclass
class Go2RSLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 environment."""
    # scene settings
    scene = Go2SimCfg(num_envs=2, env_spacing=2.0)

    # basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    
    # dummy settings
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        # viewer settings
        self.viewer.eye = [-4.0, 0.0, 5.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        # step settings
        self.decimation = 8  # step

        # simulation settings
        self.sim.dt = 0.005  # sim step every 
        self.sim.render_interval = self.decimation  
        self.sim.disable_contact_processing = True
        self.sim.render.antialiasing_mode = None
        # self.sim.physics_material = self.scene.terrain.physics_material

        # settings for rsl env control
        self.episode_length_s = 20.0 # can be ignored
        self.is_finite_horizon = False
        self.actions.joint_pos.scale = 0.25

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class Go2MujocoOnnxEnvCfg(Go2RSLEnvCfg):
    """Go2 play environment aligned with the MuJoCo-exported ONNX policy."""

    def __post_init__(self):
        super().__post_init__()

        # The MuJoCo policy was trained at 50 Hz: mujoco dt 0.005 * decimation 4.
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.actions.joint_pos.scale = 0.25

        # Match the MuJoCo/mjlab Go2 default pose stored in policy.onnx metadata.
        self.scene.unitree_go2.init_state.pos = (0.0, 0.0, 0.32)
        self.scene.unitree_go2.init_state.joint_pos = dict(
            zip(MUJOCO_GO2_JOINT_NAMES, MUJOCO_GO2_DEFAULT_JOINT_POS)
        )

        # Match the MuJoCo actuator gains more closely than Isaac's default Go2.
        apply_mujoco_onnx_sysid_params(self)

def camera_follow(env):
    # Remove conditional check and always use indexed naming
    robot_position = env.unwrapped.scene[f"unitree_go2"].data.root_state_w[0, :3].cpu().numpy()
    robot_orientation = env.unwrapped.scene[f"unitree_go2"].data.root_state_w[0, 3:7].cpu().numpy()
    rotation = R.from_quat([robot_orientation[1], robot_orientation[2], 
                            robot_orientation[3], robot_orientation[0]])
    yaw = rotation.as_euler('zyx')[0]
    yaw_rotation = R.from_euler('z', yaw).as_matrix()
    set_camera_view(
        yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position,
        robot_position
    )
