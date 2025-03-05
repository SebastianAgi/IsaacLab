# Custom DirectRL Environment for Pushing Obstacles to a Target Area

from __future__ import annotations

import torch
from pxr import UsdGeom, Usd

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class Franka3dofCfg(DirectRLEnvCfg):
    # Environment configuration
    episode_length_s = 10  # 1000 timesteps
    decimation = 2
    action_space = 3
    observation_space = 23 #34
    state_space = 0

    # Simulation configuration
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Robot configuration
    robot_name = 'franka_panda' # 'franka_panda' or 'ur10' or 'widowx'
    # robot.is_fixed_base = True

    # Spawn area configuration
    spawn_pose = [0.375, 0.19, 1.07]
    spawn_area = [[0.125, 0.05, 1.07], [0.625, 0.325, 1.07]]

    # Target area configuration
    target_pose = [0.375, -0.19, 1.07]
    target_area = [[0.1, 0.0, 1.07], [0.65, -0.37, 1.07]] # smallest corner and largest corner

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # Reward scales
    dist_reward_scale = 0.5
    action_penalty_scale = 0.5 #0.05
    target_reward_scale = 20.0
    alignment_reward_scale = 0.05

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # Robot configuration
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sebastian/IsaacLab/frank_panda_usd/Granular_franka_2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -3.14/2,
                "panda_joint5": 0.0,
                "panda_joint6": 3.14/2,
                "panda_joint7": -3.14/4,
            },
            pos=(0.0, 0.0, 1.05),
            rot=(-1.0,0.0,0.0,0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # Higher PD gains for the robot cfg
    robot_high_pd = robot.copy()
    robot_high_pd.spawn.rigid_props.disable_gravity = True
    robot_high_pd.actuators["panda_shoulder"].stiffness = 400.0
    robot_high_pd.actuators["panda_shoulder"].damping = 80.0
    robot_high_pd.actuators["panda_forearm"].stiffness = 400.0
    robot_high_pd.actuators["panda_forearm"].damping = 80.0


    # Table configuration
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sebastian/IsaacLab/Thor_table_usd/thor_table.usd", #Granular_test_bed.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0, density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    # on-table object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/MultiColorCube/multi_color_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.375, 0.2, 1.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # object_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"/home/sebastian.IsaacLab/USD_files/Beans.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(density=400.0),
    #         scale=(1.0, 1.0, 1.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.375, 0.2, 1.1), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.2, 1.2, 1.2),
            )
        },
    )

    # Current end effector pose marker
    ee_pose_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ee_pose_marker",
        markers={
            "ee": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )
        },
    )

    # Wanted end effector pose marker
    ee_goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ee_goal_marker",
        markers={
            "ee": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )
        },
    )


    # usd_file_path = f"/home/sebastian.IsaacLab/USD_files/Beans.usd"                 # Path to your USD file
    # stage = Usd.Stage.Open(usd_file_path)                                           # Load the USD stage
    # particles_prim = stage.GetPrimAtPath("/World/Particles")                        # Get the Particles Xform
    # granules = [prim.GetPath().pathString for prim in particles_prim.GetChildren()] # Get child prims inside Particle
    # num_granules = len(granules)

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/Ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
