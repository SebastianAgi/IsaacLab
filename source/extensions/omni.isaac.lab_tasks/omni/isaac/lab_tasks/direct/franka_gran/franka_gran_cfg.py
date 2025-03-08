# Custom DirectRL Environment for Pushing Obstacles to a Target Area

from __future__ import annotations

import torch
import math
from pxr import UsdGeom, Usd

from omni.isaac.lab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
import omni.isaac.lab.sim as sim_utils

@configclass
class FrankaGranCfg(DirectRLEnvCfg):
    # Number of Granules
    num_grans = 10

    # Environment configuration
    episode_length_s = 12# 480 timesteps    timesteps = episode_length_s / (decimation * dt)
    decimation = 2
    action_space = 3
    state_space = 0
    # # Observation space for 3D environment
    # observation_space = 7 + (3*num_grans)
    # Observation space for 2D environment
    observation_space = 5 + (2*num_grans)


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
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**22,
            gpu_found_lost_pairs_capacity=2**22,
        ),
    )
    
    # Robot configuration
    robot_name = 'franka_panda' # 'franka_panda' or 'ur10' or 'widowx'

    # Observation space
    # observation_space = 7 + (3*num_grans) #23 #34

    # goal circle radius
    goal_diameter = 0.1
    goal_radius = goal_diameter / 2
    spawn_area_radius = 0.05

    # Spawn area configuration
    spawn_pose = [0.25, 0.19, 1.07]
    full_spawn_area = [[0.075, 0.0, 1.05], [0.725, 0.375, 1.05]] # smallest corner and largest corner

    # Target area configuration
    target_pose = [0.375, -goal_radius, 1.07]
    target_area = [[0.075, 0.0, 1.04], [0.725, -0.375, 1.04]] # smallest corner and largest corner

    # Create common spawn point for granules objects
    init_spawn_point = [0.375, goal_radius, 1.07]

    action_scale = 1.0 #7.5
    dof_velocity_scale = 0.1

    object_scale = 0.01

    # Reward scales
    dist_reward_scale = 5.0
    action_penalty_scale = 0.05
    target_reward_scale = 5.0
    alignment_reward_scale = 0.05

    # Valid spawn area
    valid_spawn_area = [list(full_spawn_area[0]), list(full_spawn_area[1])]
    valid_spawn_area[0][0] += goal_radius
    valid_spawn_area[0][1] += goal_radius
    valid_spawn_area[1][0] -= goal_radius
    valid_spawn_area[1][1] -= goal_radius

    # Valid target area
    valid_target_area = [list(target_area[0]), list(target_area[1])]
    valid_target_area[0][0] += goal_radius
    valid_target_area[0][1] -= goal_radius
    valid_target_area[1][0] -= goal_radius
    valid_target_area[1][1] += goal_radius



    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)
    
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
            # usd_path="/home/sebastian/IsaacLab/Thor_table_usd/thor_table.usd", #Stock Thors table,
            usd_path="/home/sebastian/IsaacLab/Thor_table_usd/Granular_test_bed.usd", #Granular test bed with walls,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0, density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )
        
    # Create a dictionary to hold the rigid objects
    rigid_objects = {}

    # Generate {num_gran} Cuboid rigid objects
    for i in range(num_grans):
        rigid_objects[f"object_{i}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Object_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(object_scale, object_scale, object_scale),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4, solver_velocity_iteration_count=0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.0025, density=500.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=init_spawn_point, rot=(1.0, 0.0, 0.0, 0.0)),
        )

    # Create the object collection
    object_collection = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

    # goal object (disk to represent the goal area)
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_area",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/disk.usd",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                scale=(goal_diameter, goal_diameter, 1.0),
            )
        },
    )

    # current granule mean marker
    mean_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/mean_object_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                scale=(0.5, 0.5, 0.5),
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

    tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=None,  # the camera is already spawned in the scene
        offset=TiledCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )