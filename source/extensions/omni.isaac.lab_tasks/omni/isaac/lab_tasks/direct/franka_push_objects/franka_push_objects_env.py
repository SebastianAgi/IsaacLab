# Custom DirectRL Environment for Pushing Obstacles to a Target Area

from __future__ import annotations

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
# from omni.isaac.utils.torch_utils import get_current_stage
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector


def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
                """Compute pose in env-local coordinates."""
                world_transform = xformable.ComputeLocalToWorldTransform(0)
                world_pos = world_transform.ExtractTranslation()
                world_quat = world_transform.ExtractRotationQuat()

                px = world_pos[0] - env_pos[0]
                py = world_pos[1] - env_pos[1]
                pz = world_pos[2] - env_pos[2]
                qx = world_quat.imaginary[0]
                qy = world_quat.imaginary[1]
                qz = world_quat.imaginary[2]
                qw = world_quat.real

                return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )

@configclass
class PushObstaclesEnvCfg(DirectRLEnvCfg):
    # Environment configuration
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    num_actions = 7
    num_observations = 27
    num_states = 0

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
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
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
    )

    # Table configuration
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sebastian/IsaacLab/Thor_table_usd/thor_table.usd", #Granular_test_bed.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
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

    # Obstacles configuration
    num_objects = 1

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

    # Spawn area configuration
    spawn_pose = torch.tensor([0.375, 0.19, 1.07])
    spawn_area = torch.tensor([[0.125, 0.05, 1.07], [0.625, 0.325, 1.07]])

    # Target area configuration
    target_pose = torch.tensor([0.375, -0.19, 1.07])
    target_area = torch.tensor([[0.1, 0.0, 1.07], [0.65, -0.37, 1.07]]) # smallest corner and largest corner

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # Reward scales
    dist_reward_scale = 2.0
    action_penalty_scale = 0.05
    target_reward_scale = 10.0
    alignment_reward_scale = 1.5


class PushObstaclesEnv(DirectRLEnv):
    cfg: PushObstaclesEnvCfg
    
    def __init__(self, cfg: PushObstaclesEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize target positions and obstacles
        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_pos[:,:] = cfg.spawn_pose
        self.spawn_area = cfg.spawn_area.clone().detach().to(device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_pos[:,:] = cfg.target_pose
        self.target_area = cfg.target_area.clone().detach().to(device=self.device)
        self.randomize_target = True

        # self.obstacle_positions = define_origins(num_origins=30, spacing=0.02, offset=[0.4, 0.1, 1.1])

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Create auxiliary variables for computing applied action, observations, and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Set up grasp and push targets for pushing obstacles
        stage = stage_utils.get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )

        # The pose of the tip of the end-effector pusher center
        pusher_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_fingertip_centered_point")),
            self.device,
        )

        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_grasp_pose_pos = tf_combine(
             hand_pose_inv_rot, hand_pose_inv_pos, pusher_pose[3:7], pusher_pose[0:3]
        )
        robot_local_grasp_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_grasp_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        # self.pusher_link_idx = self._robot.find_bodies("panda_fingertip_centered_point")[0][0]
        # self.object_link_idx = self._object.find_bodies("root")[0][0]

        # Define gripper axes in the robot's local frame repeated for all environments
        self.hand_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        # Offset in the local z-axis
        self.offset = torch.tensor([0.0, 0.0, 0.272], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        # Initialize the goal rotation
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        self.goal_distances = torch.norm(self.object_pos - self.target_pos, dim=-1).unsqueeze(-1).to(self.device)
        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.swap_goal = torch.rand((self.num_envs, 1), device=self.device)
        # Set everything over 0.5 to 1, and everything below to -1
        self.swap_goal = torch.where(self.swap_goal > 0.5, 1, -1)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._table = RigidObject(self.cfg.table_cfg)
        self._object = RigidObject(self.cfg.object_cfg)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["object"] = self._object

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone, filter, and replicate the environment
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check if all obstacles are within a certain distance from the target position
        distances = torch.norm(self.object_pos - self.target_pos, dim=-1)
        terminated = torch.all(distances < 0.05, dim=-1)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics step
        self._compute_intermediate_values()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        # print('action shape:', self.actions.shape)
        # print('env 0 actions:', self.actions[0])
        
        return self._compute_rewards(
            self.actions,
            self.pusher_pos,
            self.object_pos,
            self.target_pos,
            self.robot_grasp_rot,
            self.robot_grasp_pos,
            self.hand_forward_axis,
            self.num_envs,
            self.goal_distances,
            self.cfg.dist_reward_scale,
            self.cfg.alignment_reward_scale,
            self.cfg.target_reward_scale,
            self.cfg.action_penalty_scale,
        )
        

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset the robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.05,
            0.05,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_state = self._object.data.default_root_state.clone()[env_ids]
        rand_factor = torch.rand(len(env_ids), 3, device=self.device)
        temp_pos = self.spawn_area[0] + rand_factor * (self.spawn_area[1] - self.spawn_area[0])
        temp_pos[:, 1] *= self.swap_goal.squeeze(-1)
        object_default_state[:, :3] = temp_pos + self.scene.env_origins[env_ids]
        
        # rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        # object_default_state[:, 3:7] = randomize_rotation(
        #     rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )
        object_default_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        object_default_state[:, 7:] = torch.zeros_like(self._object.data.default_root_state[env_ids, 7:])
        self._object.write_root_state_to_sim(object_default_state, env_ids)

        # Refresh intermediate values for observations and rewards
        self._compute_intermediate_values(env_ids)

        # reset goal distances
        self.goal_distances = torch.norm(self.object_pos - self.target_pos, dim=-1).unsqueeze(-1)

        # Reset episode timers and success flags
        self.episode_length_buf[env_ids] = 0
        self.successes[env_ids] = 0

        self.swap_goal *= -1

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_factor = torch.rand(len(env_ids), 3, device=self.device)
        self.target_pos[env_ids] = self.target_area[0] + rand_factor * (self.target_area[1] - self.target_area[0])

        self.target_pos[env_ids, 1] *= self.swap_goal.squeeze(-1)

        # update goal pose and markers
        goal_pos = self.target_pos[env_ids] + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

        self.reset_goal_buf[env_ids] = 0

    def _get_observations(self) -> dict:
        # Observation consists of robot joint positions, velocities, and distance to obstacles
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        object_to_target_dists = self.target_pos - self.object_pos
        # pusher_to_object_dists = self.object_pos - self.pusher_pos

        obs = torch.cat(
            (
                dof_pos_scaled, 
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale, 
                object_to_target_dists,
                self.object_pos,
                # self.object_rot,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # Auxiliary methods
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Get the hand's global position and rotation
        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        # Rotate the offset into the world frame using the hand's rotation
        offset_world = tf_vector(hand_rot, self.offset)

        # Compute the pusher_tip position
        self.pusher_pos = hand_pos + offset_world - self.scene.env_origins

        # data for object
        self.object_pos = self._object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self._object.data.root_quat_w
        self.object_velocities = self._object.data.root_vel_w
        self.object_linvel = self._object.data.root_lin_vel_w
        self.object_angvel = self._object.data.root_ang_vel_w

        # data for robot
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
        )

    def _compute_rewards(
        self,
        actions,
        pusher_pos,
        objects_pos,
        target_pos,
        robot_grasp_rot,
        robot_grasp_pos,
        hand_forward_axis,
        num_envs,
        goal_distances,
        dist_reward_scale,
        alignment_reward_scale,
        target_reward_scale,
        action_penalty_scale,
    ):
        ### Distance from end-effector to objects
        # Tanh normalized distance reward
        d = torch.norm(objects_pos - pusher_pos, p=2, dim=-1)
        object_pusher_dist_reward = 1 - torch.tanh(d / 0.1)

        # Tanh normalized distance reward
        d_target = torch.norm(target_pos - objects_pos, p=2, dim=-1)
        target_dist_reward = 1 - torch.tanh((d_target) / 0.1)

        # Compute alignment (cosine similarity)
        gripper_forward_axis_world = tf_vector(robot_grasp_rot, hand_forward_axis)

        # Define the target direction (negative z-axis)
        target_direction = torch.tensor([0, 0, -1], device=self.device)

        # Compute the cosine similarity (dot product)
        dot = torch.sum(gripper_forward_axis_world * target_direction, dim=-1)
        # dot = torch.bmm(gripper_forward_axis_world.view(num_envs, 1, 3), target_direction.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

        # Compute alignment reward
        alignment = torch.sign(dot) * dot**2



        # Regularization on the actions
        action_penalty = torch.sum(actions**2, dim=-1)

        # Total reward
        rewards = (
            - dist_reward_scale * object_pusher_dist_reward
            + target_reward_scale * target_dist_reward
            + alignment_reward_scale * alignment
            - action_penalty_scale * action_penalty
        )
        # print(
        #     f'pusher_pos env0:                  {pusher_pos[0]} \n'
        #     f'object_pos env0:                  {objects_pos[0]} \n'
        #     f'reward object_pusher_dist_reward: {- dist_reward_scale * object_pusher_dist_reward[0]} \n'
        #     f'reward target_dist_reward:        {target_reward_scale * target_dist_reward[0]} \n'
        #     f'reward alignment:                 {alignment_reward_scale * alignment[0]} \n'
        #     f'reward action_penalty:            {-action_penalty_scale * action_penalty[0]} \n'
        #     f'reward total env 0:               {rewards[0]} \n'
        #     f'---'
        # )

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos

