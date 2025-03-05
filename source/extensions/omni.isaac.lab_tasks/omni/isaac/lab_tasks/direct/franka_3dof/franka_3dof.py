# Custom DirectRL Environment for Pushing Obstacles to a Target Area

from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.core.utils.stage import get_current_stage # type: ignore # get_current_stage is in this module
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector # type: ignore
from pxr import UsdGeom
import time

from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_mul, sample_uniform, subtract_frame_transforms, matrix_from_quat, quat_from_matrix
from .franka_3dof_cfg import Franka3dofCfg


class Franka3dof(DirectRLEnv):
    """RL Environment where the action space is the end-effector pose (position + orientation)."""

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Franka3dofCfg
    
    def __init__(self, cfg: Franka3dofCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

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

        # Initialize target positions and obstacles
        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_pos[:,:] = torch.tensor(cfg.spawn_pose, dtype=torch.float, device=self.device) + self.scene.env_origins
        self.spawn_area = torch.tensor(cfg.spawn_area, dtype=torch.float, device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_pos[:,:] = torch.tensor(cfg.target_pose, dtype=torch.float, device=self.device) + self.scene.env_origins
        self.target_area = torch.tensor(cfg.target_area, dtype=torch.float, device=self.device)
        self.randomize_target = True
        self.spawn_distance = torch.norm(self.object_pos - self.target_pos, p=2, dim=-1)
        # self.action_scale = cfg.action_scale.clone().detach().to(device=self.device)

        # self.obstacle_positions = define_origins(num_origins=30, spacing=0.02, offset=[0.4, 0.1, 1.1])

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Create auxiliary variables for computing applied action, observations, and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        #unit quaternion
        self.unit_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=self.device)

        # Set up grasp and push targets for pushing obstacles
        stage = get_current_stage()
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

        # End effector pose and orientation in the robot's local frame
        self.ee_position = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device) # Position (x, y, z)
        self.ee_orientation = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device) # Quaternion (w, x, y, z)
        # self.ee_orientation[:, 0] = 1.0  # Initialize as identity quaternion
        
        ##########################################################################
        # This creates the rotation for the EE to always point towards the table #
        ##########################################################################
        temp_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device)
        temp_rot = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=self.device)
        temp_quat_from_rot = quat_from_matrix(temp_rot)
        new_quat = quat_mul(temp_quat, temp_quat_from_rot)
        self.init_ee_orientation = self.ee_orientation.clone()
        self.init_ee_orientation[:,:] = new_quat

        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_grasp_pose_pos = tf_combine(
             hand_pose_inv_rot, hand_pose_inv_pos, pusher_pose[3:7], pusher_pose[0:3]
        )
        robot_local_grasp_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_grasp_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]


        # Define gripper axes in the robot's local frame repeated for all environments
        self.hand_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        # Offset in the local z-axis
        self.offset = torch.tensor([0.0, 0.0, 0.272], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        # Initialize the goal rotation
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        
        # Markers
        # self.object_goal_marker = VisualizationMarkers(self.cfg.goal_object_cfg)
        self.ee_marker = VisualizationMarkers(self.cfg.ee_pose_marker_cfg)
        self.ee_goal_marker = VisualizationMarkers(self.cfg.ee_goal_marker_cfg)
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # self.goal_distances = torch.norm(self.object_pos - self.target_pos, dim=-1).unsqueeze(-1).to(self.device)
        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_grasp_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.swap_goal = torch.rand((self.num_envs, 1), device=self.device)
        # Set everything over 0.5 to 1, and everything below to -1
        self.swap_goal = torch.where(self.swap_goal > 0.5, 1, -1)

        # Specify robot-specific parameters (for the differential IK controller)
        if self.cfg.robot_name == "franka_panda":
            self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        elif self.cfg.robot_name == "ur10":
            self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])

        else:
            raise ValueError(f"Robot {self.cfg.robot} is not supported. Valid: franka_panda, ur10")

        self.robot_entity_cfg.resolve(self.scene)

        if self._robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]


    def _setup_scene(self):
        """Set up the scene, including loading the Franka arm and other relevant objects."""
        self._robot = Articulation(self.cfg.robot_high_pd) # GRAVITY DISABLED - diff_ik only works with gravity disabled
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

        # Create controller
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        self.ik_commands = torch.zeros(self.num_envs, self.diff_ik_controller.action_dim, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Actions are in the order [dx, dy, dz]

        # Clamp and scale the actions
        self.actions = actions.clone().clamp(-1.0, 1.0)

        delta_position = self.actions[:, :3] * self.cfg.action_scale * self.dt

        # Update the end effector position and orientation
        self.ee_position += delta_position
        self.ee_position[:, 0] = torch.clamp(self.ee_position[:, 0], -0.1, 0.75)
        self.ee_position[:, 1] = torch.clamp(self.ee_position[:, 1], -0.4, 0.4)
        self.ee_position[:, 2] = torch.clamp(self.ee_position[:, 2], 0.16, 1.0) 
        # self.ee_orientation = quat_mul(self.ee_orientation, delta_orientation) #(Hamilton product)
        
        # Set the updated pose as the IK command
        # self.ik_commands = torch.cat((self.ee_position, delta_orientation), dim=-1)
        self.ik_commands = torch.cat((self.ee_position, self.init_ee_orientation), dim=-1)
        # Set the IK command for the differential IK controller
        self.diff_ik_controller.set_command(self.ik_commands)

        # visualize the goal pose
        self.ik_commands[:, 2] += 1.05 #add 1.05 to the z value as robot is lifter by 1.05
        self.ee_goal_marker.visualize(self.ik_commands[:, 0:3] + self.scene.env_origins, self.ik_commands[:, 3:7])

    def _apply_action(self):
        """Apply actions to the simulator."""
        # Obtain quantities from simulation
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = self._robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self._robot.data.root_state_w[:, 0:7]
        joint_pos = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        
        # Compute frame in root frame
        self.ee_position, self.ee_orientation = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        # Compute the joint commands using the differential IK controller
        joint_pos_des = self.diff_ik_controller.compute(self.ee_position, self.ee_orientation, jacobian, joint_pos)
        
        # Apply the joint position targets to the robot
        self._robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)

        # Update the end effector marker position and orientation
        ee_pos = ee_pose_w[:, 0:3]
        ee_quat = ee_pose_w[:, 3:7]
        self.ee_marker.visualize(ee_pos, ee_quat)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine if environments are done."""
        # TODO: add conditions for done regarding the object and target positions
        # Done if timeout occurs
        terminated = 0# self.object_pos[:, 2] < 1.05
        timeouts = self.episode_length_buf >= self.max_episode_length -1
        
        return terminated, timeouts

    def _get_rewards(self) -> torch.Tensor:
        """Compute and return rewards for each environment."""
         # Refresh the intermediate values after the physics step
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.object_pos,
            self.target_pos,
            self.robot_grasp_pos,
            self.spawn_distance,
            self.cfg.dist_reward_scale,
            self.cfg.target_reward_scale,
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

        # EE and arm root pose in world frame
        ee_pose_w = self._robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self._robot.data.root_state_w[:, 0:7]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # End-effector pose in robot's local frame
        ee_pose = torch.cat((ee_pos_b, ee_quat_b), dim=-1)
        # Set the initial IK commands to be the current end-effector pose
        self.ik_commands = ee_pose
        self.diff_ik_controller.reset() # Reset the differential IK controller       
        self.diff_ik_controller.set_command(self.ik_commands) # Set the initial IK commands

        # reset goal position
        rand_factor = torch.rand(len(env_ids), 3, device=self.device)
        self.target_pos[env_ids] = self.target_area[0] + rand_factor * (self.target_area[1] - self.target_area[0])
        self.target_pos[env_ids,1] *= self.swap_goal.squeeze(-1)[env_ids]
        self.target_pos[env_ids] = self.target_pos[env_ids] + self.scene.env_origins[env_ids]

        # update goal pose and markers
        goal_pos = self.target_pos[env_ids] #+ self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)
        self.reset_goal_buf[env_ids] = 0

        # reset object
        object_default_state = self._object.data.default_root_state.clone()[env_ids]
        rand_factor = torch.rand(len(env_ids), 3, device=self.device)
        temp_pos = self.spawn_area[0] + rand_factor * (self.spawn_area[1] - self.spawn_area[0])
        temp_pos[env_ids, 1] *= self.swap_goal.squeeze(-1)[env_ids]
        object_default_state[env_ids, :3] = temp_pos + self.scene.env_origins[env_ids]
        object_default_state[env_ids, 3:7] = self.unit_quat
        object_default_state[env_ids, 7:] = torch.zeros_like(self._object.data.default_root_state[env_ids, 7:])
        self._object.write_root_state_to_sim(object_default_state, env_ids=env_ids)

        # Refresh intermediate values for observations and rewards
        self._compute_intermediate_values(env_ids)

        # span distance between object and target
        self.spawn_distance = torch.norm(self.object_pos - self.target_pos, dim=-1).to(self.device)

    def _get_observations(self):
        """Return the observations for the environment."""
        # OBSERVATIONS:
        # 1. End-effector pose (position + orientation)
        # 2. Pusher tip position
        # 3. Object position
        # 4. Distance between object and target
        # 5. Distance between pusher and object
        
        # end-effector pose in world frame, shape (num_envs, 7)
        ee_pose = self._robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        self.robot_grasp_pose[:, :3] = self.robot_grasp_pos
        self.robot_grasp_pose[:, 3:] = self.robot_grasp_rot

        # end-effector pose in robot's local frame, shape (num_envs, 7)
        object_to_target_dists = self.target_pos - self.object_pos
        pusher_to_object_dists = self.object_pos - self.robot_grasp_pos

        observations = torch.cat(
            (
                ee_pose,                # [x, y, z, qw, qx, qy, qz]     world frame
                self.robot_grasp_pose,  # [x, y, z, qw, qx, qy, qz]     world frame
                self.object_pos,        # [x, y, z]                     world frame
                object_to_target_dists, # [x, y, z]                     world frame
                pusher_to_object_dists, # [x, y, z]                     world frame
            ),
            dim=-1,
        )

        return {"policy": observations} #{"policy": torch.clamp(observations, -5.0, 5.0)}
    
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Get the hand's global position and rotation
        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        # data for object
        self.object_pos = self._object.data.root_pos_w[env_ids] # - self.scene.env_origins (want it in global frame)

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
        object_pos,
        target_pos,
        robot_grasp_pos,
        spawn_distance,
        dist_reward_scale,
        target_reward_scale,
    ):
        ### Distance from end-effector to objects reward
        object_pusher_dist_reward = torch.norm(object_pos - robot_grasp_pos, p=2, dim=-1) # Tanh normalized distance reward

        # Tanh normalized distance reward
        goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)#.unsqueeze(-1)

        norm_goal_dist = 1 - torch.div(goal_dist, spawn_distance)#.squeeze(-1)# goal_dist / spawn_distance

        # Total reward
        rewards = (
            - dist_reward_scale * object_pusher_dist_reward
            + target_reward_scale * norm_goal_dist
        )

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
