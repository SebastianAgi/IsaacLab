# Custom DirectRL Environment for Pushing Obstacles to a Target Area

from __future__ import annotations

import torch
import math

from omni.isaac.lab.assets.rigid_object_collection.rigid_object_collection import RigidObjectCollection
import omni.isaac.lab.sim as sim_utils
from omni.isaac.core.utils.stage import get_current_stage # type: ignore # get_current_stage is in this module
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector # type: ignore
from pxr import UsdGeom
import tqdm
import wandb

from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_mul, sample_uniform, subtract_frame_transforms, matrix_from_quat
from .franka_gran_cfg import FrankaGranCfg


class FrankaGranIK(DirectRLEnv):
    """RL Environment where the action space is the end-effector pose (position + orientation)."""

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    @torch.jit.script
    def quaternion_to_yaw(quat):
        """Convert a quaternion to a yaw angle (rotation around the z-axis)."""
        # Extract the components of the quaternion
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Compute the yaw angle
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        return yaw

    @torch.jit.script
    def get_spawn_points_hemisphere_batch(
        centers: torch.Tensor,
        spawn_area_radius: float,
        sphere_radius: float,
        n: int,
        max_attempts: int = 10000
    ) -> torch.Tensor:
        """
        Generate n non-overlapping spawn points in a hemispherical area for each center in a batch,
        using only tensor operations for the batch dimension.

        Args:
            centers (Tensor): A tensor of shape (m, 3) where m is the number of centers.
            spawn_area_radius (float): The maximum radial distance from each center.
            sphere_radius (float): The radius of each object.
            n (int): The number of spawn points to generate per center.
            max_attempts (int): Maximum candidate generation attempts.

        Returns:
            Tensor: An (m, n, 3) tensor where each (n, 3) slice contains spawn points for one center.

        Raises:
            RuntimeError: If fewer than n valid points can be generated for some centers.
        """
        m = centers.size(0)  # number of centers
        accepted = torch.empty((m, n, 3), dtype=torch.float32, device=centers.device)
        # count[i] will hold the number of spawn points accepted so far for center i.
        count = torch.zeros((m,), dtype=torch.int32, device=centers.device)
        attempts = 0
        two_radius = 2 * sphere_radius
        spawn_height = 2 * spawn_area_radius

        # Continue until every center has n points or we exceed max_attempts.
        while (count < n).any() and attempts < max_attempts:
            # Generate one candidate for each center in parallel.
            # Sample uniformly from a circular disk in the xy-plane:
            r = spawn_area_radius * torch.sqrt(torch.rand(m, device=centers.device))
            theta = 2 * math.pi * torch.rand(m, device=centers.device)
            x_offset = r * torch.cos(theta)
            y_offset = r * torch.sin(theta)

            # Sample z uniformly over the desired height.
            # For example, if spawn_height is a new parameter that defines the vertical span:
            z_offset = torch.rand(m, device=centers.device) * spawn_height

            # Form the candidate points:
            candidate = centers + torch.stack([x_offset, y_offset, z_offset], dim=1)  # shape (m, 3)

            # Only consider centers that still need points.
            still_need = (count < n)

            # For each center, we must check the candidate against all accepted spawn points so far.
            # We'll compare candidate (m,3) against accepted (m,n,3) as follows:
            #   1. Expand candidate to (m, n, 3)
            #   2. Compute Euclidean distances between candidate and each accepted point.
            candidate_exp = candidate.unsqueeze(1).expand(m, n, 3)  # shape (m, n, 3)
            diff = candidate_exp - accepted  # shape (m, n, 3)
            dists = torch.sqrt(torch.sum(diff * diff, dim=2))  # shape (m, n)

            # Because each center i has only count[i] valid accepted points (stored in accepted[i, 0:count[i]]),
            # we create a mask to “ignore” unused rows. For each center i, let j be an index:
            indices = torch.arange(n, device=centers.device).unsqueeze(0).expand(m, n)  # (m, n)
            valid_mask = indices < count.unsqueeze(1)  # (m, n) mask: True for accepted indices.

            # For j not yet used (i.e. j >= count[i]), set the distance to a large number so that they don't affect rejection.
            dists_masked = torch.where(valid_mask, dists, torch.full_like(dists, 1e9))

            # A candidate is valid for center i if:
            #   - No accepted point (if any) is closer than 2*sphere_radius.
            #   - (Or if count[i]==0, the candidate is automatically valid.)
            min_dists, _ = torch.min(dists_masked, dim=1)  # (m,)
            candidate_valid = (count == 0) | (min_dists >= two_radius)
            # Also require that the center still needs more points.
            candidate_valid = candidate_valid & still_need

            # For all centers where candidate_valid is True, update the accepted points.
            # Use advanced indexing to update accepted[i, count[i]] = candidate[i] for each valid center.
            valid_centers = torch.nonzero(candidate_valid).squeeze(1)
            if valid_centers.numel() > 0:
                # For these centers, the column to update is given by count[valid_centers].
                col_indices = count.index_select(0, valid_centers).to(torch.int64)
                accepted[valid_centers, col_indices] = candidate.index_select(0, valid_centers)
                # Increment count for these centers.
                count.index_put_((valid_centers,), count.index_select(0, valid_centers) + 1)
            attempts += 1

        if (count < n).any():
            raise RuntimeError("Could not generate enough non-overlapping spawn points for some centers.")
        return accepted

    cfg: FrankaGranCfg
    
    def __init__(self, cfg: FrankaGranCfg, render_mode: str | None = None, **kwargs):
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
        self.init_spawn_point = torch.tensor(cfg.init_spawn_point, dtype=torch.float, device=self.device)
        self.objects_pos = torch.zeros((self.num_envs, self.cfg.num_grans, 3), dtype=torch.float, device=self.device)
        self.objects_pos[:,:,:] = self.get_spawn_points_hemisphere_batch(
            self.init_spawn_point.repeat(self.num_envs, 1), 
            spawn_area_radius=0.1, 
            sphere_radius=(self.cfg.object_scale*math.sqrt(3))/2, 
            n=self.cfg.num_grans
        )
        self.spawn_area = torch.tensor(cfg.valid_spawn_area, dtype=torch.float, device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_pos[:,:] = torch.tensor(cfg.target_pose, dtype=torch.float, device=self.device)
        self.target_area = torch.tensor(cfg.valid_target_area, dtype=torch.float, device=self.device)
        self.randomize_target = True
        self.objects_to_target = self.target_pos - torch.tensor(cfg.spawn_pose, dtype=torch.float, device=self.device)
        self.spawn_distance = torch.norm(self.objects_to_target - self.target_pos, p=2, dim=-1)

        self.objects_pos_mean = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Create auxiliary variables for computing applied action, observations, and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        #unit quaternion
        self.unit_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=self.device)
        self.all_env_unit_quat = self.unit_quat.repeat((self.num_envs, 1))
        #unit lin vel & ang vel
        self.unit_vel = torch.zeros(6, dtype=torch.float, device=self.device)
        # Zero tensor
        self.zero_tensor = torch.tensor(0.0, device=self.device)
        # 0.1 tensor
        self.point1_tensor = torch.tensor(0.1, device=self.device)
        # wandb counter
        self.global_step = torch.tensor(0, device=self.device)
        # Default objects state shape (num_instances, num_objects, 13)
        self.deafult_objects_state = torch.zeros((self.num_envs, self.cfg.num_grans, 13), dtype=torch.float, device=self.device)

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
        self.ee_position[:, :3] = hand_pose[0:3]
        self.ee_orientation = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device) # Quaternion (w, x, y, z)
        self.ee_orientation[:, :] = hand_pose[3:]

        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_grasp_pose_pos = tf_combine(
             hand_pose_inv_rot, hand_pose_inv_pos, pusher_pose[3:7], pusher_pose[0:3]
        )
        robot_local_grasp_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_grasp_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]

        # Initialize the goal rotation
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0

        # Create initial object poses
        self.objects_state = torch.zeros((self.num_envs, self.cfg.num_grans, 13), dtype=torch.float, device=self.device)
        self.objects_state[:, :, :3] = self.objects_pos
        self.objects_state[:, :, 3:7] = self.unit_quat   

        self.scale_tensor = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.scale_tensor[:,:] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float, device=self.device)

        # Change the objects position to the spawn points
        self._object_collection.write_object_state_to_sim(self.objects_state)     
        
        # Markers
        self.ee_marker = VisualizationMarkers(self.cfg.ee_pose_marker_cfg)
        self.ee_goal_marker = VisualizationMarkers(self.cfg.ee_goal_marker_cfg)
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
        self.mean_marker = VisualizationMarkers(self.cfg.mean_object_cfg)

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_grasp_pose = torch.zeros((self.num_envs, 7), device=self.device)

        self.swap_goal = torch.rand((self.num_envs, 1), device=self.device)
        self.swap_goal = torch.where(self.swap_goal > 0.5, 1, -1) # Set everything over 0.5 to 1, and everything below to -1

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

        self.bap = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
        self.bapkus = torch.tensor([0.0, 0.001, 0.0, 0.0], device=self.device)


    def _setup_scene(self):
        """Set up the scene, including loading the Franka arm and other relevant objects."""
        self._robot = Articulation(self.cfg.robot_high_pd) # GRAVITY DISABLED - diff_ik only works with gravity disabled
        self._table = RigidObject(self.cfg.table_cfg)
        self._object_collection = RigidObjectCollection(self.cfg.object_collection)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_object_collections["object_collection"] = self._object_collection

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
        # Actions are in the order [dx, dy, dz, yaw_rad]

        # Clamp and scale the actions
        self.actions = actions.clone().clamp(-1.0, 1.0)

        delta_position = self.actions[:, :3] * self.cfg.action_scale * self.dt

        # Update the end effector position and orientation
        self.ee_position += delta_position
        self.ee_position[:, 0] = torch.clamp(self.ee_position[:, 0], -0.1, 0.75)
        self.ee_position[:, 1] = torch.clamp(self.ee_position[:, 1], -0.4, 0.4)
        self.ee_position[:, 2] = torch.clamp(self.ee_position[:, 2], 0.16, 0.4) 

        # Compute the new yaw angle
        delta_yaw = self.actions[:, 3] * self.cfg.action_scale * self.dt
        current_yaw = self.quaternion_to_yaw(self.ee_orientation)

        # Clamp the new yaw angle to the rotation limits
        clamped_yaw = torch.clamp(delta_yaw, self.robot_dof_lower_limits[6] + 0.785398 - current_yaw, self.robot_dof_upper_limits[6] + 0.785398 - current_yaw)

        # Convert the clamped yaw angle to a quaternion
        clamped_yaw_quat = torch.zeros((self.num_envs, 4), device=self.device)
        clamped_yaw_quat[:, 0] = torch.cos(clamped_yaw / 2)
        clamped_yaw_quat[:, 3] = torch.sin(clamped_yaw / 2)

        # Update the end effector orientation
        self.ee_orientation = quat_mul(self.ee_orientation, clamped_yaw_quat)

        # self.ee_orientation[:, :] = torch.tensor([3.5739e-08, -9.4628e-01, -3.2336e-01, -8.1139e-08], device=self.device)

        # Set the updated pose as the IK command
        self.ik_commands = torch.cat((self.ee_position, self.ee_orientation), dim=-1)
        # Set the IK command for the differential IK controller
        self.diff_ik_controller.set_command(self.ik_commands)

        # visualize the goal pose
        self.ik_commands[:, 2] += 1.05 #add 1.05 to the z value as robot is lifter by 1.05
        self.ee_goal_marker.visualize(
            translations=self.ik_commands[:,:3] + self.scene.env_origins, 
            orientations=self.ik_commands[:,3:7], 
            scales= self.scale_tensor,
            marker_indices=None)
        
        # Visualize the mean object position
        self.mean_marker.visualize(self.objects_pos_mean, self.all_env_unit_quat)

    def _apply_action(self):
        """Apply actions to the simulator."""
        # Obtain quantities from simulation
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = self._robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self._robot.data.root_state_w[:, 0:7]
        joint_pos = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        
        # Compute frame in root frame
        self.ee_position, self.ee_orientation_controller = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        # Compute the joint commands using the differential IK controller
        joint_pos_des = self.diff_ik_controller.compute(self.ee_position, self.ee_orientation_controller, jacobian, joint_pos)
        
        # Apply the joint position targets to the robot
        self._robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)

        # Update the end effector marker position and orientation
        ee_pos = ee_pose_w[:, 0:3]
        ee_quat = ee_pose_w[:, 3:7]
        self.ee_marker.visualize(ee_pos, ee_quat)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine if environments are done."""

        self._compute_intermediate_values()

        # Done if timeout occurs
        terminated = (self.objects_pos[:,:, 2] < 0.5).any(dim=1)
        terminated2 = (torch.norm(self.objects_pos_mean, p=2,dim=-1) < 0.1).any()
        terminated = terminated | terminated2
        # terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timeouts = self.episode_length_buf >= self.max_episode_length -1

        # print(f'Timesout: {self.episode_length_buf} >= {self.max_episode_length -1}')
        
        return terminated, timeouts

    def _get_rewards(self) -> torch.Tensor:
        """Compute and return rewards for each environment."""
         # Refresh the intermediate values after the physics step
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.objects_pos,
            self.objects_pos_mean,
            self.target_pos,
            self.robot_grasp_pos,
            self.spawn_distance,
            self.actions,
            self.cfg.action_penalty_scale,
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
        ee_pose_w = self._robot.data.body_state_w[env_ids, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self._robot.data.root_state_w[env_ids, 0:7]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # End-effector pose in robot's local frame
        ee_pose = torch.cat((ee_pos_b, ee_quat_b), dim=-1)
        # Set the initial IK commands to be the current end-effector pose
        self.ik_commands[env_ids] = ee_pose
        self.diff_ik_controller.reset(env_ids=env_ids) # Reset the differential IK controller       
        self.diff_ik_controller.set_command(self.ik_commands) # Set the initial IK commands

        # reset goal position
        rand_factor = torch.rand(len(env_ids), 3, device=self.device)
        self.target_pos[env_ids] = self.target_area[0] + rand_factor * (self.target_area[1] - self.target_area[0])
        self.target_pos[env_ids,1] *= self.swap_goal.squeeze(-1)[env_ids]
        # self.target_pos[env_ids,2] = 1.05
        self.target_pos[env_ids] = self.target_pos[env_ids] + self.scene.env_origins[env_ids]

        # update goal pose and markers
        self.goal_markers.visualize(self.target_pos, self.goal_rot)
        self.reset_goal_buf[env_ids] = 0

        # reset object
        objects_new_state = self.deafult_objects_state[env_ids] # shape (num_envs, num_grans, 13)
        rand_factor = torch.rand(len(env_ids), 3, device=self.device)
        new_spawn_points = self.spawn_area[0] + rand_factor * (self.spawn_area[1] - self.spawn_area[0])
        objects_new_state[:,:,:3] = self.get_spawn_points_hemisphere_batch(
            centers=new_spawn_points, 
            spawn_area_radius=self.cfg.spawn_area_radius, 
            sphere_radius=(self.cfg.object_scale*math.sqrt(3))/2, 
            n=self.cfg.num_grans
        )
        objects_new_state[:, :, 1] *= self.swap_goal[env_ids]
        objects_new_state[:, :, :3] += self.scene.env_origins[env_ids].unsqueeze(1)
        objects_new_state[:, :, 3:7] = self.unit_quat
        objects_new_state[:, :, 7:] = self.unit_vel
        self._object_collection.write_object_state_to_sim(objects_new_state, env_ids=env_ids)

        settle_steps = 30
        for _ in range(settle_steps):
            DirectRLEnv.step_physics_only(self, action=torch.zeros((self.num_envs, 4), device=self.device))

        # Refresh intermediate values for observations and rewards
        self._compute_intermediate_values(env_ids)

        # span distance between objects and target
        self.spawn_distance = torch.norm(self.objects_pos - self.target_pos.unsqueeze(1), dim=-1).to(self.device)

    def _get_observations(self):
        """
        OBSERVATIONS:
            1. End point of pusher position (panda_fingertip_centered_point)
            2. yaw of the end-effector
            3. Each object position 
            4. Target position
        """
        # Convert the end-effector orientation to only the yaw angle
        ee_yaw = self.quaternion_to_yaw(self.ee_orientation)
        
        # Reshape self.objects_pos, each_object_to_target_dists, and pusher_to_each_object_dists to 2D arrays
        objects_pos_reshaped = self.objects_pos.reshape(self.num_envs, -1)  # Shape: [num_envs, num_grans * 3]

        observations = torch.cat(
            (
                self.robot_grasp_pos,      # [x, y, z]                     in world frame
                ee_yaw.unsqueeze(1),       # [yaw_rad]                     in world frame
                objects_pos_reshaped,      # [x, y, z] x env objects       in world frame
                self.target_pos,           # [x, y, z]                     in world frame 
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
        self.objects_pos[env_ids] = self._object_collection.data.object_pos_w[env_ids] # shape (num_envs, num_grans, 3)
        self.objects_pos_mean[env_ids] = torch.mean(self.objects_pos[env_ids], dim=1, keepdim=True).squeeze(1) # shape (num_envs, 3)

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
        objects_pos,
        objects_pos_mean,
        target_pos,
        robot_grasp_pos,
        spawn_distance,
        actions,
        dist_reward_scale,
        target_reward_scale,
        action_penalty_scale,
    ):
        ### Distance from end-effector to objects reward
                                         #[env_id, obj, 3] - [env_id, 3]
        object_pusher_dist_reward = torch.norm(objects_pos - robot_grasp_pos.unsqueeze(1), p=2, dim=-1) # shape (num_envs, num_grans)
        sum_object_pusher_dist_reward = torch.sum(torch.clamp(object_pusher_dist_reward, max=1.0), dim=-1)/self.cfg.num_grans

        ## Mean distance from objects to target reward
        goal_dist = torch.norm(objects_pos_mean - target_pos, p=2, dim=-1) # shape (num_envs)
        spawn_dist_mean = torch.mean(spawn_distance, dim=-1)
        norm_goal_dist = 1- torch.div(goal_dist, spawn_dist_mean) 
        norm_goal_dist = torch.where(norm_goal_dist < self.point1_tensor, self.zero_tensor, norm_goal_dist)

        ###################################################################
        # Does the reward take into account the area or the center point? # NO... 
        ###################################################################

        # action_penalty = torch.sum(actions**2, dim=-1)

        # Total reward
        rewards = (
            - dist_reward_scale * sum_object_pusher_dist_reward
            + target_reward_scale * norm_goal_dist
            # - action_penalty_scale * action_penalty
        )
        # print(
        #     # f'pusher_pos env0:                  {pusher_pos[0]} \n'
        #     # f'object_pos env0:                  {objects_pos[0]} \n'
        #     # f'reward object_pusher_dist_reward: {- dist_reward_scale * object_pusher_dist_reward[0]} \n'
        #     f'reward target_dist_reward:        {target_reward_scale * norm_goal_dist[0]} \n'
        #     # f'reward alignment:                 {alignment_reward_scale * alignment[0]} \n'
        #     # f'reward action_penalty:            {-action_penalty_scale * action_penalty[0]} \n'
        #     # f'reward total env 0:               {rewards[0]} \n'
        #     # f'---'
        # )

        # if "log" not in self.extras:
        #     self.extras["log"] = dict()
        # self.extras["log"]["dist_reward"] = (-dist_reward_scale * sum_object_pusher_dist_reward).mean()
        # self.extras["log"]["target_reward"] = (target_reward_scale * norm_goal_dist).mean()
        # # self.extras["log"]["action_penalty"] = (-action_penalty_scale * action_penalty).mean(),
        # self.extras["log"]["total_reward"] = rewards.mean()
        
        # wandb.log(self.extras["log"], step=self.global_step)
        # self.global_step += 1

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
