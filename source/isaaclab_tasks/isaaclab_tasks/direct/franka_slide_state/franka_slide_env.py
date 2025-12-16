from __future__ import annotations

import torch

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.sensors.camera.tiled_camera import TiledCamera
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from .franka_slide_env_cfg import FrankaSlideStateEnvCfg
from isaaclab.assets.rigid_object.rigid_object import RigidObject


class FrankaSlideStateEnv(DirectRLEnv):
    """Franka-slide object to out of reach goal environment.

    The environment consists of a Franka robot that needs to slide an object
    to a goal position that is out of reach of the robot."""

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaSlideStateEnvCfg

    def __init__(self, cfg: FrankaSlideStateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: str) -> torch.Tensor:
            """Compute pose in env-local coordinates"""
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

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_pos[:,:] = torch.tensor(cfg.target_pose, dtype=torch.float, device=self.device)
        self.init_target_pos = self.target_pos.clone().detach()

        # Create initial object poses
        self.object_state = torch.zeros((self.num_envs, 13), dtype=torch.float, device=self.device)
        self.object_state[:, :3] = torch.tensor([0.0, 0.0, 0.3225], dtype=torch.float, device=self.device)
        self.object_state[:, 3:7] = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=self.device)  
        self.default_object_state = self.object_state.clone().detach()
        self.object_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.scale_tensor = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.scale_tensor[:,:] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float, device=self.device)

        # Change the objects position to the spawn points
        self._object.write_root_state_to_sim(self.object_state) 

        # Marker
        # Initialize the goal rotation
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_marker = VisualizationMarkers(self.cfg.goal_object_cfg)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_high_pd) # GRAVITY DISABLED - diff_ik only works with gravity disabled
        self._table = RigidObject(self.cfg.table_cfg)
        self._object = RigidObject(self.cfg.object_cfg)
        # self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["object"] = self._object
        # self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Tell the viewer / renderer to use this camera
        # self.sim.set_camera_view(eye=(2.274, 0.927, 1.288), target=(0.0, 0.0, 0.3))


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

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        self._compute_intermediate_values()
        # terminate if object is within goal region
        terminated = torch.norm(self.object_pos - self.target_pos, p=2, dim=-1) < 0.2
        # Done if timeout occurs
        timeouts = self.episode_length_buf >= self.max_episode_length -1

        return terminated, timeouts

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.actions,
            self.object_pos,
            self.target_pos,
            self.robot_grasp_pos,
            self.cfg.object_target_dist_reward_scale,
            self.cfg.hand_obj_dist_reward_scale,
            self.cfg.action_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset Object state
        self.object_state[env_ids, :13] = self.default_object_state[env_ids, :13]
        self.object_state[env_ids, :3] += self.scene.env_origins[env_ids]
        self._object.write_root_state_to_sim(self.object_state[env_ids], env_ids=env_ids)

        # reset goal position
        self.target_pos[env_ids] = self.init_target_pos[env_ids] + self.scene.env_origins[env_ids]

        # update goal pose and markers
        self.goal_marker.visualize(self.target_pos, self.goal_rot)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

        # span distance between objects and target
        self.spawn_distance = torch.norm(self.object_pos - self.target_pos.unsqueeze(1), dim=-1).to(self.device)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.object_pos - self.target_pos

        # print('obj pos shape:', self.object_pos.shape)
        # print('to target shape:', to_target.shape)
        # print('dof pos shape:', dof_pos_scaled.shape)
        # print('joint vel shape:', self._robot.data.joint_vel.shape)

        obs = torch.cat(
            (
                dof_pos_scaled, # 7
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale, # 7
                to_target, # 3
                self.object_pos, # 3
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # data for object
        self.object_pose[env_ids] = self._object.data.root_com_pose_w[env_ids] # shape (num_grans, 7)
        self.object_pos[env_ids] = self.object_pose[env_ids, :3] # shape (num_envs, 3)

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
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
        object_pos,
        target_pos,
        hand_pos,
        object_target_dist_reward_scale,  # kept for interface compatibility
        hand_obj_dist_reward_scale,       # kept for interface compatibility
        action_penalty_scale,
    ):
        # distances
        reach_dist = torch.norm(hand_pos - object_pos, p=2, dim=-1)  # (num_envs,)
        push_dist = torch.norm((object_pos - target_pos)[..., :2], p=2, dim=-1)  # XY only

        # 1) Hand–object proximity: encourage getting and staying near the object
        reach_reward = -reach_dist

        # 2) Object–goal progress: reward reduction in distance to goal between steps
        if not hasattr(self, "prev_push_dist"):
            self.prev_push_dist = push_dist.clone().detach()

        push_progress = self.prev_push_dist - push_dist
        push_progress_scale = 20.0
        push_progress_reward = push_progress * push_progress_scale

        # 3) Modulate push reward by hand–object distance (soft, not a hard gate)
        contact_scale = torch.exp(-reach_dist / 0.2)
        push_reward = contact_scale * push_progress_reward

        # 4) Small shaping on absolute distance to goal
        push_distance_reward = -0.5 * push_dist

        # 5) Action regularization
        action_penalty = torch.sum(actions ** 2, dim=-1)
        action_cost = action_penalty_scale * action_penalty

        # 6) Combine terms
        reward = (
            1.0 * reach_reward
            + 1.0 * push_reward
            + 0.5 * push_distance_reward
            - action_cost
        )

        # update stored distance for next step
        self.prev_push_dist = push_dist.clone().detach()

        return reward

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