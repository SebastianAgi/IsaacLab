# from __future__ import annotations

# import torch

# from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
# from pxr import UsdGeom

# import isaaclab.sim as sim_utils
# from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.assets import Articulation, ArticulationCfg
# from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
# from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.sim import SimulationCfg
# from isaaclab.sim.utils.stage import get_current_stage
# from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.utils import configclass
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# from isaaclab.utils.math import sample_uniform
# from .franka_slide_env_cfg import FrankaSlideEnvCfg
# from isaaclab.assets.rigid_object.rigid_object import RigidObject


# class FrankaSlideEnv(DirectRLEnv):
#     """Franka-slide object to out of reach goal environment.

#     The environment consists of a Franka robot that needs to slide an object
#     to a goal position that is out of reach of the robot."""

#     # pre-physics step calls
#     #   |-- _pre_physics_step(action)
#     #   |-- _apply_action()
#     # post-physics step calls
#     #   |-- _get_dones()
#     #   |-- _get_rewards()
#     #   |-- _reset_idx(env_ids)
#     #   |-- _get_observations()

#     cfg: FrankaSlideEnvCfg

#     def __init__(self, cfg: FrankaSlideEnvCfg, render_mode: str | None = None, **kwargs):
#         super().__init__(cfg, render_mode, **kwargs)

#         def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
#             """Compute pose in env-local coordinates"""
#             world_transform = xformable.ComputeLocalToWorldTransform(0)
#             world_pos = world_transform.ExtractTranslation()
#             world_quat = world_transform.ExtractRotationQuat()

#             px = world_pos[0] - env_pos[0]
#             py = world_pos[1] - env_pos[1]
#             pz = world_pos[2] - env_pos[2]
#             qx = world_quat.imaginary[0]
#             qy = world_quat.imaginary[1]
#             qz = world_quat.imaginary[2]
#             qw = world_quat.real

#             return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

#         self.dt = self.cfg.sim.dt * self.cfg.decimation

#         # create auxiliary variables for computing applied action, observations and rewards
#         self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
#         self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

#         self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
#         self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
#         self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

#         self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

#         stage = get_current_stage()
#         hand_pose = get_env_local_pose(
#             self.scene.env_origins[0],
#             UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
#             self.device,
#         )
#         lfinger_pose = get_env_local_pose(
#             self.scene.env_origins[0],
#             UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
#             self.device,
#         )
#         rfinger_pose = get_env_local_pose(
#             self.scene.env_origins[0],
#             UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
#             self.device,
#         )

#         finger_pose = torch.zeros(7, device=self.device)
#         finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
#         finger_pose[3:7] = lfinger_pose[3:7]
#         hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

#         robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
#             hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
#         )
#         robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
#         self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
#         self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

#         self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
#             (self.num_envs, 1)
#         )
#         self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
#             (self.num_envs, 1)
#         )

#         self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
#         self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
#         self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

#         self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
#         self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

#     def _setup_scene(self):
#         self._robot = Articulation(self.cfg.robot_high_pd) # GRAVITY DISABLED - diff_ik only works with gravity disabled
#         self._table = RigidObject(self.cfg.table_cfg)
#         self._object = RigidObject(self.cfg.object_cfg)
#         self.scene.articulations["robot"] = self._robot
#         self.scene.rigid_objects["table"] = self._table
#         self.scene.rigid_objects["object"] = self._object

#         self.cfg.terrain.num_envs = self.scene.cfg.num_envs
#         self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
#         self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

#         # Clone, filter, and replicate the environment
#         self.scene.clone_environments(copy_from_source=False)
#         self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
#         # Add lights
#         light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#         light_cfg.func("/World/Light", light_cfg)

#     # pre-physics step calls

#     def _pre_physics_step(self, actions: torch.Tensor):
#         self.actions = actions.clone().clamp(-1.0, 1.0)
#         targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
#         self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

#     def _apply_action(self):
#         self._robot.set_joint_position_target(self.robot_dof_targets)

#     def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#         terminated = self._cabinet.data.joint_pos[:, 3] > 0.39
#         truncated = self.episode_length_buf >= self.max_episode_length - 1
#         return terminated, truncated

#     def _get_rewards(self) -> torch.Tensor:
#         # Refresh the intermediate values after the physics steps
#         self._compute_intermediate_values()
#         robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
#         robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

#         return self._compute_rewards(
#             self.actions,
#             self._cabinet.data.joint_pos,
#             self.robot_grasp_pos,
#             self.drawer_grasp_pos,
#             self.robot_grasp_rot,
#             self.drawer_grasp_rot,
#             robot_left_finger_pos,
#             robot_right_finger_pos,
#             self.gripper_forward_axis,
#             self.drawer_inward_axis,
#             self.gripper_up_axis,
#             self.drawer_up_axis,
#             self.num_envs,
#             self.cfg.dist_reward_scale,
#             self.cfg.rot_reward_scale,
#             self.cfg.open_reward_scale,
#             self.cfg.action_penalty_scale,
#             self.cfg.finger_reward_scale,
#             self._robot.data.joint_pos,
#         )

#     def _reset_idx(self, env_ids: torch.Tensor | None):
#         super()._reset_idx(env_ids)
#         # robot state
#         joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
#             -0.125,
#             0.125,
#             (len(env_ids), self._robot.num_joints),
#             self.device,
#         )
#         joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
#         joint_vel = torch.zeros_like(joint_pos)
#         self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
#         self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

#         # cabinet state
#         zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
#         self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

#         # Need to refresh the intermediate values so that _get_observations() can use the latest values
#         self._compute_intermediate_values(env_ids)

#     def _get_observations(self) -> dict[str, torch.Tensor]:
#         pass

#     def _compute_rewards(self) -> torch.Tensor:
#         pass