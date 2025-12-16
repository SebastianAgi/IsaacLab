# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the operational space controller (OSC) with the simulator.

The OSC controller can be configured in different modes. It uses the dynamical quantities such as Jacobians and
mass matricescomputed by PhysX.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_osc.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the operational space controller.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg, RigidObject
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with a tilted wall."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/stand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.3)),
        
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/Sebastian/isaacsim/standalone_examples/testing/dark_table.usd", 
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0, density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        # physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3), rot=(0.7071068, 0, 0, 0.7071068)),
    )

    # object 
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1000.0,
            ),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0, density=500.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Object",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    robot.spawn.rigid_props.disable_gravity = True
    robot.init_state.pos=(-0.5, 0.0, 0.3)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
    """

    # Extract scene entities for readability.
    robot = scene["robot"]
    contact_forces = scene["contact_forces"]

    # Obtain indices for the end-effector and arm joints
    ee_frame_name = "panda_leftfinger"
    arm_joint_names = ["panda_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    # Create the OSC
    # For debugging: pure Cartesian pose control, no force control.
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_damping_ratio_task=2.0,
        contact_wrench_stiffness_task=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],
        contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
        nullspace_control="position",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy() # type: ignore[union-attr]
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Parameters for the randomized circle-push behavior
    circle_radius = 0.15  # meters (distance from cube center to start/end points)
    approach_height = 0.03  # meters above cube for approach
    push_distance = 0.18  # how far to push through the cube
    push_force = 2.0  # target force magnitude in Newtons

    # We will control force in the task frame across X/Y (push direction expressed in task frame).
    # OSC will be configured to regulate both X and Y force components so the push direction
    # can be arbitrary in the x/y plane without rotating the task frame yaw explicitly.

    # Basic Cartesian stiffness used for the other axes (not the push DOF(s)).
    # default_kp = torch.tensor([360.0, 360.0, 360.0, 360.0, 360.0, 360.0], device=sim.device)
    default_kp = torch.tensor([150.0, 150.0, 150.0, 50.0, 50.0, 50.0], device=sim.device)

    num_envs = scene.num_envs

    # Per-env state variables
    # 0 = go to start point; 1 = push across to end point; 2 = finished (will reset)
    behavior_state = torch.zeros(num_envs, dtype=torch.int64, device=sim.device)

    # per-env random angles for start point on circle
    angles = (2 * torch.pi * torch.rand(num_envs, device=sim.device)).to(sim.device)

    # get cube world position (center) from scene object
    cube = scene["object"]
    # cube root pos in world frame (will be updated below each loop iteration)
    cube_pos_w = cube.data.root_pos_w.clone()

    # create default cube spawn pose 
    default_root_pos = torch.tensor([0.0, 0.0, 0.35], device=sim.device)
    default_root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
    default_root_lin_vel = torch.zeros(3, device=sim.device)
    default_root_ang_vel = torch.zeros(3, device=sim.device)
    default_root_pose = torch.cat([default_root_pos, default_root_quat, default_root_lin_vel, default_root_ang_vel], dim=-1)
    default_root_pose = default_root_pose.repeat(num_envs, 1)

    # start and end positions in world frame (initialized, will be updated after reset)
    start_pos_w = torch.zeros(num_envs, 3, device=sim.device)
    end_pos_w = torch.zeros(num_envs, 3, device=sim.device)

    # a convenient helper to get a downward pointing quaternion in world frame
    # (used so the end effector points downwards). This is a simple preset quaternion
    # used in this tutorial; you may rotate yaw if you prefer explicit alignment.
    # shape (num_envs, 4) to match per-env operations in subtract_frame_transforms.
    quat_down_world = torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device).repeat(num_envs, 1)

    # Fixed task frame at the robot root: identity pose in body frame for all envs.
    # This makes OSC interpret pose_abs commands directly as EE pose in root frame.
    identity_task_pose_b = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=sim.device
    ).repeat(num_envs, 1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Update existing buffers
    # Note: We need to update buffers before the first step for the controller.
    robot.update(dt=sim_dt)

    # Get the center of the robot soft joint limits
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)

    # get the updated states
    (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces)

    # Track the given target command
    current_goal_idx = 0  # Current goal index for the arm
    command = torch.zeros(
        scene.num_envs, osc.action_dim, device=sim.device
    )  # Generic target command (here: only pose and gains).
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the body frame
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the world frame (for marker)

    # Set joint efforts to zero
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)

    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset every 500 steps or on a high root z (sanity)
        if count % 500 == 0 or (root_pose_w[:, 2] > 0.5).any():
            # reset joint state to default
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
            robot.write_data_to_sim()
            robot.reset()
            # reset cube position
            cube.write_root_state_to_sim(default_root_pose)
            cube.reset()
            # reset contact sensor
            contact_forces.reset()

            # Re-sample random angles and compute start/end points in world frame per env
            angles = (2 * torch.pi * torch.rand(num_envs, device=sim.device)).to(sim.device)
            # update cube world position
            cube_pos_w = cube.data.root_pos_w.clone()
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)
            start_pos_w[:, 0] = cube_pos_w[:, 0] + cos_a * circle_radius
            start_pos_w[:, 1] = cube_pos_w[:, 1] + sin_a * circle_radius
            start_pos_w[:, 2] = cube_pos_w[:, 2] + approach_height
            # opponent point on circle and a bit further to ensure crossing
            end_pos_w[:, 0] = cube_pos_w[:, 0] - cos_a * circle_radius*2# + push_distance)
            end_pos_w[:, 1] = cube_pos_w[:, 1] - sin_a * circle_radius*2# + push_distance)
            end_pos_w[:, 2] = cube_pos_w[:, 2] + approach_height

            # reset target pose: compute pose of start in body frame
            robot.update(sim_dt)
            _, _, _, ee_pose_b, _, root_pose_w, _, _, _, _ = update_states(
                sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces
            )

            # convert start pose from world to body frame
            start_pose_b_pos, start_pose_b_quat = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], start_pos_w, quat_down_world
            )
            start_pose_b = torch.cat([start_pose_b_pos, start_pose_b_quat], dim=-1)

            # compute end pose in body frame
            end_pose_b_pos, end_pose_b_quat = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], end_pos_w, quat_down_world
            )
            end_pose_b = torch.cat([end_pose_b_pos, end_pose_b_quat], dim=-1)

            # initialize command to go to the start pose (pure pose control)
            command = torch.zeros(num_envs, osc.action_dim, device=sim.device)
            command[:, :7] = start_pose_b
            # set default kp gains
            command[:, 7:13] = default_kp.unsqueeze(0).repeat(num_envs, 1)

            # set the osc command directly in body frame with a fixed root task frame
            osc.reset()
            osc.set_command(
                command=command,
                current_ee_pose_b=ee_pose_b,
                current_task_frame_pose_b=identity_task_pose_b,
            )

            # reset behavior state
            behavior_state[:] = 0
        else:
            # get the updated states
            (
                jacobian_b,
                mass_matrix,
                gravity,
                ee_pose_b,
                ee_vel_b,
                root_pose_w,
                ee_pose_w,
                ee_force_b,
                joint_pos,
                joint_vel,
            ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces)

            # prepare the next command per-env depending on state (pose + gains only)
            command = torch.zeros(num_envs, osc.action_dim, device=sim.device)

            # distances from EE to start and end in world frame
            ee_pos_w = ee_pose_w[:, 0:3]
            dist_to_start = torch.norm(ee_pos_w - start_pos_w, dim=-1)
            dist_to_end = torch.norm(ee_pos_w - end_pos_w, dim=-1)

            # masks
            go_mask = behavior_state == 0
            push_mask = behavior_state == 1

            # For envs that still need to go to the start point, set pose targets to start_pose_b
            if go_mask.any():
                idx = go_mask.nonzero(as_tuple=False).squeeze(-1)
                command[idx, :7] = start_pose_b[idx]
                command[idx, 7:13] = default_kp.unsqueeze(0).repeat(len(idx), 1)

                # switch to push when close enough to start
                arrived = (dist_to_start < 0.005) & go_mask
                if arrived.any():
                    behavior_state[arrived] = 1

            # For envs that are in push mode, set wrench and end pose
            if push_mask.any():
                idx = push_mask.nonzero(as_tuple=False).squeeze(-1)

                # place end pose as pose target (pure pose control)
                command[idx, :7] = end_pose_b[idx]
                command[idx, 7:13] = default_kp.unsqueeze(0).repeat(len(idx), 1)

                # check for completion (reached/passed end point)
                finished = (dist_to_end < 0.03) & push_mask
                if finished.any():
                    behavior_state[finished] = 2

            # For finished envs, keep last command zeroed (or you could set a hold pose)
            done_mask = behavior_state == 2
            if done_mask.any():
                idx = done_mask.nonzero(as_tuple=False).squeeze(-1)
                command[idx, :7] = end_pose_b[idx]
                command[idx, 7:13] = default_kp.unsqueeze(0).repeat(len(idx), 1)

            # update OSC with the absolute pose/wrench command in body frame, using
            # a fixed identity task frame at the robot root
            osc.set_command(
                command=command,
                current_ee_pose_b=ee_pose_b,
                current_task_frame_pose_b=identity_task_pose_b,
            )

            # compute the joint commands
            joint_efforts = osc.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                current_ee_force_b=ee_force_b,
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=joint_centers,
            )

            # apply actions
            robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()

        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # choose which world target to visualize: start for state 0, end for state 1 or 2
        target_pos_w = torch.where(
            (behavior_state == 0).unsqueeze(-1),
            start_pos_w,
            end_pos_w,
        )
        target_quat_w = quat_down_world  # same downward orientation

        goal_marker.visualize(target_pos_w, target_quat_w)

        # perform step
        sim.step(render=True)
        # update robot buffers
        robot.update(sim_dt)
        # update buffers
        scene.update(sim_dt)
        # update sim-time
        count += 1


# Update robot states
def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
    contact_forces,
):
    """Update the robot states.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        robot: (Articulation) Robot articulation.
        ee_frame_idx: (int) End-effector frame index.
        arm_joint_ids: (list[int]) Arm joint indices.
        contact_forces: (ContactSensor) Contact sensor.

    Returns:
        jacobian_b (torch.tensor): Jacobian in the body frame.
        mass_matrix (torch.tensor): Mass matrix.
        gravity (torch.tensor): Gravity vector.
        ee_pose_b (torch.tensor): End-effector pose in the body frame.
        ee_vel_b (torch.tensor): End-effector velocity in the body frame.
        root_pose_w (torch.tensor): Root pose in the world frame.
        ee_pose_w (torch.tensor): End-effector pose in the world frame.
        ee_force_b (torch.tensor): End-effector force in the body frame.
        joint_pos (torch.tensor): The joint positions.
        joint_vel (torch.tensor): The joint velocities.

    Raises:
        ValueError: Undefined target_type.
    """
    # obtain dynamics related quantities from simulation
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Extract end-effector velocity in the world frame
    root_vel_w = robot.data.root_vel_w  # Extract root velocity in the world frame
    relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Calculate the contact force
    ee_force_w = torch.zeros(scene.num_envs, 3, device=sim.device)
    sim_dt = sim.get_physics_dt()
    contact_forces.update(sim_dt)  # update contact sensor
    # Calculate the contact force by averaging over last four time steps (i.e., to smoothen) and
    # taking the max of three surfaces as only one should be the contact of interest
    ee_force_w, _ = torch.max(torch.mean(contact_forces.data.net_forces_w_history, dim=1), dim=1)

    # This is a simplification, only for the sake of testing.
    ee_force_b = ee_force_w

    # Get joint positions and velocities
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    )


# Update the target commands
def update_target(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    osc: OperationalSpaceController,
    root_pose_w: torch.tensor,
    ee_target_set: torch.tensor,
    current_goal_idx: int,
):
    """Update the targets for the operational space controller.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        osc: (OperationalSpaceController) Operational space controller.
        root_pose_w: (torch.tensor) Root pose in the world frame.
        ee_target_set: (torch.tensor) End-effector target set.
        current_goal_idx: (int) Current goal index.

    Returns:
        command (torch.tensor): Updated target command.
        ee_target_pose_b (torch.tensor): Updated target pose in the body frame.
        ee_target_pose_w (torch.tensor): Updated target pose in the world frame.
        next_goal_idx (int): Next goal index.

    Raises:
        ValueError: Undefined target_type.
    """

    # update the ee desired command
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:] = ee_target_set[current_goal_idx]

    # update the ee desired pose
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        elif target_type == "wrench_abs":
            pass  # ee_target_pose_b could stay at the root frame for force control, what matters is ee_target_b
        else:
            raise ValueError("Undefined target_type within update_target().")

    # update the target desired pose in world frame (for marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    next_goal_idx = (current_goal_idx + 1) % len(ee_target_set)

    return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


# Convert the target commands to the task frame
def convert_to_task_frame(osc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor):
    """Converts the target commands to the task frame.

    Args:
        osc: OperationalSpaceController object.
        command: Command to be converted.
        ee_target_pose_b: Target pose in the body frame.

    Returns:
        command (torch.tensor): Target command in the task frame.
        task_frame_pose_b (torch.tensor): Target pose in the task frame.

    Raises:
        ValueError: Undefined target_type.
    """
    command = command.clone()
    task_frame_pose_b = ee_target_pose_b.clone()

    cmd_idx = 0
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
            )
            cmd_idx += 7
        elif target_type == "wrench_abs":
            # These are already defined in target frame for ee_goal_wrench_set_tilted_task (since it is
            # easier), so not transforming
            cmd_idx += 6
        else:
            raise ValueError("Undefined target_type within _convert_to_task_frame().")

    return command, task_frame_pose_b


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))
    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
