"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


from pxr import Usd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
#disable warnings
import warnings
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.app import AppLauncher
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG
# from .franka_EE_env_cfg import FrankaEndEffectorEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

class TestSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sebastian/IsaacLab/Thor_table_usd/Granular_test_bed.usd", #Granular test bed with walls,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0, density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Robot configuration
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sebastian/IsaacLab/frank_panda_usd/Granular_franka_2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
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
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    """Configuration of Franka Emika Panda robot."""

    # prim_utils.create_prim("/World/envs/env_.*/Camera_1", "Xform")
    # prim_utils.create_prim("/World/envs/env_.*/Camera_2", "Xform")

    TiledCamera1 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera_1",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
    )
    TiledCamera2 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera_2",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
    )
    """Configuration of the camera looking at the robot scene."""

def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title.

    Args:
        images: A list of images to be plotted. Shape of each image should be (H, W, C).
        cmap: Colormap to be used for plotting. Defaults to None, in which case the default colormap is used.
        nrows: Number of rows in the grid. Defaults to 1.
        subtitles: A list of subtitles for each image. Defaults to None, in which case no subtitles are shown.
        title: Title of the grid. Defaults to None, in which case no title is shown.
        filename: Path to save the figure. Defaults to None, in which case the figure is not saved.
    """
    # show images in a grid
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    # Ensure axes is a list, even when only one subplot is created
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()

    # plot images
    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy()
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx])
    # remove extra axes if any
    for ax in axes[n_images:]:
        fig.delaxes(ax)
    # set title
    if title:
        plt.suptitle(title)

    # adjust layout to fit the title
    plt.tight_layout()
    # save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    # close the figure
    plt.close()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene.articulations["robot"]
    camera1: Camera = scene["TiledCamera1"]
    camera2: Camera = scene["TiledCamera2"]

    # Create output directory to save images
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Camera positions, targets, orientations
    camera_positions = torch.tensor([[0.4, 1.0, 1.0],[0.4, -1.0, 1.0]], device=sim.device)
    camera_targets = torch.tensor([[0.4, 0.0, 0.0],[0.4, 0.0, 0.0]], device=sim.device)
    # These orientations are in ROS-convention, and will position the cameras to view the origin
    camera_orientations = torch.tensor(  # noqa: F841
        [[-0.1759, 0.3399, 0.8205, -0.4247], [-0.4247, 0.8205, -0.3399, 0.1759]], device=sim.device
    )

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    camera2.set_world_poses_from_view(
        camera_positions[1].unsqueeze(0),
        camera_targets[0].unsqueeze(0)
    )
    camera1.set_world_poses_from_view(
        camera_positions[0].unsqueeze(0),
        camera_targets[1].unsqueeze(0)
    )
    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        # [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
        # [0.5, -0.4, 0.6, 1.0, 0.0, 0.0, 0.0],
        [0.3, -0.4, 0.6, 0.0, 0.7071, 0.7071, 0.0],
        [0.3, 0.4, 0.6, 0.0, 0.7071, 0.7071, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]


    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    
    print('###################')
    print('robot_entity_cfg: ',robot_entity_cfg)
    print('diff_ik_controller.action_dim: ',diff_ik_controller.action_dim)

    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # joint_pos_des = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
    # #make it a torch tensor
    # joint_pos_des = torch.tensor(joint_pos_des, dtype=torch.float32, device='cuda')

    print('###################')
    print(robot_entity_cfg.joint_ids)
    print('###################')

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # reset joint state
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            print('yeet this resets often')
            # reset time
            count = 0
            # # reset joint state
            # joint_pos = robot.data.default_joint_pos.clone()
            # joint_vel = robot.data.default_joint_vel.clone()
            # robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)

        else:
            # if count % 10 == 0:
            #     # update the goal
            #     ik_commands[:] = ee_goals[current_goal_idx]
            #     diff_ik_controller.set_command(ik_commands)
            #     current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            print(robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7])
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        camera1.update(sim_dt)
        camera2.update(sim_dt)
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

        print(scene["TiledCamera1"])
        print("Received shape of rgb   image: ", scene["TiledCamera1"].data.output["rgb"].shape)
        print(scene["TiledCamera2"])
        print("Received shape of rgb   image: ", scene["TiledCamera2"].data.output["rgb"].shape)
        print("-------------------------------")

        # save every 10th image (for visualization purposes only)
        # note: saving images will slow down the simulation
        if count % 10 == 0:
            # save all tiled RGB images
            tiled_images = scene["TiledCamera1"].data.output["rgb"]
            # add images from the second camera to the list
            tiled_images = torch.cat([tiled_images, scene["TiledCamera2"].data.output["rgb"]], dim=0)
            save_images_grid(
                tiled_images,
                subtitles=[f"Cam{i}" for i in range(tiled_images.shape[0])],
                title="Tiled RGB Image",
                filename=os.path.join(output_dir, "tiled_rgb", f"{count:04d}.jpg"),
            )


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Create a scene
    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.0)
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