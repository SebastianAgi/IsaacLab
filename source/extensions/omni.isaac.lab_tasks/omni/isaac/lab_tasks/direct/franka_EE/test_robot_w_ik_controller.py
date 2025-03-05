"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.lab.app import AppLauncher


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
import torch
#disable warnings
import warnings
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene.interactive_scene import InteractiveScene
from omni.isaac.lab.scene.interactive_scene_cfg import InteractiveSceneCfg
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG
# from .franka_EE_env_cfg import FrankaEndEffectorEnvCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

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
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
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



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene.articulations["robot"]

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
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


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