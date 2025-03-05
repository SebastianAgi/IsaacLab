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
    robot = scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    # print(robot_entity_cfg)
    print('robot config body IDs:',robot_entity_cfg.body_ids)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    joint_pos_des = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
    #make it a torch tensor
    joint_pos_des = torch.tensor(joint_pos_des, dtype=torch.float32, device='cuda')

    print('###################')
    print(robot_entity_cfg.joint_ids)
    print('###################')

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # print(ee_pose_w)
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)


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