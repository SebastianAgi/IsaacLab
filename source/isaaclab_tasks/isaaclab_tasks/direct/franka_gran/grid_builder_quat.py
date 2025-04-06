import matplotlib.pyplot as plt
from pytictac import Timer
import torch
import math
from isaaclab.utils.math import matrix_from_quat
import cv2
import numpy as np
import threading
import os
import wandb

@torch.jit.script
def batch_get_downward_yaw_tensor_vectorized(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Fully vectorized version that processes all quaternions at once.
    
    Args:
        quaternions: Tensor of shape [batch_size, 4] with quaternions (w,x,y,z)
        
    Returns:
        Tensor of shape [batch_size] with yaw angles in radians
    """
    # Define the six canonical face normals and associated tangents
    face_normals = torch.tensor([
        [ 1.0,  0.0,  0.0],  # +X
        [-1.0,  0.0,  0.0],  # -X 
        [ 0.0,  1.0,  0.0],  # +Y
        [ 0.0, -1.0,  0.0],  # -Y
        [ 0.0,  0.0,  1.0],  # +Z
        [ 0.0,  0.0, -1.0]   # -Z
    ], dtype=quaternions.dtype, device=quaternions.device)
    
    face_tangents = torch.tensor([
        [0.0, 0.0, -1.0],  # Tangent for +X (points along -Z)
        [0.0, 0.0, -1.0],  # Tangent for -X (points along -Z)
        [1.0, 0.0,  0.0],  # Tangent for +Y (points along +X)
        [1.0, 0.0,  0.0],  # Tangent for -Y (points along +X)
        [1.0, 0.0,  0.0],  # Tangent for +Z (points along +X)
        [1.0, 0.0,  0.0]   # Tangent for -Z (points along +X)
    ], dtype=quaternions.dtype, device=quaternions.device)
    
    # Get rotation matrices [num_envs, num_obj, 3, 3]
    R = matrix_from_quat(quaternions)

    rotated_normals = torch.einsum('abij,kj->abkj', R, face_normals)
    
    # Find the face most aligned with down direction
    down_vector = torch.tensor([0.0, 0.0, -1.0], dtype=quaternions.dtype, device=quaternions.device)

    dots = torch.einsum("abkj,j->abk", rotated_normals, down_vector)

    max_indices = dots.argmax(dim=2)  # shape: [num_envs, num_obj]
    
    # Select the corresponding face tangents: [6, 3]
    selected_tangents = face_tangents[max_indices]
    
    # Rotate the selected tangents for each quaternion
    # [num_envs, num_obj, 3] = [num_envs, num_obj, 3, 3] @ [num_envs, num_obj, 3]
    rotated_tangents = torch.einsum('abij,abj->abi', R, selected_tangents)

    yaw = torch.atan2(rotated_tangents[..., 1], rotated_tangents[..., 0])

    return yaw


@torch.jit.script
def target_layer(target_pos: torch.Tensor, target_radius: float, xy_world: torch.Tensor) -> torch.Tensor:
    """Generate a target area mask."""
    # -----------------------------------
    # Channel 0: Target area (vectorized with squared distances)
    # -----------------------------------
    # Avoid torch.norm (which computes a sqrt) by comparing squared distances.
    diff = xy_world.unsqueeze(0) - target_pos  # shape: (num_envs, H, W, 2)
    dist_sq = diff[..., 0] ** 2 + diff[..., 1] ** 2
    target_mask = dist_sq <= (target_radius ** 2)
    
    # # Expand the mask to include neighboring pixels
    # target_mask = torch.nn.functional.max_pool2d(target_mask.unsqueeze(1).float(), kernel_size=3, stride=1, padding=1).squeeze(1).to(torch.bool)
    
    return target_mask

@torch.jit.script
def EE_layer(
    ee_pos: torch.Tensor, 
    ee_orient: torch.Tensor, 
    xy_world: torch.Tensor, 
    end_effector_length: float, 
    end_effector_width: float,
    x_margin: float,  # assuming square cells for simplicity
    y_margin: float
) -> torch.Tensor:
    """Generate an end effector mask with inflated boundaries."""
    # Calculate differences and rotate into the end effector frame.
    ee_diff = xy_world.unsqueeze(0) - ee_pos  # shape: (num_envs, H, W, 2)
    cos_theta = torch.cos(ee_orient)
    sin_theta = torch.sin(ee_orient)
    x_local = cos_theta * ee_diff[..., 0] + sin_theta * ee_diff[..., 1]
    y_local = -sin_theta * ee_diff[..., 0] + cos_theta * ee_diff[..., 1]
    
    half_length = end_effector_length / 2.0
    half_width  = end_effector_width / 2.0
    # Inflate boundaries by half the grid cell size.
    
    ee_mask = (x_local >= -(half_length + x_margin)) & (x_local <= (half_length + x_margin)) & \
              (y_local >= -(half_width  + y_margin)) & (y_local <= (half_width  + y_margin))

    return ee_mask


@torch.jit.script
def object_layer(obj_mask: torch.Tensor, 
                 obj_pos_chunk: torch.Tensor, 
                 obj_orient_chunk: torch.Tensor, 
                 coords: torch.Tensor, 
                 object_size: float,
                 x_margin: float,
                 y_margin: float
                ) -> torch.Tensor:  # <-- Changed to float
    """Generate a mask for a single object."""
    # -----------------------------------
    # Channel 2: Objects (vectorized)
    # -----------------------------------    

    obj_diff = coords - obj_pos_chunk
    
    cos_theta_chunk = torch.cos(obj_orient_chunk)
    sin_theta_chunk = torch.sin(obj_orient_chunk)
    dx = obj_diff[..., 0]
    dy = obj_diff[..., 1]
    x_local = cos_theta_chunk * dx + sin_theta_chunk * dy
    y_local = -sin_theta_chunk * dx + cos_theta_chunk * dy
    
    half_side = object_size / 2
    inside = (x_local >= -(half_side + x_margin)) & (x_local <= (half_side + x_margin)) & \
                (y_local >= -(half_side + y_margin)) & (y_local <= (half_side + y_margin))
    # Reduce over objects in the chunk (logical OR across the object dimension).
    inside_any = inside.any(dim=1)
    obj_mask |= inside_any
    
    # # Expand the mask to include neighboring pixels
    # expanded_inside_any = torch.nn.functional.max_pool2d(inside_any.unsqueeze(1).float(), kernel_size=3, stride=1, padding=1).squeeze(1).to(torch.bool)
    # obj_mask |= expanded_inside_any

    return obj_mask


def save_grid_images_async(grid_tensor: torch.Tensor, 
                           chosen_env: int, 
                           step: int, 
                           base_filename: str, 
                           save_dir: str, 
                           log_to_wandb: bool = True):
    """
    Save each environment's grid (shape [3, H, W]) as an image in the specified directory.
    
    Args:
        grid_tensor: Tensor of shape [num_envs, 3, H, W] with values in [0, 1].
        base_filename: Base filename to which an index will be appended.
        save_dir: Directory in which to save the images.
    """
    # Ensure the save directory exists.
    os.makedirs(save_dir, exist_ok=True)
    
    # Detach and move the grid tensor to CPU.
    grid_np = grid_tensor.detach().cpu().numpy()  # shape: [num_envs, 3, H, W]

    # Extract one environment's grid.
    img = grid_np[chosen_env]  # shape: [3, H, W]
    # Transpose to [H, W, 3]
    img = np.transpose(img, (1, 2, 0))
    # Scale from [0, 1] to [0, 255] and convert to uint8.
    img = (img * 255).astype(np.uint8)
    # Build the filename.
    filename = os.path.join(save_dir, f"{base_filename}_{step}.png")
    
    def writer_and_logger(im, fn, global_step):
        # Save image to disk
        cv2.imwrite(fn, im)
        
        # Log to wandb if requested
        if log_to_wandb:
            try:
                # Log the image to wandb
                wandb.log({
                    "observation_grid": wandb.Image(im)
                }, step=global_step)
            except Exception as e:
                print(f"Error logging to wandb: {e}")
    
    # Launch the writer and logger in a separate thread.
    threading.Thread(target=writer_and_logger, 
                    args=(img, filename, step), 
                    daemon=True).start()


class GridObservationGenerator:
    def __init__(self, grid_shape=(256, 256), area = (0.65,0.75), device="cuda", dtype=torch.float16):
        """Initialize with fixed grid properties."""
        x_pixel_size= area[0]/grid_shape[0]
        y_pixel_size= area[1]/grid_shape[1]
        self.grid_shape = grid_shape
        self.device = device
        self.dtype = dtype
        
        # Pre-compute the grid coordinates only once.
        H, W = grid_shape
        i_coords = torch.arange(H, device=device, dtype=dtype) + 0.5
        j_coords = torch.arange(W, device=device, dtype=dtype) + 0.5
        y_grid, x_grid = torch.meshgrid(i_coords, j_coords, indexing='ij')
        
        # Store world coordinates.
        self.x_world = x_grid * x_pixel_size  # shape (H, W)
        self.y_world = y_grid * y_pixel_size  # shape (H, W)

        self.x_margin = x_pixel_size / 2.0
        self.y_margin = y_pixel_size / 2.0
        
    def generate_observations(
        self, 
        object_positions: torch.Tensor,      # shape (num_envs, num_obj, 2)
        target_area_position: torch.Tensor,    # shape (num_envs, 2)
        end_effector_position: torch.Tensor,   # shape (num_envs, 2)
        object_orientations: torch.Tensor,     # shape (num_envs, num_obj)
        end_effector_orientation: torch.Tensor, # shape (num_envs,)
        target_radius: float = 0.1,
        object_size: float = 0.01,
        end_effector_length: float = 0.01,
        end_effector_width: float = 0.05,
        chunk_size: int = 1,
    ) -> torch.Tensor:
        """Generate 3-channel grid observations for multiple environments.
        
        Returns:
            observations: torch.Tensor of shape (num_envs, 3, H, W)
                - Channel 0: Target area (circle)
                - Channel 1: Robot end-effector (rotated rectangle)
                - Channel 2: Objects (aggregated from object shapes)
        """
        num_envs = object_positions.shape[0]
        num_obj = object_positions.shape[1]
        H, W = self.grid_shape
        
        # Use a boolean tensor for intermediate storage.
        observations = torch.zeros((num_envs, 3, H, W), device=self.device, dtype=torch.bool)
        # Precompute grid coordinates with shape (H, W, 2).
        xy_world = torch.stack([self.x_world, self.y_world], dim=-1)
        
        # Layer 0: Target area
        target_pos = target_area_position.view(num_envs, 1, 1, 2)
        observations[:, 0] = target_layer(target_pos, target_radius, xy_world)

        # Layer 1: End-effector
        ee_pos = end_effector_position.view(num_envs, 1, 1, 2)
        ee_orient = end_effector_orientation.view(num_envs, 1, 1)
        observations[:, 1] = EE_layer(ee_pos, ee_orient, xy_world, end_effector_length, end_effector_width, self.x_margin, self.y_margin)
        
        # Layer 2: Objects (process objects in chunks to limit broadcast size)
        # Instead of broadcasting over all objects at once, process them in smaller chunks.
        obj_mask = torch.zeros((num_envs, H, W), device=self.device, dtype=torch.bool)
        # chunk_size = 1  # Adjust this based on your memory/performance trade-off.
        for i in range(0, num_obj, chunk_size):
            end_i = min(i + chunk_size, num_obj)
            # Extract a chunk of objects: shape (num_envs, chunk_size, 2) and (num_envs, chunk_size)
            chunk_obj_pos = object_positions[:, i:end_i, :]
            chunk_obj_orient = object_orientations[:, i:end_i]
            
            # Reshape for broadcasting: (num_envs, chunk_size, 1, 1, 2)
            obj_pos_chunk = chunk_obj_pos.view(num_envs, end_i - i, 1, 1, 2)
            obj_orient_chunk = chunk_obj_orient.view(num_envs, end_i - i, 1, 1)
            
            # Reshape grid coordinates for broadcasting: (1, 1, H, W, 2)
            coords = xy_world.view(1, 1, H, W, 2)
            
            obj_mask = object_layer(obj_mask, obj_pos_chunk, obj_orient_chunk, coords, object_size, self.x_margin, self.y_margin)
        
        observations[:, 2] = obj_mask
        
        return observations.to(dtype=self.dtype)



if __name__ == "__main__":
    # Initialize the generator.
    grid_gen = GridObservationGenerator(grid_shape=(16,16), area=(0.65, 0.75), dtype=torch.float16)
    
    # Test with 400 environments and 10 objects each.
    num_envs = 4096
    num_obj = 4
    obj_pos = torch.rand((num_envs, num_obj, 2), device='cuda') * 0.65
    obj_pos[:, 0, :] = torch.tensor([0.65/2, 0.75/2])
    obj_quat = torch.rand((num_envs, num_obj, 4), device='cuda')
    target_area = torch.rand((num_envs, 2), device='cuda') * 0.65
    ee_pos = torch.rand((num_envs, 2), device='cuda') * 0.65
    # obj_yaw = torch.rand((num_envs, num_obj), device='cuda')
    ee_yaw = torch.rand((num_envs,), device='cuda')

    print('obj_quat shape:', obj_quat.shape)

    for _ in range(10):
        with Timer("obj_yaw generation"):
            obj_yaw = batch_get_downward_yaw_tensor_vectorized(obj_quat)
    
    print("obj_yaw shape:", obj_yaw.shape)
    
    # Warm up TorchScript (first call compiles the function).
    for _ in range(10):
        with Timer("Grid generation"):
            obs = grid_gen.generate_observations(obj_pos, 
                                                 target_area, 
                                                 ee_pos, 
                                                 obj_yaw, 
                                                 ee_yaw, 
                                                 object_size=0.01)
    print("Observation shape:", obs.shape)

    print(torch.unique(obs[0]))
    
    for i in range(10):
        mask1 = obs[i]
        print(mask1.shape)
        mask1 = mask1.sum(dim=0)
        # mask1 = mask1[2]
        plt.imshow(mask1.cpu().numpy(), cmap='hot')
        plt.axis('off')
        plt.tight_layout()
        # plt.savefig(f'cube_mask_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.show()
