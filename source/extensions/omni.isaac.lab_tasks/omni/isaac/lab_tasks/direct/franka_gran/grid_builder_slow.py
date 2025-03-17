import torch
from pytictac import Timer
import matplotlib.pyplot as plt

class GridObservationGenerator:
    def __init__(self, grid_shape=(256, 256), device="cuda", dtype=torch.float16):
        """Initialize with fixed grid properties."""
        pixel_size = 0.65/grid_shape[0]
        self.grid_shape = grid_shape
        self.pixel_size = pixel_size
        self.device = device
        self.dtype = dtype
        
        # Pre-compute the grid coordinates only once
        H, W = grid_shape
        i_coords = torch.arange(H, device=device, dtype=dtype) + 0.5
        j_coords = torch.arange(W, device=device, dtype=dtype) + 0.5
        y_grid, x_grid = torch.meshgrid(i_coords, j_coords, indexing='ij')
        
        # Store world coordinates
        self.x_world = x_grid * pixel_size  # shape (H, W)
        self.y_world = y_grid * pixel_size  # shape (H, W)
        
    # @torch.jit.script
    def generate_observations(
        self, 
        object_positions: torch.Tensor,      # shape (num_envs, num_obj, 2)
        target_area_position: torch.Tensor,  # shape (num_envs, 2)
        end_effector_position: torch.Tensor, # shape (num_envs, 2) # End effector has sides 0.01 and 0.05
        object_orientations: torch.Tensor,   # shape (num_envs, num_obj)
        end_effector_orientation: torch.Tensor, # shape (num_envs,) - orientation in radians
        target_radius: float = 0.1,
        object_size: float = 0.01,
        end_effector_length: float = 0.05,   # Long side
        end_effector_width: float = 0.01     # Short side
    ) -> torch.Tensor:
        """Generate 3-channel grid observations for multiple environments.
        
        Returns:
            observations: torch.Tensor of shape (num_envs, 3, H, W)
                - Channel 0: Target area
                - Channel 1: Robot end-effector
                - Channel 2: Objects
        """
        num_envs = object_positions.shape[0]
        H, W = self.grid_shape
        # observations = torch.zeros((num_envs, 3, H, W), device=self.device)
        observations = torch.zeros((num_envs, 3, H, W), device=self.device, dtype=torch.bool)
        
        # -----------------------------------
        # Channel 0: Target area (vectorized)
        # -----------------------------------
        # Reshape for broadcasting: [num_envs, 1, 1, 2]
        target_pos = target_area_position.view(num_envs, 1, 1, 2)
        
        # Subtract from grid coordinates (broadcasting):
        # [num_envs, H, W, 2] - [num_envs, 1, 1, 2] -> [num_envs, H, W, 2]
        xy_world = torch.stack([self.x_world, self.y_world], dim=-1)
        target_diff = xy_world.unsqueeze(0) - target_pos
        
        # Calculate distance for all environments in parallel
        target_dist = torch.norm(target_diff, dim=-1)  # [num_envs, H, W]
        
        # Target mask is where distance <= target_radius
        observations[:, 0] = (target_dist <= target_radius).float() * 3
        
        # -----------------------------------
        # Channel 1: End-effector as oriented rectangle
        # -----------------------------------
        # Reshape for broadcasting: [num_envs, 1, 1, 2]
        ee_pos = end_effector_position.view(num_envs, 1, 1, 2)
        # Reshape orientation: [num_envs, 1, 1]
        ee_orient = end_effector_orientation.view(num_envs, 1, 1)
        
        # Calculate differences to each end effector
        ee_diff = xy_world.unsqueeze(0) - ee_pos  # [num_envs, H, W, 2]
        
        # Compute rotation for end effectors
        cos_theta = torch.cos(ee_orient)
        sin_theta = torch.sin(ee_orient)
        
        # Rotate coordinates for end effector frame
        dx_ee = ee_diff[..., 0]
        dy_ee = ee_diff[..., 1]
        
        # Transform to local coordinates (apply inverse rotation)
        x_local_ee = cos_theta * dx_ee + sin_theta * dy_ee
        y_local_ee = -sin_theta * dx_ee + cos_theta * dy_ee
        
        # Check if points are inside the rectangular end effector
        half_length = end_effector_length / 2
        half_width = end_effector_width / 2
        
        inside_ee = (
            (x_local_ee >= -half_length) & (x_local_ee <= half_length) &
            (y_local_ee >= -half_width) & (y_local_ee <= half_width)
        )
        
        observations[:, 1] = inside_ee.float() * 2
        
        # -----------------------------------
        # Channel 2: Objects (vectorized)
        # -----------------------------------
        num_obj = object_positions.shape[1]
        
        # Reshape for proper broadcasting:
        # Objects: [num_envs, num_obj, 1, 1, 2]
        obj_pos = object_positions.view(num_envs, num_obj, 1, 1, 2)
        # Orientations: [num_envs, num_obj, 1, 1]
        obj_orient = object_orientations.view(num_envs, num_obj, 1, 1)
        
        # Coordinates: [1, 1, H, W, 2] (will broadcast to [num_envs, num_obj, H, W, 2])
        coords = xy_world.view(1, 1, H, W, 2) 
        
        # Calculate differences to each object
        obj_diff = coords - obj_pos  # [num_envs, num_obj, H, W, 2]
        
        # Compute rotation for all objects
        cos_theta = torch.cos(obj_orient)
        sin_theta = torch.sin(obj_orient)
        
        # Apply rotations - extract x and y components for clarity
        dx = obj_diff[..., 0]
        dy = obj_diff[..., 1]
        
        # Rotate coordinates
        x_local = cos_theta * dx + sin_theta * dy
        y_local = -sin_theta * dx + cos_theta * dy
        
        # Check if grid points are inside any object
        half_side = object_size / 2
        inside = (
            (x_local >= -half_side) & (x_local <= half_side) &
            (y_local >= -half_side) & (y_local <= half_side)
        )
        
        # Aggregate across all objects (for each environment)
        observations[:, 2] = inside.any(dim=1).float()
        
        return observations.to(dtype=self.dtype)
    

if __name__ == "__main__":  

    grid = GridObservationGenerator(grid_shape=(128,128),dtype=torch.float16)

    num_envs = 100
    num_obj = 100

    obj_pos = torch.rand((num_envs, num_obj, 2), device='cuda') * 0.65
    target_area = torch.rand((num_envs, 2), device='cuda') * 0.65
    EE_pos = torch.rand((num_envs, 2), device='cuda') *0.65
    obj_yaw = torch.rand((num_envs,num_obj), device='cuda')
    EE_yaw = torch.rand((num_envs,), device='cuda')

    for i in range(100):
        with Timer("create_multi_cube_mask"):
            mask = grid.generate_observations(obj_pos, target_area, EE_pos, obj_yaw, EE_yaw)

    print(mask.shape)

    # for i in range(10):
    #     mask1 = mask[i]
    #     mask1 = mask1.sum(dim=0)
    #     plt.imshow(mask1.cpu().numpy(), cmap='hot')
    #     plt.axis('off')
    #     plt.savefig(f'cube_mask_{i}.png', bbox_inches='tight', pad_inches=0)
    #     plt.show()
        # plt.waitforbuttonpress()


    # plt.imshow(mask1.cpu().numpy(), cmap='hot')
    # plt.axis('off')
    # plt.savefig('cube_mask.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    # # If you need a 3-channel observation, you can simply stack the mask into three channels:
    # mask_3ch = mask.unsqueeze(0).repeat(3, 1, 1)  # shape: (3, H, W)
