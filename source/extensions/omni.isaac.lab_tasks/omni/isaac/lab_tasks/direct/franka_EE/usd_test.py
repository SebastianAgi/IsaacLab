from pxr import Usd
#disable warnings
import warnings

# # Custom filter to ignore specific warnings
# def custom_warning_filter(message, category, filename, lineno, file=None, line=None):
#     if "Could not open asset" in str(message):
#         return
#     return warnings.defaultaction
# warnings.filterwarnings("ignore")
# warnings.showwarning = custom_warning_filter

# usd_file_path = f"/home/sebastian/IsaacLab/USD_files/Beans.usd"

# # Load the USD stage
# stage = Usd.Stage.Open(usd_file_path)

# # Get the Particles Xform
# particles_prim = stage.GetPrimAtPath("/World/Particles")

# # Get all child prims inside Particles
# granules = [prim.GetPath().pathString for prim in particles_prim.GetChildren()]

# # Print the granules list
# print(len(granules))


## Inspecting the Franka Panda robot granular material pusher USD
usd_file_path = f"/home/sebastian/IsaacLab/frank_panda_usd/Granular_franka_2.usd"

# Load the USD stage
stage = Usd.Stage.Open(usd_file_path)

# Function to recursively print all prims in the stage
def print_all_prims(prim, indent=0):
    print(" " * indent + prim.GetPath().pathString)
    for child in prim.GetChildren():
        print_all_prims(child, indent + 2)

# Get the root prim
root_prim = stage.GetPseudoRoot()

# Print all prims starting from the root
print_all_prims(root_prim)