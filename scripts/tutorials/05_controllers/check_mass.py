import numpy as np

arr = np.load("/home/Sebastian/IsaacLab/scripts/tutorials/05_controllers/output/camera/mass/000200.npy")
print("shape:", arr.shape)
print("min:", arr.min(), "max:", arr.max(), "mean:", arr.mean())
print("unique values:", np.unique(arr)[:20])
