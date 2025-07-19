import numpy as np

def rotate_vector(v: np.ndarray, theta: float) -> np.ndarray:
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return rotation_matrix @ v  # matrix multiply

# Usage:
v = np.array([1, 0])
angle = np.pi / 6  # 30 degrees in radians

rotated_v = rotate_vector(v, angle)
print(rotated_v)  # [0.866..., 0.5]
