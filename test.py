import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2

size = 100   # grid size
scale = 0.1  # smaller = smoother terrain

x_coords = np.linspace(0, size * scale, size)
y_coords = np.linspace(0, size * scale, size)
X, Y = np.meshgrid(x_coords, y_coords)
Z = np.zeros_like(X)

for i in range(size):
    for j in range(size):
        Z[i, j] = pnoise2(X[i, j], Y[i, j])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain', linewidth=0, antialiased=False)

ax.set_title("Perlin Noise as a 3D Surface (z = pnoise2(x, y))")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Perlin value")
plt.show()
