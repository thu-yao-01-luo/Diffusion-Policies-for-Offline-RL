import numpy as np
import matplotlib.pyplot as plt

vector_array = np.random.rand(30, 30, 2)
heatmap_array = np.random.rand(30, 30)

# Assuming you have a vector array of shape (30, 30, 2) called "vector_array"
# Generate a grid of x, y coordinates
x, y = np.meshgrid(np.arange(30), np.arange(30))

# Get the x and y components of the vectors
u = vector_array[:, :, 0]
v = vector_array[:, :, 1]

# Create a quiver plot to show the distribution of the vectors
plt.figure()
plt.quiver(x, y, u, v, scale=10)
plt.imshow(heatmap_array, cmap='hot', alpha=0.5, extent=(0, 30, 0, 30), origin='lower')
# Set axis labels and plot title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Vector Distribution')

# Show the plot
# plt.show()
plt.savefig('vector_distribution.png')
