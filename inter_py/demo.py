import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Assuming you have two numpy arrays, figure1 and figure2
figure1 = np.random.rand(100, 100)
figure2 = np.random.rand(100, 100)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2)

# Plot the first figure on the first subplot
axs[0].imshow(figure1, cmap='gray')  # 'gray' colormap for grayscale images
axs[0].set_title('Figure 1')

# Plot the second figure on the second subplot
axs[1].imshow(figure2, cmap='gray')  # 'gray' colormap for grayscale images
axs[1].set_title('Figure 2')

# Adjust layout to prevent overlapping of titles and axes
plt.tight_layout()

# Render the figure on a canvas
canvas = FigureCanvas(fig)

# Convert the canvas to a numpy array
canvas.draw()
width, height = fig.get_size_inches() * fig.get_dpi()
# image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
data = np.array(canvas.buffer_rgba())
print(data.shape)
print(data.dtype)
print(data)
# Now image_np contains the numpy array representing the figure with subplots

# Optionally, you can save the image to a file as well
plt.savefig('subfigures.png')

# If you don't want to save the image, you can close the figure to release resources
plt.close()

# Now image_np contains the numpy array representing the figure with subplots
