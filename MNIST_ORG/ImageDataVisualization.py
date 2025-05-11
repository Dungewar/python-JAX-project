import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from IDXFileReaders import read_idx3_ubyte, read_idx1_ubyte

# Load the data
current_dir = Path(__file__).parent
train_images = read_idx3_ubyte(current_dir / 'train-images.idx3-ubyte')
train_labels = read_idx1_ubyte(current_dir / 'train-labels.idx1-ubyte')

# Print basic information
print(f"Image data shape: {train_images.shape}")
print(f"Label data shape: {train_labels.shape}")

# Display a grid of 5x5 random images
plt.figure(figsize=(10, 10))
for i in range(25):
    idx = np.random.randint(0, len(train_images))
    plt.subplot(5, 5, i + 1)
    plt.imshow(train_images[idx], cmap='gray')
    plt.title(f"Digit: {train_labels[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Show a single image in more detail
idx = np.random.randint(0, len(train_images))
plt.figure(figsize=(8, 8))
plt.imshow(train_images[idx], cmap='gray')
plt.colorbar()  # Show color scale
plt.title(f'Label: {train_labels[idx]}')
plt.show()

# Print pixel values for a small section
print("\nSample pixel values:")
print(train_images[idx, :28, :28])