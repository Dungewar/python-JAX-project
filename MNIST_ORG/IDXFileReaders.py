from pathlib import Path
import numpy as np


def read_idx3_ubyte(filename):
    """Read IDX3 file format (images)"""
    with open(filename, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')  # Magic number (should be 2051)
        n_images = int.from_bytes(f.read(4), 'big')  # Number of images
        n_rows = int.from_bytes(f.read(4), 'big')  # Number of rows
        n_cols = int.from_bytes(f.read(4), 'big')  # Number of columns

        # Read image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        return image_data.reshape(n_images, n_rows, n_cols)


def read_idx1_ubyte(filename):
    """Read IDX1 file format (labels)"""
    with open(filename, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')  # Magic number (should be 2049)
        n_items = int.from_bytes(f.read(4), 'big')  # Number of items

        # Read label data
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        return label_data


# Get current directory and construct paths
data_dir = Path(__file__).parent  # relative to where you run the script
train_images_path = data_dir / 'train-images.idx3-ubyte'
train_labels_path = data_dir / 'train-labels.idx1-ubyte'

# Read images with idx3 reader
train_images = read_idx3_ubyte(train_images_path)  # Use idx3 for images

# Read labels with idx1 reader
train_labels = read_idx1_ubyte(train_labels_path)  # Use idx1 for labels
