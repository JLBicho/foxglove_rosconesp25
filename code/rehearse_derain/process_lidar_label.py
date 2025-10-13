import struct
import numpy as np

file_path = "/Volumes/SandiskSSD/JLMV/Foxglove/REHEARSE RAIN/data/000/radar_pcd/000000.bin"

# Read the binary file
with open(file_path, 'rb') as file:
    binary_data = file.read()

print(binary_data[:64])  # Display the first 64 bytes for inspection

# Convert binary data to a numpy array of integers
# Assuming each label is a 4-byte integer
labels = np.frombuffer(binary_data, dtype=np.int32)

# Display the first few labels
print(labels[:20])


# Convert binary data to a numpy array of floats
# Assuming each label is a 4-byte float
labels = np.frombuffer(binary_data, dtype=np.float32)

# Display the first few labels
print(labels[0:20])

data = struct.pack("ffff", labels[0], labels[1], labels[2], labels[3])

print(data)
