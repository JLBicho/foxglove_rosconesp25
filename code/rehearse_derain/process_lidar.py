import os

import numpy as np

import foxglove
from foxglove.channels import (
    PointCloudChannel,
)
from foxglove.schemas import (
    Timestamp,
    PointCloud,
    PackedElementField,
    PackedElementFieldNumericType
)

from utils import get_metadata_from_folder

f32 = PackedElementFieldNumericType.Float32
u32 = PackedElementFieldNumericType.Uint32


def generate_lidar_viz(folder, topic_name="/lidar", fields=None) -> None:
    """
    Generates a PointCloud topic from a folder of bin files.
    Args:
        folder (str): Path to the folder containing bin files.
        topic_name (str): Name of the topic to create.
    """
    print(f"LiDAR viz from: {folder} with topic {topic_name}")

    # Define the channel for compressed images
    channel = PointCloudChannel(topic=topic_name)

    metadata = get_metadata_from_folder(folder)

    if fields is None:
        fields = [
            PackedElementField(name="x", offset=0, type=f32),
            PackedElementField(name="y", offset=4, type=f32),
            PackedElementField(name="z", offset=8, type=f32),
            PackedElementField(name="intensity", offset=12, type=u32),
        ]

    else:
        fields = [PackedElementField(name=field, offset=4*i, type=f32)
                  for i, field in enumerate(fields)]

    for i, data in enumerate(metadata):
        filename = os.path.basename(data["Filename"])
        file_path = os.path.join(folder, filename)
        stamp = int(data["Stamp"])
        sec = int(stamp / 1e9)
        nsec = int(stamp % 1e9)
        if os.path.exists(file_path):
            timestamp = Timestamp(sec=sec, nsec=nsec)
            bin_pcd = np.fromfile(file_path, dtype=np.float32)

            pc = PointCloud(
                timestamp=timestamp,
                frame_id="base",
                # 4 floats per point (x, y, z, intensity)
                point_stride=4*len(fields),
                fields=fields,
                data=bin_pcd.tobytes(),  # Convert to bytes for PointCloud
            )

            channel.log(
                pc,
                log_time=stamp,
            )
            print(f"Processed point cloud {i + 1}/{len(metadata)}: {filename}")


if __name__ == "__main__":
    print("Generating LiDAR visualization...")
    writer = foxglove.open_mcap("radar_pcd.mcap", allow_overwrite=True)
    generate_lidar_viz(
        "/Volumes/SandiskSSD/JLMV/Foxglove/REHEARSE RAIN/data/000/radar_pcd", "/radar_pcd")
