import os

from foxglove.channels import (
    CompressedImageChannel,
)
from foxglove.schemas import (
    Timestamp,
    CompressedImage,
)

from utils import get_metadata_from_folder


def generate_image_viz(folder, topic_name) -> None:
    """
    Generates a CompressedImage topic from a folder of images.
    Args:
        folder (str): Path to the folder containing images.
        topic_name (str): Name of the topic to create.
    """
    print(f"Image viz from: {folder} with topic {topic_name}")

    # Define the channel for compressed images
    channel = CompressedImageChannel(topic=topic_name)

    metadata = get_metadata_from_folder(folder)

    for i, data in enumerate(metadata):
        filename = os.path.basename(data["Filename"])
        image_path = os.path.join(folder, filename)
        stamp = int(data["Stamp"])
        sec = int(stamp / 1e9)
        nsec = int(stamp % 1e9)
        if os.path.exists(image_path):
            timestamp = Timestamp(sec=sec, nsec=nsec)
            with open(image_path, "rb") as file:
                img = file.read()
                data = img

            # Create a CompressedImage message
            compressed_image = CompressedImage(
                timestamp=timestamp,
                frame_id=topic_name,
                format="jpeg",
                data=data
            )
            channel.log(
                compressed_image,
                log_time=stamp,
            )
            print(f"Processed image {i + 1}/{len(metadata)}: {filename}")
