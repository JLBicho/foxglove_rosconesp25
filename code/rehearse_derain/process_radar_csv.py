import os

import foxglove
from foxglove.schemas import (
    Pose,
    Vector3,
    Quaternion,
    Color,
    SpherePrimitive,
    SceneEntity,
    SceneUpdate,
    Timestamp
)

from foxglove.channels import SceneUpdateChannel

from utils import open_csv, get_metadata_from_folder, generate_palette


def generate_spheres(position, intensity, cross_section, speed, confidence):
    """
    Generates a SceneEntity for the radar visualization.
    Args:
        position (tuple): Position of the entity in the scene.
        size (tuple): Size of the entity in the scene. Radius of cylinder.
        speed (float): Speed of the entity. Length of cylinder.
        confidence (float): Confidence level of the detection. Color of cylinder.
    Returns:
        SceneEntity: A SceneEntity object representing the radar data.
    """

    r, g, b = generate_palette(confidence)
    sphere = SpherePrimitive(
        pose=Pose(
            position=Vector3(
                x=position[0], y=position[1], z=position[2]
            ),
            orientation=Quaternion(x=0, y=0, z=0, w=1),
        ),
        size=Vector3(
            x=speed, y=intensity, z=cross_section),
        color=Color(r=r, g=g, b=b, a=1),
    )

    return sphere


def generate_radar_viz(folder, topic_name="/radar") -> None:
    """
    Generates a radar visualization topic from the radar images folder.
    This function reads the radar images and generates a CompressedImage topic.
    """

    print(f"RADAR viz from: {folder} with topic {topic_name}")

    radar_scene_update = SceneUpdateChannel(topic_name)

    metadata = get_metadata_from_folder(folder)

    for i, data in enumerate(metadata):
        filename = os.path.basename(data["Filename"])
        file_path = os.path.join(folder, filename)
        stamp = int(data["Stamp"])
        sec = int(stamp / 1e9)
        nsec = int(stamp % 1e9)
        if os.path.exists(file_path):
            get_csv_data, _ = open_csv(file_path)
            intensity_max = -1e3
            cross_section_max = -1e3
            velocity_max = -1e3
            confidence_max = -1e3
            confidence_min = 1e3
            spheres = []

            for row in get_csv_data:
                intensity = float(row["INTENSITY"])
                cross_section = abs(float(row["CROSS_SECTION"]))
                velocity = float(row["VELOCITY"])
                confidence = float(row["CONF"])
                intensity_max = max(intensity_max, intensity)
                cross_section_max = max(cross_section_max, cross_section)
                velocity_max = max(velocity_max, velocity)
                confidence_max = max(confidence_max, confidence)
                confidence_min = min(confidence_max, confidence)

            # print(confidence_max, confidence_min)
            for row in get_csv_data:
                position = (
                    float(row["X"]),
                    float(row["Y"]),
                    float(row["Z"]),
                )
                intensity = float(row["INTENSITY"]) / intensity_max
                cross_section = abs(
                    float(row["CROSS_SECTION"])) / cross_section_max
                speed = float(row["VELOCITY"]) / velocity_max
                confidence = (
                    float(row["CONF"])-confidence_min) / (confidence_max-confidence_min)

                sphere = generate_spheres(
                    position, intensity, cross_section, speed, confidence)

                spheres.append(sphere)

            timestamp = Timestamp(sec=sec, nsec=nsec)
            scene_entity = SceneEntity(
                timestamp=timestamp,
                id="radar",
                frame_id="base",
                spheres=spheres,
            )

            scene_update = SceneUpdate(
                entities=[scene_entity],
            )

            radar_scene_update.log(scene_update, log_time=stamp)

            print(f"Processed radar {i + 1}/{len(metadata)}: {filename}")


if __name__ == "__main__":
    writer = foxglove.open_mcap("radar.mcap", allow_overwrite=True)
    generate_radar_viz(".")
