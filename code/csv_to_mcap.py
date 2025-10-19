import os
import csv
import numpy as np
import math

import foxglove
from foxglove import Channel
from foxglove.schemas import (
    Pose,
    PoseInFrame,
    Vector3,
    Quaternion,
    Timestamp,
    PosesInFrame,
    KeyValuePair,
    LaserScan,
    FrameTransform,
    LinePrimitive,
    SceneEntity,
    SceneUpdate,
    Color,
    Point3
)
from foxglove.channels import (
    PoseInFrameChannel,
    PosesInFrameChannel,
    KeyValuePairChannel,
    LaserScanChannel,
    FrameTransformChannel,
    SceneUpdateChannel
)

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_FILE = os.path.join(ROOT_DIR, "data", "dirnd_csv",
                        "autonomous_navigation_dataset.csv")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "autonomous_navigation_dataset.mcap")


COLOR_BY_RESULT = {
    "collision": (1.0, 0, 0),  # Red
    "success": (0, 1.0, 0),  # Green
    "partial": (0, 0, 1.0),  # Blue
    "timeout": (1.0, 1.0, 0),  # Yellow
}


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert roll, pitch, yaw angles to quaternion."""

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]


def open_csv(file_path: str) -> list[dict]:
    """
        Open a CSV file and return its contents as a list of dictionaries
        Input:
            file_path (str): Path to the CSV file
        Output:
            list[dict]: List of dictionaries representing the CSV rows

    """
    with open(file_path, "r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        fields = csv_reader.fieldnames
        data = list(csv_reader)
    print(f"Available data fields: {fields}")
    return data


def get_pose_and_speed(timestamp, row, prev_ts, prev_position) -> None:
    ts = timestamp.sec
    # Create the messages starting from the basic ones
    position = Vector3(
        x=float(row["x"]),
        y=float(row["y"]))
    # Calculate orientation based on previous position
    yaw = math.atan2(
        float(row["y"]) - prev_position[1],
        float(row["x"]) - prev_position[0])
    quat = rpy_to_quaternion(
        roll=0,
        pitch=0,
        yaw=yaw)
    orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    # Calculate speed based on traveled distance divided by time taken
    if ts > 0:
        speed = math.sqrt(
            (float(row["x"]) - prev_position[0]) ** 2 +
            (float(row["y"]) - prev_position[1]) ** 2) / (ts-prev_ts)
    # First instance or if ts is 0, set speed to 0
    else:
        speed = 0.0

    pose = Pose(
        position=position,
        orientation=orientation
    )

    frame = FrameTransform(
        timestamp=timestamp,
        parent_frame_id="odom",
        child_frame_id=f"id_{row['run_id']}",
        translation=position,
        rotation=orientation
    )

    return pose, speed, frame


def get_line_scene(row, pts: list) -> tuple[SceneEntity, list]:
    pt = Point3(
        x=float(row["x"]),
        y=float(row["y"]),
        z=0
    )

    if row["timestamp"] == '0':
        pts[row['run_id']].clear()
    pts[row['run_id']].append(pt)

    color = COLOR_BY_RESULT[row["target"]]
    line = LinePrimitive(
        points=pts[row['run_id']],
        color=Color(r=color[0], g=color[1], b=color[2], a=1.0),
        thickness=0.05
    )

    line_entitity = SceneEntity(lines=[line],
                                id=f"id_{row['run_id']}",
                                frame_id="odom",
                                metadata=[KeyValuePair(
                                    key="target",
                                    value=row["target"]),
                                    KeyValuePair(
                                    key="time_taken",
                                    value=row["time_taken"])])

    return line_entitity, pts


def generate_mcap() -> None:
    writer = foxglove.open_mcap(
        OUTPUT_FILE, allow_overwrite=True)
    data = open_csv(CSV_FILE)

    # A channel will be created for each run_id
    pose_channels = {}
    poses = {}
    poses_channels = {}
    battery_channels = {}
    path_smoothness_channels = {}
    laser_channels = {}
    ultrasound_channels = {}
    frames_channels = {}
    pts = {}
    speed_channel = {}

    # Initialize channels for each run_id
    for row in data:
        pose_channels[row['run_id']] = PoseInFrameChannel(
            f"{row['run_id']}/pose")
        poses_channels[row['run_id']] = PosesInFrameChannel(
            f"{row['run_id']}/path")
        poses[row['run_id']] = []
        battery_channels[row['run_id']] = Channel(
            f"{row['run_id']}/battery")
        path_smoothness_channels[row['run_id']] = KeyValuePairChannel(
            f"{row['run_id']}/path_smoothness")
        laser_channels[row['run_id']] = LaserScanChannel(
            f"{row['run_id']}/laser")
        ultrasound_channels[row['run_id']] = LaserScanChannel(
            f"{row['run_id']}/ultrasound")
        frames_channels[row['run_id']] = FrameTransformChannel(
            f"{row['run_id']}/tf")
        speed_channel[row['run_id']] = Channel(
            f"{row['run_id']}/speed")

        pts[row['run_id']] = []

    target_result_channels = Channel("/target_result")
    time_taken_channel = Channel("/time_taken")

    scene_channels = {}
    scene_entities = {}
    results_count = {}
    results_channel = Channel("/results")
    possible_results = set(row['target'] for row in data)

    for result in possible_results:
        scene_entities[result] = []
        scene_channels[result] = SceneUpdateChannel(f"{result}/scene")
        results_count[result] = 0

    target_results = []
    prev_id = '0'
    prev_position = (0, 0)
    prev_ts = 0

    # Iterate through the data and create the necessary messages
    for row in data:
        # Get timestamp as float and convert to Timestamp
        ts = float(row["timestamp"])
        timestamp = Timestamp(sec=0, nsec=0).from_epoch_secs(ts)

        # Clear previous data if 'run_id' has changed
        if prev_id != row['run_id']:
            print("Clearing previous data for new run_id")
            scene_entities[row["target"]].clear()
            prev_position = (0, 0)

        # Get the pose, speed and frame for the current row
        pose, speed, frame = get_pose_and_speed(
            timestamp, row, prev_ts, prev_position)
        print(
            f"Run ID: {row['run_id']}, Timestamp: {ts}, Position: ({row['x']}, {row['y']}), Speed: {speed:.2f} m/s")

        # Log the pose, poses (path), frame and speed
        pose_stamped = PoseInFrame(
            pose=pose, frame_id="odom", timestamp=timestamp)
        pose_channels[row['run_id']].log(pose_stamped, log_time=int(ts*1e9))

        poses[row['run_id']].append(pose)
        poses_frame = PosesInFrame(
            timestamp=timestamp, frame_id="odom", poses=poses[row['run_id']])
        poses_channels[row['run_id']].log(poses_frame, log_time=int(ts*1e9))

        speed_channel[row['run_id']].log(
            {"speed": speed}, log_time=int(ts*1e9))

        frames_channels[row['run_id']].log(frame, log_time=int(ts*1e9))

        # Create the scene entity for the current row
        scene_entity, pts = get_line_scene(row, pts)

        # Append, log the scene entity and update the scene
        scene_entities[row["target"]].append(scene_entity)
        scene_channels[row["target"]].log(SceneUpdate(entities=scene_entities[row["target"]]),
                                          log_time=int(ts*1e9))

        # Log other data

        # Battery as JSONSchema directly
        battery = {"battery": row["battery_level"]}
        battery_channels[row["run_id"]].log(battery, log_time=int(ts*1e9))

        # Time taken as JSONSchema with X value 'run_id' and Y value 'time_taken'
        time_taken_channel.log(
            {"run_id": int(row["run_id"]),
             "time_taken": int(row["time_taken"])}, log_time=int(ts*1e9))

        # Path smoothness as KeyValuePair
        path_smoothness = KeyValuePair(
            key="path_smoothness", value=row["path_smoothness"])
        path_smoothness_channels[row["run_id"]].log(
            path_smoothness, log_time=int(ts*1e9))

        # Ultrasound scan as LaserScan in angles -90 and 90 degrees
        ultrasound_scan = LaserScan(
            timestamp=timestamp,
            frame_id=f"id_{row['run_id']}",
            start_angle=-3.141592/2,
            end_angle=+3.141592/2,
            ranges=[float(row["ultrasonic_right"]),
                    float(row["ultrasonic_left"])],
            intensities=[0, 100],
        )
        ultrasound_channels[row["run_id"]].log(
            ultrasound_scan, log_time=int(ts*1e9))

        # Register the results and count each result
        target_results.append(row["target"])
        results_count[row["target"]] += 1

        # Store the previous id, timestamp and position
        prev_id = row['run_id']
        prev_ts = ts
        prev_position = (float(row["x"]), float(row["y"]))

    target_result_channels.log(
        {"results": target_results}, log_time=int(0*1e9))
    results_channel.log(
        results_count, log_time=int(0*1e9))

    writer.close()


if __name__ == "__main__":

    generate_mcap()
