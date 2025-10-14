import json
import time
import os

import numpy as np

import foxglove

from foxglove.schemas import (
    LaserScan, PoseInFrame, Pose, Vector3, FrameTransform, Quaternion, Timestamp
)
from foxglove.channels import LaserScanChannel, PoseInFrameChannel, FrameTransformChannel
from foxglove import Channel, open_mcap


# Define folders
ROOT_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data", "irnd_json")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "output")
N_FILES = len(os.listdir(DATA_FOLDER))

# Options
LIVE_STREAM = False
RECORD_MCAP = True


if LIVE_STREAM:
    server = foxglove.start_server()

if RECORD_MCAP:
    mcap_file = open_mcap(os.path.join(
        OUTPUT_FOLDER, "irdn.mcap"), allow_overwrite=True)

# Define schemas and channels
bool_schema = {
    "type": "object",
    "properties": {
        "value": {"type": "boolean"}
    }
}

text_schema = {
    "type": "object",
    "properties": {
        "value": {"type": "string"}
    }
}

speed_schema = {
    "type": "object",
    "properties": {
        "left": {"type": "number"},
        "right": {"type": "number"}
    }
}

laser_channel = LaserScanChannel(topic="laser_scan")
pose_channel = PoseInFrameChannel(topic="pose")
tf_channel = FrameTransformChannel(topic="tf")
brake_channel = Channel(topic="brake", schema=bool_schema)
horn_channel = Channel(topic="horn", schema=bool_schema)
direction_channel = Channel(topic="direction", schema=text_schema)
speed_channel = Channel(topic="speed", schema=speed_schema)
file_channel = Channel(topic="file", schema=speed_schema)


def euler_to_quaternion(roll, pitch, yaw) -> list:
    """ Convert Euler angles to Quaternion """

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]


def get_laserscan(record, timestamp=None) -> LaserScan:
    """ Convert a record to a LaserScan message """
    scan = LaserScan(
        frame_id="laser_frame",
        timestamp=timestamp,
        start_angle=min(record["angles"]),
        end_angle=max(record["angles"]),
        ranges=record["dists"]
    )
    return scan


def get_pose_tf_timestamp(record) -> tuple[PoseInFrame, FrameTransform, Timestamp]:
    """ Convert a record to a PoseInFrame, FrameTransform and Timestamp """
    q = euler_to_quaternion(0, 0, record["pose"]["theta"])
    stamp = int(record["pose"]["stamp"])
    timestamp = Timestamp(sec=int(stamp//1e9), nsec=int(stamp % 1e9))
    pose = PoseInFrame(
        frame_id="odom",
        timestamp=timestamp,
        pose=Pose(
            position=Vector3(
                x=record["pose"]["x"],
                y=record["pose"]["y"],
            ),
            orientation=Quaternion(
                x=q[0],
                y=q[1],
                z=q[2],
                w=q[3]
            )
        )
    )
    tf = FrameTransform(
        child_frame_id="laser_frame",
        parent_frame_id="odom",
        timestamp=timestamp,
        translation=Vector3(
            x=record["pose"]["x"],
            y=record["pose"]["y"],
        ),
        rotation=Quaternion(
            x=q[0],
            y=q[1],
            z=q[2],
            w=q[3]
        )
    )
    return pose, tf, timestamp


def main() -> None:
    # Iterate over files
    for n in range(1, N_FILES+1):
        print(f"Processing file {n} of {N_FILES}")
        prev_ts_ns = 0
        file = f"{n}.json"
        filepath = os.path.join(DATA_FOLDER, file)
        with open(filepath, 'r', encoding="utf-8") as f:
            data = json.load(f)
        # Iterate over records
        for i in range(data["num_records"]):
            print(f"  Record {i+1} of {data['num_records']}")

            pose, tf, timestamp = get_pose_tf_timestamp(data["data"][i])
            laser_scan = get_laserscan(
                data["data"][i], timestamp=timestamp)

            print(f"    Timestamp: {timestamp}")

            ts_ns = int(timestamp.sec*1e9 + timestamp.nsec)
            if prev_ts_ns == 0:
                prev_ts_ns = ts_ns

            laser_channel.log(laser_scan, log_time=ts_ns)
            pose_channel.log(pose, log_time=ts_ns)
            tf_channel.log(tf, log_time=ts_ns)

            brake_channel.log(
                {"value": data["data"][i]["brake"]}, log_time=ts_ns)
            horn_channel.log(
                {"value": data["data"][i]["horn"]}, log_time=ts_ns)
            direction_channel.log(
                {"value": data["data"][i]["direction"]}, log_time=ts_ns)
            file_channel.log({"value": file}, log_time=ts_ns)
            speed_channel.log(
                {"left": float(data["data"][i]["counts_left"]),
                 "right": float(data["data"][i]["counts_right"])}, log_time=ts_ns)

            if LIVE_STREAM:
                time.sleep((ts_ns-prev_ts_ns) / 1e9)

            prev_ts_ns = ts_ns


if __name__ == "__main__":
    main()
