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


ROOT_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "output")

LIVE_STREAM = False
RECORD_MCAP = True

if LIVE_STREAM:
    server = foxglove.start_server()

if RECORD_MCAP:
    mcap_file = open_mcap(os.path.join(
        OUTPUT_FOLDER, "irdn.mcap"), allow_overwrite=True)


def euler_to_quaternion(yaw, pitch, roll):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]


def convert_to_laserscan(record):
    scan = LaserScan(
        frame_id="laser_frame",
        start_angle=min(record["angles"]),
        end_angle=max(record["angles"]),
        ranges=record["dists"]
    )
    return scan


def convert_to_poseinframe(record):
    q = euler_to_quaternion(record["pose"]["theta"], 0, 0)
    pose = PoseInFrame(
        frame_id="odom",
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
    return pose, tf


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


files = os.listdir(DATA_FOLDER)
n_files = len(files)
t = 0
for n in range(1, n_files+1):
    print(f"Processing file {n} of {n_files}")
    file = f"{n}.json"
    filepath = os.path.join(DATA_FOLDER, file)
    with open(filepath, 'r', encoding="utf-8") as f:
        data = json.load(f)
    for i in range(data["num_records"]):
        print(f"  Record {i+1} of {data['num_records']}")
        timestamp = Timestamp(sec=int(t))
        laser_channel.log(convert_to_laserscan(
            data["data"][i]), log_time=int(t*1e8))
        pose_channel.log(convert_to_poseinframe(
            data["data"][i])[0], log_time=int(t*1e8))
        tf_channel.log(convert_to_poseinframe(
            data["data"][i])[1], log_time=int(t*1e8))
        brake_channel.log(
            {"value": data["data"][i]["brake"]}, log_time=int(t*1e8))
        horn_channel.log(
            {"value": data["data"][i]["horn"]}, log_time=int(t*1e8))
        direction_channel.log(
            {"value": data["data"][i]["direction"]}, log_time=int(t*1e8))
        file_channel.log({"value": file}, log_time=int(t*1e8))
        speed_channel.log(
            {"left": float(data["data"][i]["counts_left"]), "right": float(data["data"][i]["counts_right"])}, log_time=int(t*1e8))
        t += 1

        if LIVE_STREAM:
            time.sleep(0.1)
