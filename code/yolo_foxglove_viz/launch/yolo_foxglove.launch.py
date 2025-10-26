from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    ld = []

    yolo_to_foxglove_node = Node(
        package='yolo_foxglove_viz',
        executable='yolo_to_foxglove',
        name='yolo_to_foxglove'
    )
    ld.append(yolo_to_foxglove_node)

    foxglove_bridge = ExecuteProcess(
        cmd=["ros2", "launch", "foxglove_bridge", "foxglove_bridge_launch.xml"])
    ld.append(foxglove_bridge)

    astra_camera = ExecuteProcess(
        cmd=["ros2", "launch", "astra_camera", "astra.launch.xml"])
    ld.append(astra_camera)

    return LaunchDescription(ld)
