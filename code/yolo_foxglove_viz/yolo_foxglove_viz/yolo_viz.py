import colorsys
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Pose, Point
from yolo_msgs.msg import DetectionArray, BoundingBox3D
from foxglove_msgs.msg import SceneEntity, SceneUpdate, CubePrimitive, TextPrimitive


def value_to_rgb(value: float) -> tuple[float, float, float]:
    """
    Map a float in [0, 6] to an RGB color.
    0 -> cold blue, 3 -> green, 6 -> red.
    Returns (r, g, b) as floats in [0,1] by default or ints 0..255 if as_int=True.
    """

    # clamp input
    v = max(0.0, min(6.0, float(value)))

    # Map value linearly to hue degrees: 0 -> 0 (red), 3 -> 120 (green), 6 -> 240 (blue)
    hue_deg = 40.0 * v
    h = (hue_deg % 360.0) / 360.0

    s = 1.0   # full saturation for vivid colors
    val = 1.0  # full brightness

    r, g, b = colorsys.hsv_to_rgb(h, s, val)

    return (r, g, b)


def distance3d(x, y, z) -> float:
    return (x**2 + y**2 + z**2)**0.5


def _to_tuple(q: Quaternion):
    return (q.x, q.y, q.z, q.w)


def _from_tuple(t):
    q = Quaternion()
    q.x, q.y, q.z, q.w = t
    return q


def _quaternion_multiply(a, b):
    """
    Multiply two quaternions a * b. Inputs are (x,y,z,w) tuples.
    Returns (x,y,z,w).
    """
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return (x, y, z, w)


def _quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """
    Create quaternion (x,y,z,w) from roll, pitch, yaw (radians).
    Uses standard Tait-Bryan ZYX (yaw-pitch-roll) convention internally to build
    the quaternion for the combined rotation.
    """

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    w = cr*cp*cy + sr*sp*sy
    return (x, y, z, w)


def rotate_quaternion_by_rpy(q: Quaternion, roll: float = 0.0, pitch: float = 0.0,
                             yaw: float = 0.0, post_multiply: bool = True) -> Quaternion:
    """
    Rotate a geometry_msgs.msg.Quaternion by given roll, pitch, yaw (radians).
    - q: original orientation (geometry_msgs.msg.Quaternion)
    - roll, pitch, yaw: rotation angles to apply
    - post_multiply:
        True (default): applies rotation in the local frame of q (q_out = q * q_rpy)
        False: applies rotation in the global frame (q_out = q_rpy * q)

    Returns a new geometry_msgs.msg.Quaternion.
    """
    q_tuple = _to_tuple(q)
    rpy_q = _quaternion_from_euler(roll, pitch, yaw)

    if post_multiply:
        out = _quaternion_multiply(q_tuple, rpy_q)
    else:
        out = _quaternion_multiply(rpy_q, q_tuple)

    return _from_tuple(out)
#


class YoloToFoxgloveNode(Node):
    def __init__(self):
        super().__init__('yolo_to_foxglove')
        self.pub = self.create_publisher(
            SceneUpdate, 'foxglove/scene_entities', 10)
        self.sub = self.create_subscription(
            DetectionArray, 'yolo/detections_3d', self.cb_detections, 10)
        self.get_logger().info('yolo_to_foxglove node started')
        self.scene_update = SceneUpdate()

    def get_dist_color(self, bbox3d: BoundingBox3D) -> tuple[float, float, float, float]:
        dist = distance3d(
            bbox3d.center.position.x,
            bbox3d.center.position.y,
            bbox3d.center.position.z)
        r, g, b = value_to_rgb(dist)
        return dist, r, g, b

    def detection_to_viz(self, detection) -> CubePrimitive:
        class_name = detection.class_name
        score = str(round(detection.score, 2))
        bbox3d = detection.bbox3d

        cube = CubePrimitive()
        # set cube pose from bbox3d center
        cube.pose = Pose()
        cube.pose.position = Point(
            x=bbox3d.center.position.x,
            y=bbox3d.center.position.y,
            z=bbox3d.center.position.z
        )
        # set cube size from bbox3d dimensions
        cube.size.x = bbox3d.size.x
        cube.size.y = bbox3d.size.y
        cube.size.z = bbox3d.size.z

        dist, r, g, b = self.get_dist_color(bbox3d)

        cube.color.r = r
        cube.color.g = g
        cube.color.b = b
        cube.color.a = 0.5

        label = TextPrimitive()
        label.text = class_name
        label.font_size = 0.05*dist
        label.color.r = r
        label.color.g = g
        label.color.b = b
        label.color.a = 0.5
        label.pose = Pose()
        label.pose.position.x = bbox3d.center.position.x - bbox3d.size.x / 2
        label.pose.position.y = bbox3d.center.position.y + bbox3d.size.y / 2
        label.pose.position.z = bbox3d.center.position.z + bbox3d.size.z / 2
        q = rotate_quaternion_by_rpy(
            bbox3d.center.orientation, roll=1.5707, pitch=0.0, yaw=-1.5707, post_multiply=False)
        label.pose.orientation = q

        score_txt = TextPrimitive()
        score_txt.text = score
        score_txt.font_size = 0.05*dist
        score_txt.color.r = r
        score_txt.color.g = g
        score_txt.color.b = b
        score_txt.color.a = 0.5
        score_txt.pose = Pose()
        score_txt.pose.position.x = bbox3d.center.position.x - bbox3d.size.x / 2
        score_txt.pose.position.y = bbox3d.center.position.y - bbox3d.size.y / 2
        score_txt.pose.position.z = bbox3d.center.position.z + bbox3d.size.z / 2
        score_txt.pose.orientation = q

        id_txt = TextPrimitive()
        id_txt.text = detection.id
        id_txt.font_size = 0.05*dist
        id_txt.color.r = r
        id_txt.color.g = g
        id_txt.color.b = b
        id_txt.color.a = 0.5
        id_txt.pose = Pose()
        id_txt.pose.position.x = bbox3d.center.position.x - bbox3d.size.x / 2
        id_txt.pose.position.y = bbox3d.center.position.y
        id_txt.pose.position.z = bbox3d.center.position.z + bbox3d.size.z / 2
        id_txt.pose.orientation = q
        return label, score_txt, id_txt, cube

    def cb_detections(self, msg: DetectionArray):
        self.get_logger().debug(f"Received {len(msg.detections)} detections")

        self.scene_update.entities.clear()
        for i, det in enumerate(msg.detections):
            self.get_logger().debug(f"Processing detection {i}")
            ent = SceneEntity()
            ent.frame_id = "base_link"

            label, score, id_txt, cube = self.detection_to_viz(det)

            ent.texts.append(label)
            ent.texts.append(score)
            ent.texts.append(id_txt)
            ent.cubes.append(cube)

            self.scene_update.entities.append(ent)
        self.pub.publish(self.scene_update)


def main(args=None):
    rclpy.init(args=args)
    node = YoloToFoxgloveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':

    main()
