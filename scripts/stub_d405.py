"""Synthetic D405 for stub-isolated e2e: real rendered ArUco frames.

Publishes the wrist camera's topic surface (color + camera_info +
aligned depth) with an actual DICT_4X4_50 marker warped into the frame
at the geometry a real camera at --cam-* would see for a tag at
--tag-*, plus the static TF base_link -> end_effector_link consistent
with the mount yaml — so the CLI's TagWatcher runs the REAL detection
path end to end. --no-marker publishes tagless frames (no-tag scenario).
"""

import argparse

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

from rammp_curobo.perception import mat_to_quat_xyzw, quat_to_mat
from rammp_curobo_ros.cameras import load_camera_config

W, H = 848, 480
FX = FY = 430.0
CX, CY = W / 2.0, H / 2.0
# camera looking straight down: x -> world +x, image-down -> world -y
R_CAM = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def render(tag_xyz, size_m, yaw_rad):
    """World corners of a flat tag at the given yaw (about world z)."""
    s = size_m / 2.0
    corners_tag = np.array(
        [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=float
    )
    c, sn = np.cos(yaw_rad), np.sin(yaw_rad)
    rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    return corners_tag @ rz.T + np.asarray(tag_xyz, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-x", type=float, default=0.45)
    ap.add_argument("--tag-y", type=float, default=0.02)
    ap.add_argument("--tag-z", type=float, default=0.133)
    ap.add_argument("--cam-x", type=float, default=0.42)
    ap.add_argument("--cam-y", type=float, default=0.0)
    ap.add_argument("--cam-z", type=float, default=0.45)
    ap.add_argument("--size", type=float, default=0.05)
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--tag-yaw-deg", type=float, default=0.0)
    ap.add_argument("--table-z", type=float, default=-0.027)
    ap.add_argument("--no-marker", action="store_true")
    a = ap.parse_args()

    t_cam = np.array([a.cam_x, a.cam_y, a.cam_z])
    canvas = np.full((H, W, 3), 110, np.uint8)
    # depth is spatially STRUCTURED: the tag-plane range only around the
    # tag's projected pixels, bench range everywhere else — so a
    # wrong-pixel depth lookup lands on the bench value and the e2e's
    # origin check catches it (2026-08-24 review: uniform depth made
    # refine_point's pixel math unfalsifiable)
    bench_mm = int(round((a.cam_z - a.table_z) * 1000))
    depth = np.full((H, W), bench_mm, np.uint16)
    if not a.no_marker:
        corners_w = render(
            [a.tag_x, a.tag_y, a.tag_z], a.size, np.radians(a.tag_yaw_deg)
        )
        px = []
        for cw in corners_w:
            pc = R_CAM.T @ (cw - t_cam)
            px.append([FX * pc[0] / pc[2] + CX, FY * pc[1] / pc[2] + CY])
        px = np.array(px, dtype=np.float32)
        tag_mm = int(round((a.cam_z - a.tag_z) * 1000))
        u0, u1 = int(px[:, 0].min()) - 12, int(px[:, 0].max()) + 12
        v0, v1 = int(px[:, 1].min()) - 12, int(px[:, 1].max()) + 12
        depth[max(0, v0) : v1, max(0, u0) : u1] = tag_mm
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), a.id, 200
        )
        pad = 60
        padded = np.full((200 + 2 * pad, 200 + 2 * pad), 255, np.uint8)
        padded[pad : pad + 200, pad : pad + 200] = marker
        src = np.array(
            [[pad, pad], [pad + 200, pad], [pad + 200, pad + 200], [pad, pad + 200]],
            dtype=np.float32,
        )
        hmat = cv2.getPerspectiveTransform(src, px)
        warped = cv2.warpPerspective(
            cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR),
            hmat,
            (W, H),
            canvas.copy(),
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        canvas = warped

    rclpy.init()
    node = rclpy.create_node("stub_d405")
    pub_c = node.create_publisher(Image, "/d405/d405/color/image_raw", 10)
    pub_d = node.create_publisher(
        Image, "/d405/d405/aligned_depth_to_color/image_raw", 10
    )
    pub_i = node.create_publisher(CameraInfo, "/d405/d405/color/camera_info", 10)

    # static TF so the grabber's mount composition lands the camera at
    # exactly R_CAM/t_cam: T_base_ee = T_base_cam o inv(T_mount)
    cfg = load_camera_config("camera_d405_wrist.yaml")
    qx, qy, qz, qw = cfg["mount_quat_xyzw"]
    r_mount = quat_to_mat(qx, qy, qz, qw)
    r_ee = R_CAM @ r_mount.T
    t_ee = t_cam - r_ee @ np.asarray(cfg["mount_xyz"], dtype=float)
    tf = TransformStamped()
    tf.header.stamp = node.get_clock().now().to_msg()
    tf.header.frame_id = "base_link"
    tf.child_frame_id = cfg["parent_frame"]
    tf.transform.translation.x, tf.transform.translation.y = t_ee[0], t_ee[1]
    tf.transform.translation.z = t_ee[2]
    q = mat_to_quat_xyzw(r_ee)
    tf.transform.rotation.x, tf.transform.rotation.y = float(q[0]), float(q[1])
    tf.transform.rotation.z, tf.transform.rotation.w = float(q[2]), float(q[3])
    # tool_frame too: the runner's trip-depth report looks it up
    tf2 = TransformStamped()
    tf2.header.stamp = tf.header.stamp
    tf2.header.frame_id = "base_link"
    tf2.child_frame_id = "tool_frame"
    tf2.transform.translation.x = a.tag_x
    tf2.transform.translation.y = a.tag_y
    tf2.transform.translation.z = a.tag_z
    tf2.transform.rotation.w = 1.0
    StaticTransformBroadcaster(node).sendTransform([tf, tf2])

    def publish():
        stamp = node.get_clock().now().to_msg()
        im = Image()
        im.header.stamp = stamp
        im.header.frame_id = "d405_color_optical_frame"
        im.height, im.width = H, W
        im.encoding = "bgr8"
        im.step = W * 3
        im.data = canvas.tobytes()
        pub_c.publish(im)
        dm = Image()
        dm.header.stamp = stamp
        dm.header.frame_id = "d405_color_optical_frame"
        dm.height, dm.width = H, W
        dm.encoding = "16UC1"
        dm.step = W * 2
        dm.data = depth.tobytes()
        pub_d.publish(dm)
        info = CameraInfo()
        info.header.stamp = stamp
        info.height, info.width = H, W
        info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        pub_i.publish(info)

    node.create_timer(1.0 / 15.0, publish)
    print(
        "STUB D405 READY (%s)" % ("no marker" if a.no_marker else "marker"), flush=True
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
