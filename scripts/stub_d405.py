"""Synthetic D405 + OWL stub for stub-isolated e2e: a box on a table.

Publishes the wrist camera's topic surface (color + camera_info + aligned
depth) with the depth image RAY-CAST from the table plane and a
box-shaped plateau at table_z + dims.z with the container's footprint —
the same renderer test_depth_source.py proves the detector against — plus
the static TF base_link -> end_effector_link that makes the grabber's
mount composition land the camera at exactly R_CAM / t_cam. The CLI's
BoxTopWatcher therefore runs the REAL depth path end to end (deproject,
TF lift, band, plateau, footprint, button circle). The colour frame
carries a dark disc at the button so the circle refine engages.

It also stands in for the persistent owl_detector node: a bbox around
the box's top face on /rammp_box_opening/owl_bbox, stamped with the
frame time like the real node, or a heartbeat (score -1) when --no-box.
The enable latch on owl_enable is ignored — inference costs nothing here.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray
from tf2_ros import StaticTransformBroadcaster

from rammp_curobo.perception import mat_to_quat_xyzw, quat_to_mat
from rammp_curobo_ros.cameras import load_camera_config

REPO = Path(__file__).resolve().parent.parent
# one renderer for the unit tests and for this stub: the geometry the
# detector is proven against is the geometry the e2e feeds it
sys.path.insert(0, str(REPO / "src/rammp_box_opening/test"))
from test_depth_source import H, K, W, render_depth  # noqa: E402

from rammp_box_opening.models.container import ContainerModel  # noqa: E402
from rammp_box_opening.perception.owl_source import BBOX_TOPIC  # noqa: E402

# camera looking straight down: x -> world +x, image-down -> world -y
R_CAM = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
OWL_SCORE = 0.25  # what the real node reports for the bench box


def project(p, t_cam):
    """World point -> (u, v, range) through the synthetic camera."""
    pc = R_CAM.T @ (np.asarray(p, dtype=float) - t_cam)
    return (
        K[0, 0] * pc[0] / pc[2] + K[0, 2],
        K[1, 1] * pc[1] / pc[2] + K[1, 2],
        pc[2],
    )


def top_corners(cx, cy, top_z, w, h, yaw):
    """World corners of a box's top face at the given yaw (about world z)."""
    c, s = np.cos(yaw), np.sin(yaw)
    out = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = sx * w / 2.0, sy * h / 2.0
        out.append([cx + c * lx - s * ly, cy + s * lx + c * ly, top_z])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--container",
        default=str(REPO / "src/rammp_box_opening/config/containers/oxo_pop.yaml"),
        help="container yaml: dims and button diameter come from here",
    )
    ap.add_argument("--box-x", type=float, default=0.45)
    ap.add_argument("--box-y", type=float, default=0.0)
    ap.add_argument("--box-yaw-deg", type=float, default=0.0)
    ap.add_argument("--cam-x", type=float, default=0.42)
    ap.add_argument("--cam-y", type=float, default=0.0)
    ap.add_argument("--cam-z", type=float, default=0.45)
    ap.add_argument("--table-z", type=float, default=-0.027)
    ap.add_argument(
        "--no-box", action="store_true", help="an empty table (no-box scenario)"
    )
    a = ap.parse_args()

    model = ContainerModel.load(a.container)
    t_cam = np.array([a.cam_x, a.cam_y, a.cam_z])
    top_z = a.table_z + model.dims[2]
    boxes = []
    bbox = None
    canvas = np.full((H, W, 3), 235, np.uint8)  # a white lid, like the bench
    if not a.no_box:
        box = (a.box_x, a.box_y, top_z, model.dims[0], model.dims[1], np.radians(a.box_yaw_deg))
        boxes.append(box)
        # the OWL stub boxes the container's top face, corners projected
        px = np.array([project(c, t_cam)[:2] for c in top_corners(*box)])
        bbox = [
            float(px[:, 0].min()),
            float(px[:, 1].min()),
            float(px[:, 0].max()),
            float(px[:, 1].max()),
        ]
        # the round button, centred on the lid: the circle refine aims here
        u, v, rng = project([a.box_x, a.box_y, top_z], t_cam)
        r_px = int(round(K[0, 0] * (model.button_diameter_m / 2.0) / rng))
        cv2.circle(canvas, (int(round(u)), int(round(v))), r_px, (60, 60, 60), -1)
    depth_m = render_depth(boxes, rot_cam=R_CAM, t_cam=t_cam, table_z=a.table_z)
    depth = (
        np.where(np.isfinite(depth_m), depth_m * 1000.0, 0.0).round().astype(np.uint16)
    )

    rclpy.init()
    node = rclpy.create_node("stub_d405")
    pub_c = node.create_publisher(Image, "/d405/d405/color/image_raw", 10)
    pub_d = node.create_publisher(
        Image, "/d405/d405/aligned_depth_to_color/image_raw", 10
    )
    pub_i = node.create_publisher(CameraInfo, "/d405/d405/color/camera_info", 10)
    pub_owl = node.create_publisher(Float32MultiArray, BBOX_TOPIC, 1)

    # static TF so the grabber's mount composition lands the camera at
    # exactly R_CAM/t_cam: T_base_ee = T_base_cam o inv(T_mount). No
    # tool_frame is published: the real bringup's tree has none either
    # (field 2026-08-25) and nothing in the mission looks it up.
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
    # The fingertip links the mission reads at a guard trip (contact_xyz):
    # static here, parked where the fingers would be touching the synthetic
    # button — the tip-link origin sits TIP_TO_TOOL + TCP above the pad
    # face, i.e. 19 mm above the surface (real URDF). The stub arm has no
    # FK, so this is the one honest place the harness can put them.
    tips = []
    for name, dx in (("robotiq_85_left_finger_tip_link", -0.025), ("robotiq_85_right_finger_tip_link", 0.025)):
        t = TransformStamped()
        t.header.stamp = node.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = name
        t.transform.translation.x = float(a.box_x + dx)
        t.transform.translation.y = float(a.box_y)
        t.transform.translation.z = float(top_z + 0.019)
        t.transform.rotation.w = 1.0
        tips.append(t)
    StaticTransformBroadcaster(node).sendTransform([tf, *tips])

    last_frame = {"t": None}

    def publish():
        stamp = node.get_clock().now().to_msg()
        last_frame["t"] = stamp.sec + stamp.nanosec * 1e-9
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
        info.k = [float(v) for v in K.ravel()]
        info.d = [0.0] * 5
        pub_i.publish(info)

    def owl():
        """The owl_detector node's contract (owl_node.py): a bbox with its
        frame AGE in slot 5 (float32-safe), or a heartbeat (score -1)
        when it sees nothing — the mission's rung tells 'alive, idle'
        from 'absent'."""
        msg = Float32MultiArray()
        if bbox is None or last_frame["t"] is None:
            msg.data = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]
        else:
            now = node.get_clock().now().nanoseconds * 1e-9
            msg.data = [*bbox, OWL_SCORE, max(0.0, now - last_frame["t"])]
        pub_owl.publish(msg)

    node.create_timer(1.0 / 15.0, publish)
    node.create_timer(0.5, owl)  # the real node ticks at 2 Hz
    print(
        "STUB D405 READY (%s)"
        % (
            "empty table"
            if a.no_box
            else "box top at [%.3f, %.3f, %.3f] yaw %g deg, owl bbox %s"
            % (a.box_x, a.box_y, top_z, a.box_yaw_deg, [round(v) for v in bbox])
        ),
        flush=True,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
