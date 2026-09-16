#!/usr/bin/env bash
# One-command detection bring-up for the RCHI bench (Luxray), no arm motion.
#
#   scripts/rchi_detect_bringup.sh plane [seconds] [extra detector args]      # RECOMMENDED: no TF, table plane fitted from depth
#   scripts/rchi_detect_bringup.sh <lens_to_table_m> [seconds] [extra detector args]   # fake vertical camera pose + tape measure
#
#   scripts/rchi_detect_bringup.sh plane           # camera tilt + height measured from the table itself
#   scripts/rchi_detect_bringup.sh 0.40            # lens 40 cm above the table, 120 s, overlay on
#   scripts/rchi_detect_bringup.sh 0.40 300        # 5 minutes
#   scripts/rchi_detect_bringup.sh 0.40 120 --no-circle
#
# Starts, in the background: the RealSense driver under Chris's d405 names (unless one is
# already running), a FAKE base_link->end_effector_link transform that puts the camera at
# CAM_Z looking straight down, and rqt_image_view on the overlay topic. Then runs
# detect_standalone.py in the foreground. Ctrl+C stops everything it started.
#
# The fake transform means x/y come out in a made-up frame; z is real relative to the
# table because table_z is derived from your lens-to-table measurement. Stop this script
# before bringing up any real TF source (the feeding launch / ros2_kortex).
set -eo pipefail   # no -u: ROS setup.bash reads unset variables

LENS_TO_TABLE="${1:?usage: $0 plane|<lens_to_table_m> [seconds] [detector args...]}"
SECONDS_RUN="${2:-120}"
shift $(( $# >= 2 ? 2 : $# ))
CAM_Z=0.45
if [ "$LENS_TO_TABLE" = "plane" ]; then
  PLANE_MODE=1
  DETECT_ARGS=(--table-from-depth)
else
  PLANE_MODE=0
  TABLE_Z=$(python3 -c "print(round($CAM_Z - $LENS_TO_TABLE, 4))")
  DETECT_ARGS=(--table-z "$TABLE_Z")
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_LOCALHOST_ONLY=1
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
source "$HOME/RAMMP-CuRobo/install/setup.bash"
source "$REPO/install/setup.bash"
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

PIDS=()
cleanup() {
  echo; echo "[bringup] stopping background processes"
  for p in "${PIDS[@]}"; do [ -n "$p" ] && kill -INT "$p" 2>/dev/null || true; done
  sleep 1
  for p in "${PIDS[@]}"; do [ -n "$p" ] && kill -KILL "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

# 1. camera (a second driver on the same device kills both -- reuse a running one)
if pgrep -f realsense2_camera_node >/dev/null; then
  echo "[bringup] RealSense driver already running; reusing it"
else
  echo "[bringup] starting RealSense under /d405/d405 (640x480x15, aligned depth)"
  ros2 launch realsense2_camera rs_launch.py camera_namespace:=d405 camera_name:=d405 \
    align_depth.enable:=true rgb_camera.color_profile:=640,480,15 depth_module.depth_profile:=640,480,15 \
    > /tmp/rchi_realsense.log 2>&1 &
  PIDS+=($!)
fi

# 2. fake camera pose: end_effector_link CAM_Z up, tool z pointing down (180 deg about x)
if [ "$PLANE_MODE" = 1 ]; then
  echo "[bringup] plane mode: no TF needed, the table plane is fitted from depth each frame"
elif ros2 run tf2_ros tf2_echo base_link end_effector_link 2>&1 | timeout 2 grep -q "Translation"; then
  echo "[bringup] a base_link->end_effector_link transform is ALREADY published; not adding a fake one"
else
  echo "[bringup] publishing fake base_link->end_effector_link at z=$CAM_Z looking down"
  ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z "$CAM_Z" --qx 1 --qy 0 --qz 0 --qw 0 \
    --frame-id base_link --child-frame-id end_effector_link > /tmp/rchi_tf.log 2>&1 &
  PIDS+=($!)
fi

# 3. viewer (skipped when the detector opens its own window)
if [[ " $* " == *" --window "* ]]; then
  echo "[bringup] --window requested: the detector opens its own diagnostic window; no rqt"
elif command -v rqt_image_view >/dev/null || ros2 pkg prefix rqt_image_view >/dev/null 2>&1; then
  ros2 run rqt_image_view rqt_image_view /detect_standalone/overlay > /tmp/rchi_rqt.log 2>&1 &
  PIDS+=($!)
  echo "[bringup] rqt_image_view on /detect_standalone/overlay (pick /d405/d405/color/image_raw in its dropdown for the raw feed)"
fi

# give the driver a moment to open the streams
sleep 3
[ "$PLANE_MODE" = 1 ] || echo "[bringup] table_z = $CAM_Z - $LENS_TO_TABLE = $TABLE_Z   (lid expected at table_z + 0.112)"
echo "[bringup] detector for ${SECONDS_RUN}s; Ctrl+C ends everything"
cd "$REPO"
python3 scripts/detect_standalone.py "${DETECT_ARGS[@]}" --seconds "$SECONDS_RUN" --overlay "$@"
