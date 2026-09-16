#!/usr/bin/env bash
# One-command bring-up for the SIZE-AGNOSTIC detector (design-review alternative 1).
#
#   scripts/rchi_agnostic_bringup.sh [seconds] [extra detector args]
#
#   scripts/rchi_agnostic_bringup.sh                 # 120 s, diagnostic window
#   scripts/rchi_agnostic_bringup.sh 300 --min-side-mm 50 --raise-mm 20
#
# Starts the RealSense driver under the d405 names (unless one is already running, with
# the spatial+temporal filters), then runs detect_agnostic.py --window. No TF, no arm,
# no planner: the table is fitted from depth every frame. Ctrl+C or q stops everything.
set -eo pipefail   # no -u: ROS setup.bash reads unset variables

SECONDS_RUN="${1:-120}"
shift $(( $# >= 1 ? 1 : 0 ))

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

if pgrep -f realsense2_camera_node >/dev/null; then
  echo "[bringup] RealSense driver already running; reusing it (restart it once if it predates the noise filters)"
else
  echo "[bringup] starting RealSense under /d405/d405 (640x480x15, aligned depth, spatial+temporal filters)"
  ros2 launch realsense2_camera rs_launch.py camera_namespace:=d405 camera_name:=d405 \
    align_depth.enable:=true rgb_camera.color_profile:=640,480,15 depth_module.depth_profile:=640,480,15 \
    spatial_filter.enable:=true temporal_filter.enable:=true hole_filling_filter.enable:=false \
    > /tmp/rchi_realsense.log 2>&1 &
  PIDS+=($!)
  sleep 3
fi

echo "[bringup] size-agnostic detector for ${SECONDS_RUN}s; q in the window or Ctrl+C ends everything"
cd "$REPO"
python3 scripts/detect_agnostic.py --seconds "$SECONDS_RUN" --window "$@"
