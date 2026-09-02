"""joint_state_relay — a 20 Hz copy of /joint_states for the planner.

    ros2 run rammp_box_opening joint_state_relay

The planner node subscribes to joint states with a Python callback that
runs per message; at the driver's ~100 Hz that callback alone lifts a
0.22 s cuRobo solve to ~0.37 s (measured offline 2026-09-02). The
planner only needs the CURRENT joints when a plan or execution starts,
so it is pointed (joint_states_topic parameter) at this relay instead:
latest message republished at RELAY_HZ from a separate process, leaving
the planner's interpreter free while it solves. The mission CLI keeps
the raw topic — its torque guard wants efforts at full rate.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

RELAY_HZ = 20.0
RELAY_TOPIC = "/rammp_box_opening/joint_states"


class JointStateRelay(Node):
    def __init__(self):
        super().__init__("joint_state_relay")
        hz = float(self.declare_parameter("rate_hz", RELAY_HZ).value)
        self._latest = None
        self.create_subscription(
            JointState, "/joint_states", self._on_js, qos_profile_sensor_data
        )
        self._pub = self.create_publisher(JointState, RELAY_TOPIC, qos_profile_sensor_data)
        self.create_timer(1.0 / hz, self._tick)

    def _on_js(self, msg):
        self._latest = msg

    def _tick(self):
        if self._latest is not None:
            self._pub.publish(self._latest)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
