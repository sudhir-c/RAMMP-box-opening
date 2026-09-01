"""owl_detector — persistent local bbox service for the semantic gate.

    ros2 run rammp_box_opening owl_detector

The OWL model costs tens of seconds to load, and a CLI that loads it per
run makes every mission wait (field 2026-09-01: the detect phase sat on
the preload). This node loads ONCE — at launch, overlapping the
planner's own GPU init — then runs the detector on the live colour
stream at a gentle cadence and publishes the best bbox:

    /rammp_box_opening/owl_bbox   std_msgs/Float32MultiArray
                                  [x0, y0, x1, y1, score, stamp_sec]

Published only when something clears vlm.owl_min_score; the stamp lets
the mission ignore stale sightings. The mission's owl rung reads this
topic first and only falls back to an in-process load when the node is
not running. This is also the architecture the deployment target wants:
an always-on perception service whose output the mission consumes.

Queries, model, and threshold come from the same container YAML the
mission uses (`container` parameter), so there is exactly one place to
tune them.
"""

import time
import warnings

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class OwlDetector(Node):
    def __init__(self):
        super().__init__("owl_detector")
        from rammp_box_opening.models.container import load_press_demo
        from rammp_box_opening.tasks.cli_common import default_container_yaml

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        container = self.declare_parameter("container", "").value
        # 2 Hz: at 1 Hz the brief mid-scan view of the box could fall
        # between ticks; inference is ~0.65 s so this saturates only
        # while frames actually change
        period = float(self.declare_parameter("period_s", 0.5).value)
        cfg_path = container or default_container_yaml()
        self.cfg = load_press_demo(str(cfg_path))

        self.get_logger().info("loading %s ..." % self.cfg.owl_model)
        t0 = time.monotonic()
        import torch  # noqa: F401
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        self._proc = Owlv2Processor.from_pretrained(self.cfg.owl_model)
        self._model = (
            Owlv2ForObjectDetection.from_pretrained(self.cfg.owl_model).eval().cuda()
        )
        self.get_logger().info(
            "owl_detector ready in %.1f s — %s at %.1f Hz, min score %.2f"
            % (
                time.monotonic() - t0,
                self.cfg.owl_model,
                1.0 / period,
                self.cfg.owl_min_score,
            )
        )

        from rammp_curobo_ros.seek_core import D405Grabber

        self.grab = D405Grabber(self, need_depth=False)
        self.pub = self.create_publisher(
            Float32MultiArray, "/rammp_box_opening/owl_bbox", 1
        )
        self._last_stamp = None
        self.create_timer(period, self._tick)

    def _tick(self):
        g = self.grab
        if g.color is None or g.color_stamp is None:
            return
        stamp = (g.color_stamp.sec, g.color_stamp.nanosec)
        if stamp == self._last_stamp:
            return
        self._last_stamp = stamp

        import torch

        from rammp_box_opening.perception.owl_source import pick_best_box

        h, w = g.color.shape[:2]
        queries = list(self.cfg.owl_queries)
        inputs = self._proc(text=[queries], images=[g.color], return_tensors="pt").to(
            "cuda"
        )
        with torch.no_grad():
            out = self._model(**inputs)
        res = self._proc.post_process_object_detection(
            out,
            threshold=float(self.cfg.owl_min_score),
            target_sizes=torch.tensor([[h, w]]).cuda(),
        )[0]
        best = pick_best_box(
            res["scores"].tolist(),
            res["labels"].tolist(),
            [b.tolist() for b in res["boxes"]],
            self.cfg.owl_min_score,
        )
        msg = Float32MultiArray()
        now = self.get_clock().now().nanoseconds * 1e-9
        if best is None:
            # heartbeat: the mission can tell "node alive, keep waiting"
            # from "node absent, fall back" (field 2026-09-01: without
            # this, one missed window cost a cold in-process model load)
            msg.data = [0.0, 0.0, 0.0, 0.0, -1.0, now]
        else:
            score, (x0, y0, x1, y1) = best
            msg.data = [float(x0), float(y0), float(x1), float(y1), float(score), now]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OwlDetector()
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
