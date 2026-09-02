"""Local open-vocabulary bbox: OWLv2 on the Jetson, no network in the loop.

The wheelchair will not always have internet (deployment constraint,
2026-09-01), so the semantic gate needs a local answer. OWLv2 runs in the
persistent owl_detector node (perception/owl_node.py); this module is the
mission side: the enable gate, the topic rung, and the pure helpers.

Measured on capture 20260901-130610 (8 scan-pose frames): bbox stable to
+/-1 px across frames, centred on the true box, score 0.235-0.249
against a 0.12-0.18 false-positive floor. Phrasing matters more than the
model: "a small white square box" finds it, "a food storage container"
does not — which is why the queries are CONFIG, plural, and the best
score across them wins.

Same contract and the same humility as the Claude backend: a returned
roi only ever NARROWS where the depth detector looks; every geometric
honesty gate still stands behind it. There is deliberately NO in-process
model: a cold load costs tens of seconds against a 10 s detect budget and
parked a second OWLv2 on the planner's GPU (review 2026-09-02).
"""

BBOX_TOPIC = "/rammp_box_opening/owl_bbox"
ENABLE_TOPIC = "/rammp_box_opening/owl_enable"
TOPIC_FRESH_S = 3.0
# a bbox may gate the depth watcher only when its FRAME is this recent:
# the node stamps frame time, inference is ~0.65 s, and a box seen while
# the camera was still moving must not gate a parked frame
ROI_FRESH_S = 1.5


def pick_best_box(scores, labels, boxes, min_score):
    """Highest-scoring box across all queries, or None below the floor.
    Pure, unit-testable."""
    best = None
    for s, _l, b in zip(scores, labels, boxes):
        s = float(s)
        if s >= float(min_score) and (best is None or s > best[0]):
            best = (s, [int(v) for v in b])
    return best


def classify_bbox_msg(m, now, fresh_s=TOPIC_FRESH_S):
    """One topic message -> "bbox" | "alive" | "stale". Pure, testable."""
    if m is None or now - m[5] > fresh_s:
        return "stale"
    return "bbox" if m[4] >= 0.0 else "alive"


def roi_from_bbox(m, shape, pad):
    """Padded, image-clamped roi from a bbox message. Pure."""
    h, w = shape[:2]
    x0, y0, x1, y1 = m[:4]
    return (
        max(0, int(x0) - pad),
        max(0, int(y0) - pad),
        min(w - 1, int(x1) + pad),
        min(h - 1, int(y1) + pad),
    )


class OwlRung:
    """The owl rung: reads the persistent owl_detector node's topic and
    owns the enable gate the node listens to.

    Call enable() around a detect window and disable() when the fix
    commits — inference outside the window only slows the planner. The
    node heartbeats every tick even with nothing seen, so the rung can
    wait for a live node's answer instead of guessing:

        fresh bbox       -> use it
        fresh heartbeat  -> node alive: keep waiting (it answers ~2 Hz)
        neither, ever    -> node absent: decline at once (plain depth is
                            the floor; nothing in-process to fall back to)

    A live node that finishes waiting having seen NO box is trusted: the
    rung declines — two models disagreeing about the same frames helps
    nobody. While enabled, every fresh bbox live-gates the depth watcher
    so its roi-gated samples are already collected when the arm parks.
    """

    def __init__(self, node, cfg, watcher_holder=None):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool, Float32MultiArray

        self.node = node
        self.cfg = cfg
        self.watcher_holder = watcher_holder or {}
        self.latest = None
        self.enabled = False
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._enable_pub = node.create_publisher(Bool, ENABLE_TOPIC, latched)
        self._Bool = Bool
        node.create_subscription(Float32MultiArray, BBOX_TOPIC, self._cb, 1)

    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def enable(self):
        self.enabled = True
        self._enable_pub.publish(self._Bool(data=True))

    def disable(self):
        self.enabled = False
        self._enable_pub.publish(self._Bool(data=False))

    def _cb(self, msg):
        m = list(msg.data)
        self.latest = m
        w = self.watcher_holder.get("watcher")
        if (
            w is not None
            and self.enabled
            and m[4] >= 0.0
            and self._now() - m[5] <= ROI_FRESH_S
            and w.grab.color is not None
        ):
            w.roi = roi_from_bbox(m, w.grab.color.shape, int(self.cfg.vlm_pad_px))

    def __call__(self, color_rgb, cfg_):
        import time as _t

        import rclpy as _r

        saw_alive = False
        deadline = _t.monotonic() + 2.0
        while _t.monotonic() < deadline:
            kind = classify_bbox_msg(self.latest, self._now())
            if kind == "bbox":
                roi = roi_from_bbox(self.latest, color_rgb.shape, int(cfg_.vlm_pad_px))
                return roi, "OWL node bbox (%d,%d)-(%d,%d) score %.2f" % (
                    *roi,
                    self.latest[4],
                )
            saw_alive = saw_alive or kind == "alive"
            if not saw_alive and self.latest is None and _t.monotonic() > deadline - 1.5:
                break  # nothing at all in 0.5 s: no node on the graph
            _r.spin_once(self.node, timeout_sec=0.1)
        if saw_alive:
            return None, "OWL node is live and sees no container top"
        return None, "owl_detector node not running — plain depth"


def make_topic_rung(node, cfg, watcher_holder=None):
    """Backwards-compatible constructor; returns the OwlRung callable."""
    return OwlRung(node, cfg, watcher_holder)
