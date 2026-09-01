"""Local open-vocabulary bbox: OWLv2 on the Jetson, no network in the loop.

The wheelchair will not always have internet (deployment constraint,
2026-09-01), so the semantic gate needs a local answer. OWLv2 is the model NanoOWL
accelerates; on this Orin the plain-torch model already runs one frame
in 0.62 s, and the gate fires once per mission — TensorRT is a future
speed-up, not a prerequisite.

Measured on capture 20260901-130610 (8 scan-pose frames): bbox stable to
+/-1 px across frames, centred on the true box, score 0.235-0.249
against a 0.12-0.18 false-positive floor. Phrasing matters more than the
model: "a small white square box" finds it, "a food storage container"
does not — which is why the queries are CONFIG, plural, and the best
score across them wins.

Same contract and the same humility as the Claude backend: a returned
roi only ever NARROWS where the depth detector looks; every geometric
honesty gate still stands behind it. Model load (~tens of seconds cold)
happens on a background thread started at CLI boot, overlapping the
scan motion; a mission that arrives before the model is ready waits
briefly, then falls down the backend ladder.
"""

import threading

_lock = threading.Lock()
_models = {}  # name -> (processor, model)
_ready = {}  # name -> threading.Event


def preload(model_name):
    """Start loading the model on a daemon thread; returns immediately.
    Safe to call more than once."""
    with _lock:
        if model_name in _ready:
            return
        _ready[model_name] = threading.Event()

    def _load():
        try:
            import warnings

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            import torch  # noqa: F401  (fail here, not mid-mission)
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            from transformers import logging as hf_logging

            hf_logging.set_verbosity_error()

            proc = Owlv2Processor.from_pretrained(model_name)
            model = Owlv2ForObjectDetection.from_pretrained(model_name).eval().cuda()
            with _lock:
                _models[model_name] = (proc, model)
        finally:
            _ready[model_name].set()  # set even on failure: waiters move on

    threading.Thread(target=_load, daemon=True, name="owl-preload").start()


def pick_best_box(scores, labels, boxes, min_score):
    """Highest-scoring box across all queries, or None below the floor.
    Pure, unit-testable."""
    best = None
    for s, _l, b in zip(scores, labels, boxes):
        s = float(s)
        if s >= float(min_score) and (best is None or s > best[0]):
            best = (s, [int(v) for v in b])
    return best


def owl_box_roi(color_rgb, cfg):
    """(roi, why) with the fetch_box_roi contract: padded, image-clamped
    pixel bbox or None with the reason. Never raises."""
    try:
        preload(cfg.owl_model)
        if not _ready[cfg.owl_model].wait(timeout=float(cfg.vlm_timeout_s)):
            return None, "OWL model still loading — trying the next backend"
        with _lock:
            pair = _models.get(cfg.owl_model)
        if pair is None:
            return None, "OWL model failed to load — trying the next backend"
        proc, model = pair

        import torch

        h, w = color_rgb.shape[:2]
        queries = list(cfg.owl_queries)
        inputs = proc(text=[queries], images=[color_rgb], return_tensors="pt").to(
            "cuda"
        )
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_object_detection(
            out,
            threshold=float(cfg.owl_min_score),
            target_sizes=torch.tensor([[h, w]]).cuda(),
        )[0]
        best = pick_best_box(
            res["scores"].tolist(),
            res["labels"].tolist(),
            [b.tolist() for b in res["boxes"]],
            cfg.owl_min_score,
        )
    except Exception as exc:
        return None, "OWL failed (%s)" % str(exc).split("\n")[0][:120]
    if best is None:
        return None, "OWL sees no %s >= %.2f" % (queries[0], cfg.owl_min_score)
    score, (x0, y0, x1, y1) = best
    if x1 <= x0 or y1 <= y0:
        return None, "OWL returned a degenerate bbox"
    pad = int(cfg.vlm_pad_px)
    roi = (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w - 1, x1 + pad),
        min(h - 1, y1 + pad),
    )
    return roi, "OWL bbox (%d,%d)-(%d,%d) score %.2f" % (*roi, score)


BBOX_TOPIC = "/rammp_box_opening/owl_bbox"
TOPIC_FRESH_S = 3.0


def classify_bbox_msg(m, now, fresh_s=TOPIC_FRESH_S):
    """One topic message -> "bbox" | "alive" | "stale". Pure, testable."""
    if m is None or now - m[5] > fresh_s:
        return "stale"
    return "bbox" if m[4] >= 0.0 else "alive"


def make_topic_rung(node, cfg, watcher_holder=None):
    """An owl rung that reads the persistent owl_detector node's topic.

    Returns a (color, cfg) -> (roi, why) callable with the backend
    contract. The node heartbeats every tick even with nothing seen, so
    the rung can wait for a live node's answer instead of paying a cold
    in-process model load the moment one window is missed:

        fresh bbox       -> use it
        fresh heartbeat  -> node alive: keep waiting (it answers ~1 Hz)
        neither, ever    -> node absent: in-process fallback (slow, but
                            the offline path still works)

    A live node that finishes waiting having seen NO box is trusted:
    the rung declines without the fallback — two models disagreeing
    about the same frames helps nobody.
    """
    latest = {}
    pad = int(cfg.vlm_pad_px)

    def _cb(msg):
        m = list(msg.data)
        latest["m"] = m
        # live-gate the depth watcher the moment a bbox exists — the node
        # sees the box MID-SCAN, so the watcher collects roi-gated samples
        # while the arm is still moving and the fix can commit on arrival
        w = (watcher_holder or {}).get("watcher")
        if w is not None and m[4] >= 0.0:
            w.roi = (m[0] - pad, m[1] - pad, m[2] + pad, m[3] + pad)

    from std_msgs.msg import Float32MultiArray

    node.create_subscription(Float32MultiArray, BBOX_TOPIC, _cb, 1)

    def rung(color_rgb, cfg_):
        import time as _t

        import rclpy as _r

        saw_alive = False
        deadline = _t.monotonic() + 5.0
        while _t.monotonic() < deadline:
            now = node.get_clock().now().nanoseconds * 1e-9
            kind = classify_bbox_msg(latest.get("m"), now)
            if kind == "bbox":
                x0, y0, x1, y1, score = latest["m"][:5]
                h, w = color_rgb.shape[:2]
                pad = int(cfg_.vlm_pad_px)
                roi = (
                    max(0, int(x0) - pad),
                    max(0, int(y0) - pad),
                    min(w - 1, int(x1) + pad),
                    min(h - 1, int(y1) + pad),
                )
                return roi, "OWL node bbox (%d,%d)-(%d,%d) score %.2f" % (
                    *roi,
                    score,
                )
            saw_alive = saw_alive or kind == "alive"
            _r.spin_once(node, timeout_sec=0.1)
        if saw_alive:
            return None, "OWL node is live and sees no container top"
        # node absent: the in-process model is the slow offline fallback
        return owl_box_roi(color_rgb, cfg_)

    return rung
