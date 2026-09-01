"""Local open-vocabulary bbox: OWLv2 on the Jetson, no network in the loop.

The wheelchair will not always have internet (Swapnil, 2026-09-01), so
the semantic gate needs a local answer. OWLv2 is the model NanoOWL
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
            import torch  # noqa: F401  (fail here, not mid-mission)
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

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
