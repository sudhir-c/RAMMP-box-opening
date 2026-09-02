"""VLM semantic gate for the depth pose source: WHICH surface is the box.

The depth source measures geometry to millimetres but cannot tell one
box-sized top from another (or from clutter). One Claude call per
mission answers the semantic half: the scan-pose colour frame goes up,
a pixel bbox comes back, and the depth detector runs restricted to that
region — every geometric honesty gate (band, smoothness, footprint,
border) still stands, so a wrong bbox yields a refusal, not a press.

Division of labour, measured on capture 20260901-130610:
    claude-opus-5 bbox      3.3 s, ~$0.007/mission, box dead-centre
    depth inside the bbox   fix [0.459 -0.145 0.086] — the true box
(claude-haiku-4-5 was 0.4 s faster and confidently boxed the ROBOT'S
OWN BASE — the model knob exists, but fast is not the default here.)

Failure is a fallback, never a stop: no key, no network, timeout, or
"not visible" all return (None, why) and the mission proceeds on plain
depth, which refuses honestly if the scene is ambiguous. The target is
a config STRING ("the OXO POP container" today, any container the owner
describes tomorrow) — that is the whole reason this is a VLM and not a
trained detector.
"""

import base64


def _box_schema():
    # pydantic is a dependency of the cloud rung only; keep the mission's
    # import cheap and the rung's failure mode a fallback, never a crash
    from pydantic import BaseModel

    class BoxLocation(BaseModel):
        found: bool
        x0: int
        y0: int
        x1: int
        y1: int
        confidence: float

    return BoxLocation


def __getattr__(name):
    # tests build BoxLocation results directly; keep the name importable
    # without paying the pydantic import on the mission's happy path
    if name == "BoxLocation":
        return _box_schema()
    raise AttributeError(name)


PROMPT = (
    "This is a %dx%d image from a robot's downward-looking wrist camera "
    "over a workbench. Find %s. Ignore the robot arm, its mount, game "
    "controllers, cables and furniture. Return its bounding box in PIXEL "
    "coordinates ((x0,y0) top-left, (x1,y1) bottom-right, x right, y "
    "down). If it is not visible, set found=false."
)


def fetch_box_roi(color_bgr, cfg, client=None, budget_s=None):
    """One VLM call -> (roi, why). roi is a padded, image-clamped pixel
    bbox for top_face_from_depth, or None with the reason.

    `client` is injectable for tests; by default the Anthropic SDK client
    is built from the environment (ANTHROPIC_API_KEY / an active auth
    profile) at call time, so import stays cheap and key-free. The call
    is BOUNDED: no SDK retries and a timeout no longer than the detect
    budget left — on the no-internet target the default retries could
    stall ~60 s outside the 10 s budget (review 2026-09-02)."""
    import cv2

    h, w = color_bgr.shape[:2]
    timeout = float(cfg.vlm_timeout_s)
    if budget_s is not None:
        timeout = min(timeout, float(budget_s))
        if timeout < 0.5:
            return None, "no detect budget left for a cloud call"
    try:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout, max_retries=0)
        # the grabber stores BGR, which is what imencode expects
        ok, png = cv2.imencode(".png", color_bgr)
        if not ok:
            return None, "could not encode the frame"
        resp = client.messages.parse(
            model=cfg.vlm_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.standard_b64encode(png.tobytes()).decode(
                                    "utf-8"
                                ),
                            },
                        },
                        {"type": "text", "text": PROMPT % (w, h, cfg.vlm_target)},
                    ],
                }
            ],
            output_format=_box_schema(),
        )
        loc = resp.parsed_output
    except Exception as exc:  # network, auth, timeout, refusal — all fall back
        return None, "VLM call failed (%s) — falling back to plain depth" % (
            str(exc).split("\n")[0][:120]
        )
    if not loc.found:
        return None, "VLM reports the container is not visible"
    if loc.x1 <= loc.x0 or loc.y1 <= loc.y0:
        return None, "VLM returned a degenerate bbox"
    pad = int(cfg.vlm_pad_px)
    roi = (
        max(0, loc.x0 - pad),
        max(0, loc.y0 - pad),
        min(w - 1, loc.x1 + pad),
        min(h - 1, loc.y1 + pad),
    )
    return roi, "bbox (%d,%d)-(%d,%d) conf %.2f" % (*roi, loc.confidence)


def resolve_roi(color_bgr, cfg, impls=None):
    """Walk the configured backend ladder; first roi wins.

    Ladder shape (deployment review, 2026-09-01): the wheelchair will not
    always have internet, so LOCAL comes first and the cloud is the
    fallback, with plain depth as the floor when every rung declines.
    Returns (roi | None, [per-backend status lines]). `impls` is
    injectable for tests; the owl rung needs the live node (an OwlRung
    from owl_source) and is skipped when none was supplied. The caller
    binds the cloud rung's budget (fetch_box_roi's budget_s)."""
    if impls is None:
        impls = {"claude": fetch_box_roi}
    lines = []
    for name in cfg.vlm_backends:
        fn = impls.get(name)
        if fn is None:
            lines.append("%s: no live rung — skipped" % name)
            continue
        roi, why = fn(color_bgr, cfg)
        lines.append("%s: %s" % (name, why))
        if roi is not None:
            return roi, lines
    return None, lines
