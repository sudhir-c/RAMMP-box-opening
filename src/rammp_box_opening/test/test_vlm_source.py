"""The VLM gate: bbox handling, fallbacks, and the no-network contract."""

import numpy as np
import pytest

from rammp_box_opening.models.container import load_press_demo
from rammp_box_opening.perception.vlm_source import BoxLocation, fetch_box_roi

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"


class _FakeResp:
    def __init__(self, loc):
        self.parsed_output = loc


class _FakeClient:
    def __init__(self, loc=None, exc=None):
        self.loc, self.exc = loc, exc
        self.messages = self
        self.calls = []

    def parse(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return _FakeResp(self.loc)


@pytest.fixture(scope="module")
def cfg():
    return load_press_demo(CFG)


FRAME = np.zeros((480, 848, 3), dtype=np.uint8)


def test_found_bbox_is_padded_and_clamped(cfg):
    c = _FakeClient(
        BoxLocation(found=True, x0=10, y0=5, x1=830, y1=470, confidence=0.9)
    )
    roi, why = fetch_box_roi(FRAME, cfg, client=c)
    assert roi == (0, 0, 847, 479)  # padded past the edge, clamped to image
    assert "conf 0.90" in why
    # the call used the configured model and target
    assert c.calls[0]["model"] == cfg.vlm_model
    assert cfg.vlm_target in c.calls[0]["messages"][0]["content"][1]["text"]


def test_not_visible_falls_back(cfg):
    c = _FakeClient(BoxLocation(found=False, x0=0, y0=0, x1=0, y1=0, confidence=0.0))
    roi, why = fetch_box_roi(FRAME, cfg, client=c)
    assert roi is None and "not visible" in why


def test_degenerate_bbox_falls_back(cfg):
    c = _FakeClient(
        BoxLocation(found=True, x0=100, y0=100, x1=100, y1=90, confidence=0.9)
    )
    roi, why = fetch_box_roi(FRAME, cfg, client=c)
    assert roi is None and "degenerate" in why


def test_api_failure_is_a_fallback_never_a_stop(cfg):
    c = _FakeClient(exc=RuntimeError("connection refused"))
    roi, why = fetch_box_roi(FRAME, cfg, client=c)
    assert roi is None
    assert "falling back to plain depth" in why


def test_shipped_config_uses_the_measured_model(cfg):
    """Field 2026-09-01: haiku boxed the robot's own base; opus was
    dead-on at +0.4 s. The shipped default is the one that works."""
    assert cfg.detect_source == "vlm"
    assert cfg.vlm_model == "claude-opus-5"
    assert "OXO" in cfg.vlm_target


# ---------------------------------------------------------------- ladder


def _mk(roi, why):
    return lambda color, cfg: (roi, why)


def test_ladder_prefers_the_local_backend(cfg):
    roi, lines = __import__(
        "rammp_box_opening.perception.vlm_source", fromlist=["resolve_roi"]
    ).resolve_roi(
        FRAME,
        cfg,
        impls={"owl": _mk((1, 2, 3, 4), "OWL bbox"), "claude": _mk((9, 9, 9, 9), "x")},
    )
    assert roi == (1, 2, 3, 4)
    assert lines == ["owl: OWL bbox"]  # claude never called


def test_ladder_falls_through_to_claude(cfg):
    from rammp_box_opening.perception.vlm_source import resolve_roi

    roi, lines = resolve_roi(
        FRAME,
        cfg,
        impls={
            "owl": _mk(None, "still loading"),
            "claude": _mk((5, 6, 7, 8), "bbox conf 0.95"),
        },
    )
    assert roi == (5, 6, 7, 8)
    assert [ln.split(":")[0] for ln in lines] == ["owl", "claude"]


def test_ladder_exhausted_means_plain_depth(cfg):
    from rammp_box_opening.perception.vlm_source import resolve_roi

    roi, lines = resolve_roi(
        FRAME, cfg, impls={"owl": _mk(None, "a"), "claude": _mk(None, "b")}
    )
    assert roi is None and len(lines) == 2


def test_owl_pick_best_box_floor():
    from rammp_box_opening.perception.owl_source import pick_best_box

    got = pick_best_box(
        [0.12, 0.24, 0.19],
        [0, 0, 1],
        [[0, 0, 1, 1], [10, 10, 20, 20], [5, 5, 9, 9]],
        min_score=0.18,
    )
    assert got == (0.24, [10, 10, 20, 20])
    assert pick_best_box([0.1], [0], [[0, 0, 1, 1]], 0.18) is None


def test_shipped_ladder_is_local_first(cfg):
    """The wheelchair will not always have internet (deployment, 2026-09-01):
    the local model leads, the cloud is the fallback."""
    assert cfg.vlm_backends == ("owl", "claude")
    assert cfg.owl_min_score == pytest.approx(0.18)
    assert any("white" in q for q in cfg.owl_queries)
