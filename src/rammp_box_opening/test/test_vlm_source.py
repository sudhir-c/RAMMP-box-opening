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
