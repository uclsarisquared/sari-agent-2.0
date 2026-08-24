"""Regression tests for bbox parse failures being retried and surfaced distinctly."""

import os
import sys
from io import BytesIO
from unittest.mock import patch

from PIL import Image

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vision import perception as p
from agent_core.llm import DEFAULT_API_MAX_ATTEMPTS, MalformedContentError, configure_api_retries


def _png_bytes():
    buf = BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_empty_bbox_is_a_real_negative_without_retry():
    replies = iter(["[]"])
    calls = []

    def request(*args):
        calls.append(args)
        return next(replies)

    with patch.object(p, "_bbox_request", request):
        assert p._detect_boxes_px(object(), "target") == []
    assert len(calls) == 1


def test_plain_language_absence_is_a_real_negative_without_retry():
    reply = (
        "I am sorry, but I cannot fulfill this request. The image provided does not contain "
        "any Ritz Crackers."
    )
    calls = []

    def request(*args):
        calls.append(args)
        return reply

    with patch.object(p, "_bbox_request", request):
        assert p._detect_boxes_px(object(), "Ritz Crackers") == []
    assert len(calls) == 1


def test_unrelated_plain_language_response_remains_malformed():
    configure_api_retries(2)
    calls = []

    def request(*args):
        calls.append(args)
        return "I am sorry, but I cannot fulfill this request."

    with (patch.object(p, "_bbox_request", request),
          patch("agent_core.llm.time.sleep", return_value=None)):
        try:
            p._detect_boxes_px(object(), "target")
        except p.BBoxResponseParseError:
            pass
        else:
            raise AssertionError("ambiguous prose was treated as a negative detection")
    configure_api_retries(DEFAULT_API_MAX_ATTEMPTS)
    assert len(calls) == 2


def test_empty_or_truncated_completion_is_normalized_to_bbox_parse_error():
    configure_api_retries(1)
    request_error = MalformedContentError("bounding-box response was empty or truncated")

    with patch.object(p, "_bbox_request", side_effect=request_error):
        try:
            p._detect_boxes_px(object(), "target")
        except p.BBoxResponseParseError as error:
            assert "empty or truncated" in str(error)
        else:
            raise AssertionError("request-level malformed content escaped the bbox error contract")
    configure_api_retries(DEFAULT_API_MAX_ATTEMPTS)


def test_malformed_bbox_is_retried_then_succeeds():
    configure_api_retries(2)
    replies = iter([
        "[{'box_2d': [1, 2, 3",  # truncated
        "[{'box_2d': [100, 200, 300, 400], 'label': 'item'}]",
    ])
    calls = []

    def request(*args):
        calls.append(args)
        return next(replies)

    with (patch.object(p, "_bbox_request", request),
          patch("agent_core.llm.time.sleep", return_value=None)):
        boxes = p._detect_boxes_px(object(), "target")
    configure_api_retries(DEFAULT_API_MAX_ATTEMPTS)
    assert len(calls) == 2
    assert boxes[0]["label"] == "item"


def test_invalid_bbox_shape_is_retried_and_eventually_raises():
    configure_api_retries(3)
    calls = []

    def request(*args):
        calls.append(args)
        return "[{'box_2d': [100, 200, 300]}]"

    with (patch.object(p, "_bbox_request", request),
          patch("agent_core.llm.time.sleep", return_value=None)):
        try:
            p._detect_boxes_px(object(), "target")
        except p.BBoxResponseParseError as error:
            assert "four-value box_2d" in str(error)
        else:
            raise AssertionError("malformed bbox response did not raise")
    configure_api_retries(DEFAULT_API_MAX_ATTEMPTS)
    assert len(calls) == 3


def test_center_surfaces_parse_exhaustion_as_detection_error():
    parse_error = p.BBoxResponseParseError("truncated response after retries")
    with (
        patch.object(p, "TransformAgent", return_value={}),
        patch.object(p, "RequestScreenshot", return_value={"image": _png_bytes()}),
        patch.object(p, "_detect_boxes_px", side_effect=parse_error),
    ):
        result = p.center_object_on_screen("target", max_iters=0)

    assert result["outcome"] == "detection_error"
    assert result["detected"] is False
    assert "not evidence that the target is absent" in result["center_message"]
    assert result["detection_error"] == str(parse_error)
