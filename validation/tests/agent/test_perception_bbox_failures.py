"""Regression tests for bbox parse failures being retried and surfaced distinctly."""

import os
import sys
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vision import perception as p
from agent_core.llm import (
    DEFAULT_API_MAX_ATTEMPTS, MalformedContentError, configure_api_retries,
    validate_json_schema,
)


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
    with pytest.raises(p.BBoxResponseParseError):
        p._bbox_payload("I am sorry, but I cannot fulfill this request.")


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


def test_truncated_bbox_is_rejected_without_repairing_coordinates():
    with pytest.raises(p.BBoxResponseParseError, match="never closed"):
        p._bbox_payload("[{'box_2d': [1, 2, 3")


def test_invalid_bbox_shape_and_range_are_rejected():
    with pytest.raises(p.BBoxResponseParseError, match="four-value"):
        p._bbox_dict_px({"box_2d": [100, 200, 300], "label": "item"})
    with pytest.raises(p.BBoxResponseParseError, match="between 0 and 1000"):
        p._bbox_dict_px({"box_2d": [-1, 200, 300, 400], "label": "item"})


@pytest.mark.parametrize(
    ("prompt", "maximum"),
    [(p.PERCEPTION_PROMPT, 1), (p.PERCEPTION_PROMPT_MULTI, 12)],
)
def test_bbox_request_uses_provider_aware_schema_with_expected_cardinality(prompt, maximum):
    captured = {}

    def complete(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            value=[], enforcement="native", completion=SimpleNamespace(text="[]")
        )

    with (
        patch.object(p, "_encode_image", return_value={"type": "image_url"}),
        patch.object(p, "structured_chat_completion", side_effect=complete),
    ):
        assert p._bbox_request(object(), "target", prompt, 400, 0.0) == []

    schema = captured["schema"]
    assert schema["minItems"] == 0
    assert schema["maxItems"] == maximum
    assert captured["provider"] == p._ENDPOINT_PROFILE.provider
    assert captured["signal_malformed_content_exhaustion"] is False
    validator = validate_json_schema(schema, provider=captured["provider"])
    validator.validate([{"label": "item", "box_2d": [0, 1, 999, 1000]}])
    with pytest.raises(Exception):
        validator.validate([{"box_2d": [0, 1, 999, 1000]}])
    with pytest.raises(Exception):
        validator.validate([{"label": "item", "box_2d": [0, 1, 999]}])


def test_bbox_plain_language_fallback_is_narrow_and_does_not_invent_coordinates():
    negative = "The image provided does not contain any Ritz Crackers."
    ambiguous = "I cannot fulfill this request."
    with (
        patch.object(p, "_encode_image", return_value={"type": "image_url"}),
        patch.object(
            p, "structured_chat_completion",
            side_effect=MalformedContentError("schema failure", content=negative),
        ),
    ):
        assert p._bbox_request(object(), "Ritz", p.PERCEPTION_PROMPT, 400, 0.0) == []
    with (
        patch.object(p, "_encode_image", return_value={"type": "image_url"}),
        patch.object(
            p, "structured_chat_completion",
            side_effect=MalformedContentError("schema failure", content=ambiguous),
        ),
        pytest.raises(MalformedContentError) as raised,
    ):
        p._bbox_request(object(), "Ritz", p.PERCEPTION_PROMPT, 400, 0.0)
    assert raised.value.content == ambiguous


def test_elongated_offcentre_box_preserves_each_provider_coordinate_order():
    raw = {"label": "elongated", "box_2d": [100, 200, 900, 400]}
    with patch.object(p, "BBOX_YMIN_FIRST", False):
        qwen = p._bbox_dict_px(raw)
    with patch.object(p, "BBOX_YMIN_FIRST", True):
        vertex = p._bbox_dict_px(raw)
    assert (qwen["cx"], qwen["cy"]) == pytest.approx((960, 324))
    assert (vertex["cx"], vertex["cy"]) == pytest.approx((576, 540))


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
