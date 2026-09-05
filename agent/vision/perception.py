import ast
import json
import random
import os
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from PIL import Image, ImageDraw
from io import BytesIO
import requests
import ast
import re
import math
from .ocr_client import OcrUnavailable, ocr_lines
# Repo-root secrets.env (agent/vision/ -> repo root is three parents up), resolved from __file__
# so it loads regardless of CWD.
load_dotenv(Path(__file__).resolve().parent.parent.parent / 'secrets.env')

# Agent runtime = the OpenAI-compatible endpoint from secrets.env (user directive 2026-07-19;
# OpenRouter retired on 402). This is the bounding-box/centering client - the endpoint's VL model
# replaces Gemini here, identically in BOTH A/B arms; bbox quality vs Gemini is unmeasured and
# shared, so it cannot skew the arms.
from agent_core.llm import (
    EndpointProfile, MalformedContentError, call_with_api_retries, effective_max_tokens,
    completion_result_from_response, image_url_part, structured_chat_completion,
)
from agent_core.prompt_loader import load_prompt, render_prompt
# Every LLM call in this module is perception's cost - bbox, centering and OCR-bbox reasoning alike.
# The role blocks wrap call_with_api_retries, not the lambda inside it, so a flaky call's retries are
# billed to perception as well; the endpoint charged for them.
from agent_core import token_meter
_ENDPOINT_PROFILE = EndpointProfile.from_env()
MODEL_NAME = _ENDPOINT_PROFILE.model
CLIENT = OpenAI(
    base_url=_ENDPOINT_PROFILE.base_url,
    api_key=_ENDPOINT_PROFILE.api_key,
    max_retries=0,
)
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080

# The simulator camera uses 60-degree vertical FOV (IK Humanoid Agent.prefab).
# For square pixels, both focal lengths are (H/2) / tan(vFOV/2).
# Convert pixel offsets with atan(offset/f), not a constant pixels-per-degree gain.
CAMERA_VFOV_DEG = 60.0
FOCAL_PX = (ORIGINAL_HEIGHT / 2.0) / math.tan(math.radians(CAMERA_VFOV_DEG / 2.0))

# Measured pixels-per-degree of camera rotation (phase-correlation, live, 2026-07-21): pitch
# matches the f=935 model (16.4 vs 16.3), yaw is ~11% hotter (18.1, parallax over the oblique
# shelf). Used ONLY to PREDICT where a tracked item lands after a rotation so the loop can
# re-lock the SAME instance next look - the correction angle itself still comes from the atan
# model above, damped by `gain`. Good-enough-for-tracking, not a precision constant.
PPD_YAW = 18.1
PPD_PITCH = 16.4
# Models emit trained bbox conventions regardless of prompt wording, normalized
# 0–1000: Qwen-VL uses [x1,y1,x2,y2]; Gemini/Vertex uses [y1,x1,y2,x2].
# Validate new models on elongated, off-center boxes: near-centered boxes can hide
# an axis swap. _bbox_dict_px selects the order by endpoint provider.
BBOX_YMIN_FIRST = _ENDPOINT_PROFILE.provider == "vertex"
PERCEPTION_PROMPT = load_prompt("vision/detect_one")
PERCEPTION_PROMPT_MULTI = load_prompt("vision/detect_many")
FIND_MOST_SIMILAR_OCR_BBOX_PROMPT = load_prompt("vision/match_ocr_box")

EXTRACTABLE_JSON_PATTERN = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

BBOX_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "minLength": 1},
        "box_2d": {
            "type": "array", "minItems": 4, "maxItems": 4,
            "items": {"type": "number", "minimum": 0, "maximum": 1000},
        },
    },
    "required": ["label", "box_2d"],
    "additionalProperties": False,
}


def _bbox_schema(max_entries):
    return {
        "type": "array", "minItems": 0, "maxItems": max_entries,
        "items": BBOX_ENTRY_SCHEMA,
    }

from sim.env import *
from manip.manipulation import *
# `import *` skips underscore names, so pull the per-step hand primitives scan_held_item needs
# explicitly (the public extend_left_hand_forward loops N steps without returning per-step state;
# the sweep needs the state each 0.025 m to detect the stall). Same lambdas the probe used.
from sim.env import _XTNFWD_LEFT_, _PLLBCK_LEFT_, _XTNFWD_RIGHT_, _PLLBCK_RIGHT_


def _ocr_lines(source):
    """Send a path/PIL image/PNG body to the central OCR service and return validated lines."""
    return ocr_lines(source)


def _encode_image(image: Image.Image) -> dict:
    buf = BytesIO()
    image.save(buf, format='PNG')
    return image_url_part(buf.getvalue(), "image/png")


def read_text(image_path=None):
    image_path = image_path or os.path.join(screenshot_dir(), "ClientScreenshot.png")
    return "\n".join(_ocr_lines(image_path))

def extract_text_from_image(image_path):
    # The second element was historically raw Paddle output. It is now the service's parsed line
    # list so no Paddle result shape leaks into an agent process.
    lines = _ocr_lines(image_path)
    return "\n".join(lines), lines

def find_most_similar_bbox_to_target_name(target_name, ocr_result):
    bboxes = '\n'.join([f'* {box}' for box in ocr_result])
    def request():
        response = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"{FIND_MOST_SIMILAR_OCR_BBOX_PROMPT}\n\ntarget_name={target_name}\n\n{bboxes}"}],
            temperature=0.5,
            max_tokens=effective_max_tokens(
                _ENDPOINT_PROFILE.provider, 400, "localization",
                _ENDPOINT_PROFILE.thinking_level,
            ),
            extra_body=_ENDPOINT_PROFILE.extra_body,
        )
        result = completion_result_from_response(
            response, provider=_ENDPOINT_PROFILE.provider,
            thinking_level=_ENDPOINT_PROFILE.thinking_level,
            workload="localization", requested_max_tokens=400,
        )
        if not result.text or result.finish_reason == "length":
            raise MalformedContentError(
                f"OCR box-match response was empty or truncated ({result.diagnostic()})",
                content=result.text, completion_result=result,
            )
        return result.text

    def validate(content):
        match = re.search(EXTRACTABLE_JSON_PATTERN, content or "")
        try:
            parsed = ast.literal_eval(match.group(1) if match else str(content or "").strip())
            box = parsed["box_2d"]
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError("box_2d must contain four coordinates")
        except (KeyError, TypeError, SyntaxError, ValueError) as error:
            raise MalformedContentError(
                f"OCR box-match response was malformed: {error}", content=content
            ) from error
        return box

    with token_meter.role(token_meter.ROLE_PERCEPTION):
        return call_with_api_retries(
            request,
            call_name="perception.ocr_box_match",
            validator=validate,
        )

def transform_paddle_result_to_coco_label_format(paddle_result):
    return [(b[0][0][0],b[0][0][1], b[0][2][0], b[0][2][1], b[1][0]) for b in paddle_result[0]]


def annotate_target(ymin, xmin, ymax, xmax, file_path=None,
                    source_image=None):
    """Draw the detected box on the screenshot (debug eyeballing only). Inputs are PIXEL coords -
    every caller passes pixels (bbox already scaled by ORIGINAL_W/H). The previous body re-divided
    by 1000 and re-multiplied by ORIGINAL_*, treating pixels as if normalised, so it drew a box
    shrunk ~1.08x/1.92x near the top-left and CRASHED PIL whenever an inverted box made y1<y0.
    Now: sort so min<=max and clamp to the image, so a malformed detection can't kill the run.

    ``source_image`` lets live callers annotate the exact in-memory frame they detected against.
    This matters when parallel benchmark workers share the legacy ClientScreenshot.png path.
    Annotation is debug-only, so an I/O failure is reported but must never abort an agent run."""
    try:
        if source_image is None:
            file_path = file_path or os.path.join(screenshot_dir(), "ClientScreenshot.png")
            with Image.open(file_path) as opened:
                opened.load()
                image = opened.copy()
        else:
            source_image.load()
            image = source_image.copy()
        draw = ImageDraw.Draw(image)
        W, H = image.size
        # Inputs are in the fixed ORIGINAL_WIDTH x ORIGINAL_HEIGHT virtual frame (detections are
        # 0-1000 normalised * ORIGINAL_*), so scale to the ACTUAL frame before drawing. Without this
        # a higher-res screenshot paints the box/crosshair in the top-left quadrant.
        sx, sy = W / ORIGINAL_WIDTH, H / ORIGINAL_HEIGHT
        x0, x1 = sorted((int(xmin * sx), int(xmax * sx)))
        y0, y1 = sorted((int(ymin * sy), int(ymax * sy)))
        x0, x1 = max(0, min(x0, W - 1)), max(0, min(x1, W - 1))
        y0, y1 = max(0, min(y0, H - 1)), max(0, min(y1, H - 1))
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, max(0, y0 - 12)), "Target", fill="red")
        # Debug-only annotated frame: cap on disk at 1080p (drawn at capture res, which may be
        # 4K). Distributed benchmark archives use a compact, atomic JPEG; standalone diagnostics
        # retain their historical PNG name and bytes.
        if benchmark_artifact_mode():
            out_path = artifact_path("screenshots", "annotated_target.jpg", legacy_base="")
            save_jpeg_atomic(out_path, image, quality=85)
        else:
            out_path = artifact_path("screenshots", "annotated_target.png", legacy_base="")
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            downscale_pil_for_storage(image).save(out_path)
    except Exception as e:
        print(f"[perception] target annotation skipped: {type(e).__name__}: {e}")


def _draw_debug_frame(frame_source, boxes, chosen, aim_xy, out_path):
    """Debug-only: save `frame_source` with EVERY VLM candidate box (thin yellow), the chosen /
    tracked instance (thick red), and the aim crosshair (green). This is the "what did the
    detector actually return, and which one are we driving to centre" picture. Best-effort - a
    drawing error must never take down a centring run."""
    try:
        if isinstance(frame_source, Image.Image):
            frame_source.load()
            img = frame_source.convert("RGB")
        else:
            with Image.open(frame_source) as opened:
                opened.load()
                img = opened.convert("RGB")
        draw = ImageDraw.Draw(img)
        # boxes/aim are in the fixed ORIGINAL_WIDTH x ORIGINAL_HEIGHT virtual frame; scale to the
        # ACTUAL screenshot so the crosshair sits at true centre at any capture resolution (a 4K
        # frame would otherwise draw the green aim at 960,540 - its top-left quadrant, not centre).
        W, H = img.size
        sx, sy = W / ORIGINAL_WIDTH, H / ORIGINAL_HEIGHT
        for b in boxes:
            draw.rectangle([b['xmin'] * sx, b['ymin'] * sy, b['xmax'] * sx, b['ymax'] * sy],
                           outline="yellow", width=2)
        if chosen is not None:
            draw.rectangle([chosen['xmin'] * sx, chosen['ymin'] * sy,
                            chosen['xmax'] * sx, chosen['ymax'] * sy], outline="red", width=4)
            draw.text((chosen['xmin'] * sx, max(0, chosen['ymin'] * sy - 13)), "locked", fill="red")
        ax, ay = int(aim_xy[0] * sx), int(aim_xy[1] * sy)
        draw.line([(ax - 28, ay), (ax + 28, ay)], fill="lime", width=3)
        draw.line([(ax, ay - 28), (ax, ay + 28)], fill="lime", width=3)
        # Debug-only frame: cap on disk at 1080p (the detection ran on the full-res `image`, not
        # this). The caller chooses a .jpg path in distributed benchmark mode.
        if benchmark_artifact_mode():
            save_jpeg_atomic(out_path, img, quality=85)
        else:
            downscale_pil_for_storage(img).save(out_path)
    except Exception as e:
        print(f"[CENTER] debug frame save failed: {type(e).__name__}: {e}")


class BBoxResponseParseError(MalformedContentError):
    """The VLM replied, but its bbox payload was malformed or truncated."""


_BBOX_NEGATIVE_PROSE_PATTERNS = (
    # Common VLM refusal wrapper followed by a genuine visual negative, e.g. "The image provided
    # does not contain any Ritz Crackers." Keep these patterns deliberately narrow: arbitrary prose
    # is still malformed and retried rather than silently becoming evidence that the target is gone.
    re.compile(
        r"\b(?:the\s+)?image(?:\s+provided)?\s+(?:does\s+not|doesn't)\s+"
        r"(?:contain|show|include|depict)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?(?:target|requested|specified)\s+(?:object|item)?\s*"
        r"(?:is|was)\s+not\s+(?:visible|present|found|detected)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:matching\s+)?(?:target|object|item)s?\s+"
        r"(?:is|are|was|were)\s+(?:visible|present|found|detected)\b",
        re.IGNORECASE,
    ),
)


def _is_bbox_negative_prose(text):
    """Return whether non-JSON prose unambiguously says the target is absent from the image."""
    return any(pattern.search(text) for pattern in _BBOX_NEGATIVE_PROSE_PATTERNS)


def _bbox_payload(content):
    """Parse a complete JSON/Python-literal bbox payload.

    A valid ``[]`` or an unambiguous plain-language visual negative is a negative detection result.
    Invalid or truncated output raises so it cannot be confused with "target not visible".
    """
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return content
    if not isinstance(content, str) or not content.strip():
        raise BBoxResponseParseError("empty VLM response")
    text = content.strip()
    fenced = re.fullmatch(r"```\s*(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Qwen commonly emits single-quoted Python literals despite the JSON instruction. Keep
        # supporting those, while parsing actual JSON first (including its null/true/false values).
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as error:
            if _is_bbox_negative_prose(text):
                return []
            raise BBoxResponseParseError(
                f"bbox payload is not complete valid JSON/literal: {error}"
            ) from error
    if parsed == []:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if not isinstance(parsed, list) or not parsed:
        raise BBoxResponseParseError(
            f"bbox payload must be an object, non-empty list, or [] (got {type(parsed).__name__})"
        )
    return parsed


def _bbox_dict_px(box_2d):
    if not isinstance(box_2d, dict):
        raise BBoxResponseParseError(
            f"bbox entry must be an object (got {type(box_2d).__name__})"
        )
    coords = box_2d.get('box_2d') or box_2d.get('bbox_2d')
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        raise BBoxResponseParseError("bbox entry is missing a four-value box_2d")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in coords):
        raise BBoxResponseParseError("bbox coordinates must be finite numbers")
    if any(value < 0 or value > 1000 for value in coords):
        raise BBoxResponseParseError("bbox coordinates must be between 0 and 1000")
    if BBOX_YMIN_FIRST:
        ymin = coords[0] / 1000 * ORIGINAL_HEIGHT
        xmin = coords[1] / 1000 * ORIGINAL_WIDTH
        ymax = coords[2] / 1000 * ORIGINAL_HEIGHT
        xmax = coords[3] / 1000 * ORIGINAL_WIDTH
    else:
        xmin = coords[0] / 1000 * ORIGINAL_WIDTH
        ymin = coords[1] / 1000 * ORIGINAL_HEIGHT
        xmax = coords[2] / 1000 * ORIGINAL_WIDTH
        ymax = coords[3] / 1000 * ORIGINAL_HEIGHT
    if xmin >= xmax or ymin >= ymax:
        raise BBoxResponseParseError("bbox coordinates are empty or reversed")
    return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'cx': (xmin + xmax) / 2.0, 'cy': (ymin + ymax) / 2.0,
            'label': box_2d.get('label', 'unknown')}


def _bbox_request(image, target_info, prompt, max_tokens, temperature):
    messages = [{"role": "user", "content": [
        _encode_image(image),
        {"type": "text", "text": f"{prompt}\n\ntarget_info={target_info}\n\n"},
    ]}]
    max_entries = 1 if prompt == PERCEPTION_PROMPT else 12
    try:
        structured = structured_chat_completion(
            client=CLIENT,
            provider=_ENDPOINT_PROFILE.provider,
            thinking_level=_ENDPOINT_PROFILE.thinking_level,
            default_extra_body=_ENDPOINT_PROFILE.extra_body,
            messages=messages,
            schema=_bbox_schema(max_entries),
            schema_name=f"bounding_boxes_{max_entries}",
            model=MODEL_NAME,
            temperature=temperature,
            max_tokens=max_tokens,
            workload="localization",
            call_name="perception.bounding_boxes",
            signal_malformed_content_exhaustion=False,
        )
    except MalformedContentError as error:
        if isinstance(error.content, str) and _is_bbox_negative_prose(error.content):
            return []
        raise
    print(
        "[DETECT OBJECT IN FRAME] "
        f"response_source={structured.enforcement} Response: {structured.completion.text}"
    )
    return structured.value


def _parsed_bbox_response(image, target_info, prompt, max_tokens, temperature):
    with token_meter.role(token_meter.ROLE_PERCEPTION):
        try:
            content = _bbox_request(image, target_info, prompt, max_tokens, temperature)
            return [_bbox_dict_px(entry) for entry in _bbox_payload(content)]
        except BBoxResponseParseError as error:
            detail = str(error).lower()
            category = "truncated" if "never closed" in detail else "schema_or_syntax"
            print(f"[BBOX MALFORMED] category={category} error={error}")
            raise
        except MalformedContentError as error:
            content = error.content
            category = (
                "empty" if not content else
                "truncated" if "truncated" in str(error).lower() else
                "schema_or_syntax"
            )
            print(f"[BBOX MALFORMED] category={category} error={error}")
            raise BBoxResponseParseError(
                str(error), content=content, completion_result=error.completion_result,
            ) from error


def _detect_bbox_px(image, target_info, temperature=0.0):
    """Detect ONE target box and return it in PIXELS as
    {'xmin','ymin','xmax','ymax','cx','cy','label'}, or None for a negative detection response.
    Malformed/truncated responses are retried, then raise BBoxResponseParseError.

    temperature defaults to 0.0 on purpose: this is a LOCALISATION call, not a creative
    one. At the old 0.5 the same frame returned different coordinates run to run, which a
    closed centring loop would chase as if the object itself were moving. Deterministic box
    in, deterministic correction out. Pulled out of center_object_on_screen so it (and the
    projection math) can be exercised offline on saved PNGs - see center_offline_check.py."""
    entries = _parsed_bbox_response(
        image, target_info, PERCEPTION_PROMPT, 400, temperature
    )
    return entries[0] if entries else None


def _detect_boxes_px(image, target_info, temperature=0.0):
    """ALL matching instances as a list of box dicts (same shape as _detect_bbox_px's return);
    [] for a valid structured or plain-language negative response. Malformed/truncated responses
    are retried, then raise
    BBoxResponseParseError. Lets the centring loop pick and TRACK one instance instead of taking whatever
    single box the model happens to return - the fix for the loop hopping between identical
    items on a dense shelf (measured 2026-07-21). temperature 0.0 for the same reason as
    _detect_bbox_px: a stable candidate set look to look."""
    entries = _parsed_bbox_response(
        image, target_info, PERCEPTION_PROMPT_MULTI, 1500, temperature
    )
    return entries


def bbox_to_rotation(cx, cy, aim_x, aim_y, focal_px=FOCAL_PX):
    """Camera (pitch, yaw) in DEGREES that swings the bbox centre (cx, cy) onto the aim
    point (aim_x, aim_y): pinhole/atan model, one focal length for both axes. Signs match
    the _PAN_RIGHT_/_TILT_DOWN_ primitives in env.py - object right-of-aim -> +yaw (pan
    right), object below-aim -> +pitch (tilt down) - so the pair drops straight into
    TransformAgent((0,0,0), (pitch, yaw, 0))."""
    yaw = math.degrees(math.atan2(cx - aim_x, focal_px))
    pitch = math.degrees(math.atan2(cy - aim_y, focal_px))
    return pitch, yaw


def _seed_front_instance(boxes, aim_x, aim_y, front_bias):
    """Choose an initial box by distance to aim, biased toward larger apparent area.

    Area is a depth proxy only among similar-sized products. Excessive front_bias
    can select a large off-target box; zero gives pure nearest-to-aim selection.
    The default bias needs validation on saved frames. Later looks track the chosen
    instance by predicted position rather than reseeding.
    """
    if front_bias <= 0.0 or len(boxes) == 1:
        return min(boxes, key=lambda b: (b['cx'] - aim_x) ** 2 + (b['cy'] - aim_y) ** 2)
    diag2 = float(ORIGINAL_WIDTH ** 2 + ORIGINAL_HEIGHT ** 2)
    frame_area = float(ORIGINAL_WIDTH * ORIGINAL_HEIGHT)

    def _score(b):
        d2 = ((b['cx'] - aim_x) ** 2 + (b['cy'] - aim_y) ** 2) / diag2
        area = max(0.0, (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin'])) / frame_area
        return d2 - front_bias * area

    return min(boxes, key=_score)


def center_object_on_screen(target_info, aim_norm=(0.5, 0.5), max_iters=5, tol_px=20.0,
                            gain=0.8, max_step_deg=12.0, front_bias=0, debug_dir=None):
    """Rotate toward the target with repeated detection and bounded atan-model corrections.

    Allow max_iters corrections plus a final measurement. gain damps overshoot;
    max_step_deg limits view changes that could break the instance lock. Seed with
    _seed_front_instance, then track the box nearest its predicted position.

    aim_norm is the normalized image aim (x, y). Centering moves the camera only:
    low targets may still require crouching or hand-height adjustment to grab.
    The front_bias default needs validation; zero selects purely by distance to aim.

    If debug_dir is set, save each look with candidates, the locked box, and aim.
    Return the simulator state plus centered, detected, residual_px, iters, outcome,
    center_message, box, and detection_error. Box coordinates use the virtual
    ORIGINAL_WIDTH x ORIGINAL_HEIGHT frame; callers can reuse them for cropping/OCR.
    """
    aim_x = aim_norm[0] * ORIGINAL_WIDTH
    aim_y = aim_norm[1] * ORIGINAL_HEIGHT
    state = TransformAgent((0, 0, 0), (0, 0, 0))  # read current pose (no-op move)
    residual = (None, None)
    centered = False
    detected = False        # did the detector ever return the target?
    detection_error = None  # malformed VLM output after detector-level parse retries
    stalled = False         # did the residual stop shrinking (target likely at the frame edge)?
    locked = None           # predicted (x, y) of the tracked instance; None until look 1 picks one
    box = None              # last locked box (xmin/ymin/xmax/ymax/cx/cy/label); None if never detected
    image = None            # exact final frame; private handoff for region OCR
    prev_mag = None
    no_improve = 0
    i = 0
    for i in range(1, max_iters + 2):   # max_iters rotations + one measurement-only look
        screenshot = RequestScreenshot(save_image=True)
        image = Image.open(BytesIO(screenshot["image"]))
        image.load()
        try:
            boxes = _detect_boxes_px(image, target_info)
        except BBoxResponseParseError as error:
            detection_error = str(error)
            print(f"[CENTER] look {i}: detector parse failure - {error}")
            break
        if not boxes:
            print(f"[CENTER] look {i}: target not detected - not rotating (let the caller search).")
            break
        detected = True
        # Stay on ONE instance. Look 1 SEEDS on the FRONT-of-row instance (front_bias), not merely
        # the box nearest the aim - a stacked row of identical items would otherwise sometimes lock
        # the one behind. Every look after tracks THAT instance: nearest to where it was predicted
        # to land. This is what stops the hop between identical items as they cross near the aim.
        if locked is None:
            box = _seed_front_instance(boxes, aim_x, aim_y, front_bias)
        else:
            box = min(boxes, key=lambda b: (b['cx'] - locked[0]) ** 2 + (b['cy'] - locked[1]) ** 2)
        annotate_target(box['ymin'], box['xmin'], box['ymax'], box['xmax'],
                        source_image=image)
        if debug_dir:
            suffix = ".jpg" if benchmark_artifact_mode() else ".png"
            _draw_debug_frame(image, boxes, box, (aim_x, aim_y),
                              os.path.join(debug_dir, f"look{i}_bbox{suffix}"))
        dx, dy = box['cx'] - aim_x, box['cy'] - aim_y
        residual = (round(dx, 1), round(dy, 1))
        print(f"[CENTER] look {i}: '{box['label']}' center=({box['cx']:.0f},{box['cy']:.0f}) "
              f"residual=({dx:+.0f},{dy:+.0f})px  ({len(boxes)} candidate(s))")
        if abs(dx) <= tol_px and abs(dy) <= tol_px:
            centered = True
            print(f"[CENTER] within tolerance ({tol_px:.0f}px) in {i} look(s).")
            break
        # Stall guard: if the residual stops shrinking for two looks the target is likely at the
        # frame edge (detection jitter, not a gain error). Stop instead of grinding out the budget
        # - that wasted looping is what an onlooker/learner mislabels as "centring keeps failing".
        mag = (dx * dx + dy * dy) ** 0.5
        no_improve = no_improve + 1 if (prev_mag is not None and mag >= prev_mag - 3.0) else 0
        prev_mag = mag
        if no_improve >= 2:
            stalled = True
            print(f"[CENTER] residual stopped improving ({residual}px) - stopping (target likely at frame edge).")
            break
        if i > max_iters:
            print(f"[CENTER] out of rotation budget ({max_iters}); residual {residual}px remains.")
            break
        pitch, yaw = bbox_to_rotation(box['cx'], box['cy'], aim_x, aim_y)
        pitch, yaw = pitch * gain, yaw * gain
        # Cap the per-step swing: a big rotation changes the view so much the detector returns a
        # different (smaller) candidate set and the instance lock jumps to a neighbour. Bounded
        # steps keep enough of the scene stable for the nearest-to-prediction re-lock to hold;
        # a far target just takes a few more looks (measured 2026-07-21).
        pitch = max(-max_step_deg, min(max_step_deg, pitch))
        yaw = max(-max_step_deg, min(max_step_deg, yaw))
        # Predict where THIS instance lands so the next look re-locks it, not a neighbour
        # (pan right -> item moves left, tilt down -> item moves up; measured px/deg).
        locked = (box['cx'] - yaw * PPD_YAW, box['cy'] - pitch * PPD_PITCH)
        print(f"[CENTER] rotate pitch={pitch:+.2f} yaw={yaw:+.2f} deg (gain {gain})")
        state = TransformAgent((0, 0, 0), (pitch, yaw, 0))

    # Outcome surfaced to the agent as `last_center` (see the runner loops): the actor and the
    # episodic learner must KNOW whether centring worked. Without it a silent success gets guessed
    # at - the learner once concluded "avoid center_object_on_screen" from a centre that actually
    # succeeded. The failure strings are actionable so the lesson is "get the target in view",
    # never "stop using the tool".
    if centered:
        outcome = "success"
        message = f"SUCCESS - target centred (residual {residual}px, {i} look(s))"
    elif detection_error is not None:
        outcome = "detection_error"
        message = (
            "ERROR - target detection could not be parsed after retries; this is a VLM response "
            "failure, not evidence that the target is absent. Retry center_object_on_screen"
        )
    elif not detected:
        outcome = "not_detected"
        message = ("FAILED - target not detected in the frame; tilt/pan to bring it into view, "
                   "then center again (do not abandon center_object_on_screen)")
    elif stalled:
        outcome = "stalled"
        message = (f"STALLED - centring stopped improving at {residual}px; the target is likely near "
                   "the frame edge - bring it more into view, then center again")
    else:
        outcome = "incomplete"
        message = (f"INCOMPLETE - detected but not centred within tolerance (residual {residual}px "
                   f"after {i} looks); move a little closer or center again")
    print(f"[CENTER] result: {message}")
    state = dict(state)
    state.update({'centered': centered, 'detected': detected, 'residual_px': residual,
                  'iters': i, 'outcome': outcome, 'center_message': message, 'box': box,
                  'detection_error': detection_error,
                  '_source_image': image})
    return state


# Fixed surface targets keep checkout centering consistent across callers.
# Counter aim is provisional; probe below-center aiming if LiDAR hits the front lip.
# Zero front bias avoids favoring products sitting on the surface.
COUNTER_TARGET_INFO = ("main_goal=the checkout counter surface - the flat countertop you set items "
                       "down on, not the items on it "
                       "The top surface of the counter, not the entire counter.")
COUNTER_AIM_NORM = (0.5, 0.5)   # provisional; place_probe measures whether the aim should sit lower


def center_to_counter(aim_norm=COUNTER_AIM_NORM, front_bias=0.0, **kwargs):
    """Center on the counter with a fixed target and aim.

    Keep carrying hands active at REST; disabling them drops the item.
    """
    return center_object_on_screen(COUNTER_TARGET_INFO, aim_norm=aim_norm,
                                   front_bias=front_bias, **kwargs)


# Scan alignment targets the pad specifically, excluding the screen and bagging tray.
# Wording and aim remain provisional until validated with place_probe's scanner command.
SCANNER_TARGET_INFO = ("main_goal=the barcode scanner on the checkout counter - the black square "
                       "scan pad below the scanner head with the red LED window. Not the POS "
                       "screen, not the bagging tray, and not any product.")
SCANNER_AIM_NORM = (0.5, 0.5)   # provisional; place_probe `scanner [y]` dials it like `center [y]` did


def center_to_scanner(aim_norm=SCANNER_AIM_NORM, front_bias=0.0, **kwargs):
    """Center on the scan pad. Keep carrying hands active at REST."""
    return center_object_on_screen(SCANNER_TARGET_INFO, aim_norm=aim_norm,
                                   front_bias=front_bias, **kwargs)


# Target the recessed tray: centering the whole counter can land on the scanner pad.
# Tray wording/aim need live validation. PLACE_ENVELOPE was measured against the
# counter surface and must be remeasured for this target.
TRAY_TARGET_INFO = ("main_goal=the bagging tray - the empty recessed rectangular bin next to the "
                    "self-checkout where you drop bagged items. Not the flat counter top, not the "
                    "black barcode scan pad, not the POS screen, and not any product.")
TRAY_AIM_NORM = (0.5, 0.5)   # provisional; dial deeper (y>0.5) if releases catch the tray's near lip


def center_to_tray(aim_norm=TRAY_AIM_NORM, front_bias=0.0, **kwargs):
    """Center on the tray. Keep carrying hands active at REST.

    Validate the provisional aim and remeasure the place envelope for this target.
    """
    return center_object_on_screen(TRAY_TARGET_INFO, aim_norm=aim_norm,
                                   front_bias=front_bias, **kwargs)


# The scanner has no WebSocket status query; use the POS receipt as evidence.
# OCR only the screen bbox to exclude shelf/product text. Validate the provisional
# target wording with place_probe's POS screen probe.
SCREEN_TARGET_INFO = ("main_goal=the self-checkout POS screen - the bright rectangular monitor "
                      "showing the receipt / item list and the green Start button. Not the barcode "
                      "scan pad, not the bagging tray, not any product.")
SCREEN_AIM_NORM = (0.5, 0.5)


def center_to_screen(aim_norm=SCREEN_AIM_NORM, front_bias=0.0, **kwargs):
    """center_object_on_screen bound to the POS screen: fixed target + aim. Its returned 'box' is
    the whole point here - pass it to read_text_in_box to OCR the receipt inside its own rectangle
    rather than the whole frame. Callers are CARRYING by construction (you read the screen to
    confirm the held item scanned); do NOT stow the hands. Wording is provisional until M5."""
    return center_object_on_screen(SCREEN_TARGET_INFO, aim_norm=aim_norm,
                                   front_bias=front_bias, **kwargs)


def read_text_in_box(box, pad_frac=0.04, image_path=None, source_image=None):
    """OCR only the region under `box` (a center_object_on_screen 'box' dict, whose
    xmin/ymin/xmax/ymax are in the ORIGINAL_WIDTH x ORIGINAL_HEIGHT virtual frame). Scales the box
    to the actual image size the same way annotate_target does, pads it by `pad_frac` of its size so
    a slightly-tight detection doesn't clip the edge glyphs, crops, and sends a PNG to the OCR API.

    Cropping first is the whole win over read_text(): the POS receipt is a small bright rectangle in
    a busy frame, and full-frame OCR mixes in shelf/label/button text. Live callers pass
    ``source_image`` from the centering result, so OCR cannot race another screenshot writer. The
    attempt-local saved frame is retained as a compatibility fallback for standalone callers.

    FAILURE CONTRACT — CHANGED, deliberately. This used to catch Exception, print
    ``[OCR] read_text_in_box failed (...)`` and return [], on the rule "OCR must never take down the
    caller". That rule was wrong here, because [] is not a neutral value for the callers: the
    checkout path (_fuzzy_new_lines, store_map's scan confirmation, validation acceptance smoke) diffs
    receipt lines against a baseline, and an empty read is indistinguishable from "nothing new was
    scanned". Unavailable OCR therefore must fail the run rather than silently converting the
    MEASURED scan signal into a permanent "no scan detected", i.e. exactly the fake negative the
    measure-don't-assume rule exists to prevent. So OCR failure now propagates: an empty [] returned
    from here means paddle genuinely read no text, and nothing else.

    A genuinely absent box still returns [] — "no screen was centred" is a real, non-erroneous
    answer that the caller already branches on."""
    if not box:
        return []
    try:
        if source_image is not None:
            source_image.load()
            img = source_image.convert("RGB")
        else:
            image_path = image_path or os.path.join(screenshot_dir(), "ClientScreenshot.png")
            with Image.open(image_path) as opened:
                opened.load()
                img = opened.convert("RGB")
        W, H = img.size
        sx, sy = W / ORIGINAL_WIDTH, H / ORIGINAL_HEIGHT
        x0, x1 = sorted((box['xmin'] * sx, box['xmax'] * sx))
        y0, y1 = sorted((box['ymin'] * sy, box['ymax'] * sy))
        px, py = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
        x0, x1 = max(0, int(x0 - px)), min(W, int(x1 + px))
        y0, y1 = max(0, int(y0 - py)), min(H, int(y1 + py))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return []
        crop = img.crop((x0, y0, x1, y1))
        # The crop stays in memory and is encoded as PNG by the API-only client. No scratch-file
        # handoff and no process-local Paddle model exist in an agent.
        return _ocr_lines(crop)
    except Exception as e:
        raise OcrUnavailable(
            f"read_text_in_box failed ({type(e).__name__}: {e}). Region OCR is REQUIRED by the "
            f"callers that diff receipt lines - returning no text here would silently read as "
            f"'nothing scanned'. Start the service with `uv run poe ocr-server` "
            f"(or `ocr-server-directml`)."
        ) from e


def _fuzzy_new_lines(before, after, ratio=0.8):
    """Multiset diff of OCR line lists tolerant of per-frame character jitter. A line in `after`
    counts as NEW iff no not-yet-consumed line in `before` matches it with a difflib ratio >= `ratio`.
    Consuming matches keeps multiset semantics (a second identical receipt line is genuinely new),
    while the fuzzy compare stops OCR noise (`JIN_RAKEN` one frame, `JIN_RAMEN` the next) from
    re-counting an already-present line as new - the exact-string bug the probe exposed. PURE, so it
    is unit-tested offline (tests/test_scan_diff.py)."""
    import difflib
    pool = list(before)
    new = []
    for a in after:
        hit = -1
        best = ratio
        for idx, b in enumerate(pool):
            r = difflib.SequenceMatcher(None, a, b).ratio()
            if r >= best:
                best, hit = r, idx
        if hit >= 0:
            pool.pop(hit)          # consumed - a real duplicate later still reads as new
        else:
            new.append(a)
    return new


def scan_held_item(hand="auto", baseline=None, max_extend=25, fuzzy_ratio=0.8, debug_dir=None):
    """Sweep a held item through the aligned scan pad without opening the grip.

    The caller must align first; the sweep moves only the hand, then restores REST
    and centers the POS screen. A fuzzy receipt delta supplies the scanned verdict.
    Pass the returned receipt as the next item's baseline; None assumes an empty
    checkout. The OCR verdict is separate from human scan verification.

    Return scanned, still_holding, receipt, new_lines, screen_detected, reason, and hand.
    """
    # 'auto' resolves to the hand that IS holding (left-first when both hold - name 'right' to scan
    # the other item); an explicit empty hand is refused. Subsumes the old empty-hand guard.
    from manip.manipulation import resolve_release_hand
    hand, refuse_reason = resolve_release_hand(hand)
    if hand is None:
        print(f"[scan_held_item] REFUSED: {refuse_reason}")
        return {"scanned": False, "still_holding": False, "receipt": [], "new_lines": [],
                "screen_detected": False, "reason": refuse_reason + " - nothing to scan"}
    extend_fn = _XTNFWD_LEFT_ if hand == "left" else _XTNFWD_RIGHT_
    pull_fn = _PLLBCK_LEFT_ if hand == "left" else _PLLBCK_RIGHT_
    grip_key = "leftGrippedState" if hand == "left" else "rightGrippedState"
    trans_key = "leftTranslation" if hand == "left" else "rightTranslation"

    def _hand_state():
        return TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

    set_hand_pose("grab", hand=hand)
    moved = 0
    try:
        prev = _hand_state()[trans_key]
        for _ in range(max_extend):                # full extension; the stall guard stops earlier
            cur = extend_fn()[trans_key]
            if sum((cur[k] - prev[k]) ** 2 for k in range(3)) ** 0.5 <= 1e-4:
                break                              # hand clamped at its reach limit
            prev = cur
            moved += 1
    finally:
        for _ in range(moved):                     # retract exactly as far as we extended
            pull_fn()
        set_hand_pose("rest", hand=hand)

    still_holding = bool(_hand_state().get(grip_key))

    # Measured confirmation: centre the POS screen, region-OCR the receipt, diff vs the baseline.
    res = center_to_screen(debug_dir=debug_dir)
    box = res.get("box")
    receipt = read_text_in_box(box, source_image=res.get("_source_image")) if box else []
    new = _fuzzy_new_lines(baseline or [], receipt, ratio=fuzzy_ratio)
    scanned = any(any(c.isdigit() for c in l) for l in new)

    if not box:
        reason = ("swept; POS screen not detected so the scan is UNCONFIRMED (measured channel "
                  "unavailable) - re-centre the screen or verify by eye")
    elif scanned:
        reason = f"scanned (new receipt line: {new[0]})"
    else:
        reason = ("swept but no new receipt line - not aligned on the pad, or out of the scan "
                  "zone; re-run align_to_scanner and retry")
    if not still_holding:
        reason = "the grip OPENED during the sweep (should not happen) - " + reason

    print(f"[scan_held_item] hand={hand} scanned={scanned} still_holding={still_holding} "
          f"screen_detected={bool(box)} new_lines={new}")
    return {"scanned": scanned, "still_holding": still_holding, "receipt": receipt,
            "new_lines": new, "screen_detected": bool(box), "reason": reason, "hand": hand}


# Simulator hands can clip through surfaces. Release 0.1 m before the LiDAR distance;
# scan sweeps still extend fully because they keep the grip closed.
PLACE_CLIP_STANDOFF_M = 0.1

# Accept the frame-center LiDAR ray within 60% of the tray's half-extents,
# leaving a 40% margin to each lip. Larger values permit riskier off-center drops.
PLACE_TRAY_EDGE_MARGIN = 0.6


def _ray_in_box(box, edge_margin):
    """True if the LiDAR sample point (frame centre) falls within the inner `edge_margin` fraction of
    `box`. Robust to aim_norm: computed from the box + the fixed frame centre, not the residual."""
    if not box:
        return False
    ray_x, ray_y = ORIGINAL_WIDTH / 2.0, ORIGINAL_HEIGHT / 2.0
    half_w = max(1.0, (box["xmax"] - box["xmin"]) / 2.0)
    half_h = max(1.0, (box["ymax"] - box["ymin"]) / 2.0)
    return (abs(ray_x - box["cx"]) <= half_w * edge_margin
            and abs(ray_y - box["cy"]) <= half_h * edge_margin)


def place_held_item(hand="auto", aim_norm=None, max_approach_iters=4, max_extend=25,
                    clip_standoff=PLACE_CLIP_STANDOFF_M, edge_margin=PLACE_TRAY_EDGE_MARGIN,
                    debug_dir=None):
    """Center on the tray, approach within the place envelope, and release a held item.

    Requires a scanned item at the counter. Re-center after each approach step.
    The LiDAR ray must lie inside the tray's edge margin; repeated lock failures or
    non-approachable verdicts abort while holding. Extend to distance - clip_standoff
    (or an earlier stall), release, retract, and restore REST.

    placed records release under a placeable verdict, not proof of landing in the
    tray; visual verification is separate. The envelope was measured on the counter
    surface and needs remeasurement for the tray, especially after aim changes.

    Return placed, released, verdict, distance, surface_height, iters, and reason.
    """
    from manip.manipulation import plan_place, PLACE_ENVELOPE, set_hand_pose as _set_pose, resolve_release_hand
    # 'auto' resolves to the hand that IS holding (left-first when both hold - name 'right' to place
    # the other item); an explicit empty hand is refused. Subsumes the old empty-hand guard.
    hand, refuse_reason = resolve_release_hand(hand)
    if hand is None:
        print(f"[place_held_item] REFUSED: {refuse_reason}")
        return {"placed": False, "released": False, "verdict": "no_item", "distance": None,
                "surface_height": None, "iters": 0, "reason": refuse_reason + " - nothing to place"}
    extend_fn = _XTNFWD_LEFT_ if hand == "left" else _XTNFWD_RIGHT_
    pull_fn = _PLLBCK_LEFT_ if hand == "left" else _PLLBCK_RIGHT_
    grip_fn = ToggleLeftGrip if hand == "left" else ToggleRightGrip
    grip_key = "leftGrippedState" if hand == "left" else "rightGrippedState"
    trans_key = "leftTranslation" if hand == "left" else "rightTranslation"
    aim = aim_norm or TRAY_AIM_NORM

    def _hand_state():
        return TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

    # ---- Approach: centre the tray, depth-gate the standoff, creep until 'placeable' -------------
    # The release ONLY fires on a 'placeable' verdict taken under a SUCCESSFUL centre (ready=True).
    # A stalled/incomplete centre means the LiDAR ray is not reliably on the tray, so we do NOT plan a
    # move or a release off it - we re-centre; if it never locks, the loop exhausts and we abort holding.
    plan = None
    iters = 0
    ready = False
    for iters in range(1, max_approach_iters + 1):
        res = center_to_tray(aim_norm=aim, debug_dir=debug_dir)
        box = res.get("box")
        if res.get("outcome") == "detection_error":
            return {"placed": False, "released": False, "verdict": "detection_error",
                    "distance": None, "surface_height": None, "iters": iters,
                    "reason": "bagging-tray detector returned malformed output after retries; "
                    "target visibility is unknown, so nothing was released"}
        if res.get("outcome") == "not_detected" or box is None:
            return {"placed": False, "released": False, "verdict": "not_detected", "distance": None,
                    "surface_height": None, "iters": iters,
                    "reason": "bagging tray not detected - re-approach the counter or pan it into view"}
        if not _ray_in_box(box, edge_margin):
            # The drop point (frame centre) is out near the tray lip / off the box - not dead-centre is
            # fine, but this is too far. Re-centre; do NOT release here.
            off = (round(ORIGINAL_WIDTH / 2.0 - box["cx"]), round(ORIGINAL_HEIGHT / 2.0 - box["cy"]))
            print(f"  [place] drop point near the tray edge (ray offset {off} px, outcome "
                  f"{res.get('outcome')}) - re-centring (iter {iters}/{max_approach_iters}), NOT releasing")
            continue
        plan = plan_place(RequestLidarCenter(), PLACE_ENVELOPE)
        v = plan["verdict"]
        if v == "placeable":
            ready = True
            break
        if v == "move":
            move_forward(plan["move_steps"]); continue
        if v == "back":
            move_backward(plan["move_steps"]); continue
        # crouch / bail / recenter / unavailable: don't release off the tray.
        return {"placed": False, "released": False, "verdict": v, "distance": plan["distance"],
                "surface_height": plan["surface_height"], "iters": iters,
                "reason": f"cannot place from here: {plan['reason']}"}
    if not ready:
        last_v = plan["verdict"] if plan else "centre_never_locked"
        return {"placed": False, "released": False, "verdict": last_v,
                "distance": plan["distance"] if plan else None,
                "surface_height": plan["surface_height"] if plan else None, "iters": iters,
                "reason": f"not placeable after {max_approach_iters} approach step(s) (last: {last_v}) "
                          f"- still holding rather than drop off the tray"}

    # ---- Release: extend toward the tray but STOP 0.1 m short of the surface, then open the grip ---
    # target_z is the hand-forward depth to release at: (centre distance - clip_standoff). Extending
    # to the reach clamp instead would let the hand clip THROUGH the counter and drop the item beyond
    # it (user-measured, 2026-07-23). Stop on reaching target_z OR a stall, whichever is first.
    _set_pose("grab", hand=hand)
    target_z = float(plan["distance"]) - clip_standoff
    moved = 0
    released = False
    stop = "target_depth"
    try:
        prev = _hand_state()[trans_key]
        while moved < max_extend:
            if prev[2] >= target_z:                # hand forward z at (centre dist - standoff): stop short of the surface
                break
            cur = extend_fn()[trans_key]
            if sum((cur[k] - prev[k]) ** 2 for k in range(3)) ** 0.5 <= 1e-4:
                stop = "stall"                     # reach clamp before target depth - release here (still short)
                break
            prev = cur
            moved += 1
        release_z = _hand_state()[trans_key][2]
        released = not grip_fn().get("gripped")    # open the hand; released iff the grip is now off
    finally:
        for _ in range(moved):                     # retract exactly as far as we extended
            pull_fn()
        _set_pose("rest", hand=hand)

    placed = bool(released)                         # released UNDER a 'placeable' verdict (we broke on it)
    reason = (f"released 0.1 m short of the tray (hand z={release_z:.2f} vs slant {plan['distance']:.2f} m, "
              f"stop={stop})" if placed else "extended but the grip did NOT open - still holding the item")
    print(f"[place_held_item] hand={hand} placed={placed} released={released} "
          f"verdict={plan['verdict']} slant={plan['distance']:.2f} release_z={release_z:.2f} stop={stop}")
    return {"placed": placed, "released": released, "verdict": plan["verdict"],
            "distance": plan["distance"], "surface_height": plan["surface_height"],
            "release_z": release_z, "iters": iters, "reason": reason, "hand": hand}
