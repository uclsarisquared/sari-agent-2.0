import os
from PIL import ImageDraw, Image
from sim.env import artifact_path, benchmark_artifact_mode, save_jpeg_atomic, screenshot_dir


def annotate_located_object(image: Image.Image | str, bbox, color="red", radius=20):
    """Return a copy of `image` with `bbox` (x0, y0, x1, y1) outlined and a frame-centre crosshair.
    `radius` is unused; kept for call compatibility."""
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened:
            opened.load()
            annotated_image = opened.copy()
    else:
        annotated_image = image.copy()
    width, height = annotated_image.size
    draw = ImageDraw.Draw(annotated_image)
    draw.rectangle(xy=bbox, outline=color, width=4)
    fx, fy = width // 2, height // 2
    draw.line((fx - 10, fy, fx + 10, fy), fill="blue", width=2)
    draw.line((fx, fy - 10, fx, fy + 10), fill="blue", width=2)
    return annotated_image


def annotate_boxes(roi, prefix="", file_path=None, source_image=None, output_dir=None):
    """Annotate boxes without requiring a shared screenshot/output directory.

    Live callers may provide the exact PIL ``source_image`` used for detection. In an orchestrated
    attempt the compatibility defaults resolve under SARI_RUN_DIR; a standalone invocation retains
    screenshots/ and annotations/ in the current directory.
    """
    file_path = file_path or os.path.join(screenshot_dir(), "ClientScreenshot.png")
    output_dir = output_dir or artifact_path("annotations", legacy_base="")
    os.makedirs(output_dir, exist_ok=True)
    if type(roi) != list:
        roi = [roi]
    for i, item in enumerate(roi):
        box = item["box"]
        image = annotate_located_object(
            source_image if source_image is not None else file_path,
            (box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
        )
        stem = f"{prefix+'-' if prefix else ''}{i}"
        if benchmark_artifact_mode():
            save_jpeg_atomic(os.path.join(output_dir, f"{stem}.jpg"), image, quality=85)
        else:
            image.save(os.path.join(output_dir, f"{stem}.png"))
