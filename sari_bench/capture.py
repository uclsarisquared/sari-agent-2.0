"""Bounded live preview storage and continuous per-attempt replay recording."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

CAPTURE_DIR = "capture"
LATEST_CAPTURE = "latest.jpg"
LEGACY_LATEST_CAPTURE = "latest.json"
CAPTURE_WIDTH = 960
CAPTURE_JPEG_QUALITY = 75
ARCHIVE_WIDTH = 1280
ARCHIVE_HEIGHT = 720
ARCHIVE_CRF = 28
ARCHIVE_PRESET = "veryfast"
ARCHIVE_THREADS = 1
ENCODER_QUEUE_SIZE = 8
ENCODER_GRACE_SECONDS = 15.0
CAPTURE_TIMEOUT_SECONDS = 10.0
DEFAULT_INTERVAL_SECONDS = 0.25

_CAPTURE_FRAME = re.compile(r"^frame(\d+)-(\d+)\.jpg$")
_STEP_FRAME = re.compile(r"^step(\d+)(?:_[^.]+)?\.(?:png|jpe?g)$", re.IGNORECASE)
_STOP = object()


@dataclass
class CaptureStats:
    frames: int = 0
    failures: int = 0
    acquired: int = 0
    encoded: int = 0
    repeated: int = 0
    dropped: int = 0
    acquisition_failures: int = 0
    encoder_failures: int = 0


def is_step_frame(path: Path) -> bool:
    return bool(_STEP_FRAME.match(path.name))


_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IEND = b"IEND"


def is_valid_frame(path: Path) -> bool:
    """Whether a frame's bytes look like a complete image rather than a crash artifact.

    An interrupted or incompletely persisted publication can leave an empty, zero-filled, or
    truncated file at the final path. Checking the format's header and trailer is far cheaper than
    decoding every candidate and rejects those known crash artifacts.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
            if not head:
                return False
            handle.seek(-8, os.SEEK_END)
            tail = handle.read(8)
    except OSError:
        return False
    if head[:2] == _JPEG_SOI:
        return tail[-2:] == _JPEG_EOI
    if head == _PNG_SIGNATURE:
        return _PNG_IEND in tail
    return False


def frame_timestamp_ns(path: Path) -> int:
    match = _CAPTURE_FRAME.match(path.name) if path.parent.name == CAPTURE_DIR else None
    if match:
        return int(match.group(2))
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def observation_frames(run_dir: Path) -> list[Path]:
    """Legacy numbered captures plus exact/timestamped PNG/JPEG step frames."""
    frames: list[Path] = []
    if not run_dir.is_dir():
        return frames
    capture_dir = run_dir / CAPTURE_DIR
    if capture_dir.is_dir():
        frames.extend(
            path for path in capture_dir.iterdir()
            if path.is_file() and _CAPTURE_FRAME.match(path.name)
        )
    for leg_dir in run_dir.iterdir():
        if leg_dir.is_dir() and leg_dir.name.startswith("leg"):
            frames.extend(path for path in leg_dir.iterdir() if path.is_file() and is_step_frame(path))
    return frames


def latest_step_frame(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    frames: list[Path] = []
    for leg_dir in run_dir.iterdir():
        if leg_dir.is_dir() and leg_dir.name.startswith("leg"):
            frames.extend(
                path for path in leg_dir.iterdir()
                if path.is_file() and is_step_frame(path)
            )
    return next(
        (path for path in sorted(frames, key=frame_timestamp_ns, reverse=True)
         if is_valid_frame(path)),
        None,
    )


def latest_capture(run_dir: Path) -> Path | None:
    """The fixed live image, falling back to old pointer/numbered archives."""
    capture_dir = run_dir / CAPTURE_DIR
    fixed = capture_dir / LATEST_CAPTURE
    if fixed.is_file() and is_valid_frame(fixed):
        return fixed
    pointer = capture_dir / LEGACY_LATEST_CAPTURE
    rejected: Path | None = None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        name = payload.get("file") if isinstance(payload, dict) else None
        if isinstance(name, str) and _CAPTURE_FRAME.match(name):
            path = capture_dir / name
            if path.is_file() and is_valid_frame(path):
                return path
            rejected = path
    except (OSError, ValueError):
        pass
    if not capture_dir.is_dir():
        return None
    frames = sorted((
        p for p in capture_dir.iterdir()
        if p.is_file() and _CAPTURE_FRAME.match(p.name) and p != rejected
    ), key=frame_timestamp_ns, reverse=True)
    return next((path for path in frames if is_valid_frame(path)), None)


def latest_observation(run_dir: Path) -> Path | None:
    frames = [path for path in (latest_step_frame(run_dir), latest_capture(run_dir)) if path]
    return max(frames, key=frame_timestamp_ns) if frames else None


async def request_screenshot(commands_uri: str) -> bytes:
    import websockets

    async with websockets.connect(commands_uri, max_size=None) as websocket:
        await websocket.send(json.dumps({
            "command": "RequestScreenshot", "prefix": "", "suffix": "",
            "folder_name": "", "save_image": False,
        }))
        payload = await websocket.recv()
    if not isinstance(payload, bytes):
        raise TypeError(f"RequestScreenshot returned {type(payload).__name__}, expected bytes")
    return payload


def _jpeg_versions(image_bytes: bytes) -> tuple[bytes, bytes]:
    """Create the 960px live JPEG and a letterboxed 1280x720 encoder frame."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as source:
        rgb = source.convert("RGB")
        live = rgb
        if live.width > CAPTURE_WIDTH:
            live = live.resize(
                (CAPTURE_WIDTH, max(1, round(live.height * CAPTURE_WIDTH / live.width))),
                Image.Resampling.LANCZOS,
            )
        live_out = io.BytesIO()
        live.save(live_out, format="JPEG", quality=CAPTURE_JPEG_QUALITY)

        scale = min(ARCHIVE_WIDTH / rgb.width, ARCHIVE_HEIGHT / rgb.height)
        fitted = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
            Image.Resampling.LANCZOS,
        )
        archive = Image.new("RGB", (ARCHIVE_WIDTH, ARCHIVE_HEIGHT), "black")
        archive.paste(fitted, ((ARCHIVE_WIDTH - fitted.width) // 2,
                               (ARCHIVE_HEIGHT - fitted.height) // 2))
        archive_bytes = archive.tobytes()
    return live_out.getvalue(), archive_bytes


def _publish_latest(run_dir: Path, jpeg: bytes) -> Path:
    capture_dir = run_dir / CAPTURE_DIR
    capture_dir.mkdir(parents=True, exist_ok=True)
    final = capture_dir / LATEST_CAPTURE
    fd, temp_name = tempfile.mkstemp(prefix=".latest.", suffix=".jpg.tmp", dir=capture_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(jpeg)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, final)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise
    return final


def _save_jpeg(run_dir: Path, image_bytes: bytes, *_unused: object) -> Path:
    """Compatibility helper: atomically replace the one fixed live JPEG."""
    live, _ = _jpeg_versions(image_bytes)
    return _publish_latest(run_dir, live)


def replay_part_path(run_dir: Path, run_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip(".-") or "attempt"
    return run_dir / f".replay.{safe}.part.mp4"


def archive_command(part_path: Path, fps: float) -> list[str]:
    return [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{ARCHIVE_WIDTH}x{ARCHIVE_HEIGHT}", "-framerate", f"{fps:.12g}",
        "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-crf", str(ARCHIVE_CRF), "-preset", ARCHIVE_PRESET, "-threads", str(ARCHIVE_THREADS),
        "-pix_fmt", "yuv420p", "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
        str(part_path),
    ]


class AttemptRecorder:
    """Acquires independently of a monotonic encoder clock and never blocks on ffmpeg."""

    def __init__(self, run_dir: Path, commands_uri: str, interval: float, *, run_id: str = "",
                 fetch: Callable[[str], Awaitable[bytes]] = request_screenshot,
                 stats: CaptureStats | None = None, log: Callable[[str], None] | None = None) -> None:
        if not (interval > 0):
            raise ValueError("continuous recording requires a positive interval")
        self.run_dir = run_dir
        self.commands_uri = commands_uri
        self.interval = interval
        self.fetch = fetch
        self.stats = stats or CaptureStats()
        self.log = log
        self.run_id = run_id
        self.queue: queue.Queue[bytes | object] = queue.Queue(maxsize=ENCODER_QUEUE_SIZE)
        self.stopping = threading.Event()
        self.stop_event = asyncio.Event()
        self.latest_native: bytes | None = None
        self.latest_archive: bytes | None = None
        self.generation = 0
        self.last_emitted_generation = -1
        self.last_step_ns = 0
        self.part_path = replay_part_path(run_dir, run_id)
        self.process: subprocess.Popen[bytes] | None = None

    def _start_encoder(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.run_dir.glob(".replay.*.part.mp4"):
            with contextlib.suppress(OSError):
                stale.unlink()
        # An active execution cannot have a completed replay. This also prevents a prior execution
        # that reused the directory from looking ready if its replacement encoder later fails.
        with contextlib.suppress(OSError):
            (self.run_dir / "replay.mp4").unlink(missing_ok=True)
        if shutil.which("ffmpeg") is None:
            self.stats.encoder_failures += 1
            return
        try:
            self.process = subprocess.Popen(
                archive_command(self.part_path, 1.0 / self.interval),
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            self.stats.encoder_failures += 1
            if self.log:
                self.log(f"continuous replay unavailable: {error}")

    async def _accept(self, native: bytes, *, fetched: bool) -> None:
        live, archive = await asyncio.to_thread(_jpeg_versions, native)
        await asyncio.to_thread(_publish_latest, self.run_dir, live)
        self.latest_native = native
        self.latest_archive = archive
        self.generation += 1
        self.stats.acquired += 1
        if fetched:
            self.stats.frames += 1

    async def _acquire(self) -> None:
        next_at = time.monotonic()
        while not self.stopping.is_set():
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))
            next_at += self.interval
            step = latest_step_frame(self.run_dir)
            step_ns = frame_timestamp_ns(step) if step else 0
            if step is not None and step_ns > self.last_step_ns:
                self.last_step_ns = step_ns
                try:
                    await self._accept(await asyncio.to_thread(step.read_bytes), fetched=False)
                    if next_at < time.monotonic():
                        next_at = time.monotonic() + self.interval
                    continue
                except Exception:
                    self.stats.acquisition_failures += 1
                    self.stats.failures += 1
            fetch_task = asyncio.create_task(self.fetch(self.commands_uri))
            try:
                native = await asyncio.wait_for(fetch_task, CAPTURE_TIMEOUT_SECONDS)
                await self._accept(native, fetched=True)
            except asyncio.CancelledError:
                outcome = await asyncio.gather(fetch_task, return_exceptions=True)
                error = outcome[0]
                if isinstance(error, Exception) and self.log:
                    detail = " ".join(str(error).splitlines()) or "no details"
                    self.log(f"preview capture cleanup: {type(error).__name__}: {detail}")
                raise
            except Exception:
                self.stats.acquisition_failures += 1
                self.stats.failures += 1
            if next_at < time.monotonic():
                next_at = time.monotonic() + self.interval

    async def _tick(self) -> None:
        next_at = time.monotonic()
        while not self.stopping.is_set():
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))
            next_at += self.interval
            frame = self.latest_archive
            if frame is None or self.process is None:
                continue
            if self.generation == self.last_emitted_generation:
                self.stats.repeated += 1
            self.last_emitted_generation = self.generation
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                self.stats.dropped += 1

    def _encode(self) -> None:
        while True:
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                if self.stopping.is_set():
                    return
                continue
            if frame is _STOP:
                return
            process = self.process
            if process is None or process.stdin is None or process.poll() is not None:
                self.stats.encoder_failures += 1
                continue
            try:
                process.stdin.write(frame)
                process.stdin.flush()
                self.stats.encoded += 1
            except (BrokenPipeError, OSError, ValueError):
                self.stats.encoder_failures += 1

    async def run(self) -> CaptureStats:
        self._start_encoder()
        acquisition = asyncio.create_task(self._acquire())
        ticker = asyncio.create_task(self._tick())
        writer = threading.Thread(target=self._encode, name="sari-replay-encoder", daemon=True)
        writer.start()
        try:
            await self.stop_event.wait()
        finally:
            acquisition.cancel()
            ticker.cancel()
            self.stopping.set()
            # The sentinel belongs behind already accepted frames. If the queue is full, discard
            # its oldest video frame; shutdown remains bounded and acquisition was never blocked.
            while True:
                try:
                    self.queue.put_nowait(_STOP)
                    break
                except queue.Full:
                    with contextlib.suppress(queue.Empty):
                        self.queue.get_nowait()
                        self.stats.dropped += 1
            # Cancellation of a screenshot websocket can take a few scheduler turns on Python
            # 3.10. The stop flag prevents another acquisition even if cleanup finishes later;
            # recorder finalization itself must not wait on that network path.
            await asyncio.sleep(0)
            for task in (acquisition, ticker):
                task.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception()
                )
            deadline = time.monotonic() + ENCODER_GRACE_SECONDS
            while writer.is_alive() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            if writer.is_alive():
                self.stats.encoder_failures += 1
                if self.process is not None:
                    with contextlib.suppress(OSError):
                        self.process.terminate()
                deadline = time.monotonic() + 2.0
                while writer.is_alive() and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
            await self._finalize_encoder()
        return self.stats

    async def _finalize_encoder(self) -> None:
        process = self.process
        if process is None:
            self.part_path.unlink(missing_ok=True)
            return
        if process.stdin is not None:
            # Every frame was explicitly flushed by the writer. Closing the descriptor directly
            # delivers EOF without letting BufferedWriter.close() block the event loop on a sick
            # encoder's pipe.
            with contextlib.suppress(OSError):
                os.close(process.stdin.fileno())
            process.stdin = None
        deadline = time.monotonic() + ENCODER_GRACE_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if process.poll() is None:
            self.stats.encoder_failures += 1
            process.terminate()
            deadline = time.monotonic() + 2.0
            while process.poll() is None and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            if process.poll() is None:
                with contextlib.suppress(OSError):
                    process.kill()
        from sari_bench import video
        replay = self.run_dir / video.REPLAY_NAME
        if process.returncode == 0 and self.stats.encoded and video.valid_replay(self.part_path):
            os.replace(self.part_path, replay)
        else:
            self.stats.encoder_failures += 1
            self.part_path.unlink(missing_ok=True)


async def record_previews(run_dir: Path, commands_uri: str, interval: float, *,
                          fetch: Callable[[str], Awaitable[bytes]] = request_screenshot,
                          stats: CaptureStats | None = None,
                          log: Callable[[str], None] | None = None,
                          run_id: str = "") -> CaptureStats:
    """Run until cancelled, finalizing and atomically publishing the replay on cancellation."""
    recorder = AttemptRecorder(run_dir, commands_uri, interval, run_id=run_id, fetch=fetch,
                               stats=stats, log=log)
    task = asyncio.create_task(recorder.run())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        recorder.stop_event.set()
        await asyncio.shield(task)
        raise
