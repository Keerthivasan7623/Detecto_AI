"""Video / image helpers: resizing, browser-safe H.264 re-encode, housekeeping."""
import shutil
import subprocess
import time
from pathlib import Path

import cv2


def get_ffmpeg():
    """System ffmpeg if present, otherwise the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def make_browser_playable(src: Path) -> Path:
    """OpenCV's mp4v output does not play in browsers. Re-encode to H.264 (yuv420p)."""
    ffmpeg = get_ffmpeg()
    if not ffmpeg:
        return src
    dst = src.with_name(src.stem + "_h264.mp4")
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=900)
        if result.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            src.unlink(missing_ok=True)
            return dst
    except Exception:
        pass
    return src


def resize_keep_aspect(frame, max_width: int):
    """Downscale to max_width (even dimensions, required by H.264). Never upscales."""
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        scale = max_width / w
        w, h = max_width, int(h * scale)
    w -= w % 2
    h -= h % 2
    if (w, h) != (frame.shape[1], frame.shape[0]):
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return frame


def cleanup_old_files(directory: Path, days: int = 14):
    """Delete generated files older than `days` so outputs/ does not grow forever."""
    cutoff = time.time() - days * 86400
    for p in Path(directory).glob("*"):
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass
