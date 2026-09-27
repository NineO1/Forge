"""FFmpeg/FFprobe wrappers: path resolution, cutting, encoding, probing."""

import json
import os
import subprocess

from .utils import JobState, setup_logging

log = setup_logging()


class FFmpegError(Exception):
    pass


def ffmpeg_path():
    r"""Resolve ffmpeg.exe: project bin first, then PATH."""
    local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "ffmpeg.exe")
    if os.path.isfile(local):
        return local
    return "ffmpeg"


def ffprobe_path():
    r"""Resolve ffprobe.exe: project bin first, then PATH."""
    local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "ffprobe.exe")
    if os.path.isfile(local):
        return local
    return "ffprobe"


def _run(cmd, job_state=None, desc="ffmpeg"):
    """Run a subprocess; kill-safe; raises FFmpegError on failure."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        _, err = proc.communicate()
    except KeyboardInterrupt:
        proc.kill()
        raise FFmpegError(f"{desc} interrupted")
    if proc.returncode != 0:
        msg = err.decode("utf-8", errors="replace")[-800:]
        if job_state:
            job_state.log_error(desc, msg)
        raise FFmpegError(f"{desc} failed (exit {proc.returncode}): {msg}")
    if job_state and job_state.cancelled:
        proc.kill()
        raise FFmpegError(f"{desc} cancelled")
    return proc.returncode


def probe(video_path):
    """Return dict of stream/container info via ffprobe."""
    cmd = [
        ffprobe_path(), "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    out = subprocess.run(
        cmd, capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if out.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {out.stderr[-300:]}")
    return json.loads(out.stdout)


def video_duration(video_path):
    """Duration in seconds (float)."""
    info = probe(video_path)
    return float(info["format"]["duration"])


def cut_clip(source, start, end, dest, job_state=None, copy_mode=True):
    """Cut a clip from source between start/end seconds, exactly to dest path."""
    dur = max(0.0, end - start)
    if copy_mode:
        vf = ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        vf = ["-c:v", "libx264", "-c:a", "aac"]
    cmd = [ffmpeg_path(), "-y", "-ss", f"{start:.3f}", "-i", source,
           "-t", f"{dur:.3f}", *vf, "-map", "0", dest]
    _run(cmd, job_state=job_state, desc=f"cut_clip({os.path.basename(dest)})")


def extract_audio(source, dest_wav, job_state=None):
    """Extract 16kHz mono WAV (whisper-ready) from source."""
    cmd = [ffmpeg_path(), "-y", "-i", source, "-vn", "-ar", "16000",
           "-ac", "1", "-c:a", "pcm_s16le", dest_wav]
    _run(cmd, job_state=job_state, desc="extract_audio")


def nvenc_supported():
    """Check whether NVENC (4090) is available in this ffmpeg build."""
    try:
        proc = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-encoders"],
            capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return "h264_nvenc" in proc.stdout
    except Exception:
        return False