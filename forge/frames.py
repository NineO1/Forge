"""Frame extraction, thumbnails, and contact sheets."""

import os

from .ffmpeg_tools import ffmpeg_path, probe, FFmpegError
from .utils import JobState, setup_logging

log = setup_logging()

def extract_frames_at_times(video_path, times_secs, output_dir, prefix="frame", job_state=None):
    """Extract single frames at given times (in seconds) to PNG files.

    Args:
        video_path: source video
        times_secs: list of [sec1, sec2, ...] timestamps
        output_dir: where to save PNGs (creates frames/ subdir)
        prefix: filename prefix
        job_state: optional progress tracking

    Returns: list of (timestamp_sec, output_path) tuples.
    """
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    out_subdir = os.path.join(output_dir, "frames")
    total = len(times_secs)
    frames = []

    for idx, t in enumerate(times_secs):
        if job_state and job_state.cancelled:
            raise InterruptedError("Frame extraction cancelled")

        base_name = f"{prefix}_{idx+1:04d}_t{int(t):06d}.png"
        out_path = os.path.join(out_subdir, base_name)

        cmd = [
            ffmpeg_path(), "-y", "-i", video_path,
            "-ss", f"{t:.3f}", "-vframes", "1", "-vf", "scale=iw:-1",
            out_path,
        ]
        try:
            subprocess_run(cmd, job_state=job_state, desc=f"frame_{idx+1}")
            frames.append((t, out_path))
        except FFmpegError as e:
            log.error(f"Failed frame at {t}s: {e}")
            if job_state:
                job_state.log_error(f"frame_{idx+1}", str(e))

        if job_state:
            progress = ((idx + 1) / total) * 100
            job_state.update(progress=progress, message=f"Extracting frame {idx+1}/{total}")

    log.info(f"Extracted {len(frames)} frames to {out_subdir}\\")
    return frames

def create_contact_sheet(video_path, output_dir, cols=4, rows=4, job_state=None):
    """Generate a contact sheet grid image from evenly-spaced frames.

    Uses FFmpeg's tile filter to build one JPEG from multiple frames.
    """
    from .ffmpeg_tools import video_duration as get_duration

    dur = get_duration(video_path)
    num_frames = cols * rows
    timestamps = [i * dur / num_frames for i in range(num_frames)]

    # Ensure output directory exists before muxing
    os.makedirs(output_dir, exist_ok=True)

    # Build tile command (ffmpeg -vf "scale=iw:-1,tile={cols}x{rows}")
    tile_filter = f"scale=iw:-1,tile={cols}x{rows}"
    out_path = os.path.join(output_dir, "contact_sheet.jpg")

    cmd = [
        ffmpeg_path(), "-y", "-i", video_path,
        "-vf", tile_filter,
        "-frames:v", "1",
        "-q:v", "2",
        out_path,
    ]

    log.info(f"Creating contact sheet ({cols}x{rows}) from {dur:.1f}s video")
    try:
        subprocess_run(cmd, job_state=job_state, desc="contact_sheet")
        log.info(f"Contact sheet saved to {out_path}")
        return out_path
    except FFmpegError as e:
        log.error(f"Contact sheet creation failed: {e}")
        if job_state:
            job_state.log_error("contact_sheet", str(e))
        return None

def extract_thumbnails(video_path, output_dir, num_thumbs=5, job_state=None):
    """Extract N evenly-spaced thumbnail JPGs (scaled to max 320px wide)."""
    from .ffmpeg_tools import video_duration as get_duration

    dur = get_duration(video_path)
    timestamps = [i * dur / (num_thumbs - 1) if num_thumbs > 1 else dur/2 for i in range(num_thumbs)]
    # Clamp inside the file tail: seeking at exact EOF yields no frame
    timestamps = [min(t, max(0.0, dur - 0.5)) for t in timestamps]

    thumbs = []
    for idx, t in enumerate(timestamps):
        if job_state and job_state.cancelled:
            raise InterruptedError("Thumbnail extraction cancelled")

        out_path = os.path.join(output_dir, f"thumb_{idx+1:02d}.jpg")
        cmd = [
            ffmpeg_path(), "-y", "-i", video_path,
            "-ss", f"{t:.3f}", "-vframes", "1",
            "-vf", "scale='min(320,iw)':-1'",
            "-q:v", "3",
            out_path,
        ]
        try:
            subprocess_run(cmd, job_state=job_state, desc=f"thumb_{idx+1}")
            thumbs.append((t, out_path))
        except FFmpegError as e:
            log.error(f"Thumbnail {idx+1} at {t}s failed: {e}")
            if job_state:
                job_state.log_error(f"thumb_{idx+1}", str(e))

        if job_state:
            progress = ((idx + 1) / num_thumbs) * 100
            job_state.update(progress=progress, message=f"Thumbnail {idx+1}/{num_thumbs}")

    return thumbs

def subprocess_run(cmd, job_state=None, desc="ffmpeg"):
    """Helper wrapper for subprocess with proper Windows flags."""
    import subprocess

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
        msg = err.decode("utf-8", errors="replace")[-500:]
        if job_state:
            job_state.log_error(desc, msg)
        raise FFmpegError(f"{desc} failed (exit {proc.returncode}): {msg}")
    if job_state and job_state.cancelled:
        proc.kill()
        raise FFmpegError(f"{desc} cancelled")