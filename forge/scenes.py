"""Scene/shot detection using PySceneDetect + FFmpeg splitting."""

import os

from .ffmpeg_tools import cut_clip, video_duration, FFmpegError
from .utils import JobState, setup_logging

log = setup_logging()

def detect_scenes(video_path, threshold=27.0, min_scene_len=1):
    """Return list of (start_sec, end_sec) tuples via PySceneDetect.

    Args:
        video_path: input video
        threshold: sensitivity (lower = more cuts)
        min_scene_len: minimum scene length in seconds

    Returns: [] if no scenes detected, list of (start, end) otherwise.
    """
    try:
        from scenedetect import detect, ContentDetector
    except ImportError as e:
        raise ImportError("scenedetect not installed; run pip install scenedetect[opencv]") from e

    log.info(f"Detecting scenes in {video_path}...")
    scene_list = detect(video_path, ContentDetector(threshold=threshold))

    # Filter scenes shorter than min_scene_len
    scenes = [
        (start.get_seconds(), end.get_seconds())
        for start, end in scene_list
        if end.get_seconds() - start.get_seconds() >= min_scene_len
    ]
    log.info(f"Detected {len(scenes)} scenes")
    return scenes

def extract_scene_clips(source_video, output_dir, scenes, job_state=None):
    """Cut each scene to a separate clip file in output_dir/scenes/.

    Returns list of (scene_index, output_path) tuples.
    """
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)

    if not scenes:
        log.warning("No scenes detected; returning empty list")
        return []

    total = len(scenes)
    clips = []

    for idx, (start, end) in enumerate(scenes):
        if job_state and job_state.cancelled:
            raise InterruptedError("Scene extraction cancelled")

        base_name = f"{idx+1:03d}"
        out_path = os.path.join(output_dir, "samples", f"{base_name}.mp4")

        # Progress update (rough estimate: 2 seconds per clip cut on typical hardware)
        progress = ((idx + 1) / total) * 100
        if job_state:
            job_state.update(progress=progress, message=f"Cutting scene {idx+1}/{total}")

        log.debug(f"Cutting scene {idx+1}: {start:.2f}s -> {end:.2f}s -> {out_path}")
        try:
            cut_clip(source_video, start, end, out_path, job_state=job_state, copy_mode=True)
            clips.append((idx + 1, out_path))
        except FFmpegError as e:
            log.error(f"Failed to cut scene {idx+1}: {e}")
            if job_state:
                job_state.log_error(f"scene_{idx+1}", str(e))

    log.info(f"Extracted {len(clips)} scene clips to {output_dir}\\samples\\")
    return clips