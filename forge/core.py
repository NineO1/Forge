"""Forge orchestrator: runs selected features over one video, shared JobState."""

import os
import time

from . import FORGE_VERSION
from .utils import JobState, make_output_dir, setup_logging
from .ffmpeg_tools import video_duration

log = setup_logging()

STAGE_ORDER = ["scenes", "detect", "transcribe", "frames"]

def run_job(video_path, output_root=None, features=None, job_state=None,
            scene_threshold=27.0, scene_min_len=1, detect_labels=None,
            detect_sample_step=15, language=None, words_per_line=8,
            max_lines=2, sheet_cols=4, sheet_rows=4):
    """Run selected Forge features over a video.

    Args:
        video_path: input video file
        output_root: base output folder (default: project output folder)
        features: dict {"scenes":bool, "detect":bool, "transcribe":bool, "frames":bool}
        job_state: optional pre-existing JobState (one created if None)
        Remaining args: per-feature tunables (defaults are the validated ones).

    Returns:
        summary: dict with per-feature results, timings, output dir
    """
    if features is None:
        features = {"scenes": True, "detect": True, "transcribe": True, "frames": True}
    if output_root is None:
        output_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

    active = [f for f in STAGE_ORDER if features.get(f)]
    if not active:
        raise ValueError("No features selected")

    if job_state is None:
        job_state = JobState()

    out_dir = make_output_dir(output_root, video_path)
    summary = {
        "version": FORGE_VERSION,
        "video": video_path,
        "output_dir": out_dir,
        "duration": video_duration(video_path),
        "stages": {},
    }

    job_state.start(f"Processing {os.path.basename(video_path)} ({len(active)} features)")
    log.info(f"Forge {FORGE_VERSION} | {video_path}")
    log.info(f"Features: {active} -> {out_dir}")

    stage_span = 100.0 / len(active)
    stage_start_progress = {}

    # Compute progress windows per stage
    for i, feat in enumerate(active):
        stage_start_progress[feat] = i * stage_span

    for feat in active:
        if job_state.cancelled:
            break
        t0 = time.time()
        stage_base = stage_start_progress[feat]
        try:
            # ---- Sub-JobState wrapper: maps stage 0-100 into overall window ----
            sub = _SubJobState(job_state, stage_base, stage_span)

            if feat == "scenes":
                from .scenes import detect_scenes, extract_scene_clips
                scenes = detect_scenes(video_path, threshold=scene_threshold,
                                       min_scene_len=scene_min_len)
                clips = extract_scene_clips(video_path, out_dir, scenes,
                                            job_state=sub)
                summary["stages"][feat] = {"scenes": len(scenes), "clips": len(clips)}

            elif feat == "detect":
                from .detect import run_detection
                tl_path, timeline, stats = run_detection(
                    video_path, out_dir, labels=detect_labels,
                    sample_step=detect_sample_step, job_state=sub)
                summary["stages"][feat] = {"hits": stats["hits"],
                                           "labels": stats["labels_found"],
                                           "timeline": tl_path}

            elif feat == "transcribe":
                from .transcribe import transcribe_to_srt
                srt_path, segments, info = transcribe_to_srt(
                    video_path, out_dir, language=language,
                    words_per_line=words_per_line, max_lines=max_lines,
                    job_state=sub)
                summary["stages"][feat] = {"srt": srt_path,
                                           "segments": len(segments),
                                           "language": info.language}

            elif feat == "frames":
                from .frames import create_contact_sheet, extract_thumbnails
                sheet = create_contact_sheet(video_path, out_dir,
                                             cols=sheet_cols, rows=sheet_rows,
                                             job_state=sub)
                thumbs = extract_thumbnails(video_path, out_dir,
                                            num_thumbs=5, job_state=sub)
                summary["stages"][feat] = {"sheet": sheet, "thumbs": len(thumbs)}

            summary["stages"][feat]["time"] = round(time.time() - t0, 1)

        except InterruptedError:
            log.warning(f"{feat} cancelled")
            summary["stages"][feat] = {"cancelled": True}
            break
        except Exception as e:
            log.error(f"{feat} failed: {e}")
            job_state.log_error(feat, str(e))
            summary["stages"][feat] = {"error": str(e)}

    job_state.finish("Processing complete" if not job_state.cancelled else "Cancelled")
    summary["elapsed"] = job_state.elapsed_str
    summary["errors"] = job_state.errors
    return summary


class _SubJobState:
    """Wraps a JobState, remapping 0-100 stage progress into an overall window."""

    def __init__(self, parent, base_pct, span_pct):
        self.parent = parent
        self.base = base_pct
        self.span = span_pct

    @property
    def cancelled(self):
        return self.parent.cancelled

    def update(self, progress=None, message=None):
        mapped = None
        if progress is not None:
            mapped = self.base + (max(0.0, min(100.0, progress)) / 100.0) * self.span
        self.parent.update(progress=mapped, message=message)

    def log_error(self, stage, error_text):
        self.parent.log_error(stage, error_text)