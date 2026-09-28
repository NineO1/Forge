"""
Forge Polish Pass - Upscale + Frame Interpolation (optional post-processing).

Stage 1: Lanczos upscale (crisp, sharp detail from low-res generation).
Stage 2: minterpolate motion-compensated frame interpolation (true in-between
         frames, not duplication - ffmpeg-native, no extra models needed).

Heavy on CPU for interpolation; upscale is fast. Both run via bundled ffmpeg.
"""

import os
import subprocess


def _ffmpeg():
    from .ffmpeg_tools import ffmpeg_path
    return ffmpeg_path()


def _run(cmd):
    r = subprocess.run(
        cmd, capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return r.returncode == 0


def upscale_video(video_path, output_path, scale_factor=2, job_state=None):
    """Lanczos upscale. scale_factor=2 turns 704x480 into 1408x960."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input not found: {video_path}")
    if job_state:
        job_state.update(progress=10, message=f"Upscaling x{scale_factor}...")

    vf = f"scale=iw*{scale_factor}:ih*{scale_factor}:flags=lanczos"
    cmd = [_ffmpeg(), "-y", "-i", video_path,
           "-vf", vf,
           "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-an",
           output_path]
    if not _run(cmd) or not os.path.isfile(output_path):
        raise RuntimeError("Upscale failed - ffmpeg reported an error")
    if job_state:
        job_state.update(progress=45, message="Upscale done")
    return output_path


def interpolate_frames(video_path, output_path, target_fps=48, job_state=None):
    """Motion-compensated frame interpolation to target fps.

    NOTE: minterpolate is CPU-heavy and slow (can take several minutes per
    clip). It produces true interpolated frames, not duplicated ones.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input not found: {video_path}")
    if job_state:
        job_state.update(progress=50, message=f"Interpolating to {target_fps}fps (slow)...")

    vf = (f"minterpolate=fps={target_fps}:mi_mode=mci:"
          f"mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    cmd = [_ffmpeg(), "-y", "-i", video_path,
           "-vf", vf,
           "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-an",
           output_path]
    if not _run(cmd) or not os.path.isfile(output_path):
        raise RuntimeError("Interpolation failed - ffmpeg reported an error")
    if job_state:
        job_state.update(progress=80, message="Interpolation done")
    return output_path


def polish_video(input_path, output_path, do_upscale=True, do_interpolate=False,
                 target_scale=2, target_fps=48, job_state=None):
    """Full polish pipeline: upscale -> interpolate. Returns final path."""
    temp_step = None
    try:
        if do_upscale:
            temp_step = os.path.splitext(input_path)[0] + "_upscaled_tmp.mp4"
            intermediate = upscale_video(input_path, temp_step, target_scale,
                                         job_state)
        else:
            intermediate = input_path

        if do_interpolate:
            result = interpolate_frames(intermediate, output_path, target_fps,
                                         job_state)
        elif temp_step:
            # Rename temp to final so the caller finds the expected path
            os.replace(temp_step, output_path)
            result = output_path
        else:
            result = input_path
        return result
    finally:
        if temp_step and os.path.isfile(temp_step) and \
                temp_step not in (output_path, input_path):
            try:
                os.remove(temp_step)
            except OSError:
                pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m forge.polish <in.mp4> <out.mp4> [--up] [--interp]")
    else:
        polish_video(sys.argv[1], sys.argv[2],
                     do_upscale="--up" in sys.argv,
                     do_interpolate="--interp" in sys.argv)
        print(f"Polished: {sys.argv[1]} -> {sys.argv[2]}")