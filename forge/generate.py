"""AI generation: text-to-image, image-to-image, text-to-video, image-to-video.

All models run locally on the 4090 via diffusers. Heavy imports are lazy -
nothing here loads until a generation function is actually called.
Models download to models\\ on first use (SDXL ~7GB, LTX-Video ~10GB).
"""

import os
import re

import torch

import gc

from .utils import JobState, setup_logging

log = setup_logging()


def _cleanup_gpu():
    """Force GPU memory release after any diffusers/torch pipeline."""
    try:
        import torch
        if torch.cuda.is_available():
            gc.collect()
            peak = torch.cuda.max_memory_reserved() / (1024 ** 3)
            if peak > 0.01:
                log.info(f"[VRAM] peak reserved this job: {peak:.1f}GB")
            torch.cuda.empty_cache()
    except Exception:
        pass

def _snap(v, base, lo, hi):
    """Snap a value to the nearest multiple of base, clamped."""
    v = max(lo, min(hi, v))
    return max(lo, (round(v / base) * base))

MODELS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Keep ALL HuggingFace caches inside the project so the distributed
# package is self-contained (Choice A: ready to run, no re-downloads).
os.environ["HF_HOME"] = os.path.join(MODELS_ROOT, "hf")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

def _cuda_kwargs():
    if torch.cuda.is_available():
        return {"torch_dtype": torch.float16}
    return {"torch_dtype": torch.float32}

def _steps(n, pipe, job_state, label):
    """Callback that reports diffusion progress into JobState."""
    def cb(p, i, t, kwargs):
        if job_state:
            job_state.update(progress=(i + 1) / n * 100,
                             message=f"{label}: step {i + 1}/{n}")
        return {}
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_begin_index(None) if hasattr(pipe.scheduler, "set_begin_index") else None
    return cb

# ---------------------------------------------------------------- text->image
def text_to_image(prompt, output_dir, filename=None, negative_prompt="",
    width=1024, height=1024, steps=35, seed=None, job_state=None):
    """Generate an image from text. Returns output PNG path.

    filename used EXACTLY as typed (no prefixes/suffixes), default gen_image.png
    """
    from diffusers import StableDiffusionXLPipeline

    out_dir = os.path.normpath(os.path.join(output_dir, "generated"))
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{os.path.splitext(filename)[0]}.png" if filename else "gen_image.png"
    out_path = os.path.join(out_dir, fname)

    if job_state:
        job_state.update(progress=5, message="Loading SDXL (first run downloads ~7GB)...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", **_cuda_kwargs())
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen = gen.manual_seed(seed) if seed is not None else gen

    if job_state:
        job_state.update(progress=10, message="Generating image...")

    cb = _steps(steps, pipe, job_state, "t2i")
    image = pipe(prompt=prompt, negative_prompt=negative_prompt,
                 width=width, height=height, num_inference_steps=steps,
                 generator=gen, callback_on_step_end=cb).images[0]

    image.save(out_path)
    del image
    del pipe
    log.info(f"text->image saved: {out_path}")
    if job_state:
        job_state.update(progress=100, message="Image done")
    _cleanup_gpu()
    return out_path

# -------------------------------------------------------------- image->image
def image_to_image(prompt, init_image, output_dir, filename=None,
                   negative_prompt="", strength=0.75, steps=30, seed=None,
                   job_state=None):
    """Modify an image with a prompt (AI modifier). Returns output PNG path.

    init_image: path to source image (a Forge frame/thumbnail works great)
    strength: 0.0 = barely changed, 1.0 = fully redrawn (default 0.75)
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline
    from diffusers.utils import load_image

    out_dir = os.path.normpath(os.path.join(output_dir, "generated"))
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{os.path.splitext(filename)[0]}.png" if filename else "modified_image.png"
    out_path = os.path.join(out_dir, fname)

    if job_state:
        job_state.update(progress=5, message="Loading SDXL...")

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", **_cuda_kwargs())
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    src = load_image(init_image)

    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen = gen.manual_seed(seed) if seed is not None else gen

    cb = _steps(steps, pipe, job_state, "i2i")
    image = pipe(prompt=prompt, negative_prompt=negative_prompt,
                 image=src, strength=strength, num_inference_steps=steps,
                 generator=gen, callback_on_step_end=cb).images[0]

    image.save(out_path)
    del image
    del src
    del pipe
    log.info(f"image->image saved: {out_path}")
    if job_state:
        job_state.update(progress=100, message="Modified image done")
    _cleanup_gpu()
    return out_path

# ---------------------------------------------------------------- video helpers
def _fix_frame_count(num_frames):
    """LTX requires frame count = multiple of 8, plus 1 (9, 17, 25...)."""
    if num_frames < 9:
        return 9
    return ((num_frames - 1) // 8) * 8 + 1

# ------------------------------------------------------------------ VRAM governor
# Pre-flight check: projects peak VRAM for a resolution x frame-count and
# refuses combos above 90% of currently free VRAM.
# CALIBRATE ONCE: run t2v at Sweet spot (704x480) @ ~5s (121 frames), copy
# the "[VRAM] peak reserved this job" number from the log into _BASE_PEAK_GB.

_BASE_PEAK_GB = 15.0          # placeholder - replace with your measured peak
_STATIC_GB = 10.5             # persistent portion (pipeline + weights)
_BASE_W, _BASE_H, _BASE_F = 704, 480, 121
_VRAM_CAP = 0.90

def _vram_projected_gb(width, height, num_frames):
    base_work = _BASE_W * _BASE_H * (_BASE_F - 1)
    work = width * height * (num_frames - 1)
    act_base = _BASE_PEAK_GB - _STATIC_GB
    return _STATIC_GB + act_base * (work / base_work)

def _vram_guard(width, height, num_frames):
    """Raise a clear error if this combo projects above the 90% cap."""
    import torch
    if not torch.cuda.is_available():
        return
    projected = _vram_projected_gb(width, height, num_frames)
    free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    cap_gb = free_gb * _VRAM_CAP
    log.info(f"[VRAM] free={free_gb:.1f}GB cap={cap_gb:.1f}GB "
             f"projected={projected:.1f}GB for {width}x{height}@{num_frames}f")
    torch.cuda.reset_peak_memory_stats()
    if projected > cap_gb:
        raise RuntimeError(
            f"{width}x{height} @ {num_frames} frames projects to "
            f"{projected:.1f}GB, above the 90% guard ({cap_gb:.1f}GB free). "
            f"Lower duration or resolution.")

LTX_MODEL = "Lightricks/LTX-Video"

def _ltx_pipe(cls):
    if torch.cuda.is_available():
        kwargs = {"torch_dtype": torch.bfloat16}
    else:
        kwargs = {}
    pipe = cls.from_pretrained(LTX_MODEL, **kwargs)
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------- text->video
def text_to_video(prompt, output_dir, filename=None, negative_prompt="",
    width=704, height=480, num_frames=121, steps=35, guidance_scale=4.5,
    seed=None, job_state=None):
    """Generate a video from text. Returns output MP4 path."""
    from diffusers import LTXPipeline
    from diffusers.utils import export_to_video

    out_dir = os.path.normpath(os.path.join(output_dir, "generated"))
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{os.path.splitext(filename)[0]}.mp4" if filename else "gen_video.mp4"
    out_path = os.path.join(out_dir, fname)

    num_frames = _fix_frame_count(num_frames)
    if job_state:
        job_state.update(progress=5, message=f"Loading LTX-Video (first run ~10GB)...")

    pipe = _ltx_pipe(LTXPipeline)

    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen = gen.manual_seed(seed) if seed is not None else gen

    cb = _steps(steps, pipe, job_state, "t2v")
    width = _snap(width, 32, 256, 1280) # raised for 16:9 options
    height = _snap(height, 32, 256, 736)
    num_frames = _snap(num_frames - 1, 8, 8, 240) + 1 # up to ~10s @ 24fps
    _vram_guard(width, height, num_frames)
    log.info(f"snapped: {width}x{height} @ {num_frames} frames")
    video = pipe(prompt=prompt, negative_prompt=negative_prompt,
                width=width, height=height, num_frames=num_frames,
                num_inference_steps=steps, guidance_scale=guidance_scale,
                generator=gen, callback_on_step_end=cb).frames[0]

    if job_state:
        job_state.update(progress=100, message="Decoding frames & encoding video...")
    export_to_video(video, out_path, fps=24)  # LTX-Video native fps
    del video
    del pipe
    log.info(f"text->video saved: {out_path}")
    if job_state:
        job_state.update(progress=100, message="Video done")
    _cleanup_gpu()
    return out_path

# ------------------------------------------------------------- image->video
def image_to_video(prompt, init_image, output_dir, filename=None,
    width=704, height=480, num_frames=121, steps=35, guidance_scale=5.0,
    strength=0.75, seed=None, job_state=None):
    """Animate a still image into a video. Returns output MP4 path.

    init_image: source image (Forge frame, thumbnail, or any file)
    """
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image

    out_dir = os.path.normpath(os.path.join(output_dir, "generated"))
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{os.path.splitext(filename)[0]}.mp4" if filename else "animated.mp4"
    out_path = os.path.join(out_dir, fname)

    num_frames = _fix_frame_count(num_frames)
    if job_state:
        job_state.update(progress=5, message="Loading LTX-Video...")

    pipe = _ltx_pipe(LTXImageToVideoPipeline)

    src = load_image(init_image)

    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen = gen.manual_seed(seed) if seed is not None else gen

    cb = _steps(steps, pipe, job_state, "i2v")
    width = _snap(width, 32, 256, 1280) # raised for 16:9 options
    height = _snap(height, 32, 256, 736)
    num_frames = _snap(num_frames - 1, 8, 8, 240) + 1 # up to ~10s @ 24fps
    _vram_guard(width, height, num_frames)
    log.info(f"snapped: {width}x{height} @ {num_frames} frames")
    video = pipe(prompt=prompt, negative_prompt=negative_prompt,
                image=src, width=width, height=height,
                num_frames=num_frames, num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=gen, callback_on_step_end=cb).frames[0]

    if job_state:
        job_state.update(progress=100, message="Decoding frames & encoding video...")
    export_to_video(video, out_path, fps=24)  # LTX-Video native fps
    del video
    del src
    del pipe
    log.info(f"image->video saved: {out_path}")
    if job_state:
        job_state.update(progress=100, message="Animated video done")
    _cleanup_gpu()
    return out_path