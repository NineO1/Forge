"""Transcription via faster-whisper (large-v3, GPU) with word-level SRT output."""

import os

from .utils import JobState, setup_logging, timestamp_str

log = setup_logging()

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "whisper")

def _get_model(model_size="large-v3", device=None):
    """Load (and first-run download) the faster-whisper model."""
    from faster_whisper import WhisperModel

    os.makedirs(MODEL_DIR, exist_ok=True)

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    log.info(f"Loading Whisper {model_size} on {device} ({compute_type})")
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type,
                            download_root=MODEL_DIR), device
    except Exception as e:
        log.warning(f"GPU load failed ({e}); falling back to CPU")
        return WhisperModel(model_size, device="cpu", compute_type="int8",
                            download_root=MODEL_DIR), "cpu"

def transcribe(video_path, model=None, language=None, job_state=None,
               words_per_line=8, model_size="large-v3"):
    """Transcribe a video's audio. Returns (segments, info).

    Args:
        video_path: input video (audio extracted internally to 16kHz mono WAV)
        model: pre-loaded WhisperModel, or None to load one
        language: None = auto-detect, or explicit code like 'en', 'fr', 'ja'
        job_state: progress/cancel tracking
        words_per_line: SRT line grouping (word count, per Captiona convention)

    Returns:
        segments: list of dicts with words [{word, start, end}, ...]
        info: model info dict (language, duration, etc.)
    """
    from .ffmpeg_tools import extract_audio

    if job_state:
        job_state.update(message="Extracting audio...")

    # Extract to temp WAV next to model dir, cleaned up after
    tmp_wav = os.path.join(MODEL_DIR, "_tmp_audio16k.wav")
    try:
        extract_audio(video_path, tmp_wav, job_state=job_state)
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}") from e

    if model is None:
        if job_state:
            job_state.update(progress=5, message="Loading Whisper model (first run downloads it)...")
        model, device = _get_model(model_size)
    else:
        device = getattr(model, "_device", "unknown")

    if job_state:
        job_state.update(progress=10, message="Transcribing...")

    lang_note = f" (forced: {language})" if language else " (auto-detect)"
    log.info(f"Transcribing {video_path}{lang_note}")

    try:
        raw_iter, info = model.transcribe(
            tmp_wav,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )

        segments = []
        total_dur = info.duration or 1.0

        for seg in raw_iter:
            if job_state and job_state.cancelled:
                raise InterruptedError("Transcription cancelled")

            words = []
            for w in (seg.words or []):
                if job_state and job_state.cancelled:
                    raise InterruptedError("Transcription cancelled")
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                })
            segments.append({"start": seg.start, "end": seg.end,
                             "text": seg.text.strip(), "words": words})

            if job_state:
                pct = 10 + (seg.end / total_dur) * 85
                job_state.update(progress=pct,
                                 message=f"Transcribing... {seg.end:.0f}s/{total_dur:.0f}s")

        log.info(f"Transcribed {len(segments)} segments "
                 f"(language: {info.language}, prob {info.language_probability:.2f})")
        return segments, info

    finally:
        # Clean temp audio
        try:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except OSError:
            pass

def write_srt(segments, srt_path, words_per_line=8, max_lines=2, job_state=None):
    """Write word-level SRT grouped by word count per line. No mid-word breaks.

    Lines are grouped purely by word count (words_per_line), with at most
    max_lines lines per subtitle cue.
    """
    from .utils import timestamp_str

    cues = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            continue
        # Chunk words into lines of words_per_line
        lines = [words[i:i + words_per_line]
                 for i in range(0, len(words), words_per_line)]
        # Group lines into cues of max_lines
        for i in range(0, len(lines), max_lines):
            group = lines[i:i + max_lines]
            start = group[0][0]["start"]
            end = group[-1][-1]["end"]
            text = "\n".join(" ".join(w["word"] for w in line) for line in group)
            cues.append((start, end, text))

    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(cues, 1):
            f.write(f"{idx}\n")
            f.write(f"{timestamp_str(start)} --> {timestamp_str(end)}\n")
            f.write(f"{text}\n\n")

    log.info(f"Wrote SRT with {len(cues)} cues to {srt_path}")
    return srt_path

def transcribe_to_srt(video_path, output_dir, language=None, words_per_line=8,
                      max_lines=2, model=None, job_state=None, model_size="large-v3"):
    """Full pipeline: transcribe video, write SRT into output_dir/srt/.

    Returns (srt_path, segments, info).
    """
    if job_state:
        job_state.update(progress=2, message="Starting transcription...")

    segments, info = transcribe(video_path, model=model, language=language,
                                 job_state=job_state, words_per_line=words_per_line,
                                 model_size=model_size)

    srt_dir = os.path.join(output_dir, "srt")
    os.makedirs(srt_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    srt_path = os.path.join(srt_dir, f"{base}.srt")

    write_srt(segments, srt_path, words_per_line=words_per_line,
              max_lines=max_lines, job_state=job_state)

    # Plain transcript as bonus
    txt_path = os.path.join(srt_dir, f"{base}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(seg["text"] for seg in segments))

    if job_state:
        job_state.update(progress=100, message="Transcription complete")
    return srt_path, segments, info