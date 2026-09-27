"""Object/person detection via MediaPipe Tasks with timeline JSON and annotated preview."""

import json
import os
import urllib.request

from .utils import setup_logging

log = setup_logging()

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "mediapipe")
DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
)
DETECTOR_MODEL_PATH = os.path.join(MODELS_DIR, "efficientdet_lite0.tflite")

def _ensure_model():
    """Download detector model once into models/mediapipe/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.isfile(DETECTOR_MODEL_PATH):
        log.info(f"Downloading detector model (one time, ~4MB)...")
        urllib.request.urlretrieve(DETECTOR_MODEL_URL, DETECTOR_MODEL_PATH)
    return DETECTOR_MODEL_PATH

def detect_objects(video_path, labels=None, sample_step=15, job_state=None,
                   annotate=True, output_dir=None):
    """Scan a video for objects/people. Returns (timeline, stats).

    Args:
        video_path: input video
        labels: list of COCO labels to keep, e.g. ['person', 'car'].
                None = keep everything detected.
        sample_step: analyze every Nth frame (15 ~= 2x/sec at 30fps)
        job_state: progress/cancel tracking
        annotate: also write annotated preview MP4 with boxes drawn
        output_dir: required when annotate=True

    Returns:
        timeline: list of {t, kind, score, box:[x,y,w,h] normalized 0-1}
        stats: dict with counts
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    options = mp_vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        score_threshold=0.5,
        max_results=10,
    )
    detector = mp_vision.ObjectDetector.create_from_options(options)

    labels = [l.lower() for l in labels] if labels else None
    timeline = []
    stats = {"frames_scanned": 0, "hits": 0, "duration": 0.0,
             "labels_found": {}}

    writer = None
    annotated_path = None
    if annotate:
        if not output_dir:
            raise ValueError("output_dir required when annotate=True")
        det_dir = os.path.join(output_dir, "detect")
        os.makedirs(det_dir, exist_ok=True)
        annotated_path = os.path.join(det_dir, "preview_annotated.mp4")
        log.info(f"Annotated preview will be written to {annotated_path}")

    try:
        frame_idx = 0
        last_ms = -1
        while True:
            if job_state and job_state.cancelled:
                raise InterruptedError("Detection cancelled")

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                t = frame_idx / fps
                ts_ms = int(t * 1000)
                if ts_ms <= last_ms:
                    ts_ms = last_ms + 1
                last_ms = ts_ms

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_image, ts_ms)

                annotated = frame.copy() if annotate else None

                for det in result.detections:
                    cat = det.categories[0]
                    kind = cat.category_name.lower()
                    if labels and kind not in labels:
                        continue
                    b = det.bounding_box
                    hit = {
                        "t": round(t, 3),
                        "kind": kind,
                        "score": round(cat.score, 3),
                        "box": [round(b.origin_x / w, 4),
                                round(b.origin_y / h, 4),
                                round(b.width / w, 4),
                                round(b.height / h, 4)],
                    }
                    timeline.append(hit)
                    stats["hits"] += 1
                    stats["labels_found"][kind] = stats["labels_found"].get(kind, 0) + 1

                    if annotate:
                        x, y, bw, bh = hit["box"]
                        p1 = (int(x * w), int(y * h))
                        p2 = (int((x + bw) * w), int((y + bh) * h))
                        color = (80, 170, 80) if kind == "person" else (255, 160, 60)
                        cv2.rectangle(annotated, p1, p2, color, 2)
                        cv2.putText(annotated, f"{kind} {hit['score']}",
                                    (p1[0], p1[1] - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if annotate:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            annotated_path, fourcc, fps, (w, h))
                    writer.write(annotated)

                stats["frames_scanned"] += 1

                if job_state and total_frames:
                    pct = 5 + (frame_idx / total_frames) * 90
                    job_state.update(
                        progress=pct,
                        message=f"Detecting... frame {frame_idx}/{total_frames}")

            frame_idx += 1

    finally:
        cap.release()
        detector.close()
        if writer is not None:
            writer.release()

    stats["duration"] = round(frame_idx / fps, 2)
    log.info(f"Detection done: {stats['hits']} hits, "
             f"labels: {stats['labels_found']} over {stats['duration']}s")
    return timeline, stats

def save_timeline(timeline, stats, output_dir):
    """Write timeline.json into output_dir/detect/."""
    det_dir = os.path.join(output_dir, "detect")
    os.makedirs(det_dir, exist_ok=True)
    out = os.path.join(det_dir, "timeline.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "hits": timeline}, f, indent=2)
    log.info(f"Timeline saved to {out}")
    return out

def run_detection(video_path, output_dir, labels=None, sample_step=15,
                  job_state=None, annotate=True):
    """Full pipeline: detect + save timeline (+ optional annotated preview).

    Returns (timeline_path, timeline, stats).
    """
    if job_state:
        job_state.update(progress=2, message="Starting detection...")
    timeline, stats = detect_objects(
        video_path, labels=labels, sample_step=sample_step,
        job_state=job_state, annotate=annotate, output_dir=output_dir)
    tl_path = save_timeline(timeline, stats, output_dir)
    if job_state:
        job_state.update(progress=100, message="Detection complete")
    return tl_path, timeline, stats