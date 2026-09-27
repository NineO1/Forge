"""Forge shared utilities: logging, progress, cancel flag, output naming."""

import logging
import os
import sys
import time
from datetime import datetime

class JobState:
    """Tracks progress, messages, and cancellation for a running job.

    GUI (Phase 3) plugs into this via callbacks; CLI polls it directly.
    """

    def __init__(self):
        self.cancelled = False
        self.progress = 0.0          # 0.0 - 100.0 overall
        self.message = ""            # current status text
        self.started_at = None
        self.finished_at = None
        self.errors = []             # list of (stage, error_text)

    def start(self, message="Starting"):
        self.cancelled = False
        self.progress = 0.0
        self.message = message
        self.errors.clear()
        self.started_at = time.time()

    def finish(self, message="Done"):
        self.finished_at = time.time()
        self.message = message

    def update(self, progress=None, message=None):
        if progress is not None:
            self.progress = max(0.0, min(100.0, float(progress)))
        if message is not None:
            self.message = message

    def log_error(self, stage, error_text):
        self.errors.append((stage, str(error_text)))

    def cancel(self):
        self.cancelled = True

    @property
    def elapsed_str(self):
        if self.started_at is None:
            return "0s"
        end = self.finished_at or time.time()
        secs = int(end - self.started_at)
        if secs < 60:
            return f"{secs}s"
        mins, secs = divmod(secs, 60)
        return f"{mins}m {secs}s"


def setup_logging(log_file=None, level=logging.INFO):
    """Configure logging to console and optionally a file. Off by default."""
    root = logging.getLogger("forge")
    root.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    if not root.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        root.addHandler(console)

    if log_file and not any(
        isinstance(h, logging.FileHandler) for h in root.handlers
    ):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return root


def make_output_dir(output_root, source_path, allow_custom=None):
    """Create output/<video name>/ and return it.

    Custom name (if given) is used EXACTLY as typed - no prefixes/suffixes.
    """
    if allow_custom:
        name = os.path.splitext(os.path.basename(allow_custom))[0]
    else:
        name = os.path.splitext(os.path.basename(source_path))[0]
    safe = "".join(c if c not in '\\/:*?"<>|' else "_" for c in name).strip()
    if not safe:
        safe = "untitled"
    out_dir = os.path.join(output_root, safe)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def timestamp_str(seconds):
    """Seconds -> HH:MM:SS.mmm string (SRT-compatible)."""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"