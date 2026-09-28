"""Forge GUI - PyQt6 dark theme frontend for the Forge engine.
Batch 2 rev4: forced GPU cleanup, guaranteed completion signal.
"""

import os
import sys
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QCheckBox, QProgressBar,
    QTextEdit, QGroupBox, QMessageBox, QScrollArea, QRadioButton,
    QComboBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction

from forge import FORGE_VERSION

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

ACCENT = "#6F00FF"
READY_BORDER = "border: 1px solid #6F00FF;"
DISABLED_STYLE = "QGroupBox { color: #777777; }"

class LiveLog(QTextEdit):
    def append_line(self, text):
        self.append(text)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

class JobWorker(QThread):
    """Runs a job function in background; ensures clean GPU shutdown."""

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, job_fn):
        super().__init__()
        self.job_fn = job_fn
        self._last_emit = 0.0
        self._last_pct = 0.0
        self._abort = False

    def _emit_progress(self, pct, msg):
        """Always emit final 100% regardless of throttle."""
        if pct >= 99.9:
            self.progress.emit(100.0, msg or "Finalizing...")
            return
        now = time.monotonic()
        if now - self._last_emit >= 0.25 or pct != self._last_pct:
            self._last_emit = now
            self._last_pct = pct
            self.progress.emit(pct, msg or "")

    def run(self):
        from forge.utils import JobState
        import gc

        worker = self

        class GuiState(JobState):
            def update(self, progress=None, message=None):
                super().update(progress, message)
                if progress is not None:
                    worker._emit_progress(progress, message or "")

        gs = GuiState()
        result = None
        try:
            result = self.job_fn(gs)
        except Exception as e:
            import traceback
            self.failed.emit(traceback.format_exc(limit=8))
            return
        finally:
            # Force GPU cleanup after any diffusers/torch usage
            try:
                import torch
                if torch.cuda.is_available():
                    del gs
                    gc.collect()
                    torch.cuda.empty_cache()
            except Exception:
                pass

        self.finished_ok.emit(result)

    def cancel(self):
        self._abort = True

class ForgeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Forge {FORGE_VERSION}")
        self.setMinimumSize(1000, 720)
        self.worker = None

        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff;
                font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; }
            QGroupBox { border: 1px solid #3a3a3a; border-radius: 4px;
                margin-top: 12px; padding-top: 8px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; padding: 0 8px; }
            QPushButton { background-color: #6F00FF; color: white; border: none;
                border-radius: 4px; padding: 8px 16px; }
            QPushButton:hover { background-color: #8A3FFF; }
            QPushButton:disabled { background-color: #4a4a4a; color: #999999; }
            QCheckBox { spacing: 8px; color: #ffffff; }
            QRadioButton { spacing: 8px; color: #ffffff; }
            QLineEdit { background-color: #2a2a2a; border: 1px solid #3a3a3a;
                border-radius: 4px; padding: 6px; color: #ffffff; }
            QLineEdit:focus { border: 1px solid #6F00FF; }
            QTextEdit { background-color: #0a0a0a; border: 1px solid #3a3a3a;
                border-radius: 4px; color: #33ff33;
                font-family: "Consolas", monospace; }
            QProgressBar { border: 1px solid #3a3a3a; border-radius: 4px;
                text-align: center; background-color: #2a2a2a; }
            QProgressBar::chunk { background-color: #6F00FF; }
            QMenuBar { background-color: #1e1e1e; color: #ffffff; }
            QMenuBar::item:selected, QMenu::item:selected { background-color: #6F00FF; }
            QMenu { background-color: #1e1e1e; color: #ffffff; border: 1px solid #3a3a3a; }
        """)

        self._create_menus()
        self._create_ui()
        self._connect_signals()
        self.refresh_relevance()

    def _create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        exit_a = QAction("Exit", self)
        exit_a.triggered.connect(self.close)
        file_menu.addAction(exit_a)
        help_menu = menubar.addMenu("Help")
        about_a = QAction("About Forge", self)
        about_a.triggered.connect(lambda: QMessageBox.information(
            self, "About Forge",
            f"Forge {FORGE_VERSION}\n\n"
            "Video analysis & AI generation\n"
            "CUDA-accelerated\n\n"
            "Concept and vision: ThNwFx\n"
            "Built in collaboration with Lumo (Proton AI)\n\n"
            "This app would not exist without them."))
        help_menu.addAction(about_a)

    def _create_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(16, 16, 16, 16)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        lay = QVBoxLayout(content)
        lay.setSpacing(12)

        # INPUT
        g_in = QGroupBox("INPUT")
        v = QHBoxLayout(g_in)
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Import a video OR an image (drag-drop or browse)...")
        self.input_btn = QPushButton("Browse...")
        v.addWidget(QLabel("Source:"))
        v.addWidget(self.input_path, 1)
        v.addWidget(self.input_btn)
        self.g_in = g_in

        # OUTPUT
        g_out = QGroupBox("OUTPUT")
        v = QHBoxLayout(g_out)
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Default: project output\\ ...")
        self.output_btn = QPushButton("Browse...")
        v.addWidget(QLabel("Destination:"))
        v.addWidget(self.output_path, 1)
        v.addWidget(self.output_btn)
        self.g_out = g_out

        # EXTRACTION FEATURES
        g_feat = QGroupBox("EXTRACTION FEATURES")
        v = QVBoxLayout(g_feat)
        self.feat_status = QLabel("")
        v.addWidget(self.feat_status)
        self.cb_scenes = QCheckBox("Scene Detection - cut samples")
        self.cb_detect = QCheckBox("Object Detection - timeline + preview")
        self.cb_transcribe = QCheckBox("Transcription - SRT")
        self.cb_frames = QCheckBox("Frames - contact sheet + thumbnails")
        for cb in (self.cb_scenes, self.cb_detect, self.cb_transcribe, self.cb_frames):
            v.addWidget(cb)
        self.g_feat = g_feat

        # GENERATION
        g_gen = QGroupBox("AI GENERATION")
        v = QVBoxLayout(g_gen)
        self.gen_status = QLabel("")
        v.addWidget(self.gen_status)
        v.addWidget(QLabel("Prompt:"))
        prow = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Describe what to generate...")
        self.magic_btn = QPushButton("\u2728 Magic")
        self.magic_btn.setToolTip("Rewrite this prompt into a cinematic, motion-rich version")
        prow.addWidget(self.prompt_input, 1)
        prow.addWidget(self.magic_btn)
        v.addLayout(prow)
        # MAGIC PANEL - read-only suggestion display
        self.magic_panel = QTextEdit()
        self.magic_panel.setReadOnly(True)
        self.magic_panel.setPlaceholderText(
            "Press Magic to get a cinematic suggestion for your prompt...")
        self.magic_panel.setMaximumHeight(90)
        self.magic_panel.setStyleSheet(
            "QTextEdit { color: #cfb2ff; font-family: 'Segoe UI'; }")
        mrow = QHBoxLayout()
        mrow.addStretch()
        self.magic_insert_btn = QPushButton("Insert")
        self.magic_insert_btn.setFixedWidth(90)
        self.magic_clear_btn = QPushButton("Clear")
        self.magic_clear_btn.setFixedWidth(90)
        mrow.addWidget(self.magic_insert_btn)
        mrow.addWidget(self.magic_clear_btn)
        v.addWidget(self.magic_panel)
        v.addLayout(mrow)

        v.addWidget(QLabel("Negative prompt (optional):"))
        self.neg_prompt_input = QLineEdit()
        v.addWidget(self.neg_prompt_input)

        # POLISH OPTIONS
        pol_row = QHBoxLayout()
        self.pol_upscale = QCheckBox("Upscale 2x (1080p target)")
        self.pol_interp = QCheckBox("Frame interpolate to 48fps")
        pol_row.addWidget(self.pol_upscale)
        pol_row.addWidget(self.pol_interp)
        v.addLayout(pol_row)

        row = QHBoxLayout()
        self.rb_t2i = QRadioButton("Text -> Image")
        self.rb_i2i = QRadioButton("Image -> Image")
        self.rb_t2v = QRadioButton("Text -> Video")
        self.rb_i2v = QRadioButton("Image -> Video")
        self.rb_t2i.setChecked(True)
        for rb in (self.rb_t2i, self.rb_i2i, self.rb_t2v, self.rb_i2v):
            row.addWidget(rb)
            rb.toggled.connect(lambda *_: self.refresh_relevance())
        v.addLayout(row)

        v.addWidget(QLabel("Init image (for Image-> modes):"))
        self.init_img_path = QLineEdit()
        self.init_img_path.setPlaceholderText("Browse for an image (Image-> modes)")
        self.init_img_btn = QPushButton("Browse...")
        irow = QHBoxLayout()
        irow.addWidget(self.init_img_path, 1)
        irow.addWidget(self.init_img_btn)
        v.addLayout(irow)

        # GEN SETTINGS
        srow1 = QHBoxLayout()
        srow1.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.res_combo.addItem("Sweet spot (704x480)", (704, 480))
        self.res_combo.addItem("Widescreen 16:9 (1024x576)", (1024, 576))
        self.res_combo.addItem("Compact 16:9 (512x288)", (512, 288))
        self.res_combo.addItem("Classic 4:3 (640x480)", (640, 480))
        srow1.addWidget(self.res_combo, 1)
        srow1.addWidget(QLabel("Steps:"))
        self.steps_combo = QComboBox()
        self.steps_combo.addItem("30 (fast)", 30)
        self.steps_combo.addItem("50 (default)", 50)
        self.steps_combo.addItem("70 (slow)", 70)
        self.steps_combo.setCurrentIndex(1)
        srow1.addWidget(self.steps_combo)
        v.addLayout(srow1)

        srow2 = QHBoxLayout()
        srow2.addWidget(QLabel("Duration (video):"))
        self.dur_combo = QComboBox()
        self.dur_combo.addItem("~3s (73 frames)", 73)
        self.dur_combo.addItem("~5s (121 frames)", 121)
        self.dur_combo.addItem("~8s (193 frames)", 193)
        self.dur_combo.addItem("~10s (241 frames)", 241)
        self.dur_combo.setCurrentIndex(1)
        srow2.addWidget(self.dur_combo, 1)
        srow2.addWidget(QLabel("Seed (0=random):"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        srow2.addWidget(self.seed_spin)
        v.addLayout(srow2)
        self.g_gen = g_gen

        # ACTION
        g_act = QGroupBox("ACTION")
        v = QHBoxLayout(g_act)
        self.start_btn = QPushButton("START")
        self.start_btn.setMinimumWidth(140)
        self.cancel_btn = QPushButton("CANCEL")
        self.cancel_btn.setEnabled(False)
        v.addWidget(self.start_btn)
        v.addWidget(self.cancel_btn)
        v.addStretch()
        self.g_act = g_act

        # LOG
        g_log = QGroupBox("LIVE LOG")
        v = QVBoxLayout(g_log)
        self.log = LiveLog()
        self.log.setMaximumHeight(180)
        self.copy_log_btn = QPushButton("Copy Log")
        v.addWidget(self.log)
        v.addWidget(self.copy_log_btn)
        self.g_log = g_log

        # STATUS
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setValue(0)

        lay.addWidget(self.g_in)
        lay.addWidget(self.g_out)
        lay.addWidget(self.g_feat)
        lay.addWidget(self.g_gen)
        lay.addWidget(self.g_act)
        lay.addWidget(self.g_log)
        lay.addWidget(self.status_label)
        lay.addWidget(self.progress)
        lay.addStretch()
        outer.addWidget(scroll)
        self.setAcceptDrops(True)

    def _connect_signals(self):
        self.input_btn.clicked.connect(self.browse_source)
        self.output_btn.clicked.connect(lambda: self.browse_folder(self.output_path))
        self.init_img_btn.clicked.connect(self.browse_init_image)
        self.input_path.textChanged.connect(self.refresh_relevance)
        self.prompt_input.textChanged.connect(self.refresh_relevance)
        self.magic_btn.clicked.connect(self.do_magic)
        self.magic_insert_btn.clicked.connect(self.insert_magic)
        self.magic_clear_btn.clicked.connect(self.clear_magic)
        self.init_img_path.textChanged.connect(self.refresh_relevance)
        self.copy_log_btn.clicked.connect(self.copy_log)
        self.start_btn.clicked.connect(self.start_processing)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        for cb in (self.cb_scenes, self.cb_detect, self.cb_transcribe, self.cb_frames):
            cb.stateChanged.connect(lambda *_: self.refresh_relevance())

    def do_magic(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            self.magic_panel.setPlainText(
                "Write a prompt first, then press Magic.")
            return
        from forge.magic import enhance_prompt
        enhanced, neg_added = enhance_prompt(prompt)
        self.magic_panel.setPlainText(enhanced)
        self.append_log(f"[MAGIC] Original prompt: {prompt}")
        self.append_log(f"[MAGIC] Extended negative prompt will be applied")

    def insert_magic(self):
        text = self.magic_panel.toPlainText().strip()
        if not text or text.startswith("Write a prompt"):
            return
        self.prompt_input.setText(text)
        self.append_log("[MAGIC] Enhanced prompt inserted into prompt field")

    def clear_magic(self):
        self.magic_panel.clear()

    def input_type(self):
        ext = os.path.splitext(self.input_path.text())[1].lower()
        if ext in VIDEO_EXTS:
            return "video"
        if ext in IMAGE_EXTS:
            return "image"
        return None

    def effective_init_image(self):
        p = self.init_img_path.text().strip()
        if p and os.path.isfile(p):
            return p
        return None

    def refresh_relevance(self):
        itype = self.input_type()
        prompt = self.prompt_input.text().strip()
        init_img = self.effective_init_image()

        is_video = itype == "video"
        for cb in (self.cb_scenes, self.cb_detect, self.cb_transcribe, self.cb_frames):
            cb.setEnabled(is_video)
        if itype == "image":
            self.g_feat.setStyleSheet(DISABLED_STYLE)
            self.feat_status.setText("Not applicable - image input")
        elif is_video:
            self.g_feat.setStyleSheet("")
            n_sel = sum(cb.isChecked() for cb in
                        (self.cb_scenes, self.cb_detect, self.cb_transcribe, self.cb_frames))
            self.feat_status.setText(f"Will run: {n_sel} extraction feature(s)" if n_sel
                                     else "Select features to run on this video")
        else:
            self.g_feat.setStyleSheet("")
            self.feat_status.setText("Import a video to enable")

        rb, mode = self._current_gen_mode()
        needs_image = mode in ("i2i", "i2v")
        self.dur_combo.setEnabled(mode in ("t2v", "i2v"))

        gen_ready = bool(prompt) or (needs_image and init_img) or \
                    (needs_image and itype == "image")
        video_feed_note = needs_image and not init_img and itype == "video"

        if itype == "image" and not init_img:
            self.init_img_path.setPlaceholderText(
                f"Using your imported image: {os.path.basename(self.input_path.text())}")
        elif video_feed_note:
            self.init_img_path.setPlaceholderText(
                "Auto: middle frame of your video will be used")
        else:
            self.init_img_path.setPlaceholderText(
                "Browse for an image (Image-> modes)")

        if gen_ready:
            self.g_gen.setStyleSheet(f"QGroupBox {{ {READY_BORDER} }}")
            if prompt or not needs_image:
                note = " (using middle frame of video)" if video_feed_note else ""
                self.gen_status.setText(f"READY: {rb.text()} will run{note}")
            else:
                self.gen_status.setText(
                    f"READY: {rb.text()} will run using your imported image")
        else:
            self.g_gen.setStyleSheet("")
            if not prompt and not needs_image:
                self.gen_status.setText("Enter a prompt or import an image")
            elif needs_image:
                self.gen_status.setText(
                    f"{rb.text()} needs an image - import one, or a video for auto-feed")
            else:
                self.gen_status.setText("Enter a prompt")

    def _current_gen_mode(self):
        pairs = [(self.rb_t2i, "t2i"), (self.rb_i2i, "i2i"),
                 (self.rb_t2v, "t2v"), (self.rb_i2v, "i2v")]
        for rb, mode in pairs:
            if rb.isChecked():
                return rb, mode
        return self.rb_t2i, "t2i"

    def browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "",
            "Media (*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.input_path.setText(path)

    def browse_init_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Init Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.init_img_path.setText(path)

    def browse_folder(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            line_edit.setText(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            ext = os.path.splitext(path)[1].lower()
            if ext in VIDEO_EXTS or ext in IMAGE_EXTS:
                self.input_path.setText(path)

    def append_log(self, text):
        self.log.append_line(text)

    def copy_log(self):
        self.log.selectAll()
        self.log.copy()
        self.append_log("[LOG COPIED]")

    def _auto_frame_from_video(self, video_path, output_dir):
        try:
            from forge.ffmpeg_tools import ffmpeg_path, video_duration
            import subprocess
            os.makedirs(output_dir, exist_ok=True)
            dur = video_duration(video_path)
            mid = max(0.0, dur / 2 - 0.5)
            out = os.path.join(output_dir, "auto_init_frame.png")
            r = subprocess.run(
                [ffmpeg_path(), "-y", "-ss", f"{mid:.3f}", "-i", video_path,
                 "-frames:v", "1", out],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if r.returncode == 0 and os.path.isfile(out):
                self.append_log(f"[INFO] Auto-fed middle frame ({mid:.1f}s) as init image")
                return out
        except Exception as e:
            self.append_log(f"[WARN] Auto-frame failed: {e}")
        return None

    def start_processing(self):
        if self.worker and self.worker.isRunning():
            self.append_log("[ERROR] A job is already running")
            return

        src = self.input_path.text().strip()
        itype = self.input_type()
        prompt = self.prompt_input.text().strip()
        neg_prompt = self.neg_prompt_input.text().strip()

        feats = {
            "scenes": itype == "video" and self.cb_scenes.isChecked(),
            "detect": itype == "video" and self.cb_detect.isChecked(),
            "transcribe": itype == "video" and self.cb_transcribe.isChecked(),
            "frames": itype == "video" and self.cb_frames.isChecked(),
        }

        prompt = self.prompt_input.text().strip()
        neg_prompt = self.neg_prompt_input.text().strip()
        from forge.magic import EXTENDED_NEGATIVE_PROMPT
        if EXTENDED_NEGATIVE_PROMPT and not neg_prompt.strip():
            neg_prompt = EXTENDED_NEGATIVE_PROMPT
        gen_w, gen_h = self.res_combo.currentData()
        gen_steps = self.steps_combo.currentData()
        gen_frames = self.dur_combo.currentData()
        gen_seed = self.seed_spin.value() or None

        rb, mode = self._current_gen_mode()
        needs_image = mode in ("i2i", "i2v")
        init_img = self.effective_init_image()
        run_gen = bool(prompt) or (needs_image and init_img) or \
                  (needs_image and itype == "image")

        if not any(feats.values()) and not run_gen:
            self.append_log("[ERROR] Nothing to do - select features or write a prompt")
            return

        output_root = self.output_path.text().strip() or \
            os.path.join(PROJECT_ROOT, "output")
        src_name = os.path.splitext(os.path.basename(src))[0] if src else "generation"
        base_out = os.path.join(output_root, src_name)

        if itype == "image" and not init_img:
            init_img = src
            self.append_log("[INFO] Imported image used as generation source")

        gui = self

        def job_fn(ws):
            results = {"stages": {}, "elapsed": "?", "output_dir": base_out}

            if any(feats.values()):
                from forge.core import run_job
                sub = run_job(src, output_root=output_root, features=feats,
                              job_state=ws)
                results["stages"].update(sub.get("stages", {}))
                results["elapsed"] = sub.get("elapsed", "?")
                results["output_dir"] = sub.get("output_dir", base_out)

            if run_gen:
                from forge.generate import text_to_image, image_to_image, \
                    text_to_video, image_to_video
                init_img_local = init_img
                if needs_image and not init_img_local:
                    if itype == "video":
                        auto = gui._auto_frame_from_video(src, output_root)
                        if not auto:
                            results["stages"]["generate"] = {
                                "error": "no init image available"}
                            return results
                        init_img_local = auto
                    elif itype == "image":
                        init_img_local = src

                out_dir = results["output_dir"]
                ws.update(progress=0, message=f"Generating ({mode})...")
                if mode == "t2i":
                    p = text_to_image(prompt, out_dir, filename="generated.png",
                                      negative_prompt=neg_prompt, width=gen_w,
                                      height=gen_h, steps=gen_steps,
                                      seed=gen_seed, job_state=ws)
                elif mode == "i2i":
                    p = image_to_image(prompt, init_img_local, out_dir,
                                       filename="generated.png",
                                       negative_prompt=neg_prompt, job_state=ws)
                elif mode == "t2v":
                    p = text_to_video(prompt, out_dir, filename="generated.mp4",
                                      negative_prompt=neg_prompt, width=gen_w,
                                      height=gen_h, steps=gen_steps,
                                      num_frames=gen_frames, seed=gen_seed,
                                      job_state=ws)
                else:
                    p = image_to_video(prompt, init_img_local, out_dir,
                                       filename="generated.mp4",
                                       negative_prompt=neg_prompt, width=gen_w,
                                       height=gen_h, steps=gen_steps,
                                       num_frames=gen_frames, seed=gen_seed,
                                       job_state=ws)
                results["stages"]["generate"] = {"output": p}
                ws.update(progress=80, message="Generation done - polishing...")
                
                # Post-process if requested
                do_up = self.pol_upscale.isChecked()
                do_int = self.pol_interp.isChecked()
                if do_up or do_int:
                    from forge.polish import polish_video
                    polished = p.replace(".mp4", "_polished.mp4")
                    polished = polish_video(p, polished, do_upscale=do_up,
                                            do_interpolate=do_int, job_state=ws)
                    results["stages"]["generate"]["output"] = polished
                    results["stages"]["polish"] = {"output": polished}
                ws.update(progress=100, message="Polish complete")
            return results

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.append_log(f"[INFO] START: {os.path.basename(src) if src else 'generation only'}")
        self.append_log(f"[INFO] Modes: {[k for k, v in feats.items() if v]}"
                        + (f" + {mode}" if run_gen else ""))

        self.worker = JobWorker(job_fn)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_progress(self, pct, msg):
        self.progress.setValue(int(pct))
        if msg:
            self.status_label.setText(msg)

    def on_finished(self, result):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(100)
        self.status_label.setText("Done")
        self.append_log(f"[DONE] Elapsed: {result.get('elapsed', '?')}")
        for feat, res in result.get("stages", {}).items():
            if "error" in res:
                self.append_log(f"[ERROR] {feat}: {res['error']}")
            elif "output" in res:
                self.append_log(f"[OK] {feat} -> {res['output']}")
            else:
                self.append_log(f"[OK] {feat} completed")
        self.append_log(f"[OUTPUT] {result.get('output_dir')}")

    def on_failed(self, err):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Failed")
        self.append_log(f"[ERROR] {err}")

    def cancel_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.append_log("[INFO] Cancel requested - current stage will finish")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ForgeGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()