"""
security_ui.py — Desktop UI (Single Launch)
============================================
Starts the full security pipeline internally and provides a tabbed
PyQt5 interface for monitoring and control.

Single entry point — run this instead of security_app.py:
    python security_ui.py

Tabs:
    1. Live Monitor  — video feed (only streams when tab is active)
    2. Logs          — searchable JSONL log viewer
    3. Faces         — register / view known / review unknowns / promote
    4. Query         — multi-turn LLM chat over incident history
    5. Settings      — config toggles and thresholds

Alerts:
    LOW/MEDIUM   → toast notification (auto-dismisses after 5s)
    HIGH/CRITICAL → modal popup (must dismiss)
"""

import sys
import cv2
import time
import json
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QScrollArea, QFrame,
    QSplitter, QListWidget, QListWidgetItem, QSlider,
    QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QDialog, QDialogButtonBox, QMessageBox, QGridLayout,
    QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QRect, QEasingCurve, QSize, QPoint
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette,
    QTextCursor, QIcon
)

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    PATHS, CAMERA, MOTION, YOLO, FACENET, LLM, ALERTS, UI,
    init_dirs, validate, warmup_chromadb
)
from modules.motion     import MotionDetector
from modules.detector   import DangerDetector
from modules.recognizer import FaceRecognizer
from modules.logger     import SecurityLogger
from modules.alerter    import Alerter

# ══════════════════════════════════════════════════════════════════════════════
# DARK THEME
# ══════════════════════════════════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #2d2d4e;
    background: #1a1a2e;
}
QTabBar::tab {
    background: #16213e;
    color: #888;
    padding: 10px 22px;
    border: 1px solid #2d2d4e;
    border-bottom: none;
    min-width: 110px;
}
QTabBar::tab:selected {
    background: #1a1a2e;
    color: #e0e0e0;
    border-top: 2px solid #4a9eff;
}
QTabBar::tab:hover { background: #0f3460; color: #e0e0e0; }
QPushButton {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #4a9eff;
    border-radius: 5px;
    padding: 7px 18px;
    font-weight: 500;
}
QPushButton:hover { background-color: #4a9eff; color: #1a1a2e; }
QPushButton:pressed { background-color: #2a6eff; }
QPushButton:disabled { background-color: #2d2d4e; color: #555; border-color: #444; }
QPushButton#danger {
    border-color: #ff4444;
    color: #ff4444;
}
QPushButton#danger:hover { background-color: #ff4444; color: #fff; }
QPushButton#success {
    border-color: #44bb44;
    color: #44bb44;
}
QPushButton#success:hover { background-color: #44bb44; color: #fff; }
QLineEdit {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #2d2d4e;
    border-radius: 4px;
    padding: 6px 10px;
}
QLineEdit:focus { border-color: #4a9eff; }
QTextEdit {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #2d2d4e;
    border-radius: 4px;
    padding: 8px;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 12px;
}
QListWidget {
    background-color: #16213e;
    border: 1px solid #2d2d4e;
    border-radius: 4px;
    color: #e0e0e0;
}
QListWidget::item { padding: 4px 8px; }
QListWidget::item:selected { background-color: #0f3460; color: #4a9eff; }
QGroupBox {
    border: 1px solid #2d2d4e;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    color: #888;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
QSlider::groove:horizontal {
    background: #2d2d4e; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #4a9eff; width: 14px; height: 14px;
    border-radius: 7px; margin: -5px 0;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #4a9eff; border-radius: 3px;
    background: #16213e;
}
QCheckBox::indicator:checked { background: #4a9eff; }
QScrollArea { border: none; background: transparent; }
QLabel { color: #e0e0e0; }
QSpinBox, QDoubleSpinBox {
    background: #16213e; color: #e0e0e0;
    border: 1px solid #2d2d4e; border-radius: 4px;
    padding: 4px 8px;
}
"""

SEVERITY_COLORS = {
    "CRITICAL": "#ff2222",
    "HIGH":     "#ff6600",
    "MEDIUM":   "#ffaa00",
    "LOW":      "#44bb44",
    "NONE":     "#888888",
}

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE THREAD
# Runs the full security pipeline in a background thread
# Emits signals to update the UI safely from the main thread
# ══════════════════════════════════════════════════════════════════════════════

class PipelineThread(QThread):
    """
    Runs the full security pipeline (motion → YOLO → FaceNet → alert → log)
    in a background thread. Emits Qt signals to bridge results to the UI.

    Signals:
        frame_ready:   new processed frame as QPixmap
        alert_fired:   new alert dict when severity is not NONE
        status_update: status string for status bar
    """
    frame_ready   = pyqtSignal(QPixmap)
    alert_fired   = pyqtSignal(dict)
    status_update = pyqtSignal(str)

    def __init__(self, modules: dict):
        super().__init__()
        self.modules       = modules
        self.running       = False
        self.stream_video  = False   # only process video when Live Monitor tab is active
        self.show_mask     = False

        # Shared state — read by UI, written by pipeline
        self._last_yolo    = {"objects": [], "alert_level": 0}
        self._last_facenet = {"faces": [], "known_count": 0, "unknown_count": 0}
        self._last_alert   = {"severity": "NONE", "suppressed": True}
        self._fps          = 0.0
        self._lock         = threading.Lock()

        # FaceNet background thread
        self._facenet_running = threading.Event()
        self._facenet_lock    = threading.Lock()

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(CAMERA["index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA["height"])

        if not cap.isOpened():
            self.status_update.emit("Camera failed to open")
            return

        motion_detector = self.modules["motion"]
        danger_detector = self.modules["detector"]
        face_recognizer = self.modules["recognizer"]
        security_logger = self.modules["logger"]
        alerter         = self.modules["alerter"]

        frame_count = 0
        fps_timer   = time.time()

        self.status_update.emit("Pipeline running")

        def run_facenet_async(frame_copy):
            try:
                result = face_recognizer.recognize(frame_copy)
                with self._facenet_lock:
                    self._last_facenet = result
            except Exception as e:
                pass
            finally:
                self._facenet_running.clear()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_update.emit("Camera read failed")
                break

            frame_count += 1

            # FPS
            if frame_count % 30 == 0:
                elapsed   = time.time() - fps_timer
                self._fps = 30 / elapsed if elapsed > 0 else 0
                fps_timer = time.time()

            # Motion detection
            should_infer, motion_mask = motion_detector.detect(frame)

            # Inference gate
            if should_infer:
                with self._lock:
                    self._last_yolo = danger_detector.detect(frame)

                if not self._facenet_running.is_set():
                    self._facenet_running.set()
                    t = threading.Thread(
                        target=run_facenet_async,
                        args=(frame.copy(),),
                        daemon=True
                    )
                    t.start()

                with self._lock:
                    yolo_out    = self._last_yolo
                    facenet_out = self._last_facenet

                alert = alerter.generate(yolo_out, facenet_out)
                security_logger.log_event(yolo_out, facenet_out, alert)

                with self._lock:
                    self._last_alert = alert

                # Emit alert signal for notifications
                if (not alert["suppressed"] and
                        alert["severity"] not in ("NONE", "suppressed")):
                    self.alert_fired.emit(alert)

            # Only build and emit frame if Live Monitor tab is watching
            if self.stream_video:
                with self._lock:
                    yolo_out    = self._last_yolo
                    facenet_out = self._last_facenet
                    alert       = self._last_alert

                vis = danger_detector.draw(frame, yolo_out)
                vis = face_recognizer.draw(vis, facenet_out)
                vis = self._draw_status_bar(vis, should_infer)

                # Convert to QPixmap
                rgb    = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg   = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.frame_ready.emit(pixmap)

        cap.release()
        self.status_update.emit("Pipeline stopped")

    def _draw_status_bar(self, frame: np.ndarray, motion: bool) -> np.ndarray:
        """Draws minimal status bar at top of frame."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 30), (25, 25, 40), -1)
        motion_text  = "● MOTION" if motion else "○ idle"
        motion_color = (0, 220, 0) if motion else (100, 100, 100)
        cv2.putText(frame, motion_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
        cv2.putText(frame, f"{self._fps:.1f}fps", (w - 70, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        return frame

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


# ══════════════════════════════════════════════════════════════════════════════
# LLM QUERY THREAD
# ══════════════════════════════════════════════════════════════════════════════

class LLMQueryThread(QThread):
    """Runs alerter.query_stream() in background, emits tokens as they arrive."""
    token_ready      = pyqtSignal(str)
    response_complete = pyqtSignal(str)
    error_occurred   = pyqtSignal(str)

    def __init__(self, alerter: Alerter, message: str):
        super().__init__()
        self.alerter  = alerter
        self.message  = message

    def run(self):
        try:
            full = ""
            for token in self.alerter.query_stream(self.message):
                full += token
                self.token_ready.emit(token)
            self.response_complete.emit(full)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TOAST NOTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

class ToastNotification(QFrame):
    """
    Animated toast notification that appears bottom-right and auto-dismisses.
    Used for LOW and MEDIUM severity alerts.
    """

    def __init__(self, parent, alert: dict):
        super().__init__(parent)
        severity = alert.get("severity", "LOW")
        color    = SEVERITY_COLORS.get(severity, "#888")

        self.setFixedSize(340, 80)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #16213e;
                border: 1px solid {color};
                border-left: 4px solid {color};
                border-radius: 6px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Severity label
        sev_label = QLabel(f"● {severity}")
        sev_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
        layout.addWidget(sev_label)

        # Summary text
        lines   = [l for l in alert.get("alert_text", "").split("\n") if l.strip()]
        summary = lines[1] if len(lines) > 1 else lines[0] if lines else ""
        msg     = QLabel(summary[:55] + "..." if len(summary) > 55 else summary)
        msg.setStyleSheet("color: #ccc; font-size: 11px;")
        msg.setWordWrap(True)
        layout.addWidget(msg)

        # Position bottom-right of parent
        self._reposition()

        # Auto-dismiss after 5 seconds
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.close)
        self._timer.start(5000)

        self.show()

    def _reposition(self):
        if self.parent():
            pw = self.parent().width()
            ph = self.parent().height()
            self.move(pw - self.width() - 16, ph - self.height() - 50)


# ══════════════════════════════════════════════════════════════════════════════
# ALERT POPUP DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class AlertPopup(QDialog):
    """
    Modal alert dialog for HIGH and CRITICAL severity.
    Must be dismissed by the user before they can continue.
    """

    def __init__(self, parent, alert: dict):
        super().__init__(parent)
        severity = alert.get("severity", "HIGH")
        color    = SEVERITY_COLORS.get(severity, "#ff6600")

        self.setWindowTitle(f"⚠ Security Alert — {severity}")
        self.setMinimumWidth(420)
        self.setStyleSheet(DARK_STYLE + f"""
            QDialog {{ border: 2px solid {color}; }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        # Severity header
        header = QLabel(f"⚠  {severity} ALERT")
        header.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {color};")
        layout.addWidget(sep)

        # Alert text
        text_view = QTextEdit()
        text_view.setReadOnly(True)
        text_view.setPlainText(alert.get("alert_text", ""))
        text_view.setMaximumHeight(120)
        layout.addWidget(text_view)

        # Timestamp + source
        meta = QLabel(
            f"Time: {alert.get('timestamp', '')[:19].replace('T', ' ')}  |  "
            f"Source: {alert.get('source', '?').upper()}"
        )
        meta.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(meta)

        # Dismiss button
        btn = QPushButton("Acknowledge Alert")
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: #fff;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
        """)
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class LiveMonitorTab(QWidget):
    """
    Shows live video feed with detection overlays.
    Video only streams when this tab is active — stops when you switch away.
    """

    def __init__(self, pipeline: PipelineThread):
        super().__init__()
        self.pipeline = pipeline
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header
        header = QHBoxLayout()
        title  = QLabel("Live Monitor")
        title.setFont(QFont("SF Pro Display", 15, QFont.Bold))
        header.addWidget(title)
        header.addStretch()

        self.status_label = QLabel("○ Camera off")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        # Video feed
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet(
            "background: #0d0d1a; border: 1px solid #2d2d4e; border-radius: 6px;"
        )
        self.video_label.setText("Camera feed will appear here.\nStart the pipeline first.")
        layout.addWidget(self.video_label, stretch=1)

        # Controls
        controls = QHBoxLayout()

        self.start_btn = QPushButton("▶  Start Pipeline")
        self.start_btn.setObjectName("success")
        self.start_btn.clicked.connect(self._start_pipeline)
        controls.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop Pipeline")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_pipeline)
        controls.addWidget(self.stop_btn)

        controls.addStretch()

        self.mask_btn = QPushButton("◈  Motion Mask")
        self.mask_btn.setCheckable(True)
        self.mask_btn.clicked.connect(self._toggle_mask)
        controls.addWidget(self.mask_btn)

        self.reset_btn = QPushButton("↺  Reset Background")
        self.reset_btn.clicked.connect(self._reset_bg)
        controls.addWidget(self.reset_btn)

        layout.addLayout(controls)

        # Connect pipeline signals
        self.pipeline.frame_ready.connect(self._update_frame)
        self.pipeline.status_update.connect(self._update_status)

    def on_tab_activated(self):
        """Called when this tab becomes active — start streaming video."""
        self.pipeline.stream_video = True

    def on_tab_deactivated(self):
        """Called when switching away — stop streaming to save CPU."""
        self.pipeline.stream_video = False
        self.video_label.setText("Switch to Live Monitor tab to see video.")

    def _start_pipeline(self):
        if not self.pipeline.isRunning():
            self.pipeline.start()
        self.pipeline.stream_video = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop_pipeline(self):
        self.pipeline.stop()
        self.pipeline.stream_video = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("Pipeline stopped.")

    def _toggle_mask(self, checked: bool):
        self.pipeline.show_mask = checked

    def _reset_bg(self):
        self.pipeline.modules["motion"].reset()

    def _update_frame(self, pixmap: QPixmap):
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _update_status(self, status: str):
        color = "#44bb44" if "running" in status.lower() else "#ff4444"
        self.status_label.setText(f"● {status}")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 12px;")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LOGS
# ══════════════════════════════════════════════════════════════════════════════

class LogsTab(QWidget):
    """Searchable JSONL alert log viewer with live refresh."""

    def __init__(self):
        super().__init__()
        self.log_path  = PATHS["alert_log"]
        self.all_lines = []
        self._build_ui()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.load_logs)
        self.refresh_timer.start(UI["log_refresh_ms"])
        self.load_logs()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header
        header = QHBoxLayout()
        title  = QLabel("Incident Logs")
        title.setFont(QFont("SF Pro Display", 15, QFont.Bold))
        header.addWidget(title)
        header.addStretch()

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #888; font-size: 12px;")
        header.addWidget(self.count_label)

        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.clicked.connect(self.load_logs)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        # Search
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Search logs... (e.g. CRITICAL, unknown, 14:32)"
        )
        self.search_input.textChanged.connect(self._apply_search)
        search_row.addWidget(self.search_input)

        clear_btn = QPushButton("✕")
        clear_btn.setMaximumWidth(36)
        clear_btn.clicked.connect(lambda: self.search_input.clear())
        search_row.addWidget(clear_btn)
        layout.addLayout(search_row)

        # Log viewer
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("SF Mono", 11))
        self.log_view.setPlaceholderText(
            "Logs will appear here once the pipeline generates alerts..."
        )
        layout.addWidget(self.log_view, stretch=1)

    def load_logs(self):
        if not self.log_path.exists():
            return
        self.all_lines = self.log_path.read_text().strip().splitlines()
        self.count_label.setText(f"{len(self.all_lines)} entries")
        self._apply_search(self.search_input.text())

    def _apply_search(self, query: str):
        lines = self.all_lines
        if query.strip():
            lines = [l for l in lines if query.lower() in l.lower()]

        formatted = []
        for line in reversed(lines):
            try:
                obj      = json.loads(line)
                ts       = obj.get("timestamp", "")[:19].replace("T", " ")
                severity = obj.get("severity", "NONE")
                summary  = obj.get("alert_text", "").split("\n")[0][:100]
                formatted.append(f"[{ts}] [{severity:8}] {summary}")
            except Exception:
                formatted.append(line)

        self.log_view.setPlainText("\n".join(formatted))
        self.log_view.moveCursor(QTextCursor.Start)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FACES
# ══════════════════════════════════════════════════════════════════════════════

class FacesTab(QWidget):
    """
    Face management tab.
    - Register new faces via webcam
    - View known faces list
    - Review unknown face logs
    - Promote unknown → known DB
    - Clear unknown logs + cache
    """

    def __init__(self, modules: dict):
        super().__init__()
        self.modules    = modules
        self.recognizer = modules["recognizer"]
        self._build_ui()
        self._refresh_known()
        self._refresh_unknowns()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # ── Left panel — known faces ──────────────────────────────────────────
        left = QVBoxLayout()

        known_group = QGroupBox("Known Faces")
        known_layout = QVBoxLayout(known_group)

        self.known_list = QListWidget()
        known_layout.addWidget(self.known_list)

        register_btn = QPushButton("＋  Register New Face")
        register_btn.setObjectName("success")
        register_btn.clicked.connect(self._register_face)
        known_layout.addWidget(register_btn)

        left.addWidget(known_group)
        layout.addLayout(left, stretch=1)

        # ── Right panel — unknown faces ───────────────────────────────────────
        right = QVBoxLayout()

        unknown_group = QGroupBox("Unknown Face Logs")
        unknown_layout = QVBoxLayout(unknown_group)

        self.unknown_list = QListWidget()
        self.unknown_list.itemClicked.connect(self._preview_unknown)
        unknown_layout.addWidget(self.unknown_list)

        # Preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(140)
        self.preview_label.setStyleSheet(
            "background: #0d0d1a; border: 1px solid #2d2d4e; border-radius: 4px;"
        )
        self.preview_label.setText("Click an unknown face to preview")
        unknown_layout.addWidget(self.preview_label)

        # Name input for promotion
        name_row = QHBoxLayout()
        self.promote_name = QLineEdit()
        self.promote_name.setPlaceholderText("Enter name to promote selected...")
        name_row.addWidget(self.promote_name)

        promote_btn = QPushButton("↑ Promote")
        promote_btn.clicked.connect(self._promote_unknown)
        name_row.addWidget(promote_btn)
        unknown_layout.addLayout(name_row)

        # Clear button
        clear_btn = QPushButton("🗑  Clear Unknown Logs + Cache")
        clear_btn.setObjectName("danger")
        clear_btn.clicked.connect(self._clear_unknowns)
        unknown_layout.addWidget(clear_btn)

        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.clicked.connect(self._refresh_unknowns)
        unknown_layout.addWidget(refresh_btn)

        right.addWidget(unknown_group)
        layout.addLayout(right, stretch=1)

    def _refresh_known(self):
        self.known_list.clear()
        for name in self.recognizer.known_people:
            self.known_list.addItem(f"  ✓  {name}")

    def _refresh_unknowns(self):
        self.unknown_list.clear()
        log_files = sorted(
            PATHS["unknown_logs_dir"].glob("*.jpg"), reverse=True
        )[:50]
        for f in log_files:
            self.unknown_list.addItem(f.name)
        self.unknown_list.setProperty("files", [str(f) for f in log_files])

    def _preview_unknown(self, item: QListWidgetItem):
        """Shows a preview of the selected unknown face crop."""
        idx      = self.unknown_list.row(item)
        files    = sorted(PATHS["unknown_logs_dir"].glob("*.jpg"), reverse=True)[:50]
        if idx < len(files):
            pixmap = QPixmap(str(files[idx]))
            self.preview_label.setPixmap(
                pixmap.scaled(self.preview_label.size(),
                              Qt.KeepAspectRatio,
                              Qt.SmoothTransformation)
            )

    def _register_face(self):
        """Opens a dialog to get the name then triggers webcam registration."""
        name, ok = self._ask_name("Register New Face", "Enter name:")
        if not ok or not name:
            return

        self._show_info(
            f"Registering '{name}'...\n\n"
            f"4 shots will be captured with 1.5s delay between each.\n"
            f"Vary your head angle slightly between shots."
        )
        try:
            self.recognizer.register_person(name=name, n_shots=4, delay_between=1.5)
            self.modules["logger"].sync_known_faces(PATHS["known_faces_dir"])
            self._refresh_known()
            self._show_info(f"'{name}' registered successfully.")
        except Exception as e:
            self._show_error(f"Registration failed: {e}")

    def _promote_unknown(self):
        """Promotes selected unknown face to known DB."""
        item = self.unknown_list.currentItem()
        if not item:
            self._show_error("Select an unknown face first.")
            return

        name = self.promote_name.text().strip()
        if not name:
            self._show_error("Enter a name to promote to.")
            return

        files = sorted(PATHS["unknown_logs_dir"].glob("*.jpg"), reverse=True)[:50]
        idx   = self.unknown_list.row(item)
        if idx >= len(files):
            return

        try:
            self.recognizer.promote_unknown(str(files[idx]), name)
            self.modules["logger"].sync_known_faces(PATHS["known_faces_dir"])
            self._refresh_known()
            self._refresh_unknowns()
            self.promote_name.clear()
            self._show_info(f"Promoted to known: '{name}'")
        except Exception as e:
            self._show_error(f"Promotion failed: {e}")

    def _clear_unknowns(self):
        """Clears all unknown logs and cache after confirmation."""
        reply = QMessageBox.question(
            self, "Confirm Clear",
            "Clear all unknown face logs and cache?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.recognizer.clear_unknown_data(clear_logs=True, clear_cache=True)
            self._refresh_unknowns()
            self.preview_label.setText("Unknown logs cleared.")

    def _ask_name(self, title: str, prompt: str) -> tuple:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setStyleSheet(DARK_STYLE)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(prompt))
        input_field = QLineEdit()
        layout.addWidget(input_field)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        ok = dialog.exec_() == QDialog.Accepted
        return input_field.text().strip(), ok

    def _show_info(self, msg: str):
        QMessageBox.information(self, "Info", msg)

    def _show_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — QUERY
# ══════════════════════════════════════════════════════════════════════════════

class QueryTab(QWidget):
    """
    Multi-turn LLM chat interface connected directly to alerter.query_stream().
    Has access to full incident log history via RAG context injection.
    """

    def __init__(self, alerter: Alerter):
        super().__init__()
        self.alerter    = alerter
        self.llm_thread = None
        self._build_ui()
        self._add_welcome()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header
        header = QHBoxLayout()
        title  = QLabel("Incident Query")
        title.setFont(QFont("SF Pro Display", 15, QFont.Bold))
        header.addWidget(title)
        header.addStretch()

        self.model_label = QLabel(f"Model: {LLM['model']}")
        self.model_label.setStyleSheet("color: #888; font-size: 11px;")
        header.addWidget(self.model_label)

        clear_btn = QPushButton("↺ New Chat")
        clear_btn.clicked.connect(self._clear_chat)
        header.addWidget(clear_btn)
        layout.addLayout(header)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display, stretch=1)

        # Suggested queries
        suggest_label = QLabel("Suggested:")
        suggest_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(suggest_label)

        suggest_row = QHBoxLayout()
        for s in ["How many alerts today?", "Any critical incidents?",
                  "Unknown faces detected?", "Summarise recent activity"]:
            btn = QPushButton(s)
            btn.setStyleSheet("""
                QPushButton {
                    background: #0d1b2a; border: 1px solid #2d2d4e;
                    border-radius: 12px; padding: 4px 12px; font-size: 11px;
                }
                QPushButton:hover { border-color: #4a9eff; color: #4a9eff; }
            """)
            btn.clicked.connect(lambda _, text=s: self._send(text))
            suggest_row.addWidget(btn)
        layout.addLayout(suggest_row)

        # Input row
        input_row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(
            "Ask about incidents, alerts, or patterns..."
        )
        self.input_field.returnPressed.connect(self._on_send)
        input_row.addWidget(self.input_field)

        self.send_btn = QPushButton("Send ▶")
        self.send_btn.setMinimumWidth(80)
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.send_btn)
        layout.addLayout(input_row)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.status_label)

    def _add_welcome(self):
        self.chat_display.append(
            '<span style="color:#4a9eff"><b>System:</b></span> '
            'Security analyst ready. Ask me about incidents, alerts, or patterns.'
        )

    def _on_send(self):
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.clear()
        self._send(text)

    def _send(self, text: str):
        if self.llm_thread and self.llm_thread.isRunning():
            return
        if self.llm_thread:
            try:
                self.llm_thread.token_ready.disconnect()
                self.llm_thread.response_complete.disconnect()
                self.llm_thread.error_occurred.disconnect()
            except Exception:
                pass

        self.chat_display.append(
            f'<span style="color:#e0e0e0"><b>You:</b></span> {text}'
        )
        self.chat_display.append(
            '<span style="color:#4a9eff"><b>Assistant:</b></span> '
        )

        self._current_response = ""
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.status_label.setText("● Thinking...")

        self.llm_thread = LLMQueryThread(self.alerter, text)
        self.llm_thread.token_ready.connect(self._on_token)
        self.llm_thread.response_complete.connect(self._on_done)
        self.llm_thread.error_occurred.connect(self._on_error)
        self.llm_thread.start()

    def _on_token(self, token: str):
        self._current_response += token
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertText(
            f'Assistant: {self._current_response}'
        )
        self.chat_display.ensureCursorVisible()

    def _on_done(self, _):
        self.chat_display.append('<hr style="border-color:#2d2d4e;">')
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status_label.setText("")

    def _on_error(self, error: str):
        self.chat_display.append(
            f'<span style="color:#ff4444"><b>Error:</b></span> {error}<br>'
            f'Make sure Ollama is running: <code>ollama serve</code>'
        )
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status_label.setText("")

    def _clear_chat(self):
        self.chat_display.clear()
        self.alerter.reset_conversation()
        self._add_welcome()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

class SettingsTab(QWidget):
    """
    Live config toggles. Changes take effect immediately without restart.
    """

    def __init__(self, modules: dict):
        super().__init__()
        self.modules = modules
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Settings")
        title.setFont(QFont("SF Pro Display", 15, QFont.Bold))
        layout.addWidget(title)

        # ── Alert mode ────────────────────────────────────────────────────────
        alert_group = QGroupBox("Alert Mode")
        alert_layout = QVBoxLayout(alert_group)

        self.llm_toggle = QCheckBox("Use LLM alerts (requires Ollama running)")
        self.llm_toggle.setChecked(LLM["use_llm_alerts"])
        self.llm_toggle.toggled.connect(self._toggle_llm)
        alert_layout.addWidget(self.llm_toggle)

        alert_layout.addWidget(QLabel(
            "When off: fast rule-based alerts. "
            "When on: context-aware LLM alerts (~2-3s latency)."
        ))
        layout.addWidget(alert_group)

        # ── Detection ─────────────────────────────────────────────────────────
        detect_group = QGroupBox("Detection")
        detect_layout = QGridLayout(detect_group)

        detect_layout.addWidget(QLabel("YOLO confidence threshold:"), 0, 0)
        self.yolo_conf = QDoubleSpinBox()
        self.yolo_conf.setRange(0.1, 0.9)
        self.yolo_conf.setSingleStep(0.05)
        self.yolo_conf.setValue(YOLO["conf"])
        self.yolo_conf.valueChanged.connect(
            lambda v: YOLO.update({"conf": v})
        )
        detect_layout.addWidget(self.yolo_conf, 0, 1)

        detect_layout.addWidget(QLabel("Face similarity threshold:"), 1, 0)
        self.face_thresh = QDoubleSpinBox()
        self.face_thresh.setRange(0.4, 0.95)
        self.face_thresh.setSingleStep(0.01)
        self.face_thresh.setValue(FACENET["similarity_threshold"])
        self.face_thresh.valueChanged.connect(
            lambda v: FACENET.update({"similarity_threshold": v})
        )
        detect_layout.addWidget(self.face_thresh, 1, 1)
        layout.addWidget(detect_group)

        # ── Motion ────────────────────────────────────────────────────────────
        motion_group = QGroupBox("Motion Detection")
        motion_layout = QGridLayout(motion_group)

        motion_layout.addWidget(QLabel("Motion threshold (contour area):"), 0, 0)
        self.motion_thresh = QSpinBox()
        self.motion_thresh.setRange(100, 5000)
        self.motion_thresh.setSingleStep(100)
        self.motion_thresh.setValue(MOTION["threshold"])
        self.motion_thresh.valueChanged.connect(
            lambda v: MOTION.update({"threshold": v})
        )
        motion_layout.addWidget(self.motion_thresh, 0, 1)

        motion_layout.addWidget(QLabel("Idle inference every N frames:"), 1, 0)
        self.idle_n = QSpinBox()
        self.idle_n.setRange(10, 300)
        self.idle_n.setSingleStep(10)
        self.idle_n.setValue(MOTION.get("idle_every_n", 60))
        self.idle_n.valueChanged.connect(
            lambda v: MOTION.update({"idle_every_n": v})
        )
        motion_layout.addWidget(self.idle_n, 1, 1)
        layout.addWidget(motion_group)

        # ── Camera ────────────────────────────────────────────────────────────
        cam_group = QGroupBox("Camera")
        cam_layout = QGridLayout(cam_group)

        cam_layout.addWidget(QLabel("Camera index:"), 0, 0)
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 5)
        self.cam_index.setValue(CAMERA["index"])
        self.cam_index.valueChanged.connect(
            lambda v: CAMERA.update({"index": v})
        )
        cam_layout.addWidget(self.cam_index, 0, 1)
        cam_layout.addWidget(
            QLabel("Change takes effect on next pipeline start."), 1, 0, 1, 2
        )
        layout.addWidget(cam_group)

        layout.addStretch()

        # Reset button
        reset_btn = QPushButton("Reset All to Defaults")
        reset_btn.setObjectName("danger")
        reset_btn.clicked.connect(self._reset_defaults)
        layout.addWidget(reset_btn)

    def _toggle_llm(self, checked: bool):
        self.modules["alerter"].set_mode(checked)
        LLM["use_llm_alerts"] = checked

    def _reset_defaults(self):
        YOLO["conf"]                    = 0.4
        FACENET["similarity_threshold"] = 0.68
        MOTION["threshold"]             = 500
        MOTION["idle_every_n"]          = 60
        self.yolo_conf.setValue(0.4)
        self.face_thresh.setValue(0.68)
        self.motion_thresh.setValue(500)
        self.idle_n.setValue(60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class SecurityApp(QMainWindow):
    """
    Main application window.
    Initialises all backend modules and wires them to the UI tabs.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offline AI Security Monitor")
        self.setMinimumSize(1100, 750)
        self.resize(1280, 820)
        self.setStyleSheet(DARK_STYLE)

        # Active toasts list — track so we can reposition if multiple
        self._toasts = []

        # ── Initialise backend modules ────────────────────────────────────────
        self._init_backend()

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()

        # ── Alert polling timer ───────────────────────────────────────────────
        # Polls pipeline for new alerts every 500ms
        self.pipeline.alert_fired.connect(self._handle_alert)

    def _init_backend(self):
        """Initialises all modules and wraps them for the pipeline thread."""
        init_dirs()
        warmup_chromadb()
        validate()

        self.modules = {
            "motion":     MotionDetector(MOTION),
            "detector":   DangerDetector(YOLO),
            "recognizer": FaceRecognizer(FACENET, PATHS),
            "logger":     SecurityLogger(PATHS, LLM),
            "alerter":    Alerter(LLM, ALERTS, None),  # logger connected below
        }

        # Connect logger to alerter
        self.modules["logger"].sync_known_faces(PATHS["known_faces_dir"])
        self.modules["alerter"].logger = self.modules["logger"]

        # Create pipeline thread with shared modules
        self.pipeline = PipelineThread(self.modules)

        try:
            import requests
            requests.get("http://localhost:11434", timeout=1)
        except Exception:
            print("[Setup] Starting Ollama server...")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            import time
            time.sleep(3)  # give server time to start

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.monitor_tab  = LiveMonitorTab(self.pipeline)
        self.logs_tab     = LogsTab()
        self.faces_tab    = FacesTab(self.modules)
        self.query_tab    = QueryTab(self.modules["alerter"])
        self.settings_tab = SettingsTab(self.modules)

        self.tabs.addTab(self.monitor_tab,  "📹  Live Monitor")
        self.tabs.addTab(self.logs_tab,     "📋  Logs")
        self.tabs.addTab(self.faces_tab,    "👤  Faces")
        self.tabs.addTab(self.query_tab,    "💬  Query")
        self.tabs.addTab(self.settings_tab, "⚙   Settings")

        layout.addWidget(self.tabs)

        # Status bar
        self.statusBar().showMessage(
            f"  Offline AI Security Monitor  |  "
            f"Model: {LLM['model']}  |  "
            f"Camera: {CAMERA['index']}"
        )
        self.statusBar().setStyleSheet(
            "QStatusBar { background: #0d0d1a; color: #555; font-size: 11px; "
            "border-top: 1px solid #2d2d4e; }"
        )

        # Update status bar from pipeline
        self.pipeline.status_update.connect(
            lambda s: self.statusBar().showMessage(f"  Pipeline: {s}")
        )

    def _on_tab_changed(self, index: int):
        """Start/stop video streaming based on which tab is active."""
        if index == 0:
            self.monitor_tab.on_tab_activated()
        else:
            self.monitor_tab.on_tab_deactivated()

    def _handle_alert(self, alert: dict):
        """Routes alert to toast or popup based on severity."""
        severity = alert.get("severity", "NONE")

        if severity in ("HIGH", "CRITICAL"):
            popup = AlertPopup(self, alert)
            popup.exec_()
        elif severity in ("LOW", "MEDIUM"):
            toast = ToastNotification(self, alert)
            self._toasts.append(toast)
            # Clean up dismissed toasts
            self._toasts = [t for t in self._toasts if t.isVisible()]

    def closeEvent(self, event):
        """Clean shutdown."""
        self.pipeline.stop()
        # Clear unknown logs and cache on exit
        self.modules["recognizer"].clear_unknown_data(
            clear_logs=True,
            clear_cache=True
        )
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Offline AI Security Monitor")
    window = SecurityApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()