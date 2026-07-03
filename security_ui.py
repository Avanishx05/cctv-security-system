"""
Offline AI Security Application — Desktop UI
=============================================
PyQt5 tabbed interface with:
  Tab 1 — Live Monitor  : webcam feed + real-time detection overlay
  Tab 2 — Alerts        : live alert feed with severity colour coding
  Tab 3 — Logs          : searchable JSONL log viewer
  Tab 4 — Query         : chat-style LLM interface over incident history

Requirements:
    pip install PyQt5 opencv-python ollama chromadb

Usage:
    python security_ui.py
"""

import sys
import json
import time
import threading
import cv2
import ollama
import chromadb
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QScrollArea, QFrame,
    QSplitter, QListWidget, QListWidgetItem, QComboBox
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QTextCursor
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "camera_index":       0,
    "frame_width":        1280,
    "frame_height":       720,
    "alert_log_path":     "alert_log.jsonl",
    "chroma_db_path":     "./chroma_db",
    "llm_model":          "llama3.2:3b",
    "llm_temperature":    0.3,
    "llm_max_tokens":     300,
    "ui_refresh_ms":      33,        # ~30fps UI refresh
    "alert_refresh_ms":   2000,      # Alert panel refresh interval
    "log_refresh_ms":     5000,      # Log panel refresh interval
    "max_alerts_shown":   50,        # Max alerts shown in alert tab
}

# Severity colour map — used across all tabs
SEVERITY_COLORS = {
    "CRITICAL": "#ff2222",
    "HIGH":     "#ff6600",
    "MEDIUM":   "#ffaa00",
    "LOW":      "#44bb44",
    "NONE":     "#888888",
}

# ══════════════════════════════════════════════════════════════════════════════
# DARK THEME STYLESHEET
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
    padding: 8px 20px;
    border: 1px solid #2d2d4e;
    border-bottom: none;
    min-width: 100px;
}
QTabBar::tab:selected {
    background: #1a1a2e;
    color: #e0e0e0;
    border-top: 2px solid #4a9eff;
}
QTabBar::tab:hover {
    background: #0f3460;
    color: #e0e0e0;
}
QPushButton {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #4a9eff;
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #4a9eff;
    color: #1a1a2e;
}
QPushButton:pressed {
    background-color: #2a6eff;
}
QPushButton:disabled {
    background-color: #2d2d4e;
    color: #555;
    border-color: #444;
}
QLineEdit {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #2d2d4e;
    border-radius: 4px;
    padding: 6px 10px;
}
QLineEdit:focus {
    border-color: #4a9eff;
}
QTextEdit {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #2d2d4e;
    border-radius: 4px;
    padding: 8px;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 12px;
}
QScrollArea {
    border: none;
    background: transparent;
}
QLabel {
    color: #e0e0e0;
}
QListWidget {
    background-color: #16213e;
    border: 1px solid #2d2d4e;
    border-radius: 4px;
    color: #e0e0e0;
}
QListWidget::item:selected {
    background-color: #0f3460;
    color: #4a9eff;
}
QComboBox {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #2d2d4e;
    border-radius: 4px;
    padding: 4px 10px;
}
QSplitter::handle {
    background: #2d2d4e;
}
QFrame[frameShape="4"] {  /* HLine */
    color: #2d2d4e;
}
"""

# ══════════════════════════════════════════════════════════════════════════════
# WEBCAM THREAD
# Runs camera capture in a separate thread to avoid blocking the UI
# ══════════════════════════════════════════════════════════════════════════════

class WebcamThread(QThread):
    """
    Captures webcam frames in a background thread.
    Emits frame_ready signal with each new frame as a QPixmap.
    Emits status_changed when camera connects/disconnects.
    """
    frame_ready    = pyqtSignal(QPixmap)
    status_changed = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config  = config
        self.running = False
        self.cap     = None

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.config["camera_index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.config["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["frame_height"])

        if not self.cap.isOpened():
            self.status_changed.emit("Camera not available")
            return

        self.status_changed.emit("Camera connected")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_changed.emit("Camera disconnected")
                break

            # Convert BGR → RGB → QImage → QPixmap
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg  = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.frame_ready.emit(pixmap)

            # Throttle to ~30fps
            time.sleep(1 / 30)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.quit()
        self.wait()


# ══════════════════════════════════════════════════════════════════════════════
# LLM QUERY THREAD
# Runs LLM inference in background so UI doesn't freeze
# ══════════════════════════════════════════════════════════════════════════════

class LLMThread(QThread):
    """
    Runs Ollama LLM inference in a background thread.
    Emits response_ready with the generated text when done.
    Emits error_occurred if something goes wrong.
    """
    response_ready  = pyqtSignal(str)
    error_occurred  = pyqtSignal(str)
    token_streamed  = pyqtSignal(str)   # For streaming token-by-token

    def __init__(self, messages: list, config: dict):
        super().__init__()
        self.messages = messages
        self.config   = config

    def run(self):
        try:
            # Stream response token by token for better UX
            full_response = ""
            stream = ollama.chat(
                model=self.config["llm_model"],
                messages=self.messages,
                stream=True,
                options={
                    "temperature": self.config["llm_temperature"],
                    "num_predict": self.config["llm_max_tokens"],
                }
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                full_response += token
                self.token_streamed.emit(token)

            self.response_ready.emit(full_response)

        except Exception as e:
            self.error_occurred.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class LiveMonitorTab(QWidget):
    """
    Shows live webcam feed with start/stop controls and camera status.
    Detection overlay is handled by the main security_app.py loop —
    this tab displays the raw feed for preview/monitoring.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config        = config
        self.webcam_thread = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        title  = QLabel("Live Monitor")
        title.setFont(QFont("SF Pro Display", 16, QFont.Bold))
        header.addWidget(title)
        header.addStretch()

        self.status_label = QLabel("● Idle")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        # ── Video feed ────────────────────────────────────────────────────────
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet(
            "background: #0d0d1a; border: 1px solid #2d2d4e; border-radius: 6px;"
        )
        self.video_label.setText("Camera feed will appear here.\nClick Start to begin.")
        layout.addWidget(self.video_label, stretch=1)

        # ── Controls ──────────────────────────────────────────────────────────
        controls = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Camera")
        self.stop_btn  = QPushButton("■  Stop Camera")
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)

        controls.addStretch()
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch()
        layout.addLayout(controls)

        # ── Info bar ──────────────────────────────────────────────────────────
        info = QLabel(
            "Note: Detection overlay runs in security_app.py. "
            "This tab shows the raw camera feed for monitoring."
        )
        info.setStyleSheet("color: #666; font-size: 11px;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

    def start_camera(self):
        self.webcam_thread = WebcamThread(self.config)
        self.webcam_thread.frame_ready.connect(self._update_frame)
        self.webcam_thread.status_changed.connect(self._update_status)
        self.webcam_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_camera(self):
        if self.webcam_thread:
            self.webcam_thread.stop()
            self.webcam_thread = None
        self.video_label.setText("Camera stopped.")
        self.status_label.setText("● Idle")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _update_frame(self, pixmap: QPixmap):
        # Scale pixmap to fit label while preserving aspect ratio
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _update_status(self, status: str):
        color = "#44bb44" if "connected" in status.lower() else "#ff4444"
        self.status_label.setText(f"● {status}")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 12px;")

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ALERTS
# ══════════════════════════════════════════════════════════════════════════════

class AlertsTab(QWidget):
    """
    Displays real-time alert feed loaded from alert_log.jsonl.
    Auto-refreshes every N seconds. Colour coded by severity.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config     = config
        self.log_path   = Path(config["alert_log_path"])
        self.last_count = 0
        self._build_ui()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_alerts)
        self.refresh_timer.start(config["alert_refresh_ms"])
        self.refresh_alerts()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        title  = QLabel("Alert Feed")
        title.setFont(QFont("SF Pro Display", 16, QFont.Bold))
        header.addWidget(title)
        header.addStretch()

        self.alert_count_label = QLabel("0 alerts")
        self.alert_count_label.setStyleSheet("color: #888; font-size: 12px;")
        header.addWidget(self.alert_count_label)

        # Severity filter
        self.severity_filter = QComboBox()
        self.severity_filter.addItems(["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
        self.severity_filter.currentTextChanged.connect(self.refresh_alerts)
        header.addWidget(self.severity_filter)

        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.clicked.connect(self.refresh_alerts)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        # ── Alert list ────────────────────────────────────────────────────────
        self.alert_list = QListWidget()
        self.alert_list.setSpacing(2)
        self.alert_list.itemClicked.connect(self._show_alert_detail)
        layout.addWidget(self.alert_list, stretch=1)

        # ── Detail panel ──────────────────────────────────────────────────────
        detail_label = QLabel("Alert Detail")
        detail_label.setStyleSheet("color: #888; font-size: 11px; font-weight: bold;")
        layout.addWidget(detail_label)

        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setMaximumHeight(160)
        self.detail_view.setPlaceholderText("Click an alert to see full details...")
        layout.addWidget(self.detail_view)

    def refresh_alerts(self):
        """Reloads alerts from JSONL log file and updates the list."""
        if not self.log_path.exists():
            self.alert_count_label.setText("No log file found")
            return

        # Read all alerts
        alerts = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    alerts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Apply severity filter
        selected_filter = self.severity_filter.currentText()
        if selected_filter != "All":
            alerts = [a for a in alerts if a.get("severity") == selected_filter]

        # Sort newest first
        alerts = sorted(alerts, key=lambda x: x.get("timestamp", ""), reverse=True)
        alerts = alerts[:self.config["max_alerts_shown"]]

        self.alert_count_label.setText(f"{len(alerts)} alert(s)")

        # Rebuild list only if count changed (avoid flicker)
        if len(alerts) == self.last_count and selected_filter == "All":
            return
        self.last_count = len(alerts)

        self.alert_list.clear()
        for alert in alerts:
            severity  = alert.get("severity", "NONE")
            timestamp = alert.get("timestamp", "")[:19].replace("T", " ")
            summary   = alert.get("alert_text", "").split("\n")[0][:80]
            color     = SEVERITY_COLORS.get(severity, "#888")

            item = QListWidgetItem(f"[{severity:8}]  {timestamp}  {summary}")
            item.setForeground(QColor(color))
            item.setData(Qt.UserRole, alert)  # Store full alert for detail view
            self.alert_list.addItem(item)

    def _show_alert_detail(self, item: QListWidgetItem):
        """Shows full alert JSON in the detail panel when an alert is clicked."""
        alert = item.data(Qt.UserRole)
        if alert:
            self.detail_view.setPlainText(json.dumps(alert, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LOGS
# ══════════════════════════════════════════════════════════════════════════════

class LogsTab(QWidget):
    """
    Searchable raw log viewer.
    Displays the full alert_log.jsonl with text search filtering.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config   = config
        self.log_path = Path(config["alert_log_path"])
        self._build_ui()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.load_logs)
        self.refresh_timer.start(config["log_refresh_ms"])
        self.load_logs()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        title  = QLabel("Incident Logs")
        title.setFont(QFont("SF Pro Display", 16, QFont.Bold))
        header.addWidget(title)
        header.addStretch()

        self.log_count_label = QLabel("")
        self.log_count_label.setStyleSheet("color: #888; font-size: 12px;")
        header.addWidget(self.log_count_label)
        layout.addLayout(header)

        # ── Search bar ────────────────────────────────────────────────────────
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search logs... (e.g. CRITICAL, unknown, 14:32)")
        self.search_input.textChanged.connect(self._apply_search)
        search_row.addWidget(self.search_input)

        clear_btn = QPushButton("✕ Clear")
        clear_btn.setMaximumWidth(80)
        clear_btn.clicked.connect(lambda: self.search_input.clear())
        search_row.addWidget(clear_btn)

        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.setMaximumWidth(90)
        refresh_btn.clicked.connect(self.load_logs)
        search_row.addWidget(refresh_btn)
        layout.addLayout(search_row)

        # ── Log viewer ────────────────────────────────────────────────────────
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("SF Mono", 11))
        self.log_view.setPlaceholderText("Logs will appear here once the security app generates alerts...")
        layout.addWidget(self.log_view, stretch=1)

    def load_logs(self):
        """Loads raw JSONL log and displays prettily formatted."""
        if not self.log_path.exists():
            self.log_view.setPlaceholderText(
                f"Log file not found: {self.log_path}\n"
                f"Start security_app.py to generate logs."
            )
            return

        lines = self.log_path.read_text().strip().splitlines()
        self.all_lines = lines
        self.log_count_label.setText(f"{len(lines)} entries")
        self._apply_search(self.search_input.text())

    def _apply_search(self, query: str):
        """Filters log lines by search query and updates display."""
        if not hasattr(self, "all_lines"):
            return

        lines = self.all_lines
        if query.strip():
            lines = [l for l in lines if query.lower() in l.lower()]

        # Format each line for display
        formatted = []
        for line in reversed(lines):  # newest first
            try:
                obj      = json.loads(line)
                ts       = obj.get("timestamp", "")[:19].replace("T", " ")
                severity = obj.get("severity", "NONE")
                summary  = obj.get("alert_text", "").split("\n")[0][:100]
                formatted.append(f"[{ts}] [{severity:8}] {summary}")
            except Exception:
                formatted.append(line)

        self.log_view.setPlainText("\n".join(formatted))
        # Scroll to top (newest)
        self.log_view.moveCursor(QTextCursor.Start)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LLM QUERY (CHAT STYLE)
# ══════════════════════════════════════════════════════════════════════════════

class QueryTab(QWidget):
    """
    Chat-style LLM interface for querying incident history.
    Multi-turn conversation — maintains message history for context.
    LLM has access to recent alert logs as RAG context.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config      = config
        self.log_path    = Path(config["alert_log_path"])
        self.messages    = []   # Full conversation history for multi-turn
        self.llm_thread  = None
        self._build_ui()
        self._add_system_message()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        title  = QLabel("Incident Query")
        title.setFont(QFont("SF Pro Display", 16, QFont.Bold))
        header.addWidget(title)
        header.addStretch()

        self.model_label = QLabel(f"Model: {self.config['llm_model']}")
        self.model_label.setStyleSheet("color: #888; font-size: 11px;")
        header.addWidget(self.model_label)

        clear_btn = QPushButton("↺ New Chat")
        clear_btn.clicked.connect(self._clear_chat)
        header.addWidget(clear_btn)
        layout.addLayout(header)

        # ── Chat display ──────────────────────────────────────────────────────
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("SF Pro Display", 12))
        layout.addWidget(self.chat_display, stretch=1)

        # ── Suggested prompts ─────────────────────────────────────────────────
        suggest_label = QLabel("Suggested queries:")
        suggest_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(suggest_label)

        suggest_row = QHBoxLayout()
        suggestions = [
            "How many alerts today?",
            "Any critical incidents?",
            "Unknown faces detected?",
            "Summarise recent activity",
        ]
        for s in suggestions:
            btn = QPushButton(s)
            btn.setStyleSheet(
                "QPushButton { background: #0d1b2a; border: 1px solid #2d2d4e; "
                "border-radius: 12px; padding: 4px 12px; font-size: 11px; }"
                "QPushButton:hover { border-color: #4a9eff; color: #4a9eff; }"
            )
            btn.clicked.connect(lambda checked, text=s: self._send_message(text))
            suggest_row.addWidget(btn)
        layout.addLayout(suggest_row)

        # ── Input row ─────────────────────────────────────────────────────────
        input_row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask about incidents, alerts, or patterns...")
        self.input_field.returnPressed.connect(self._on_send)
        input_row.addWidget(self.input_field)

        self.send_btn = QPushButton("Send ▶")
        self.send_btn.setMinimumWidth(80)
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.send_btn)
        layout.addLayout(input_row)

        # ── Status ────────────────────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.status_label)

    def _add_system_message(self):
        """Sets up the system prompt with security analyst persona."""
        system_content = (
            "You are a security incident analyst for an offline AI CCTV security system. "
            "You have access to detection logs from the system which records weapon detections "
            "(danger tiers: risky, dangerous) and face recognition events (known/unknown faces). "
            "Answer questions about incidents factually and concisely. "
            "When asked for summaries, structure your response clearly. "
            "If no log data is available for a query, say so directly."
        )
        self.messages = [{"role": "system", "content": system_content}]
        self._append_chat("System", "Security Analyst ready. Ask me about incidents, alerts, or patterns.", "#4a9eff")

    def _load_log_context(self, n_recent: int = 20) -> str:
        """
        Loads recent alerts from the log file to inject as RAG context.
        Returns a formatted string summary of recent events.
        """
        if not self.log_path.exists():
            return "No incident logs available yet."

        lines = self.log_path.read_text().strip().splitlines()
        if not lines:
            return "Log file exists but contains no entries."

        # Take most recent N entries
        recent = lines[-n_recent:]
        parsed = []
        for line in recent:
            try:
                obj      = json.loads(line)
                ts       = obj.get("timestamp", "")[:19].replace("T", " ")
                severity = obj.get("severity", "NONE")
                text     = obj.get("alert_text", "").replace("\n", " ")
                parsed.append(f"[{ts}] [{severity}] {text}")
            except Exception:
                continue

        if not parsed:
            return "Could not parse log entries."

        return f"Recent {len(parsed)} incidents:\n" + "\n".join(parsed)

    def _on_send(self):
        """Handles send button click and Enter key."""
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.clear()
        self._send_message(text)

    def _send_message(self, text: str):
        """Sends a user message and triggers LLM response."""
        if self.llm_thread and self.llm_thread.isRunning():
            return  # Don't stack requests

        # Show user message
        self._append_chat("You", text, "#e0e0e0")

        # Build context-injected user message
        # Inject current log context with every message for RAG
        log_context = self._load_log_context()
        context_message = (
            f"Current incident log context:\n{log_context}\n\n"
            f"User question: {text}"
        )

        # Add to conversation history
        self.messages.append({"role": "user", "content": context_message})

        # Disable input while LLM is thinking
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.status_label.setText("● Thinking...")

        # Start assistant response placeholder
        self._append_chat("Assistant", "", "#4a9eff")
        self._current_response = ""

        # Run LLM in background thread
        self.llm_thread = LLMThread(self.messages, self.config)
        self.llm_thread.token_streamed.connect(self._on_token)
        self.llm_thread.response_ready.connect(self._on_response_complete)
        self.llm_thread.error_occurred.connect(self._on_llm_error)
        self.llm_thread.start()

    def _on_token(self, token: str):
        """Appends each streamed token to the last assistant message."""
        self._current_response += token
        # Update last line in chat display with accumulated response
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Replace last assistant line with updated content
        doc  = self.chat_display.document()
        last = doc.lastBlock()
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertText(f"Assistant: {self._current_response}")
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _on_response_complete(self, full_response: str):
        """Called when LLM finishes generating."""
        # Add assistant response to conversation history
        self.messages.append({"role": "assistant", "content": full_response})

        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status_label.setText("")
        self._append_chat_separator()

    def _on_llm_error(self, error: str):
        """Handles LLM errors — shows message and re-enables input."""
        self._append_chat(
            "System",
            f"Error: {error}\nMake sure Ollama is running: ollama serve",
            "#ff4444"
        )
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status_label.setText("")

    def _append_chat(self, sender: str, message: str, color: str):
        """Appends a message to the chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        if message:
            self.chat_display.append(f'<span style="color:{color}"><b>{sender}:</b></span> {message}')
        else:
            self.chat_display.append(f'<span style="color:{color}"><b>{sender}:</b></span> ')

        self.chat_display.ensureCursorVisible()

    def _append_chat_separator(self):
        """Adds a visual separator between exchanges."""
        self.chat_display.append('<hr style="border-color: #2d2d4e;">')

    def _clear_chat(self):
        """Resets conversation history and display."""
        self.chat_display.clear()
        self.messages = []
        self._add_system_message()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class SecurityApp(QMainWindow):
    """
    Main application window.
    Houses all four tabs in a QTabWidget.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offline AI Security Monitor")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        self.setStyleSheet(DARK_STYLE)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Tab widget ────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.monitor_tab = LiveMonitorTab(CONFIG)
        self.alerts_tab  = AlertsTab(CONFIG)
        self.logs_tab    = LogsTab(CONFIG)
        self.query_tab   = QueryTab(CONFIG)

        self.tabs.addTab(self.monitor_tab, "📹  Live Monitor")
        self.tabs.addTab(self.alerts_tab,  "🚨  Alerts")
        self.tabs.addTab(self.logs_tab,    "📋  Logs")
        self.tabs.addTab(self.query_tab,   "💬  Query")

        layout.addWidget(self.tabs)

        # ── Status bar ────────────────────────────────────────────────────────
        self.statusBar().showMessage(
            f"  Offline AI Security Monitor  |  Log: {CONFIG['alert_log_path']}  "
            f"|  LLM: {CONFIG['llm_model']}"
        )
        self.statusBar().setStyleSheet(
            "QStatusBar { background: #0d0d1a; color: #555; font-size: 11px; "
            "border-top: 1px solid #2d2d4e; }"
        )

    def closeEvent(self, event):
        """Clean shutdown — stop webcam thread if running."""
        self.monitor_tab.stop_camera()
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
