"""
modules/logger.py — Logging Module
====================================
Handles two types of logging:
  1. JSONL file logging  — one JSON object per line, human readable,
                           used by the UI alert and log tabs
  2. ChromaDB logging    — vector-searchable event store,
                           used by the LLM query interface for RAG retrieval

Usage:
    from modules.logger import SecurityLogger
    from config import PATHS, LLM

    logger = SecurityLogger(PATHS, LLM)

    # Log a detection event
    event_id = logger.log_event(yolo_output, facenet_output, alert)

    # Query past events (for LLM RAG context)
    context = logger.get_recent_context(n=10)
    similar = logger.search_similar("unknown face with weapon")
"""

import json
import chromadb
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from config import PATHS, LLM


class SecurityLogger:
    """
    Dual-channel logger for detection events and alerts.

    Channel 1 — JSONL file:
        Append-only log of all alert events.
        Read by the UI (alerts tab, logs tab).
        Human readable, easy to back up or export.

    Channel 2 — ChromaDB:
        Vector-searchable store of detection event descriptions.
        Queried by the LLM for RAG context retrieval.
        Enables semantic search — "find events similar to this one."

    Channel 3 — In-memory ring buffer:
        Last N detection events kept in memory.
        Used for fast pattern detection without hitting disk.
    """

    def __init__(self, paths: dict = PATHS, llm_config: dict = LLM):
        """
        Args:
            paths:      PATHS dict from config.py
            llm_config: LLM dict from config.py
        """
        self.alert_log_path = paths["alert_log"]
        self.chroma_db_path = paths["chroma_db"]
        self.history_window = llm_config["history_window"]

        # Ensure log directory exists
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

        # ── In-memory ring buffer ─────────────────────────────────────────────
        # Stores last N detection events for fast pattern detection
        # Does not persist between runs — rebuilt as events come in
        self._recent = deque(maxlen=self.history_window)

        # ── ChromaDB setup ────────────────────────────────────────────────────
        # PersistentClient = data survives between app restarts
        self._chroma  = chromadb.PersistentClient(path=str(self.chroma_db_path))

        # Detection history collection
        # Each document = one frame's detection event as a text description
        self._events  = self._chroma.get_or_create_collection(
            name="detection_history",
            metadata={"description": "Frame-level detection events"}
        )

        # Known faces collection
        # Stores registered person metadata for LLM context
        self._faces   = self._chroma.get_or_create_collection(
            name="known_faces",
            metadata={"description": "Registered known faces metadata"}
        )

        print(f"[SecurityLogger] JSONL log: {self.alert_log_path}")
        print(f"[SecurityLogger] ChromaDB: {self.chroma_db_path}")
        print(f"[SecurityLogger] Existing events in DB: {self._events.count()}")

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ──────────────────────────────────────────────────────────────────────────

    def log_event(self, yolo_output: dict, facenet_output: dict,
                  alert: dict) -> str:
        """
        Logs a complete detection event to all three channels.

        Args:
            yolo_output:    output dict from DangerDetector.detect()
            facenet_output: output dict from FaceRecognizer.recognize()
            alert:          output dict from Alerter.generate()

        Returns:
            event_id: unique string ID for this event
        """
        timestamp = datetime.now()
        event_id  = f"event_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

        # ── Build human-readable event description ────────────────────────────
        # This is what gets stored in ChromaDB and retrieved for LLM context
        event_text = self._build_event_text(
            timestamp, yolo_output, facenet_output
        )

        # ── Channel 1: JSONL file (only for non-trivial alerts) ───────────────
        if not alert.get("suppressed") and alert.get("severity") != "NONE":
            self._write_jsonl(
                event_id, timestamp, yolo_output, facenet_output, alert
            )

        # ── Channel 2: ChromaDB (all events for RAG) ──────────────────────────
        self._write_chroma(
            event_id, timestamp, event_text, yolo_output, facenet_output
        )

        # ── Channel 3: In-memory ring buffer ─────────────────────────────────
        self._recent.append({
            "event_id":      event_id,
            "timestamp":     timestamp,
            "alert_level":   yolo_output["alert_level"],
            "unknown_faces": facenet_output["unknown_count"],
            "event_text":    event_text
        })

        return event_id

    def sync_known_faces(self, known_faces_dir: Path):
        """
        Syncs the known faces registry into ChromaDB.
        Call this at startup and after registering new faces.
        Gives the LLM context about who is authorised to be in the space.

        Args:
            known_faces_dir: Path to known_faces/ folder
        """
        if not known_faces_dir.exists():
            return

        person_dirs = [d for d in known_faces_dir.iterdir() if d.is_dir()]
        synced = 0

        for person_dir in person_dirs:
            name   = person_dir.name
            images = list(person_dir.glob("*.jpg"))
            if not images:
                continue

            reg_date = datetime.fromtimestamp(
                images[0].stat().st_mtime
            ).strftime("%Y-%m-%d")

            doc_text = (
                f"{name} is a registered known person. "
                f"Registered on {reg_date} with {len(images)} reference image(s). "
                f"Should NOT trigger unknown face alerts."
            )

            self._faces.upsert(
                ids=[name],
                documents=[doc_text],
                metadatas=[{
                    "name":     name,
                    "registered": reg_date,
                    "n_images": len(images)
                }]
            )
            synced += 1

        print(f"[SecurityLogger] Synced {synced} known people to ChromaDB.")

    def get_recent_context(self, n: int = None) -> str:
        """
        Returns a formatted string of the most recent N detection events.
        Used as RAG context for LLM alert generation and query responses.

        Args:
            n: number of recent events to include (defaults to history_window)

        Returns:
            Formatted multi-line string of recent events
        """
        n       = n or self.history_window
        events  = list(self._recent)[-n:]

        if not events:
            return "No recent detection events."

        lines = [
            f"  - {e['event_text']}"
            for e in reversed(events)   # newest first
        ]

        return f"Recent {len(lines)} detection events:\n" + "\n".join(lines)

    def get_log_context(self, n: int = None) -> str:
        """
        Reads the last N alert entries from the JSONL log file.
        Used as RAG context for the LLM query interface.
        Includes more detail than get_recent_context (full alert text).

        Args:
            n: number of entries to include

        Returns:
            Formatted multi-line string of recent alerts
        """
        n = n or 20

        if not self.alert_log_path.exists():
            return "No alert log found."

        lines = self.alert_log_path.read_text().strip().splitlines()
        if not lines:
            return "Alert log is empty."

        recent  = lines[-n:]
        parsed  = []

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

        return f"Recent {len(parsed)} alerts:\n" + "\n".join(parsed)

    def search_similar(self, query: str, n_results: int = 5) -> str:
        """
        Semantic search over detection history in ChromaDB.
        Finds past events similar to the query string.
        Used by LLM query interface for context-aware responses.

        Args:
            query:     natural language search query
            n_results: number of similar events to return

        Returns:
            Formatted string of matching past events
        """
        if self._events.count() == 0:
            return "No events in database yet."

        try:
            results = self._events.query(
                query_texts=[query],
                n_results=min(n_results, self._events.count())
            )

            docs = results.get("documents", [[]])[0]
            if not docs:
                return "No similar events found."

            lines = [f"  - {doc}" for doc in docs]
            return f"Similar past events:\n" + "\n".join(lines)

        except Exception as e:
            return f"Search error: {e}"

    def get_known_faces_context(self) -> str:
        """
        Returns a summary of registered known faces for LLM context.
        """
        if self._faces.count() == 0:
            return "No known faces registered in the system."

        all_faces = self._faces.get()
        names     = [m["name"] for m in all_faces["metadatas"]]

        return (
            f"Registered known people ({len(names)}): {', '.join(names)}. "
            f"Any face NOT in this list is unknown and should be flagged."
        )

    @property
    def recent_events(self) -> list:
        """Returns the raw list of recent detection events from ring buffer."""
        return list(self._recent)

    @property
    def event_count(self) -> int:
        """Total events stored in ChromaDB."""
        return self._events.count()

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _build_event_text(self, timestamp: datetime,
                          yolo_output: dict,
                          facenet_output: dict) -> str:
        """
        Builds a human-readable description of a detection event.
        This is what gets stored in ChromaDB and retrieved for LLM context.
        """
        parts = [f"[{timestamp.strftime('%H:%M:%S')}]"]

        # Describe objects
        if yolo_output["objects"]:
            descs = [
                f"{o['class']} (conf={o['confidence']:.2f})"
                for o in yolo_output["objects"]
            ]
            tier_labels = {0: "safe", 1: "risky", 2: "dangerous"}
            tier_label  = tier_labels.get(yolo_output["alert_level"], "unknown")
            parts.append(
                f"Objects: {', '.join(descs)}. "
                f"Alert level: {tier_label}."
            )
        else:
            parts.append("No objects detected.")

        # Describe faces
        if facenet_output["faces"]:
            known_names   = [
                f["name"] for f in facenet_output["faces"] if f["is_known"]
            ]
            unknown_count = facenet_output["unknown_count"]

            if known_names:
                parts.append(f"Known people: {', '.join(known_names)}.")
            if unknown_count > 0:
                parts.append(f"{unknown_count} unknown face(s) detected.")
        else:
            parts.append("No faces detected.")

        return " ".join(parts)

    def _write_jsonl(self, event_id: str, timestamp: datetime,
                     yolo_output: dict, facenet_output: dict,
                     alert: dict):
        """
        Appends a structured alert record to the JSONL log file.
        One JSON object per line — easy to parse, grep, or tail.
        """
        record = {
            "event_id":      event_id,
            "timestamp":     timestamp.isoformat(),
            "severity":      alert.get("severity", "NONE"),
            "source":        alert.get("source", "placeholder"),
            "alert_text":    alert.get("alert_text", ""),
            "alert_level":   yolo_output["alert_level"],
            "objects":       yolo_output["objects"],
            "unknown_faces": facenet_output["unknown_count"],
            "known_faces":   facenet_output["known_count"],
            "faces": [
                {k: v for k, v in f.items() if k != "log_path"}
                for f in facenet_output["faces"]
            ]
        }

        with open(self.alert_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _write_chroma(self, event_id: str, timestamp: datetime,
                      event_text: str, yolo_output: dict,
                      facenet_output: dict):
        """
        Stores a detection event in ChromaDB for semantic search.
        ChromaDB auto-embeds the document text using its default embedder.
        """
        try:
            self._events.upsert(
                ids=[event_id],
                documents=[event_text],
                metadatas=[{
                    "timestamp":     timestamp.isoformat(),
                    "alert_level":   yolo_output["alert_level"],
                    "unknown_faces": facenet_output["unknown_count"],
                    "known_faces":   facenet_output["known_count"],
                    "n_objects":     len(yolo_output["objects"])
                }]
            )
        except Exception as e:
            print(f"[SecurityLogger] ChromaDB write error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run directly to test logger
# python modules/logger.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from config import PATHS, LLM, init_dirs
    init_dirs()

    logger = SecurityLogger(PATHS, LLM)

    # Simulate a detection event
    yolo_output = {
        "objects": [
            {"class": "dangerous", "danger_tier": 2,
             "confidence": 0.91, "bbox": [100, 150, 200, 300]}
        ],
        "alert_level": 2
    }
    facenet_output = {
        "faces": [
            {"name": "Unknown", "confidence": 0.41,
             "is_known": False, "bbox": [80, 50, 160, 140], "log_path": None}
        ],
        "known_count":   0,
        "unknown_count": 1
    }
    alert = {
        "severity":   "CRITICAL",
        "alert_text": "SEVERITY: CRITICAL\nSUMMARY: Unknown person with dangerous object.\nACTION: Immediate review.",
        "source":     "placeholder",
        "suppressed": False
    }

    event_id = logger.log_event(yolo_output, facenet_output, alert)
    print(f"\nLogged event: {event_id}")
    print(f"Total events in ChromaDB: {logger.event_count}")
    print(f"\nRecent context:\n{logger.get_recent_context()}")
    print(f"\nSearch results:\n{logger.search_similar('unknown face dangerous weapon')}")