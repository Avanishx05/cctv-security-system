"""
config.py — Single source of truth for the Offline AI Security Application
==========================================================================
All modules import from here. Edit this file to tune behaviour.
Never hardcode paths or hyperparameters inside modules.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# All paths are relative to the project root (where security_app.py lives)
# ══════════════════════════════════════════════════════════════════════════════

# Project root — resolves to the directory this file is in
ROOT = Path(__file__).parent

PATHS = {
    # ── Model weights ─────────────────────────────────────────────────────────
    "yolo_model":        ROOT / "models" / "danger_detection_best.pt",

    # ── Face recognition ──────────────────────────────────────────────────────
    "known_faces_dir":   ROOT / "known_faces",      # one subfolder per person
    "unknown_logs_dir":  ROOT / "unknown_logs",     # timestamped unknown crops
    "embeddings_cache":  ROOT / "embeddings" / "known_embeddings.pkl",
    "unknown_cache":     ROOT / "embeddings" / "unknown_cache.pkl",     # dedup cache for unknown faces

    # ── Logs and DB ───────────────────────────────────────────────────────────
    "alert_log":         ROOT / "logs" / "alert_log.jsonl",
    "app_log":           ROOT / "logs" / "app.log",
    "chroma_db":         ROOT / "chroma_db",
}

# ══════════════════════════════════════════════════════════════════════════════
# CAMERA
# ══════════════════════════════════════════════════════════════════════════════

CAMERA = {
    "index":        0,       # 0 = built-in webcam, 1+ = external
    "width":        1280,
    "height":       720,
    "fps_target":   30,
}

# ══════════════════════════════════════════════════════════════════════════════
# MOTION DETECTION
# Gates all downstream inference — nothing runs without motion
# ══════════════════════════════════════════════════════════════════════════════

MOTION = {
    "threshold":         500,    # Min contour area (px²) to count as motion
                                 # Lower = more sensitive, higher = less noise
    "blur_kernel":       21,     # Gaussian blur kernel size (must be odd)
    "bg_update_rate":    0.05,   # How fast background model adapts (0-1)
                                 # Higher = adapts faster to lighting changes
    "cooldown_s":        0.5,    # Min seconds between motion-triggered inferences
    "idle_every_n":      60,     # Also run inference every N frames when idle
                                 # At 30fps: 60 = every 2 seconds
}

# ══════════════════════════════════════════════════════════════════════════════
# YOLOV8 — WEAPON / DANGER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

YOLO = {
    "model_path":   PATHS["yolo_model"],
    "conf":         0.4,         # Confidence threshold (0-1)
                                 # Lower = more detections, more false positives
    "imgsz":        640,         # Input resolution — must match training size
    "device":       "mps",       # mps = Apple Silicon, cuda = NVIDIA, cpu = fallback

    # Class names after relabelling (2-class setup)
    "class_names":  ["risky", "dangerous"],

    # Danger tier mapping — class name → tier ID
    # Used for overlay colours and alert severity
    "tier_map": {
        "risky":     1,
        "dangerous": 2,
    },

    # Display config per tier — colour in BGR, label text
    "tier_display": {
        -1: {"label": "Unknown",   "color": (128, 128, 128)},
        0:  {"label": "Safe",      "color": (0, 200, 0)},
        1:  {"label": "Risky",     "color": (0, 165, 255)},
        2:  {"label": "DANGEROUS", "color": (0, 0, 255)},
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# FACENET — FACE RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

FACENET = {
    "model_name":           "Facenet512",   # 512-d embeddings
    "detector_backend":     "mtcnn",        # Face detector (most accurate in deepface)
    "similarity_threshold": 0.68,           # Cosine similarity cutoff
                                            # Lower = stricter matching
    "alert_on_unknown":     True,           # Trigger alert when unknown face detected
    "min_face_confidence":  0.85,           # Min MTCNN detection confidence to process

    # Haar cascade pre-filter (cascaded inference gate)
    # Runs before FaceNet to skip expensive embedding on frames with no faces
    "haar_scale_factor":    1.1,
    "haar_min_neighbours":  4,
    "haar_min_size":        (30, 30),

    # Unknown face deduplication cache
    "unknown_dedup_threshold": 0.68,   # same as recognition threshold
    "unknown_cache_max":       500,    # max entries before oldest are pruned
}

# ══════════════════════════════════════════════════════════════════════════════
# RAG / LLM — ALERT GENERATION AND QUERY
# ══════════════════════════════════════════════════════════════════════════════

LLM = {
    "model":            "llama3.2:3b",      # Ollama model name
    "host":             "http://localhost:11434",
    "temperature":      0.3,                # Low = consistent, factual alerts
    "max_tokens":       150,                # Keep alerts concise
    "query_max_tokens": 500,                # More tokens for query responses
    "alert_cooldown_s": 30,                 # Min seconds between same-type alerts
    "history_window":   20,                 # Recent events kept in memory
    "log_context_n":    20,                 # Recent log entries injected as RAG context

    # Use LLM for real-time alerts (True) or rule-based placeholder (False)
    # Set True once you've verified Ollama is running correctly
    "use_llm_alerts":   False,
}

# ══════════════════════════════════════════════════════════════════════════════
# ALERTS
# ══════════════════════════════════════════════════════════════════════════════

ALERTS = {
    # Severity levels in priority order
    "severity_levels": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"],

    # Rules for placeholder (rule-based) alert generation
    # Evaluated in order — first match wins
    "rules": [
        {
            "condition": {"min_tier": 2, "min_unknowns": 1},
            "severity":  "CRITICAL",
            "summary":   "Unknown person detected with dangerous object.",
            "action":    "Immediate review required."
        },
        {
            "condition": {"min_tier": 2, "min_unknowns": 0},
            "severity":  "HIGH",
            "summary":   "Dangerous object detected.",
            "action":    "Verify identity of person in frame."
        },
        {
            "condition": {"min_tier": 1, "min_unknowns": 1},
            "severity":  "MEDIUM",
            "summary":   "Unknown person detected with risky object.",
            "action":    "Monitor situation closely."
        },
        {
            "condition": {"min_tier": 0, "min_unknowns": 1},
            "severity":  "LOW",
            "summary":   "Unknown face detected, no weapons.",
            "action":    "Log for review."
        },
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

UI = {
    "alert_refresh_ms":  2000,   # How often alert tab polls log file
    "log_refresh_ms":    5000,   # How often log tab polls log file
    "max_alerts_shown":  50,     # Max alerts shown in alert tab
    "video_fps":         30,     # Target webcam display FPS

    # Severity colours for UI (hex, used in PyQt5)
    "severity_colors": {
        "CRITICAL": "#ff2222",
        "HIGH":     "#ff6600",
        "MEDIUM":   "#ffaa00",
        "LOW":      "#44bb44",
        "NONE":     "#888888",
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# ENSURE REQUIRED DIRECTORIES EXIST
# Called once at startup by security_app.py
# ══════════════════════════════════════════════════════════════════════════════

def init_dirs():
    """Creates all required project directories if they don't exist."""
    dirs = [
        PATHS["known_faces_dir"],
        PATHS["unknown_logs_dir"],
        PATHS["embeddings_cache"].parent,
        PATHS["alert_log"].parent,
        PATHS["chroma_db"],
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)




def warmup_chromadb():
    """
    Pre-downloads the ChromaDB embedding model (~90MB) so it does not
    interrupt the first run of the app mid-startup.
    Safe to call multiple times — skips download if already cached.
    """
    try:
        import chromadb
        print("[Setup] Warming up ChromaDB embedding model (one-time download)...")
        client = chromadb.PersistentClient(path=str(PATHS["chroma_db"]))
        col    = client.get_or_create_collection("warmup")
        col.upsert(ids=["warmup"], documents=["warmup"])
        client.delete_collection("warmup")
        print("[Setup] ChromaDB ready.")
    except Exception as e:
        print(f"[Setup] ChromaDB warmup failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# Catches missing files early before the main loop starts
# ══════════════════════════════════════════════════════════════════════════════

def validate():
    """
    Checks that critical files exist before starting the app.
    Prints warnings for optional files (embeddings) that can be created at runtime.
    Raises FileNotFoundError for required files (model weights).
    """
    errors   = []
    warnings = []

    # Required — app cannot run without this
    if not PATHS["yolo_model"].exists():
        errors.append(
            f"YOLOv8 weights not found: {PATHS['yolo_model']}\n"
            f"Copy best.pt to models/best.pt"
        )

    # Optional — app runs with degraded functionality without these
    if not PATHS["embeddings_cache"].exists():
        warnings.append(
            f"No face embeddings found at {PATHS['embeddings_cache']}. "
            f"All faces will be flagged as Unknown until you register known faces."
        )

    if not PATHS["known_faces_dir"].exists():
        warnings.append("known_faces/ directory not found — will be created.")

    for w in warnings:
        print(f"[WARNING] {w}")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        raise FileNotFoundError("Missing required files. See errors above.")

    print("[OK] Config validated.")


if __name__ == "__main__":
    # Run directly to validate config and print all settings
    print("=== Offline AI Security App — Config ===\n")
    init_dirs()
    validate()
    print("\nPaths:")
    for k, v in PATHS.items():
        print(f"  {k:20} {v}")
    print(f"\nDevice: {YOLO['device']}")
    print(f"LLM:    {LLM['model']} (alerts={'LLM' if LLM['use_llm_alerts'] else 'placeholder'})")
    print(f"Camera: {CAMERA['index']} ({CAMERA['width']}x{CAMERA['height']})")