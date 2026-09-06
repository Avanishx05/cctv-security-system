# Offline AI Security Application

A fully offline, real-time CCTV security system combining weapon detection,
face recognition, and an LLM-powered alert and query interface.

## What it does

- **Weapon detection** — YOLOv8 classifies objects into danger tiers (risky / dangerous)
- **Face recognition** — FaceNet identifies known vs unknown faces in real time
- **Motion gating** — inference only runs when motion is detected, saving compute
- **Smart alerts** — rule-based or LLM-generated alerts with context from recent history
- **Query interface** — chat with a local LLM about past incidents via the desktop UI

Everything runs **fully offline** on your machine. No cloud, no API keys.

---

## Requirements

- Python 3.10+
- macOS (Apple Silicon recommended) / Linux
- [Ollama](https://ollama.com) for LLM alerts and query interface
- Trained YOLOv8 weights (`danger_detection_best.pt`)

---

## First-time setup

### 1. Clone and enter the project

```bash
git clone git@github.com:Avanishx05/cctv-security-system.git
cd cctv-project
```

### 2. Add your model weights 

(https://drive.google.com/file/d/1XmUoePCm7nD0FxojouKHUcpd5yLpATzW/view?usp=sharing use my current weights if you do not wish to train your own model)

Copy your trained YOLOv8 weights into the `models/` folder:

```bash
mkdir -p models
cp /path/to/danger_detection_best.pt models/
```

### 3. Run setup

```bash
bash setup.sh
```

This will:
- Install all Python dependencies
- Create required project folders
- Pre-download the ChromaDB embedding model (~90MB, one-time)
- Install and pull Llama 3.2 3B via Ollama (~2GB, one-time)
- Verify your model weights are in place

### 4. Start the app

```bash
python3 security_app.py
```

---

## Project structure

```
cctv-project/
├── models/
│   └── danger_detection_best.pt   # YOLOv8 fine-tuned weights (you provide)
├── modules/
│   ├── __init__.py
│   ├── motion.py                  # Motion detection (frame differencing)
│   ├── detector.py                # YOLOv8 weapon detector
│   ├── recognizer.py              # FaceNet face recognizer
│   ├── logger.py                  # JSONL + ChromaDB event logger
│   └── alerter.py                 # Rule-based + LLM alert generator
├── embeddings/
│   └── known_embeddings.pkl       # Face embeddings DB (auto-generated)
├── known_faces/                   # Registered face images (auto-generated)
├── unknown_logs/                  # Unknown face crops (auto-generated)
├── logs/                          # App + alert logs (auto-generated)
├── chroma_db/                     # ChromaDB vector store (auto-generated)
├── config.py                      # All settings — edit this to tune behaviour
├── security_app.py                # Main pipeline
├── security_ui.py                 # Desktop UI (PyQt5)
├── setup.sh                       # First-time setup script
└── requirements.txt               # Python dependencies
```

---

## Controls (inside security_app.py)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `F` | Register a new face via webcam |
| `L` | Toggle LLM / rule-based alert mode |
| `M` | Toggle motion mask debug view |
| `R` | Reset background model |

---

## Desktop UI

Run the UI separately alongside the main app:

```bash
python3 security_ui.py
```

The UI has four tabs:
- **Live Monitor** — raw webcam feed
- **Alerts** — live alert feed with severity colour coding
- **Logs** — searchable raw incident log
- **Query** — chat-style LLM interface for querying past incidents

---

## Configuration

All settings are in `config.py`. Key things to tune:

| Setting | Location | Default | Description |
|---------|----------|---------|-------------|
| `camera_index` | `CAMERA` | `0` | Webcam index |
| `conf` | `YOLO` | `0.4` | Detection confidence threshold |
| `similarity_threshold` | `FACENET` | `0.68` | Face match strictness |
| `use_llm_alerts` | `LLM` | `False` | Enable LLM alerts |
| `idle_every_n` | `MOTION` | `60` | Idle inference interval (frames) |
| `alert_cooldown_s` | `LLM` | `30` | Min seconds between same-type alerts |

---

## Enabling LLM alerts

LLM alerts are disabled by default. To enable:

1. Make sure Ollama is running: `ollama serve`
2. In `config.py`, set `"use_llm_alerts": True` in the `LLM` dict
3. Or press `L` inside the running app to toggle live

---

## Registering known faces

1. Start the app: `python3 security_app.py`
2. Press `F` to open face registration
3. Enter the person's name when prompted in the terminal
4. Hold still and vary your head angle slightly between shots
5. The app rebuilds the embedding database automatically

To add more faces later, press `F` again at any time.

---

## Danger tiers

| Tier | Label | Colour | Examples |
|------|-------|--------|---------|
| 1 | Risky | Orange | Blunt weapons, glass |
| 2 | Dangerous | Red | Guns, pistols, knives |

---

## Alert severity logic (rule-based mode)

| Condition | Severity |
|-----------|----------|
| Unknown face + dangerous object | CRITICAL |
| Dangerous object only | HIGH |
| Unknown face + risky object | MEDIUM |
| Unknown face only | LOW |
| Known face / no threats | NONE |

---

## Troubleshooting

**`ModuleNotFoundError`** — run `python3 -m pip install -r requirements.txt`

**Camera not opening** — change `camera_index` in `config.py` to `1`

**All faces showing as Unknown** — press `F` to register known faces

**LLM alerts not working** — run `ollama serve` in a separate terminal, then press `L` in the app

**ChromaDB downloading on startup** — run `bash setup.sh` first to pre-download
