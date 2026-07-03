"""
modules/recognizer.py — FaceNet Face Recognition Module
========================================================
MTCNN face detection + FaceNet512 embedding + cosine similarity matching.
Uses a Haar cascade pre-filter to skip FaceNet on frames with no faces
(cascaded inference gate — cheap model guards expensive model).

Usage:
    from modules.recognizer import FaceRecognizer
    from config import FACENET, PATHS

    recognizer = FaceRecognizer(FACENET, PATHS)
    result     = recognizer.recognize(frame)

    # result = {
    #     'faces': [
    #         {
    #             'name':       str,    # matched name or 'Unknown'
    #             'confidence': float,  # cosine similarity score
    #             'is_known':   bool,
    #             'bbox':       [x1, y1, x2, y2],
    #             'log_path':   str or None
    #         }
    #     ],
    #     'known_count':   int,
    #     'unknown_count': int
    # }
"""

import cv2
import uuid
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from deepface import DeepFace
from config import FACENET, PATHS



class UnknownFaceCache:
    """
    Persistent deduplication cache for unknown face detections.

    Prevents the same unknown person from being logged repeatedly.
    Each cache entry stores:
        - embedding:   512-d FaceNet vector for this unknown face
        - first_seen:  ISO timestamp of first detection
        - log_path:    path to the saved crop on first detection
        - seen_count:  how many times this person has been detected

    Cache persists to disk between app sessions.
    On load, if cache file exists it is restored from pkl.
    """

    def __init__(self, cache_path, threshold: float = 0.68, max_entries: int = 500):
        """
        Args:
            cache_path: Path to pkl file for persistent storage
            threshold:  Cosine similarity above which = same person (skip log)
            max_entries: Max cache size before oldest entries pruned
        """
        self.cache_path  = Path(cache_path)
        self.threshold   = threshold
        self.max_entries = max_entries
        self._cache      = []   # list of dicts, ordered oldest → newest
        self._load()

    def is_duplicate(self, embedding: list) -> bool:
        """
        Checks if this embedding matches any cached unknown face.
        Returns True if a match is found (skip logging), False if new.
        Also increments seen_count on match.
        """
        for entry in self._cache:
            score = self._cosine_similarity(embedding, entry["embedding"])
            if score >= self.threshold:
                entry["seen_count"] += 1
                entry["last_seen"]   = datetime.now().isoformat()
                self._save()
                return True
        return False

    def add(self, embedding: list, log_path: str):
        """
        Adds a new unknown face to the cache.
        Prunes oldest entries if cache exceeds max_entries.
        """
        self._cache.append({
            "embedding":  embedding,
            "first_seen": datetime.now().isoformat(),
            "last_seen":  datetime.now().isoformat(),
            "log_path":   log_path,
            "seen_count": 1
        })

        # Prune oldest entries if over limit
        if len(self._cache) > self.max_entries:
            self._cache = self._cache[-self.max_entries:]

        self._save()

    def clear(self):
        """Clears the cache entirely — useful for testing or resetting."""
        self._cache = []
        self._save()
        print("[UnknownFaceCache] Cache cleared.")

    @property
    def size(self) -> int:
        return len(self._cache)

    def _cosine_similarity(self, a: list, b: list) -> float:
        a = np.array(a)
        b = np.array(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _save(self):
        """Persists cache to pkl file."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f)

    def _load(self):
        """Loads cache from pkl file if it exists."""
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)
            print(f"[UnknownFaceCache] Loaded {len(self._cache)} cached unknowns.")
        else:
            print("[UnknownFaceCache] No existing cache — starting fresh.")


class FaceRecognizer:
    """
    Face recognition pipeline:
        Haar pre-filter → MTCNN detection → FaceNet512 embedding
        → cosine similarity → known / unknown decision

    Handles:
    - Loading and caching face embeddings from pkl file
    - Logging unknown face crops to unknown_logs/
    - Deduplicating unknown faces via persistent embedding cache
    - Rebuilding embedding cache when new faces are registered
    """

    def __init__(self, config: dict = FACENET, paths: dict = PATHS):
        """
        Args:
            config: FACENET dict from config.py
            paths:  PATHS dict from config.py
        """
        self.config           = config
        self.model_name       = config["model_name"]
        self.detector_backend = config["detector_backend"]
        self.threshold        = config["similarity_threshold"]
        self.min_confidence   = config["min_face_confidence"]
        self.alert_on_unknown = config["alert_on_unknown"]

        self.known_faces_dir  = paths["known_faces_dir"]
        self.unknown_logs_dir = paths["unknown_logs_dir"]
        self.embeddings_cache = paths["embeddings_cache"]
        self.unknown_cache_path = paths.get(
            "unknown_cache",
            Path("data") / "unknown_cache.pkl"
        )

        # Ensure directories exist
        self.unknown_logs_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Load Haar cascade for pre-filter gate
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(haar_path)

        # Unknown face deduplication cache
        self._unknown_cache = UnknownFaceCache(
            cache_path=paths["unknown_cache"],
            threshold=config["unknown_dedup_threshold"],
            max_entries=config["unknown_cache_max"]
        )

        # Load face embedding database
        self._db = self._load_embeddings()

        # Load unknown face deduplication cache
        # Structure: {unknown_id: {embedding, first_seen, last_seen, count, crop_path}}
        self._unknown_cache = self._load_unknown_cache()

        print(f"[FaceRecognizer] Loaded {len(self._db)} known people.")
        print(f"[FaceRecognizer] Unknown cache: {len(self._unknown_cache)} entries.")
        print(f"[FaceRecognizer] Threshold: {self.threshold} | Model: {self.model_name}")

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ──────────────────────────────────────────────────────────────────────────

    def recognize(self, frame: np.ndarray) -> dict:
        """
        Full recognition pipeline for a single frame.

        Steps:
        1. Haar pre-filter — skip FaceNet if no face-shaped region found
        2. MTCNN face detection — precise bounding boxes + confidence
        3. FaceNet512 embedding — 512-d vector per face
        4. Cosine similarity — match against known DB
        5. Log unknown faces to disk

        Args:
            frame: BGR numpy array from webcam

        Returns:
            {
                'faces': [
                    {
                        'name':       str,
                        'confidence': float,
                        'is_known':   bool,
                        'bbox':       [x1, y1, x2, y2],
                        'log_path':   str or None
                    }
                ],
                'known_count':   int,
                'unknown_count': int
            }
        """
        # ── Step 1: Haar pre-filter (cascaded inference gate) ─────────────────
        # Cheap check before running expensive FaceNet
        # If no face-shaped region exists skip the whole pipeline
        if not self._haar_check(frame):
            return self._empty_result()

        # ── Step 2: MTCNN face detection ──────────────────────────────────────
        try:
            detected = DeepFace.extract_faces(
                frame,
                detector_backend=self.detector_backend,
                enforce_detection=False  # return [] instead of raising if no face
            )
        except Exception as e:
            print(f"[FaceRecognizer] Detection error: {e}")
            return self._empty_result()

        if not detected:
            return self._empty_result()

        face_results = []

        for face_data in detected:
            # Skip low-confidence detections (blurry, partial, side-on faces)
            if face_data.get("confidence", 1.0) < self.min_confidence:
                continue

            # Extract bounding box from DeepFace facial_area format
            region = face_data["facial_area"]
            bbox   = [
                region["x"],
                region["y"],
                region["x"] + region["w"],
                region["y"] + region["h"]
            ]

            # ── Step 3: FaceNet embedding ─────────────────────────────────────
            embedding = self._get_embedding(frame)
            if embedding is None:
                continue

            # ── Step 4: Identity matching ─────────────────────────────────────
            match = self._identify(embedding)

            # ── Step 5: Log unknown faces (with dedup check) ─────────────────
            log_path = None
            if not match["is_known"]:
                log_path = self._log_unknown(
                    frame, bbox, match["confidence"], embedding
                )

            face_results.append({
                "name":       match["name"],
                "confidence": match["confidence"],
                "is_known":   match["is_known"],
                "bbox":       bbox,
                "log_path":   log_path
            })

        known_count   = sum(1 for f in face_results if f["is_known"])
        unknown_count = len(face_results) - known_count

        return {
            "faces":         face_results,
            "known_count":   known_count,
            "unknown_count": unknown_count
        }

    def draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draws face bounding boxes and identity labels onto frame.
        Green = known, Red = unknown.

        Args:
            frame:  BGR numpy array
            result: output dict from recognize()

        Returns:
            BGR numpy array with face overlays drawn
        """
        vis = frame.copy()

        for face in result["faces"]:
            x1, y1, x2, y2 = face["bbox"]
            color = (0, 200, 0) if face["is_known"] else (0, 0, 255)
            label = f"{face['name']} ({face['confidence']:.2f})"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label below the box so it doesn't overlap with detector labels
            cv2.putText(vis, label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return vis

    def reload_embeddings(self):
        """
        Reloads the face embedding database from disk.
        Call this after registering new faces without restarting the app.
        """
        self._db = self._load_embeddings()
        print(f"[FaceRecognizer] Reloaded embeddings: {len(self._db)} people.")

    def register_person(self, name: str, n_shots: int = 4,
                        delay_between: float = 1.5,
                        camera_index: int = 0):
        """
        Registers a new known person by capturing webcam shots.
        Saves images to known_faces/{name}/ and rebuilds embedding cache.

        Can be called at any time while the app is running —
        reload_embeddings() is called automatically after registration
        so the main loop picks up the new face immediately.

        Args:
            name:          Person identifier (folder name + display label)
            n_shots:       Number of images to capture (3-5 recommended)
            delay_between: Seconds between shots — gives time to vary angle
            camera_index:  Camera to use for capture (default = 0)

        Usage:
            recognizer.register_person("avi", n_shots=4)
        """
        import time

        person_dir = self.known_faces_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)

        # Find next image index — avoids overwriting existing shots
        existing  = list(person_dir.glob("*.jpg"))
        start_idx = len(existing)

        print(f"[FaceRecognizer] Registering '{name}' — {n_shots} shots.")
        print("Slightly vary head angle between shots for better coverage.")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera_index} for registration."
            )

        # Warmup — let exposure settle
        for _ in range(5):
            cap.read()

        captured = 0
        for i in range(n_shots):
            print(f"  Shot {i + 1}/{n_shots} in {delay_between}s...")
            time.sleep(delay_between)

            ret, frame = cap.read()
            if not ret:
                print(f"  Shot {i + 1} failed — camera read error.")
                continue

            # Verify a face is actually in the frame before saving
            try:
                faces = DeepFace.extract_faces(
                    frame,
                    detector_backend=self.detector_backend,
                    enforce_detection=True
                )
                if not faces:
                    print(f"  Shot {i + 1} skipped — no face detected.")
                    continue
            except Exception:
                print(f"  Shot {i + 1} skipped — face detection failed.")
                continue

            img_path = person_dir / f"img_{start_idx + i}.jpg"
            cv2.imwrite(str(img_path), frame)
            captured += 1
            print(f"  Saved: {img_path.name}")

        cap.release()

        print(f"[FaceRecognizer] Registered {captured}/{n_shots} valid shots for '{name}'.")

        if captured == 0:
            print("[FaceRecognizer] No valid shots captured — registration failed.")
            return

        # Rebuild embedding cache and reload into memory
        self._build_embeddings()
        self._db = self._load_embeddings()
        print(f"[FaceRecognizer] '{name}' added. Known people: {self.known_people}")

    def promote_unknown(self, unknown_image_path: str, name: str):
        """
        Promotes a logged unknown face into the known faces database.
        Copies the image to known_faces/{name}/ and rebuilds embeddings.

        Args:
            unknown_image_path: path to a file in unknown_logs/
            name: identity to assign to this face
        """
        import shutil

        person_dir = self.known_faces_dir / name
        person_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest      = person_dir / f"promoted_{timestamp}.jpg"
        shutil.copy(unknown_image_path, dest)
        print(f"[FaceRecognizer] Promoted → known_faces/{name}/{dest.name}")

        # Rebuild embedding cache
        self._build_embeddings()
        self._db = self._load_embeddings()
        print(f"[FaceRecognizer] Embeddings rebuilt: {len(self._db)} people.")

    @property
    def known_people(self) -> list:
        """Returns list of registered person names."""
        return list(self._db.keys())

    @property
    def db_size(self) -> int:
        """Returns number of registered people."""
        return len(self._db)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _haar_check(self, frame: np.ndarray) -> bool:
        """
        Runs fast Haar cascade check to see if any face-shaped regions exist.
        Returns True if at least one candidate region found, False otherwise.
        This gates FaceNet — if False, skip embedding entirely.
        """
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates = self._haar.detectMultiScale(
            gray,
            scaleFactor=self.config["haar_scale_factor"],
            minNeighbors=self.config["haar_min_neighbours"],
            minSize=self.config["haar_min_size"]
        )
        return len(candidates) > 0

    def _get_embedding(self, frame: np.ndarray) -> list | None:
        """
        Computes FaceNet512 embedding for the dominant face in the frame.
        Returns 512-d list or None if embedding fails.
        """
        try:
            result = DeepFace.represent(
                img_path=frame,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            if result:
                return result[0]["embedding"]
        except Exception as e:
            print(f"[FaceRecognizer] Embedding error: {e}")
        return None

    def _identify(self, embedding: list) -> dict:
        """
        Compares embedding against all known faces using cosine similarity.
        Returns best match dict with name, confidence, is_known.
        """
        if not self._db:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}

        best_name  = "Unknown"
        best_score = 0.0

        for name, data in self._db.items():
            score = self._cosine_similarity(embedding, data["mean_embedding"])
            if score > best_score:
                best_score = score
                best_name  = name

        is_known = best_score >= self.threshold

        return {
            "name":       best_name if is_known else "Unknown",
            "confidence": round(best_score, 4),
            "is_known":   is_known
        }

    def _cosine_similarity(self, a: list, b: list) -> float:
        """
        Cosine similarity between two embedding vectors.
        Returns float in [0, 1] — higher = more similar faces.
        """
        a = np.array(a)
        b = np.array(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _log_unknown(self, frame: np.ndarray, bbox: list,
                     confidence: float, embedding: list) -> str | None:
        """
        Deduplication-aware unknown face logger.

        Compares the face embedding against the persistent unknown cache.
        - Match found (sim >= threshold): updates last_seen + count, returns None
          (no new crop saved — already have this person on file)
        - No match: saves crop, adds new entry to cache, persists cache to disk

        Args:
            frame:      BGR numpy array
            bbox:       [x1, y1, x2, y2] face region
            confidence: best similarity score against known DB
            embedding:  FaceNet512 embedding of this face

        Returns:
            Path to saved crop if new unknown, None if duplicate
        """
        # ── Check against unknown cache ───────────────────────────────────────
        best_id    = None
        best_score = 0.0

        for uid, entry in self._unknown_cache.items():
            score = self._cosine_similarity(embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_id    = uid

        if best_score >= self.threshold and best_id is not None:
            # Duplicate — update existing cache entry, skip saving new crop
            self._unknown_cache[best_id]["last_seen"] = datetime.now().isoformat()
            self._unknown_cache[best_id]["count"]    += 1
            self._save_unknown_cache()
            return None   # no new crop saved

        # ── New unknown — save crop and add to cache ──────────────────────────
        x1, y1, x2, y2 = bbox
        pad = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename  = f"unknown_{timestamp}_sim{confidence:.2f}.jpg"
        path      = self.unknown_logs_dir / filename
        cv2.imwrite(str(path), crop)

        # Add to cache
        uid = str(uuid.uuid4())[:8]
        self._unknown_cache[uid] = {
            "embedding":  embedding,
            "first_seen": datetime.now().isoformat(),
            "last_seen":  datetime.now().isoformat(),
            "count":      1,
            "crop_path":  str(path)
        }
        self._save_unknown_cache()

        return str(path)

    def _load_unknown_cache(self) -> dict:
        """Loads the unknown face deduplication cache from disk."""
        if not self.unknown_cache_path.exists():
            return {}
        try:
            with open(self.unknown_cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[FaceRecognizer] Could not load unknown cache: {e}")
            return {}

    def _save_unknown_cache(self):
        """Persists the unknown face cache to disk."""
        try:
            with open(self.unknown_cache_path, "wb") as f:
                pickle.dump(self._unknown_cache, f)
        except Exception as e:
            print(f"[FaceRecognizer] Could not save unknown cache: {e}")

    def clear_unknown_cache(self):
        """
        Clears the unknown face cache entirely.
        Call this to reset deduplication history (e.g. new monitoring session).
        """
        self._unknown_cache = {}
        if self.unknown_cache_path.exists():
            self.unknown_cache_path.unlink()
        print("[FaceRecognizer] Unknown cache cleared.")

    @property
    def unknown_cache_size(self) -> int:
        """Returns number of unique unknown faces in the cache."""
        return len(self._unknown_cache)

    def _load_embeddings(self) -> dict:
        """
        Loads face embedding database from pkl cache.
        Returns empty dict if cache doesn't exist yet.
        """
        if not self.embeddings_cache.exists():
            print(
                f"[FaceRecognizer] No embeddings cache found at "
                f"{self.embeddings_cache}. All faces will be Unknown."
            )
            return {}

        with open(self.embeddings_cache, "rb") as f:
            db = pickle.load(f)

        return db

    def _build_embeddings(self):
        """
        Rebuilds the embedding cache from known_faces/ folder.
        Called after registering new faces or promoting unknowns.
        Saves to pkl cache on disk.
        """
        print("[FaceRecognizer] Rebuilding embedding cache...")
        db = {}

        person_dirs = [
            d for d in self.known_faces_dir.iterdir() if d.is_dir()
        ]

        for person_dir in person_dirs:
            name   = person_dir.name
            images = list(person_dir.glob("*.jpg"))

            if not images:
                continue

            print(f"  Processing {name} ({len(images)} images)...")
            embeddings = []

            for img_path in images:
                try:
                    result = DeepFace.represent(
                        img_path=str(img_path),
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=True
                    )
                    embeddings.append(result[0]["embedding"])
                except Exception as e:
                    print(f"    Skipping {img_path.name}: {e}")

            if embeddings:
                db[name] = {
                    "embeddings":     embeddings,
                    "mean_embedding": np.mean(embeddings, axis=0).tolist()
                }
                print(f"    Stored {len(embeddings)} embeddings for {name}.")

        # Save to pkl
        self.embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(self.embeddings_cache, "wb") as f:
            pickle.dump(db, f)

        print(f"[FaceRecognizer] Cache saved: {len(db)} people.")

    def _empty_result(self) -> dict:
        """Returns an empty result dict — used when no faces detected."""
        return {"faces": [], "known_count": 0, "unknown_count": 0}


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run directly to test face recognition on webcam
# python modules/recognizer.py
# Press Q to quit
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from config import CAMERA

    recognizer = FaceRecognizer()
    cap        = cv2.VideoCapture(CAMERA["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA["height"])

    print("Face recognizer test — press Q to quit")
    print(f"Known people: {recognizer.known_people}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = recognizer.recognize(frame)
        vis    = recognizer.draw(frame, result)

        if result["faces"]:
            for face in result["faces"]:
                status = "KNOWN" if face["is_known"] else "UNKNOWN"
                print(f"  {status}: {face['name']} (conf={face['confidence']:.3f})")

        cv2.imshow("Face Recognizer Test", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()