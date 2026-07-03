"""
modules/alerter.py — Alert Generation Module
=============================================
Two alert modes controlled by config.LLM['use_llm_alerts']:

  False (default) — Rule-based placeholder alerts
      Fast, no LLM needed, deterministic output.
      Rules defined in config.ALERTS['rules'].

  True — LLM-generated alerts via Ollama (Llama 3.2 3B)
      Context-aware, uses RAG context from SecurityLogger.
      Requires Ollama server running: ollama serve

Both modes return identical output schema for drop-in compatibility.

Usage:
    from modules.alerter import Alerter
    from modules.logger import SecurityLogger
    from config import LLM, ALERTS

    logger  = SecurityLogger()
    alerter = Alerter(LLM, ALERTS, logger)

    alert = alerter.generate(yolo_output, facenet_output)

    # alert = {
    #     'alert_text': str,    # generated alert message
    #     'severity':   str,    # LOW / MEDIUM / HIGH / CRITICAL / NONE
    #     'timestamp':  str,    # ISO format
    #     'source':     str,    # 'placeholder' or 'llm'
    #     'suppressed': bool    # True if cooldown prevented alert
    # }
"""

import ollama
from datetime import datetime
from config import LLM, ALERTS


# ── System prompt for LLM alert generation ────────────────────────────────────
# Instructs the LLM to act as a security analyst and format alerts consistently
ALERT_SYSTEM_PROMPT = """You are a security monitoring AI for an offline CCTV system.
You receive real-time detection data from computer vision models that detect
weapons (danger tiers: risky, dangerous) and identify faces (known/unknown).

Generate concise, factual security alerts. Do not speculate or be alarmist.
Prioritise unknown faces combined with dangerous objects as the highest threat.

Always respond in this exact format:
SEVERITY: [LOW / MEDIUM / HIGH / CRITICAL]
SUMMARY: One sentence describing what was detected.
ACTION: One sentence recommending what to do.
CONTEXT: One sentence noting any relevant patterns.

Keep the entire response under 100 words."""

# ── System prompt for LLM query interface ────────────────────────────────────
# Used by the UI query tab for multi-turn incident analysis
QUERY_SYSTEM_PROMPT = """You are a security incident analyst for an offline AI CCTV system.
You have access to detection logs that record:
  - Weapon detections (danger tiers: risky=1, dangerous=2)
  - Face recognition events (known/unknown faces)
  - Timestamps and severity levels for each alert

Answer questions about incidents factually and concisely.
Structure your responses clearly when asked for summaries.
If no relevant log data exists for a query, say so directly.
Do not invent incidents that are not in the provided logs."""


class Alerter:
    """
    Alert generation for detection events.

    Supports two modes:
    - Placeholder: fast rule-based alerts, no LLM needed
    - LLM: context-aware alerts via Ollama, requires server running

    Also handles the multi-turn LLM query interface used by the UI query tab.
    """

    def __init__(self, llm_config: dict = LLM,
                 alert_config: dict = ALERTS,
                 logger=None):
        """
        Args:
            llm_config:    LLM dict from config.py
            alert_config:  ALERTS dict from config.py
            logger:        SecurityLogger instance for RAG context retrieval
                           Required for LLM mode, optional for placeholder mode
        """
        self.llm_config    = llm_config
        self.alert_config  = alert_config
        self.logger        = logger
        self.use_llm       = llm_config["use_llm_alerts"]
        self.model         = llm_config["model"]
        self.temperature   = llm_config["temperature"]
        self.max_tokens    = llm_config["max_tokens"]
        self.cooldown_s    = llm_config["alert_cooldown_s"]

        # Cooldown tracker — prevents alert spam for repeated same-type events
        # Key: alert type string, Value: datetime of last alert
        self._last_alert_times = {}

        # Multi-turn conversation history for the query interface
        # Reset between sessions via reset_conversation()
        self._conversation = []

        mode = "LLM" if self.use_llm else "placeholder"
        print(f"[Alerter] Mode: {mode} | Model: {self.model}")

        # Verify Ollama is reachable if LLM mode is active
        if self.use_llm:
            self._check_ollama()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ──────────────────────────────────────────────────────────────────────────

    def generate(self, yolo_output: dict, facenet_output: dict) -> dict:
        """
        Generates an alert for a detection event.
        Routes to placeholder or LLM based on config.LLM['use_llm_alerts'].

        Args:
            yolo_output:    output dict from DangerDetector.detect()
            facenet_output: output dict from FaceRecognizer.recognize()

        Returns:
            {
                'alert_text': str,
                'severity':   str,   # LOW / MEDIUM / HIGH / CRITICAL / NONE
                'timestamp':  str,
                'source':     str,   # 'placeholder' or 'llm'
                'suppressed': bool
            }
        """
        # ── Cooldown check ────────────────────────────────────────────────────
        # Don't fire same alert type more than once per cooldown period
        alert_key = self._alert_key(yolo_output, facenet_output)
        if self._is_cooling_down(alert_key):
            return self._suppressed_alert()

        # ── Route to correct mode ─────────────────────────────────────────────
        if self.use_llm:
            alert = self._llm_alert(yolo_output, facenet_output)
        else:
            alert = self._placeholder_alert(yolo_output, facenet_output)

        # Update cooldown tracker if alert was meaningful
        if not alert["suppressed"] and alert["severity"] != "NONE":
            self._last_alert_times[alert_key] = datetime.now()

        return alert

    def query(self, user_message: str) -> str:
        """
        Multi-turn LLM query over incident history.
        Maintains conversation context between calls.
        Injects recent log entries as RAG context with each message.

        Args:
            user_message: natural language question about incidents

        Returns:
            LLM response string (streamed token by token internally,
            returned as complete string)
        """
        if not self._conversation:
            # First message — initialise with system prompt
            self._conversation = [
                {"role": "system", "content": QUERY_SYSTEM_PROMPT}
            ]

        # Build context-injected user message
        # Injects log context with every turn for up-to-date RAG
        log_context = (
            self.logger.get_log_context()
            if self.logger else "No logger connected."
        )
        known_context = (
            self.logger.get_known_faces_context()
            if self.logger else ""
        )

        context_message = (
            f"Current security system context:\n"
            f"{known_context}\n\n"
            f"{log_context}\n\n"
            f"User question: {user_message}"
        )

        self._conversation.append({
            "role":    "user",
            "content": context_message
        })

        try:
            response = ollama.chat(
                model=self.model,
                messages=self._conversation,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.llm_config["query_max_tokens"]
                }
            )
            answer = response["message"]["content"].strip()

        except Exception as e:
            answer = (
                f"LLM query failed: {e}\n"
                f"Make sure Ollama is running: ollama serve"
            )

        # Add assistant response to history for multi-turn context
        self._conversation.append({
            "role":    "assistant",
            "content": answer
        })

        return answer

    def query_stream(self, user_message: str):
        """
        Streaming version of query() — yields tokens as they're generated.
        Used by the UI query tab for real-time token display.

        Args:
            user_message: natural language question

        Yields:
            str: individual tokens as generated by the LLM
        """
        if not self._conversation:
            self._conversation = [
                {"role": "system", "content": QUERY_SYSTEM_PROMPT}
            ]

        log_context = (
            self.logger.get_log_context()
            if self.logger else "No logger connected."
        )
        known_context = (
            self.logger.get_known_faces_context()
            if self.logger else ""
        )

        context_message = (
            f"Current security system context:\n"
            f"{known_context}\n\n"
            f"{log_context}\n\n"
            f"User question: {user_message}"
        )

        self._conversation.append({
            "role": "user", "content": context_message
        })

        full_response = ""

        try:
            stream = ollama.chat(
                model=self.model,
                messages=self._conversation,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.llm_config["query_max_tokens"]
                }
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                full_response += token
                yield token

        except Exception as e:
            error_msg = (
                f"LLM error: {e}. "
                f"Make sure Ollama is running: ollama serve"
            )
            yield error_msg
            full_response = error_msg

        # Save complete response to conversation history
        self._conversation.append({
            "role": "assistant", "content": full_response
        })

    def reset_conversation(self):
        """Resets the multi-turn conversation history for a fresh session."""
        self._conversation = []
        print("[Alerter] Conversation history cleared.")

    def set_mode(self, use_llm: bool):
        """
        Switches between LLM and placeholder alert modes at runtime.
        Useful for toggling from the UI without restarting the app.

        Args:
            use_llm: True = LLM alerts, False = placeholder alerts
        """
        if use_llm and not self.use_llm:
            self._check_ollama()
        self.use_llm = use_llm
        mode = "LLM" if use_llm else "placeholder"
        print(f"[Alerter] Switched to {mode} mode.")

    # ──────────────────────────────────────────────────────────────────────────
    # PLACEHOLDER ALERT (RULE-BASED)
    # ──────────────────────────────────────────────────────────────────────────

    def _placeholder_alert(self, yolo_output: dict,
                           facenet_output: dict) -> dict:
        """
        Generates a rule-based alert without calling the LLM.
        Rules are defined in config.ALERTS['rules'] and evaluated in order.
        First matching rule wins.

        Fast, deterministic, no dependencies beyond config.
        """
        alert_level   = yolo_output["alert_level"]
        unknown_count = facenet_output["unknown_count"]
        known_names   = [
            f["name"] for f in facenet_output["faces"] if f["is_known"]
        ]
        object_names  = [o["class"] for o in yolo_output["objects"]]

        # Evaluate rules in priority order
        for rule in self.alert_config["rules"]:
            cond = rule["condition"]
            if (alert_level   >= cond["min_tier"] and
                    unknown_count >= cond["min_unknowns"]):

                # Enrich summary with detected object names if available
                summary = rule["summary"]
                if object_names and "{objects}" in summary:
                    summary = summary.replace(
                        "{objects}", ", ".join(object_names)
                    )

                alert_text = (
                    f"SEVERITY: {rule['severity']}\n"
                    f"SUMMARY: {summary}\n"
                    f"ACTION: {rule['action']}"
                )

                return self._build_alert(
                    alert_text, rule["severity"], "placeholder"
                )

        # No rule matched — known faces or empty frame
        if known_names:
            return self._build_alert(
                f"SEVERITY: NONE\n"
                f"SUMMARY: Known person detected: {', '.join(known_names)}.\n"
                f"ACTION: No action required.",
                "NONE", "placeholder"
            )

        return self._build_alert(
            "SEVERITY: NONE\nSUMMARY: Motion detected, no threats.\nACTION: No action required.",
            "NONE", "placeholder"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # LLM ALERT
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_alert(self, yolo_output: dict, facenet_output: dict) -> dict:
        """
        Generates a context-aware alert using Llama 3.2 3B via Ollama.
        Falls back to placeholder if Ollama is unavailable.

        Injects:
        - Recent detection history (from logger ring buffer)
        - Known faces registry (from logger ChromaDB)
        """
        # Skip LLM for trivial events — no objects and no unknowns
        if yolo_output["alert_level"] == 0 and facenet_output["unknown_count"] == 0:
            return self._placeholder_alert(yolo_output, facenet_output)

        # Build structured event description
        objects_desc = ", ".join(
            f"{o['class']} (conf={o['confidence']:.2f})"
            for o in yolo_output["objects"]
        ) or "none"

        faces_desc = ", ".join(
            f"{'UNKNOWN' if not f['is_known'] else f['name']} "
            f"(similarity={f['confidence']:.2f})"
            for f in facenet_output["faces"]
        ) or "none"

        # Get RAG context from logger
        recent_context = (
            self.logger.get_recent_context(n=5)
            if self.logger else "No context available."
        )
        known_context = (
            self.logger.get_known_faces_context()
            if self.logger else ""
        )

        user_prompt = (
            f"CURRENT DETECTION:\n"
            f"  Objects detected: {objects_desc}\n"
            f"  Faces detected: {faces_desc}\n"
            f"  Overall alert tier: {yolo_output['alert_level']}\n\n"
            f"REGISTERED KNOWN PEOPLE:\n  {known_context}\n\n"
            f"RECENT HISTORY:\n{recent_context}\n\n"
            f"Generate a security alert for this event."
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": ALERT_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            alert_text = response["message"]["content"].strip()
            severity   = self._extract_severity(alert_text)
            return self._build_alert(alert_text, severity, "llm")

        except Exception as e:
            print(f"[Alerter] LLM unavailable ({e}), falling back to placeholder.")
            return self._placeholder_alert(yolo_output, facenet_output)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _check_ollama(self):
        """Verifies Ollama server is reachable. Warns but does not raise."""
        try:
            ollama.list()
            print(f"[Alerter] Ollama server reachable. Model: {self.model}")
        except Exception as e:
            print(
                f"[Alerter] WARNING: Ollama not reachable ({e}). "
                f"Start with: ollama serve\n"
                f"Falling back to placeholder alerts until server is available."
            )

    def _alert_key(self, yolo_output: dict, facenet_output: dict) -> str:
        """
        Builds a string key identifying the alert type.
        Used for cooldown tracking — same key = same alert type.
        """
        tier     = yolo_output["alert_level"]
        unknowns = min(facenet_output["unknown_count"], 1)  # cap at 1
        return f"tier{tier}_unknown{unknowns}"

    def _is_cooling_down(self, alert_key: str) -> bool:
        """Returns True if this alert type is still in cooldown."""
        last = self._last_alert_times.get(alert_key)
        if last is None:
            return False
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed < self.cooldown_s

    def _extract_severity(self, alert_text: str) -> str:
        """
        Extracts severity level from LLM-generated alert text.
        Looks for SEVERITY: keyword, falls back to MEDIUM if not found.
        """
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]:
            if level in alert_text.upper():
                return level
        return "MEDIUM"

    def _build_alert(self, alert_text: str, severity: str,
                     source: str) -> dict:
        """Builds the standard alert output dict."""
        return {
            "alert_text": alert_text,
            "severity":   severity,
            "timestamp":  datetime.now().isoformat(),
            "source":     source,
            "suppressed": False
        }

    def _suppressed_alert(self) -> dict:
        """Returns a suppressed alert dict for cooldown events."""
        return {
            "alert_text": "",
            "severity":   "suppressed",
            "timestamp":  datetime.now().isoformat(),
            "source":     "cooldown",
            "suppressed": True
        }


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run directly to test alerter
# python modules/alerter.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from config import LLM, ALERTS, PATHS, init_dirs
    from modules.logger import SecurityLogger

    init_dirs()
    logger  = SecurityLogger(PATHS, LLM)
    alerter = Alerter(LLM, ALERTS, logger)

    # Test placeholder alerts
    print("\n=== PLACEHOLDER ALERT TESTS ===\n")

    scenarios = [
        (
            {"objects": [{"class": "dangerous", "danger_tier": 2, "confidence": 0.91, "bbox": []}], "alert_level": 2},
            {"faces": [{"name": "Unknown", "confidence": 0.41, "is_known": False, "bbox": [], "log_path": None}], "known_count": 0, "unknown_count": 1},
            "Unknown + dangerous object"
        ),
        (
            {"objects": [{"class": "risky", "danger_tier": 1, "confidence": 0.78, "bbox": []}], "alert_level": 1},
            {"faces": [{"name": "avi", "confidence": 0.89, "is_known": True, "bbox": [], "log_path": None}], "known_count": 1, "unknown_count": 0},
            "Known face + risky object"
        ),
        (
            {"objects": [], "alert_level": 0},
            {"faces": [], "known_count": 0, "unknown_count": 0},
            "No threats"
        ),
    ]

    for yolo, facenet, label in scenarios:
        alert = alerter.generate(yolo, facenet)
        print(f"Scenario: {label}")
        print(f"Severity: {alert['severity']} | Source: {alert['source']}")
        print(f"{alert['alert_text']}\n")

    # Test LLM query
    print("\n=== LLM QUERY TEST ===\n")
    print("Testing query interface (requires Ollama running)...")
    response = alerter.query("Summarise any recent security incidents.")
    print(f"Response:\n{response}")