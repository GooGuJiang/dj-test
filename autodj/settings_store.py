from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping


class SettingsStore:
    """Atomic JSON settings storage for the desktop application."""

    VERSION = 1

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(
            path
            or os.environ.get("AUTODJ_SETTINGS_PATH", "")
            or (Path.home() / ".config" / "beatthis-muq-auto-dj" / "settings.json")
        ).expanduser()
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        try:
            if not self.path.is_file():
                return {}
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            values = data.get("values", data)
            return dict(values) if isinstance(values, dict) else {}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}

    def save(self, values: Mapping[str, Any]) -> None:
        payload = {
            "version": self.VERSION,
            "values": dict(values),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(self.path)
