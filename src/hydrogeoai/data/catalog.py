"""Versioned dataset catalog (local JSON registry).

Dataset versions are content hashes of the processed observation table, so a version string
identifies exactly the data a model was trained/evaluated on. Lifecycle:
    registered -> checked (QC run) -> approved -> archived
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import data_dir
from ..utils import utcnow

STATUSES = ("registered", "checked", "approved", "archived")


class DatasetCatalog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else data_dir() / "catalog.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = json.loads(self.path.read_text()) if self.path.exists() else {"datasets": {}}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._db, indent=2, default=str))

    @staticmethod
    def key(name: str, version: str) -> str:
        return f"{name}@{version}"

    def register(self, name: str, version: str, files: dict[str, str], metadata: dict) -> str:
        k = self.key(name, version)
        entry = self._db["datasets"].get(k, {})
        entry.update({"name": name, "version": version, "files": files, "metadata": metadata,
                      "status": entry.get("status", "registered"), "registered": entry.get("registered", utcnow()),
                      "updated": utcnow()})
        entry.setdefault("history", []).append({"event": "register", "time": utcnow()})
        self._db["datasets"][k] = entry
        self._save()
        return k

    def set_status(self, key: str, status: str, note: str = "") -> dict:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        e = self._db["datasets"][key]
        if status == "approved" and e["status"] not in ("checked", "approved"):
            raise ValueError("A dataset must pass QC ('checked') before approval")
        e["status"] = status
        e["updated"] = utcnow()
        e.setdefault("history", []).append({"event": status, "note": note, "time": utcnow()})
        self._save()
        return e

    def get(self, key: str) -> dict:
        return self._db["datasets"][key]

    def list(self) -> list[dict]:
        return [{"key": k, **{f: v[f] for f in ("name", "version", "status", "updated")}}
                for k, v in self._db["datasets"].items()]

    def latest(self, name: str, status: str | None = None) -> dict | None:
        c = [v for v in self._db["datasets"].values() if v["name"] == name and (status is None or v["status"] == status)]
        return max(c, key=lambda v: v["updated"]) if c else None
