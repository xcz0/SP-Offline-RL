"""Filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
