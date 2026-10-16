"""Profiles: named sets of settings stored in ~/.prompt_llm/profiles.json."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

HOME = Path(os.environ.get("PROMPT_LLM_HOME") or Path.home() / ".prompt_llm")
FIELDS = ("provider", "model", "api_key", "base_url", "system", "temperature")

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ProfileError(Exception):
    """Raised for missing or invalid profiles."""


def _path() -> Path:
    return HOME / "profiles.json"


def _migrate_legacy() -> dict:
    """Read 0.1.x files (config.json plus one <name>.json per profile) as profiles."""
    data = {"default": None, "profiles": {}}
    if not HOME.is_dir():
        return data
    files = sorted(HOME.glob("*.json"), key=lambda f: f.stem != "config")  # config.json first
    for file in files:
        name = "default" if file.stem == "config" else file.stem
        if not _NAME.match(name) or name in data["profiles"]:
            continue
        try:
            values = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(values, dict):
            data["profiles"][name] = {k: v for k, v in values.items() if k in FIELDS}
    if "default" in data["profiles"]:
        data["default"] = "default"
    return data


def _load() -> dict:
    path = _path()
    if not path.exists():
        return _migrate_legacy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ProfileError(f"Could not read {path}: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("profiles", {}), dict):
        raise ProfileError(f"{path} is malformed")
    data.setdefault("profiles", {})
    data.setdefault("default", None)
    return data


def _save(data: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    path = _path()
    tmp = path.with_name(path.name + ".tmp")
    # Profiles may hold API keys, so keep the file readable by the owner only.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _require(data: dict, name: str) -> dict:
    if name not in data["profiles"]:
        raise ProfileError(f"Profile {name!r} does not exist. See `prompt_llm profile list`.")
    return data["profiles"][name]


def get(name: Optional[str] = None) -> Tuple[Optional[str], dict]:
    """Return (name, settings) of the given profile, or of the default one."""
    data = _load()
    name = name or data["default"]
    if not name:
        return None, {}
    return name, dict(_require(data, name))


def list_all() -> Tuple[Optional[str], Dict[str, dict]]:
    data = _load()
    return data["default"], data["profiles"]


def save(name: str, values: dict, unset=(), make_default: bool = False) -> dict:
    """Create or update a profile. The first profile saved becomes the default."""
    if not _NAME.match(name):
        raise ProfileError(f"Invalid profile name {name!r}: use letters, digits, '.', '_' or '-'")
    data = _load()
    profile = data["profiles"].setdefault(name, {})
    profile.update(values)
    for key in unset:
        profile.pop(key, None)
    if make_default or not data["default"]:
        data["default"] = name
    _save(data)
    return profile


def set_default(name: str) -> None:
    data = _load()
    _require(data, name)
    data["default"] = name
    _save(data)


def remove(name: str) -> None:
    data = _load()
    _require(data, name)
    del data["profiles"][name]
    if data["default"] == name:
        data["default"] = None
    _save(data)
