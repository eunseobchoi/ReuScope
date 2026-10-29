"""Resolve externally supplied dataset paths."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


def resolve_cv2024_or_kvasir_path(
    rel_path: str,
    *,
    cv2024_root: Path | None,
    kvasir_data_root: Path | None,
) -> Path:
    """Resolve CV2024 paths and Kvasir labelled-image prefixes."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("Expected a nonblank image_path string")
    if "\x00" in rel_path:
        raise ValueError("image_path must not contain NUL characters")
    rel = rel_path.replace("\\", "/")
    raw_path = Path(rel)
    if raw_path.is_absolute():
        return raw_path
    for prefix in ("kvasir_capsule_split_1/", "kvasir_capsule/labelled_images/", "labelled_images/"):
        if rel.startswith(prefix):
            if kvasir_data_root is None:
                raise ValueError("Set KVASIR_ROOT for Kvasir image paths")
            suffix = rel[len(prefix):]
            parts = suffix.split("/")
            if (not suffix.strip() or Path(suffix).is_absolute() or PureWindowsPath(suffix).drive
                    or ".." in parts or parts[-1] in ("", ".")):
                raise ValueError(f"Malformed Kvasir image path: {rel_path}")
            if prefix == "kvasir_capsule_split_1/":
                if len(parts) < 2 or parts[-2] in ("", "."):
                    raise ValueError(f"Malformed Kvasir split_1 path: {rel_path}")
                return kvasir_data_root / parts[-2] / parts[-1]
            return kvasir_data_root / suffix
    if cv2024_root is None:
        raise ValueError("Set CV2024_ROOT for relative image paths")
    return cv2024_root / rel


def require_directory(name):
    """Require an explicit existing directory in an environment variable."""
    value = os.environ.get(name)
    if not value or not Path(value).is_dir():
        raise ValueError(f"Set {name} to an existing directory")
    return Path(value).resolve()
