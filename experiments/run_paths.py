from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable


_PATH_KEYS: tuple[str, ...] = (
    "variants_root",
    "wall_root",
    "figures_root",
    "demo_artifacts_root",
)


def _append_run_name(path_like: str, run_name: str | None) -> Path:
    path = Path(path_like)
    if run_name:
        return (path / run_name).resolve()
    return path.resolve()


def resolve_path(
    cfg: Dict[str, Any],
    key: str,
    run_name: str | None = None,
) -> Path:
    raw = cfg["paths"][key]
    return _append_run_name(str(raw), run_name)


def resolve_paths(
    cfg: Dict[str, Any],
    run_name: str | None = None,
    keys: Iterable[str] = _PATH_KEYS,
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for key in keys:
        if key in cfg.get("paths", {}):
            out[key] = resolve_path(cfg, key=key, run_name=run_name)
    return out


def run_scoped_file(path_like: str, run_name: str | None = None) -> Path:
    path = Path(path_like)
    if run_name:
        return (path.parent / run_name / path.name).resolve()
    return path.resolve()

