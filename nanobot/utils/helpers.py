"""Utility functions for nanobot."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path() -> Path:
    """~/.nanobot data directory."""
    return ensure_dir(Path.home() / ".nanobot")


def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve and ensure workspace path. Defaults to ~/.nanobot/workspace."""
    path = Path(workspace).expanduser() if workspace else Path.home() / ".nanobot" / "workspace"
    return ensure_dir(path)


def timestamp() -> str:
    """Current ISO timestamp."""
    return datetime.now().isoformat()


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')

def safe_filename(name: str) -> str:
    """Replace unsafe path characters with underscores."""
    return _UNSAFE_CHARS.sub("_", name).strip()


def normalize_tool_result(result: Any) -> tuple[Any, list[str]]:
    """
    Normalize various tool result formats into (content, media_paths).
    
    Supported formats:
    - str with __MEDIA_PATH__: /path
    - dict with _media_path or _media_paths keys
    - Any other type (treated as content with no media)
    """
    import json
    media_paths = []
    content = result

    if isinstance(result, dict):
        # 1. Extract media keys
        m_paths = result.pop("_media_path", None) or result.pop("_media_paths", [])
        if isinstance(m_paths, str):
            media_paths = [m_paths]
        elif isinstance(m_paths, list):
            media_paths = m_paths
        
        # 2. If result is now empty or just has 'status', normalize content
        if not result or (len(result) == 1 and "status" in result):
            content = "Resource captured."
        else:
            content = result

    elif isinstance(result, str):
        # 3. Support legacy magic strings
        matches = re.findall(r"__MEDIA_PATH__:\s*([^\s]+)", result)
        if matches:
            media_paths = matches
            # Clean up the magic strings from content
            clean_content = re.sub(r"__MEDIA_PATH__:\s*[^\s]+", "", result).strip()
            content = clean_content or "Resource captured."

    return content, media_paths


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """Sync bundled templates to workspace. Only creates missing files."""
    from importlib.resources import files as pkg_files
    try:
        tpl = pkg_files("nanobot") / "templates"
    except Exception:
        return []
    if not tpl.is_dir():
        return []

    added: list[str] = []

    def _write(src, dest: Path):
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8") if src else "", encoding="utf-8")
        added.append(str(dest.relative_to(workspace)))

    for item in tpl.iterdir():
        if item.name.endswith(".md"):
            _write(item, workspace / item.name)
    _write(tpl / "memory" / "MEMORY.md", workspace / "memory" / "MEMORY.md")
    _write(None, workspace / "memory" / "HISTORY.md")
    (workspace / "skills").mkdir(exist_ok=True)

    if added and not silent:
        from rich.console import Console
        for name in added:
            Console().print(f"  [dim]Created {name}[/dim]")
    return added
