from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "YAML config requires PyYAML. Install with `pip install pyyaml` "
            "or reinstall QueryOS from the updated requirements."
        ) from exc

    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"config file not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config YAML root must be a mapping/object")
    return loaded


def cfg_get(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def pick(cli_value: Any, config: Dict[str, Any], path: str, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    return cfg_get(config, path, default)


def pick_bool(cli_value: bool, config: Dict[str, Any], path: str, default: bool = False) -> bool:
    if cli_value:
        return True
    value = cfg_get(config, path, default)
    return bool(value)
