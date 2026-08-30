"""Strict loader for the distributed-benchmark watcher's TOML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli


class WatchConfigError(ValueError):
    """The watcher configuration is malformed or contains an unsupported option."""


_SCHEMA: dict[str, dict[str, type[bool]]] = {
    "discord": {
        "enable": bool,
        "collapse_alerts": bool,
    },
}


class WatchConfig:
    """Validated watcher-only settings."""

    def __init__(self, path: Path, values: dict[str, dict[str, Any]]) -> None:
        self.path = path
        self._values = values

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._values.get(section, {}).get(key, default)


def load_watch_config(path: str | Path) -> WatchConfig:
    """Load the watcher configuration and reject settings that belong to another process."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as handle:
            raw = tomli.load(handle)
    except FileNotFoundError as error:
        raise WatchConfigError(f"watch config does not exist: {config_path}") from error
    except (OSError, tomli.TOMLDecodeError) as error:
        raise WatchConfigError(f"could not load watch config {config_path}: {error}") from error

    unknown_sections = sorted(set(raw) - set(_SCHEMA))
    if unknown_sections:
        raise WatchConfigError(
            f"{config_path}: unknown section(s): {', '.join(unknown_sections)}"
        )

    values: dict[str, dict[str, Any]] = {}
    for section, section_values in raw.items():
        if not isinstance(section_values, dict):
            raise WatchConfigError(f"{config_path}: [{section}] must be a TOML table")
        unknown_keys = sorted(set(section_values) - set(_SCHEMA[section]))
        if unknown_keys:
            names = ", ".join(f"{section}.{key}" for key in unknown_keys)
            raise WatchConfigError(f"{config_path}: unknown option(s): {names}")
        values[section] = {}
        for key, value in section_values.items():
            if not isinstance(value, bool):
                raise WatchConfigError(
                    f"{config_path}: {section}.{key} must be a bool, got {type(value).__name__}"
                )
            values[section][key] = value
    return WatchConfig(config_path, values)
