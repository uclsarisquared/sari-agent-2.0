"""Strict loader for the distributed-benchmark watcher's TOML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sari_runconfig import TomlConfig, load_toml_tables, reject_unknown_keys


class WatchConfigError(ValueError):
    """The watcher configuration is malformed or contains an unsupported option."""


_SCHEMA: dict[str, dict[str, type[bool]]] = {
    "discord": {
        "enable": bool,
        "collapse_alerts": bool,
    },
}


class WatchConfig(TomlConfig):
    """Validated watcher-only settings."""


def load_watch_config(path: str | Path) -> WatchConfig:
    """Load the watcher configuration and reject settings that belong to another process."""
    config_path, raw = load_toml_tables(path, _SCHEMA, WatchConfigError, "watch")

    values: dict[str, dict[str, Any]] = {}
    for section, section_values in raw.items():
        reject_unknown_keys(config_path, section, section_values, _SCHEMA[section], WatchConfigError)
        values[section] = {}
        for key, value in section_values.items():
            if not isinstance(value, bool):
                raise WatchConfigError(
                    f"{config_path}: {section}.{key} must be a bool, got {type(value).__name__}"
                )
            values[section][key] = value
    return WatchConfig(config_path, values)
