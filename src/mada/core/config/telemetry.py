# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Telemetry configuration.

MADA-specific telemetry settings live here. MSAF-provided OpenTelemetry knobs
(OTEL_EXPORTER_OTLP_ENDPOINT, ENABLE_SENSITIVE_DATA, etc.) stay as environment
variables so we don't have to shadow every option Microsoft Agent Framework
adds — this dataclass only covers switches MADA itself owns.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TelemetryConfig:
    """
    Configuration for MADA's telemetry wrapper.

    Attributes:
        enabled: When True, `setup_telemetry()` calls MSAF's
            `configure_otel_providers()`. Defaults to False — telemetry is
            opt-in and also requires the `telemetry` extra to be installed.
    """

    enabled: bool = False


def load_telemetry_config(entry: Optional[Dict[str, Any]]) -> TelemetryConfig:
    """
    Build a `TelemetryConfig` from an optional config-file block.
    """
    if entry is None:
        return TelemetryConfig()
    if not isinstance(entry, dict):
        raise ValueError("'telemetry' must be an object")
    enabled = entry.get("enabled", False)
    if not isinstance(enabled, bool):
        # Reject values like the string "true" or int 1 that would look
        # truthy but aren't the bool we expect.
        raise ValueError(
            f"'telemetry.enabled' must be a bool, got "
            f"{type(enabled).__name__}: {enabled!r}"
        )
    return TelemetryConfig(enabled=enabled)
