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
        disabled: When True, `setup_telemetry()` returns without calling
            MSAF's `configure_otel_providers()`. Defaults to False.
    """

    disabled: bool = False


def load_telemetry_config(entry: Optional[Dict[str, Any]]) -> TelemetryConfig:
    """
    Build a `TelemetryConfig` from an optional config-file block.
    """
    if entry is None:
        return TelemetryConfig()
    if not isinstance(entry, dict):
        raise ValueError("'telemetry' must be an object")
    disabled = entry.get("disabled", False)
    if not isinstance(disabled, bool):
        # Reject truthy strings like "false" that would silently disable telemetry.
        raise ValueError(
            f"'telemetry.disabled' must be a bool, got "
            f"{type(disabled).__name__}: {disabled!r}"
        )
    return TelemetryConfig(disabled=disabled)
