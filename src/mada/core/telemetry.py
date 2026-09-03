# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""OpenTelemetry setup.

Wraps Microsoft Agent Framework's built-in observability. Telemetry is
opt-in: install the `telemetry` extra and set `telemetry.enabled: true`
in the config file. See the README for the Aspire Dashboard quickstart.
"""

import logging

LOG = logging.getLogger(__name__)


def setup_telemetry(enabled: bool = False) -> None:
    """
    Configure OpenTelemetry providers when enabled.
    """
    if not enabled:
        return

    try:
        from agent_framework.observability import configure_otel_providers

        # Console exporter dumps every span plus a full metric snapshot every
        # 5s to stdout, so keep it off regardless of ENABLE_CONSOLE_EXPORTERS.
        configure_otel_providers(enable_console_exporters=False)
    except Exception as e:
        LOG.warning("Telemetry setup failed, continuing without it: %s", e)
