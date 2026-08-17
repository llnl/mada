# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""OpenTelemetry setup.

Wraps Microsoft Agent Framework's built-in observability. Setting
OTEL_EXPORTER_OTLP_ENDPOINT turns on export; see the README for the
Aspire Dashboard quickstart. MADA_DISABLE_TELEMETRY=true skips setup.
"""

import logging
import os

LOG = logging.getLogger(__name__)


def setup_telemetry() -> None:
    if os.environ.get("MADA_DISABLE_TELEMETRY", "").lower() in ("1", "true", "yes"):
        return

    try:
        from agent_framework.observability import configure_otel_providers

        # Console exporter dumps every span plus a full metric snapshot every
        # 5s to stdout, so keep it off regardless of ENABLE_CONSOLE_EXPORTERS.
        configure_otel_providers(enable_console_exporters=False)
    except Exception as e:
        LOG.warning("Telemetry setup failed, continuing without it: %s", e)
