# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
TLS helpers shared by MADA HTTP clients.

This module centralizes certificate verification behavior so MCP HTTP tools and
OpenAI-compatible model clients resolve CA bundles the same way.
"""

import logging
import os
import ssl

import truststore


LOG = logging.getLogger(__name__)


def resolve_httpx_verify_value(*, verify: bool = True) -> bool | ssl.SSLContext | str:
    """
    Return the verify value to pass to ``httpx`` clients.

    Resolution order for ``verify=True`` is:

    1. ``SSL_CERT_FILE``
    2. ``REQUESTS_CA_BUNDLE``
    3. System trust store via ``truststore``
    """
    if verify is False:
        return False

    for env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        cert_path = os.getenv(env_var)
        if not cert_path:
            continue
        if os.path.exists(cert_path):
            return cert_path
        LOG.warning(
            "Ignoring %s=%r because the file does not exist.", env_var, cert_path
        )

    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
