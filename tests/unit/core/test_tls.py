# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ssl

import pytest

from mada.core.tls import resolve_httpx_verify_value


@pytest.mark.unit
def test_resolve_httpx_verify_value_uses_ssl_cert_file(tmp_path, monkeypatch):
    bundle = tmp_path / "ca.pem"
    bundle.write_text("test bundle", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    assert resolve_httpx_verify_value() == str(bundle)


@pytest.mark.unit
def test_resolve_httpx_verify_value_uses_truststore_when_no_bundle(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    assert isinstance(resolve_httpx_verify_value(), ssl.SSLContext)
