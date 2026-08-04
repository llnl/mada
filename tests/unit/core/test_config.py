# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from mada.core.config import (
    DEFAULT_ORCHESTRATION_MODE,
    AppConfig,
    PostgreSQLConfig,
    SQLiteConfig,
    load_orchestration_config,
)


@pytest.mark.unit
class TestSQLiteConfig:
    def test_default_path(self):
        """Test that the default SQLite path exists and is expanded correctly."""
        config = SQLiteConfig()
        assert config.path.parent.exists(), "'~/.mada/' directory should exist."
        assert "~" not in str(config.path)
        assert str(config.path) == str(config.path.expanduser()), (
            "SQLite path should be expanded correctly."
        )


@pytest.mark.unit
class TestPostgreSQLConfig:
    def test_connection_string_initialization(self):
        """Test that PostgreSQLConfig initializes correctly with a connection string."""
        connection_string = "postgresql://user:password@localhost:5432/testdb"
        config = PostgreSQLConfig(connection_string=connection_string)
        assert config.get_connection_string() == connection_string, (
            "PostgreSQL connection string should match the initialized value."
        )

    def test_individual_fields_initialization(self):
        """Test that PostgreSQLConfig constructs a connection string from individual fields."""
        config = PostgreSQLConfig(
            host="localhost",
            port=5432,
            database="testdb",
            user="user",
            password="password",
        )
        expected_connection_string = "postgresql://user:password@localhost:5432/testdb"
        assert config.get_connection_string() == expected_connection_string, (
            "PostgreSQL connection string should be constructed from individual fields."
        )

    def test_missing_fields_validation(self):
        """Test that PostgreSQLConfig raises a ValueError when required fields are missing."""
        with pytest.raises(ValueError):
            PostgreSQLConfig(
                host="localhost",
                port=5432,
                database="testdb",
                user="user",
                # Missing password
            )

    def test_env_var_expansion(self, monkeypatch):
        """Test that PostgreSQLConfig expands environment variables correctly."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_USER", "user")
        monkeypatch.setenv("DB_PASSWORD", "password")
        monkeypatch.setenv("DB_DATABASE", "testdb")

        config = PostgreSQLConfig(
            host="${DB_HOST}",
            port=5432,
            database="${DB_DATABASE}",
            user="${DB_USER}",
            password="${DB_PASSWORD}",
        )
        expected_connection_string = "postgresql://user:password@localhost:5432/testdb"
        assert config.get_connection_string() == expected_connection_string, (
            "PostgreSQL connection string should expand environment variables correctly."
        )


@pytest.mark.unit
class TestOrchestrationConfig:
    def test_load_orchestration_config_defaults_when_omitted(self):
        config = load_orchestration_config(None)

        assert config.mode == DEFAULT_ORCHESTRATION_MODE
        assert config.participants is None

    def test_load_orchestration_config_defaults_for_empty_object(self):
        config = load_orchestration_config({})

        assert config.mode == DEFAULT_ORCHESTRATION_MODE
        assert config.participants is None

    @pytest.mark.parametrize("invalid_value", [False, []])
    def test_load_orchestration_config_rejects_non_object_blocks(self, invalid_value):
        with pytest.raises(ValueError, match="'orchestration' must be an object"):
            load_orchestration_config(invalid_value)


@pytest.mark.unit
class TestAppConfig:
    def test_app_config_loads_verify_settings(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MADA_CA_BUNDLE", "/tmp/mada-ca.pem")

        config = AppConfig.from_dict(
            {
                "model": {
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                    "api_key": "sk-test",
                    "base_url": "https://example.invalid/v1",
                    "verify": False,
                },
                "agents": [
                    {
                        "agent_name": "TestAgent",
                        "description": "Test agent",
                        "instructions": "You are a test agent.",
                        "mcp_servers": ["test_server"],
                    }
                ],
                "database": {"type": "sqlite", "path": str(tmp_path / "mada.db")},
                "mcp_servers": {
                    "test_server": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.invalid/mcp",
                        "verify": "${MADA_CA_BUNDLE}",
                    }
                },
            }
        )

        assert config.model.verify is False
        assert config.mcp_servers["test_server"].verify == "/tmp/mada-ca.pem"
