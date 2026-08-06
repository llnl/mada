# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
from pathlib import Path

import pytest

from mada.core.config import (
    A2AConfig,
    DEFAULT_ORCHESTRATION_MODE,
    AppConfig,
    PostgreSQLConfig,
    RemoteA2AAgentConfig,
    SQLiteConfig,
    load_config_from_json,
    load_a2a_agents_config,
    load_a2a_config,
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
class TestA2AConfig:
    def test_load_a2a_config_defaults_when_omitted(self):
        config = load_a2a_config(None)

        assert config.name == "MADA"
        assert config.description == "MADA multi-agent orchestration service"
        assert config.skills == []

    def test_load_a2a_config_accepts_metadata(self):
        config = load_a2a_config(
            {
                "name": "MADA A2A",
                "description": "Agent card description",
                "version": "1.2.3",
                "url": "https://mada.example/a2a",
                "card_path": "/tmp/mada-card.json",
                "skills": [{"id": "workflow", "name": "Workflow"}],
            }
        )

        assert config == A2AConfig(
            name="MADA A2A",
            description="Agent card description",
            version="1.2.3",
            url="https://mada.example/a2a",
            card_path="/tmp/mada-card.json",
            skills=[{"id": "workflow", "name": "Workflow"}],
        )

    @pytest.mark.parametrize("invalid_value", [False, []])
    def test_load_a2a_config_rejects_non_object_blocks(self, invalid_value):
        with pytest.raises(ValueError, match="'a2a.self' must be an object"):
            load_a2a_config(invalid_value)

    def test_load_a2a_config_resolves_relative_card_path(self, tmp_path: Path):
        config = load_a2a_config(
            {"card_path": "agent_cards/mada-card.json"},
            card_path_base=tmp_path,
        )

        assert config.card_path == str(
            (tmp_path / "agent_cards/mada-card.json").resolve()
        )


@pytest.mark.unit
class TestRemoteA2AAgentConfig:
    def test_load_a2a_agents_config_defaults_when_omitted(self):
        assert load_a2a_agents_config(None) == {}

    def test_load_a2a_agents_config_accepts_remote_agents(self):
        config = load_a2a_agents_config(
            {
                "optimizer": {
                    "url": "https://optimizer.example/a2a",
                    "card_url": "https://optimizer.example/.well-known/agent-card.json",
                    "timeout": 30,
                    "api_key": "secret",
                    "headers": {"x-trace": "enabled"},
                }
            }
        )

        assert config == {
            "optimizer": RemoteA2AAgentConfig(
                url="https://optimizer.example/a2a",
                card_url="https://optimizer.example/.well-known/agent-card.json",
                timeout=30,
                api_key="secret",
                headers={"x-trace": "enabled"},
            )
        }

    @pytest.mark.parametrize("invalid_value", [False, []])
    def test_load_a2a_agents_config_rejects_non_object_blocks(self, invalid_value):
        with pytest.raises(ValueError, match="'a2a.agents' must be an object"):
            load_a2a_agents_config(invalid_value)


@pytest.mark.unit
class TestAppA2AConfig:
    def _base_config(self):
        return {
            "model": {
                "provider": "openai",
                "model": "gpt-test",
                "api_key": "test-key",
                "base_url": "https://llm.example/v1",
            },
            "agents": [
                {
                    "agent_name": "WorkerAgent",
                    "description": "Does work",
                    "domain": "test",
                    "mcp_servers": [],
                    "instructions": "Help with tests.",
                }
            ],
        }

    def test_from_dict_accepts_nested_a2a_config(self):
        config_dict = self._base_config()
        config_dict["a2a"] = {
            "agents": {
                "ursa": {
                    "url": "https://ursa.example/a2a",
                    "card_url": "https://ursa.example/.well-known/agent-card.json",
                }
            },
            "self": {
                "name": "MADA A2A",
                "url": "https://mada.example/a2a",
            },
        }

        config = AppConfig.from_dict(config_dict)

        assert config.a2a.name == "MADA A2A"
        assert config.a2a.url == "https://mada.example/a2a"
        assert config.a2a_agents == {
            "ursa": RemoteA2AAgentConfig(
                url="https://ursa.example/a2a",
                card_url="https://ursa.example/.well-known/agent-card.json",
            )
        }

    def test_load_config_from_json_resolves_nested_a2a_self_card_path(
        self, tmp_path: Path
    ):
        config_dict = self._base_config()
        config_dict["a2a"] = {
            "self": {
                "card_path": "agent_cards/mada-card.json",
            }
        }
        config_path = tmp_path / "mada.json"
        config_path.write_text(json.dumps(config_dict), encoding="utf-8")

        config = load_config_from_json(str(config_path))

        assert config.a2a.card_path == str(
            (tmp_path / "agent_cards/mada-card.json").resolve()
        )

    def test_from_dict_rejects_non_object_nested_a2a_config(self):
        config_dict = self._base_config()
        config_dict["a2a"] = []

        with pytest.raises(ValueError, match="'a2a' must be an object"):
            AppConfig.from_dict(config_dict)

    def test_from_dict_rejects_legacy_top_level_a2a_keys(self):
        config_dict = self._base_config()
        config_dict["a2a_self"] = {
            "name": "Legacy",
            "url": "https://legacy.example/a2a",
        }
        config_dict["a2a_agents"] = {
            "legacy": {
                "url": "https://legacy-agent.example/a2a",
            }
        }

        with pytest.raises(
            ValueError,
            match="Use 'a2a.self' and 'a2a.agents' for A2A configuration",
        ):
            AppConfig.from_dict(config_dict)


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
