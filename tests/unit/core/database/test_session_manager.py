# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Unit tests for the `mada/core/database/session_manager.py` module.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from mada.core.database.session_manager import ChatSessionManager


@pytest.fixture
def mock_db():
    """Create a mocked chat database with default return values."""
    db = MagicMock()
    db.load_session.return_value = []
    db.list_sessions.return_value = []
    return db


@pytest.fixture
def mock_factory(mock_db):
    """Patch the database factory so it returns the mocked database."""
    with patch("mada.core.database.session_manager.ChatDatabaseFactory") as factory_cls:
        factory = MagicMock()
        factory.create.return_value = mock_db
        factory_cls.return_value = factory
        yield factory


class TestInit:
    """
    Tests for the constructor of `ChatSessionManager`.
    """

    def test_init_uses_provided_session_id(self, mock_factory):
        """Initialize with an explicit session ID and verify it is preserved."""
        config = MagicMock()
        config.session_id = "session-123"
        config.type = "sqlite"

        manager = ChatSessionManager(database_config=config)

        assert manager.current_session_id == "session-123"
        mock_factory.create.assert_called_once_with(config.type, config)

    def test_init_generates_session_id_when_missing(self, mock_factory):
        """Initialize without a session ID and verify one is generated."""
        config = MagicMock()
        config.session_id = None
        config.type = "sqlite"

        with patch(
            "mada.core.database.session_manager.uuid.uuid4",
            return_value="generated-id",
        ):
            manager = ChatSessionManager(database_config=config)

        assert manager.current_session_id == "generated-id"
        mock_factory.create.assert_called_once_with(config.type, config)

    def test_init_uses_default_sqlite_config_when_none_passed(self, mock_factory):
        """Initialize without config and verify SQLiteConfig is used by default."""
        with patch("mada.core.database.session_manager.SQLiteConfig") as sqlite_cls:
            sqlite_config = MagicMock()
            sqlite_config.type = "sqlite"
            sqlite_cls.return_value = sqlite_config

            with patch(
                "mada.core.database.session_manager.uuid.uuid4",
                return_value="generated-id",
            ):
                manager = ChatSessionManager()

        assert manager.current_session_id == "generated-id"
        sqlite_cls.assert_called_once()
        mock_factory.create.assert_called_once_with(sqlite_config.type, sqlite_config)


class TestCreateSessionID:
    """
    Tests for the `create_session_id` method of `ChatSessionManager`.
    """

    def test_create_session_id_returns_uuid_string(self, mock_factory):
        """Verify create_session_id returns the UUID string from uuid4."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="s1", type="sqlite")
        )

        with patch(
            "mada.core.database.session_manager.uuid.uuid4",
            return_value="abc-123",
        ):
            assert manager.create_session_id() == "abc-123"


class TestCreateNewSession:
    """
    Tests for the `create_new_session` method of `ChatSessionManager`.
    """

    def test_create_new_session_uses_provided_session_id(self, mock_factory):
        """Verify create_new_session creates a session with the supplied ID."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )

        manager.create_new_session("new-session")

        manager.chat_db.create_session.assert_called_once_with("new-session")

    def test_create_new_session_defaults_to_current_session(self, mock_factory):
        """Verify create_new_session falls back to the current session ID."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )

        manager.create_new_session()

        manager.chat_db.create_session.assert_called_once_with("current")


class TestSelectSession:
    """
    Tests for the `select_session` method of `ChatSessionManager`.
    """

    def test_select_session_updates_current_and_loads_history(self, mock_factory):
        """Verify select_session switches the active session and loads history."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )
        manager.chat_db.load_session.return_value = [{"role": "user", "content": "hi"}]

        history = manager.select_session("session-2")

        assert manager.current_session_id == "session-2"
        assert history == [{"role": "user", "content": "hi"}]
        manager.chat_db.load_session.assert_called_once_with("session-2")

    def test_select_session_removes_primary_from_context_sessions(self, mock_factory):
        """Verify selecting a primary session prunes it from additional context."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )
        manager.set_context_sessions(["session-2", "current", "session-3"])

        manager.select_session("session-2")

        assert manager.current_session_id == "session-2"
        assert manager.get_context_session_ids() == ["session-3"]


class TestLoadHistory:
    """
    Tests for the `load_history` method of `ChatSessionManager`.
    """

    def test_load_history_returns_empty_list_when_db_returns_none(self, mock_factory):
        """Verify load_history returns an empty list when the database returns None."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )
        manager.chat_db.load_session.return_value = None

        assert manager.load_history() == []

    def test_load_history_returns_db_result(self, mock_factory):
        """Verify load_history returns the database-provided history."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )
        expected = [{"role": "assistant", "content": "hello"}]
        manager.chat_db.load_session.return_value = expected

        assert manager.load_history() == expected

    def test_load_history_accepts_explicit_session_id(self, mock_factory):
        """Verify load_history can read a non-active session without switching state."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )
        expected = [{"role": "assistant", "content": "from other session"}]
        manager.chat_db.load_session.return_value = expected

        assert manager.load_history("other-session") == expected
        assert manager.current_session_id == "current"
        manager.chat_db.load_session.assert_called_once_with("other-session")


class TestContextSessions:
    """
    Tests for the context-session helpers on `ChatSessionManager`.
    """

    def test_set_context_sessions_dedupes_and_skips_primary(self, mock_factory):
        """Verify additional context sessions are normalized and exclude the primary."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="primary", type="sqlite")
        )

        selected = manager.set_context_sessions(["context-a", "primary", "context-a", ""])

        assert selected == ["context-a"]
        assert manager.get_context_session_ids() == ["context-a"]

    def test_get_loaded_session_ids_returns_context_then_primary(self, mock_factory):
        """Verify the full loaded-session list is deterministic."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="primary", type="sqlite")
        )
        manager.set_context_sessions(["context-a", "context-b"])

        assert manager.get_loaded_session_ids() == [
            "context-a",
            "context-b",
            "primary",
        ]

    def test_load_histories_reads_each_requested_session_once(self, mock_factory):
        """Verify load_histories reads multiple sessions without mutating active state."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="primary", type="sqlite")
        )
        manager.chat_db.load_session.side_effect = lambda session_id: [
            {"role": "assistant", "content": session_id}
        ]

        histories = manager.load_histories(["context-a", "context-a", "context-b"])

        assert histories == {
            "context-a": [{"role": "assistant", "content": "context-a"}],
            "context-b": [{"role": "assistant", "content": "context-b"}],
        }
        assert manager.current_session_id == "primary"


class TestAddMessage:
    """
    Tests for the `add_message` method of `ChatSessionManager`.
    """

    def test_add_message_passes_current_session(self, mock_factory):
        """Verify add_message writes to the current session."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )

        manager.add_message("user", "hello")

        manager.chat_db.add_message.assert_called_once_with("current", "user", "hello")


class TestListSessions:
    """
    Tests for the `list_sessions` method of `ChatSessionManager`.
    """

    def test_list_sessions_delegates_to_db(self, mock_factory):
        """Verify list_sessions returns the database session list."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )
        expected = [("s1", datetime.now(timezone.utc))]
        manager.chat_db.list_sessions.return_value = expected

        assert manager.list_sessions() == expected


class TestDeleteSession:
    """
    Tests for the `delete_session` method of `ChatSessionManager`.
    """

    def test_delete_session_uses_given_session_id(self, mock_factory):
        """Verify delete_session deletes the explicitly provided session."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )

        manager.delete_session("to-delete")

        manager.chat_db.delete_session.assert_called_once_with("to-delete")

    def test_delete_session_defaults_to_current_session(self, mock_factory):
        """Verify delete_session falls back to the current session when omitted."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )

        manager.delete_session()

        manager.chat_db.delete_session.assert_called_once_with("current")


class TestDeleteAllSessions:
    """
    Tests for the `delete_all_sessions` method of `ChatSessionManager`.
    """

    def test_delete_all_sessions_flushes_and_clears_current_session(self, mock_factory):
        """Verify delete_all_sessions flushes the database and clears state."""
        manager = ChatSessionManager(
            database_config=MagicMock(session_id="current", type="sqlite")
        )

        manager.delete_all_sessions(confirm=False)

        manager.chat_db.flush_database.assert_called_once_with(confirm=False)
        assert manager.current_session_id is None
