# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Integration tests for the
`tests/integration/core/database/test_session_manager_integration.py`
module.
"""

import pytest

from mada.core.database.session_manager import ChatSessionManager


class TestSQLiteSessionManagerIntegration:
    """Integration tests for ChatSessionManager backed by SQLite."""

    def test_init_creates_manager_with_sqlite_db(self, sqlite_db):
        """Verify the manager can be initialized with a real SQLite database."""
        manager = ChatSessionManager(database_config=sqlite_db.db_config)

        assert manager.current_session_id is not None
        assert manager.chat_db is not None

    def test_create_and_load_session_history(self, sqlite_db):
        """Verify a session can be created, written to, and loaded back."""
        manager = ChatSessionManager(database_config=sqlite_db.db_config)

        manager.create_new_session("session-1")
        manager.add_message("user", "hello")
        manager.add_message("assistant", "hi")

        history = manager.load_history()

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "hi"

    def test_select_session_switches_history(self, sqlite_db):
        """Verify select_session changes the active session and returns its history."""
        manager = ChatSessionManager(database_config=sqlite_db.db_config)

        manager.create_new_session("session-a")
        manager.add_message("user", "message a")

        manager.select_session("session-b")
        manager.create_new_session("session-b")
        manager.add_message("assistant", "message b")

        history = manager.select_session("session-b")

        assert manager.current_session_id == "session-b"
        assert len(history) == 1
        assert history[0]["content"] == "message b"

    def test_list_sessions_returns_created_sessions(self, sqlite_db):
        """Verify list_sessions returns session records stored in SQLite."""
        manager = ChatSessionManager(database_config=sqlite_db.db_config)

        manager.create_new_session("session-x")
        manager.create_new_session("session-y")

        sessions = manager.list_sessions()

        assert len(sessions) >= 2
        session_ids = [session_id for session_id, _ in sessions]
        assert "session-x" in session_ids
        assert "session-y" in session_ids

    def test_delete_session_removes_session(self, sqlite_db):
        """Verify delete_session removes a session from SQLite."""
        manager = ChatSessionManager(database_config=sqlite_db.db_config)

        manager.create_new_session("session-delete")
        manager.add_message("user", "to be deleted")

        manager.delete_session("session-delete")

        assert manager.chat_db.load_session("session-delete") == []


class TestPostgreSQLSessionManagerIntegration:
    """Integration tests for ChatSessionManager backed by PostgreSQL."""

    @pytest.mark.allocation_required
    def test_init_creates_manager_with_postgres_db(self, postgresql_db):
        """Verify the manager can be initialized with a real PostgreSQL database."""
        manager = ChatSessionManager(database_config=postgresql_db.db_config)

        assert manager.current_session_id is not None
        assert manager.chat_db is not None

    @pytest.mark.allocation_required
    def test_create_and_load_session_history(self, postgresql_db):
        """Verify a session can be created, written to, and loaded back in PostgreSQL."""
        manager = ChatSessionManager(database_config=postgresql_db.db_config)

        manager.create_new_session("session-1")
        manager.add_message("user", "hello")
        manager.add_message("assistant", "hi")

        history = manager.load_history()

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "hi"

    @pytest.mark.allocation_required
    def test_select_session_switches_history(self, postgresql_db):
        """Verify select_session changes the active session and returns its history."""
        manager = ChatSessionManager(database_config=postgresql_db.db_config)

        manager.create_new_session("session-a")
        manager.add_message("user", "message a")

        manager.select_session("session-b")
        manager.create_new_session("session-b")
        manager.add_message("assistant", "message b")

        history = manager.select_session("session-b")

        assert manager.current_session_id == "session-b"
        assert len(history) == 1
        assert history[0]["content"] == "message b"

    @pytest.mark.allocation_required
    def test_list_sessions_returns_created_sessions(self, postgresql_db):
        """Verify list_sessions returns session records stored in PostgreSQL."""
        manager = ChatSessionManager(database_config=postgresql_db.db_config)

        manager.create_new_session("session-x")
        manager.create_new_session("session-y")

        sessions = manager.list_sessions()

        assert len(sessions) >= 2
        session_ids = [session_id for session_id, _ in sessions]
        assert "session-x" in session_ids
        assert "session-y" in session_ids

    @pytest.mark.allocation_required
    def test_delete_session_removes_session(self, postgresql_db):
        """Verify delete_session removes a session from PostgreSQL."""
        manager = ChatSessionManager(database_config=postgresql_db.db_config)

        manager.create_new_session("session-delete")
        manager.add_message("user", "to be deleted")

        manager.delete_session("session-delete")

        assert manager.chat_db.load_session("session-delete") == []
