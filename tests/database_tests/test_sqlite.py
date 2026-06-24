# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime
from typing import Tuple

from mada.core.config import SQLiteConfig
from mada.core.database import SQLiteChatDatabase


@pytest.fixture
def sqlite_db_unit(tmp_path) -> Tuple[SQLiteChatDatabase, MagicMock]:
    """Fixture to initialize SQLiteChatDatabase with SQLiteConfig."""
    db_config = SQLiteConfig(path=tmp_path / "sqlite_db_unit_test.db")
    db = SQLiteChatDatabase(db_config=db_config)
    mock_conn = MagicMock()
    db._connect = MagicMock(return_value=mock_conn)
    return db, mock_conn


@pytest.fixture
def sqlite_db_integration(tmp_path) -> SQLiteChatDatabase:
    """Fixture to initialize SQLiteChatDatabase with SQLiteConfig for integration tests."""
    db_config = SQLiteConfig(path=tmp_path / "sqlite_db_integration_test.db")
    db = SQLiteChatDatabase(db_config=db_config)
    return db


@pytest.mark.unit
class TestSQLiteChatDatabaseUnit:
    class TestInitDB:
        def test_init_db_creates_table(self, sqlite_db_unit):
            """Test that the init_db method creates the sessions table."""
            db, mock_conn = sqlite_db_unit

            db.init_db()

            mock_conn.__enter__.return_value.execute.assert_called_once_with("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    last_updated TIMESTAMP,
                    messages TEXT
                )
            """)

    class TestSaveSession:
        @patch("mada.core.database.sqlite.datetime")
        def test_save_session_inserts_data(self, mock_datetime, sqlite_db_unit):
            """Test that save_session inserts data into the database."""
            db, mock_conn = sqlite_db_unit

            session_id = "test_session"
            messages = [{"user": "Hello", "bot": "Hi"}]

            mock_datetime.now.return_value = datetime(2025, 12, 18, 17, 58, 43)
            db.save_session(session_id, messages)

            mock_conn.__enter__.return_value.execute.assert_called_once_with(
                """
                INSERT OR REPLACE INTO sessions (session_id, last_updated, messages)
                VALUES (?, ?, ?)
            """,
                (session_id, mock_datetime.now.return_value, json.dumps(messages)),
            )

        @patch("mada.core.database.sqlite.datetime")
        def test_save_session_handles_empty_messages(
            self, mock_datetime, sqlite_db_unit
        ):
            """Test that save_session handles empty messages gracefully."""
            db, mock_conn = sqlite_db_unit

            session_id = "empty_session"
            messages = []

            mock_datetime.now.return_value = datetime(2025, 12, 18, 17, 58, 43)
            db.save_session(session_id, messages)

            mock_conn.__enter__.return_value.execute.assert_called_once_with(
                """
                INSERT OR REPLACE INTO sessions (session_id, last_updated, messages)
                VALUES (?, ?, ?)
            """,
                (session_id, mock_datetime.now.return_value, json.dumps(messages)),
            )

    class TestLoadSession:
        def test_load_session_returns_messages(self, sqlite_db_unit):
            """Test that load_session retrieves messages from the database."""
            db, mock_conn = sqlite_db_unit
            mock_conn.__enter__.return_value.execute.return_value.fetchone.return_value = [
                json.dumps([{"user": "Hello", "bot": "Hi"}])
            ]

            session_id = "test_session"
            loaded_messages = db.load_session(session_id)

            assert loaded_messages == [{"user": "Hello", "bot": "Hi"}]
            mock_conn.__enter__.return_value.execute.assert_called_once_with(
                "SELECT messages FROM sessions WHERE session_id = ?", (session_id,)
            )

        def test_load_session_returns_empty_list_if_not_found(self, sqlite_db_unit):
            """Test that load_session returns an empty list if the session is not found."""
            db, mock_conn = sqlite_db_unit
            mock_conn.__enter__.return_value.execute.return_value.fetchone.return_value = None

            loaded_messages = db.load_session("nonexistent_session")

            assert loaded_messages == []
            mock_conn.__enter__.return_value.execute.assert_called_once_with(
                "SELECT messages FROM sessions WHERE session_id = ?",
                ("nonexistent_session",),
            )

    class TestListSessions:
        def test_list_sessions_returns_all_sessions(self, sqlite_db_unit):
            """Test that list_sessions returns all sessions in the database."""
            db, mock_conn = sqlite_db_unit
            mock_conn.__enter__.return_value.execute.return_value.fetchall.return_value = [
                ("session_1", datetime(2025, 12, 18, 17, 58, 43)),
                ("session_2", datetime(2025, 12, 18, 17, 59, 43)),
            ]

            sessions = db.list_sessions()

            assert len(sessions) == 2
            assert sessions[0][0] == "session_1"
            assert sessions[1][0] == "session_2"
            mock_conn.__enter__.return_value.execute.assert_called_once_with(
                "SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC"
            )

        def test_list_sessions_returns_empty_list_if_no_sessions(self, sqlite_db_unit):
            """Test that list_sessions returns an empty list if there are no sessions."""
            db, mock_conn = sqlite_db_unit
            mock_conn.__enter__.return_value.execute.return_value.fetchall.return_value = []

            sessions = db.list_sessions()

            assert sessions == []
            mock_conn.__enter__.return_value.execute.assert_called_once_with(
                "SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC"
            )

    class TestDeleteSession:
        def test_delete_session_removes_data(self, sqlite_db_unit):
            """Test that delete_session removes a session from the database."""
            db, mock_conn = sqlite_db_unit

            session_id = "test_session"
            db.delete_session(session_id)

            mock_conn.__enter__.return_value.execute.assert_called_once_with(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )

    class TestFlushDatabase:
        @patch.object(SQLiteChatDatabase, "confirm_db_flush", return_value=True)
        def test_flush_database_removes_all_data(
            self, mock_confirm_flush, sqlite_db_unit
        ):
            """Test that flush_database removes all data from the database."""
            db, mock_conn = sqlite_db_unit

            db.flush_database()

            mock_confirm_flush.assert_called_once()
            mock_conn.__enter__.return_value.execute.assert_called_once_with(
                "DELETE FROM sessions"
            )

        @patch.object(SQLiteChatDatabase, "confirm_db_flush", return_value=False)
        def test_flush_database_cancels_on_user_decline(
            self, mock_confirm_flush, sqlite_db_unit
        ):
            """Test that flush_database does not remove data if user declines."""
            db, mock_conn = sqlite_db_unit

            db.flush_database()

            mock_confirm_flush.assert_called_once()
            mock_conn.__enter__.return_value.execute.assert_not_called()


@pytest.mark.integration
class TestSQLiteChatDatabaseIntegration:
    class TestInitDB:
        def test_init_db_creates_table(self, sqlite_db_integration):
            """Integration test for init_db method."""
            # NOTE: don't need to call init_db as that's done automatically in the constructor

            # Connect to the actual SQLite database file
            with sqlite_db_integration._connect() as conn:
                result = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
                ).fetchone()

            assert result is not None, "Table 'sessions' was not created."

    class TestSaveSession:
        def test_save_session_inserts_data(self, sqlite_db_integration):
            """Integration test for save_session method."""
            session_id = "test_save_session_inserts_data"
            messages = [{"user": "Hello", "bot": "Hi"}]

            sqlite_db_integration.save_session(session_id, messages)

            # Verify data in the actual SQLite database file
            with sqlite_db_integration._connect() as conn:
                result = conn.execute(
                    "SELECT session_id, messages FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

            assert result is not None, "Session data was not saved."
            assert result[0] == session_id, "Session ID mismatch."
            assert json.loads(result[1]) == messages, "Messages mismatch."

        def test_save_session_handles_empty_messages(self, sqlite_db_integration):
            """Integration test for save_session method with empty messages."""
            session_id = "test_save_session_handles_empty_messages"
            messages = []

            sqlite_db_integration.save_session(session_id, messages)

            # Verify data in the actual SQLite database file
            with sqlite_db_integration._connect() as conn:
                result = conn.execute(
                    "SELECT session_id, messages FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

            assert result is not None, "Session data was not saved."
            assert result[0] == session_id, "Session ID mismatch."
            assert json.loads(result[1]) == messages, "Messages mismatch."

    class TestLoadSession:
        def test_load_session_returns_messages(self, sqlite_db_integration):
            """Integration test for load_session method."""
            session_id = "test_load_session_returns_messages"
            messages = [{"user": "Hello", "bot": "Hi"}]

            # Save session first
            sqlite_db_integration.save_session(session_id, messages)

            # Load session
            loaded_messages = sqlite_db_integration.load_session(session_id)

            assert loaded_messages == messages, (
                "Loaded messages do not match saved messages."
            )

        def test_load_session_returns_empty_list_if_not_found(
            self, sqlite_db_integration
        ):
            """Integration test for load_session method when session is not found."""
            session_id = "nonexistent_session"

            # Load session
            loaded_messages = sqlite_db_integration.load_session(session_id)

            assert loaded_messages == [], "Expected empty list for nonexistent session."

    class TestListSessions:
        def test_list_sessions_returns_all_sessions(self, sqlite_db_integration):
            """Integration test for list_sessions method."""
            session_1 = "test_list_sessions_returns_all_sessions_1"
            session_2 = "test_list_sessions_returns_all_sessions_2"
            messages_1 = [{"user": "Hi", "bot": "Hello"}]
            messages_2 = [{"user": "How are you?", "bot": "I'm fine, thank you."}]

            # Save sessions
            sqlite_db_integration.save_session(session_1, messages_1)
            sqlite_db_integration.save_session(session_2, messages_2)

            # List sessions
            sessions = sqlite_db_integration.list_sessions()

            assert len(sessions) == 2, "Expected two sessions in the database."
            assert sessions[0][0] == session_2, (
                "Expected test_list_sessions_returns_all_sessions_2 to be listed first."
            )
            assert sessions[1][0] == session_1, (
                "Expected test_list_sessions_returns_all_sessions_1 to be listed second."
            )

        def test_list_sessions_returns_empty_list_if_no_sessions(
            self, sqlite_db_integration
        ):
            """Integration test for list_sessions method when no sessions exist."""
            # List sessions
            sessions = sqlite_db_integration.list_sessions()

            assert sessions == [], "Expected empty list when no sessions exist."

    class TestDeleteSession:
        def test_delete_session_removes_data(self, sqlite_db_integration):
            """Integration test for delete_session method."""
            session_id = "test_delete_session_removes_data"
            messages = [{"user": "Hello", "bot": "Hi"}]

            # Save session first
            sqlite_db_integration.save_session(session_id, messages)

            # Delete session
            sqlite_db_integration.delete_session(session_id)

            # Verify session is deleted
            with sqlite_db_integration._connect() as conn:
                result = conn.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

            assert result is None, "Session was not deleted."

    class TestFlushDatabase:
        @patch("builtins.input", return_value="y")
        def test_flush_database_removes_all_data(
            self, mock_input, sqlite_db_integration
        ):
            """Integration test for flush_database method."""
            session_1 = "test_flush_database_removes_all_data_1"
            session_2 = "test_flush_database_removes_all_data_2"
            messages_1 = [{"user": "Hi", "bot": "Hello"}]
            messages_2 = [{"user": "How are you?", "bot": "I'm fine, thank you."}]

            # Save sessions
            sqlite_db_integration.save_session(session_1, messages_1)
            sqlite_db_integration.save_session(session_2, messages_2)

            # Flush database
            sqlite_db_integration.flush_database()

            # Verify all sessions are deleted
            sessions = sqlite_db_integration.list_sessions()
            assert sessions == [], "Database was not flushed."

        @patch("builtins.input", return_value="n")
        def test_flush_database_cancels_on_user_decline(
            self, mock_input, sqlite_db_integration
        ):
            """Test that flush_database does not remove data if user declines."""
            session = "test_flush_database_cancels_on_user_decline"
            messages = [{"user": "Hi", "bot": "Hello"}]

            # Save a session to the database
            sqlite_db_integration.save_session(session, messages)

            # Flush database
            sqlite_db_integration.flush_database()

            # Verify nothing happened since the user declined the flush
            sessions = sqlite_db_integration.list_sessions()
            assert len(sessions) == 1, "Expected the session to remain in the database."
            assert sessions[0][0] == session, (
                "Expected the session ID to match the saved session."
            )
