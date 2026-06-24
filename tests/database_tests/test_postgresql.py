# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime
from typing import Tuple

from mada.core.config import PostgreSQLConfig
from mada.core.database import PostgreSQLChatDatabase


@pytest.fixture
def postgresql_db_unit() -> Tuple[PostgreSQLChatDatabase, MagicMock]:
    """Fixture to initialize PostgreSQLChatDatabase with PostgreSQLConfig."""
    db_config = PostgreSQLConfig(
        host="localhost",
        port=5432,
        database="test_db_unit",
        user="test_user",
        password="test_password",
    )

    patcher = patch.object(PostgreSQLChatDatabase, "_connect", return_value=MagicMock())
    mock_connect = patcher.start()

    # Set up the context manager chain
    mock_conn = mock_connect.return_value
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None

    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__enter__.return_value = mock_cursor
    mock_cursor.__exit__.return_value = None

    # Configure the mocked cursor methods
    mock_cursor.execute = MagicMock()
    mock_cursor.fetchone = MagicMock(return_value=None)
    mock_cursor.fetchall = MagicMock(return_value=[])

    yield PostgreSQLChatDatabase(db_config=db_config), mock_cursor

    patcher.stop()


@pytest.fixture
def postgresql_db_integration(
    postgres_connection: PostgreSQLConfig,
) -> PostgreSQLChatDatabase:
    """Fixture to initialize PostgreSQLChatDatabase with PostgreSQLConfig for integration tests."""
    db = PostgreSQLChatDatabase(db_config=postgres_connection)
    db.flush_database(confirm=False)  # Ensure we have a clean database each time
    return db


@pytest.mark.unit
class TestPostgreSQLChatDatabaseUnit:
    class TestInitDB:
        def test_init_db_creates_table(self, postgresql_db_unit):
            """Test that the init_db method creates the sessions table."""
            _, mock_conn = postgresql_db_unit

            mock_conn.execute.assert_called_once_with("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        last_updated TIMESTAMP,
                        messages TEXT
                    )
                """)

    class TestSaveSession:
        @patch("mada.core.database.postgresql.datetime")
        def test_save_session_inserts_data(self, mock_datetime, postgresql_db_unit):
            """Test that save_session inserts data into the database."""
            db, mock_conn = postgresql_db_unit

            session_id = "test_session"
            messages = [{"user": "Hello", "bot": "Hi"}]

            mock_datetime.now.return_value = datetime(2025, 12, 18, 17, 58, 43)
            db.save_session(session_id, messages)

            mock_conn.execute.assert_any_call(
                """
                    INSERT INTO sessions (session_id, last_updated, messages)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE
                    SET last_updated = EXCLUDED.last_updated,
                        messages = EXCLUDED.messages
                """,
                (session_id, mock_datetime.now.return_value, json.dumps(messages)),
            )

        @patch("mada.core.database.postgresql.datetime")
        def test_save_session_handles_empty_messages(
            self, mock_datetime, postgresql_db_unit
        ):
            """Test that save_session handles empty messages gracefully."""
            db, mock_conn = postgresql_db_unit

            session_id = "empty_session"
            messages = []

            mock_datetime.now.return_value = datetime(2025, 12, 18, 17, 58, 43)
            db.save_session(session_id, messages)

            mock_conn.execute.assert_any_call(
                """
                    INSERT INTO sessions (session_id, last_updated, messages)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE
                    SET last_updated = EXCLUDED.last_updated,
                        messages = EXCLUDED.messages
                """,
                (session_id, mock_datetime.now.return_value, json.dumps(messages)),
            )

    class TestLoadSession:
        def test_load_session_returns_messages(self, postgresql_db_unit):
            """Test that load_session retrieves messages from the database."""
            db, mock_conn = postgresql_db_unit
            mock_conn.fetchone.return_value = [
                json.dumps([{"user": "Hello", "bot": "Hi"}])
            ]

            session_id = "test_session"
            loaded_messages = db.load_session(session_id)

            assert loaded_messages == [{"user": "Hello", "bot": "Hi"}]
            mock_conn.execute.assert_any_call(
                "SELECT messages FROM sessions WHERE session_id = %s", (session_id,)
            )

        def test_load_session_returns_empty_list_if_not_found(self, postgresql_db_unit):
            """Test that load_session returns an empty list if the session is not found."""
            db, mock_conn = postgresql_db_unit
            mock_conn.fetchone.return_value = None

            loaded_messages = db.load_session("nonexistent_session")

            assert loaded_messages == []
            mock_conn.execute.assert_any_call(
                "SELECT messages FROM sessions WHERE session_id = %s",
                ("nonexistent_session",),
            )

    class TestListSessions:
        def test_list_sessions_returns_all_sessions(self, postgresql_db_unit):
            """Test that list_sessions returns all sessions in the database."""
            db, mock_conn = postgresql_db_unit
            mock_conn.fetchall.return_value = [
                ("session_1", datetime(2025, 12, 18, 17, 58, 43)),
                ("session_2", datetime(2025, 12, 18, 17, 59, 43)),
            ]

            sessions = db.list_sessions()

            assert len(sessions) == 2
            assert sessions[0][0] == "session_1"
            assert sessions[1][0] == "session_2"
            mock_conn.execute.assert_any_call(
                "SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC"
            )

        def test_list_sessions_returns_empty_list_if_no_sessions(
            self, postgresql_db_unit
        ):
            """Test that list_sessions returns an empty list if there are no sessions."""
            db, mock_conn = postgresql_db_unit
            mock_conn.fetchall.return_value = []

            sessions = db.list_sessions()

            assert sessions == []
            mock_conn.execute.assert_any_call(
                "SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC"
            )

    class TestDeleteSession:
        def test_delete_session_removes_data(self, postgresql_db_unit):
            """Test that delete_session removes a session from the database."""
            db, mock_conn = postgresql_db_unit

            session_id = "test_session"
            db.delete_session(session_id)

            mock_conn.execute.assert_any_call(
                "DELETE FROM sessions WHERE session_id = %s", (session_id,)
            )

    class TestFlushDatabase:
        @patch.object(PostgreSQLChatDatabase, "confirm_db_flush", return_value=True)
        def test_flush_database_removes_all_data(
            self, mock_confirm_flush, postgresql_db_unit
        ):
            """Test that flush_database removes all data from the database."""
            db, mock_conn = postgresql_db_unit

            db.flush_database()

            mock_confirm_flush.assert_called_once()
            mock_conn.execute.assert_any_call("DELETE FROM sessions")

        @patch.object(PostgreSQLChatDatabase, "confirm_db_flush", return_value=False)
        def test_flush_database_cancels_on_user_decline(
            self, mock_confirm_flush, postgresql_db_unit
        ):
            """Test that flush_database does not remove data if user declines."""
            db, mock_conn = postgresql_db_unit

            db.flush_database()

            mock_confirm_flush.assert_called_once()
            mock_conn.execute.assert_called_once()  # Should only be called upon initialization


@pytest.mark.integration
@pytest.mark.allocation_required
class TestPostgreSQLChatDatabaseIntegration:
    class TestInitDB:
        def test_init_db_creates_table(self, postgresql_db_integration):
            """Integration test for init_db method."""
            # Verify table creation
            with postgresql_db_integration._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name = 'sessions'
                """)
                result = cursor.fetchone()

            assert result is not None, "Table 'sessions' was not created."
            print("done with init test")

    class TestSaveSession:
        def test_save_session_inserts_data(self, postgresql_db_integration):
            """Integration test for save_session method."""
            session_id = "test_save_session_inserts_data"
            messages = [{"user": "Hello", "bot": "Hi"}]

            postgresql_db_integration.save_session(session_id, messages)

            # Verify data in the database
            with postgresql_db_integration._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, messages
                    FROM sessions
                    WHERE session_id = %s
                """,
                    (session_id,),
                )
                result = cursor.fetchone()

            assert result is not None, "Session data was not saved."
            assert result[0] == session_id, "Session ID mismatch."
            assert json.loads(result[1]) == messages, "Messages mismatch."

        def test_save_session_handles_empty_messages(self, postgresql_db_integration):
            """Integration test for save_session method with empty messages."""
            session_id = "test_save_session_handles_empty_messages"
            messages = []

            postgresql_db_integration.save_session(session_id, messages)

            # Verify data in the database
            with postgresql_db_integration._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, messages
                    FROM sessions
                    WHERE session_id = %s
                """,
                    (session_id,),
                )
                result = cursor.fetchone()

            assert result is not None, "Session data was not saved."
            assert result[0] == session_id, "Session ID mismatch."
            assert json.loads(result[1]) == messages, "Messages mismatch."

    class TestLoadSession:
        def test_load_session_returns_messages(self, postgresql_db_integration):
            """Integration test for load_session method."""
            session_id = "test_load_session_returns_messages"
            messages = [{"user": "Hello", "bot": "Hi"}]

            # Save session first
            postgresql_db_integration.save_session(session_id, messages)

            # Load session
            loaded_messages = postgresql_db_integration.load_session(session_id)

            assert loaded_messages == messages, (
                "Loaded messages do not match saved messages."
            )

        def test_load_session_returns_empty_list_if_not_found(
            self, postgresql_db_integration
        ):
            """Integration test for load_session method when session is not found."""
            session_id = "nonexistent_session"

            # Load session
            loaded_messages = postgresql_db_integration.load_session(session_id)

            assert loaded_messages == [], "Expected empty list for nonexistent session."

    class TestListSessions:
        def test_list_sessions_returns_all_sessions(self, postgresql_db_integration):
            """Integration test for list_sessions method."""
            session_1 = "test_list_sessions_returns_all_sessions_1"
            session_2 = "test_list_sessions_returns_all_sessions_2"
            messages_1 = [{"user": "Hi", "bot": "Hello"}]
            messages_2 = [{"user": "How are you?", "bot": "I'm fine, thank you."}]

            # Save sessions
            postgresql_db_integration.save_session(session_1, messages_1)
            postgresql_db_integration.save_session(session_2, messages_2)

            # List sessions
            sessions = postgresql_db_integration.list_sessions()

            assert len(sessions) == 2, "Expected two sessions in the database."
            assert sessions[0][0] == session_2, (
                "Expected test_list_sessions_returns_all_sessions_2 to be listed first."
            )
            assert sessions[1][0] == session_1, (
                "Expected test_list_sessions_returns_all_sessions_1 to be listed second."
            )

        def test_list_sessions_returns_empty_list_if_no_sessions(
            self, postgresql_db_integration
        ):
            """Integration test for list_sessions method when no sessions exist."""
            # List sessions
            sessions = postgresql_db_integration.list_sessions()

            assert sessions == [], "Expected empty list when no sessions exist."

    class TestDeleteSession:
        def test_delete_session_removes_data(self, postgresql_db_integration):
            """Integration test for delete_session method."""
            session_id = "test_delete_session_removes_data"
            messages = [{"user": "Hello", "bot": "Hi"}]

            # Save session first
            postgresql_db_integration.save_session(session_id, messages)

            # Delete session
            postgresql_db_integration.delete_session(session_id)

            # Verify session is deleted
            with postgresql_db_integration._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id
                    FROM sessions
                    WHERE session_id = %s
                """,
                    (session_id,),
                )
                result = cursor.fetchone()

            assert result is None, "Session was not deleted."

    class TestFlushDatabase:
        @patch("builtins.input", return_value="y")
        def test_flush_database_removes_all_data(
            self, mock_input, postgresql_db_integration
        ):
            """Integration test for flush_database method."""
            session_1 = "test_flush_database_removes_all_data_1"
            session_2 = "test_flush_database_removes_all_data_2"
            messages_1 = [{"user": "Hi", "bot": "Hello"}]
            messages_2 = [{"user": "How are you?", "bot": "I'm fine, thank you."}]

            # Save sessions
            postgresql_db_integration.save_session(session_1, messages_1)
            postgresql_db_integration.save_session(session_2, messages_2)

            # Flush database
            postgresql_db_integration.flush_database()

            # Verify all sessions are deleted
            sessions = postgresql_db_integration.list_sessions()
            assert sessions == [], "Database was not flushed."

        @patch("builtins.input", return_value="n")
        def test_flush_database_cancels_on_user_decline(
            self, mock_input, postgresql_db_integration
        ):
            """Test that flush_database does not remove data if user declines."""
            session = "test_flush_database_cancels_on_user_decline"
            messages = [{"user": "Hi", "bot": "Hello"}]

            # Save a session to the database
            postgresql_db_integration.save_session(session, messages)

            # Flush database
            postgresql_db_integration.flush_database()

            # Verify nothing happened since the user declined the flush
            sessions = postgresql_db_integration.list_sessions()
            assert len(sessions) == 1, "Expected the session to remain in the database."
            assert sessions[0][0] == session, (
                "Expected the session ID to match the saved session."
            )
