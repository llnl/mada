"""
Unit tests for the `mada/core/database/postgresql.py` module.
"""

from datetime import datetime
from typing import Tuple
from unittest.mock import MagicMock, patch

import pytest

from mada.core.config import PostgreSQLConfig
from mada.core.database import PostgreSQLChatDatabase


@pytest.fixture
def postgresql_db() -> Tuple[PostgreSQLChatDatabase, MagicMock]:
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

    yield PostgreSQLChatDatabase(db_config=db_config), mock_cursor, mock_conn

    patcher.stop()


class TestInitDB:
    def test_init_db_creates_table(self, postgresql_db):
        """Test that the init_db method creates the sessions table."""
        _, mock_cursor, _ = postgresql_db

        mock_cursor.execute.assert_any_call("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        last_updated TIMESTAMP
                    )
                """)

        mock_cursor.execute.assert_any_call("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id SERIAL PRIMARY KEY,
                        session_id TEXT REFERENCES sessions(session_id),
                        role TEXT,
                        content TEXT,
                        timestamp TIMESTAMP
                    )
                """)


class TestCreateSession:
    def test_create_session_inserts_session(self, postgresql_db):
        db, mock_cursor, _ = postgresql_db
        session_id = "test_session"

        mock_cursor.execute.reset_mock()

        db.create_session(session_id)

        mock_cursor.execute.assert_called_once()
        sql, params = mock_cursor.execute.call_args[0]

        assert "INSERT INTO sessions" in sql
        assert "ON CONFLICT (session_id) DO NOTHING" in sql
        assert params[0] == session_id
        assert isinstance(params[1], datetime)


class TestAddMessage:
    def test_add_message_with_explicit_timestamp(self, postgresql_db):
        db, mock_cursor, mock_conn = postgresql_db
        session_id = "test_session"
        role = "user"
        content = "Hello"
        timestamp = datetime(2025, 1, 1, 12, 0, 0)

        mock_cursor.execute.reset_mock()
        mock_conn.commit.reset_mock()

        db.add_message(session_id, role, content, timestamp)

        assert mock_cursor.execute.call_count == 3

        expected_calls = [
            (
                """
                    INSERT INTO sessions (session_id, last_updated)
                    VALUES (%s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id, timestamp),
            ),
            (
                """
                    INSERT INTO messages (session_id, role, content, timestamp)
                    VALUES (%s, %s, %s, %s)
                """,
                (session_id, role, content, timestamp),
            ),
            (
                """
                    UPDATE sessions SET last_updated = %s WHERE session_id = %s
                """,
                (timestamp, session_id),
            ),
        ]

        actual_calls = [call.args for call in mock_cursor.execute.call_args_list]
        for expected in expected_calls:
            assert expected in actual_calls

        mock_conn.commit.assert_called_once()

    def test_add_message_without_timestamp_uses_current_time(self, postgresql_db):
        db, mock_cursor, mock_conn = postgresql_db
        session_id = "test_session"
        role = "assistant"
        content = "Hi there"

        mock_cursor.execute.reset_mock()
        mock_conn.commit.reset_mock()

        db.add_message(session_id, role, content)

        assert mock_cursor.execute.call_count == 3

        first_call = mock_cursor.execute.call_args_list[0].args
        second_call = mock_cursor.execute.call_args_list[1].args
        third_call = mock_cursor.execute.call_args_list[2].args

        assert first_call[1][0] == session_id
        assert isinstance(first_call[1][1], datetime)

        assert second_call[1][0] == session_id
        assert second_call[1][1] == role
        assert second_call[1][2] == content
        assert isinstance(second_call[1][3], datetime)

        assert isinstance(third_call[1][0], datetime)
        assert third_call[1][1] == session_id

        mock_conn.commit.assert_called_once()


class TestLoadSession:
    def test_load_session_returns_messages(self, postgresql_db):
        """Test that load_session retrieves messages from the database."""
        db, mock_cursor, _ = postgresql_db

        mock_cursor.execute.reset_mock()

        mock_cursor.fetchall.return_value = [
            ("user", "Hello", datetime(2025, 12, 18, 17, 58, 43)),
            ("assistant", "Hi", datetime(2025, 12, 18, 17, 58, 44)),
        ]

        session_id = "test_session"
        loaded_messages = db.load_session(session_id)

        assert loaded_messages == [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": datetime(2025, 12, 18, 17, 58, 43),
            },
            {
                "role": "assistant",
                "content": "Hi",
                "timestamp": datetime(2025, 12, 18, 17, 58, 44),
            },
        ]

        mock_cursor.execute.assert_called_once_with(
            """
                    SELECT role, content, timestamp FROM messages
                    WHERE session_id = %s
                    ORDER BY message_id ASC
                """,
            (session_id,),
        )

    def test_load_session_returns_empty_list_if_not_found(self, postgresql_db):
        """Test that load_session returns an empty list if the session has no messages."""
        db, mock_cursor, _ = postgresql_db

        mock_cursor.execute.reset_mock()

        mock_cursor.fetchall.return_value = []

        loaded_messages = db.load_session("nonexistent_session")

        assert loaded_messages == []
        mock_cursor.execute.assert_called_once_with(
            """
                    SELECT role, content, timestamp FROM messages
                    WHERE session_id = %s
                    ORDER BY message_id ASC
                """,
            ("nonexistent_session",),
        )


class TestListSessions:
    def test_list_sessions_returns_all_sessions(self, postgresql_db):
        """Test that list_sessions returns all sessions in the database."""
        db, mock_cursor, _ = postgresql_db

        mock_cursor.execute.reset_mock()

        mock_cursor.fetchall.return_value = [
            ("session_1", datetime(2025, 12, 18, 17, 58, 43)),
            ("session_2", datetime(2025, 12, 18, 17, 59, 43)),
        ]

        sessions = db.list_sessions()

        assert len(sessions) == 2
        assert sessions[0][0] == "session_1"
        assert sessions[1][0] == "session_2"
        mock_cursor.execute.assert_called_once_with(
            "SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC"
        )

    def test_list_sessions_returns_empty_list_if_no_sessions(self, postgresql_db):
        """Test that list_sessions returns an empty list if there are no sessions."""
        db, mock_cursor, _ = postgresql_db

        mock_cursor.execute.reset_mock()

        mock_cursor.fetchall.return_value = []

        sessions = db.list_sessions()

        assert sessions == []
        mock_cursor.execute.assert_called_once_with(
            "SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC"
        )


class TestDeleteSession:
    def test_delete_session_removes_data(self, postgresql_db):
        """Test that delete_session removes a session from the database."""
        db, mock_cursor, mock_conn = postgresql_db

        mock_cursor.execute.reset_mock()
        mock_conn.commit.reset_mock()

        session_id = "test_session"
        db.delete_session(session_id)

        mock_cursor.execute.assert_any_call("DELETE FROM messages WHERE session_id = %s", (session_id,))
        mock_cursor.execute.assert_any_call("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        mock_conn.commit.assert_called_once()


class TestFlushDatabase:
    @patch.object(PostgreSQLChatDatabase, "confirm_db_flush", return_value=True)
    def test_flush_database_removes_all_data(self, mock_confirm_flush, postgresql_db):
        """Test that flush_database removes all data from the database."""
        db, mock_cursor, mock_conn = postgresql_db

        mock_cursor.execute.reset_mock()
        mock_conn.commit.reset_mock()

        db.flush_database()

        mock_confirm_flush.assert_called_once()
        mock_cursor.execute.assert_any_call("DELETE FROM messages")
        mock_cursor.execute.assert_any_call("DELETE FROM sessions")
        mock_conn.commit.assert_called_once()

    @patch.object(PostgreSQLChatDatabase, "confirm_db_flush", return_value=False)
    def test_flush_database_cancels_on_user_decline(self, mock_confirm_flush, postgresql_db):
        """Test that flush_database does not remove data if user declines."""
        db, mock_cursor, _ = postgresql_db

        mock_cursor.execute.reset_mock()

        db.flush_database()

        mock_confirm_flush.assert_called_once()
        mock_cursor.assert_not_called()
