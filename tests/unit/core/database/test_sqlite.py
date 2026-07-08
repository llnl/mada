# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Unit tests for the `mada/core/database/postgresql.py` module.
"""

import textwrap
from datetime import datetime
from typing import Tuple
from unittest.mock import MagicMock, patch

import pytest

from mada.core.config import SQLiteConfig
from mada.core.database.sqlite import SQLiteChatDatabase


def normalize_sql(sql: str) -> str:
    """Normalize SQL strings for reliable test comparisons."""
    return textwrap.dedent(sql).strip()


@pytest.fixture
def sqlite_db(tmp_path) -> Tuple[SQLiteChatDatabase, MagicMock]:
    """Fixture to initialize SQLiteChatDatabase with SQLiteConfig."""
    db_config = SQLiteConfig(path=tmp_path / "sqlite_db_unit_test.db")
    db = SQLiteChatDatabase(db_config=db_config)

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None

    db._connect = MagicMock(return_value=mock_conn)
    return db, mock_conn


class TestInitDB:
    def test_init_db_creates_tables(self, sqlite_db):
        """Test that the init_db method creates the sessions and messages tables."""
        db, mock_conn = sqlite_db

        db.init_db()

        actual_sql_calls = [
            normalize_sql(call.args[0]) for call in mock_conn.execute.call_args_list
        ]

        assert (
            normalize_sql("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                last_updated TIMESTAMP
            )
        """)
            in actual_sql_calls
        )

        assert (
            normalize_sql("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        """)
            in actual_sql_calls
        )


class TestCreateSession:
    @patch("mada.core.database.sqlite.datetime")
    def test_create_session_inserts_data(self, mock_datetime, sqlite_db):
        """Test that create_session inserts a session into the database."""
        db, mock_conn = sqlite_db

        session_id = "test_session"
        mock_datetime.now.return_value = datetime(2025, 12, 18, 17, 58, 43)

        db.create_session(session_id)

        sql, params = mock_conn.execute.call_args[0]
        assert normalize_sql(sql) == normalize_sql("""
            INSERT OR IGNORE INTO sessions (session_id, last_updated)
            VALUES (?, ?)
        """)
        assert params == (session_id, mock_datetime.now.return_value)


class TestAddMessage:
    def test_add_message_with_explicit_timestamp(self, sqlite_db):
        """Test that add_message inserts message data and updates session."""
        db, mock_conn = sqlite_db

        session_id = "test_session"
        role = "user"
        content = "Hello"
        timestamp = datetime(2025, 1, 1, 12, 0, 0)

        db.add_message(session_id, role, content, timestamp)

        assert mock_conn.execute.call_count == 3

        actual_calls = [
            (normalize_sql(call.args[0]), call.args[1])
            for call in mock_conn.execute.call_args_list
        ]

        expected_calls = [
            (
                normalize_sql("""
                    INSERT OR IGNORE INTO sessions (session_id, last_updated)
                    VALUES (?, ?)
                """),
                (session_id, timestamp),
            ),
            (
                normalize_sql("""
                    INSERT INTO messages (session_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?)
                """),
                (session_id, role, content, timestamp),
            ),
            (
                normalize_sql("""
                    UPDATE sessions SET last_updated = ? WHERE session_id = ?
                """),
                (timestamp, session_id),
            ),
        ]

        for expected in expected_calls:
            assert expected in actual_calls

    def test_add_message_without_timestamp_uses_current_time(self, sqlite_db):
        """Test that add_message uses datetime.now() when no timestamp is provided."""
        db, mock_conn = sqlite_db

        session_id = "test_session"
        role = "assistant"
        content = "Hi there"

        db.add_message(session_id, role, content)

        assert mock_conn.execute.call_count == 3

        first_call = mock_conn.execute.call_args_list[0].args
        second_call = mock_conn.execute.call_args_list[1].args
        third_call = mock_conn.execute.call_args_list[2].args

        assert normalize_sql(first_call[0]) == normalize_sql("""
            INSERT OR IGNORE INTO sessions (session_id, last_updated)
            VALUES (?, ?)
        """)
        assert first_call[1][0] == session_id
        assert isinstance(first_call[1][1], datetime)

        assert normalize_sql(second_call[0]) == normalize_sql("""
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
        """)
        assert second_call[1][0] == session_id
        assert second_call[1][1] == role
        assert second_call[1][2] == content
        assert isinstance(second_call[1][3], datetime)

        assert normalize_sql(third_call[0]) == normalize_sql("""
            UPDATE sessions SET last_updated = ? WHERE session_id = ?
        """)
        assert isinstance(third_call[1][0], datetime)
        assert third_call[1][1] == session_id


class TestLoadSession:
    def test_load_session_returns_messages(self, sqlite_db):
        """Test that load_session retrieves messages from the database."""
        db, mock_conn = sqlite_db

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("user", "Hello", datetime(2025, 12, 18, 17, 58, 43)),
            ("assistant", "Hi", datetime(2025, 12, 18, 17, 58, 44)),
        ]
        mock_conn.execute.return_value = mock_cursor

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

        sql, params = mock_conn.execute.call_args[0]
        assert normalize_sql(sql) == normalize_sql("""
            SELECT role, content, timestamp FROM messages
            WHERE session_id = ?
            ORDER BY message_id ASC
        """)
        assert params == (session_id,)

    def test_load_session_returns_empty_list_if_not_found(self, sqlite_db):
        """Test that load_session returns an empty list if the session is not found."""
        db, mock_conn = sqlite_db

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        loaded_messages = db.load_session("nonexistent_session")

        assert loaded_messages == []

        sql, params = mock_conn.execute.call_args[0]
        assert normalize_sql(sql) == normalize_sql("""
            SELECT role, content, timestamp FROM messages
            WHERE session_id = ?
            ORDER BY message_id ASC
        """)
        assert params == ("nonexistent_session",)


class TestListSessions:
    def test_list_sessions_returns_all_sessions(self, sqlite_db):
        """Test that list_sessions returns all sessions in the database."""
        db, mock_conn = sqlite_db

        mock_conn.execute.return_value.fetchall.return_value = [
            ("session_1", "2025-12-18T17:58:43"),
            ("session_2", "2025-12-18T17:59:43"),
        ]

        sessions = db.list_sessions()

        assert len(sessions) == 2
        assert sessions[0][0] == "session_1"
        assert sessions[1][0] == "session_2"
        assert sessions[0][1] == datetime(2025, 12, 18, 17, 58, 43)
        assert sessions[1][1] == datetime(2025, 12, 18, 17, 59, 43)

        sql = mock_conn.execute.call_args[0][0]
        assert normalize_sql(sql) == normalize_sql("""
            SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC
        """)

    def test_list_sessions_returns_empty_list_if_no_sessions(self, sqlite_db):
        """Test that list_sessions returns an empty list if there are no sessions."""
        db, mock_conn = sqlite_db
        mock_conn.execute.return_value.fetchall.return_value = []

        sessions = db.list_sessions()

        assert sessions == []

        sql = mock_conn.execute.call_args[0][0]
        assert normalize_sql(sql) == normalize_sql("""
            SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC
        """)


class TestDeleteSession:
    def test_delete_session_removes_data(self, sqlite_db):
        """Test that delete_session removes a session from the database."""
        db, mock_conn = sqlite_db

        session_id = "test_session"
        db.delete_session(session_id)

        actual_calls = [
            (normalize_sql(call.args[0]), call.args[1])
            for call in mock_conn.execute.call_args_list
        ]

        assert (
            normalize_sql("DELETE FROM messages WHERE session_id = ?"),
            (session_id,),
        ) in actual_calls

        assert (
            normalize_sql("DELETE FROM sessions WHERE session_id = ?"),
            (session_id,),
        ) in actual_calls


class TestFlushDatabase:
    @patch.object(SQLiteChatDatabase, "confirm_db_flush", return_value=True)
    def test_flush_database_removes_all_data(self, mock_confirm_flush, sqlite_db):
        """Test that flush_database removes all data from the database."""
        db, mock_conn = sqlite_db

        db.flush_database()

        mock_confirm_flush.assert_called_once()

        actual_sql_calls = [
            normalize_sql(call.args[0]) for call in mock_conn.execute.call_args_list
        ]

        assert normalize_sql("DELETE FROM messages") in actual_sql_calls
        assert normalize_sql("DELETE FROM sessions") in actual_sql_calls

    @patch.object(SQLiteChatDatabase, "confirm_db_flush", return_value=False)
    def test_flush_database_cancels_on_user_decline(
        self, mock_confirm_flush, sqlite_db
    ):
        """Test that flush_database does not remove data if user declines."""
        db, mock_conn = sqlite_db

        db.flush_database()

        mock_confirm_flush.assert_called_once()
        mock_conn.execute.assert_not_called()
