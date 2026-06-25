# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Integration tests for SQLiteChatDatabase.
"""

from unittest.mock import patch


class TestInitDB:
    def test_init_db_creates_tables(self, sqlite_db):
        """Integration test for init_db method."""
        # NOTE: don't need to call init_db as that's done automatically in the constructor

        # Connect to the actual SQLite database file
        with sqlite_db._connect() as conn:
            sessions_result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            ).fetchone()
            messages_result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
            ).fetchone()

        assert sessions_result is not None, "Table 'sessions' was not created."
        assert messages_result is not None, "Table 'messages' was not created."


class TestCreateSession:
    def test_create_session_inserts_data(self, sqlite_db):
        """Integration test for create_session method."""
        session_id = "test_create_session_inserts_data"

        sqlite_db.create_session(session_id)

        # Verify data in the actual SQLite database file
        with sqlite_db._connect() as conn:
            result = conn.execute(
                "SELECT session_id, last_updated FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        assert result is not None, "Session data was not saved."
        assert result[0] == session_id, "Session ID mismatch."
        assert result[1] is not None, "last_updated was not saved."


class TestAddMessage:
    def test_add_message_inserts_data(self, sqlite_db):
        """Integration test for add_message method."""
        session_id = "test_add_message_inserts_data"
        role = "user"
        content = "Hello"

        sqlite_db.add_message(session_id, role, content)

        # Verify data in the actual SQLite database file
        with sqlite_db._connect() as conn:
            session_result = conn.execute(
                "SELECT session_id, last_updated FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            message_result = conn.execute(
                """
                SELECT session_id, role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY message_id ASC
                """,
                (session_id,),
            ).fetchone()

        assert session_result is not None, "Session data was not saved."
        assert session_result[0] == session_id, "Session ID mismatch."
        assert session_result[1] is not None, "last_updated was not saved."

        assert message_result is not None, "Message data was not saved."
        assert message_result[0] == session_id, "Message session ID mismatch."
        assert message_result[1] == role, "Role mismatch."
        assert message_result[2] == content, "Content mismatch."
        assert message_result[3] is not None, "Timestamp was not saved."


class TestLoadSession:
    def test_load_session_returns_messages(self, sqlite_db):
        """Integration test for load_session method."""
        session_id = "test_load_session_returns_messages"

        # Save session first
        sqlite_db.add_message(session_id, "user", "Hello")
        sqlite_db.add_message(session_id, "assistant", "Hi")

        # Load session
        loaded_messages = sqlite_db.load_session(session_id)

        assert len(loaded_messages) == 2, "Loaded messages count mismatch."
        assert loaded_messages[0]["role"] == "user", "First message role mismatch."
        assert loaded_messages[0]["content"] == "Hello", "First message content mismatch."
        assert loaded_messages[1]["role"] == "assistant", "Second message role mismatch."
        assert loaded_messages[1]["content"] == "Hi", "Second message content mismatch."

    def test_load_session_returns_empty_list_if_not_found(self, sqlite_db):
        """Integration test for load_session method when session is not found."""
        session_id = "nonexistent_session"

        # Load session
        loaded_messages = sqlite_db.load_session(session_id)

        assert loaded_messages == [], "Expected empty list for nonexistent session."


class TestListSessions:
    def test_list_sessions_returns_all_sessions(self, sqlite_db):
        """Integration test for list_sessions method."""
        session_1 = "test_list_sessions_returns_all_sessions_1"
        session_2 = "test_list_sessions_returns_all_sessions_2"

        # Save sessions
        sqlite_db.add_message(session_1, "user", "Hi")
        sqlite_db.add_message(session_2, "user", "How are you?")

        # List sessions
        sessions = sqlite_db.list_sessions()

        assert len(sessions) == 2, "Expected two sessions in the database."
        assert sessions[0][0] == session_2, "Expected test_list_sessions_returns_all_sessions_2 to be listed first."
        assert sessions[1][0] == session_1, "Expected test_list_sessions_returns_all_sessions_1 to be listed second."

    def test_list_sessions_returns_empty_list_if_no_sessions(self, sqlite_db):
        """Integration test for list_sessions method when no sessions exist."""
        # List sessions
        sessions = sqlite_db.list_sessions()

        assert sessions == [], "Expected empty list when no sessions exist."


class TestDeleteSession:
    def test_delete_session_removes_data(self, sqlite_db):
        """Integration test for delete_session method."""
        session_id = "test_delete_session_removes_data"

        # Save session first
        sqlite_db.add_message(session_id, "user", "Hello")
        sqlite_db.add_message(session_id, "assistant", "Hi")

        # Delete session
        sqlite_db.delete_session(session_id)

        # Verify session is deleted
        with sqlite_db._connect() as conn:
            session_result = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            message_result = conn.execute(
                "SELECT session_id FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()

        assert session_result is None, "Session was not deleted."
        assert message_result is None, "Session messages were not deleted."


class TestFlushDatabase:
    @patch("builtins.input", return_value="y")
    def test_flush_database_removes_all_data(self, mock_input, sqlite_db):
        """Integration test for flush_database method."""
        session_1 = "test_flush_database_removes_all_data_1"
        session_2 = "test_flush_database_removes_all_data_2"

        # Save sessions
        sqlite_db.add_message(session_1, "user", "Hi")
        sqlite_db.add_message(session_2, "user", "How are you?")

        # Flush database
        sqlite_db.flush_database()

        # Verify all sessions are deleted
        sessions = sqlite_db.list_sessions()
        assert sessions == [], "Database was not flushed."

        with sqlite_db._connect() as conn:
            messages = conn.execute("SELECT * FROM messages").fetchall()

        assert messages == [], "Messages were not flushed."

    @patch("builtins.input", return_value="n")
    def test_flush_database_cancels_on_user_decline(self, mock_input, sqlite_db):
        """Test that flush_database does not remove data if user declines."""
        session = "test_flush_database_cancels_on_user_decline"

        # Save a session to the database
        sqlite_db.add_message(session, "user", "Hi")

        # Flush database
        sqlite_db.flush_database()

        # Verify nothing happened since the user declined the flush
        sessions = sqlite_db.list_sessions()
        assert len(sessions) == 1, "Expected the session to remain in the database."
        assert sessions[0][0] == session, "Expected the session ID to match the saved session."
