# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Integration tests for PostgreSQLChatDatabase.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from mada.core.database import PostgreSQLChatDatabase


@pytest.mark.allocation_required
class TestPostgreSQLChatDatabaseIntegration:
    class TestInitDB:
        def test_init_db_creates_tables(self, postgresql_db):
            """Integration test for init_db method."""
            postgresql_db.init_db()

            with postgresql_db._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_name IN ('sessions', 'messages')
                    """)
                    result = cursor.fetchall()

            table_names = {row[0] for row in result}
            assert "sessions" in table_names
            assert "messages" in table_names

    class TestCreateSession:
        def test_create_session_inserts_data(self, postgresql_db):
            """Integration test for create_session method."""
            session_id = "test_create_session_inserts_data"

            postgresql_db.create_session(session_id)

            with postgresql_db._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT session_id, last_updated
                        FROM sessions
                        WHERE session_id = %s
                    """,
                        (session_id,),
                    )
                    result = cursor.fetchone()

            assert result is not None
            assert result[0] == session_id
            assert result[1] is not None

        def test_create_session_handles_duplicate_session(self, postgresql_db):
            """Integration test that duplicate create_session does not fail."""
            session_id = "test_create_session_handles_duplicate_session"

            postgresql_db.create_session(session_id)
            postgresql_db.create_session(session_id)

            with postgresql_db._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT COUNT(*)
                        FROM sessions
                        WHERE session_id = %s
                    """,
                        (session_id,),
                    )
                    result = cursor.fetchone()

            assert result[0] == 1

    class TestAddMessage:
        def test_add_message_inserts_message_and_session(self, postgresql_db):
            """Integration test for add_message method."""
            session_id = "test_add_message_inserts_message_and_session"
            timestamp = datetime(2025, 12, 18, 17, 58, 43)

            postgresql_db.add_message(
                session_id=session_id,
                role="user",
                content="Hello",
                timestamp=timestamp,
            )

            with postgresql_db._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT session_id, last_updated
                        FROM sessions
                        WHERE session_id = %s
                    """,
                        (session_id,),
                    )
                    session_result = cursor.fetchone()

                    cursor.execute(
                        """
                        SELECT session_id, role, content, timestamp
                        FROM messages
                        WHERE session_id = %s
                    """,
                        (session_id,),
                    )
                    message_result = cursor.fetchone()

            assert session_result is not None
            assert session_result[0] == session_id
            assert session_result[1] == timestamp

            assert message_result is not None
            assert message_result[0] == session_id
            assert message_result[1] == "user"
            assert message_result[2] == "Hello"
            assert message_result[3] == timestamp

        def test_add_message_updates_last_updated(self, postgresql_db):
            """Integration test that add_message updates session last_updated."""
            session_id = "test_add_message_updates_last_updated"
            first_timestamp = datetime(2025, 12, 18, 17, 58, 43)
            second_timestamp = datetime(2025, 12, 18, 17, 59, 43)

            postgresql_db.add_message(session_id, "user", "Hello", first_timestamp)
            postgresql_db.add_message(session_id, "assistant", "Hi", second_timestamp)

            with postgresql_db._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT last_updated
                        FROM sessions
                        WHERE session_id = %s
                    """,
                        (session_id,),
                    )
                    result = cursor.fetchone()

            assert result is not None
            assert result[0] == second_timestamp

    class TestLoadSession:
        def test_load_session_returns_messages(self, postgresql_db):
            """Integration test for load_session method."""
            session_id = "test_load_session_returns_messages"
            timestamp_1 = datetime(2025, 12, 18, 17, 58, 43)
            timestamp_2 = datetime(2025, 12, 18, 17, 58, 44)

            postgresql_db.add_message(session_id, "user", "Hello", timestamp_1)
            postgresql_db.add_message(session_id, "assistant", "Hi", timestamp_2)

            loaded_messages = postgresql_db.load_session(session_id)

            assert loaded_messages == [
                {"role": "user", "content": "Hello", "timestamp": timestamp_1},
                {"role": "assistant", "content": "Hi", "timestamp": timestamp_2},
            ]

        def test_load_session_returns_empty_list_if_not_found(self, postgresql_db):
            """Integration test for load_session method when session is not found."""
            session_id = "nonexistent_session"

            loaded_messages = postgresql_db.load_session(session_id)

            assert loaded_messages == []

    class TestListSessions:
        def test_list_sessions_returns_all_sessions(self, postgresql_db):
            """Integration test for list_sessions method."""
            session_1 = "test_list_sessions_returns_all_sessions_1"
            session_2 = "test_list_sessions_returns_all_sessions_2"
            timestamp_1 = datetime(2025, 12, 18, 17, 58, 43)
            timestamp_2 = datetime(2025, 12, 18, 17, 59, 43)

            postgresql_db.add_message(session_1, "user", "Hi", timestamp_1)
            postgresql_db.add_message(session_2, "user", "How are you?", timestamp_2)

            sessions = postgresql_db.list_sessions()

            assert len(sessions) == 2
            assert sessions[0][0] == session_2
            assert sessions[0][1] == timestamp_2
            assert sessions[1][0] == session_1
            assert sessions[1][1] == timestamp_1

        def test_list_sessions_returns_empty_list_if_no_sessions(self, postgresql_db):
            """Integration test for list_sessions method when no sessions exist."""
            sessions = postgresql_db.list_sessions()

            assert sessions == []

    class TestDeleteSession:
        def test_delete_session_removes_data(self, postgresql_db):
            """Integration test for delete_session method."""
            session_id = "test_delete_session_removes_data"

            postgresql_db.add_message(
                session_id, "user", "Hello", datetime(2025, 12, 18, 17, 58, 43)
            )
            postgresql_db.delete_session(session_id)

            with postgresql_db._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT session_id
                        FROM sessions
                        WHERE session_id = %s
                    """,
                        (session_id,),
                    )
                    session_result = cursor.fetchone()

                    cursor.execute(
                        """
                        SELECT session_id
                        FROM messages
                        WHERE session_id = %s
                    """,
                        (session_id,),
                    )
                    message_result = cursor.fetchone()

            assert session_result is None
            assert message_result is None

    class TestFlushDatabase:
        @patch.object(PostgreSQLChatDatabase, "confirm_db_flush", return_value=True)
        def test_flush_database_removes_all_data(self, mock_confirm, postgresql_db):
            """Integration test for flush_database method."""
            session_1 = "test_flush_database_removes_all_data_1"
            session_2 = "test_flush_database_removes_all_data_2"

            postgresql_db.add_message(
                session_1, "user", "Hi", datetime(2025, 12, 18, 17, 58, 43)
            )
            postgresql_db.add_message(
                session_2, "assistant", "Hello", datetime(2025, 12, 18, 17, 59, 43)
            )

            postgresql_db.flush_database()

            sessions = postgresql_db.list_sessions()
            assert sessions == []
            mock_confirm.assert_called_once()

        @patch.object(PostgreSQLChatDatabase, "confirm_db_flush", return_value=False)
        def test_flush_database_cancels_on_user_decline(
            self, mock_confirm, postgresql_db
        ):
            """Integration test that flush_database does not remove data if user declines."""
            session_id = "test_flush_database_cancels_on_user_decline"

            postgresql_db.add_message(
                session_id, "user", "Hi", datetime(2025, 12, 18, 17, 58, 43)
            )

            postgresql_db.flush_database()

            sessions = postgresql_db.list_sessions()
            assert len(sessions) == 1
            assert sessions[0][0] == session_id
            mock_confirm.assert_called_once()
