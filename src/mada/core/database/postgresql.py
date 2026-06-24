# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
PostgreSQL database implementation for chat history management.
"""

import logging
from datetime import datetime
from typing import List, Tuple

import psycopg2

from mada.core.database.base_db import BaseChatDatabase


LOG = logging.getLogger(__name__)


class PostgreSQLChatDatabase(BaseChatDatabase):
    """
    PostgreSQLChatDatabase is a concrete implementation of BaseChatDatabase,
    designed to manage chat history using a PostgreSQL database.

    Attributes:
        db_config (core.config.PostgreSQLConfig): Configuration object containing database
            connection details.

    Methods:
        init_db: Initializes the database by creating the necessary table if it
            doesn't exist.
        add_message: Saves a message in the database.
        create_session: Create a new session entry in the database.
        load_session: Loads a session's messages from the database.
        list_sessions: Lists all sessions in the database.
        delete_session: Deletes a session from the database.
        confirm_db_flush: Confirms that the user wants to flush the entire database.
        flush_database: Removes all data from the database, effectively resetting
            it.
    """

    def _connect(self):
        """
        Establish a connection to the PostgreSQL database.

        Returns:
            psycopg2.connection: Database connection object.
        """
        return psycopg2.connect(
            self.db_config.get_connection_string(), sslmode="require"
        )

    def init_db(self):
        """
        Initialize the database by creating the necessary table if it doesn't exist.
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        last_updated TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id SERIAL PRIMARY KEY,
                        session_id TEXT REFERENCES sessions(session_id),
                        role TEXT,
                        content TEXT,
                        timestamp TIMESTAMP
                    )
                """)
            conn.commit()

    def add_message(
        self, session_id: str, role: str, content: str, timestamp: datetime = None
    ):
        """
        Add a single message to the messages table in the database.

        Args:
            session_id (str): The ID of the session that this message is associated with
            role (str): The role (user or assistant) to designate who wrote the message
            content (str): The message contents
            timestamp (datetime): The time that the message was created
        """
        if timestamp is None:
            timestamp = datetime.now()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                # Ensure the session exists
                cursor.execute(
                    """
                    INSERT INTO sessions (session_id, last_updated)
                    VALUES (%s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                """,
                    (session_id, timestamp),
                )
                # Insert the message
                cursor.execute(
                    """
                    INSERT INTO messages (session_id, role, content, timestamp)
                    VALUES (%s, %s, %s, %s)
                """,
                    (session_id, role, content, timestamp),
                )
                # Update session's last_updated
                cursor.execute(
                    """
                    UPDATE sessions SET last_updated = %s WHERE session_id = %s
                """,
                    (timestamp, session_id),
                )
            conn.commit()

    def create_session(self, session_id: str):
        """
        Create a new session entry.

        Args:
            session_id (str): The ID of the session to create.
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (session_id, last_updated)
                    VALUES (%s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                """,
                    (session_id, datetime.now()),
                )

    def load_session(self, session_id: str) -> List[dict]:
        """
        Load a session's messages from the database.

        Args:
            session_id (str): The ID of the session to load.

        Returns:
            List of messages from the session.
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT role, content, timestamp FROM messages
                    WHERE session_id = %s
                    ORDER BY message_id ASC
                """,
                    (session_id,),
                )
                return [
                    {"role": row[0], "content": row[1], "timestamp": row[2]}
                    for row in cursor.fetchall()
                ]

    def list_sessions(self) -> List[Tuple[str, datetime]]:
        """
        List all sessions in the database.

        Returns:
            List of tuples containing session_id and last_updated.
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC"
                )
                return [
                    (session_id, last_updated)
                    for session_id, last_updated in cursor.fetchall()
                ]

    def delete_session(self, session_id: str):
        """
        Delete a session from the database.

        Args:
            session_id (str): The ID of the session to delete.
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM messages WHERE session_id = %s", (session_id,)
                )
                cursor.execute(
                    "DELETE FROM sessions WHERE session_id = %s", (session_id,)
                )
            conn.commit()

    def flush_database(self, confirm: bool = True):
        """
        Remove all data from the database, effectively resetting it.

        Warning:
            This operation is irreversible and will delete all sessions.

        Args:
            confirm (bool): If True, confirm that the user wants to flush the
                database prior to deleting anything. Otherwise, don't ask.
        """

        def _flush_db():
            LOG.info("Flushing the database...")
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM messages")
                    cursor.execute("DELETE FROM sessions")
                conn.commit()
            LOG.info("Database successfully flushed.")

        if confirm:
            if self.confirm_db_flush():
                _flush_db()
            else:
                LOG.info("Database flush cancelled.")
        else:
            _flush_db()
