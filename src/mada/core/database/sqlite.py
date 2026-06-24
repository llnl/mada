# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
SQLite database implementation for chat history management.
"""

import logging
import sqlite3
from datetime import datetime
from typing import List, Tuple

from mada.core.database.base_db import BaseChatDatabase

LOG = logging.getLogger(__name__)


class SQLiteChatDatabase(BaseChatDatabase):
    """
    SQLiteChatDatabase is a concrete implementation of BaseChatDatabase,
    designed to manage chat history using an SQLite database.

    Attributes:
        db_config (core.config.SQLiteConfig): Configuration object containing database
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

    def _connect(self) -> sqlite3.Connection:
        """
        Establish a connection to the SQLite database.

        Returns:
            sqlite3.Connection: Database connection object.
        """
        return sqlite3.connect(self.db_config.path)

    def init_db(self):
        """
        Initialize the database by creating the necessary tables if they don't exist.
        """
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    last_updated TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)

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
            # Ensure the session exists in the sessions table
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, last_updated)
                VALUES (?, ?)
            """,
                (session_id, timestamp),
            )
            # Insert the message
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (session_id, role, content, timestamp),
            )
            # Update last_updated in sessions
            conn.execute(
                """
                UPDATE sessions SET last_updated = ? WHERE session_id = ?
            """,
                (timestamp, session_id),
            )

    def create_session(self, session_id: str):
        """
        Create a new session entry.

        Args:
            session_id (str): The ID of the session to create.
        """
        with self._connect() as conn:
            # Ensure the session exists in the sessions table
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, last_updated)
                VALUES (?, ?)
            """,
                (session_id, datetime.now()),
            )

    def load_session(self, session_id: str) -> List[dict]:
        """
        Load a sessions messages from the database.

        Args:
            session_id (str): The ID of the session to load.

        Returns:
            List of messages from the session.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT role, content, timestamp FROM messages
                WHERE session_id = ?
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
            results = conn.execute(
                "SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC"
            ).fetchall()
            return [
                (session_id, datetime.fromisoformat(last_updated))
                for session_id, last_updated in results
            ]

    def delete_session(self, session_id: str):
        """
        Delete a session from the database based on its session_id.

        Args:
            session_id (str): The ID of the session to delete.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

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
                conn.execute("DELETE FROM messages")
                conn.execute("DELETE FROM sessions")
            LOG.info("Database successfully flushed.")

        if confirm:
            if self.confirm_db_flush():
                _flush_db()
            else:
                LOG.info("Database flush cancelled.")
        else:
            _flush_db()
