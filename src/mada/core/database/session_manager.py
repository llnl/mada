# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Chat session management utilities.

This module provides the `ChatSessionManager` class, which offers a high level
interface for creating, selecting, and managing chat sessions backed by a
pluggable database implementation.

The manager abstracts away the underlying database type through the
`BaseChatDatabase` interface and `ChatDatabaseFactory`, allowing clients to
work with chat sessions using a simple API
"""

import uuid
from datetime import datetime
from typing import Dict, List, Tuple

from mada.core.database import BaseChatDatabase, ChatDatabaseFactory
from mada.core.config import DatabaseConfig, SQLiteConfig


class ChatSessionManager:
    """
    Manage chat sessions and their message histories using a backing database.

    This class provides a thin abstraction around a `BaseChatDatabase`
    implementation so that callers can interact with chat sessions using a
    simple, stateful interface. One writable primary session is tracked via
    `current_session_id`, while optional read-only context sessions are tracked
    separately in `context_session_ids` and can be merged into model context
    without changing the persistence target.

    The underlying database implementation is created using
    `ChatDatabaseFactory`. If no `database_config` is supplied, a default
    `SQLiteConfig` is used. If the provided configuration does not specify
    a `session_id`, a new UUID based session ID is created automatically.

    Attributes:
        chat_db: Concrete implementation of `BaseChatDatabase` used to
            store and retrieve chat data.
        current_session_id: Identifier of the active chat session. This is
            either taken from the provided `DatabaseConfig` or generated
            using `create_session_id`.
        context_session_ids: Additional session IDs that should be loaded as
            read-only model context alongside the primary session.

    Methods:
        create_session_id: Generates a new unique identifier for a chat
            session.
        select_session: Sets the given session as the current session and
            returns its message history.
        load_history: Loads the message history for the current session from
            the database.
        add_message: Adds a message entry to the current session in the
            database.
        list_sessions: Lists all sessions known to the database, including
            their last updated time.
        delete_session: Deletes the current session from the database.
    """

    def __init__(self, database_config: DatabaseConfig = None):
        """
        Constructor for the ChatSessionManager.

        Args:
            database_config: The database configuration object to
                use for connecting to the database. If no database
                configuration is provided, default to SQLite.
        """
        self.chat_db: BaseChatDatabase = self._connect_to_db(database_config)

        # Store the current session ID
        if database_config is not None and database_config.session_id is not None:
            self.current_session_id = database_config.session_id
        else:
            self.current_session_id = self.create_session_id()
        self.context_session_ids: List[str] = []

    def _connect_to_db(self, database_config: DatabaseConfig) -> BaseChatDatabase:
        """
        Given a database configuration, connect to the database.

        Args:
            database_config: The database configuration object to
                use for connecting to the database. If no database
                configuration is provided, default to SQLite.

        Returns:
            A connected database instance.
        """
        if database_config is None:
            database_config = SQLiteConfig()
        chat_db_factory = ChatDatabaseFactory()
        chat_db = chat_db_factory.create(database_config.type, database_config)
        return chat_db

    def create_session_id(self) -> str:
        """
        Create a new chat session ID.

        Returns:
            A new session ID.
        """
        return str(uuid.uuid4())

    def create_new_session(self, session_id: str = None):
        """
        Save an empty session to the database.

        Args:
            session_id (str): Specific session ID to assign to the new session.
        """
        if not session_id:
            session_id = self.current_session_id
        self.chat_db.create_session(session_id)

    @staticmethod
    def _dedupe_session_ids(session_ids: List[str]) -> List[str]:
        """
        Return session IDs with empty values and duplicates removed in order.
        """
        deduped_session_ids = []
        seen_session_ids = set()

        for session_id in session_ids:
            normalized_session_id = str(session_id or "").strip()
            if not normalized_session_id or normalized_session_id in seen_session_ids:
                continue
            deduped_session_ids.append(normalized_session_id)
            seen_session_ids.add(normalized_session_id)

        return deduped_session_ids

    def select_session(self, session_id: str) -> List[Dict]:
        """
        Set the given session as the writable primary session and return its history.

        If the selected session is already present in `context_session_ids`, it
        is removed from that read-only list so the primary session is not loaded
        twice.

        Args:
            session_id (str): The ID of the session to select.

        Returns:
            The session chat history.
        """
        self.current_session_id = session_id
        self.context_session_ids = [
            existing_session_id
            for existing_session_id in self.context_session_ids
            if existing_session_id != session_id
        ]
        return self.load_history(session_id)

    def set_context_sessions(self, session_ids: List[str] | None = None) -> List[str]:
        """
        Store the read-only chat sessions that should be loaded into model context.

        Args:
            session_ids: Optional list of session IDs to attach as additional
                context sources.

        Returns:
            The normalized list of tracked context session IDs.
        """
        primary_session_id = self.current_session_id
        deduped_session_ids = self._dedupe_session_ids(list(session_ids or []))
        self.context_session_ids = [
            session_id
            for session_id in deduped_session_ids
            if session_id != primary_session_id
        ]
        return list(self.context_session_ids)

    def get_context_session_ids(self) -> List[str]:
        """
        Return the additional chat sessions loaded as read-only context.
        """
        return list(self.context_session_ids)

    def get_loaded_session_ids(
        self,
        primary_session_id: str | None = None,
        context_session_ids: List[str] | None = None,
    ) -> List[str]:
        """
        Return the effective session IDs loaded into model context.

        Args:
            primary_session_id: Optional explicit primary session. Defaults to the
                active session.
            context_session_ids: Optional explicit context sessions. Defaults to
                the tracked additional context sessions.

        Returns:
            Ordered session IDs containing the additional context sessions first
            and the primary writable session last.
        """
        # Resolve the primary session ID and context session IDs, defaulting to the
        # current session and tracked context sessions if not provided.
        resolved_primary_session_id = primary_session_id or self.current_session_id
        resolved_context_session_ids = list(
            self.context_session_ids
            if context_session_ids is None
            else context_session_ids
        )

        # Deduplicate the context session IDs.
        loaded_session_ids = self._dedupe_session_ids(resolved_context_session_ids)

        # If the primary session ID is present, ensure it is the last entry in the list.
        if resolved_primary_session_id:
            loaded_session_ids = [
                session_id
                for session_id in loaded_session_ids
                if session_id != resolved_primary_session_id
            ]
            loaded_session_ids.append(resolved_primary_session_id)

        return loaded_session_ids

    def load_history(self, session_id: str = None) -> List[Dict]:
        """
        Load the message history from one persisted session.

        Args:
            session_id: Optional explicit session to load. Defaults to the
                currently selected session.

        Returns:
            A list of messages for the given session.
        """
        resolved_session_id = session_id or self.current_session_id
        if not resolved_session_id:
            return []
        return self.chat_db.load_session(resolved_session_id) or []

    def load_histories(self, session_ids: List[str]) -> Dict[str, List[Dict]]:
        """
        Load histories for multiple sessions without mutating the active session.

        Args:
            session_ids: Session IDs to load.

        Returns:
            Mapping of session ID to stored messages.
        """
        return {
            session_id: self.load_history(session_id)
            for session_id in self._dedupe_session_ids(session_ids)
        }

    def add_message(self, role: str, message: str):
        """
        Add a message to the database.

        Args:
            role (str): The role (user or assistant) to designate who wrote the message
            message (str): The message contents
        """
        self.chat_db.add_message(self.current_session_id, role, message)

    def list_sessions(self) -> List[Tuple[str, datetime]]:
        """
        List all sessions currently stored in the database.

        Returns:
            A list of sessions where each session has (session ID, last updated).
        """
        return self.chat_db.list_sessions()

    def delete_session(self, session_id: str = None):
        """
        Given a session ID, delete the session from the database.

        If no session ID is provided, the currently tracked session is deleted.

        Args:
            session_id (str): Specific session ID to delete from the database.
        """
        if session_id is None:
            session_id = self.current_session_id
        self.chat_db.delete_session(session_id)

    def delete_all_sessions(self, confirm: bool = True):
        """
        Delete all sessions from the database.

        Warning:
            This operation is irreversible and will delete all chat history.

        Args:
            confirm (bool): If True, prompt the user for confirmation before
                deleting. If False, delete without prompting.
        """
        self.chat_db.flush_database(confirm=confirm)
        self.current_session_id = None
