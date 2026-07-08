# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base class for chat history database implementations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple

from mada.core.config import DatabaseConfig


class BaseChatDatabase(ABC):
    """
    Abstract base class for chat history database implementations.

    This class defines the common interface that all database implementations
    must adhere to, ensuring consistency across different database types.

    Attributes:
        db_config (core.config.DatabaseConfig): Configuration object containing database
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

    def __init__(self, db_config: DatabaseConfig):
        """
        Initialize the database by creating necessary tables or structures.

        Args:
            db_config (core.config.DatabaseConfig): Configuration settings for the database.
        """
        self.db_config = db_config
        self.init_db()

    @abstractmethod
    def init_db(self):
        """
        Initialize the database by creating necessary tables or structures.
        """

    @abstractmethod
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

    @abstractmethod
    def create_session(self, session_id: str):
        """
        Create a new session entry.

        Args:
            session_id (str): The ID of the session to create.
        """

    @abstractmethod
    def load_session(self, session_id: str) -> List:
        """
        Load a session's messages from the database.

        Args:
            session_id (str): The ID of the session to load.

        Returns:
            list: List of messages from the session.
        """

    @abstractmethod
    def list_sessions(self) -> List[Tuple[str, datetime]]:
        """
        List all sessions in the database.

        Returns:
            list: List of tuples containing session_id and last_updated.
        """

    @abstractmethod
    def delete_session(self, session_id: str):
        """
        Delete a session from the database.

        Args:
            session_id (str): The ID of the session to delete.
        """

    def confirm_db_flush(self) -> bool:
        """
        Confirm that the user wants to flush the entire database.

        Returns:
            True if the user wants to flush the entire database. False otherwise.
        """
        valid_inputs = ["y", "n"]
        user_input = (
            input("Are you sure you want to flush the entire database? (y/n): ")
            .strip()
            .lower()
        )
        while user_input not in valid_inputs:
            user_input = (
                input("Invalid input. Use 'y' for 'yes' or 'n' for 'no': ")
                .strip()
                .lower()
            )

        if user_input == "y":
            return True
        return False

    @abstractmethod
    def flush_database(self, confirm: bool = True):
        """
        Remove all data from the database, effectively resetting it.

        Warning:
            This operation is irreversible and will delete all sessions.

        Args:
            confirm (bool): If True, confirm that the user wants to flush the
                database prior to deleting anything. Otherwise, don't ask.
        """
