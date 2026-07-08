# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Factory class for creating chat database instances.
"""

import logging
from typing import Any, Dict, List

from mada.common import MADAUnsupportedDatabase
from mada.core.database.base_db import BaseChatDatabase
from mada.core.database.sqlite import SQLiteChatDatabase
from mada.core.database.postgresql import PostgreSQLChatDatabase
from mada.core.config import DatabaseConfig


LOG = logging.getLogger(__name__)


class ChatDatabaseFactory:
    """
    Factory class for creating instances of chat database implementations.

    This class is responsible for registering, validating, and creating
    instances of supported `BaseChatDatabase` implementations (e.g.,
    `SQLiteChatDatabase`).

    Attributes:
        _registry (Dict[str, Dict[str, BaseChatDatabase]]): Maps canonical
            database names to their classes.
        _aliases (Dict[str, str]): Maps alias names to canonical database names.

    Methods:
        register: Register a new database and its optional aliases.
        list_available: Return a list of all registered database names.
        create: Instantiate a registered database by name or alias.
    """

    def __init__(self):
        """
        Initialize the chat database factory.
        """
        # Map canonical names to implementation classes or instances
        self._registry: Dict[str, Dict[str, BaseChatDatabase]] = {}

        # Map aliases to canonical names (e.g., legacy names or shorthand)
        self._aliases: Dict[str, str] = {}

        # Register built-in implementations, if any
        self._register_builtins()

    def _register_builtins(self):
        """
        Register built-in components.
        """
        self.register("sqlite", SQLiteChatDatabase)
        self.register(
            "postgresql", PostgreSQLChatDatabase, aliases=["postgres", "psql"]
        )

    def register(
        self, name: str, database_class: Any, aliases: List[str] = None
    ) -> None:
        """
        Register a new database implementation.

        Args:
            name: Canonical name for the database.
            database_class: The class or implementation to register.
            aliases: Optional alternative names for this database.

        Raises:
            TypeError: If the database_class fails validation.
        """
        if not issubclass(database_class, BaseChatDatabase):
            raise TypeError(f"{database_class} must inherit from `BaseChatDatabase`")

        self._registry[name] = database_class
        LOG.debug(f"Registered database: {name}")

        if aliases:
            for alias in aliases:
                self._aliases[alias] = name
                LOG.debug(f"Registered alias '{alias}' for database '{name}'")

    def list_available(self) -> List[str]:
        """
        Return a list of supported database names.

        Returns:
            A list of canonical names for all available databases.
        """
        return list(self._registry.keys())

    def _get_database_class(
        self, canonical_name: str, name_from_user: str
    ) -> Dict[str, BaseChatDatabase]:
        """
        Retrieve a registered database by its canonical name.

        This method ensures that the database being requested exists. If the
        requested database is not found in the registry, it raises a descriptive
        error with a list of available databases.

        Args:
            canonical_name: The canonical name of the database (resolved from alias).
            name_from_user: The original name or alias provided by the user (used in error messages).

        Returns:
            The dictionary containing info about which chat database and configuration
                are linked to the database at `canonical_name` in the registry.

        Raises:
            MADAUnsupportedDatabase: When the database being requested
                is not in the registry.
        """
        # Grab the database information from the registry and ensure it's supported
        database_class = self._registry.get(canonical_name)
        if database_class is None:
            available = ", ".join(self.list_available())
            raise MADAUnsupportedDatabase(
                f"Database '{name_from_user}' is not supported. "
                f"Available databases: {available}"
            )

        return database_class

    def create(self, database_type: str, config: DatabaseConfig) -> Any:
        """
        Instantiate and return a database of the specified type.

        Args:
            database_type: The name or alias of the database to create.
            config: Database configuration settings. The class passed in here
                should match the database you're trying to create. For example,
                if you want to create a SQLite database, `config` should be an
                instance of [`SQLiteConfig`][core.config.database.SQLiteConfig].

        Returns:
            An instance of the requested database.

        Raises:
            ValueError: If the database instantiation fails.
        """
        # Resolve alias
        canonical_name = self._aliases.get(database_type, database_type)

        # Get the class associated with the name
        database_class = self._get_database_class(canonical_name, database_type)

        # Create and return an instance of the database_class
        try:
            instance = database_class(config)
            LOG.info(f"Created database '{canonical_name}'")
            return instance
        except Exception as e:
            raise ValueError(
                f"Failed to create database '{canonical_name}': {e}"
            ) from e
