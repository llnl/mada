# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Database configuration definitions and loading utilities.

This module defines configuration dataclasses for supported database backends
used by the application. It includes `SQLiteConfig` for file-based SQLite
storage, `PostgreSQLConfig` for PostgreSQL connection settings, and
`load_database_config` for constructing the appropriate configuration object
from a dictionary.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union


LOG = logging.getLogger("mada-interface")


@dataclass
class SQLiteConfig:
    """
    Configuration for SQLite database.

    Attributes:
        type (str): The type of database that this config is used for.
        path (str | Path): The path for SQLite to use as the store.
        session_id (str): An optional session ID to connect to. If no
            session ID is provided, a new session (chat) will be created.
    """

    type: str = "sqlite"
    path: str | Path = Path.home() / ".mada" / "chat_history.db"
    session_id: str = None

    def __post_init__(self):
        """
        After initialization, expand variables in the path and
        make sure it exists.
        """
        # Expand variables and convert to Path object
        self.path = Path(os.path.expandvars(self.path))

        # Ensure parent directory exists
        if self.path.parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class PostgreSQLConfig:
    """
    Configuration for PostgreSQL database.

    When connecting to PostgreSQL, we use a connection string.
    This class allows you to provide the connection string yourself
    or individual settings that will eventually be converted to a
    connection string.

    If both a connection string and individual settings are
    provided, the connection string will take precedence.

    These entries can all include shell variables.

    Attributes:
        type (str): The type of database that this config is used for.
        session_id (str): An optional session ID to connect to. If no
            session ID is provided, a new session (chat) will be created.
        connection_string (str | None): A PostgreSQL connection string.
        host (str | None): The service host. In LaunchIT, this is the
            'service-host' setting.
        port (int): The service port. In LaunchIT, this is the
            'service-port' setting.
        database (str | None): The database name. In LaunchIT, this is
            the 'database-name' setting.
        user (str | None): The database user. In LaunchIT, this is the
            'database-user' setting.
        password (str | None): The database password. In LaunchIT, this
            is the 'database-password' setting. You will likely want to
            store this as an environment variable and then pass in a
            reference to that environment variable.

    Methods:
        get_connection_string: Get the connection string for the database.
    """

    type: str = "postgresql"
    session_id: str = None

    # Option 1: Connection string (takes precedence)
    connection_string: str | None = None

    # Option 2: Individual fields
    host: str | None = None
    port: int = 5432
    database: str | None = None
    user: str | None = None
    password: str | None = None

    def __post_init__(self):
        """
        After initialization, expand variables and ensure required
        settings have been provided (if necessary).

        Raises:
            ValueError: If a required setting is missing.
        """
        # Expand environment variables
        if self.connection_string:
            self.connection_string = os.path.expandvars(self.connection_string)
        else:
            # Must have individual fields
            required = ["host", "port", "database", "user", "password"]
            missing = [f for f in required if getattr(self, f) is None]
            if missing:
                raise ValueError(
                    f"PostgreSQL requires either 'connection_string' or "
                    f"all of: {', '.join(required)}. Missing: {', '.join(missing)}"
                )

            # Expand env vars in individual fields
            if self.host:
                self.host = os.path.expandvars(self.host)
            if self.database:
                self.database = os.path.expandvars(self.database)
            if self.user:
                self.user = os.path.expandvars(self.user)
            if self.password:
                self.password = os.path.expandvars(self.password)

    def get_connection_string(self) -> str:
        """
        Build the connection string (if necessary) and return it.

        Returns:
            The connection string for the database.
        """
        if self.connection_string:
            return self.connection_string

        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


DatabaseConfig = Union[SQLiteConfig | PostgreSQLConfig]


def load_database_config(config_dict: dict) -> DatabaseConfig:
    """
    Factory function to create the appropriate config from a dict.

    Args:
        config_dict: The configuration for the database provided by
            the user.

    Returns:
        An instance of the appropriate database class.

    Raises:
        ValueError: If the database type is unsupported.
    """
    db_type = config_dict.get("type", "sqlite")

    if db_type == "sqlite":
        return SQLiteConfig(**config_dict)
    elif db_type in ["postgresql", "postgres", "psql"]:
        return PostgreSQLConfig(**config_dict)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
