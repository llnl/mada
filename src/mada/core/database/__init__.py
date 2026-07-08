# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
The database package provides database implementations for managing chat history.

Modules:
    base_db: Defines the abstract base class for chat database implementations.
    postgresql: Implements chat history management using PostgreSQL.
    sqlite: Implements chat history management using SQLite.
    db_factory: Provides a factory for creating instances of chat database implementations.
    session_manager: High-level API for interacting with the database.
"""

from mada.core.database.base_db import BaseChatDatabase
from mada.core.database.postgresql import PostgreSQLChatDatabase
from mada.core.database.sqlite import SQLiteChatDatabase
from mada.core.database.db_factory import ChatDatabaseFactory
from mada.core.database.session_manager import ChatSessionManager

__all__ = [
    "BaseChatDatabase",
    "PostgreSQLChatDatabase",
    "SQLiteChatDatabase",
    "ChatDatabaseFactory",
    "ChatSessionManager",
]
