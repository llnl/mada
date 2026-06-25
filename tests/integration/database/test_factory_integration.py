# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Integration tests for ChatDatabaseFactory.
"""

import pytest

from mada.core.config import PostgreSQLConfig, SQLiteConfig
from mada.core.database.db_factory import ChatDatabaseFactory
from mada.core.database.postgresql import PostgreSQLChatDatabase
from mada.core.database.sqlite import SQLiteChatDatabase


class TestCreate:
    def test_sqlite_creation(self, tmp_path):
        """
        Verify that a SQLite database instance can be created successfully.
        """
        factory = ChatDatabaseFactory()
        sqlite_instance = factory.create("sqlite", SQLiteConfig(path=tmp_path / "test_basic_creation.db"))
        assert isinstance(sqlite_instance, SQLiteChatDatabase)

    @pytest.mark.allocation_required
    def test_postgres_creation(self, postgres_connection: PostgreSQLConfig):
        """
        Verify that a PostgreSQL database instance can be created successfully.
        """
        factory = ChatDatabaseFactory()
        postgres_instance = factory.create("postgresql", postgres_connection)
        assert isinstance(postgres_instance, PostgreSQLChatDatabase)

    @pytest.mark.allocation_required
    def test_postgres_creation_with_alias(self, postgres_connection: PostgreSQLConfig):
        """
        Verify that a PostgreSQL database instance can be created using its alias.
        """
        factory = ChatDatabaseFactory()
        postgres_instance_alias = factory.create("psql", postgres_connection)
        assert isinstance(postgres_instance_alias, PostgreSQLChatDatabase)
