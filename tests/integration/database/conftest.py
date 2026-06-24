"""
Fixtures for files in this `database/` test directory.
"""

import pytest

from mada.core.config import PostgreSQLConfig, SQLiteConfig
from mada.core.database import PostgreSQLChatDatabase, SQLiteChatDatabase


@pytest.fixture
def sqlite_db(tmp_path) -> SQLiteChatDatabase:
    """Fixture to initialize SQLiteChatDatabase with SQLiteConfig for integration tests."""
    db_config = SQLiteConfig(path=tmp_path / "sqlite_db_integration_test.db")
    db = SQLiteChatDatabase(db_config=db_config)
    return db


@pytest.fixture
def postgresql_db(postgres_connection: PostgreSQLConfig) -> PostgreSQLChatDatabase:
    """Fixture to initialize PostgreSQLChatDatabase for integration tests."""
    db = PostgreSQLChatDatabase(db_config=postgres_connection)
    db.init_db()
    db.flush_database(confirm=False)
    return db
