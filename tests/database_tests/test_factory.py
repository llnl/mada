# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from mada.core.database.db_factory import ChatDatabaseFactory
from mada.core.database.sqlite import SQLiteChatDatabase
from mada.core.database.postgresql import PostgreSQLChatDatabase
from mada.core.database.base_db import BaseChatDatabase
from mada.core.config import DatabaseConfig, SQLiteConfig, PostgreSQLConfig
from mada.common import MADAUnsupportedDatabase


class DummyValidDB(BaseChatDatabase):
    """
    Simple valid BaseChatDatabase subclass for testing registration and creation.
    """

    def __init__(self, db_config):
        # Call BaseChatDatabase constructor so it sets db_config and calls init_db
        self.init_called = False
        super().__init__(db_config)

    def init_db(self):
        # Mark that init_db was called by the BaseChatDatabase __init__
        self.init_called = True

    def save_session(self, session_id, messages):
        pass

    def load_session(self, session_id):
        return []

    def list_sessions(self):
        return []

    def delete_session(self, session_id):
        pass

    def flush_database(self, confirm: bool = True):
        pass


class DummyInvalidDB:
    """Does not inherit from BaseChatDatabase, should fail validation."""

    pass


class TestInitialization:
    @pytest.mark.unit
    def test_factory_registers_builtins_on_init(self):
        """
        Verify that the factory registers built-in database types and their aliases
        during initialization.
        """
        factory = ChatDatabaseFactory()

        # Built in registrations
        assert "sqlite" in factory._registry
        assert "postgresql" in factory._registry

        # Check that aliases were added
        assert "postgres" in factory._aliases
        assert "psql" in factory._aliases


class TestCreate:
    @pytest.mark.unit
    def test_basic_creation(self):
        """
        Test the creation of a database instance using a registered database type.
        """
        factory = ChatDatabaseFactory()
        factory.register("dummy_db", DummyValidDB)

        dummy_instance = factory.create("dummy_db", DatabaseConfig)
        assert isinstance(dummy_instance, DummyValidDB)
        assert dummy_instance.init_called

    @pytest.mark.unit
    def test_alias_creation(self):
        """
        Test the creation of a database instance using an alias for a registered database type.
        """
        factory = ChatDatabaseFactory()
        factory.register("dummy_db", DummyValidDB, aliases=["dummy", "legacy_dummy"])

        dummy_instance1 = factory.create("dummy", DatabaseConfig)
        assert isinstance(dummy_instance1, DummyValidDB)
        assert dummy_instance1.init_called

        dummy_instance2 = factory.create("legacy_dummy", DatabaseConfig)
        assert isinstance(dummy_instance2, DummyValidDB)
        assert dummy_instance2.init_called

    @pytest.mark.unit
    def test_create_unsupported_name_raises_custom_exception(self):
        """
        Ensure that attempting to create a database with an unsupported name raises
        the appropriate custom exception.
        """
        factory = ChatDatabaseFactory()
        with pytest.raises(MADAUnsupportedDatabase):
            factory.create("not_registered", DatabaseConfig)

    @pytest.mark.integration
    def test_sqlite_creation(self, tmp_path):
        """
        Verify that an SQLite database instance can be created successfully.
        """
        """Check that sqlite creation works"""
        factory = ChatDatabaseFactory()
        sqlite_instance = factory.create(
            "sqlite", SQLiteConfig(path=tmp_path / "test_basic_creation.db")
        )
        assert isinstance(sqlite_instance, SQLiteChatDatabase)

    @pytest.mark.allocation_required
    @pytest.mark.integration
    def test_postgres_creation(self, postgres_connection: PostgreSQLConfig):
        """
        Verify that a PostgreSQL database instance can be created successfully.
        """
        """Check that postgresql create works"""
        factory = ChatDatabaseFactory()
        postgres_instance = factory.create("postgresql", postgres_connection)
        assert isinstance(postgres_instance, PostgreSQLChatDatabase)

    @pytest.mark.allocation_required
    @pytest.mark.integration
    def test_postgres_creation_with_alias(self, postgres_connection: PostgreSQLConfig):
        """
        Verify that a PostgreSQL database instance can be created using its alias.
        """
        """Check postgres via alias 'psql'"""
        factory = ChatDatabaseFactory()
        postgres_instance_alias = factory.create("psql", postgres_connection)
        assert isinstance(postgres_instance_alias, PostgreSQLChatDatabase)


class TestRegister:
    @pytest.mark.unit
    def test_register_valid_subclass(self):
        """
        Test the registration of a valid subclass of BaseChatDatabase.
        """
        factory = ChatDatabaseFactory()
        factory.register("dummy_db", DummyValidDB)
        assert "dummy_db" in factory._registry

    @pytest.mark.unit
    def test_register_invalid_subclass(self):
        """
        Ensure that attempting to register a class that does not inherit from
        BaseChatDatabase raises a TypeError.
        """
        factory = ChatDatabaseFactory()

        with pytest.raises(TypeError) as excinfo:
            factory.register("invalid_db", DummyInvalidDB)

        assert "must inherit from `BaseChatDatabase`" in str(excinfo.value)

    @pytest.mark.unit
    def test_register_with_aliases(self):
        """
        Test the registration of a database type with multiple aliases.
        """
        factory = ChatDatabaseFactory()

        factory.register("dummy_db", DummyValidDB, aliases=["dummy", "legacy_dummy"])

        assert "dummy_db" in factory._registry
        assert "dummy" in factory._aliases
        assert "legacy_dummy" in factory._aliases


class TestListAvailable:
    @pytest.mark.unit
    def test_list_available_returns_only_canonical_names(self):
        """
        Verify that the list_available method returns only canonical names
        and excludes aliases.
        """
        factory = ChatDatabaseFactory()

        factory.register("dummy_db", DummyValidDB, aliases=["dummy_alias"])

        # Aliases should not appear here
        available = factory.list_available()
        assert "dummy_db" in available
        assert "dummy_alias" not in available


@pytest.mark.integration
def test_e2e_factory_life_cycle():
    """
    Perform an end-to-end test of the factory lifecycle, including initialization,
    registration, alias handling, and instance creation.
    """
    # Initialization
    factory = ChatDatabaseFactory()

    # Check list_available for built ins
    available = factory.list_available()
    assert len(available) == 2
    assert "sqlite" in available
    assert "postgresql" in available

    # Check that aliases were added
    assert len(factory._aliases) == 2
    assert "postgres" in factory._aliases
    assert "psql" in factory._aliases

    # Register a new database with aliases
    factory.register("dummy", DummyValidDB, aliases=["dummy_db"])

    # Check list_available again
    new_available = factory.list_available()
    assert len(new_available) == 3
    assert "dummy" in new_available

    # Check that the new alias was added
    assert len(factory._aliases) == 3
    assert "dummy_db" in factory._aliases

    # Create an instance
    instance = factory.create("dummy", DatabaseConfig)
    assert isinstance(instance, DummyValidDB)
    assert instance.init_called
