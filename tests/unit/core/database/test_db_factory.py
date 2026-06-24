"""
Unit tests for the `mada/core/database/db_factory.py` module.
"""

import pytest

from mada.common import MADAUnsupportedDatabase
from mada.core.config import DatabaseConfig
from mada.core.database.db_factory import ChatDatabaseFactory


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
    def test_basic_creation(self, dummy_valid_db_class):
        """
        Test the creation of a database instance using a registered database type.
        """
        factory = ChatDatabaseFactory()
        factory.register("dummy_db", dummy_valid_db_class)

        dummy_instance = factory.create("dummy_db", DatabaseConfig)
        assert isinstance(dummy_instance, dummy_valid_db_class)
        assert dummy_instance.init_called

    @pytest.mark.unit
    def test_alias_creation(self, dummy_valid_db_class):
        """
        Test the creation of a database instance using an alias for a registered database type.
        """
        factory = ChatDatabaseFactory()
        factory.register("dummy_db", dummy_valid_db_class, aliases=["dummy", "legacy_dummy"])

        dummy_instance1 = factory.create("dummy", DatabaseConfig)
        assert isinstance(dummy_instance1, dummy_valid_db_class)
        assert dummy_instance1.init_called

        dummy_instance2 = factory.create("legacy_dummy", DatabaseConfig)
        assert isinstance(dummy_instance2, dummy_valid_db_class)
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


class TestRegister:
    def test_register_valid_subclass(self, dummy_valid_db_class):
        """
        Test the registration of a valid subclass of BaseChatDatabase.
        """
        factory = ChatDatabaseFactory()
        factory.register("dummy_db", dummy_valid_db_class)
        assert "dummy_db" in factory._registry

    def test_register_invalid_subclass(self):
        """
        Ensure that attempting to register a class that does not inherit from
        BaseChatDatabase raises a TypeError.
        """
        factory = ChatDatabaseFactory()

        with pytest.raises(TypeError) as excinfo:
            factory.register("invalid_db", DummyInvalidDB)

        assert "must inherit from `BaseChatDatabase`" in str(excinfo.value)

    def test_register_with_aliases(self, dummy_valid_db_class):
        """
        Test the registration of a database type with multiple aliases.
        """
        factory = ChatDatabaseFactory()

        factory.register("dummy_db", dummy_valid_db_class, aliases=["dummy", "legacy_dummy"])

        assert "dummy_db" in factory._registry
        assert "dummy" in factory._aliases
        assert "legacy_dummy" in factory._aliases


class TestListAvailable:
    def test_list_available_returns_only_canonical_names(self, dummy_valid_db_class):
        """
        Verify that the list_available method returns only canonical names
        and excludes aliases.
        """
        factory = ChatDatabaseFactory()

        factory.register("dummy_db", dummy_valid_db_class, aliases=["dummy_alias"])

        # Aliases should not appear here
        available = factory.list_available()
        assert "dummy_db" in available
        assert "dummy_alias" not in available
