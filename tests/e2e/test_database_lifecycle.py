"""
End-to-end tests for the database functionality.
"""

from mada.core.config import DatabaseConfig
from mada.core.database.db_factory import ChatDatabaseFactory


def test_e2e_factory_life_cycle(dummy_valid_db_class):
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
    factory.register("dummy", dummy_valid_db_class, aliases=["dummy_db"])

    # Check list_available again
    new_available = factory.list_available()
    assert len(new_available) == 3
    assert "dummy" in new_available

    # Check that the new alias was added
    assert len(factory._aliases) == 3
    assert "dummy_db" in factory._aliases

    # Create an instance
    instance = factory.create("dummy", DatabaseConfig)
    assert isinstance(instance, dummy_valid_db_class)
    assert instance.init_called
