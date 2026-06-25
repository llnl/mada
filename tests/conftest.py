# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Pytest configuration file for custom options and fixtures.

This file defines custom command-line options, setup hooks, and global fixtures
used across the test suite. It includes configurations for environment variable
checks, temporary directories, and PostgreSQL integration tests.
"""

import os
from pathlib import Path
from time import sleep

import pytest

# from jeds.postgresql import PostgreSQL
from testcontainers.postgres import PostgresContainer

from mada.core.config import PostgreSQLConfig
from mada.core.database import BaseChatDatabase

TESTS_DIR = Path(__file__).parent
E2E_DIR = TESTS_DIR / "e2e"
INTEGRATION_DIR = TESTS_DIR / "integration"
UNIT_DIR = TESTS_DIR / "unit"


################################
# Custom Pytest Configurations #
################################


def pytest_collection_modifyitems(config, items):
    """
    Modifies pytest items prior to running the tests. In our case,
    this is specifically marking tests appropriately.
    """
    # Resolve test directories
    e2e_dir = E2E_DIR.resolve()
    integration_dir = INTEGRATION_DIR.resolve()
    unit_dir = UNIT_DIR.resolve()

    # Loop through and mark tests appropriately
    for item in items:
        try:
            path = item.path.resolve()
        except Exception:
            continue

        # Mark end-to-end tests
        if e2e_dir in path.parents or path == e2e_dir:
            item.add_marker(pytest.mark.e2e)

        # Mark integration tests
        if integration_dir in path.parents or path == integration_dir:
            item.add_marker(pytest.mark.integration)

        # Mark unit tests
        if unit_dir in path.parents or path == unit_dir:
            item.add_marker(pytest.mark.unit)


def pytest_addoption(parser):
    """
    Add custom command-line options to pytest.

    Args:
        parser: The pytest command-line option parser.
    """
    parser.addoption(
        "--include-allocation-required",
        action="store_true",
        default=False,
        help="Run tests marked with 'allocation_required'.",
    )


def pytest_runtest_setup(item):
    """
    Perform setup checks before running a test.

    This hook checks for the presence of required environment variables for tests
    marked with `@pytest.mark.requires_env`. If any required environment variables
    are missing, the test is skipped.

    Args:
        item: The pytest test item being set up.
    """
    requires_env_mark = item.get_closest_marker("requires_env")
    if requires_env_mark:
        # marker can be used as @pytest.mark.requires_env("VAR")
        # or @pytest.mark.requires_env("VAR", "OTHER_VAR")
        env_vars = requires_env_mark.args
        missing = [var for var in env_vars if os.getenv(var) is None]

        if missing:
            pytest.skip(
                f"Skipping {item.name} because required env var(s) not set: {', '.join(missing)}"
            )


###################
# Global Fixtures #
###################


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory: pytest.TempdirFactory) -> Path:
    """
    Creates a temporary directory for the entire test session.

    Pytest's `tmp_path` is function scoped so you can't use it for fixtures
    with a longer scope like "session". This fixture provides a workaround.

    Args:
        tmp_path_factory: A pytest fixture for creating temporary directories.

    Returns:
        Path: The path to the temporary directory created for the session.
    """
    # Create a temporary directory named 'session_tmp_' within the pytest temporary root
    tmp_dir = tmp_path_factory.mktemp("session_tmp_")
    return tmp_dir


@pytest.fixture(scope="session")
def postgres_connection(session_tmp_path: Path, request: pytest.FixtureRequest):
    """
    Fixture for establishing a connection to PostgreSQL for integration tests.

    An allocation IS REQUIRED to run any tests that use this fixture. Use the
    @pytest.mark.allocation_required decorator to mark such tests. There is a
    check at the beginning of this fixture that will skip the test in case the
    decorator is not present.

    This utilizes the JEDS library to spin up a PostgreSQL server in a container.
    Once all tests finish running, the container is stopped.

    Args:
        session_tmp_path: A fixture for the temporary directory for the entire test session.
        request: A fixture providing information about the requesting test function.
    """
    # Ensure allocation-required tests are included if this fixture is used
    include_alloc_reqd_tests = request.config.getoption(
        "--include-allocation-required", False
    )
    if not include_alloc_reqd_tests:
        pytest.skip(
            "Test requires allocation; use --include-allocation-required when running tests to enable this test."
        )

    # Spin up container using testcontainer library
    postgres_client = PostgresContainer("postgres:16")
    postgres_client.start()

    postgres_config = PostgreSQLConfig(
        host=postgres_client.get_container_host_ip(),
        port=postgres_client.get_exposed_port(5432),
        database=postgres_client.dbname,
        user=postgres_client.username,
        password=postgres_client.password,
        sslmode="disable",
    )

    sleep(5)  # Give some time for the client to spin up

    yield postgres_config

    # Stop container after tests have ran
    postgres_client.stop()


@pytest.fixture
def dummy_valid_db_class() -> "DummyValidDB":  # noqa: F821
    """
    A fixture that provides a valid database class that inherits from
    `BaseChatDatabase`.

    This fixture is used across unit, integration, and e2e tests which
    is why it's located here.

    Returns:
        The class representation for `DummyValidDB` so mocking can take
        place in tests.
    """

    class DummyValidDB(BaseChatDatabase):
        """
        Simple valid `BaseChatDatabase` subclass for testing registration and creation.
        """

        def __init__(self, db_config):
            self.init_called = False
            super().__init__(db_config)

        def init_db(self):
            self.init_called = True

        def add_message(self, session_id, role, content, timestamp):
            pass

        def create_session(self, session_id):
            pass

        def load_session(self, session_id):
            return []

        def list_sessions(self):
            return []

        def delete_session(self, session_id):
            pass

        def flush_database(self, confirm: bool = True):
            pass

    return DummyValidDB
