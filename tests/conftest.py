# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Pytest configuration file for custom options and fixtures.

This file defines custom command-line options, setup hooks, and global fixtures
used across the test suite. It includes configurations for environment variable
checks, temporary directories, and PostgreSQL integration tests.
"""

import os
import pytest
from pathlib import Path
from time import sleep

from jeds.postgresql import PostgreSQL

from mada.core.config import PostgreSQLConfig


################################
# Custom Pytest Configurations #
################################


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
                f"Skipping {item.name} because required env var(s) not set: "
                f"{', '.join(missing)}"
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

    # Spin up container using JEDS
    postgres_client = PostgreSQL()
    postgres_client.run()

    postgres_config = PostgreSQLConfig(
        host=postgres_client.conf["service-host"],
        port=postgres_client.conf["service-port"],
        database=postgres_client.conf["database-name"],
        user=postgres_client.conf["database-user"],
        password=postgres_client.conf["database-password"],
    )

    sleep(5)  # Give some time for the client to spin up

    yield postgres_config

    # Stop container after tests have ran
    postgres_client.stop()
