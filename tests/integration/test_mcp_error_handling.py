# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Test that MCP server connection failures are handled gracefully
without blocking the system.
"""

import asyncio
import pytest
import logging

from mada.core.config import (
    OpenAIModelConfig,
    SQLiteConfig,
    AgentConfig,
    MCPServerConfig,
)
from mada.core.orchestrator import MADAOrchestrator
from mada.core.database import ChatSessionManager


@pytest.fixture
def model_config():
    """Create a test model config."""
    return OpenAIModelConfig(
        provider="openai",
        model="gpt-4",
        api_key="sk-test-fake-key-for-testing",  # Fake key for testing
        base_url="https://api.openai.com/v1",
    )


@pytest.fixture
def database_config():
    """Create a test database config using in-memory database."""
    import tempfile

    # Create a temporary file for the test database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    return SQLiteConfig(path=temp_db.name)


@pytest.fixture
def session_manager(database_config):
    """Create a test session manager."""
    manager = ChatSessionManager(database_config)
    manager.chat_db.init_db()  # Initialize the database tables
    manager.create_new_session("test_session")
    return manager


@pytest.fixture
def failing_mcp_server():
    """Create an MCP server config that will fail to connect."""
    return MCPServerConfig(
        transport="streamable-http",
        url="http://localhost:9999/nonexistent",  # This port should be closed
        description="Test MCP server that will fail to connect",
    )


@pytest.fixture
def agent_with_failing_mcp(failing_mcp_server):
    """Create an agent config that uses a failing MCP server."""
    return AgentConfig(
        agent_name="TestAgent",
        description="Test agent with failing MCP server",
        instructions="You are a test agent.",
        mcp_servers=["failing_server"],
    )


@pytest.mark.asyncio
async def test_failed_mcp_connection_does_not_block(
    model_config,
    database_config,
    session_manager,
    failing_mcp_server,
    agent_with_failing_mcp,
    caplog,
):
    """
    Test that a failed MCP connection doesn't block the system.

    This test verifies that when an MCP server fails to connect,
    the orchestrator continues to function and doesn't leave
    any hanging processes or blocked I/O.
    """
    caplog.set_level(logging.INFO)

    mcp_servers = {"failing_server": failing_mcp_server}

    async with MADAOrchestrator(
        model_config=model_config, session_manager=session_manager
    ) as orchestrator:
        # Initialize orchestrator with failing MCP server
        status, tools = await orchestrator.initialize_orchestrator(
            [agent_with_failing_mcp], mcp_servers
        )

        # Verify that initialization completed (even with failed server)
        assert status is not None
        assert "Connection Successful" in status or "WARNING" in status

        # Verify that the error was logged
        assert any(
            "Cannot connect to MCP server" in record.message
            for record in caplog.records
        )

        # Verify that no tools were registered (since the server failed)
        assert len(tools) == 0

        # Test that async operations don't block
        # This simulates being able to get user input
        async def simulated_async_operation():
            await asyncio.sleep(0.1)
            return "operation completed"

        try:
            result = await asyncio.wait_for(simulated_async_operation(), timeout=2.0)
            assert result == "operation completed"
        except asyncio.TimeoutError:
            pytest.fail("Async operation timed out - system appears to be blocked!")


@pytest.mark.asyncio
async def test_multiple_failed_mcp_connections(model_config, session_manager):
    """
    Test that multiple failed MCP connections don't block the system.
    """

    # Create multiple failing servers
    mcp_servers = {
        "failing_server_1": MCPServerConfig(
            transport="streamable-http",
            url="http://localhost:9999/server1",
            description="Test server 1",
        ),
        "failing_server_2": MCPServerConfig(
            transport="streamable-http",
            url="http://localhost:9998/server2",
            description="Test server 2",
        ),
    }

    agent_config = AgentConfig(
        agent_name="TestAgent",
        description="Test agent with multiple failing MCP servers",
        instructions="You are a test agent.",
        mcp_servers=["failing_server_1", "failing_server_2"],
    )

    async with MADAOrchestrator(
        model_config=model_config, session_manager=session_manager
    ) as orchestrator:
        status, tools = await orchestrator.initialize_orchestrator(
            [agent_config], mcp_servers
        )

        # Verify initialization completed
        assert status is not None

        # No tools should be registered
        assert len(tools) == 0

        # Verify system is not blocked
        async def quick_check():
            return True

        result = await asyncio.wait_for(quick_check(), timeout=1.0)
        assert result is True


@pytest.mark.asyncio
async def test_mixed_successful_and_failed_mcp_connections(
    model_config, session_manager
):
    """
    Test that when some MCP servers succeed and others fail,
    the system continues with the successful connections.
    """

    # Note: This test assumes we can't easily create a working MCP server
    # So we'll just test that the system doesn't block with multiple failures
    mcp_servers = {
        "failing_server": MCPServerConfig(
            transport="streamable-http",
            url="http://localhost:9999/fail",
            description="Test failing server",
        ),
    }

    agent_config = AgentConfig(
        agent_name="TestAgent",
        description="Test agent",
        instructions="You are a test agent.",
        mcp_servers=["failing_server"],
    )

    async with MADAOrchestrator(
        model_config=model_config, session_manager=session_manager
    ) as orchestrator:
        # This should not raise an exception or block
        status, tools = await orchestrator.initialize_orchestrator(
            [agent_config], mcp_servers
        )

        assert status is not None
        # System should still be responsive
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_no_async_generator_warnings_on_cleanup(model_config, session_manager):
    """
    Test that failing MCP connections don't trigger async generator warnings during cleanup.

    This test ensures that when an MCP server fails to connect, the async generators
    are properly cleaned up without triggering RuntimeWarnings.
    """
    import warnings

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        mcp_servers = {
            "failing_server": MCPServerConfig(
                transport="streamable-http",
                url="http://localhost:9999/test",
                description="Test failing server",
            )
        }

        agent_config = AgentConfig(
            agent_name="TestAgent",
            description="Test agent",
            instructions="You are a test agent.",
            mcp_servers=["failing_server"],
        )

        async with MADAOrchestrator(
            model_config=model_config, session_manager=session_manager
        ) as orchestrator:
            status, tools = await orchestrator.initialize_orchestrator(
                [agent_config], mcp_servers
            )
            assert status is not None

        # Check for async generator warnings
        async_warnings = [
            warning
            for warning in w
            if "async_generator" in str(warning.message).lower()
        ]

        # There should be no async generator warnings
        assert len(async_warnings) == 0, (
            f"Found {len(async_warnings)} async generator warning(s): {[str(w.message) for w in async_warnings]}"
        )
