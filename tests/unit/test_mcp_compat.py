import mada

from mcp.shared.exceptions import McpError
from mcp.shared import exceptions as mcp_exceptions
from mcp.types import InitializeResult, Implementation, ServerCapabilities, Tool


def test_mcp_initialize_result_exposes_legacy_protocol_version_alias():
    assert mada.__version__

    result_kwargs = {
        "capabilities": ServerCapabilities(),
    }
    if "protocol_version" in InitializeResult.model_fields:
        result_kwargs["protocol_version"] = "2025-06-18"
        result_kwargs["server_info"] = Implementation(
            name="test-server", version="1.0.0"
        )
    else:
        result_kwargs["protocolVersion"] = "2025-06-18"
        result_kwargs["serverInfo"] = Implementation(
            name="test-server", version="1.0.0"
        )

    result = InitializeResult.model_construct(
        **result_kwargs,
    )

    assert result.protocolVersion == "2025-06-18"


def test_mcp_shared_exceptions_exposes_legacy_mcp_error_alias():
    assert mcp_exceptions.McpError is McpError


def test_mcp_tool_exposes_legacy_schema_aliases():
    tool_kwargs = {
        "name": "demo",
    }
    if "input_schema" in Tool.model_fields:
        tool_kwargs["input_schema"] = {"type": "object"}
        tool_kwargs["output_schema"] = None
    else:
        tool_kwargs["inputSchema"] = {"type": "object"}
        tool_kwargs["outputSchema"] = None

    tool = Tool.model_construct(**tool_kwargs)

    assert tool.inputSchema == {"type": "object"}
    assert tool.outputSchema is None
