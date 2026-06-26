# MADA

Multi-agent orchestration system for coordinating complex engineering workflows using MCP (Model Context Protocol) servers.

## Overview

MADA coordinates multiple specialized agents to execute complex engineering workflows. Each agent connects to MCP servers that provide domain-specific tools for simulation, geometry, job management, and inverse design.


### Project Structure

```
mada/
├── src/mada/
│   ├── core/                 # Core orchestration logic
│   │   ├── config.py         # Configuration classes
│   │   ├── coordinator.py    # Agent coordination base class
│   │   ├── model_client.py   # LLM client factory
│   │   ├── orchestrator.py   # Main multi-agent orchestrator
│   │   └── utils.py          # Utility functions
│   ├── interfaces/           # User interfaces
│   │   ├── cli/              # Command-line interface
│   │   │   └── main.py       # CLI implementation
│   │   ├── openai_api/       # OpenAI-compatible HTTP API
│   │   │   └── main.py       # FastAPI launcher
│   │   └── gradio/           # Web interface
│   │       ├── interface.py  # Gradio interface components
│   │       ├── mcp_client_wrapper.py # MCP client session
│   │       └── main.py       # Gradio launcher
│   └── main.py               # Main entry point
├── configs/                  # Configuration files
│   └── orchestrator_config.json # Main configuration
├── examples/                 # Usage examples
│   └── basic_usage.py        # Basic usage example
└── pyproject.toml            # Package configuration
```

## Architecture

### Core Components

- **Orchestrator** (`src/mada/core/orchestrator.py`) - Main coordination engine
- **Agents** - Specialized agents for different domains
- **MCP Servers** - External tool providers
- **Interfaces** - CLI, Gradio web, and OpenAI-compatible API interfaces


## Installation


```bash
# Clone the repository
git clone https://github.com/llnl/mada.git
cd mada
```


```bash
# Create python virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (--pre required for agent-framework pre-release)
pip install --pre -e .
```


## Prerequisites

1. **MCP Servers**: Install and configure MADA Tools:
   ```bash
   # SSH access (preferred)
   git clone git@github.com:llnl/mada-tools.git

   # HTTPS access (alternative)
   git clone https://github.com/llnl/mada-tools.git
   ```

2. **Install**: Install mada-tools following README

## Configuration

### Main Configuration

MADA uses a single JSON configuration file that defines all agents and their MCP server connections. The configuration file specifies:

- **Model settings**: LLM model, API credentials, and connection details
- **Agent definitions**: Specialized agents and their MCP server connections
- **MCP server endpoints**: Transport types, URLs, and descriptions
- **Orchestration settings**: Timeouts, concurrency limits, coordination options
- **Interface configuration**: UI titles, descriptions, and placeholders

See [`configs/mada_config.json`](configs/mada_config.json) for the complete configuration.

### Environment Variables

MADA uses environment variables for API credentials, which are referenced in the configuration file. Set these environment variables:

```bash
# Required: LLM API configuration
export API_KEY="your-api-key"
export API_BASE_URL="https://api.openai.com/v1/responses"  # or your preferred endpoint
```

These are referenced in the config file using environment variable expansion:
```json
{
  "model": {
    "model": "gpt-4o",
    "api_key": "${API_KEY}",
    "base_url": "${API_BASE_URL:-https://livai-api.llnl.gov/v1}"
  }
}
```

### MCP Server Setup

Ensure MCP servers are running on the expected ports. From the mada-tools directory:

```bash
# Start individual servers
mada-mcp-flux --port 8001 &
mada-mcp-professor --port 8005 &

# Or start with specific config
mada-mcp-flux --config configs/development.json --port 8001 &
```

**Important**: Update the MCP server URLs in your `mada_config.json` to point to the correct endpoints where your servers are running. The default configuration assumes servers are running locally on ports 8001-8005.

## Usage

### CLI Interface

Launch the interactive command-line interface:

```bash
# Using the CLI entry point
mada-cli configs/mada_config.json

# Or using the module directly
python -m mada.interfaces.cli.main configs/mada_config.json
```

The CLI provides an interactive prompt where you can chat directly with the multi-agent team.

### Gradio Web Interface

Launch the web-based interface:

```bash
# Using the Gradio entry point
mada-gradio configs/mada_config.json

# Or using the module directly
python -m mada.interfaces.gradio.main configs/mada_config.json
```

The web interface will be available at `http://localhost:7860`.

### OpenAI-Compatible API

Launch the OpenAI-compatible API server:

```bash
# Using the dedicated entry point
mada-openai-api configs/orchestrator_config.json

# Or using the main entry point
mada openai-api configs/orchestrator_config.json
```

The API will be available at `http://localhost:8000/v1` and can be connected directly from Open WebUI or any other OpenAI-compatible client.

## Examples

### Basic Usage

Get started quickly with the basic example:

```bash
# Basic CLI interface using main configuration
python examples/basic_usage.py
```

## Example User Commands

Once MADA is running, you can interact with the multi-agent team using natural language commands. Below are some general commands you can try out:

**Ask about available agents:**
```
what are the agents
```

**Check job status:**
```
What is the status of my running jobs?
```

### Adding New Agents

1. Add agent configuration to `mada_config.json`:
   ```json
   {
     "agent_name": "new_agent",
     "description": "Description of what this agent does",
     "mcp_servers": ["server1", "server2"],
     "domain": "agent_domain",
     "instructions": "Custom system prompt for this agent"
   }
   ```
2. Ensure the MCP servers the agent needs are running
3. MADA will automatically create and manage the agent

### Adding New MCP Servers

1. Create the MCP server using the [`mada-tools` repository](https://github.com/llnl/mada-tools)
2. Add server definition to `mcp_servers` section in `mada_config.json`:
   ```json
   {
     "mcp_servers": {
       "new_server": {
         "transport": "streamable-http",
         "url": "http://localhost:8006",
         "description": "Description of the server"
       }
     }
   }
   ```
3. Update agent's `mcp_servers` list to include the new server
4. Start the server on the specified port


## Troubleshooting

### Common Issues

#### MCP Server Connection Failed

   - Verify servers are running on expected ports
   - Ensure transport type matches server configuration

#### API Key Issues

   - Verify `API_KEY` environment variable is set. If using LivAI's endpoint, request a LivAPI key.
   - Confirm `API_BASE_URL` is correct

#### Agent Initialization Failed

   - Check agent configuration in JSON file
   - Verify required MCP servers are available


#### "Address already in use" Error

If you see an error like:

```
bind [127.0.0.1]:7860: Address already in use
```

it means port 7860 on your local machine is already occupied by another process. To free it:

Identify which process is listening on 7860

```
lsof -iTCP:7860 -sTCP:LISTEN
```

Suppose the output shows PID 12345, then kill it:

```
kill -9 12345
```

After freeing the port, try setting up the SSH tunnel again.

#### Gradio timeout

Use Safari.


#### SSL certificate errors

If you encounter SSL errors (which sometimes surface as an "Agent error") update your certificate bundle:

```
python_cert_path=$(python -c "import certifi; print(certifi.where())")
cat /etc/ssl/certs/ca-bundle.crt >> "$python_cert_path"
```


## Related Projects

- **MADA Tools**: AI tooling for MADA
  - SSH: `git@github.com:llnl/mada-tools.git`
  - HTTPS: `https://github.com/llnl/mada-tools.git`
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework): Multi-agent orchestration framework
- [MCP](https://modelcontextprotocol.io/): Model Context Protocol specification


## Release

MADA is distributed under the terms of the Apache License (Version 2.0) WITH LLVM Exception.

All new contributions must be made under the Apache 2.0 License WITH LLVM Exception.

See [LICENSE](./LICENSE), [COPYRIGHT](./COPYRIGHT), and [NOTICE](./NOTICE) for details.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

LLNL-CODE-2019927
