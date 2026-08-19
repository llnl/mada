# Configuring MADA

In MADA there are three required configuration options:

- [Agent Configuration](#agent-configuration)
- [MCP Server Configuration](#mcp-server-configuration)
- [Model Configuration](#model-configuration)

Additionally, there are optional configuration options:

- [Database Configuration](#optional-database-configuration)
- [Gradio Interface Configuration](#optional-gradio-interface-configuration)
- [Orchestration Configuration](#optional-orchestration-configuration)
- [A2A Configuration](#optional-a2a-configuration)

## Agent Configuration

Agent configuration defines the autonomous agents that MADA will orchestrate in your multi-agent system. Each agent acts as an independent process capable of performing specific tasks, interpreting instructions, or providing specialized services. Proper agent configuration ensures that MADA can delegate tasks effectively and maintain clear communication between agents.

### Fields

| Field Name        | Description                                               | Required? | Default   |
| ----------------- | --------------------------------------------------------- | --------- | --------- |
| `agent_name`      | Unique identifier for the agent.                          | Yes       | N/A       |
| `description`     | Human-readable description of the agent's purpose.        | Yes       | N/A       |
| `mcp_servers`     | List of MCP server names this agent should connect to.    | Yes       | N/A       |
| `domain`          | The domain or specialization of the agent.                | No        | None      |
| `instructions`    | System prompt to initialize the agent's behavior.         | No        | None      |
| `server_path`     | **[Set for deprecation]** The file path to a Python script containing the agent's MCP tool definitions. The path can be absolute or relative to your project directory. | Yes (unless `mcp_servers` is given) | N/A     |

!!! note

    The names listed in the `mcp_servers` field should match existing names in the ["mcp servers"](#mcp-server-configuration) section of your configuration.

### Example

```json
"agents": [
    {
        "agent_name": "JobManagementAgent",
        "description": "Generates parameter samples and executes computational workflows",
        "domain": "job_management",
        "mcp_servers": ["flux"],
        "instructions": [
            "You are a Job Management Agent specialized in parameter generation and job execution.",
            "You have access to Flux for executing simulation runs."
            "You handle the complete workflow from parameter generation through job execution."
        ]
    },
    {
        "agent_name": "InverseDesignAgent",
        "description": "Analyzes simulation results and calculates Quality of Interest (QoI)",
        "domain": "inverse_design",
        "mcp_servers": ["professor"],
        "incstructions": [
            "You are an Inverse Design Agent specialized in analyzing simulation results and calculating Quality of Interest (QoI).",
            "You have access to Professor for surrogate modeling.",
            "Your responsibilities include analyzing completed simulation results, calculating QoI metrics, identifying best designs, and building surrogate models for optimization.",
            "You do NOT generate parameters or execute jobs - that is handled by the JobManagementAgent.",
        ]
    }
],
```

### How Agent Configuration Works

When MADA starts, it reads the agent configuration and creates the selected specialist agents. Each specialist can connect to named MCP servers from the `mcp_servers` block, or use the legacy `server_path` setting for a directly launched stdio MCP server.

MADA then runs the configured orchestration mode:

- `agent-as-tool`: Exposes specialists as tools to a visible planning agent.
- `magentic`: Coordinates specialists in a peer group chat through a hidden manager.

Each agent's configuration allows you to:

- Assign clear roles and responsibilities to agents.
- Extend MADA's capabilities by adding new agents with specialized MCP tools.
- Customize agent behavior and context using system messages.

### The Planning Agent

MADA includes a special coordinator called the **planning agent**. This coordinator is automatically added to orchestration and **does not need to be defined in your agent configuration**. In `agent-as-tool` mode it acts as the visible planner that delegates to specialists as tools. In `magentic` mode the same optional `PlanningAgent` configuration is reused as the hidden Magentic manager that coordinates the peer specialist group chat.

The default, base instructions for the `agent-as-tool` planning agent are:

```
You are a planning agent for the MADA multi-agent system.

Your specialist agents (available as tools) can be delegated tasks.
```

If you'd like to modify the coordinator instructions for either mode, add an agent entry to the `"agents"` list in your configuration file with `"agent_name": "PlanningAgent"`. For example:

```json
"agents": [
    {
        "agent_name": "PlanningAgent",
        "instructions": "You are the planning agent for MADA. Use your specialist agents to help you accomplish tasks. These agents can be used as tools."
    }
]
```

!!! note

    `PlanningAgent` is never part of `orchestration.participants`. In `magentic` mode it customizes the hidden manager instead of joining the visible specialist list.

    The `agent-as-tool` planning agent will always include core instructions that cannot be modified. These instructions describe each specialist agent you define, and include the following guidelines:

    ```
    Guidelines:
    - Delegate to specialist agents when the request matches their expertise
    - Delegate to remote A2A agents when their descriptions match the request
    - Answer directly only for questions about the system itself
    - Avoid infinite loops between agents
    - After receiving results, synthesize and respond to the user
    ```

## MCP Server Configuration

!!! warning

    The MCP servers that you define in this configuration must be spun up *prior* to launching MADA. The configuration here just points to an already existing server.

In order for [agents](#agent-configuration) to utilize MCP tool calls, you must point them to MCP servers where these tool calls exist. This setup supports flexible deployment scenarios, allowing agents to connect to remote servers over HTTP or launch local MCP server processes as needed.

A typical agent configuration might include multiple MCP servers, each with different transport methods or tool sets, allowing for scalable and modular tool call management. Proper configuration ensures reliable communication and access to the full range of MCP tools available in your environment.

### Fields

| Field Name    | Description                                                   | Required? | Default   |
| ------------- | ------------------------------------------------------------- | --------- | --------- |
| `transport`   | Transport method ('streamable-http' or 'stdio').              | Yes       | N/A       |
| `url`         | URL for when `transport` is set to 'streamable-http'.         | No        | None      |
| `command`     | Command to launch server when `transport` is set to `stdio`.  | No        | None      |
| `description` | Human-readable description of the server.                     | No        | None      |
| `verify`      | TLS verification for `streamable-http`. Use `true` for env/system trust, `false` to disable verification, or a CA bundle path. | No | `true` |

### Example

```json
"mcp_servers": {
    "flux": {
        "transport": "streamable-http",
        "url": "http://localhost:8001/mcp",
        "description": "Flux workload manager for job execution",
        "verify": true
    },
    "merlin": {
        "transport": "streamable-http",
        "url": "http://localhost:8002/mcp",
        "description": "Merlin workflow orchestration"
    },
    "professor": {
        "transport": "streamable-http",
        "url": "http://localhost:8005/mcp",
        "description": "Professor surrogate modeling"
    }
}
```

## Model Configuration

Model configuration tells MADA which language model to use for agent conversations, and how to connect to the selected provider. Depending on the provider, this may include values such as the model name, API key, base URL, region, or other authentication settings.

For OpenAI-compatible providers, `verify` controls TLS verification. Use
`true` to keep MADA's default env/system trust resolution, `false` to disable
verification, or a CA bundle path string.

MADA supports multiple providers for model configuration. Each provider has its own required and optional fields. Refer to the following documentation for provider-specific details:

- [OpenAI Model Configuration](./models/openai.md)
- [LivAI Model Configuration](./models/livai.md)
- [AWS Bedrock Model Configuration](./models/bedrock.md)

### Environment Variables

You can set your API key and base URL using environment variables. If you do not specify these fields directly in your configuration, MADA will look for the following environment variables:

- **API_KEY:** Used as the default for the `api_key` field if not provided in the config.
- **API_BASE_URL:** Used as the default for the  `base_url` field if not provided in the config.

This allows you to keep sensitive information like API keys out of your configuration files.

**Example (Linux/macOS):**

```bash
export API_KEY="sk-xxxx..."
export API_BASE_URL="https://api.openai.com/v1/responses"
```

### Example

```json
"model": {
    "model": "o3",
    "api_key": "${API_KEY}",
    "base_url": "${API_BASE_URL:-https://api.openai.com/v1/responses}",
    "verify": true,
    "extra": {
        "temperature": 0.7,
        "max_tokens": 2048
    }
}
```

### How Model Configuration Works

When you start MADA, it reads your model configuration to connect to the right language model (such as OpenAI's GPT-4 or LivAI). MADA uses the information you provide—like the model name, API key, and any extra options—to set up everything needed for agents to communicate with the model.

**What happens behind the scenes:**

- MADA matches your chosen model name to the correct backend.
- It applies your settings (like temperature or max tokens) to customize how the model responds.
- It securely connects using your API key and the specified endpoint.
- Once set up, all agent conversations and completions use this model automatically.

**You only need to:**

- Fill out the model configuration section with your preferred model and settings.
- Make sure your API key and endpoint are correct.

## (Optional) Orchestration Configuration

MADA now exposes the orchestration pattern as an explicit top-level config block. Two
internal modes are supported:

- `agent-as-tool`: the existing planner-plus-specialist flow
- `magentic`: a peer specialist group chat coordinated by a hidden manager

### Fields

| Field Name     | Description                                                                 | Required? | Default           |
| -------------- | --------------------------------------------------------------------------- | --------- | ----------------- |
| `mode`         | Internal orchestration mode. Supported values are `agent-as-tool` and `magentic`. | No        | `agent-as-tool`   |
| `participants` | Optional list of specialist agent names to include. `PlanningAgent` is excluded. | No        | All non-`PlanningAgent` agents |

### Example

```json
"orchestration": {
    "mode": "agent-as-tool",
    "participants": ["JobManagementAgent", "InverseDesignAgent", "GeometryAgent"]
}
```

If `participants` is omitted, MADA includes every configured agent except
`PlanningAgent`.

In `magentic` mode, the `participants` list still refers only to specialist
agents. If a `PlanningAgent` config is present, its instructions customize the
hidden Magentic manager. Otherwise MADA uses its built-in manager instructions.

### Magentic Example

```json
"orchestration": {
    "mode": "magentic",
    "participants": ["JobManagementAgent", "InverseDesignAgent"]
}
```

## (Optional) A2A Configuration

MADA can participate in Agent-to-Agent (A2A) workflows in two directions:

- `a2a.agents` is the client-side configuration. It lists remote A2A agents
  that MADA can call as tools from the orchestrator.
- `a2a.self` is the server-side configuration. It describes MADA's own A2A
  identity when you run MADA with `mada-a2a` so other agents can discover and
  call MADA.

These settings do not replace CLI or Gradio. CLI and Gradio are interactive
interfaces for users. A2A mode starts an HTTP service so other A2A agents can
discover MADA and delegate tasks to it.

In code, the same split is reflected by the modules: `mada.core.a2a_client`
handles outbound calls from MADA to remote A2A agents, while
`mada.interfaces.a2a.main` exposes MADA itself as an inbound A2A service.

### Remote A2A Agents

Use `a2a.agents` when the MADA orchestrator should delegate work to other A2A
agents. Each configured remote agent is exposed to the planning agent as a tool,
using the remote agent card for routing context. If a configured remote A2A
agent card cannot be fetched, MADA starts without that remote tool and reports
a warning in the startup status.

#### Fields

| Field Name    | Description                                                                 | Required? | Default |
| ------------- | --------------------------------------------------------------------------- | --------- | ------- |
| `url`         | JSON-RPC endpoint for the remote A2A agent.                                  | Yes       | N/A     |
| `card_url`    | Explicit URL for the remote agent card. If omitted, MADA tries standard A2A card paths derived from `url`. | No | None |
| `timeout`     | HTTP timeout in seconds for calls to the remote agent.                       | No        | `180`   |
| `api_key`     | Optional API key sent as `x-api-key`.                                        | No        | None    |
| `headers`     | Additional HTTP headers to send to the remote agent.                         | No        | `{}`    |

#### Example

```json
"a2a": {
  "agents": {
    "LangChainAgent": {
      "url": "http://localhost:9111/",
      "card_url": "http://localhost:9111/.well-known/agent-card.json"
    },
    "GoogleADKAgent": {
      "url": "http://localhost:9112/",
      "card_url": "http://localhost:9112/.well-known/agent-card.json"
    }
  }
}
```

The example MCP servers are used inside the remote A2A agents, not as local
MADA MCP servers. The MADA orchestrator should report `0 MCP Servers` and `2
remote A2A agents` for this config. The remote agent card endpoints must be
reachable so MADA can discover each remote agent's skills. Install optional
dependencies and launch the MCP servers and A2A agents with the config path:

```bash
pip install -e ".[a2a-examples]"
python examples/a2a/a2a_table_mcp_server.py --port 9101
python examples/a2a/a2a_average_mcp_server.py --port 9102
python examples/a2a/a2a_langchain_agent.py --port 9111 --config configs/example_a2a_agents.json --mcp-url http://localhost:9101/mcp
python examples/a2a/a2a_google_adk_agent.py --port 9112 --config configs/example_a2a_agents.json --mcp-url http://localhost:9102/mcp
```

Use each example agent's `--model`, `--api-key`, and `--base-url` flags when
you want that remote agent to use a different model endpoint from MADA. The
Google ADK example also accepts `--provider`.

### MADA's A2A Agent Card

Use `a2a.self` when you want MADA itself to be discoverable by other A2A agents.
This block is used by `mada-a2a` and `mada a2a`; it is not used by CLI or Gradio
mode. These are commands within this repo and not actual MADA repos like `mada-tools`.

The `card_path` value points to a standalone A2A agent card JSON file. Relative
paths are resolved relative to the configuration file. When the card is served,
MADA overwrites the card's `url` field with the runtime public URL from
`a2a.self.url` or `--public-url`, and advertises A2A protocol `1.0.0`.

#### Fields

| Field Name  | Description                                                                 | Required? | Default |
| ----------- | --------------------------------------------------------------------------- | --------- | ------- |
| `card_path` | Path to MADA's standalone A2A agent card JSON file.                          | No        | None    |
| `url`       | Public URL advertised in the served agent card.                              | No        | Runtime host and port |
| `name`      | Name used by the generated card fallback when no `card_path` is supplied.    | No        | `MADA`  |
| `description` | Description used by the generated card fallback when no `card_path` is supplied. | No | `MADA multi-agent orchestration service` |
| `skills`    | Skills used by the generated card fallback when no `card_path` is supplied.  | No        | Derived from configured agents |

#### Example

```json
"a2a": {
  "self": {
    "card_path": "agent_cards/mada_orchestrator_card.json",
    "url": "http://localhost:9120",
  }
}
```

Launch MADA as an A2A service with:

```bash
mada-a2a --port 9120 configs/example_a2a_agents.json
```

Other A2A agents can then discover MADA at:

```text
http://localhost:9120/.well-known/agent-card.json
```

## (Optional) Database Configuration

If you want to customize your database settings, you can set this in the configuration file. There are two database options:

- [SQLite](#sqlite-configuration) (default)
- [PostgreSQL](#postgresql-configuration)

There is more details on configuration for both types below.

### SQLite Configuration

SQLite is the default configuration in MADA. If no configuration settings are provided, MADA will default to using a SQLite file that is created and stored at `~/.mada/chat_history.db`. This file path can be changed using these configuration settings.

#### Fields

| Field Name    | Description                                               | Required? | Default                   |
| ------------- | --------------------------------------------------------- | --------- | ------------------------- |
| `type`        | The type of database (either `sqlite` or `postgresql`)    | Yes       | `sqlite`                  |
| `path`        | The path to the SQLite database file to use               | Yes       | `~/.mada/chat_history.db` |

#### Example

```json
"database": {
    "type": "sqlite",
    "path": "/custom/sqlite/path.db
}
```

### PostgreSQL Configuration

If you would like to use something other than SQLite, MADA also provides support for PostgreSQL databases. Here, you'll need to fill out *either* the `connection_string` field *or* every other field listed in the table below.

#### Fields

| Field Name            | Description                                                           | Required? | Default   |
| --------------------- | --------------------------------------------------------------------- | --------- | --------- |
| `type`                | The type of database (either `sqlite` or `postgresql`)                | Yes       | `sqlite`  |
| `connection_string`   | A PostgreSQL connection string                                        | No        | None      |
| `host`                | The service host. In LaunchIT, this is the 'service-host'.            | No        | None      |
| `port`                | The service port. In LaunchIT, this is the 'service-port'.            | No        | None      |
| `database`            | The database name. In LaunchIT, this is the 'database-name'.          | No        | None      |
| `user`                | The database user. In LaunchIT, this is the 'database-user'.          | No        | None      |
| *`password`           | The database password. In LaunchIT, this is the 'database-password'.  | No        | None      |

*You may want to store the password as an environment variable and then pass in a reference to that environment variable.

#### Examples

Below is an example configuration that uses a connection string:

```json
"database": {
    "type": "postgresql",
    "connection_string": "postgresql://username:password@localhost:5432/dbname"
}
```

Or, you can instead provide each setting individually:

```json
"database": {
    "type": "postgresql",
    "host": "localhost",
    "port": 5432,
    "database": "dbname",
    "user": "username",
    "password": "password"
}
```

## (Optional) Gradio Interface Configuration

If you are running MADA in [Gradio mode](./usage/gradio.md), you can customize the appearance and behavior of the web interface using the Gradio interface configuration. This includes the interface title, description, chat input placeholder, and layout options.

### Fields

| Field Name                    | Description                                           | Required? | Default                       |
| ----------------------------- | ----------------------------------------------------- | --------- | ----------------------------- |
| `title`                       | Title displayed at the top of the interface.          | Yes       | N/A                           |
| `description`                 | Brief description shown below the title.              | Yes       | N/A                           |
| `chat_placeholder`            | Placeholder text in the chat input field.             | No        | `"Type your message here"`    |
| `port`                        | Port to run the Gradio interface on.                  | No        | 7860                          |
| `share`                       | Whether to create a public sharing link.              | No        | `False`                       |
| `connection_accordion_open`   | Whether the connection accordion is open by default.  | No        | `False`                       |
| `connection_accordion_label`  | Label for the connection accordion section.           | No        | `"Connect to MCP Servers"`    |
| `accordion_kwargs`            | Additional customization for the accordion component. | No        | None                          |
| `dataframe_kwargs`            | Customization for the agents `DataFrame`.             | No        | None                          |
| `chat_interface_kwargs`       | Customization for the chat interface.                 | No        | None                          |

### Example

```json
"interface": {
    "title": "Multi-Agent Chat",
    "description": "Interact with multiple agents in a single workspace.",
    "chat_placeholder": "Type your message here",
    "connection_accordion_open": true,
    "connection_accordion_label": "Connect to MCP Servers",
    "accordion_kwargs": {
        "style": "background-color: #f5f5f5"
    },
    "dataframe_kwargs": {
        "page_size": 10
    },
    "chat_interface_kwargs": {
        "theme": "dark"
    }
}
```

## Full Example Configuration File

The below example combines all configuration options discussed in the previous sections of this page.

```json
--8<-- "configs/mada_config.json"
```
