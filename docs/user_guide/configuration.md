# Configuring MADA

In MADA there are three required configuration options:

- [Agent Configuration](#agent-configuration)
- [MCP Server Configuration](#mcp-server-configuration)
- [Model Configuration](#model-configuration)

Additionally, there are optional configuration options:

- [Database Configuration](#optional-database-configuration)
- [Gradio Interface Configuration](#optional-gradio-interface-configuration)
- [Orchestration Configuration](#optional-orchestration-configuration)

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

When MADA starts, it reads the agent configuration and loads each agent by executing the specified server path Python file. The functions decorated with `@mcp.tool` within these scripts become the MCP tools that MADA can call during a session. Agents communicate within a group chat, and the [planning agent](#the-planning-agent) coordinates which helper agent should handle each user request.

Each agent's configuration allows you to:

- Assign clear roles and responsibilities to agents.
- Extend MADA's capabilities by adding new agents with specialized MCP tools.
- Customize agent behavior and context using system messages.

### The Planning Agent

MADA includes a special agent called the **planning agent**. This agent is automatically added to every group chat session and **does not need to be defined in your agent configuration**. The planning agent's role is to interpret user input, determine which helper agent is best suited to handle each request, and coordinate task delegation.

The default, base instructions for the planning agent are:

```python
--8<-- "src/mada/core/orchestrator.py:200:203"
```

If you'd like to modify these instructions, add an agent entry to the `"agents"` list in your configuration file with `"agent_name": "PlanningAgent"`. For example:

```json
"agents": [
    {
        "agent_name": "PlanningAgent",
        "instructions": "You are the planning agent for MADA. Use your specialist agents to help you accomplish tasks. These agents can be used as tools."
    }
]
```

!!! note

    The planning agent will always include core instructions that cannot be modified. These instructions describe each specialist agent you define, and include the following guidelines:

    ```python
    --8<-- "src/mada/core/orchestrator.py:212:216"
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

MADA now exposes the orchestration pattern as an explicit top-level config block. In
this release, the current planner-plus-specialist flow is named `agent-as-tool` and
it is the only supported mode.

### Fields

| Field Name     | Description                                                                 | Required? | Default           |
| -------------- | --------------------------------------------------------------------------- | --------- | ----------------- |
| `mode`         | Internal orchestration mode. Only `agent-as-tool` is supported right now.   | No        | `agent-as-tool`   |
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
