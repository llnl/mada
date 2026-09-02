# MADA Orchestrator Codebase Architecture

## Top-Level Repository Structure

At the top-level of the repository you'll find:

- Project configuration files (pyproject.toml, mkdocs.yaml, etc.)
- Directories to the rest of the project, including:
    - **Configuration:** Example configuration files
    - **Documentation:** Containing files for the documentation
    - **Examples:** Example agents for using the orchestrator
    - **Source Code:** All of the source code for the MADA Orchestrator project
    - **Tests:** All of the test files for testing the source code

Below is a visual representation of the top-level repository:

```bash
mada/
├── src/
├── examples/
├── configs/
├── docs/
├── .gitignore
├── pyproject.toml
├── mkdocs.yaml
└── README.md
```

## Source Code Structure

The source code for the MADA Orchestrator repository is designed with the goal of modularity and extensibility. There are multiple directories to keep components organized:

- **Core:** Implements the shared orchestrator runtime, mode-specific orchestration strategies, coordinator primitives, configuration, and persistence.
- **Common:** Contains shared utilities and exception handling.
- **Interfaces:** Provides CLI and Gradio interfaces for user interaction.
- **Config:** Manages configuration files and models.
- **Database:** Handles database interactions, including support for SQLite and PostgreSQL.

Below is a visual representation of the source code structure:

```bash
src/
└── mada/
    ├── core/                     # Core orchestration logic
    │   ├── config/                 # Configuration management
    │   ├── database/               # Database operations
    │   ├── orchestration/          # Mode-specific strategy implementations
    │   ├── coordinator.py          # Task coordination
    │   └── orchestrator.py         # Shared orchestration runtime
    ├── common/                   # Shared utilities
    ├── interfaces/               # User interfaces
    │   ├── cli/                    # Command-line interface
    │   └── gradio/                 # Web-based interface
```

## Orchestration Modes

`src/mada/core/orchestration/` contains the internal strategy boundary for MADA orchestration, while `src/mada/core/orchestrator.py` owns shared state, MCP connection primitives, agent creation helpers, and session persistence. The current implementation supports two modes:

- `agent-as-tool`: builds a reusable planning-agent session and exposes specialist agents as tools
- `magentic`: builds a fresh peer-agent workflow per request and uses `PlanningAgent` only as the hidden manager configuration source

In both modes, participant resolution follows the same rules: `PlanningAgent` is excluded, omitted participants means all non-`PlanningAgent` agents, and ordering is preserved while duplicates are collapsed.

## Test Code Structure

Tests should follow the same directory structure as the [source code](#source-code-structure). There should be organizational directories and appropriate test files underneath them. For example, tests for the coordinator module will live at `tests/core/coordinator/` just like its source code lives at `mada/core/coordinator.py`.

There can be a `conftest.py` file in each testing directory to help define shared fixtures for individual components. There will also be a `conftest.py` file at the top-level directory for fixtures that are shared across the entire test suite.

For more on testing, see the [MADA Orchestrator Testing Guide](./testing.md).
