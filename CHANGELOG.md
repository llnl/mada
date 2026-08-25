# Changelog

## Unreleased

### Added
- DOI link to README
- Magentic Orchestration Functionality
- Safeguards for the orchestration switching and database consistency for async functionality
- Test coverage for Magentic orchestration

### Changed
- OpenAI-compatible model configs and streamable MCP server configs now support a `verify` setting for TLS verification control.

### Fixed
- OpenAI-compatible HTTP clients and streamable MCP HTTP clients now honor configured `verify` values when resolving TLS verification.
- Specialist agents now honor agent-level extra settings, including default_options, matching the existing PlanningAgent behavior.
- Wrong task planning from multi-turn conversations
- Tool call detection in nested Agent Framework event structures (including typed records)
- Incomplete streaming when final result differs from accumulated chunks

## 0.2.0 - 2026-07-07

### Added
- `bump_version.py` script for changing version
- `CHANGELOG` to track changes across releases
- workflows for running CI
- `pre-commit` and `ruff` for linting
- support for Windows OS
- workflows for publishing develop and stable versions of documentation
- Orchestration configuration layer and behavior selection through a mode-specific strategy, preserves existing CLI, Gradio, and OpenAI API interfaces.
- `orchestration.py` , `orchestrator.py` large updates to support the new pattern selection layer
- Adds A2A capabilities, enabling the agent to connect with other agents and be accessed by them through A2A.

### Changed
- re-architected the test suite into unit/integration/e2e tests
- converted interfaces to use async tools

### Fixed
- broken tests in the test suite
