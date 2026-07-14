# Changelog

## Unreleased

### Added
-

### Changed
- OpenAI-compatible model configs and streamable MCP server configs now support a `verify` setting for TLS verification control.

### Fixed
- OpenAI-compatible HTTP clients and streamable MCP HTTP clients now honor configured `verify` values when resolving TLS verification.

## 0.2.0 - 2026-07-07

### Added
- `bump_version.py` script for changing version
- `CHANGELOG` to track changes across releases
- workflows for running CI
- `pre-commit` and `ruff` for linting
- support for Windows OS
- workflows for publishing develop and stable versions of documentation

### Changed
- re-architected the test suite into unit/integration/e2e tests

### Fixed
- broken tests in the test suite
