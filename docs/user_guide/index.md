# MADA User Guide

This user guide will cover [installation](./installation.md), [configuration](./configuration.md), and [usage](./usage/index.md) of MADA.

## What is MADA?

MADA is a framework designed to facilitate collaboration between multiple specialized agents within a unified system. Built on the Agent Framework, MADA supports an `agent-as-tool` mode where a [planning agent](./configuration.md#the-planning-agent) delegates tasks to relevant specialists, and a `magentic` mode where specialists participate in a peer group chat coordinated by a hidden manager. MADA provides CLI, Gradio, and OpenAI-compatible interfaces for interacting with the configured agent team.

## How Does MADA Work?

MADA begins by reading your [configuration](./configuration.md), which specifies the [model](./configuration.md#model-configuration), the [agents](./configuration.md#agent-configuration), and the optional [orchestration mode](./configuration.md#optional-orchestration-configuration). If you are using the [Gradio run mode](./usage/index.md#gradio-mode-overview), you can also provide [interface](./configuration.md#optional-gradio-interface-configuration) settings for customizing the Gradio web application UI.

After orchestration is initialized, MADA waits for user input. In `agent-as-tool` mode, the planning agent selects the appropriate specialist by calling that specialist as a tool. In `magentic` mode, the hidden manager coordinates a fresh peer workflow and returns the final synthesized answer. Responses are streamed back through the chosen interface.
