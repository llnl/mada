# MADA User Guide

This user guide will cover [installation](./installation.md), [configuration](./configuration.md), and [usage](./usage/index.md) of MADA.

## What is MADA?

MADA is a framework designed to facilitate collaboration between multiple specialized agents within a unified system. Built on the Agent Framework, MADA supports multiple [orchestration modes](./configuration.md#optional-orchestration-configuration) for coordinating how agents work together. MADA provides CLI, Gradio, and OpenAI-compatible interfaces for interacting with the configured agent team.

## How Does MADA Work?

MADA begins by reading your [configuration](./configuration.md), which specifies the [model](./configuration.md#model-configuration), the [agents](./configuration.md#agent-configuration), and the optional [orchestration mode](./configuration.md#optional-orchestration-configuration). If you are using the [Gradio run mode](./usage/index.md#gradio-mode-overview), you can also provide [interface](./configuration.md#optional-gradio-interface-configuration) settings for customizing the Gradio web application UI.

After orchestration is initialized, MADA waits for user input. The configured [orchestration mode](./configuration.md#optional-orchestration-configuration) determines how specialists coordinate to process requests. Responses are streamed back through the chosen interface.
