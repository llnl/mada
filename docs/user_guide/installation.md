# Installing and Setting Up MADA

This guide will help you install and set up MADA for basic use or for development.

For basic installation and setup, follow the instructions at the following sections:

1. [Basic Installation](#basic-installation)
2. [Environment Variable Setup](#environment-variable-setup)

Developers, see above and also [Installing Optional Dependencies](#installing-optional-dependencies)

## Basic Installation

1. First, create a virtual environment:

    ```bash
    python -m venv mada_venv
    ```

2. Activate the virtual environment:

    === "bash"

        ```bash
        source mada_venv/bin/activate
        ```

    === "csh"

        ```csh
        source mada_venv/bin/activate.csh
        ```

3. Finally, install MADA with pip:

    ```bash
    pip install mada
    ```

## Developer Setup

1. First, clone the repository:

    <!-- This repo will exist after open sourcing -->
    ```bash
    git clone https://github.com/llnl/mada.git
    ```

2. Next, move into the cloned repository:

    ```bash
    cd mada/
    ```

2. Now, create a virtual environment:

    ```bash
    python -m venv mada_venv
    ```

3. Activate the virtual environment:

    === "bash"

        ```bash
        source mada_venv/bin/activate
        ```

    === "csh"

        ```csh
        source mada_venv/bin/activate.csh
        ```

4. Finally, install an editable version of MADA with pip and all development dependencies:

    ```bash
    pip install -e .[dev]
    ```

## Environment Variable Setup

MADA requires certain environment variables to be set for API authentication and configuration. These will be used by your [Model Configuration](./configuration.md#model-configuration).

| Variable       | Required | Default Value                         | Purpose                    |
| -------------- | -------- | ------------------------------------- | -------------------------- |
| `API_KEY`      | Yes      | None                                  | API key for authentication |
| `API_BASE_URL` | No       | https://api.openai.com/v1/responses   | API endpoint               |

You can set these variables from the command line with:

```bash
export API_KEY=<your_api_key_here>
export API_BASE_URL=<custom_api_url>  # Optional
```

For developers, you can set these variables using a `.env` file in the project root. Put the following in your file:

```bash
API_KEY=<your_api_key_here>
API_BASE_URL=<custom_api_url>  # Optional
```

## Installing Optional Dependencies

There are two sets of optional dependencies that can be installed: one set for tests and the other for documentation.

These can be installed together:

=== "Shorthand"

    ```bash
    pip install -e .[dev]
    ```

=== "Verbose"

    ```bash
    pip install -e .[tests,docs]
    ```

Or separately:

=== "Install Test Dependencies"

    ```bash
    pip install -e .[tests]
    ```

=== "Install Documentation Dependencies"

    ```bash
    pip install -e .[docs]
    ```
