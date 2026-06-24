# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Utility functions for configuration-related modules.
"""

import os
import re


def expand_env_vars(value: str | None) -> str | None:
    """
    Expand environment variable references in configuration values.

    Supports formats:
    - ${VAR_NAME} - expands to os.getenv("VAR_NAME")
    - ${VAR_NAME:-default} - expands with default value if not set

    Args:
        value:
            String that may contain environment variable references
            or None.

    Returns:
        String with environment variables expanded or None.
    """
    if value is None:
        return

    def replace_env_var(match):
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default_value = var_expr.split(":-", 1)
            return os.getenv(var_name, default_value)
        else:
            return os.getenv(var_expr, match.group(0))  # Return original if not found

    # Pattern matches ${VAR_NAME} or ${VAR_NAME:-default}
    pattern = r"\$\{([^}]+)\}"
    return re.sub(pattern, replace_env_var, value)
