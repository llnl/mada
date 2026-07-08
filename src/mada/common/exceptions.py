# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Custom exceptions for MADA.
"""


class MADAUnsupportedDatabase(Exception):
    """
    Used when trying to initialize a connection to an unsupported
    database type.
    """
