# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""tt-smi utility for P100 hardware monitoring."""

import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_tt_smi_output() -> str:
    """Get raw tt-smi output.

    Returns:
        Raw tt-smi output as string, or error message if command fails.
    """
    try:
        result = subprocess.run(["tt-smi"], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            return result.stdout
        else:
            return f"tt-smi error (code {result.returncode}):\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "tt-smi command timed out after 10 seconds"
    except FileNotFoundError:
        return "tt-smi not found. Is it installed and in PATH?"
    except Exception as e:
        return f"Error running tt-smi: {str(e)}"


def get_tt_smi_list() -> str:
    """Get tt-smi -ls output (list devices).

    Returns:
        Raw tt-smi -ls output as string.
    """
    try:
        result = subprocess.run(
            ["tt-smi", "-ls"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            return result.stdout
        else:
            return f"tt-smi -ls error (code {result.returncode}):\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "tt-smi -ls command timed out after 10 seconds"
    except FileNotFoundError:
        return "tt-smi not found. Is it installed and in PATH?"
    except Exception as e:
        return f"Error running tt-smi -ls: {str(e)}"
