# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Environment variable loader for P100 Dashboard."""

import os
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv


def load_env_vars(env_file: Optional[Path] = None) -> Dict[str, str]:
    """Load environment variables from .env file.

    Args:
        env_file: Path to .env file. If None, looks for .env in repo root.

    Returns:
        Dictionary of environment variables relevant to the dashboard.
    """
    if env_file is None:
        # Look for .env in repo root (parent of frontend/)
        env_file = Path(__file__).parent.parent.parent / ".env"

    # Load .env file if it exists
    if env_file.exists():
        load_dotenv(env_file)

    # Define relevant environment variables with defaults
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "JWT_SECRET": os.getenv("JWT_SECRET", ""),
        "SERVICE_PORT": os.getenv("SERVICE_PORT", "8000"),
        "DASHBOARD_PORT": os.getenv("DASHBOARD_PORT", "8050"),
        "CACHE_ROOT": os.getenv("CACHE_ROOT", str(Path.home() / "cache_root")),
        "DEPLOY_URL": os.getenv("DEPLOY_URL", "http://127.0.0.1"),
        "DEVICE_ID": os.getenv("DEVICE_ID", "0"),
    }

    return env_vars


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent.parent


def mask_sensitive_value(value: str, visible_chars: int = 4) -> str:
    """Mask a sensitive value, showing only last N characters.

    Args:
        value: The value to mask
        visible_chars: Number of characters to show at the end

    Returns:
        Masked string like "••••••••abcd"
    """
    if not value:
        return ""
    if len(value) <= visible_chars:
        return "•" * len(value)
    return "•" * (len(value) - visible_chars) + value[-visible_chars:]
