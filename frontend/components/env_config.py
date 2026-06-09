# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Environment configuration component."""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback


def create_env_config_component(env_vars):
    """Create the environment configuration component.

    Args:
        env_vars: Dictionary of environment variables

    Returns:
        Dash component with editable environment variables
    """
    # Define which variables are sensitive (to be masked)
    sensitive_vars = ["HF_TOKEN", "JWT_SECRET"]

    # Define variable descriptions
    descriptions = {
        "HF_TOKEN": "Hugging Face token for model downloads",
        "JWT_SECRET": "Secret key for API authentication",
        "SERVICE_PORT": "Port for vLLM server (default: 8000)",
        "DASHBOARD_PORT": "Port for this dashboard (default: 8050)",
        "CACHE_ROOT": "Directory for model cache",
        "DEPLOY_URL": "Base URL for deployment",
        "DEVICE_ID": "P100 device ID (default: 0)",
    }

    rows = []
    for var_name, default_value in env_vars.items():
        is_sensitive = var_name in sensitive_vars

        # Create input group with visibility toggle for sensitive vars
        if is_sensitive:
            input_component = html.Div(
                [
                    dbc.Input(
                        id=f"env-{var_name}",
                        type="password",
                        value=default_value,
                        placeholder=f"Enter {var_name}",
                    ),
                    dbc.Button(
                        "👁️",
                        id=f"toggle-{var_name}",
                        color="light",
                        size="sm",
                        className="mt-1",
                    ),
                ]
            )
        else:
            input_component = dbc.Input(
                id=f"env-{var_name}",
                type="text",
                value=default_value,
                placeholder=f"Enter {var_name}",
            )

        row = dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(var_name, className="fw-bold"),
                        html.Small(
                            descriptions.get(var_name, ""),
                            className="text-muted d-block",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(input_component, width=9),
            ],
            className="mb-3",
        )

        rows.append(row)

    return html.Div(
        [
            html.H4("Environment Configuration", className="mb-3"),
            html.P(
                "Override environment variables for this session. "
                "Values are loaded from .env by default.",
                className="text-muted",
            ),
            html.Hr(),
            html.Div(rows),
            html.Hr(),
            dbc.Button(
                "Reset to Defaults",
                id="reset-env-btn",
                color="secondary",
                className="me-2",
            ),
            dbc.Button("Apply Changes", id="apply-env-btn", color="primary"),
            html.Div(id="env-status", className="mt-3"),
        ]
    )


def register_env_callbacks(app, env_vars):
    """Register callbacks for environment configuration component.

    Args:
        app: Dash application instance
        env_vars: Dictionary of environment variable names
    """
    sensitive_vars = ["HF_TOKEN", "JWT_SECRET"]

    # Register toggle callbacks for sensitive fields
    for var_name in sensitive_vars:
        if var_name in env_vars:

            @app.callback(
                Output(f"env-{var_name}", "type"),
                Input(f"toggle-{var_name}", "n_clicks"),
                State(f"env-{var_name}", "type"),
                prevent_initial_call=True,
            )
            def toggle_visibility(n_clicks, current_type, var=var_name):
                if n_clicks:
                    return "text" if current_type == "password" else "password"
                return current_type
