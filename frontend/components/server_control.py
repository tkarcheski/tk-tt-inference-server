# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Server control component with start/stop buttons."""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_server_control_component():
    """Create the server control component.

    Returns:
        Dash component with server controls
    """
    return html.Div(
        [
            html.H4("Server Control", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Status", className="card-title"),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "●",
                                                        id="server-status-dot",
                                                        style={
                                                            "color": "#dc3545",  # Red
                                                            "fontSize": "24px",
                                                            "marginRight": "10px",
                                                        },
                                                    ),
                                                    html.Span(
                                                        "Stopped",
                                                        id="server-status-text",
                                                        className="fw-bold",
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                "Model: -",
                                                id="server-model-info",
                                                className="text-muted mt-2",
                                            ),
                                            html.Div(
                                                "Port: -",
                                                id="server-port-info",
                                                className="text-muted",
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Actions", className="card-title"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                "▶ Start Server",
                                                                id="start-server-btn",
                                                                color="success",
                                                                className="w-100",
                                                                size="lg",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                "⏹ Stop Server",
                                                                id="stop-server-btn",
                                                                color="danger",
                                                                className="w-100",
                                                                size="lg",
                                                                disabled=True,
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                id="server-action-status",
                                                className="mt-3",
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        width=8,
                    ),
                ]
            ),
            dcc.Interval(
                id="server-status-interval",
                interval=2000,  # 2 seconds
                n_intervals=0,
            ),
        ]
    )


def create_model_selector_component():
    """Create the model selector dropdown.

    Returns:
        Dash component with model selection
    """
    return html.Div(
        [
            html.Hr(),
            html.H5("Model Selection", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Model", className="fw-bold"),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=[
                                    {
                                        "label": "GPT-OSS-20B (P100) - Experimental",
                                        "value": "gpt-oss-20b",
                                    },
                                    {
                                        "label": "Llama-3.1-8B (P100) - Recommended",
                                        "value": "Llama-3.1-8B",
                                    },
                                ],
                                value="Llama-3.1-8B",  # Default to 8B for better compatibility
                                disabled=False,
                            ),
                            html.Small(
                                id="model-description",
                                children="Context: 64k tokens | Device: P100 | Smaller, better compatibility",
                                className="text-muted d-block mt-1",
                            ),
                            html.Br(),
                            dbc.Label("Max Context Length", className="fw-bold mt-2"),
                            dcc.Dropdown(
                                id="context-length-dropdown",
                                options=[
                                    {"label": "512 tokens (safest)", "value": "512"},
                                    {"label": "1024 tokens (safe)", "value": "1024"},
                                    {
                                        "label": "2048 tokens (recommended)",
                                        "value": "2048",
                                    },
                                    {
                                        "label": "4096 tokens (experimental)",
                                        "value": "4096",
                                    },
                                    {"label": "8192 tokens (risky)", "value": "8192"},
                                    {
                                        "label": "16384 tokens (very risky)",
                                        "value": "16384",
                                    },
                                ],
                                value="2048",
                                clearable=False,
                            ),
                            html.Small(
                                id="context-warning",
                                children="Change triggers automatic server restart",
                                className="text-warning d-block mt-1",
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Device", className="fw-bold"),
                            dcc.Dropdown(
                                id="device-dropdown",
                                options=[
                                    {"label": "P100 (Blackhole)", "value": "p100"}
                                ],
                                value="p100",
                                disabled=True,
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )
