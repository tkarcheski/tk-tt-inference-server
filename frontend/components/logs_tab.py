# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Logs tab component."""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_logs_tab():
    """Create the logs tab content.

    Returns:
        Dash component with log viewer
    """
    return html.Div(
        [
            html.H4("Server Logs", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                "🔄 Refresh",
                                id="refresh-logs-btn",
                                color="primary",
                                className="me-2",
                            ),
                            dbc.Button(
                                "⏸ Pause",
                                id="pause-logs-btn",
                                color="secondary",
                                className="me-2",
                            ),
                            dcc.Interval(
                                id="logs-interval",
                                interval=2000,  # 2 seconds
                                n_intervals=0,
                            ),
                        ],
                        width=12,
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            "Server Logs",
                                            dbc.Badge(
                                                "Live",
                                                color="success",
                                                className="ms-2",
                                                id="logs-live-badge",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Pre(
                                                id="server-logs-output",
                                                className="bg-dark text-light p-3",
                                                style={
                                                    "maxHeight": "500px",
                                                    "overflow": "auto",
                                                    "fontSize": "11px",
                                                    "fontFamily": "monospace",
                                                    "whiteSpace": "pre-wrap",
                                                },
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ],
                        width=12,
                    ),
                ]
            ),
        ]
    )
