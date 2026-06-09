# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Hardware configuration tab component."""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_hardware_tab():
    """Create the hardware configuration tab content.

    Returns:
        Dash component with tt-smi output display
    """
    return html.Div(
        [
            html.H4("Hardware Configuration", className="mb-3"),
            html.P(
                "Real-time P100 (Blackhole) hardware status via tt-smi.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                "🔄 Refresh",
                                id="refresh-tt-smi-btn",
                                color="primary",
                                className="mb-3",
                            ),
                            dcc.Interval(
                                id="tt-smi-interval",
                                interval=5000,  # 5 seconds
                                n_intervals=0,
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("tt-smi Output"),
                                    dbc.CardBody(
                                        [
                                            html.Pre(
                                                id="tt-smi-output",
                                                className="bg-dark text-light p-3",
                                                style={
                                                    "maxHeight": "400px",
                                                    "overflow": "auto",
                                                    "fontSize": "12px",
                                                    "fontFamily": "monospace",
                                                },
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Device List (tt-smi -ls)"),
                                    dbc.CardBody(
                                        [
                                            html.Pre(
                                                id="tt-smi-list-output",
                                                className="bg-dark text-light p-3",
                                                style={
                                                    "maxHeight": "400px",
                                                    "overflow": "auto",
                                                    "fontSize": "12px",
                                                    "fontFamily": "monospace",
                                                },
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )
