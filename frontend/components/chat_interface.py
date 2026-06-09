# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Chat interface component."""

import dash_bootstrap_components as dbc
from dash import html, dcc, State


def create_chat_interface_component():
    """Create the chat interface component.

    Returns:
        Dash component with chat UI
    """
    return html.Div(
        [
            # Chat messages display
            dbc.Card(
                [
                    dbc.CardHeader("Chat"),
                    dbc.CardBody(
                        [
                            html.Div(
                                id="chat-messages",
                                className="chat-container",
                                style={
                                    "maxHeight": "400px",
                                    "overflowY": "auto",
                                    "padding": "10px",
                                    "backgroundColor": "#f8f9fa",
                                    "borderRadius": "5px",
                                    "marginBottom": "15px",
                                },
                            ),
                        ],
                        style={"padding": "15px"},
                    ),
                ],
                className="mb-3",
            ),
            # Input area
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Textarea(
                                id="chat-input",
                                placeholder="Type your message here...",
                                style={"minHeight": "80px"},
                                disabled=True,
                            ),
                        ],
                        width=12,
                    ),
                ],
                className="mb-2",
            ),
            # Controls
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                "📤 Send",
                                id="send-btn",
                                color="primary",
                                disabled=True,
                                className="me-2",
                            ),
                            dbc.Button(
                                "🗑️ Clear",
                                id="clear-chat-btn",
                                color="secondary",
                                className="me-2",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [html.Div(id="chat-status", className="text-muted")], width=8
                    ),
                ]
            ),
            # Performance metrics display
            html.Hr(),
            html.H6("Performance Metrics", className="mt-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "0.00s",
                                                id="metric-ttft",
                                                className="card-title text-center",
                                            ),
                                            html.P(
                                                "TTFT",
                                                className="card-text text-center text-muted",
                                            ),
                                        ]
                                    )
                                ],
                                color="info",
                                outline=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "0.00",
                                                id="metric-tps",
                                                className="card-title text-center",
                                            ),
                                            html.P(
                                                "Tokens/sec",
                                                className="card-text text-center text-muted",
                                            ),
                                        ]
                                    )
                                ],
                                color="success",
                                outline=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "0",
                                                id="metric-tokens",
                                                className="card-title text-center",
                                            ),
                                            html.P(
                                                "Total Tokens",
                                                className="card-text text-center text-muted",
                                            ),
                                        ]
                                    )
                                ],
                                color="warning",
                                outline=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                "0.00s",
                                                id="metric-total-time",
                                                className="card-title text-center",
                                            ),
                                            html.P(
                                                "Total Time",
                                                className="card-text text-center text-muted",
                                            ),
                                        ]
                                    )
                                ],
                                color="secondary",
                                outline=True,
                            )
                        ],
                        width=3,
                    ),
                ]
            ),
            # Parameter controls
            html.Hr(),
            html.H6("Generation Parameters", className="mt-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Max Tokens", className="fw-bold"),
                            dcc.Slider(
                                id="max-tokens-slider",
                                min=64,
                                max=2048,
                                step=64,
                                value=256,
                                marks={
                                    64: "64",
                                    512: "512",
                                    1024: "1024",
                                    2048: "2048",
                                },
                                disabled=True,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Temperature", className="fw-bold"),
                            dcc.Slider(
                                id="temperature-slider",
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                value=0.7,
                                marks={0.0: "0.0", 0.5: "0.5", 1.0: "1.0"},
                                disabled=True,
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            # Hidden stores for state
            dcc.Store(id="chat-history", data=[]),
            dcc.Store(id="prompt-history", data=[]),
            dcc.Store(id="streaming-text", data=""),
            dcc.Interval(
                id="streaming-interval",
                interval=100,  # 100ms for smooth streaming
                n_intervals=0,
                disabled=True,
            ),
        ]
    )


def create_message_bubble(role, content, is_streaming=False):
    """Create a chat message bubble.

    Args:
        role: 'user' or 'assistant'
        content: Message content
        is_streaming: Whether this is a streaming message

    Returns:
        HTML component for message bubble
    """
    is_user = role == "user"

    bubble_style = {
        "maxWidth": "80%",
        "padding": "10px 15px",
        "borderRadius": "15px",
        "marginBottom": "10px",
        "wordWrap": "break-word",
    }

    if is_user:
        bubble_style.update(
            {
                "backgroundColor": "#007bff",
                "color": "white",
                "marginLeft": "auto",
                "borderBottomRightRadius": "5px",
            }
        )
        avatar = "👤"
    else:
        bubble_style.update(
            {
                "backgroundColor": "#e9ecef",
                "color": "black",
                "marginRight": "auto",
                "borderBottomLeftRadius": "5px",
            }
        )
        avatar = "🤖"

    if is_streaming:
        bubble_style["border"] = "2px dashed #28a745"

    return html.Div(
        [
            html.Span(f"{avatar} ", style={"marginRight": "5px"}),
            html.Div(content, style=bubble_style),
        ],
        style={
            "display": "flex",
            "flexDirection": "row-reverse" if is_user else "row",
            "alignItems": "flex-start",
            "marginBottom": "15px",
        },
    )
