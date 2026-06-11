#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""P100 Dashboard - Main Dash application."""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
from dash.exceptions import PreventUpdate

from utils import (
    load_env_vars,
    ServerManager,
    VLLMClient,
    get_tt_smi_output,
    get_tt_smi_list,
)
from components import (
    create_prompt_library_component,
    create_env_config_component,
    create_server_control_component,
    create_model_selector_component,
    create_hardware_tab,
    create_logs_tab,
    create_chat_interface_component,
    create_message_bubble,
    register_env_callbacks,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
env_vars = load_env_vars()

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.title = "P100 Testing Dashboard"

# Initialize server manager (model will be set dynamically)
server_manager = None
current_model = "Llama-3.1-8B"  # Default model

# App layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "🚀 P100 Testing Dashboard",
                            className="text-primary mt-4 mb-2",
                        ),
                        html.P(
                            "Test and monitor models on Tenstorrent P100 (Blackhole)",
                            className="text-muted",
                        ),
                    ]
                )
            ]
        ),
        html.Hr(),
        # Main tabs
        dbc.Tabs(
            [
                # Chat Tab
                dbc.Tab(
                    label="💬 Chat",
                    tab_id="chat-tab",
                    children=[
                        html.Div(
                            [
                                html.Br(),
                                # Server Control Section
                                create_server_control_component(),
                                create_model_selector_component(),
                                html.Hr(),
                                # Prompt Library
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                create_prompt_library_component(),
                                            ],
                                            width=12,
                                        ),
                                    ]
                                ),
                                html.Br(),
                                # Chat Interface
                                create_chat_interface_component(),
                            ],
                            className="p-3",
                        )
                    ],
                ),
                # Hardware Config Tab
                dbc.Tab(
                    label="⚙️ Hardware",
                    tab_id="hardware-tab",
                    children=[
                        html.Div(
                            [
                                html.Br(),
                                create_hardware_tab(),
                            ],
                            className="p-3",
                        )
                    ],
                ),
                # Environment Tab
                dbc.Tab(
                    label="🔧 Environment",
                    tab_id="env-tab",
                    children=[
                        html.Div(
                            [
                                html.Br(),
                                create_env_config_component(env_vars),
                            ],
                            className="p-3",
                        )
                    ],
                ),
                # Logs Tab
                dbc.Tab(
                    label="📜 Logs",
                    tab_id="logs-tab",
                    children=[
                        html.Div(
                            [
                                html.Br(),
                                create_logs_tab(),
                            ],
                            className="p-3",
                        )
                    ],
                ),
            ],
            id="main-tabs",
            active_tab="chat-tab",
        ),
        # Footer
        html.Hr(),
        html.Footer(
            [
                html.P(
                    "P100 Testing Dashboard - Experimental | Built with Dash",
                    className="text-center text-muted",
                ),
            ],
            className="mt-4 mb-4",
        ),
        # Global stores
        dcc.Store(id="current-env-vars", data=env_vars),
        dcc.Store(id="selected-model", data="Llama-3.1-8B"),
        dcc.Store(id="context-length", data="2048"),
        dcc.Store(id="server-config-restart-trigger", data=0),
    ],
    fluid=True,
)


# Callbacks


@app.callback(
    [
        Output("server-status-dot", "style"),
        Output("server-status-text", "children"),
        Output("server-model-info", "children"),
        Output("server-port-info", "children"),
        Output("start-server-btn", "disabled"),
        Output("stop-server-btn", "disabled"),
        Output("chat-input", "disabled"),
        Output("send-btn", "disabled"),
        Output("max-tokens-slider", "disabled"),
        Output("temperature-slider", "disabled"),
        Output("server-action-status", "children"),
    ],
    [
        Input("server-status-interval", "n_intervals"),
        Input("start-server-btn", "n_clicks"),
        Input("stop-server-btn", "n_clicks"),
        Input("server-config-restart-trigger", "data"),
    ],
    [
        State("current-env-vars", "data"),
        State("model-dropdown", "value"),
        State("context-length", "data"),
    ],
    prevent_initial_call=False,
)
def update_server_status(
    n_intervals,
    start_clicks,
    stop_clicks,
    restart_trigger,
    current_env,
    selected_model,
    context_length,
):
    """Update server status and control buttons."""
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    status_message = ""

    # Use selected model from dropdown
    if not selected_model:
        selected_model = "Llama-3.1-8B"

    # Use context length
    if not context_length:
        context_length = "2048"

    # Initialize or update server manager with selected model and context length
    global server_manager, current_model
    if server_manager is None or current_model != selected_model:
        server_manager = ServerManager(
            current_env, model=selected_model, max_context=context_length
        )
        current_model = selected_model
    else:
        server_manager.env_vars = current_env
        server_manager.max_context = context_length

    # Handle button clicks or restart trigger
    if triggered_id == "start-server-btn" and start_clicks:
        logger.info(
            f"Starting server with model: {selected_model}, context: {context_length}..."
        )
        try:
            if server_manager.is_running():
                status_message = dbc.Alert(
                    "✅ Server is already running.", color="info"
                )
            elif server_manager.start():
                status_message = dbc.Alert(
                    "✅ Server starting... This may take several minutes.",
                    color="success",
                )
            else:
                status_message = dbc.Alert(
                    "❌ Server failed to start. Check the Logs tab.",
                    color="danger",
                )
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            status_message = dbc.Alert(
                f"❌ Failed to start server: {str(e)}", color="danger"
            )

    elif triggered_id == "stop-server-btn" and stop_clicks:
        logger.info("Stopping server...")
        try:
            server_manager.stop()
            status_message = dbc.Alert("✅ Server stopped.", color="success")
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            status_message = dbc.Alert(
                f"❌ Failed to stop server: {str(e)}", color="danger"
            )

    # Check current status
    try:
        status = server_manager.get_status()
        is_running = status.get("running", False)
        service_port = status.get("port") or current_env.get("SERVICE_PORT", "8000")
    except Exception as e:
        logger.error(f"Failed to get server status: {e}")
        is_running = False
        service_port = current_env.get("SERVICE_PORT", "8000")

    if is_running:
        dot_style = {"color": "#28a745", "fontSize": "24px", "marginRight": "10px"}
        status_text = "Running"
        model_info = f"Model: {selected_model} | Device: P100"
        port_info = f"Port: {service_port}"
        start_disabled = True
        stop_disabled = False
        chat_disabled = False
    else:
        dot_style = {"color": "#dc3545", "fontSize": "24px", "marginRight": "10px"}
        status_text = "Stopped"
        model_info = "Model: -"
        port_info = "Port: -"
        start_disabled = False
        stop_disabled = True
        chat_disabled = True

    return (
        dot_style,
        status_text,
        model_info,
        port_info,
        start_disabled,
        stop_disabled,
        chat_disabled,
        chat_disabled,
        chat_disabled,
        chat_disabled,
        status_message if status_message else None,
    )


@app.callback(
    [
        Output("tt-smi-output", "children"),
        Output("tt-smi-list-output", "children"),
    ],
    [
        Input("tt-smi-interval", "n_intervals"),
        Input("refresh-tt-smi-btn", "n_clicks"),
    ],
    prevent_initial_call=False,
)
def update_tt_smi(n_intervals, n_clicks):
    """Update tt-smi output display."""
    try:
        return get_tt_smi_output(), get_tt_smi_list()
    except Exception as e:
        logger.error(f"Failed to get hardware info: {e}")
        return f"Error: {str(e)}", ""


@app.callback(
    Output("server-logs-output", "children"),
    [
        Input("logs-interval", "n_intervals"),
        Input("refresh-logs-btn", "n_clicks"),
    ],
    prevent_initial_call=False,
)
def update_logs(n_intervals, n_clicks):
    """Update server logs display."""
    if server_manager is None:
        return "Server has not been started yet."
    try:
        return "".join(server_manager.get_logs(lines=100)) or "No logs available."
    except Exception as e:
        return f"Error fetching logs: {str(e)}"


@app.callback(
    [
        Output("chat-messages", "children"),
        Output("chat-history", "data"),
        Output("chat-input", "value"),
        Output("chat-status", "children"),
        Output("metric-ttft", "children"),
        Output("metric-tps", "children"),
        Output("metric-tokens", "children"),
        Output("metric-total-time", "children"),
    ],
    [
        Input("send-btn", "n_clicks"),
        Input("clear-chat-btn", "n_clicks"),
        Input("prompt-dropdown", "value"),
    ],
    [
        State("chat-input", "value"),
        State("chat-history", "data"),
        State("max-tokens-slider", "value"),
        State("temperature-slider", "value"),
        State("current-env-vars", "data"),
        State("model-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def handle_chat(
    send_clicks,
    clear_clicks,
    selected_prompt,
    input_value,
    chat_history,
    max_tokens,
    temperature,
    current_env,
    selected_model,
):
    """Handle chat messages and responses."""
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Handle clear button
    if triggered_id == "clear-chat-btn" and clear_clicks:
        return [], [], "", "Chat cleared", "0.00s", "0.00", "0", "0.00s"

    # Get the prompt to send
    prompt = None
    if triggered_id == "prompt-dropdown" and selected_prompt:
        if selected_prompt == "__custom__":
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                "Enter custom prompt below",
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        prompt = selected_prompt
    elif triggered_id == "send-btn" and send_clicks and input_value:
        prompt = input_value

    if not prompt:
        raise PreventUpdate

    # Add user message to history
    if chat_history is None:
        chat_history = []

    chat_history.append({"role": "user", "content": prompt})

    # Create message bubbles
    message_bubbles = []
    for msg in chat_history:
        message_bubbles.append(create_message_bubble(msg["role"], msg["content"]))

    # Add streaming message placeholder
    message_bubbles.append(create_message_bubble("assistant", "", is_streaming=True))

    # Call the vLLM server directly
    try:
        current_env = current_env or {}
        service_port = current_env.get("SERVICE_PORT", "8000")
        jwt_secret = (
            server_manager.get_current_jwt_secret()
            if server_manager is not None
            else current_env.get("JWT_SECRET")
        )
        client = VLLMClient(
            f"http://localhost:{service_port}", jwt_secret=jwt_secret
        )
        result = client.chat_completion(
            model=selected_model,
            messages=chat_history,
            max_tokens=max_tokens or 256,
            temperature=temperature or 0.7,
        )
        if result.get("error"):
            raise RuntimeError(result["text"])

        # Add assistant response to history
        chat_history.append({"role": "assistant", "content": result["text"]})

        # Recreate message bubbles with response
        message_bubbles = []
        for msg in chat_history:
            message_bubbles.append(create_message_bubble(msg["role"], msg["content"]))

        # Update metrics
        ttft = f"{result.get('ttft', 0):.2f}s"
        tps = f"{result.get('tps', 0):.2f}"
        tokens = str(result.get("tokens", 0))
        total_time = f"{result.get('total_time', 0):.2f}s"

        return (
            message_bubbles,
            chat_history,
            "",
            f"✅ Response received ({result.get('tokens', 0)} tokens)",
            ttft,
            tps,
            tokens,
            total_time,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_msg = str(e)
        if "Connection refused" in error_msg:
            error_msg = "Server not running. Please start the server first."
        elif "Server not running" in error_msg:
            error_msg = "Server not running. Please start the server first."

        chat_history.append({"role": "assistant", "content": f"Error: {error_msg}"})

        message_bubbles = []
        for msg in chat_history:
            message_bubbles.append(create_message_bubble(msg["role"], msg["content"]))

        return (
            message_bubbles,
            chat_history,
            "",
            f"❌ Error: {error_msg}",
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )


@app.callback(
    Output("current-env-vars", "data"),
    [Input("apply-env-btn", "n_clicks")],
    [State(f"env-{var}", "value") for var in env_vars.keys()],
    prevent_initial_call=True,
)
def apply_env_changes(n_clicks, *values):
    """Apply environment variable changes."""
    if not n_clicks:
        raise PreventUpdate

    new_env = {}
    for i, var_name in enumerate(env_vars.keys()):
        new_env[var_name] = values[i] if values[i] else env_vars[var_name]

    # Update server manager if it exists
    global server_manager
    if server_manager is not None:
        server_manager.env_vars = new_env

    return new_env


@app.callback(
    Output("model-description", "children"),
    Output("selected-model", "data"),
    Input("model-dropdown", "value"),
)
def update_model_description(selected_model):
    """Update model description when selection changes."""
    if selected_model == "gpt-oss-20b":
        return (
            "Context: 32k tokens | Quantization: mxfp4 | Device: P100 | May have core limitations",
            selected_model,
        )
    elif selected_model == "Llama-3.1-8B":
        return (
            "Context: 64k tokens | Device: P100 | Smaller, better compatibility",
            selected_model,
        )
    else:
        return "Context: varies | Device: P100", selected_model


@app.callback(
    Output("context-length", "data"),
    Output("server-config-restart-trigger", "data"),
    Output("server-action-status", "children", allow_duplicate=True),
    Input("context-length-dropdown", "value"),
    State("server-status-text", "children"),
    State("server-config-restart-trigger", "data"),
    prevent_initial_call=True,
)
def handle_context_length_change(context_length, server_status, current_trigger):
    """Handle context length changes - store value and trigger restart if server is running."""
    global server_manager

    # Store the new context length
    logger.info(f"Context length changed to: {context_length}")

    # Check if server is running
    if server_status == "Running" and server_manager is not None:
        logger.info(
            "Server is running, stopping for restart with new context length..."
        )
        # Stop the server
        server_manager.stop()
        # Increment trigger to signal restart needed
        return (
            context_length,
            current_trigger + 1,
            dbc.Alert(
                f"🔄 Context length changed to {context_length}. Server restarting...",
                color="warning",
            ),
        )

    return context_length, current_trigger, dash.no_update


if __name__ == "__main__":
    dashboard_port = int(env_vars.get("DASHBOARD_PORT", "8050"))
    logger.info(f"Starting P100 Dashboard on port {dashboard_port}")
    app.run(debug=True, host="0.0.0.0", port=dashboard_port)
