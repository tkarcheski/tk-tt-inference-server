# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Prompt library component with suggested prompts."""

import dash_bootstrap_components as dbc
from dash import html, dcc


# Curated prompt suggestions organized by category
PROMPT_SUGGESTIONS = {
    "Tenstorrent Knowledge": [
        "Explain the architecture of Tenstorrent's Blackhole chip",
        "Compare Wormhole and Blackhole - what are the key differences?",
        "How does Tenstorrent's approach to AI acceleration differ from GPUs?",
        "What makes the P100 card unique for inference workloads?",
        "Explain dataflow architecture and why it matters for AI",
    ],
    "Code Generation": [
        "Write a Python function to calculate Fibonacci numbers recursively",
        "Create a bash script that monitors CPU usage and logs to a file",
        "Write a Rust function to parse JSON with error handling",
        "Create a Dockerfile for a Python ML application",
        "Write a regex pattern to validate email addresses",
        "Debug this Python code:\n\ndef buggy():\n    return 1/0",
        "Generate a Makefile for a C++ project with multiple targets",
        "Write a SQL query to find duplicate records in a table",
        "Create a Python class for a thread-safe queue",
        "Write unit tests for a function that sorts a list",
    ],
    "Creative Writing": [
        "Write a haiku about artificial intelligence",
        "Tell me a short story about a robot learning to paint",
        "Write a poem about the speed of light",
        "Create a dialogue between two AI assistants",
        "Describe a futuristic city in 2150",
        "Write a joke about programmers",
        "Create an analogy for neural networks",
    ],
    "Reasoning & Analysis": [
        "Compare Python and Rust for systems programming",
        "What are the trade-offs between accuracy and speed in ML models?",
        "Analyze the complexity of quicksort vs mergesort",
        "Explain the CAP theorem with examples",
        "When should you use SQL vs NoSQL databases?",
        "Compare containerization vs virtualization",
        "What are the benefits and drawbacks of microservices?",
    ],
    "System Testing": [
        "Generate a long response to test maximum context length",
        "Write code with many tokens to test throughput",
        "Create a complex nested data structure as JSON",
        "Generate a detailed technical specification document",
        "Write a comprehensive guide with multiple sections",
    ],
}


def create_prompt_library_component():
    """Create the prompt library dropdown component.

    Returns:
        Dash component with dropdown for prompt selection
    """
    # Flatten prompts for dropdown, with category prefix
    dropdown_options = []
    for category, prompts in PROMPT_SUGGESTIONS.items():
        for prompt in prompts:
            dropdown_options.append(
                {
                    "label": f"[{category}] {prompt[:60]}{'...' if len(prompt) > 60 else ''}",
                    "value": prompt,
                }
            )

    # Add custom option
    dropdown_options.insert(0, {"label": "✏️ Custom prompt...", "value": "__custom__"})

    return html.Div(
        [
            dbc.Label("Select a Prompt", className="fw-bold"),
            dcc.Dropdown(
                id="prompt-dropdown",
                options=dropdown_options,
                placeholder="Choose a suggested prompt or select custom...",
                searchable=True,
                clearable=True,
                style={"marginBottom": "10px"},
            ),
        ]
    )


def get_all_prompts() -> list:
    """Get all prompts as a flat list.

    Returns:
        List of all prompt strings
    """
    prompts = []
    for category_prompts in PROMPT_SUGGESTIONS.values():
        prompts.extend(category_prompts)
    return prompts
