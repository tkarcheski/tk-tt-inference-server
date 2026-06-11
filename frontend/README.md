# P100 Testing Dashboard

A Dash-based web interface for testing and monitoring GPT-OSS-20B on Tenstorrent P100 (Blackhole) cards.

## Features

### 🤖 Chat Interface
- **Server Control**: Start/stop vLLM server with one click
- **Model Selection**: Currently supports GPT-OSS-20B on P100
- **Prompt Library**: 30+ curated prompts organized by category:
  - Tenstorrent Knowledge (architecture, comparisons)
  - Code Generation (Python, Rust, SQL, etc.)
  - Creative Writing (stories, poems, analogies)
  - Reasoning & Analysis (comparisons, trade-offs)
  - System Testing (long responses, throughput)
- **Real-time Chat**: Send messages and receive streaming responses
- **Performance Metrics**: TTFT, TPS, token count, total latency
- **Parameter Controls**: Adjustable max_tokens and temperature

### ⚙️ Hardware Config
- Real-time tt-smi output display
- Device list monitoring
- Auto-refresh every 5 seconds

### 🔧 Environment
- Override environment variables from UI
- Sensitive values (HF_TOKEN, JWT_SECRET) are masked
- Session-only changes (not saved to .env)
- Support for SERVICE_PORT, DASHBOARD_PORT, CACHE_ROOT, etc.

### 📜 Logs
- Live server log streaming
- Auto-refresh every 2 seconds
- Pause/resume log updates

## Quick Start

### Prerequisites
1. Python 3.8+
2. Docker (without sudo)
3. tt-smi installed
4. .env file with required variables

### Installation

```bash
# From the repository root
cd scripts/p100
./start-dashboard.sh
```

This will:
1. Create a Python virtual environment (`.venv_frontend`)
2. Install required dependencies
3. Check environment configuration
4. Start the dashboard on port 8050 (or DASHBOARD_PORT from .env)

### Manual Start

```bash
cd frontend
pip install -r requirements.txt
python3 app.py
```

## Usage

### Starting the Server

1. Open the dashboard at `http://localhost:8050`
2. Go to the **Chat** tab
3. Click **▶ Start Server**
4. Wait for the server to initialize (first run takes 10-30 minutes for cache generation)
5. Once status shows "Running", you're ready to chat

### Sending Messages

1. Select a prompt from the dropdown, or type your own
2. Adjust **Max Tokens** and **Temperature** sliders if desired
3. Click **📤 Send** or press Enter
4. Watch the response stream in the chat window
5. View performance metrics below the chat

### Monitoring Hardware

1. Go to the **Hardware** tab
2. View tt-smi output showing device status, temperature, and memory
3. Click **🔄 Refresh** for manual update or wait for auto-refresh

### Environment Variables

1. Go to the **Environment** tab
2. Edit variables as needed (sensitive values are masked)
3. Click **👁️** to toggle visibility of secrets
4. Click **Apply Changes** to update the session

**Note**: Changes are session-only and won't be saved to .env file.

### Viewing Logs

1. Go to the **Logs** tab
2. Watch real-time server logs
3. Click **⏸ Pause** to stop auto-refresh
4. Click **🔄 Refresh** for manual update

## Configuration

### Environment Variables

The dashboard reads from your `.env` file. Key variables:

- `HF_TOKEN`: Hugging Face token for model downloads
- `JWT_SECRET`: Secret key for API authentication
- `SERVICE_PORT`: vLLM server port (default: 8000)
- `DASHBOARD_PORT`: Dashboard port (default: 8050)
- `CACHE_ROOT`: Model cache directory
- `DEVICE_ID`: P100 device ID (default: 0)

### Prompt Suggestions

Prompts are defined in `frontend/components/prompt_library.py`. To add your own:

```python
PROMPT_SUGGESTIONS = {
    "Your Category": [
        "Your prompt here",
        "Another prompt",
    ],
    # ... existing categories
}
```

## Architecture

```
frontend/
├── app.py                    # Main Dash application
├── requirements.txt          # Python dependencies
├── components/
│   ├── __init__.py
│   ├── chat_interface.py    # Chat UI components
│   ├── env_config.py        # Environment variable editor
│   ├── hardware_tab.py      # tt-smi display
│   ├── logs_tab.py          # Log viewer
│   ├── prompt_library.py    # Suggested prompts
│   └── server_control.py    # Start/stop buttons
└── utils/
    ├── __init__.py
    ├── api_client.py        # vLLM API client
    ├── env_loader.py        # .env file handling
    ├── server_manager.py    # Docker server control
    └── tt_smi.py           # tt-smi wrapper
```

## Troubleshooting

### Dashboard won't start

```bash
# Check Python version
python3 --version  # Must be 3.8+

# Check if port is in use
lsof -i :8050

# Try different port
DASHBOARD_PORT=8051 ./scripts/p100/start-dashboard.sh
```

### Server fails to start

1. Check Docker is running: `docker ps`
2. Verify device access: `tt-smi -ls`
3. Check environment variables in **Environment** tab
4. View logs in **Logs** tab for errors

### API connection errors

1. Ensure server status shows "Running"
2. Check SERVICE_PORT matches in Environment tab
3. Verify JWT_SECRET is set correctly
4. Try refreshing the page

### Performance issues

- First run requires cache generation (10-30 minutes)
- Subsequent runs are much faster
- Monitor device temperature in Hardware tab
- Reduce max_tokens or concurrency if needed

## Development

### Adding New Features

1. Add components to `frontend/components/`
2. Register callbacks in `frontend/app.py`
3. Update utils in `frontend/utils/`

## Known Issues

- Streaming responses may have slight delay in UI update
- Chat history is lost on page refresh (will add persistence later)
- Only single session supported (multi-session tabs are UI-only for now)

## Future Enhancements

- [ ] Multi-session support with database persistence
- [ ] Response streaming with real-time token display
- [ ] Batch testing interface
- [ ] Custom prompt saving
- [ ] Performance benchmarking charts
- [ ] Model comparison tools

## License

SPDX-License-Identifier: Apache-2.0

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
