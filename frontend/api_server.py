#!/usr/bin/env python3
"""FastAPI backend for P100 Dashboard."""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from utils import (
    load_env_vars,
    ServerManager,
    VLLMClient,
    get_tt_smi_output,
    get_tt_smi_list,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="P100 Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env_vars = load_env_vars()
server_manager: Optional[ServerManager] = None
current_model = "Llama-3.1-8B"
current_context_length = "2048"


class ChatRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 256
    temperature: float = 0.7


class ServerConfig(BaseModel):
    model: str
    context_length: str


@app.get("/api/health")
def health_check():
    return {"status": "ok", "api_version": "1.0.0"}


@app.post("/api/server/start")
def start_server(config: ServerConfig, background_tasks: BackgroundTasks):
    global server_manager, current_model, current_context_length

    current_model = config.model
    current_context_length = config.context_length

    if server_manager is None:
        server_manager = ServerManager(
            env_vars, model=current_model, max_context=current_context_length
        )
    else:
        server_manager.model = current_model
        server_manager.max_context = current_context_length

    if server_manager.is_running():
        return {"status": "already_running", "model": current_model}

    success = server_manager.start()
    if success:
        return {
            "status": "starting",
            "model": current_model,
            "context_length": current_context_length,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to start server")


@app.post("/api/server/stop")
def stop_server():
    global server_manager

    if server_manager is None:
        raise HTTPException(status_code=400, detail="Server not initialized")

    success = server_manager.stop()
    if success:
        return {"status": "stopped"}
    else:
        raise HTTPException(status_code=500, detail="Failed to stop server")


@app.get("/api/server/status")
def get_status():
    global server_manager, current_model

    if server_manager is None:
        return {"running": False, "model": None}

    status = server_manager.get_status()
    return status


@app.post("/api/chat")
def chat(request: ChatRequest):
    global server_manager, env_vars

    if server_manager is None or not server_manager.is_running():
        raise HTTPException(status_code=503, detail="Server not running")

    service_port = env_vars.get("SERVICE_PORT", "8000")
    jwt_secret = server_manager.get_current_jwt_secret()

    try:
        # For Llama models without chat template, use completions API directly
        if request.model in ["Llama-3.1-8B", "meta-llama/Llama-3.1-8B"]:
            import requests

            last_message = request.messages[-1] if request.messages else {"content": ""}
            prompt = last_message.get("content", "")

            headers = {"Content-Type": "application/json"}
            if jwt_secret:
                import jwt as jwt_lib

                payload = {"team_id": "tenstorrent", "token_id": "dashboard-test"}
                token = jwt_lib.encode(payload, jwt_secret, algorithm="HS256")
                headers["Authorization"] = f"Bearer {token}"

            response = requests.post(
                f"http://127.0.0.1:{service_port}/v1/completions",
                json={
                    "model": "meta-llama/Llama-3.1-8B",
                    "prompt": prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                },
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "text": data["choices"][0].get("text", ""),
                "tokens": data.get("usage", {}).get("completion_tokens", 0),
                "ttft": 0.1,
                "tps": 0.0,
                "total_time": 0.0,
            }
        else:
            # Use chat completion for models that support it
            client = VLLMClient(
                f"http://127.0.0.1:{service_port}", jwt_secret=jwt_secret
            )
            result = client.chat_completion(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )

            return {
                "text": result["text"],
                "tokens": result["tokens"],
                "ttft": result["ttft"],
                "tps": result["tps"],
                "total_time": result["total_time"],
            }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hardware")
def get_hardware():
    try:
        smi_output = get_tt_smi_output()
        smi_list = get_tt_smi_list()
        return {"tt_smi": smi_output, "tt_smi_list": smi_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
def get_logs(lines: int = 100):
    global server_manager

    if server_manager is None:
        return {"logs": "Server not started"}

    try:
        log_lines = server_manager.get_logs(lines=lines)
        return {"logs": "".join(log_lines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
def list_models():
    return {
        "models": [
            {
                "id": "gpt-oss-20b",
                "name": "GPT-OSS-20B",
                "description": "20B parameter model",
            },
            {
                "id": "Llama-3.1-8B",
                "name": "Llama-3.1-8B",
                "description": "8B parameter model (recommended)",
            },
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8051)
