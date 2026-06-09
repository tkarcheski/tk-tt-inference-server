# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""API client for vLLM streaming chat completions."""

import json
import logging
import time
import jwt
import requests
from typing import Dict, List, Optional, Callable, Generator

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for vLLM API with streaming support."""

    def __init__(self, base_url: str, jwt_secret: Optional[str] = None):
        """Initialize client.

        Args:
            base_url: Base URL for API (e.g., http://127.0.0.1:8000)
            jwt_secret: JWT secret for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.jwt_secret = jwt_secret
        self.headers = self._get_headers()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authorization."""
        headers = {"Content-Type": "application/json"}

        if self.jwt_secret:
            # Generate JWT token
            payload = {"team_id": "tenstorrent", "token_id": "dashboard-test"}
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            headers["Authorization"] = f"Bearer {token}"

        return headers

    def check_health(self) -> bool:
        """Check if server is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health", headers=self.headers, timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
        callback: Optional[Callable[[str, Dict], None]] = None,
    ) -> Dict:
        """Send chat completion request.

        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
            callback: Optional callback for streaming updates (token, metrics)

        Returns:
            Dictionary with response and metrics
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if stream:
            payload["stream_options"] = {"include_usage": True}

        start_time = time.time()
        first_token_time = None
        full_text = ""
        token_count = 0
        usage = None

        try:
            response = requests.post(
                url, json=payload, headers=self.headers, stream=stream, timeout=300
            )
            response.raise_for_status()

            if stream:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:].strip()

                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)

                            # Check for usage info
                            if (
                                "usage" in data
                                and data["usage"] is not None
                                and not data.get("choices")
                            ):
                                usage = data["usage"]
                                continue

                            # Extract content
                            if data.get("choices"):
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")

                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.time()

                                    full_text += content
                                    token_count += 1

                                    # Calculate current metrics
                                    current_time = time.time()
                                    metrics = {
                                        "ttft": first_token_time - start_time
                                        if first_token_time
                                        else 0,
                                        "tokens": token_count,
                                        "total_time": current_time - start_time,
                                    }

                                    if callback:
                                        callback(content, metrics)

                        except json.JSONDecodeError:
                            continue
            else:
                # Non-streaming response
                data = response.json()
                full_text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                token_count = usage.get("completion_tokens", 0)

            end_time = time.time()
            total_time = end_time - start_time

            # Calculate final metrics
            ttft = first_token_time - start_time if first_token_time else 0
            gen_time = (
                max(end_time - first_token_time, 0.0001)
                if first_token_time
                else total_time
            )
            tps = (token_count - 1) / gen_time if token_count > 1 else 0

            return {
                "text": full_text,
                "tokens": token_count,
                "ttft": ttft,
                "tps": tps,
                "total_time": total_time,
                "usage": usage or {},
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Chat completion request failed: {e}")
            # Try falling back to completions endpoint for models without chat template
            if "404" in str(e) or "Not Found" in str(e):
                logger.info("Falling back to /v1/completions endpoint")
                return self._completion_fallback(
                    model, messages, max_tokens, temperature, start_time
                )
            return {
                "text": f"Error: {str(e)}",
                "tokens": 0,
                "ttft": 0,
                "tps": 0,
                "total_time": time.time() - start_time,
                "usage": {},
                "error": True,
            }

    def _completion_fallback(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        start_time: float,
    ) -> Dict:
        """Fallback to /v1/completions for models without chat template."""
        url = f"{self.base_url}/v1/completions"

        # Extract the last message content as prompt
        prompt = messages[-1].get("content", "") if messages else ""

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(
                url, json=payload, headers=self.headers, timeout=300
            )
            response.raise_for_status()

            data = response.json()
            full_text = data["choices"][0].get("text", "")
            usage = data.get("usage", {})
            token_count = usage.get("completion_tokens", 0)

            end_time = time.time()
            total_time = end_time - start_time

            return {
                "text": full_text,
                "tokens": token_count,
                "ttft": 0.1,  # Approximate
                "tps": token_count / max(total_time, 0.001),
                "total_time": total_time,
                "usage": usage,
            }
        except Exception as e:
            logger.error(f"Fallback completion failed: {e}")
            return {
                "text": f"Error: {str(e)}",
                "tokens": 0,
                "ttft": 0,
                "tps": 0,
                "total_time": time.time() - start_time,
                "usage": {},
                "error": True,
            }
