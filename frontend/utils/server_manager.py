# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Server manager for controlling vLLM Docker containers."""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, List
import os
import secrets

logger = logging.getLogger(__name__)


class ServerManager:
    """Manages vLLM server lifecycle via Docker/run.py."""

    # Model configurations
    MODEL_CONFIGS = {
        "gpt-oss-20b": {
            "impl": "gpt-oss",
            "docker_image": "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.7.0-5491d3c-f49265a",
            "needs_p100_workarounds": True,
        },
        "Llama-3.1-8B": {
            "impl": "tt-transformers",
            "docker_image": "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.7.0-55fd115-aa4ae1e",
            "needs_p100_workarounds": False,
        },
    }

    def __init__(
        self,
        env_vars: Dict[str, str],
        model: str = "Llama-3.1-8B",
        max_context: str = "2048",
    ):
        """Initialize server manager.

        Args:
            env_vars: Dictionary of environment variables
            model: Model name to run (default: Llama-3.1-8B)
            max_context: Maximum context length in tokens (default: 2048)
        """
        self.env_vars = env_vars
        self.process: Optional[subprocess.Popen] = None
        self.repo_root = Path(__file__).parent.parent.parent
        self.model = model
        self.device = "p100"
        self.max_context = max_context

        # Get model config
        if model not in self.MODEL_CONFIGS:
            logger.warning(f"Unknown model {model}, using default config")
            self.model_config = self.MODEL_CONFIGS["Llama-3.1-8B"]
        else:
            self.model_config = self.MODEL_CONFIGS[model]

    def _get_docker_image(self) -> str:
        """Get Docker image for the selected model."""
        return self.model_config["docker_image"]

    def start(self) -> bool:
        """Start the vLLM server.

        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running():
            logger.info("Server is already running")
            return True

        try:
            # Build command to run server
            # Use system python3 which has yaml and other dependencies
            python_executable = "/home/tyler/.tenstorrent-venv/bin/python3"

            # Get model-specific configuration
            impl = self.model_config["impl"]
            needs_workarounds = self.model_config["needs_p100_workarounds"]

            cmd = [
                python_executable,
                "run.py",
                "--model",
                self.model,
                "--device",
                self.device,
                "--impl",
                impl,
                "--workflow",
                "server",
                "--docker-server",
                "--device-id",
                self.env_vars.get("DEVICE_ID", "0"),
                "--dev-mode",
                "--override-docker-image",
                self._get_docker_image(),
            ]

            # Add P100-specific workarounds to prevent crashes
            cmd.extend(
                [
                    "--disable-trace-capture",  # Disable trace capture to prevent L1 buffer crashes
                    "--skip-system-sw-validation",  # Skip validation that fails on P100
                ]
            )

            if needs_workarounds:
                cmd.extend(
                    [
                        "--vllm-override-args",  # Reduce concurrency for P100 core limits
                        '{"max_num_seqs": 8}',
                        "--override-tt-config",
                        '{"trace_region_size": 20000000}',  # Smaller trace region
                    ]
                )

            # Generate a fresh JWT_SECRET for this session
            jwt_secret = self._generate_jwt_secret()
            self.current_jwt_secret = jwt_secret
            logger.info("Generated new JWT_SECRET for this session")

            # Set environment variables
            env = os.environ.copy()
            env.update(self.env_vars)
            env["AUTOMATIC_HOST_SETUP"] = "1"
            env["MODEL_SOURCE"] = "huggingface"
            env["JWT_SECRET"] = jwt_secret  # Override with auto-generated secret
            env["MAX_CONTEXT_LEN"] = (
                self.max_context
            )  # Pass context length to container
            env["MAX_MODEL_LEN"] = self.max_context  # Alternative name for vLLM

            logger.info(f"Starting server with command: {' '.join(cmd)}")

            # Start process
            self.process = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Give it a moment to start
            time.sleep(2)

            # Check if process is still running
            if self.process.poll() is None:
                logger.info("Server process started successfully")
                return True
            else:
                stdout, _ = self.process.communicate()
                logger.error(f"Server failed to start: {stdout}")
                return False

        except Exception as e:
            logger.error(f"Error starting server: {e}")
            return False

    def stop(self) -> bool:
        """Stop the vLLM server.

        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if self.process and self.process.poll() is None:
                # Try to terminate gracefully first
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop
                    self.process.kill()
                    self.process.wait()

                logger.info("Server process stopped")

            # Also try to stop any running Docker containers
            self._stop_docker_containers()

            self.process = None
            return True

        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            return False

    def _stop_docker_containers(self):
        """Stop any running vLLM Docker containers."""
        try:
            # Find containers with "tt-inference-server" in the name
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", "name=tt-inference-server"],
                capture_output=True,
                text=True,
            )

            container_ids = result.stdout.strip().split("\n")
            container_ids = [cid for cid in container_ids if cid]

            for cid in container_ids:
                logger.info(f"Stopping Docker container: {cid}")
                subprocess.run(["docker", "stop", cid], capture_output=True)

        except Exception as e:
            logger.error(f"Error stopping Docker containers: {e}")

    def is_running(self) -> bool:
        """Check if server is running.

        Returns:
            True if server is running, False otherwise
        """
        # Check if our process is running
        if self.process and self.process.poll() is None:
            return True

        # Check if Docker container is running (filter for tt-inference-server)
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", "name=tt-inference-server"],
                capture_output=True,
                text=True,
            )
            return bool(result.stdout.strip())
        except:
            return False

    def get_logs(self, lines: int = 50) -> List[str]:
        """Get recent server logs.

        Args:
            lines: Number of lines to return

        Returns:
            List of log lines
        """
        log_lines = []

        # Try to get logs from process output
        if self.process:
            # Note: This is tricky with Popen, we'd need to capture output differently
            # For now, we'll read from log files
            pass

        # Try to read from workflow logs directory
        log_dir = self.repo_root / "workflow_logs" / "docker_server"
        if log_dir.exists():
            # Get most recent log file
            log_files = sorted(
                log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True
            )
            if log_files:
                try:
                    with open(log_files[0], "r") as f:
                        all_lines = f.readlines()
                        log_lines = (
                            all_lines[-lines:] if len(all_lines) > lines else all_lines
                        )
                except Exception as e:
                    logger.error(f"Error reading log file: {e}")

        return log_lines

    def _generate_jwt_secret(self) -> str:
        """Generate a secure random JWT secret (32 bytes)."""
        return secrets.token_urlsafe(32)  # 32 bytes = 256 bits, URL-safe base64

    def get_current_jwt_secret(self) -> Optional[str]:
        """Get the current JWT secret (auto-generated or from env)."""
        return getattr(self, "current_jwt_secret", None) or self.env_vars.get(
            "JWT_SECRET"
        )

    def get_status(self) -> Dict:
        """Get server status.

        Returns:
            Dictionary with status information
        """
        is_running = self.is_running()

        return {
            "running": is_running,
            "model": self.model if is_running else None,
            "device": self.device if is_running else None,
            "port": self.env_vars.get("SERVICE_PORT", "8000") if is_running else None,
        }
