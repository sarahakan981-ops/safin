"""Constants for RootAgent."""

import os
from typing import Optional

# Load .env file before reading environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Application Info
APP_NAME = "RootAgent"
APP_VERSION = "0.2.0"
APP_DESCRIPTION = "AI penetration testing"

# Agent States
AGENT_STATE_IDLE = "idle"
AGENT_STATE_THINKING = "thinking"
AGENT_STATE_EXECUTING = "executing"
AGENT_STATE_WAITING_INPUT = "waiting_input"
AGENT_STATE_COMPLETE = "complete"
AGENT_STATE_ERROR = "error"

# Tool Categories
TOOL_CATEGORY_EXECUTION = "execution"
TOOL_CATEGORY_WEB = "web"
TOOL_CATEGORY_NETWORK = "network"
TOOL_CATEGORY_RECON = "reconnaissance"
TOOL_CATEGORY_EXPLOITATION = "exploitation"
TOOL_CATEGORY_MCP = "mcp"

# Default Timeouts (in seconds)
DEFAULT_COMMAND_TIMEOUT = 300
DEFAULT_VPN_TIMEOUT = 30
DEFAULT_MCP_TIMEOUT = 60

# Docker Settings
DOCKER_SANDBOX_IMAGE = "ghcr.io/sarahakan981-ops/rootagent:kali"
DOCKER_NETWORK_MODE = "bridge"

# RAG Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RAG_TOP_K = 3

# Memory Settings
MEMORY_RESERVE_RATIO = 0.8  # Reserve 20% of context for response

# OpenAI-compatible endpoint
# Returns the configured base URL with no trailing slash, or None if not set.
def get_openai_api_base() -> Optional[str]:
    """Return the configured OpenAI-compatible API base URL, or None."""
    base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    return base.rstrip("/") if base else None

# LLM Defaults (set ROOTAGENT_MODEL in .env or shell)
DEFAULT_MODEL = os.environ.get(
    "ROOTAGENT_MODEL"
)  # No fallback - requires configuration
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 8192

# Agent Defaults
AGENT_MAX_ITERATIONS = int(os.environ.get("ROOTAGENT_AGENT_MAX_ITERATIONS", "30"))
ORCHESTRATOR_MAX_ITERATIONS = int(
    os.environ.get("ROOTAGENT_ORCHESTRATOR_MAX_ITERATIONS", "50")
)

# File Extensions
KNOWLEDGE_TEXT_EXTENSIONS = [".txt", ".md"]
KNOWLEDGE_DATA_EXTENSIONS = [".json"]

# MCP Transport Types
MCP_TRANSPORT_STDIO = "stdio"
MCP_TRANSPORT_SSE = "sse"

# Exit Commands
EXIT_COMMANDS = ["exit", "quit", "q", "bye"]
