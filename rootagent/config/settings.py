"""Application settings for RootAgent."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .constants import (
    AGENT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    get_openai_api_base,
)


@dataclass
class Settings:
    """Application settings."""

    # LLM Settings
    model: str = field(default_factory=lambda: DEFAULT_MODEL)
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_context_tokens: int = 128000

    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    # Optional OpenAI-compatible API base URL (e.g. a relay or local inference server).
    # Reads OPENAI_API_BASE or OPENAI_BASE_URL; None means use the provider default.
    openai_api_base: Optional[str] = field(default_factory=get_openai_api_base)
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    # Paths
    knowledge_path: Path = field(default_factory=lambda: Path("knowledge"))
    mcp_config_path: Path = field(default_factory=lambda: Path("mcp.json"))

    # Docker Settings
    container_name: str = "rootagent-sandbox"
    docker_image: str = "ghcr.io/sarahakan981-ops/rootagent:kali"

    # Agent Settings
    max_iterations: int = AGENT_MAX_ITERATIONS

    # VPN Settings
    vpn_config_path: Optional[Path] = None

    # Interface Settings
    default_interface: str = "tui"  # "tui" or "cli"

    # Prompt Modules
    prompt_modules: List[str] = field(default_factory=list)

    # Target Settings
    target: Optional[str] = None
    scope: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        if isinstance(self.knowledge_path, str):
            self.knowledge_path = Path(self.knowledge_path)
        if isinstance(self.mcp_config_path, str):
            self.mcp_config_path = Path(self.mcp_config_path)
        if isinstance(self.vpn_config_path, str):
            self.vpn_config_path = Path(self.vpn_config_path)

    def __repr__(self) -> str:
        # Mask API keys so they never appear in logs or tracebacks.
        return (
            f"Settings(model={self.model!r}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, max_context_tokens={self.max_context_tokens}, "
            f"openai_api_key={'***' if self.openai_api_key else None}, "
            f"anthropic_api_key={'***' if self.anthropic_api_key else None}, "
            f"target={self.target!r}, scope={self.scope!r}, "
            f"max_iterations={self.max_iterations})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def update_settings(**kwargs) -> Settings:
    """Update global settings with new values."""
    global _settings
    _settings = Settings(**kwargs)
    return _settings
