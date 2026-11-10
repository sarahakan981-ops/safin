"""Tests for OpenAI-compatible API base handling in the LLM layer."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rootagent.llm.config import ModelConfig
from rootagent.llm.llm import LLM
from rootagent.llm.memory import ConversationMemory


class TestIsOpenAICompatibleModel:
    def _make_llm(self, model: str) -> LLM:
        llm = LLM.__new__(LLM)
        llm.model = model
        llm.config = ModelConfig()
        return llm

    def test_openai_prefix(self):
        assert self._make_llm("openai/my-relay-model")._is_openai_compatible_model()

    def test_gpt_prefix(self):
        assert self._make_llm("gpt-5")._is_openai_compatible_model()
        assert self._make_llm("gpt-4.1-mini")._is_openai_compatible_model()

    def test_o_series(self):
        assert self._make_llm("o1-mini")._is_openai_compatible_model()
        assert self._make_llm("o3-mini")._is_openai_compatible_model()
        assert self._make_llm("o4-preview")._is_openai_compatible_model()

    def test_chatgpt_prefix(self):
        assert self._make_llm("chatgpt-4o")._is_openai_compatible_model()

    def test_anthropic_not_openai(self):
        assert not self._make_llm("claude-sonnet-4-20250514")._is_openai_compatible_model()
        assert not self._make_llm("anthropic/claude-opus-4")._is_openai_compatible_model()

    def test_ollama_not_openai(self):
        assert not self._make_llm("ollama/llama3")._is_openai_compatible_model()

    def test_gemini_not_openai(self):
        assert not self._make_llm("gemini/gemini-2.5-flash")._is_openai_compatible_model()


class TestApplyApiBase:
    def _make_llm(self, model: str, api_base: str | None) -> LLM:
        llm = LLM.__new__(LLM)
        llm.model = model
        llm.config = ModelConfig.__new__(ModelConfig)
        llm.config.openai_api_base = api_base
        return llm

    def test_injects_api_base_for_openai_model(self):
        llm = self._make_llm("openai/my-model", "https://relay.example/v1")
        kwargs = {"model": "openai/my-model"}
        llm._apply_api_base(kwargs)
        assert kwargs["api_base"] == "https://relay.example/v1"

    def test_no_injection_when_api_base_is_none(self):
        llm = self._make_llm("openai/my-model", None)
        kwargs = {"model": "openai/my-model"}
        llm._apply_api_base(kwargs)
        assert "api_base" not in kwargs

    def test_no_injection_for_anthropic_model(self):
        llm = self._make_llm("claude-sonnet-4-20250514", "https://relay.example/v1")
        kwargs = {"model": "claude-sonnet-4-20250514"}
        llm._apply_api_base(kwargs)
        assert "api_base" not in kwargs

    def test_no_injection_for_ollama_model(self):
        llm = self._make_llm("ollama/llama3", "https://relay.example/v1")
        kwargs = {"model": "ollama/llama3"}
        llm._apply_api_base(kwargs)
        assert "api_base" not in kwargs


class TestGeneratePassesApiBase:
    @pytest.mark.asyncio
    async def test_generate_passes_openai_api_base_to_litellm(self):
        captured_kwargs = []

        async def fake_acompletion(**kwargs):
            captured_kwargs.append(kwargs)
            message = SimpleNamespace(content="ok", tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=None, model=kwargs["model"])

        llm = LLM.__new__(LLM)
        llm.model = "openai/my-model"
        llm.config = ModelConfig(openai_api_base="https://relay.example/v1")
        llm.memory = ConversationMemory(max_tokens=100_000)
        llm._litellm = SimpleNamespace(acompletion=fake_acompletion)

        response = await llm.generate(
            system_prompt="system",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert response.content == "ok"
        assert captured_kwargs[0].get("api_base") == "https://relay.example/v1"

    @pytest.mark.asyncio
    async def test_generate_does_not_pass_api_base_for_anthropic(self):
        captured_kwargs = []

        async def fake_acompletion(**kwargs):
            captured_kwargs.append(kwargs)
            message = SimpleNamespace(content="ok", tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=None, model=kwargs["model"])

        llm = LLM.__new__(LLM)
        llm.model = "claude-sonnet-4-20250514"
        llm.config = ModelConfig(openai_api_base="https://relay.example/v1")
        llm.memory = ConversationMemory(max_tokens=100_000)
        llm._litellm = SimpleNamespace(acompletion=fake_acompletion)

        await llm.generate(
            system_prompt="system",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert "api_base" not in captured_kwargs[0]
