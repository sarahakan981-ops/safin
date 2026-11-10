"""Tests for should_use_local_embeddings() decision logic."""

import os
from unittest.mock import patch

import pytest

from rootagent.knowledge.embeddings import should_use_local_embeddings


class TestShouldUseLocalEmbeddings:
    def _env(self, **overrides):
        """Build a clean env dict with only the keys we care about."""
        base = {k: v for k, v in os.environ.items()
                if k not in ("ROOTAGENT_EMBEDDINGS", "OPENAI_API_BASE",
                              "OPENAI_BASE_URL", "OPENAI_API_KEY")}
        base.update(overrides)
        return base

    def test_explicit_local_always_true(self):
        with patch.dict(os.environ, self._env(
            ROOTAGENT_EMBEDDINGS="local",
            OPENAI_API_KEY="sk-123",
        ), clear=True):
            assert should_use_local_embeddings() is True

    def test_explicit_openai_always_false(self):
        with patch.dict(os.environ, self._env(
            ROOTAGENT_EMBEDDINGS="openai",
        ), clear=True):
            assert should_use_local_embeddings() is False

    def test_no_key_defaults_to_local(self):
        with patch.dict(os.environ, self._env(), clear=True):
            assert should_use_local_embeddings() is True

    def test_openai_key_set_defaults_to_remote(self):
        with patch.dict(os.environ, self._env(OPENAI_API_KEY="sk-test"), clear=True):
            assert should_use_local_embeddings() is False

    def test_official_openai_base_with_key_uses_remote(self):
        with patch.dict(os.environ, self._env(
            OPENAI_API_KEY="sk-test",
            OPENAI_API_BASE="https://api.openai.com/v1",
        ), clear=True):
            assert should_use_local_embeddings() is False

    def test_custom_api_base_forces_local_even_with_key(self):
        """A non-official relay likely doesn't support the embeddings endpoint."""
        with patch.dict(os.environ, self._env(
            OPENAI_API_KEY="sk-test",
            OPENAI_API_BASE="https://my-relay.example/v1",
        ), clear=True):
            assert should_use_local_embeddings() is True

    def test_explicit_override_beats_custom_base(self):
        """ROOTAGENT_EMBEDDINGS=openai should win even with a custom base."""
        with patch.dict(os.environ, self._env(
            ROOTAGENT_EMBEDDINGS="openai",
            OPENAI_API_BASE="https://my-relay.example/v1",
        ), clear=True):
            assert should_use_local_embeddings() is False
