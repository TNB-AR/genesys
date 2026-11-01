"""WS-verify: the provider factory wires spaces/audit and degrades without an embedder."""
from __future__ import annotations

import importlib.util

import papez.providers as providers


def test_local_embedder_needs_sentence_transformers(monkeypatch):
    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        if name == "sentence_transformers":
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    for choice in ("local", "openai"):
        monkeypatch.setenv("GENESYS_EMBEDDER", choice)
        # Keyword-only recall instead of ModuleNotFoundError on the first embed().
        assert providers._make_embedder() is None


def test_get_providers_wires_spaces_and_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(providers, "_instance", None)
    monkeypatch.setenv("GENESYS_PERSIST_PATH", str(tmp_path / "memories.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    built = providers.get_providers()
    try:
        assert built.tools.spaces is built.spaces
        assert built.tools.audit is built.audit
    finally:
        providers._instance = None
