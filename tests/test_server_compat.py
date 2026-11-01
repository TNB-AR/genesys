"""The stdio server registers its tools correctly on both mcp 1.x and 2.x.

``papez.server`` feature-detects the SDK's handler-registration API at import.
These tests exercise the registered handlers the way the SDK's runner does
(not the plain ``list_tools``/``call_tool`` functions other tests call), so a
regression in either adapter shows up regardless of which SDK is installed.
The last test drives a real ``python -m papez`` subprocess over stdio.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from typing import Any

import mcp.types as t
import pytest
from mcp.server import Server

import papez.server as srv

# ``website_url`` and ``inputSchema`` validation in the call_tool decorator both
# arrived partway through the 1.x line; the floor (mcp 1.0) has neither.
_SERVER_ACCEPTS_WEBSITE_URL = "website_url" in inspect.signature(Server.__init__).parameters
_SDK_VALIDATES_INPUT = srv.SDK_TOOL_API == "request_handlers" or (
    hasattr(Server, "call_tool") and "validate_input" in inspect.signature(Server.call_tool).parameters
)
requires_input_validation = pytest.mark.skipif(
    not _SDK_VALIDATES_INPUT, reason="this mcp release's call_tool decorator does not validate inputSchema"
)

CORE_TOOLS = {
    "memory_store", "memory_amend", "memory_recall", "memory_search", "memory_traverse",
    "memory_explain", "pin_memory", "unpin_memory", "list_core_memories", "delete_memory",
    "memory_stats", "set_core_preferences", "promote_to_org",
}
# Mind Spaces (spec 6.8). Count assertions use len(srv._TOOL_DEFS) (D31).
SPACE_TOOLS = {
    "space_create", "space_list", "space_get", "space_update", "space_member_add",
    "space_member_remove", "space_member_list", "space_archive", "space_unarchive",
    "space_erase", "promote_to_space",
}
EXPECTED_TOOLS = CORE_TOOLS | SPACE_TOOLS


def _wire(model: Any) -> dict[str, Any]:
    """Model as MCP wire JSON (camelCase). Attribute names differ between mcp 1.x and 2.x."""
    dumped: dict[str, Any] = model.model_dump(by_alias=True, exclude_none=True)
    return dumped


async def _list_tools_via_sdk() -> Any:
    if srv.SDK_TOOL_API == "decorators":
        handler = srv.app.request_handlers[t.ListToolsRequest]
        return (await handler(t.ListToolsRequest(method="tools/list"))).root
    entry = srv.app.get_request_handler("tools/list")
    assert entry is not None
    return await entry.handler(None, t.PaginatedRequestParams())


async def _call_tool_via_sdk(name: str, arguments: dict[str, Any]) -> Any:
    params = t.CallToolRequestParams(name=name, arguments=arguments)
    if srv.SDK_TOOL_API == "decorators":
        handler = srv.app.request_handlers[t.CallToolRequest]
        return (await handler(t.CallToolRequest(method="tools/call", params=params))).root
    entry = srv.app.get_request_handler("tools/call")
    assert entry is not None
    return await entry.handler(None, params)


class TestRegistration:
    def test_sdk_api_is_feature_detected(self):
        legacy = hasattr(Server, "list_tools") and hasattr(Server, "call_tool")
        assert srv.SDK_TOOL_API == ("decorators" if legacy else "request_handlers")

    def test_server_identity(self):
        assert srv.app.name == "papez"
        opts = srv.app.create_initialization_options()
        assert opts.server_name == "papez"
        assert opts.capabilities.tools is not None
        if _SERVER_ACCEPTS_WEBSITE_URL:
            assert srv.app.website_url == srv.SERVER_WEBSITE_URL
            assert opts.website_url == srv.SERVER_WEBSITE_URL

    async def test_tools_list_exposes_every_tool_with_its_schema(self):
        result = await _list_tools_via_sdk()
        assert isinstance(result, t.ListToolsResult)
        by_name = {tool.name: tool for tool in result.tools}
        assert EXPECTED_TOOLS <= set(by_name)
        assert set(by_name) == {d["name"] for d in srv._TOOL_DEFS}
        assert len(result.tools) == len(srv._TOOL_DEFS)
        for declared in srv._TOOL_DEFS:
            served = _wire(by_name[declared["name"]])
            assert served["inputSchema"] == declared["inputSchema"]
            assert served["description"] == declared["description"]


class TestDispatchMatchesDefs:
    """Every schema property is dispatched and every dispatched arg is in the schema (D31)."""

    def test_same_tool_names(self):
        assert set(srv._TOOL_DISPATCH) == {d["name"] for d in srv._TOOL_DEFS}

    @pytest.mark.parametrize("definition", srv._TOOL_DEFS, ids=lambda d: d["name"])
    def test_properties_and_required_match_dispatch(self, definition):
        method, required, optional = srv._TOOL_DISPATCH[definition["name"]]
        schema = definition["inputSchema"]
        props = set(schema.get("properties", {}))
        assert props == set(required) | set(optional), definition["name"]
        assert set(schema.get("required", [])) == set(required), definition["name"]
        params = inspect.signature(method).parameters
        for arg in (*required, *optional):
            assert arg in params, f"{definition['name']}: handler has no parameter {arg!r}"

    @pytest.mark.parametrize("definition", srv._TOOL_DEFS, ids=lambda d: d["name"])
    def test_no_none_anywhere_in_schema(self, definition):
        def walk(value: Any) -> None:
            assert value is not None, definition["name"]
            if isinstance(value, dict):
                for v in value.values():
                    walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)

        walk(definition["inputSchema"])

    def test_include_personal_dispatch_default_is_none_without_json_default(self):
        for name in ("memory_recall", "memory_search", "memory_traverse"):
            _, _, optional = srv._TOOL_DISPATCH[name]
            assert optional["include_personal"] is None
            schema = next(d for d in srv._TOOL_DEFS if d["name"] == name)["inputSchema"]
            assert "default" not in schema["properties"]["include_personal"]

    def test_read_space_tools_are_retryable_and_writes_are_not(self):
        assert {"space_list", "space_get", "space_member_list"} <= srv._RETRYABLE_TOOLS
        for name in ("space_create", "space_update", "space_member_add", "space_member_remove",
                     "space_archive", "space_unarchive", "space_erase", "promote_to_space"):
            assert name not in srv._RETRYABLE_TOOLS

    def test_stdio_constants(self):
        assert srv.STDIO_LOCAL_USER == "stdio_local_user"
        assert srv.STDIO_LOCAL_ORG == "stdio_local_org"


class TestCallToolParity:
    @requires_input_validation
    async def test_missing_required_argument_is_an_sdk_validation_error(self):
        result = await _call_tool_via_sdk("memory_explain", {})
        assert isinstance(result, t.CallToolResult)
        assert _wire(result)["isError"] is True
        assert result.content[0].text == "Input validation error: 'node_id' is a required property"

    @requires_input_validation
    async def test_wrong_argument_type_is_an_sdk_validation_error(self):
        result = await _call_tool_via_sdk("memory_recall", {"query": "x", "k": "ten"})
        assert _wire(result)["isError"] is True
        assert result.content[0].text.startswith("Input validation error: ")

    async def test_unknown_tool_is_a_papez_error_payload(self):
        result = await _call_tool_via_sdk("not_a_tool", {})
        assert _wire(result).get("isError", False) is False
        payload = json.loads(result.content[0].text)
        assert payload == {"error": "Unknown tool: not_a_tool", "retryable": False}

    async def test_tool_exception_becomes_error_payload_not_protocol_error(self, monkeypatch):
        async def boom(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setitem(srv._TOOL_DISPATCH, "memory_stats", (boom, [], {}))
        result = await _call_tool_via_sdk("memory_stats", {})
        assert _wire(result).get("isError", False) is False
        payload = json.loads(result.content[0].text)
        assert payload == {"error": "RuntimeError: boom", "retryable": True}

    async def test_successful_call_returns_json_text(self):
        result = await _call_tool_via_sdk("memory_stats", {})
        assert _wire(result).get("isError", False) is False
        assert result.content[0].type == "text"
        assert isinstance(json.loads(result.content[0].text), dict)


class TestStdioRoundTrip:
    async def test_python_m_papez_serves_tools_over_stdio(self, tmp_path):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        import papez

        env = {
            # Run the SAME source tree as this test session (a worktree or a
            # checkout other than the editable install's).
            "PYTHONPATH": os.path.dirname(os.path.dirname(papez.__file__)),
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(tmp_path)),
            "GENESYS_PERSIST_PATH": str(tmp_path / "memories.json"),
            "GENESYS_MAINTENANCE_INTERVAL_S": "0",
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
        }
        params = StdioServerParameters(command=sys.executable, args=["-m", "papez"], env=env, cwd=str(tmp_path))

        async def scenario() -> None:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                info = _wire(await session.initialize())["serverInfo"]
                assert info["name"] == "papez"
                if "websiteUrl" in info:  # only SDKs whose Server accepts website_url advertise it
                    assert info["websiteUrl"] == srv.SERVER_WEBSITE_URL

                listed = await session.list_tools()
                assert {tool.name for tool in listed.tools} == {d["name"] for d in srv._TOOL_DEFS}
                assert EXPECTED_TOOLS <= {tool.name for tool in listed.tools}

                # Mind Spaces over stdio: the local user creates a space in the
                # local org, writes into it, and the next call already sees it.
                created = json.loads((await session.call_tool("space_create", {"name": "Round Trip"})).content[0].text)
                assert created["space"]["slug"] == "round-trip"
                assert created["space"]["org_id"] == srv.STDIO_LOCAL_ORG
                assert created["space"]["your_role"] == "owner"
                # The provider factory falls back to keyword-only recall when
                # no embedder can run (no OpenAI key, no sentence-transformers),
                # so store/recall work over stdio on every machine.
                stored = json.loads((await session.call_tool(
                    "memory_store", {"content": "round trip space memory", "space": "round-trip", "tags": ["Trip"]},
                )).content[0].text)
                assert stored["status"] == "stored"
                assert stored["space"]["slug"] == "round-trip"
                listed_spaces = json.loads((await session.call_tool("space_list", {})).content[0].text)
                assert [s["memory_count"] for s in listed_spaces["spaces"]] == [1]
                recalled = json.loads((await session.call_tool(
                    "memory_recall", {"query": "round trip", "space": "round-trip"},
                )).content[0].text)
                assert [r["id"] for r in recalled["results"]] == [stored["node_id"]]
                assert recalled["results"][0]["tags"] == ["trip"]
                missing = json.loads((await session.call_tool(
                    "memory_recall", {"query": "x", "space": "nope"},
                )).content[0].text)
                assert missing == {"error": "Space not found", "code": "space_not_found"}

                ok = await session.call_tool("memory_stats", {})
                assert _wire(ok).get("isError", False) is False
                assert isinstance(json.loads(ok.content[0].text), dict)

                if _SDK_VALIDATES_INPUT:
                    bad = await session.call_tool("memory_explain", {})
                    assert _wire(bad)["isError"] is True
                    assert bad.content[0].text == "Input validation error: 'node_id' is a required property"

                unknown = await session.call_tool("not_a_tool", {})
                assert _wire(unknown).get("isError", False) is False
                assert json.loads(unknown.content[0].text) == {"error": "Unknown tool: not_a_tool", "retryable": False}

        await asyncio.wait_for(scenario(), timeout=60)
        # The registry sidecar is written lazily, on the first mutation.
        assert (tmp_path / "memories.spaces.json").exists()
