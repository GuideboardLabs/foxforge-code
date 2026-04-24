from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from shared_tools.model_routing import load_model_routing

from . import tools

try:  # pragma: no cover - optional dependency
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover
    FastMCP = None


def create_server(repo_root: Path) -> Any:
    if FastMCP is None:
        raise RuntimeError("MCP SDK is not installed. Install `mcp==1.27.*` to enable server mode.")

    root = Path(repo_root).resolve()
    server = FastMCP("foxforge")

    @server.tool()
    def forage(query: str, depth: str = "quick") -> dict[str, Any]:
        return tools.forage(root, query=query, depth=depth)

    @server.tool()
    def recall(
        query: str,
        kinds: list[str] | None = None,
        project: str = "general",
        conversation_id: str = "",
    ) -> dict[str, Any]:
        return tools.recall(
            root,
            query=query,
            kinds=kinds,
            project=project,
            conversation_id=conversation_id,
        )

    @server.tool()
    def make_artifact(type_id: str, spec: str) -> dict[str, Any]:
        return tools.make_artifact(root, type_id=type_id, spec=spec)

    return server


def run_stdio(repo_root: Path) -> None:
    server = create_server(repo_root)
    server.run()


def run_http(repo_root: Path, *, host: str = "127.0.0.1", port: int = 9876) -> None:
    server = create_server(repo_root)
    routing = load_model_routing(Path(repo_root))
    ack = bool(routing.get("mcp.acknowledge_unsafe_http", False)) if isinstance(routing, dict) else False
    if not ack:
        raise RuntimeError(
            "HTTP MCP transport requires explicit acknowledgement: set "
            "`mcp.acknowledge_unsafe_http=true` in config.ini ([inference])."
        )
    token_path = Path(repo_root) / "Runtime" / "state" / "mcp_token"
    token = token_path.read_text(encoding="utf-8").strip() if token_path.exists() else ""
    if not token:
        raise RuntimeError("HTTP MCP transport requires Runtime/state/mcp_token bearer token file.")
    os.environ.setdefault("FOXFORGE_MCP_BEARER_TOKEN", token)
    server.run(transport="streamable-http", host=host, port=int(port))
