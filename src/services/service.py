"""
src/services/service.py — CMN-C1-042

Service-layer placeholder for scaffold-structure compliance. The domain logic for
this template lives in the node classes (src/nodes/); the LLM client is dependency-
injected into the graph at construction time (see src/graph/graph.py), and the CRM
schemas / nuance dictionary are loaded from config/ by the respective nodes. No
agenticstar imports.
"""

from __future__ import annotations

from typing import Any


def build_llm_client(config: dict[str, Any] | None = None) -> Any:
    """Return the production LLM client, or None to let StructuredExtractNode default.

    Placeholder seam: in production this constructs the LLM client from `config`
    (model, endpoint, credentials via InvocationContext); in CI/local it returns
    None and StructuredExtractNode falls back to its deterministic stub path.
    """
    return (config or {}).get("llm_client")
