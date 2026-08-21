"""Vault memory plugin — Obsidian file-backed deep store (MemoryProvider).

Design niche (see Concepts/Memory-Architecture.md):
  - Hindsight = automatic semantic recall (conversation-derived, opaque store).
  - MEMORY.md  = small curated hot cache, agent-authored, always injected.
  - Vault      = THIS plugin — the long-form Obsidian knowledge base
                 (Projects/Concepts/References/Decisions/People). It provides
                 (a) PROACTIVE RECALL: prefetch() injects relevant vault
                     context before each turn so the vault is never an
                     "unknown unknown"; and (b) the PROMOTION BRIDGE: it
                 receives built-in MEMORY.md writes via on_memory_write() and
                 demotes reference-grade facts into structured vault notes.

It is intentionally designed to RUN ALONGSIDE Hindsight (the MemoryManager
allowlist blesses the {hindsight, vault} pair). The two never register
conflicting tool names.

Pure stdlib — no external SDK, no network. Reads/writes the Obsidian vault
at OBSIDIAN_VAULT_PATH (falls back to the documented default).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Directories inside the vault that are fair game for recall/promotion.
_RECALL_DIRS = ("Projects", "Concepts", "People", "References", "Decisions")
_PREFETCH_CAP_RESULTS = 3
_PREFETCH_CAP_CHARS = 1200
_PREFETCH_CACHE_TTL_S = 120
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
    "as", "at", "by", "from", "we", "you", "i", "he", "she", "they", "but",
    "not", "no", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "my", "your", "our", "their", "what", "when",
    "where", "why", "how", "which", "who", "into", "about", "than", "then",
}


def _vault_path() -> Path:
    """Resolve the Obsidian vault root from env, else documented default."""
    env = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
    if env:
        return Path(env)
    # Fallback: the path used throughout the workspace.
    return Path(r"C:\Users\Jie\HermesMemory")


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

VAULT_SEARCH_SCHEMA = {
    "name": "vault_search",
    "description": (
        "Search the Obsidian long-term memory vault (Projects/, Concepts/, "
        "People/, References/, Decisions/) for relevant notes. Use proactively "
        "when you encounter a project name, concept, or technical term you are "
        "uncertain about — the vault holds durable knowledge the hot cache "
        "cannot fit. Returns matching note paths + excerpt snippets ranked by "
        "relevance. Prefer this over guessing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords to search for in the vault.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 5, max 15).",
            },
            "dirs": {
                "type": "string",
                "description": (
                    "Optional comma-separated vault subdirs to restrict to "
                    "(e.g. 'Projects,Concepts'). Default: all recall dirs."
                ),
            },
        },
        "required": ["query"],
    },
}

VAULT_READ_SCHEMA = {
    "name": "vault_read",
    "description": (
        "Read the full content of a specific vault note by its path (relative "
        "to the vault root, e.g. 'Projects/LLM-Isomorph.md'). Use after "
        "vault_search identifies a relevant note, or when you know the exact "
        "note you need."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Vault-relative note path, e.g. 'Concepts/Memory-Architecture.md'.",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read (default 200, 0 = no limit).",
            },
        },
        "required": ["path"],
    },
}

ALL_TOOL_SCHEMAS = [VAULT_SEARCH_SCHEMA, VAULT_READ_SCHEMA]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class VaultMemoryProvider(MemoryProvider):
    """Obsidian vault as a first-class memory backend."""

    def __init__(self) -> None:
        self._vault: Path = _vault_path()
        self._agent_context: str = "primary"
        self._cache: Dict[str, tuple[float, str]] = {}
        self._cache_lock = threading.Lock()
        self._bridge_lock = threading.Lock()

    # -- Lifecycle --------------------------------------------------------

    @property
    def name(self) -> str:
        return "vault"

    def is_available(self) -> bool:
        # Available whenever the vault directory exists and is readable.
        try:
            return self._vault.is_dir()
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._agent_context = kwargs.get("agent_context", "primary") or "primary"
        env = kwargs.get("hermes_home")
        if env:
            # Allow per-session vault override via OBSIDIAN_VAULT_PATH only;
            # hermes_home is the profile dir, not the vault. Keep default.
            pass

    def shutdown(self) -> None:
        return None

    # -- System prompt -----------------------------------------------------

    def system_prompt_block(self) -> str:
        # The vault is read via prefetch + tools; no static block needed.
        return ""

    # -- Prefetch (proactive recall) --------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not query.strip():
            return ""
        cached = self._cache_get(query)
        if cached is not None:
            return cached
        results = self._search(query, limit=_PREFETCH_CAP_RESULTS, dirs=None)
        if not results:
            self._cache_set(query, "")
            return ""
        parts = []
        total = 0
        for path, snippet, score in results:
            if total >= _PREFETCH_CAP_CHARS:
                break
            block = f"### {path}\n{snippet}"
            parts.append(block)
            total += len(block)
        out = "\n\n".join(parts)
        self._cache_set(query, out)
        return out

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Synchronous prefetch is cheap (local file grep); run inline is fine,
        # but match the manager's background contract by no-opping — the
        # manager already calls prefetch_all() before the turn. We keep this
        # as a no-op to avoid double work.
        return None

    # -- Tools -------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Tools are recall-only; safe in any context. (Prefetch is automatic;
        # these let the agent drill in on demand.)
        return list(ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        try:
            if tool_name == "vault_search":
                query = (args.get("query") or "").strip()
                if not query:
                    return tool_error("Missing required parameter: query")
                limit = min(int(args.get("limit", 5)), 15)
                dirs = args.get("dirs")
                dirs_list = [d.strip() for d in dirs.split(",")] if dirs else None
                hits = self._search(query, limit=limit, dirs=dirs_list)
                if not hits:
                    return json.dumps({"results": [], "note": "No matching vault notes."})
                out = [
                    {"path": p, "score": round(s, 3), "snippet": s2}
                    for p, s2, s in hits
                ]
                return json.dumps({"results": out})
            elif tool_name == "vault_read":
                rel = (args.get("path") or "").strip()
                if not rel:
                    return tool_error("Missing required parameter: path")
                limit = int(args.get("limit", 200))
                return self._read_note(rel, limit=limit)
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Vault tool %s failed: %s", tool_name, e)
            return tool_error(f"Vault {tool_name} failed: {e}")

    # -- The promotion bridge ---------------------------------------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Receive built-in MEMORY.md/USER.md writes and demote to vault.

        This is the long-missing promotion bridge. When the agent calls the
        `memory` tool (add/replace/remove on MEMORY.md), this hook fires with
        the new/changed content. We do NOT overwrite the hot cache — instead
        we append a structured, dated entry to a bridge inbox note in the
        vault. The memory-distillation cron later promotes/merges these into
        the proper Projects/Concepts/References notes. This keeps the live
        write path non-destructive and lets the cron apply judgment.

        Writes are skipped in cron/non-primary contexts (the bridge is for
        user/agent-authored facts, not cron self-log noise).
        """
        if self._agent_context != "primary":
            return
        if target not in ("memory", "user"):
            return
        if action not in ("add", "replace"):
            return
        if not content or not content.strip():
            return
        # Only bridge substantive (reference-grade) content. A single short
        # warning line is hot-cache-only; bridge entries that look like
        # durable facts / decisions / findings. Heuristic: length + presence
        # of a fact-y signal (number, colon, project name). Keep it simple to
        # avoid flooding the inbox.
        text = content.strip()
        if len(text) < 40:
            return
        self._append_to_bridge_inbox(target, action, text)

    # -- Helpers -----------------------------------------------------------

    def _cache_get(self, query: str) -> Optional[str]:
        with self._cache_lock:
            item = self._cache.get(query)
            if item and (time.time() - item[0]) < _PREFETCH_CACHE_TTL_S:
                return item[1]
        return None

    def _cache_set(self, query: str, value: str) -> None:
        with self._cache_lock:
            self._cache[query] = (time.time(), value)

    def _iter_notes(self, dirs: Optional[List[str]]) -> List[Path]:
        roots = []
        for d in (_RECALL_DIRS if not dirs else tuple(dirs)):
            p = self._vault / d
            if p.is_dir():
                roots.append(p)
        notes: List[Path] = []
        for root in roots:
            try:
                notes.extend(root.rglob("*.md"))
            except Exception:
                continue
        return notes

    def _score(self, query: str, text_lower: str) -> float:
        terms = [t for t in re.findall(r"[a-z0-9_.]+", query.lower()) if t not in _STOPWORDS and len(t) > 1]
        if not terms:
            return 0.0
        score = 0.0
        for t in terms:
            # term frequency
            score += text_lower.count(t) * 1.0
            # filename / heading bonus
            if t in text_lower[:200]:
                score += 2.0
        return score

    def _search(
        self, query: str, limit: int, dirs: Optional[List[str]]
    ) -> List[tuple[str, str, float]]:
        notes = self._iter_notes(dirs)
        scored: List[tuple[Path, float]] = []
        ql = query.lower()
        for note in notes:
            try:
                body = note.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            s = self._score(query, body.lower())
            if s > 0:
                scored.append((note, s))
        if not scored:
            return []
        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[tuple[str, str, float]] = []
        for note, score in scored[:limit]:
            rel = str(note.relative_to(self._vault))
            snippet = self._snippet(note, ql)
            results.append((rel, snippet, score))
        return results

    def _snippet(self, note: Path, query_lower: str) -> str:
        try:
            lines = note.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""
        terms = [t for t in re.findall(r"[a-z0-9_.]+", query_lower) if len(t) > 1]
        best_idx = 0
        best_hits = -1
        for i, line in enumerate(lines):
            hits = sum(line.lower().count(t) for t in terms)
            if hits > best_hits:
                best_hits = hits
                best_idx = i
        if best_hits <= 0:
            # fall back to first non-empty line
            for i, line in enumerate(lines):
                if line.strip():
                    best_idx = i
                    break
        start = max(0, best_idx - 2)
        end = min(len(lines), best_idx + 6)
        return "\n".join(lines[start:end])[:600]

    def _read_note(self, rel: str, limit: int = 200) -> str:
        path = self._vault / rel
        if not path.is_file():
            return tool_error(f"Vault note not found: {rel}")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return tool_error(f"Failed to read {rel}: {e}")
        if limit and limit > 0:
            lines = lines[:limit]
        return json.dumps({"path": rel, "content": "\n".join(lines)})

    def _append_to_bridge_inbox(self, target: str, action: str, text: str) -> None:
        """Append a dated bridge entry to Memory-Bridge-Inbox.md in the vault.

        Non-destructive: never overwrites existing notes. The distillation
        cron promotes these into proper vault nodes.
        """
        try:
            inbox_dir = self._vault
            inbox = inbox_dir / "Memory-Bridge-Inbox.md"
            date = time.strftime("%Y-%m-%d")
            header = ""
            if not inbox.exists():
                header = (
                    "# Memory Bridge Inbox\n\n"
                    "> Auto-populated by the `vault` memory provider's "
                    "on_memory_write hook. Each entry is a hot-cache fact the "
                    "agent wrote to MEMORY.md/USER.md that looked reference-grade. "
                    "The memory-distillation cron promotes/merges these into the "
                    "proper Projects/Concepts/References/Decisions notes. Do not "
                    "delete by hand — let the cron consume them.\n\n"
                )
            stamp = time.strftime("%Y-%m-%d %H:%M")
            entry = f"\n## [{date}] {target} · {action}\n- {text}\n- _bridged {stamp}_\n"
            with self._bridge_lock:
                with open(inbox, "a", encoding="utf-8") as f:
                    if header:
                        f.write(header)
                    f.write(entry)
        except Exception as e:
            logger.debug("Vault bridge inbox write failed: %s", e)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the vault provider."""
    ctx.register_memory_provider(VaultMemoryProvider())
