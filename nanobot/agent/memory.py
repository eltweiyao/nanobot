"""Memory system for persistent agent memory, supporting both file-based and pgvector-based storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg2
from loguru import logger
from psycopg2.extras import RealDictCursor

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session
    from nanobot.config.schema import VectorMemoryConfig


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                    "atomic_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: list of standalone, important facts extracted from this conversation.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class PgVectorStore:
    """Backend for pgvector-based memory storage."""

    def __init__(self, db_url: str, user_id: str):
        self.db_url = db_url
        self.user_id = user_id
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
        return self._conn

    def search_facts(self, embedding: list[float], limit: int = 5) -> list[str]:
        """Search for relevant facts using cosine similarity."""
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT content FROM agent_facts "
                    "WHERE user_id = %s "
                    "ORDER BY embedding <=> %s::vector "
                    "LIMIT %s",
                    (self.user_id, embedding, limit),
                )
                return [row["content"] for row in cur.fetchall()]
        except Exception:
            logger.exception("Failed to search facts in pgvector")
            return []

    def add_fact(self, content: str, embedding: list[float], category: str = "general"):
        """Add or update an atomic fact."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_facts (user_id, content, embedding, category) "
                    "VALUES (%s, %s, %s::vector, %s)",
                    (self.user_id, content, embedding, category),
                )
            conn.commit()
        except Exception:
            logger.exception("Failed to add fact to pgvector")

    def add_event(self, summary: str, embedding: list[float], session_id: str | None = None):
        """Add a conversation event summary."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_events (user_id, summary, embedding, session_id) "
                    "VALUES (%s, %s, %s::vector, %s)",
                    (self.user_id, summary, embedding, session_id),
                )
            conn.commit()
        except Exception:
            logger.exception("Failed to add event to pgvector")


class MemoryStore:
    """Two-layer memory: Vector/File facts + History log, with multi-user isolation."""

    def __init__(self, workspace: Path, vector_config: VectorMemoryConfig | None = None, user_id: str = "default"):
        self.workspace = workspace
        self.user_id = user_id
        self.vector_config = vector_config
        
        # Isolate file storage by user_id
        self.memory_root = ensure_dir(workspace / "memory")
        if user_id == "default":
            self.memory_dir = self.memory_root
        else:
            # Create a subfolder for the specific user (e.g. workspace/memory/user_123/)
            self.memory_dir = ensure_dir(self.memory_root / user_id)
            
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        
        self.vector_store = None
        if vector_config and vector_config.enabled:
            self.vector_store = PgVectorStore(vector_config.db_url, user_id)

    def read_long_term(self) -> str:
        """Read fallback MEMORY.md file."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    async def get_memory_context(self, provider: LLMProvider | None = None, query: str | None = None) -> str:
        """Get combined memory context. Uses vector search if query is provided."""
        parts = []

        # 1. Semantic Facts (Vector)
        if self.vector_store and provider and query and self.vector_config:
            try:
                embedding = await provider.embed(
                    query, 
                    model=self.vector_config.embedding_model,
                    api_key=self.vector_config.embedding_api_key,
                    api_base=self.vector_config.embedding_api_base
                )
                facts = self.vector_store.search_facts(embedding, limit=self.vector_config.limit)
                if facts:
                    parts.append("### Relevant Facts (from memory)\n- " + "\n- ".join(facts))
            except Exception as e:
                logger.warning("Vector memory search failed: {}", e)

        # 2. Legacy Long-term Facts (Always load if small, or as fallback)
        long_term = self.read_long_term()
        if long_term:
            parts.append(f"### Core Facts\n{long_term}")

        return "\n\n".join(parts) if parts else ""

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into persistent storage via LLM tool call."""
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation (archive_all): {} messages", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return True
            logger.info("Memory consolidation: {} to consolidate, {} keep", len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation. Identify any new atomic facts for long-term storage."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)

            # 1. Update History (Legacy)
            if entry := args.get("history_entry"):
                self.append_history(entry)
                # Also save to vector if enabled
                if self.vector_store and self.vector_config:
                    embedding = await provider.embed(
                        entry, 
                        model=self.vector_config.embedding_model,
                        api_key=self.vector_config.embedding_api_key,
                        api_base=self.vector_config.embedding_api_base
                    )
                    self.vector_store.add_event(entry, embedding, session_id=session.key)

            # 2. Update Long-term (Legacy File)
            if update := args.get("memory_update"):
                if update != current_memory:
                    self.write_long_term(update)

            # 3. Save Atomic Facts (New Vector feature)
            if self.vector_store and (facts := args.get("atomic_facts")) and self.vector_config:
                for fact in facts:
                    embedding = await provider.embed(
                        fact, 
                        model=self.vector_config.embedding_model,
                        api_key=self.vector_config.embedding_api_key,
                        api_base=self.vector_config.embedding_api_base
                    )
                    self.vector_store.add_fact(fact, embedding)

            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info("Memory consolidation done: {} messages", len(old_messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False
