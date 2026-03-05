"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


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
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_atomic_memories",
            "description": "Save extracted atomic facts and preferences to vector memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "The fact or preference text."},
                                "category": {"type": "string", "enum": ["fact", "preference", "task"], "description": "The type of memory."},
                                "importance": {"type": "number", "minimum": 0, "maximum": 1, "description": "Subjective importance (0-1)."}
                            },
                            "required": ["content", "category"]
                        }
                    }
                },
                "required": ["memories"]
            }
        }
    }
]


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
        vector_store: Any = None,
        user_id: str = "default",
        session_id: str | None = None,
    ) -> bool:
        """Consolidate old messages into persistent memory via LLM tool call.

        If vector_store is provided, it extracts atomic facts. Otherwise, it updates MEMORY.md.
        Returns True on success, False on failure.
        """
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

        # Process in chunks of 50 messages to avoid context overflow and keep LLM focused
        CHUNK_SIZE = 50
        chunks = [old_messages[i:i + CHUNK_SIZE] for i in range(0, len(old_messages), CHUNK_SIZE)]
        
        success = True
        for idx, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.info("Consolidating chunk {}/{} ({} messages)", idx + 1, len(chunks), len(chunk))
            
            lines = []
            for m in chunk:
                if not m.get("content"):
                    continue
                tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
                lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

            if not lines:
                continue

            if vector_store:
                # Vector-based consolidation (Atomic Facts)
                prompt = f"""Process this conversation segment and extract new atomic facts or user preferences.
Focus on information that is worth remembering long-term.
You MUST call 'save_atomic_memories' to store your findings.

## Conversation Segment to Process
{chr(10).join(lines)}"""
                tool_choice = {"type": "function", "function": {"name": "save_atomic_memories"}}
            else:
                # File-based consolidation (Markdown)
                current_memory = self.read_long_term()
                prompt = f"""Process this conversation segment and call the 'save_memory' tool with your consolidation.

## Current Long-term Memory (for context)
{current_memory or "(empty)"}

## Conversation Segment to Process
{chr(10).join(lines)}"""
                tool_choice = {"type": "function", "function": {"name": "save_memory"}}

            try:
                response = await provider.chat(
                    messages=[
                        {"role": "system", "content": "You are a precise memory consolidation agent. You always use the provided tools to save extracted information."},
                        {"role": "user", "content": prompt},
                    ],
                    tools=_SAVE_MEMORY_TOOL,
                    tool_choice=tool_choice,
                    model=model,
                )

                if not response.has_tool_calls:
                    logger.warning("Memory consolidation (chunk {}): LLM did not call a tool, skipping this chunk", idx + 1)
                    success = False
                    continue

                for tool_call in response.tool_calls:
                    args = tool_call.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error("Failed to parse tool arguments as JSON: {}", args)
                            continue

                    if tool_call.name == "save_atomic_memories" and vector_store:
                        memories = args.get("memories", [])
                        for m in memories:
                            await vector_store.add_memory(
                                user_id=user_id,
                                session_id=session_id,
                                content=m["content"],
                                category=m["category"],
                                metadata={"importance": m.get("importance", 1.0)}
                            )
                        logger.info("Chunk {}: stored {} atomic memories in pgvector", idx + 1, len(memories))
                    
                    elif tool_call.name == "save_memory":
                        if entry := args.get("history_entry"):
                            self.append_history(entry)
                        if update := args.get("memory_update"):
                            self.write_long_term(update)
                        logger.info("Chunk {}: updated MEMORY.md/HISTORY.md", idx + 1)

            except Exception:
                logger.exception("Memory consolidation failed for chunk {}", idx + 1)
                success = False

        if success:
            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
        
        return success
