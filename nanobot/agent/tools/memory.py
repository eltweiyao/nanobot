"""Memory search tools for semantic retrieval."""

from typing import Any, Optional

from nanobot.agent.tools.base import Tool
from nanobot.agent.vector_memory import VectorMemoryStore


class SearchMemoryTool(Tool):
    """Tool to search through semantic memory."""

    def __init__(self, vector_memory: Optional[VectorMemoryStore] = None):
        self._vector_memory = vector_memory
        self._channel: str = "default"
        self._chat_id: Optional[str] = None

    def set_context(self, channel: str, chat_id: str, *args: Any, **kwargs: Any) -> None:
        """Update routing context."""
        self._channel = channel
        self._chat_id = chat_id

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return (
            "Search through long-term semantic memory for relevant facts, preferences, or past interactions. "
            "Use this when you need to recall specific information that isn't in the current conversation history."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "The search query (e.g. 'What is the user's favorite programming language?')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, limit: int = 5, **kwargs: Any) -> str:
        if not self._vector_memory:
            return "Error: Vector memory is not enabled in this workspace."

        try:
            results = await self._vector_memory.query_relevant_memories(
                user_id=self._channel,
                query=query,
                session_id=self._chat_id,
                limit=limit
            )
            
            if not results:
                return f"No relevant memories found for query: '{query}'"
                
            return f"Found relevant memories:\n\n{results}"
        except Exception as e:
            return f"Error searching memory: {str(e)}"
