"""Vector-based semantic memory using pgvector and DashScope."""

import asyncio
import dashscope
from http import HTTPStatus
from typing import List, Optional
from loguru import logger
from nanobot.utils.db import DatabaseManager

class VectorMemoryStore:
    """Store and retrieve memories using vector similarity search."""

    def __init__(self, db: DatabaseManager, api_key: str, model: str = "text-embedding-v3"):
        self.db = db
        self.api_key = api_key
        self.model = model
        # Note: dashscope.api_key is a global setting in the SDK
        dashscope.api_key = api_key

    async def get_embeddings(self, text: str) -> Optional[List[float]]:
        """Call DashScope text-embedding-v3 API (offloaded to thread)."""
        if not self.api_key:
            return None
            
        def _call():
            return dashscope.TextEmbedding.call(
                model=self.model,
                input=text
            )

        try:
            resp = await asyncio.to_thread(_call)
            if resp.status_code == HTTPStatus.OK:
                # text-embedding-v3 returns a list of embeddings for the input list
                return resp.output['embeddings'][0]['embedding']
            else:
                logger.error("DashScope embedding failed ({}): {}", resp.code, resp.message)
                return None
        except Exception as e:
            logger.error("DashScope embedding error: {}", e)
            return None

    async def add_memory(self, user_id: int, content: str):
        """Add a factual snippet to vector memories for a specific user."""
        if not self.db or not self.db.enabled or not self.db.pool:
            return

        embedding = await self.get_embeddings(content)
        if not embedding:
            return

        try:
            async with self.db.pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO vector_memories (user_id, content, embedding) VALUES ($1, $2, $3)",
                    user_id, content, embedding
                )
                logger.info("Added vector memory for user {}: {}", user_id, (content[:50] + "...") if len(content) > 50 else content)
        except Exception as e:
            logger.error("Failed to add vector memory: {}", e)

    async def search_memories(self, user_id: int, query: str, limit: int = 5) -> List[str]:
        """Retrieve relevant memory snippets for a user using vector similarity."""
        if not self.db or not self.db.enabled or not self.db.pool:
            return []

        embedding = await self.get_embeddings(query)
        if not embedding:
            return []

        try:
            async with self.db.pool.acquire() as conn:
                # ORDER BY embedding <=> $2::vector uses cosine distance (pgvector)
                rows = await conn.fetch(
                    """
                    SELECT content FROM vector_memories 
                    WHERE user_id = $1 
                    ORDER BY embedding <=> $2::vector 
                    LIMIT $3
                    """,
                    user_id, embedding, limit
                )
                return [row['content'] for row in rows]
        except Exception as e:
            logger.error("Failed to search vector memories: {}", e)
            return []
