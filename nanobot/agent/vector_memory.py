import json
import logging
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from nanobot.providers.base import LLMProvider

logger = logging.getLogger(__name__)

class VectorMemoryStore:
    """
    Tiered memory store using PostgreSQL with pgvector.
    Supports multi-user and multi-session isolation with semantic retrieval.
    """

    def __init__(
        self,
        db_url: str,
        provider: LLMProvider,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
    ):
        self.db_url = db_url
        self.provider = provider
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.embedding_api_base = embedding_api_base
        self._init_db()

    def _init_db(self):
        """Initialize database schema if it doesn't exist."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS agent_memories (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(255) NOT NULL,
                            session_id VARCHAR(255),
                            content TEXT NOT NULL,
                            embedding VECTOR(1024),
                            category VARCHAR(50) DEFAULT 'fact',
                            importance FLOAT DEFAULT 1.0,
                            metadata JSONB,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_session ON agent_memories (user_id, session_id);")
                    # HNSW index for fast vector search
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_embedding ON agent_memories USING hnsw (embedding vector_cosine_ops);")
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize vector memory database: {e}")
            raise

    async def add_memory(
        self,
        user_id: str,
        content: str,
        session_id: Optional[str] = None,
        category: str = "fact",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Generate embedding and store a new memory."""
        try:
            embedding = await self.provider.embed(
                content, 
                model=self.embedding_model,
                api_key=self.embedding_api_key,
                api_base=self.embedding_api_base
            )
            
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO agent_memories (user_id, session_id, content, embedding, category, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (user_id, session_id, content, embedding, category, Json(metadata or {})),
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise

    async def query_relevant_memories(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """
        Search for relevant memories using semantic similarity.
        Prioritizes current session, then user-level preferences/facts.
        """
        try:
            query_embedding = await self.provider.embed(
                query, 
                model=self.embedding_model,
                api_key=self.embedding_api_key,
                api_base=self.embedding_api_base
            )
            
            # Tier 1: Current Session Memories (Top 3)
            session_memories = []
            if session_id:
                session_memories = self._search(
                    user_id=user_id,
                    session_id=session_id,
                    embedding=query_embedding,
                    limit=3
                )
            
            # Tier 2: Global User Preferences/Facts (Top 2)
            global_memories = self._search(
                user_id=user_id,
                session_id=None, # Explicitly search across sessions
                embedding=query_embedding,
                category="preference",
                limit=2
            )
            
            # Combine and format
            all_memories = session_memories + global_memories
            if not all_memories:
                return ""
            
            formatted = []
            for m in all_memories:
                prefix = f"[{m['category']}]"
                formatted.append(f"{prefix} {m['content']}")
                
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            return ""

    def _search(
        self,
        user_id: str,
        embedding: List[float],
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Internal search helper."""
        query = """
            SELECT content, category, (embedding <=> %s::vector) as distance
            FROM agent_memories
            WHERE user_id = %s
        """
        params = [embedding, user_id]
        
        if session_id:
            query += " AND session_id = %s"
            params.append(session_id)
        elif session_id is None and category != "preference":
            # If we're searching globally and it's not a specific preference, 
            # we might want to include only those with session_id NULL or just everything for that user.
            # For now, let's say NULL session_id means global.
            query += " AND session_id IS NULL"
            
        if category:
            query += " AND category = %s"
            params.append(category)
            
        query += " ORDER BY distance ASC LIMIT %s"
        params.append(limit)
        
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, tuple(params))
                return cur.fetchall()
