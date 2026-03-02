"""PostgreSQL and pgvector database utility."""

import asyncpg
from typing import Optional
from loguru import logger
from nanobot.config.schema import Config

class DatabaseManager:
    """Manages PostgreSQL connection pool and table initialization."""
    
    _pool: Optional[asyncpg.Pool] = None

    def __init__(self, config: Config):
        self.db_config = config.database
        self.enabled = self.db_config.enabled

    async def connect(self):
        """Establish connection pool and initialize tables."""
        if not self.enabled:
            return

        if not self._pool:
            try:
                self._pool = await asyncpg.create_pool(
                    host=self.db_config.host,
                    port=self.db_config.port,
                    user=self.db_config.user,
                    password=self.db_config.password,
                    database=self.db_config.database,
                    min_size=1,
                    max_size=10
                )
                logger.info("Connected to PostgreSQL database at {}:{}", self.db_config.host, self.db_config.port)
                await self._init_tables()
            except Exception as e:
                logger.error("Failed to connect to database: {}", e)
                self.enabled = False

    async def _init_tables(self):
        """Initialize required tables and extensions."""
        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            try:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Users table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # User identities mapping table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_identities (
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        channel VARCHAR(50),
                        sender_id VARCHAR(255),
                        PRIMARY KEY (channel, sender_id)
                    )
                """)
                
                # Vector memories table (1536 dimensions for text-embedding-v3)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS vector_memories (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        embedding vector(1536),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.info("Database tables and pgvector extension initialized")
            except Exception as e:
                logger.error("Failed to initialize database tables: {}", e)

    async def get_user_id(self, channel: str, sender_id: str) -> Optional[int]:
        """Get or create a global user_id for a given channel and sender_id."""
        if not self._pool or not self.enabled:
            return None
        
        try:
            async with self._pool.acquire() as conn:
                # Try to find existing identity
                row = await conn.fetchrow(
                    "SELECT user_id FROM user_identities WHERE channel = $1 AND sender_id = $2",
                    channel, sender_id
                )
                if row:
                    return row['user_id']
                
                # Create new user and identity in a transaction
                async with conn.transaction():
                    user_id = await conn.fetchval("INSERT INTO users DEFAULT VALUES RETURNING id")
                    await conn.execute(
                        "INSERT INTO user_identities (user_id, channel, sender_id) VALUES ($1, $2, $3)",
                        user_id, channel, sender_id
                    )
                    logger.info("Created new user ID {} for {}:{}", user_id, channel, sender_id)
                    return user_id
        except Exception as e:
            logger.error("Database error in get_user_id: {}", e)
            return None

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    @property
    def pool(self) -> Optional[asyncpg.Pool]:
        return self._pool
