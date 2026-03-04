"""HTTP callback server for WeCom webhook."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.channels.wecom import WecomChannel

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import PlainTextResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. WeCom callback server unavailable.")

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False
    logger.warning("uvicorn not installed. WeCom callback server unavailable.")


class WecomCallbackServer:
    """
    HTTP callback server for WeCom webhook.
    
    Runs on port 18790 (same as gateway) and handles:
    - WeCom verification (GET /wecom/callback)
    - Message callbacks (POST /wecom/callback)
    """

    def __init__(self, channel: WecomChannel, port: int = 18790):
        self.channel = channel
        self.port = port
        self._server: Any = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the callback server."""
        if not FASTAPI_AVAILABLE or not UVICORN_AVAILABLE:
            logger.error("FastAPI or uvicorn not installed. Cannot start WeCom callback server.")
            return

        app = FastAPI(title="nanobot WeCom Callback")

        @app.get("/wecom/callback")
        async def get_callback(request: Request) -> Response:
            """Handle WeCom verification request."""
            status_code, body = await self.channel.handle_callback(request)
            return PlainTextResponse(content=body, status_code=status_code)

        @app.post("/wecom/callback")
        async def post_callback(request: Request) -> Response:
            """Handle WeCom message callback."""
            status_code, body = await self.channel.handle_callback(request)
            return PlainTextResponse(content=body, status_code=status_code)

        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        
        logger.info("WeCom callback server started on port {}", self.port)
        logger.info("Callback URL: http://<your-server>:{} /wecom/callback", self.port)
        
        await self._server.serve()

    async def stop(self) -> None:
        """Stop the callback server."""
        if self._server:
            self._server.should_exit = True
        logger.info("WeCom callback server stopped")

