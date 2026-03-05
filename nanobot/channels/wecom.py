"""WeCom (Enterprise WeChat) channel implementation using callback mode."""

from __future__ import annotations

import asyncio
import hashlib
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Any

import httpx
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import WecomConfig

try:
    import base64
    from Crypto.Cipher import AES
    WECOM_CRYPTO_AVAILABLE = True
except ImportError:
    WECOM_CRYPTO_AVAILABLE = False
    logger.warning("pycryptodome not installed. WeCom message encryption/decryption unavailable.")

MAX_SEEN_MESSAGE_IDS = 2000


class WecomChannel(BaseChannel):
    """
    WeCom (Enterprise WeChat) channel using callback mode.

    Requires:
    - CorpID (企业 ID)
    - AgentID (应用 ID)
    - Secret (应用 Secret)
    - Token (回调验证 Token)
    - EncodingAESKey (消息加密密钥)
    - Callback URL: http://your-server:18790/wecom/callback

    Setup:
    1. Create enterprise app in WeCom Admin Panel
    2. Configure callback URL in App settings
    3. Enable message receiving
    """

    name = "wecom"
    callback_path = "/wecom/callback"

    def __init__(self, config: WecomConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: WecomConfig = config
        self._http_client: httpx.AsyncClient | None = None
        self._access_token: str | None = None
        self._token_expires_at: float = 0
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()

    async def start(self) -> None:
        """Start the WeCom bot."""
        if not self.config.corp_id or not self.config.agent_id or not self.config.secret:
            logger.error("WeCom corp_id, agent_id, and secret must be configured")
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=30.0)
        
        logger.info("WeCom bot started")
        logger.info("Callback URL: http://<gateway-host>:18790{}", self.callback_path)

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the WeCom bot."""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
        logger.info("WeCom bot stopped")

    async def _get_access_token(self) -> str | None:
        """Get WeCom access token (cached)."""
        if self._access_token and time.time() < self._token_expires_at - 300:
            return self._access_token

        try:
            url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.config.corp_id}&corpsecret={self.config.secret}"
            response = await self._http_client.get(url)
            data = response.json()
            
            if data.get("errcode") == 0:
                self._access_token = data["access_token"]
                self._token_expires_at = time.time() + data.get("expires_in", 7200)
                return self._access_token
            else:
                logger.error("Failed to get WeCom access token: {}", data)
                return None
        except Exception as e:
            logger.error("Error getting WeCom access token: {}", e)
            return None

    async def send(self, msg: OutboundMessage) -> None:
        """Send message to WeCom."""
        if not self._http_client:
            logger.error("WeCom HTTP client not initialized")
            return

        access_token = await self._get_access_token()
        if not access_token:
            return

        content = msg.content
        if msg.metadata.get("_tool_hint"):
            return
        if msg.metadata.get("_progress"):
            return

        try:
            payload = {
                "touser": msg.chat_id,
                "msgtype": "text",
                "agentid": self.config.agent_id,
                "text": {"content": content},
                "safe": 0
            }

            url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}"
            response = await self._http_client.post(url, json=payload)
            data = response.json()

            if data.get("errcode") == 0:
                logger.debug("WeCom message sent to {}: {}", msg.chat_id, content[:50])
            else:
                logger.error("Failed to send WeCom message: {}", data)

        except Exception as e:
            logger.error("Error sending WeCom message: {}", e)

    async def handle_callback(self, request: Any) -> tuple[int, str]:
        """
        Handle WeCom callback request.
        
        Args:
            request: FastAPI Request object
            
        Returns:
            Tuple of (status_code, response_body)
        """
        try:
            query_params = dict(request.query_params)
            msg_signature = query_params.get("msg_signature", "")
            timestamp = query_params.get("timestamp", "")
            nonce = query_params.get("nonce", "")
            echostr = query_params.get("echostr", "")

            # Verification request (GET)
            if echostr:
                if self._verify_signature(msg_signature, timestamp, nonce, echostr):
                    decrypted_echostr = self._decrypt_msg(echostr)
                    return 200, decrypted_echostr
                else:
                    logger.warning("WeCom callback signature verification failed (GET)")
                    return 403, "Invalid signature"

            # Message callback (POST)
            body = await request.body()
            if not body:
                return 400, "Empty body"
            
            # Parse XML body
            root = ET.fromstring(body)
            encrypt_node = root.find("Encrypt")
            if encrypt_node is None or not encrypt_node.text:
                logger.error("WeCom callback missing Encrypt node")
                return 400, "Invalid XML"
            
            encrypt_msg = encrypt_node.text
            
            if self._verify_signature(msg_signature, timestamp, nonce, encrypt_msg):
                decrypted_xml = self._decrypt_msg(encrypt_msg)
                if not decrypted_xml:
                    return 500, "Decryption failed"
                
                message_data = self._parse_xml(decrypted_xml)
                await self._process_message(message_data)
                
                return 200, "success"
            else:
                logger.warning("WeCom callback signature verification failed (POST)")
                return 403, "Invalid signature"

        except Exception as e:
            logger.error("Error handling WeCom callback: {}", e)
            return 500, "Internal error"

    def _verify_signature(self, msg_signature: str, timestamp: str, nonce: str, data: str) -> bool:
        """Verify WeCom request signature."""
        try:
            token = self.config.token
            if not token:
                logger.warning("WeCom token not configured, skipping signature verification")
                return True

            sorted_list = sorted([token, timestamp, nonce, data])
            concatenated = "".join(sorted_list)
            signature = hashlib.sha1(concatenated.encode()).hexdigest()
            
            return signature == msg_signature
        except Exception as e:
            logger.error("Signature verification error: {}", e)
            return False

    def _decrypt_msg(self, encrypted_msg: str) -> str:
        """Decrypt WeCom message."""
        if not WECOM_CRYPTO_AVAILABLE:
            logger.error("AES decryption requested but pycryptodome not installed")
            return ""

        try:
            aes_key = base64.b64decode(self.config.encoding_aes_key + "=")
            iv = aes_key[:16]
            cipher = AES.new(aes_key, AES.MODE_CBC, iv)
            
            encrypted = base64.b64decode(encrypted_msg)
            decrypted = cipher.decrypt(encrypted)
            
            # PKCS7 unpadding
            padding_len = decrypted[-1]
            if padding_len < 1 or padding_len > 32:
                padding_len = 0
            decrypted = decrypted[:-padding_len]
            
            # WeCom message structure:
            # random(16 bytes) + msg_len(4 bytes) + msg + corp_id
            content = decrypted[16:]
            msg_len = int.from_bytes(content[:4], byteorder="big")
            message = content[4:4+msg_len].decode("utf-8")
            
            # Optional: verify corp_id
            received_corp_id = content[4+msg_len:].decode("utf-8")
            if received_corp_id != self.config.corp_id:
                logger.warning("WeCom corp_id mismatch: expected {}, got {}", self.config.corp_id, received_corp_id)
            
            return message
        except Exception as e:
            logger.error("Message decryption error: {}", e)
            return ""

    def _parse_xml(self, xml_str: str) -> dict[str, str]:
        """Parse WeCom XML message into a flat dictionary."""
        try:
            root = ET.fromstring(xml_str)
            return {child.tag: child.text for child in root if child.text is not None}
        except Exception as e:
            logger.error("Error parsing WeCom XML: {}", e)
            return {}

    async def _process_message(self, message_data: dict) -> None:
        """Process incoming WeCom message."""
        try:
            msg_type = message_data.get("MsgType", "")
            event = message_data.get("Event", "")
            
            # Support text messages and specific events
            if msg_type != "text" and msg_type != "event":
                return
            
            if msg_type == "event" and event not in ("enter_agent", "click", "view"):
                return

            from_user = message_data.get("FromUserName", "")
            to_agent = message_data.get("ToUserName", "")
            content = message_data.get("Content", "")
            msg_id = message_data.get("MsgId", str(time.time()))
            create_time = int(message_data.get("CreateTime", time.time()))

            # Deduplication
            if msg_id in self._processed_message_ids:
                return
            self._processed_message_ids[msg_id] = None
            if len(self._processed_message_ids) > MAX_SEEN_MESSAGE_IDS:
                self._processed_message_ids.popitem(last=False)

            # Access control
            if self.config.allow_from and from_user not in self.config.allow_from:
                if "*" not in self.config.allow_from:
                    logger.warning("WeCom message from unauthorized user: {}", from_user)
                    return

            inbound_msg = {
                "channel": "wecom",
                "chat_id": from_user,
                "author": from_user,
                "content": content,
                "timestamp": create_time,
                "message_id": msg_id,
                "metadata": {
                    "to_user": to_agent,
                    "agent_id": self.config.agent_id,
                    "msg_type": msg_type,
                    "event": event
                }
            }

            await self.bus.publish_inbound(inbound_msg)
            
            logger.debug("WeCom message received from {}: {}", from_user, (content or "")[:50])

        except Exception as e:
            logger.error("Error processing WeCom message: {}", e)
