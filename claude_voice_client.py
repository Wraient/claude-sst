"""Claude Conversational Audio / Voice Mode Client.

Maintains conversation context, streams user audio, updates prompt in real time,
and yields both the live user prompt and Claude's assistant response tokens.
"""

import asyncio
import json
import logging
import urllib.parse
import uuid
from typing import AsyncGenerator, Dict, Any, Optional
import aiohttp

from auth import auth_manager

logger = logging.getLogger("claude_voice")


class ClaudeVoiceSession:
    """Session connected to Claude's Conversational Voice Mode WebSocket."""

    def __init__(
        self,
        conversation_uuid: Optional[str] = None,
        language: str = "en",
        voice: str = "buttery",
        sample_rate: int = 16000,
        model: Optional[str] = None,
    ):
        self.conversation_uuid = conversation_uuid or str(uuid.uuid4())
        self.language = language
        self.voice = voice
        self.sample_rate = sample_rate
        self.model = model

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._is_closed = False

    async def _ensure_conversation_exists(self):
        """Create conversation in Claude if it doesn't already exist."""
        _, _, org_uuid = auth_manager.get_credentials()
        headers = auth_manager.get_headers()
        headers["Content-Type"] = "application/json"

        url = f"https://claude.ai/api/organizations/{org_uuid}/chat_conversations"
        payload = {
            "uuid": self.conversation_uuid,
            "name": "Audio Mode Session",
            "include_conversation_preferences": True,
        }
        if self.model:
            payload["model"] = self.model

        async with aiohttp.ClientSession() as s:
            try:
                async with s.post(url, headers=headers, json=payload, timeout=10) as resp:
                    if resp.status in (200, 201):
                        logger.info(f"Initialized conversation {self.conversation_uuid}")
                    else:
                        logger.debug(f"Conversation create returned status {resp.status}")
            except Exception as e:
                logger.warning(f"Error creating conversation: {e}")

    async def connect(self):
        """Connect to Claude's Voice Mode WebSocket."""
        await self._ensure_conversation_exists()
        _, _, org_uuid = auth_manager.get_credentials()
        headers = auth_manager.get_headers()

        params = {
            "input_encoding": "linear16",
            "input_sample_rate": str(self.sample_rate),
            "input_channels": "1",
            "output_format": "pcm_16000",
            "language": self.language,
            "timezone": "Asia/Kolkata",
            "tts_speed": "1.00",
            "server_interrupt_enabled": "true",
            "client_aec": "true",
            "client_platform": "web_claude_ai",
            "entry_source": "chat_button",
            "connect_reason": "initial",
            "voice": self.voice,
        }
        if self.model:
            params["model"] = self.model

        url = (
            f"wss://claude.ai/api/ws/voice/organizations/{org_uuid}/"
            f"chat_conversations/{self.conversation_uuid}?{urllib.parse.urlencode(params)}"
        )
        logger.info(f"Connecting to Claude Voice Mode WS: {url}")

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(url, headers=headers, heartbeat=20.0)
            self._is_closed = False
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            logger.info("Connected to Claude Voice Mode WebSocket.")
        except Exception as e:
            if self._session:
                await self._session.close()
            raise RuntimeError(f"Failed to connect to Claude Voice Mode: {e}")

    async def _keepalive_loop(self):
        """Send periodic keep_alive control frames."""
        try:
            while not self._is_closed and self._ws and not self._ws.closed:
                await asyncio.sleep(4.0)
                if not self._is_closed and self._ws and not self._ws.closed:
                    await self._ws.send_str(json.dumps({"type": "keep_alive"}))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Voice keepalive error: {e}")

    async def send_audio(self, chunk: bytes):
        """Send raw audio chunk (16kHz 16-bit mono PCM)."""
        if self._ws and not self._ws.closed:
            await self._ws.send_bytes(chunk)
        else:
            raise RuntimeError("Cannot send audio: WebSocket closed")

    async def send_interrupt(self):
        """Interrupt current assistant playback / generation."""
        if self._ws and not self._ws.closed:
            await self._ws.send_str(json.dumps({"type": "interrupt"}))

    async def receive_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield structured conversation events (user prompt updates & assistant text)."""
        if not self._ws:
            raise RuntimeError("Session not connected")

        current_prompt = ""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue

                    mtype = data.get("type")

                    # Live user speech prompt as spoken
                    if mtype == "transcript_interim":
                        text = data.get("text", "")
                        current_prompt = text
                        yield {
                            "type": "user_interim",
                            "text": text,
                            "conversation_uuid": self.conversation_uuid,
                        }

                    # User stopped speaking turn
                    elif mtype == "user_input_end":
                        yield {
                            "type": "user_speech_end",
                            "final_prompt": current_prompt,
                            "conversation_uuid": self.conversation_uuid,
                        }

                    # Human message committed
                    elif mtype == "message_complete":
                        sender = data.get("data", {}).get("sender")
                        text = data.get("data", {}).get("text")
                        yield {
                            "type": "message_complete",
                            "sender": sender,
                            "text": text,
                            "conversation_uuid": self.conversation_uuid,
                        }

                    # Assistant text generation token by token
                    elif mtype == "message_sse":
                        ev = data.get("event", {})
                        if ev.get("type") == "content_block_delta":
                            delta = ev.get("data", {}).get("delta", {})
                            token = delta.get("text", "")
                            if token:
                                yield {
                                    "type": "assistant_text_delta",
                                    "text": token,
                                    "conversation_uuid": self.conversation_uuid,
                                }

                    elif mtype == "tts_word":
                        yield {
                            "type": "tts_word",
                            "text": data.get("text", ""),
                            "pts_ms": data.get("pts_ms"),
                        }

                    elif mtype in ("playback_start", "playback_end"):
                        yield {"type": mtype}

                    elif mtype == "error":
                        yield {"type": "error", "error": data}

                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Assistant PCM audio chunk
                    yield {
                        "type": "assistant_audio",
                        "audio": msg.data,
                    }

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            await self.close()

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True

        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        logger.info(f"Claude Voice session {self.conversation_uuid} closed.")
