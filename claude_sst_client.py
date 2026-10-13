"""Client for Anthropic Claude Web Speech-to-Text (SST) streaming WebSocket."""

import asyncio
import json
import logging
import urllib.parse
from typing import AsyncGenerator, Optional, Dict, Any
import aiohttp

from auth import auth_manager

logger = logging.getLogger("claude_sst")


class ClaudeSSTSession:
    """A single streaming session connected to Claude's Speech-to-Text WebSocket."""

    def __init__(
        self,
        language: str = "en",
        sample_rate: int = 16000,
        encoding: str = "linear16",
        stt_provider: str = "deepgram-nova3",
        use_conversation_engine: bool = True,
    ):
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.stt_provider = stt_provider
        self.use_conversation_engine = use_conversation_engine

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._is_closed = False

    async def connect(self):
        """Establish WebSocket connection to Claude SST gateway."""
        _, _, org_uuid = auth_manager.get_credentials()
        headers = auth_manager.get_headers()

        params = {
            "encoding": self.encoding,
            "sample_rate": str(self.sample_rate),
            "channels": "1",
            "endpointing_ms": "300",
            "utterance_end_ms": "1000",
            "language": self.language,
            "use_conversation_engine": "true" if self.use_conversation_engine else "false",
            "stt_provider": self.stt_provider,
            "client_platform": "web_claude_ai",
            "organization_uuid": org_uuid,
            "keepalive_ack": "true",
        }

        url = f"wss://claude.ai/api/ws/speech_to_text/voice_stream?{urllib.parse.urlencode(params)}"
        logger.info(f"Connecting to Claude SST: {url}")

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(
                url,
                headers=headers,
                heartbeat=20.0,
            )
            self._is_closed = False
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            logger.info("Connected to Claude SST WebSocket.")
        except Exception as e:
            if self._session:
                await self._session.close()
            raise RuntimeError(f"Failed to connect to Claude SST WebSocket: {e}")

    async def _keepalive_loop(self):
        """Send periodic KeepAlive frames to maintain connection."""
        try:
            while not self._is_closed and self._ws and not self._ws.closed:
                await asyncio.sleep(4.0)
                if not self._is_closed and self._ws and not self._ws.closed:
                    await self._ws.send_str(json.dumps({"type": "KeepAlive"}))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Keepalive loop exception: {e}")

    async def send_audio(self, chunk: bytes):
        """Send a binary chunk of audio (e.g. 16kHz 16-bit mono PCM)."""
        if self._ws and not self._ws.closed:
            await self._ws.send_bytes(chunk)
        else:
            raise RuntimeError("Cannot send audio: WebSocket is closed")

    async def receive_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield structured transcript events from the Claude SST server."""
        if not self._ws:
            raise RuntimeError("Session not connected")

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue

                    msg_type = data.get("type")
                    text = data.get("data")

                    if msg_type == "TranscriptInterim":
                        yield {"type": "interim", "text": text}
                    elif msg_type == "TranscriptText":
                        yield {"type": "final", "text": text}
                    elif msg_type == "TranscriptEndpoint":
                        yield {"type": "endpoint"}
                    elif msg_type == "TranscriptError":
                        yield {"type": "error", "error": data.get("description")}
                    elif msg_type == "KeepAliveAck":
                        pass
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            await self.close()

    async def close(self):
        """Clean up WebSocket and HTTP session."""
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
        logger.info("Claude SST session closed.")
