"""Claude Conversational Audio / Voice Mode Client.

Supports:
1. Full Voice Conversation (User speaks -> Prompt updates -> Claude answers).
2. Conversational SST Only (User speaks -> Prompt updates with full conversational engine
   -> Claude answer is suppressed & interrupted -> Clean prompt returned without Claude reply).
"""

import asyncio
import json
import logging
import urllib.parse
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List
import aiohttp

from auth import auth_manager

logger = logging.getLogger("claude_voice")


def combine_transcripts(history: List[str], current: str) -> str:
    """Seamlessly combine committed turn history and current live utterance,

    avoiding duplicate overlapping boundary words across pauses and turns.
    """
    cleaned_history = [h.strip() for h in history if h and h.strip()]
    cleaned_current = current.strip() if current else ""

    if not cleaned_history:
        return cleaned_current
    if not cleaned_current:
        return " ".join(cleaned_history)

    full_history = " ".join(cleaned_history)

    # If current already contains full_history
    if cleaned_current.startswith(full_history):
        return cleaned_current

    # If current already contains the last turn
    last_item = cleaned_history[-1]
    if cleaned_current.startswith(last_item):
        prefix = " ".join(cleaned_history[:-1])
        return f"{prefix} {cleaned_current}".strip()

    # Word-level suffix overlap check to eliminate duplicate boundary words
    hist_words = full_history.split()
    curr_words = cleaned_current.split()
    max_overlap = min(len(hist_words), len(curr_words), 8)
    overlap_len = 0
    for k in range(max_overlap, 0, -1):
        h_slice = [w.lower().strip(".,?!;:'\"") for w in hist_words[-k:]]
        c_slice = [w.lower().strip(".,?!;:'\"") for w in curr_words[:k]]
        if h_slice == c_slice:
            overlap_len = k
            break

    if overlap_len > 0:
        return " ".join(hist_words[:-overlap_len] + curr_words)

    return f"{full_history} {cleaned_current}".strip()


class ClaudeVoiceSession:
    """Session connected to Claude's Conversational Voice Mode WebSocket."""

    def __init__(
        self,
        conversation_uuid: Optional[str] = None,
        language: str = "en",
        voice: str = "buttery",
        sample_rate: int = 16000,
        model: Optional[str] = None,
        suppress_reply: bool = False,
        ephemeral: bool = False,
    ):
        self.conversation_uuid = conversation_uuid or str(uuid.uuid4())
        self.language = language
        self.voice = voice
        self.sample_rate = sample_rate
        self.model = model
        self.suppress_reply = suppress_reply
        self.ephemeral = ephemeral or suppress_reply

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._is_closed = False
        self._interrupted = False

        # Accumulation state across pauses and multi-turn utterances
        self._history_utterances: List[str] = []
        self._current_turn_text: str = ""
        self._accumulated_text: str = ""

    @property
    def accumulated_text(self) -> str:
        """Returns the full accumulated transcript from the moment recording started."""
        return combine_transcripts(self._history_utterances, self._current_turn_text)

    async def _ensure_conversation_exists(self):
        """Create conversation in Claude if it doesn't already exist."""
        _, _, org_uuid = auth_manager.get_credentials()
        headers = auth_manager.get_headers()
        headers["Content-Type"] = "application/json"

        url = f"https://claude.ai/api/organizations/{org_uuid}/chat_conversations"
        payload = {
            "uuid": self.conversation_uuid,
            "name": "Conversational SST" if self.suppress_reply else "Audio Mode Session",
            "include_conversation_preferences": True,
        }
        if self.model:
            payload["model"] = self.model

        async with aiohttp.ClientSession() as s:
            try:
                async with s.post(url, headers=headers, json=payload, timeout=10) as resp:
                    if resp.status in (200, 201):
                        logger.info(f"Initialized conversation {self.conversation_uuid}")
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
        if self.ephemeral:
            params["chat_memory_mode"] = "disabled"
        if self.model:
            params["model"] = self.model

        url = (
            f"wss://claude.ai/api/ws/voice/organizations/{org_uuid}/"
            f"chat_conversations/{self.conversation_uuid}?{urllib.parse.urlencode(params)}"
        )
        logger.info(f"Connecting to Claude Voice Mode WS: {url} (suppress_reply={self.suppress_reply})")

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(url, headers=headers, heartbeat=20.0)
            self._is_closed = False
            self._interrupted = False
            self._history_utterances = []
            self._current_turn_text = ""
            self._accumulated_text = ""
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
            self._interrupted = True
            await self._ws.send_str(json.dumps({"type": "interrupt"}))

    async def receive_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield structured conversation events (user prompt updates & assistant text).

        Maintains cumulative prompt history so speech across pauses is never lost.
        """
        if not self._ws:
            raise RuntimeError("Session not connected")

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
                        self._current_turn_text = text
                        full_prompt = self.accumulated_text
                        self._accumulated_text = full_prompt
                        yield {
                            "type": "interim" if self.suppress_reply else "user_interim",
                            "text": full_prompt,
                            "turn_text": text,
                            "conversation_uuid": self.conversation_uuid,
                        }

                    # User stopped speaking a turn
                    elif mtype == "user_input_end":
                        if self.suppress_reply:
                            # Immediately tell server to interrupt and NOT generate an answer
                            await self.send_interrupt()
                        full_prompt = self.accumulated_text
                        self._accumulated_text = full_prompt
                        yield {
                            "type": "interim" if self.suppress_reply else "user_speech_end",
                            "text": full_prompt,
                            "final_prompt": full_prompt,
                            "conversation_uuid": self.conversation_uuid,
                        }

                    # Human message committed
                    elif mtype == "message_complete":
                        data_payload = data.get("data", {})
                        if data_payload.get("sender") == "human":
                            contents = data_payload.get("content", [])
                            turn_text = ""
                            if contents and isinstance(contents, list) and isinstance(contents[0], dict):
                                turn_text = contents[0].get("text", "")
                            if not turn_text:
                                turn_text = data_payload.get("text", "") or self._current_turn_text

                            if turn_text:
                                self._history_utterances.append(turn_text.strip())
                                self._current_turn_text = ""

                            full_prompt = self.accumulated_text
                            self._accumulated_text = full_prompt
                            yield {
                                "type": "final" if self.suppress_reply else "message_complete",
                                "text": full_prompt,
                                "sender": "human",
                                "conversation_uuid": self.conversation_uuid,
                            }
                        else:
                            if not self.suppress_reply:
                                yield {
                                    "type": "message_complete",
                                    "sender": data_payload.get("sender"),
                                    "text": data_payload.get("text"),
                                    "conversation_uuid": self.conversation_uuid,
                                }

                    # Assistant text generation token by token
                    elif mtype == "message_sse":
                        if self.suppress_reply:
                            # If server started generating before interrupt processed, send another interrupt
                            if not self._interrupted:
                                await self.send_interrupt()
                            continue

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
                        if not self.suppress_reply:
                            yield {
                                "type": "tts_word",
                                "text": data.get("text", ""),
                                "pts_ms": data.get("pts_ms"),
                            }

                    elif mtype in ("playback_start", "playback_end"):
                        if not self.suppress_reply:
                            yield {"type": mtype}

                    elif mtype == "error":
                        yield {"type": "error", "error": data}

                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Assistant PCM audio chunk
                    if not self.suppress_reply:
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
            if self.suppress_reply and not self._interrupted:
                try:
                    await self._ws.send_str(json.dumps({"type": "interrupt"}))
                except Exception:
                    pass
            await self._ws.close()
        self._ws = None

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

        # If ephemeral, delete the temporary conversation from Claude
        if self.ephemeral:
            asyncio.create_task(self._cleanup_conversation())

        logger.info(f"Claude Voice session {self.conversation_uuid} closed.")

    async def _cleanup_conversation(self):
        """Delete ephemeral conversation from Claude backend."""
        try:
            _, _, org_uuid = auth_manager.get_credentials()
            headers = auth_manager.get_headers()
            url = f"https://claude.ai/api/organizations/{org_uuid}/chat_conversations/{self.conversation_uuid}"
            async with aiohttp.ClientSession() as s:
                async with s.delete(url, headers=headers, timeout=10) as resp:
                    logger.debug(f"Cleanup conversation {self.conversation_uuid}: status {resp.status}")
        except Exception as e:
            logger.debug(f"Cleanup conversation failed (ignored): {e}")
