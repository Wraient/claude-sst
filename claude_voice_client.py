"""Client for Anthropic Claude Web Voice Mode (Conversational Audio) WebSocket.

Supports real-time streaming speech-to-text, prompt extraction, and automated
speech-interruption to suppress Claude's voice output for pure SST usage.
Includes multi-account pooling, automatic rotation, and failover on rate limits.
Includes real-time DC offset blocking and multi-utterance pause accumulation.
"""

import asyncio
import json
import logging
import urllib.parse
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List
import aiohttp
import numpy as np

from auth import auth_manager
from account_pool import account_pool

logger = logging.getLogger("claude_voice")


class StreamingDCBlocker:
    """Real-time DC offset and subsonic microphone rumble filter.

    Maintains a low-latency running estimate of DC bias and centers 16-bit linear PCM
    around 0 to avoid choking Claude's VAD/AEC speech detector.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.running_dc: float = 0.0
        self.initialized: bool = False

    def process_bytes(self, pcm_bytes: bytes) -> bytes:
        if not pcm_bytes:
            return pcm_bytes
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return pcm_bytes
        chunk_mean = float(np.mean(samples))
        if not self.initialized:
            self.running_dc = chunk_mean
            self.initialized = True
        else:
            self.running_dc = (1.0 - self.alpha) * self.running_dc + self.alpha * chunk_mean
        centered = samples - self.running_dc
        return np.clip(centered, -32768, 32767).astype(np.int16).tobytes()


def combine_transcripts(history: List[str], current: str) -> str:
    """Combines a list of finalized utterances with the current utterance across pauses.

    Prevents repetitive duplication by checking if current starts with or contains
    parts of previous utterances, or vice-versa.
    """
    valid_parts = [p.strip() for p in history if p and p.strip()]
    curr = current.strip()

    if not valid_parts:
        return curr
    if not curr:
        return " ".join(valid_parts)

    combined = list(valid_parts)
    last_hist = combined[-1]

    # If the current utterance already contains the last history item, replace or append cleanly
    if curr.startswith(last_hist):
        combined[-1] = curr
    elif last_hist.endswith(curr):
        pass  # Already included
    else:
        # Check for word-level suffix/prefix overlap
        last_words = last_hist.split()
        curr_words = curr.split()
        overlap_found = False
        for k in range(min(len(last_words), len(curr_words)), 0, -1):
            lw = [w.lower().strip(".,?!;:-_") for w in last_words[-k:]]
            cw = [w.lower().strip(".,?!;:-_") for w in curr_words[:k]]
            if lw == cw:
                # Overlap of k words
                merged = last_words + curr_words[k:]
                combined[-1] = " ".join(merged)
                overlap_found = True
                break
        if not overlap_found:
            combined.append(curr)

    return " ".join(combined)


class ClaudeVoiceSession:
    """A single bidirectional session connected to Claude's Voice Mode WebSocket.

    Extracts user prompt deltas and accumulated text in real-time, while suppressing Claude's audio reply.
    Rotates accounts and handles rate-limits automatically.
    """

    def __init__(
        self,
        voice: str = "buttery",
        language: str = "en",
        suppress_reply: bool = True,
        ephemeral: bool = True,
        sample_rate: int = 16000,
        model: Optional[str] = None,
        account: Optional[Dict[str, Any]] = None,
    ):
        self.voice = voice
        self.language = language
        self.suppress_reply = suppress_reply
        self.ephemeral = ephemeral
        self.sample_rate = sample_rate
        self.model = model
        self.account = account
        self.conversation_uuid = str(uuid.uuid4())

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._is_closed = False
        self._interrupted = False
        self._dc_blocker = StreamingDCBlocker(alpha=0.05)

        # Multi-turn transcript accumulation:
        # Keeps history of all confirmed utterances across pauses
        self._history_utterances: List[str] = []
        self._current_turn_text: str = ""
        self._accumulated_text: str = ""
        self.turn_completed_event = asyncio.Event()

    @property
    def accumulated_text(self) -> str:
        """Returns the full accumulated transcript from the moment recording started."""
        return combine_transcripts(self._history_utterances, self._current_turn_text)

    async def connect(self, max_retries: int = 3):
        """Connect to Claude's Voice Mode WebSocket with automatic account failover."""
        attempt = 0
        tried_ids: List[str] = []
        last_error = None

        while attempt < max_retries:
            attempt += 1
            self.conversation_uuid = str(uuid.uuid4())
            if not self.account or self.account["id"] in tried_ids:
                try:
                    self.account = account_pool.get_next_account(exclude_ids=tried_ids)
                except Exception as e:
                    logger.error(f"No available healthy Claude account in pool: {e}")
                    raise RuntimeError(f"All Claude accounts in pool are rate-limited or unavailable: {e}")

            acc_id = self.account["id"]
            acc_name = self.account.get("name", acc_id)
            tried_ids.append(acc_id)
            session_key = self.account["sessionKey"]
            org_uuid = self.account.get("orgUuid", "")

            # Ensure orgUuid exists
            if not org_uuid:
                valid, org_uuid, org_name, err = await account_pool.validate_session_key(session_key)
                if not valid:
                    account_pool.mark_dead(acc_id, reason=err or "Invalid session")
                    continue
                self.account["orgUuid"] = org_uuid

            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                "Cookie": f"sessionKey={session_key}; lastActiveOrg={org_uuid}",
                "Origin": "https://claude.ai",
                "Referer": "https://claude.ai/new",
            }

            # 1. Create conversation in Claude
            conv_url = f"https://claude.ai/api/organizations/{org_uuid}/chat_conversations"
            conv_payload = {
                "uuid": self.conversation_uuid,
                "name": "Conversational SST" if self.suppress_reply else "Audio Mode Session",
                "include_conversation_preferences": True,
            }
            if self.model:
                conv_payload["model"] = self.model

            try:
                timeout = aiohttp.ClientTimeout(total=8)
                async with aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.post(conv_url, headers={**headers, "Content-Type": "application/json"}, json=conv_payload) as resp:
                        if resp.status == 429:
                            account_pool.mark_rate_limited(acc_id, cooldown_seconds=900, reason="Conversation HTTP 429")
                            continue
                        elif resp.status in (401, 403):
                            account_pool.mark_dead(acc_id, reason=f"Conversation HTTP {resp.status}")
                            continue
                        elif resp.status in (200, 201):
                            logger.info(f"Initialized conversation {self.conversation_uuid} with account '{acc_name}'")
            except Exception as e:
                logger.warning(f"Notice creating conversation with account '{acc_name}': {e}")

            # 2. Connect Voice Mode WebSocket
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
            logger.info(f"Connecting to Claude Voice Mode WS: {url} via account '{acc_name}'")

            self._session = aiohttp.ClientSession()
            try:
                self._ws = await self._session.ws_connect(url, headers=headers, heartbeat=20.0)
                self._is_closed = False
                self._interrupted = False
                self._history_utterances = []
                self._current_turn_text = ""
                self._accumulated_text = ""
                self._keepalive_task = asyncio.create_task(self._keepalive_loop())
                logger.info(f"Connected to Claude Voice Mode WebSocket successfully via '{acc_name}'!")
                return
            except aiohttp.WSServerHandshakeError as e:
                if self._session:
                    await self._session.close()
                if e.status == 429:
                    account_pool.mark_rate_limited(acc_id, cooldown_seconds=900, reason="WS Handshake 429")
                elif e.status in (401, 403):
                    account_pool.mark_dead(acc_id, reason=f"WS Handshake {e.status}")
                else:
                    logger.error(f"WS handshake failed for '{acc_name}': {e}")
                last_error = e
            except Exception as e:
                if self._session:
                    await self._session.close()
                err_str = str(e).lower()
                if "429" in err_str:
                    account_pool.mark_rate_limited(acc_id, cooldown_seconds=900, reason=str(e))
                elif "401" in err_str or "403" in err_str:
                    account_pool.mark_dead(acc_id, reason=str(e))
                else:
                    logger.error(f"Connection failed for '{acc_name}': {e}")
                last_error = e

        raise RuntimeError(f"Failed to connect to Claude Voice Mode after trying {len(tried_ids)} accounts: {last_error}")

    async def _keepalive_loop(self):
        """Send periodic keep-alive / ping frames."""
        try:
            while not self._is_closed and self._ws and not self._ws.closed:
                await asyncio.sleep(5.0)
                if not self._is_closed and self._ws and not self._ws.closed:
                    await self._ws.send_str(json.dumps({"type": "keep_alive"}))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Keepalive loop stopped: {e}")

    async def send_audio_chunk(self, pcm_chunk: bytes):
        """Send raw PCM linear16 16kHz audio chunk to Claude after stripping DC offset."""
        if self._ws and not self._ws.closed:
            self.turn_completed_event.clear()
            cleaned = self._dc_blocker.process_bytes(pcm_chunk)
            await self._ws.send_bytes(cleaned)

    async def send_interrupt(self):
        """Send interrupt signal to Claude to stop audio generation immediately."""
        if self._ws and not self._ws.closed:
            await self._ws.send_str(
                json.dumps(
                    {
                        "type": "interrupt",
                        "reason": "user_speech_detected",
                    }
                )
            )
            self._interrupted = True

    async def flush_and_finish_turn(self):
        """Send silence padding and VAD signals so Claude finalizes the current utterance."""
        if not self._ws or self._ws.closed:
            return
        self.turn_completed_event.clear()

        # Send 0.4s silence to guarantee VAD triggers end-of-speech
        silence_chunk = b"\x00\x00" * int(self.sample_rate * 0.4)
        await self._ws.send_bytes(silence_chunk)

        try:
            await self._ws.send_str(json.dumps({"type": "vad_speech_ended"}))
            await self._ws.send_str(json.dumps({"type": "stop_audio"}))
        except Exception:
            pass

    async def events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield parsed events from Claude Voice WebSocket.

        Intercepts:
        - prompt_delta: Real-time user transcription chunk.
        - prompt: Updated accumulated user prompt.
        - assistant_audio / assistant_text: Claude's response (suppressed if suppress_reply=True).
        """
        if not self._ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    event_type = data.get("type", "")

                    # 1. User Prompt Updates (The speech-to-text output)
                    if event_type in ("transcript_interim", "prompt_delta", "prompt"):
                        txt = data.get("text", "")
                        if txt:
                            seq = data.get("utterance_seq")
                            # Detect when a new utterance begins after a pause
                            is_new_utterance = False
                            if seq == 0 and self._current_turn_text.strip():
                                if not txt.strip().startswith(self._current_turn_text.strip()):
                                    is_new_utterance = True
                            elif self._current_turn_text.strip() and len(txt.strip()) < len(self._current_turn_text.strip()) * 0.4:
                                if not txt.strip().startswith(self._current_turn_text.strip()[:10]):
                                    is_new_utterance = True

                            if is_new_utterance:
                                curr = self._current_turn_text.strip()
                                if not self._history_utterances or self._history_utterances[-1] != curr:
                                    self._history_utterances.append(curr)
                                self._current_turn_text = ""

                            self._current_turn_text = txt
                            self._accumulated_text = self.accumulated_text
                            yield {
                                "type": "interim",
                                "text": self._accumulated_text,
                                "delta": txt,
                                "full_text": self._accumulated_text,
                            }

                    elif event_type == "transcript_empty":
                        # Silence detected / pause between utterances:
                        # Lock in any current turn text into history so it is never lost
                        if self._current_turn_text.strip():
                            curr = self._current_turn_text.strip()
                            if not self._history_utterances or self._history_utterances[-1] != curr:
                                self._history_utterances.append(curr)
                            self._current_turn_text = ""

                    # 2. Assistant Turn Starting / Turn End -> Suppress reply!
                    elif (
                        event_type in ("assistant_audio", "assistant_delta", "turn_start", "tts_word", "playback_end", "user_input_end")
                        or (event_type == "message_sse" and data.get("event", {}).get("type") in ("message_start", "content_block_start"))
                    ):
                        self.turn_completed_event.set()
                        # Archive current turn text into history
                        if self._current_turn_text.strip():
                            curr = self._current_turn_text.strip()
                            if not self._history_utterances or self._history_utterances[-1] != curr:
                                self._history_utterances.append(curr)
                            self._current_turn_text = ""

                        if self.suppress_reply:
                            # Immediately interrupt Claude so it doesn't speak or generate audio
                            await self.send_interrupt()
                            yield {
                                "type": "final",
                                "text": self.accumulated_text,
                                "full_text": self.accumulated_text,
                            }
                        else:
                            yield data

                    elif event_type == "error":
                        err_msg = data.get("message", "Unknown error")
                        logger.error(f"Claude Voice error event: {data}")
                        if "rate limit" in err_msg.lower() or "429" in err_msg:
                            if self.account:
                                account_pool.mark_rate_limited(self.account["id"], cooldown_seconds=900, reason=err_msg)
                        yield {"type": "error", "error": err_msg}

                    else:
                        yield data

                elif msg.type == aiohttp.WSMsgType.BINARY:
                    if not self.suppress_reply:
                        yield {"type": "audio", "data": msg.data}

                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break

        except Exception as e:
            logger.error(f"Exception in voice event loop: {e}")
            raise
        finally:
            # Mark account success if transcript was acquired
            if self.account and self.accumulated_text.strip():
                account_pool.mark_success(self.account["id"])

    async def close(self):
        """Clean up WebSocket and session resources."""
        self._is_closed = True
        if self._keepalive_task:
            self._keepalive_task.cancel()

        if self._ws and not self._ws.closed:
            if self.suppress_reply:
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
            if not self.account:
                return
            org_uuid = self.account.get("orgUuid", "")
            session_key = self.account.get("sessionKey", "")
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                "Cookie": f"sessionKey={session_key}; lastActiveOrg={org_uuid}",
                "Origin": "https://claude.ai",
                "Referer": "https://claude.ai/new",
            }
            url = f"https://claude.ai/api/organizations/{org_uuid}/chat_conversations/{self.conversation_uuid}"
            async with aiohttp.ClientSession() as s:
                async with s.delete(url, headers=headers, timeout=10) as resp:
                    logger.debug(f"Cleanup conversation {self.conversation_uuid}: status {resp.status}")
        except Exception as e:
            logger.debug(f"Cleanup conversation ignored: {e}")

    # Aliases for backward compatibility
    send_audio = send_audio_chunk
    receive_events = events
