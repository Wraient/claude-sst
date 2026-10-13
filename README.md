# Claude Web Speech-to-Text (SST) Local Proxy

A local bidirectional streaming proxy and OpenAI-compatible API for Anthropic Claude Web's Speech-to-Text (SST) engine (Deepgram Nova-3).

---

## Architecture & How It Works

Claude.ai web client uses an internal WebSocket gateway for dictation:
`wss://claude.ai/api/ws/speech_to_text/voice_stream`

### Query Parameters
- `encoding`: `linear16` (16-bit little-endian PCM) or `opus`
- `sample_rate`: `16000`
- `channels`: `1`
- `endpointing_ms`: `300`
- `utterance_end_ms`: `1000`
- `language`: `en` (or target ISO code)
- `use_conversation_engine`: `true`
- `stt_provider`: `deepgram-nova3`
- `client_platform`: `web_claude_ai`
- `organization_uuid`: `<org_uuid>`
- `keepalive_ack`: `true`

### Protocol Framing
1. **Client -> Server Audio**:
   - Raw binary audio frames (`linear16` PCM: 16000Hz, 16-bit, 1 channel).
   - Periodic JSON keepalives: `{"type": "KeepAlive"}`.
2. **Server -> Client Events**:
   - `{"type": "TranscriptInterim", "data": "word..."}`: Real-time interim hypothesis.
   - `{"type": "TranscriptText", "data": "word..."}`: Streamed progressive text.
   - `{"type": "TranscriptEndpoint"}`: Speech boundary / utterance detected.
   - `{"type": "KeepAliveAck"}`: Server heartbeat response.

---

## Features

- **Automatic Cookie Decryption**: Reads your active session and org cookies directly from local Google Chrome (`v11` AES-128-CBC via Gnome Keyring), so you never have to manually copy session keys.
- **Bidirectional Streaming WebSocket API**: Connect via `ws://localhost:8000/v1/audio/transcriptions/stream` or `ws://localhost:8000/ws` to stream microphone audio in real time and receive live word-by-word text.
- **OpenAI-Compatible REST API**: `POST /v1/audio/transcriptions` supporting multipart file uploads (MP3, WAV, M4A, WebM, etc.) with automatic on-the-fly transcoding via `ffmpeg`.
- **Interactive Web UI**: Visit `http://localhost:8000` for an in-browser dictation interface with real-time waveform visualizer, live transcription, and audio file drag-and-drop.

---

## Quick Start

### 1. Start the Local Proxy Server
```bash
python3 server.py
```
Server starts on `http://127.0.0.1:8000`.

### 2. Run Test Suite
```bash
python3 test_client.py
```

### 3. Use the Interactive Web UI
Open your browser to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

Click the microphone button and start speaking.

---

## API Reference

### 1. OpenAI-Compatible REST API
```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@speech.wav" \
  -F "model=claude-sst"
```

Response:
```json
{
  "text": "Hi there. I just want to share how thrilled I am to work with you and I can't wait to get started. So what's the game plan?"
}
```

### 2. Bidirectional Streaming WebSocket
Endpoint: `ws://127.0.0.1:8000/ws` or `ws://127.0.0.1:8000/v1/audio/transcriptions/stream`

- **Send**: Raw binary chunks of 16kHz, 16-bit mono signed PCM (`Int16Array`).
- **Receive**: JSON events:
  ```json
  {"type": "final", "text": "Hi there."}
  ```

---

## Environment Variables (Optional Overrides)

If you are not using Chrome on Linux or wish to run headlessly on a server:
- `CLAUDE_SESSION_KEY`: `sk-ant-sid02-...`
- `CLAUDE_ORG_UUID`: Organization UUID from `claude.ai`
- `PORT`: Server port (default: `8000`)
