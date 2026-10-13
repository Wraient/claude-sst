# Claude Web Audio & Speech-to-Text (SST) Local Gateway

A local bidirectional streaming proxy and API for Anthropic Claude Web's Voice & Dictation engines.

Supports two distinct modes:
1. **Conversational Audio Mode** (`/ws/voice`): Full bidirectional conversation with Claude. Updates your prompt dynamically as you speak, keeps track of conversational context, and streams Claude's thought and response tokens back in real time.
2. **Pure Dictation SST Mode** (`/ws/sst` or `/ws`): Fast raw speech-to-text dictation without assistant reasoning or replies (using Deepgram Nova-3).
3. **OpenAI-Compatible REST API** (`POST /v1/audio/transcriptions`): For transcribing recorded audio files (WAV, MP3, M4A, WebM, etc.).

---

## Architecture & Reverse-Engineered Endpoints

Claude Web uses two distinct WebSocket services:

| Feature | Endpoint | Description |
| :--- | :--- | :--- |
| **Conversational Audio** | `wss://claude.ai/api/ws/voice/organizations/{org_id}/chat_conversations/{conv_uuid}` | Live prompt transcription, conversational turn management, context memory, and streaming assistant responses. |
| **Dictation SST** | `wss://claude.ai/api/ws/speech_to_text/voice_stream` | Low-latency speech-to-text dictation using `deepgram-nova3` directly into input boxes. |

### Audio Specifications
- **Format**: Linear PCM 16-bit little-endian (`linear16` / `s16le`)
- **Sample Rate**: 16,000 Hz
- **Channels**: 1 (mono)
- **Framing**: ~40ms - 100ms chunks (1280 - 3200 bytes per chunk)

---

## Features

- **Automatic Cookie Decryption**: Decrypts active `sessionKey`, `lastActiveOrg`, and Cloudflare clearance cookies from Google Chrome's Linux Gnome Keyring (`v11` AES-128-CBC) with zero configuration.
- **Conversational Audio Mode**:
  - Live interim prompt updates as words are spoken (`user_interim`).
  - Turn boundary endpointing (`user_speech_end`).
  - Token-by-token assistant text generation (`assistant_text_delta`).
  - Full conversation context retention across multiple voice turns.
- **Pure Dictation SST Mode**:
  - Fast interim and final transcription hypotheses (`interim`, `final`).
- **OpenAI-Compatible REST Endpoint**: `POST /v1/audio/transcriptions` with on-the-fly `ffmpeg` transcoding.
- **Interactive Web UI**: Built-in web client at `http://localhost:8000` with visualizer, dual-pane conversation/prompt monitor, mode switcher, and mic recording.

---

## Quick Start

### 1. Launch the Server
```bash
python3 server.py
```
Server runs on `http://127.0.0.1:8000`.

### 2. Run Test Suite
```bash
python3 test_client.py
```

### 3. Open Web UI
Visit [http://localhost:8000](http://localhost:8000) in your browser:
- Switch between **Conversational Audio Mode** and **Dictation SST**.
- Tap the microphone button and start speaking.

---

## API Reference

### 1. Conversational Audio WebSocket (`/ws/voice`)
Connect to: `ws://127.0.0.1:8000/ws/voice?conversation_uuid=<uuid>`

- **Send**: Binary 16kHz 16-bit mono PCM chunks.
- **Receive**:
  ```json
  {"type": "user_interim", "text": "Hi there I just want to"}
  {"type": "user_speech_end", "final_prompt": "Hi there. I just want to..."}
  {"type": "assistant_text_delta", "text": "Hello! How can I help you today?"}
  ```

### 2. Dictation SST WebSocket (`/ws/sst` or `/ws`)
Connect to: `ws://127.0.0.1:8000/ws/sst`

- **Send**: Binary 16kHz 16-bit mono PCM chunks.
- **Receive**:
  ```json
  {"type": "final", "text": "Transcribed speech text here..."}
  ```

### 3. OpenAI-Compatible REST API
```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=claude-sst"
```
