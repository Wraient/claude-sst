# Claude Web Audio & Speech-to-Text (SST) Local Gateway

A local bidirectional streaming proxy and API for Anthropic Claude Web's Voice & Dictation engines.

---

## 3 Operating Modes Available

| Mode | Endpoint | Description |
| :--- | :--- | :--- |
| **Smart Conversational SST** *(Recommended)* | `ws://localhost:8000/ws/conversational_sst` | **Dynamic prompt updating & context-aware transcription WITHOUT Claude answering.** As you speak, the prompt refines in real time. Claude's LLM reply is automatically interrupted and suppressed, and temporary sessions are cleanly deleted. |
| **Full Conversational Voice** | `ws://localhost:8000/ws/voice` | Full two-way voice chat. Updates prompt live, then streams Claude's contextual assistant reply (tokens + audio). |
| **Raw Dictation SST** | `ws://localhost:8000/ws/sst` | Raw low-latency Deepgram Nova-3 transcription. |

---

## How Smart Conversational SST Works (No Claude Answer)

Claude's conversational voice model (`/api/ws/voice/organizations/{org_id}/chat_conversations/{conv_uuid}`) provides far superior, grammatically refined prompt transcription compared to raw dictation.

To get this conversational prompt transcription **without Claude answering or cluttering your account**, the proxy implements a 3-part pipeline:
1. **Live Prompt Extraction**: Consumes `transcript_interim` events from Claude's voice WebSocket, continuously streaming refined prompt sentences to your client.
2. **Instant Server Interruption**: As soon as speech pause/end is detected (`user_input_end`), the proxy immediately transmits an upstream `{"type": "interrupt"}` control frame to Claude's server, aborting assistant LLM inference and TTS before any reply is produced.
3. **Incognito & Ephemeral Cleanup**: Establishes the session with `chat_memory_mode: disabled` and issues a `DELETE /api/organizations/{org}/chat_conversations/{conv_uuid}` when closed, leaving zero trace in your chat sidebar.

---

## Quick Start

### 1. Launch Proxy Server
```bash
python3 server.py
```
Runs at `http://127.0.0.1:8000`.

### 2. Run Test Suite
```bash
python3 test_client.py
```

### 3. Open Interactive Web UI
Visit [http://localhost:8000](http://localhost:8000):
- Select **Smart Conversational SST (No Claude Answer)**.
- Tap the microphone button and start speaking naturally.
- Watch your prompt update live, refined as you speak, with zero assistant responses.

---

## API Reference

### 1. Smart Conversational SST WebSocket
Connect to: `ws://127.0.0.1:8000/ws/conversational_sst` (or `/ws/smart_sst`)

- **Send**: Raw binary chunks of 16kHz, 16-bit mono signed PCM (`Int16Array`).
- **Receive**:
  ```json
  {"type": "interim", "text": "Hi there I just want to"}
  {"type": "final", "text": "Hi there. I just want to share how thrilled I am to work with you."}
  ```

### 2. OpenAI-Compatible REST API
```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@speech.wav"
```
Defaults to the high-quality Conversational engine. Response:
```json
{
  "text": "Hi there. I just want to share how thrilled I am to work with you. And I can't wait to get started. So what's the game plan?"
}
```
