# Claude Web Audio & Speech-to-Text (SST) Local Gateway

A local bidirectional streaming proxy and API for Anthropic Claude Web's Voice & Dictation engines. Operates as a persistent `systemd` user service alongside `opencode-cc` (8787), `muse-cc` (8788), and `mcode-cc` (8789) on consistent port **8790**.

---

## Operating Modes

| Mode | Endpoint | Description |
| :--- | :--- | :--- |
| **Non-Streaming REST API** | `POST http://127.0.0.1:8790/v1/audio/transcriptions` (or `/speech-to-text`) | Accepts WAV/MP3/M4A/WebM audio, accumulates transcription in memory, and returns a complete JSON response (`{"text": "...", "transcript": "...", "transcription": "..."}`). Fully compatible with OpenAI and Sarvam API formats used in `hyprwhspr`. |
| **Smart Conversational SST (Streaming)** | `ws://127.0.0.1:8790/ws/conversational_sst` | **Dynamic prompt updating & context-aware transcription WITHOUT Claude answering.** As you speak, the prompt refines in real time. Claude's LLM reply is automatically interrupted and suppressed, and temporary sessions are cleanly deleted. |
| **Full Conversational Voice** | `ws://127.0.0.1:8790/ws/voice` | Full two-way voice chat. Updates prompt live, then streams Claude's contextual assistant reply (tokens + audio). |
| **Raw Dictation SST** | `ws://127.0.0.1:8790/ws/sst` | Raw low-latency Deepgram Nova-3 transcription. |

---

## Systemd Service

Runs automatically on boot and restarts on failure via systemd:
```bash
# Check status
systemctl --user status claude-sst.service

# Restart service
systemctl --user restart claude-sst.service

# View live logs
journalctl --user -u claude-sst.service -f
```

- **Service File**: `~/.config/systemd/user/claude-sst.service`
- **Launcher Binary**: `~/.local/bin/claude-sst`
- **Environment**: `~/.config/claude-sst/env` (`PORT=8790`, `HOST=127.0.0.1`)

---

## OpenWhisper (`hyprwhspr`) Integration

Integrated directly into `hyprwhspr` with zero code modifications to OpenWhisper.

### Switch Profiles
```bash
# Switch to Claude Local Proxy
hyprwhspr-backend claude

# Switch to OpenAI
hyprwhspr-backend openai

# Switch to Sarvam
hyprwhspr-backend sarvam

# Check active backend status
hyprwhspr-backend status
```

### Configuration Details (`~/.config/hyprwhspr/config.claude.json`)
```json
{
  "transcription_backend": "rest-api",
  "rest_endpoint_url": "http://127.0.0.1:8790/v1/audio/transcriptions",
  "rest_headers": {
    "Authorization": "Bearer local-claude-sst"
  },
  "rest_body": {
    "model": "conversational",
    "language": "en"
  }
}
```

---

## API Reference

### 1. Non-Streaming REST API (OpenAI & Sarvam Format)
```bash
curl -X POST http://127.0.0.1:8790/v1/audio/transcriptions \
  -H "Authorization: Bearer local" \
  -F "file=@audio.wav"
```
Response:
```json
{
  "text": "Hi there. I just want to share how thrilled I am to work with you.",
  "transcript": "Hi there. I just want to share how thrilled I am to work with you.",
  "transcription": "Hi there. I just want to share how thrilled I am to work with you."
}
```

### 2. Streaming Conversational SST WebSocket
Connect to: `ws://127.0.0.1:8790/ws/conversational_sst` (or `/ws/smart_sst`)
- **Send**: Raw binary chunks of 16kHz, 16-bit mono signed PCM (`Int16Array`).
- **Receive**:
  ```json
  {"type": "interim", "text": "Hi there I just want to"}
  {"type": "final", "text": "Hi there. I just want to share how thrilled I am to work with you."}
  ```

---

## Interactive Web UI
Visit [http://127.0.0.1:8790](http://127.0.0.1:8790) in your browser:
- Test all 3 modes with live microphone visualization and real-time prompt updating.
