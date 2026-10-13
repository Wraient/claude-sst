"""Local Proxy Server for Claude Web Speech-to-Text (SST) & Conversational Audio Mode.

Provides:
1. Conversational Audio Mode WebSocket (/ws/voice or /v1/voice/stream):
   - Updates prompt dynamically as spoken
   - Maintains conversation context
   - Streams Claude assistant text responses token by token
   - Streams Claude audio responses
2. Pure Dictation SST WebSocket (/ws/sst or /ws or /v1/audio/transcriptions/stream):
   - Fast raw speech-to-text transcription without assistant replies
3. OpenAI-Compatible REST API (POST /v1/audio/transcriptions)
4. Interactive Web UI with Mode Switcher (GET /)
5. Health Check (GET /health)
"""

import os
import json
import logging
import asyncio
import subprocess
from typing import Optional
from aiohttp import web

from auth import auth_manager
from claude_sst_client import ClaudeSSTSession
from claude_voice_client import ClaudeVoiceSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("claude_proxy")

routes = web.RouteTableDef()


# =====================================================================
# HTML / Web UI
# =====================================================================
HTML_INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Claude Audio & SST &bull; Local Proxy</title>
  <style>
    :root {
      --bg: #131316;
      --card-bg: #1e1e24;
      --card-inner: #16161a;
      --border: #32323a;
      --claude-accent: #d97757;
      --claude-accent-dark: #b85c3e;
      --claude-purple: #9065b0;
      --text: #f4f4f7;
      --text-muted: #8e8e99;
      --success: #22c55e;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body {
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 2rem 1rem;
    }
    .container {
      max-width: 860px;
      width: 100%;
    }
    header {
      margin-bottom: 1.5rem;
      text-align: center;
    }
    h1 {
      font-size: 2.1rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }
    h1 span { color: var(--claude-accent); }
    p.subtitle {
      color: var(--text-muted);
      margin-top: 0.4rem;
      font-size: 0.95rem;
    }
    .mode-selector {
      display: flex;
      justify-content: center;
      margin: 1.25rem 0 1.75rem;
      gap: 0.75rem;
    }
    .mode-btn {
      padding: 0.65rem 1.25rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--card-bg);
      color: var(--text-muted);
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: 600;
      transition: all 0.2s;
    }
    .mode-btn.active {
      background: var(--claude-accent);
      color: white;
      border-color: var(--claude-accent);
      box-shadow: 0 0 15px rgba(217, 119, 87, 0.4);
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .card-title {
      font-size: 1.05rem;
      font-weight: 600;
      margin-bottom: 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .mic-controls {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1rem;
      padding: 0.75rem 0;
    }
    .mic-btn {
      width: 76px;
      height: 76px;
      border-radius: 50%;
      background: var(--claude-accent);
      border: none;
      color: white;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
      box-shadow: 0 0 20px rgba(217, 119, 87, 0.4);
    }
    .mic-btn:hover {
      transform: scale(1.05);
      background: var(--claude-accent-dark);
    }
    .mic-btn.recording {
      background: var(--danger);
      animation: pulse 1.4s infinite;
      box-shadow: 0 0 25px rgba(239, 68, 68, 0.6);
    }
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.08); }
      100% { transform: scale(1); }
    }
    .mic-btn svg { width: 34px; height: 34px; fill: currentColor; }
    .status-text {
      font-size: 0.95rem;
      color: var(--text-muted);
      font-weight: 500;
    }
    .visualizer-canvas {
      width: 100%;
      height: 40px;
      border-radius: 6px;
      background: rgba(0,0,0,0.3);
    }
    .dual-pane {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }
    @media (max-width: 700px) {
      .dual-pane { grid-template-columns: 1fr; }
    }
    .pane-box {
      background: var(--card-inner);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      min-height: 220px;
      max-height: 380px;
      overflow-y: auto;
      font-size: 1rem;
      line-height: 1.6;
      word-wrap: break-word;
      white-space: pre-wrap;
    }
    .pane-header {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
      display: flex;
      justify-content: space-between;
    }
    .prompt-accent { color: var(--claude-accent); }
    .claude-accent { color: #a78bfa; }
    .actions {
      display: flex;
      gap: 0.75rem;
      margin-top: 1rem;
      justify-content: flex-end;
    }
    button.btn {
      padding: 0.5rem 1rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #2a2a32;
      color: #eee;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 500;
      transition: background 0.15s;
    }
    button.btn:hover { background: #3b3b46; }
    .code-snippet {
      background: #101013;
      border-radius: 6px;
      padding: 0.75rem 1rem;
      font-family: monospace;
      font-size: 0.85rem;
      color: #e4e4e7;
      overflow-x: auto;
      margin-top: 0.5rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1><span>Claude Audio</span> Local Gateway</h1>
      <p class="subtitle">Conversational Voice Mode (Context + Live Prompts) &amp; Speech-to-Text</p>
    </header>

    <div class="mode-selector">
      <button class="mode-btn active" id="modeVoiceBtn" onclick="switchMode('voice')">
        Conversational Audio Mode (Live Prompt &amp; Context)
      </button>
      <button class="mode-btn" id="modeSstBtn" onclick="switchMode('sst')">
        Pure Dictation SST (Transcribe Only)
      </button>
    </div>

    <div class="card">
      <div class="card-title">
        <span id="activeModeTitle">Conversational Audio Mode</span>
        <span style="font-size: 0.8rem; color: var(--text-muted);">Real-time Bidirectional Stream</span>
      </div>

      <div class="mic-controls">
        <button id="micBtn" class="mic-btn" title="Toggle Recording">
          <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
        </button>
        <div id="statusText" class="status-text">Click microphone to speak</div>
        <canvas id="viz" class="visualizer-canvas" height="40" width="800"></canvas>
      </div>

      <!-- Conversational View -->
      <div class="dual-pane" id="dualPane">
        <div>
          <div class="pane-header">
            <span class="prompt-accent">Live Updated Prompt (Your Speech)</span>
            <span id="promptStatus"></span>
          </div>
          <div id="promptBox" class="pane-box">Speak to watch your prompt update dynamically...</div>
        </div>
        <div>
          <div class="pane-header">
            <span class="claude-accent">Claude Response (In Context)</span>
            <span id="claudeStatus"></span>
          </div>
          <div id="claudeBox" class="pane-box">Claude's streamed response will appear here...</div>
        </div>
      </div>

      <!-- Pure SST View (Hidden in voice mode) -->
      <div id="sstPane" style="display: none;">
        <div class="pane-header">
          <span class="prompt-accent">Transcript (Speech-to-Text)</span>
        </div>
        <div id="sstBox" class="pane-box">Transcription will appear here as you speak...</div>
      </div>

      <div class="actions">
        <button class="btn" id="newChatBtn" onclick="resetConversation()">New Session</button>
        <button class="btn" onclick="clearBoxes()">Clear</button>
        <button class="btn" onclick="copyResult()">Copy Text</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">API Integration Endpoints</div>
      <p style="font-size:0.9rem; color:var(--text-muted); margin-bottom: 0.5rem;">
        Connect to either streaming mode from your code or CLI:
      </p>
      <div class="code-snippet">
# 1. Conversational Audio Mode (Live prompt updates + Claude context response):<br>
ws://localhost:8000/ws/voice?conversation_uuid=&lt;uuid&gt;<br>
Stream 16kHz PCM audio &rarr; Receive {type: 'user_interim', text: '...'} &amp; {type: 'assistant_text_delta', text: '...'}
      </div>
      <div class="code-snippet">
# 2. Pure Dictation SST (Transcribe only):<br>
ws://localhost:8000/ws/sst (or /v1/audio/transcriptions/stream)<br>
Stream 16kHz PCM audio &rarr; Receive {type: 'final', text: '...'}
      </div>
      <div class="code-snippet">
# 3. OpenAI-Compatible File Transcription (REST):<br>
curl -X POST http://localhost:8000/v1/audio/transcriptions -F "file=@audio.mp3"
      </div>
    </div>
  </div>

  <script>
    let currentMode = 'voice';
    let ws = null;
    let audioContext = null;
    let micStream = null;
    let processor = null;
    let isRecording = false;
    let convUuid = null;

    const micBtn = document.getElementById('micBtn');
    const statusText = document.getElementById('statusText');
    const promptBox = document.getElementById('promptBox');
    const claudeBox = document.getElementById('claudeBox');
    const sstBox = document.getElementById('sstBox');
    const dualPane = document.getElementById('dualPane');
    const sstPane = document.getElementById('sstPane');
    const activeModeTitle = document.getElementById('activeModeTitle');
    const canvas = document.getElementById('viz');
    const canvasCtx = canvas.getContext('2d');

    function switchMode(mode) {
      if (isRecording) stopRecording();
      currentMode = mode;
      document.getElementById('modeVoiceBtn').classList.toggle('active', mode === 'voice');
      document.getElementById('modeSstBtn').classList.toggle('active', mode === 'sst');

      if (mode === 'voice') {
        activeModeTitle.innerText = 'Conversational Audio Mode';
        dualPane.style.display = 'grid';
        sstPane.style.display = 'none';
      } else {
        activeModeTitle.innerText = 'Pure Dictation SST Mode';
        dualPane.style.display = 'none';
        sstPane.style.display = 'block';
      }
    }

    function resetConversation() {
      convUuid = ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
        (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
      );
      promptBox.innerText = 'New session ready. Start speaking!';
      claudeBox.innerText = '';
      if (isRecording) stopRecording();
    }
    resetConversation();

    function clearBoxes() {
      promptBox.innerText = '';
      claudeBox.innerText = '';
      sstBox.innerText = '';
    }

    function copyResult() {
      const text = currentMode === 'voice'
        ? `Prompt: ${promptBox.innerText}\\n\\nClaude: ${claudeBox.innerText}`
        : sstBox.innerText;
      navigator.clipboard.writeText(text);
      alert('Copied to clipboard!');
    }

    // Visualizer loop
    let analyser = null;
    function drawViz() {
      requestAnimationFrame(drawViz);
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      if (!isRecording || !analyser) {
        canvasCtx.fillStyle = '#222';
        canvasCtx.fillRect(0, canvas.height/2 - 1, canvas.width, 2);
        return;
      }
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteTimeDomainData(dataArray);

      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = currentMode === 'voice' ? '#d97757' : '#22c55e';
      canvasCtx.beginPath();
      const sliceWidth = canvas.width * 1.0 / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;
        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
        x += sliceWidth;
      }
      canvasCtx.lineTo(canvas.width, canvas.height / 2);
      canvasCtx.stroke();
    }
    drawViz();

    async function startRecording() {
      try {
        const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsPath = currentMode === 'voice'
          ? `/ws/voice?conversation_uuid=${convUuid}`
          : `/ws/sst`;

        ws = new WebSocket(`${wsProto}//${window.location.host}${wsPath}`);

        ws.onopen = async () => {
          statusText.innerText = "Listening... Speak now";
          micBtn.classList.add('recording');
          isRecording = true;

          micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              sampleRate: { ideal: 16000 },
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true
            }
          });

          audioContext = new (window.AudioContext || window.webkitAudioContext)();
          const source = audioContext.createMediaStreamSource(micStream);
          analyser = audioContext.createAnalyser();
          analyser.fftSize = 512;
          source.connect(analyser);

          const inputSampleRate = audioContext.sampleRate;
          const targetSampleRate = 16000;
          const bufferSize = 2048;

          processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
          processor.onaudioprocess = (e) => {
            if (!isRecording || ws.readyState !== WebSocket.OPEN) return;
            const inputData = e.inputBuffer.getChannelData(0);

            // Resample to 16000Hz
            const downsampledLength = Math.floor(inputData.length * (targetSampleRate / inputSampleRate));
            const pcm16 = new Int16Array(downsampledLength);
            for (let i = 0; i < downsampledLength; i++) {
              const originalIndex = Math.floor(i * (inputSampleRate / targetSampleRate));
              let s = Math.max(-1, Math.min(1, inputData[originalIndex]));
              pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            ws.send(pcm16.buffer);
          };

          source.connect(processor);
          processor.connect(audioContext.destination);
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          const t = data.type;

          if (currentMode === 'voice') {
            if (t === 'user_interim') {
              promptBox.innerText = data.text;
              promptBox.scrollTop = promptBox.scrollHeight;
            } else if (t === 'user_speech_end') {
              if (data.final_prompt) promptBox.innerText = data.final_prompt;
              claudeBox.innerText = '';
              statusText.innerText = "Claude is thinking and replying...";
            } else if (t === 'assistant_text_delta') {
              claudeBox.innerText += data.text;
              claudeBox.scrollTop = claudeBox.scrollHeight;
            } else if (t === 'playback_end') {
              statusText.innerText = "Listening... Speak whenever you're ready";
            }
          } else {
            // Pure SST
            if (t === 'interim' || t === 'final') {
              sstBox.innerText = data.text;
              sstBox.scrollTop = sstBox.scrollHeight;
            }
          }
        };

        ws.onerror = (e) => {
          console.error("WS error:", e);
          stopRecording();
        };

        ws.onclose = () => {
          stopRecording();
        };

      } catch (err) {
        console.error("Recording error:", err);
        statusText.innerText = "Error: " + err.message;
        stopRecording();
      }
    }

    function stopRecording() {
      if (!isRecording) return;
      isRecording = false;
      micBtn.classList.remove('recording');
      statusText.innerText = "Stopped. Click microphone to speak.";

      if (processor) {
        processor.disconnect();
        processor = null;
      }
      if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
      }
      if (audioContext) {
        audioContext.close();
        audioContext = null;
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
      ws = null;
    }

    micBtn.addEventListener('click', () => {
      if (isRecording) stopRecording();
      else startRecording();
    });
  </script>
</body>
</html>
"""


# =====================================================================
# REST: Home & Health Check
# =====================================================================
@routes.get("/")
async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=HTML_INDEX, content_type="text/html")


@routes.get("/health")
async def handle_health(request: web.Request) -> web.Response:
    try:
        _, session_key, org_uuid = auth_manager.get_credentials()
        return web.json_response({
            "status": "healthy",
            "authenticated": bool(session_key),
            "org_uuid": org_uuid,
            "session_prefix": session_key[:15] + "..." if session_key else None,
        })
    except Exception as e:
        return web.json_response({"status": "unhealthy", "error": str(e)}, status=500)


# =====================================================================
# WebSocket: Mode 1 - Conversational Voice Mode (Context + Live Prompts)
# =====================================================================
@routes.get("/v1/voice/stream")
@routes.get("/ws/voice")
async def handle_ws_voice(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    conv_uuid = request.query.get("conversation_uuid")
    language = request.query.get("language", "en")
    voice = request.query.get("voice", "buttery")
    model = request.query.get("model")

    logger.info(f"Client connected to Conversational Voice Mode (conv: {conv_uuid})")

    voice_session = ClaudeVoiceSession(
        conversation_uuid=conv_uuid,
        language=language,
        voice=voice,
        model=model,
    )

    try:
        await voice_session.connect()
    except Exception as e:
        logger.error(f"Failed to connect upstream to Claude Voice Mode: {e}")
        await ws.send_json({"type": "error", "error": f"Upstream connection failed: {e}"})
        await ws.close()
        return ws

    async def forward_client_to_upstream():
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    await voice_session.send_audio(msg.data)
                elif msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "interrupt":
                            await voice_session.send_interrupt()
                        elif data.get("action") == "stop":
                            break
                    except Exception:
                        pass
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    break
        except Exception as e:
            logger.debug(f"Voice client forward error: {e}")

    async def forward_upstream_to_client():
        try:
            async for event in voice_session.receive_events():
                if ws.closed:
                    break
                if event.get("type") == "assistant_audio":
                    # Audio bytes to client if needed
                    # Only send if client requested binary audio, or ignore
                    pass
                else:
                    await ws.send_json(event)
        except Exception as e:
            logger.debug(f"Voice downstream forward error: {e}")

    client_task = asyncio.create_task(forward_client_to_upstream())
    server_task = asyncio.create_task(forward_upstream_to_client())

    done, pending = await asyncio.wait(
        [client_task, server_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

    await voice_session.close()
    if not ws.closed:
        await ws.close()

    logger.info("Voice streaming session ended.")
    return ws


# =====================================================================
# WebSocket: Mode 2 - Pure Dictation SST (/ws, /ws/sst)
# =====================================================================
@routes.get("/v1/audio/transcriptions/stream")
@routes.get("/ws/sst")
@routes.get("/ws")
async def handle_ws_sst(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    language = request.query.get("language", "en")
    logger.info(f"Client connected to Dictation SST stream (language: {language})")

    claude_session = ClaudeSSTSession(language=language)
    try:
        await claude_session.connect()
    except Exception as e:
        logger.error(f"Failed to connect upstream to Claude SST: {e}")
        await ws.send_json({"type": "error", "error": f"Upstream connection failed: {e}"})
        await ws.close()
        return ws

    async def forward_upstream():
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    await claude_session.send_audio(msg.data)
                elif msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get("action") == "stop":
                            break
                    except Exception:
                        pass
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    break
        except Exception as e:
            logger.debug(f"Client forward error: {e}")

    async def forward_downstream():
        try:
            async for event in claude_session.receive_events():
                if not ws.closed:
                    await ws.send_json(event)
        except Exception as e:
            logger.debug(f"Downstream forward error: {e}")

    upstream_task = asyncio.create_task(forward_upstream())
    downstream_task = asyncio.create_task(forward_downstream())

    done, pending = await asyncio.wait(
        [upstream_task, downstream_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

    await claude_session.close()
    if not ws.closed:
        await ws.close()

    logger.info("SST streaming session finished.")
    return ws


# =====================================================================
# REST: OpenAI-Compatible Transcriptions (POST /v1/audio/transcriptions)
# =====================================================================
@routes.post("/v1/audio/transcriptions")
async def handle_transcriptions(request: web.Request) -> web.Response:
    reader = await request.multipart()
    audio_bytes = None
    language = "en"

    while True:
        field = await reader.next()
        if field is None:
            break
        if field.name in ("file", "audio"):
            audio_bytes = await field.read()
        elif field.name == "language":
            language = (await field.read()).decode("utf-8").strip()

    if not audio_bytes:
        return web.json_response({"error": "No audio file provided in request"}, status=400)

    # Transcode audio to 16kHz 16-bit mono PCM via ffmpeg
    try:
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-i", "pipe:0",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pcm_data, err = proc.communicate(input=audio_bytes, timeout=30)
        if proc.returncode != 0:
            logger.error(f"FFmpeg error: {err.decode('utf-8', errors='ignore')}")
            return web.json_response({"error": "Audio transcoding failed"}, status=400)
    except Exception as e:
        logger.error(f"Failed to transcode audio: {e}")
        return web.json_response({"error": f"Transcoder error: {e}"}, status=500)

    # Stream to Claude SST and collect final text
    final_text = ""
    claude_session = ClaudeSSTSession(language=language)
    try:
        await claude_session.connect()

        async def send_audio_chunks():
            chunk_size = 3200
            for i in range(0, len(pcm_data), chunk_size):
                await claude_session.send_audio(pcm_data[i:i+chunk_size])
                await asyncio.sleep(0.04)
            for _ in range(12):
                await claude_session.send_audio(b"\x00" * 3200)
                await asyncio.sleep(0.04)
            await asyncio.sleep(0.8)
            await claude_session.close()

        sender = asyncio.create_task(send_audio_chunks())

        async for event in claude_session.receive_events():
            if event.get("type") in ("final", "interim"):
                txt = event.get("text", "")
                if txt:
                    final_text = txt

        await sender
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return web.json_response({"error": str(e)}, status=500)
    finally:
        await claude_session.close()

    return web.json_response({"text": final_text})


def create_app() -> web.Application:
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.add_routes(routes)
    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Claude Audio & SST Proxy on http://127.0.0.1:{port}")
    web.run_app(app, host="127.0.0.1", port=port)
