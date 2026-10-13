"""Local Proxy Server for Claude Web Speech-to-Text (SST) & Conversational Audio Mode.

Endpoints:
1. Smart Conversational SST (/ws/conversational_sst or /ws/smart_sst):
   - Uses Claude's Conversational Voice pipeline to update prompt contextually
   - Suppresses Claude's assistant reply via instant interrupt
   - Deletes ephemeral conversation cleanly (no stray chats)
   - Output: Real-time updated prompt text without Claude answering

2. Full Conversational Voice Mode (/ws/voice):
   - Updates prompt dynamically
   - Claude assistant answers in context (text tokens + audio)

3. Raw Dictation SST (/ws/sst or /ws):
   - Direct Deepgram Nova-3 dictation stream

4. OpenAI & Sarvam Compatible REST API:
   - POST /v1/audio/transcriptions
   - POST /speech-to-text
   - POST /transcribe
   - Keeps transcription in memory and returns complete JSON response (text, transcript, transcription)

5. Interactive Web UI (GET /)
"""

import os
import json
import uuid
import base64
import logging
import asyncio
import subprocess
from typing import Optional
import numpy as np
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
      --claude-blue: #3b82f6;
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
      max-width: 900px;
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
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .mode-btn {
      padding: 0.65rem 1.1rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--card-bg);
      color: var(--text-muted);
      cursor: pointer;
      font-size: 0.88rem;
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
    .mode-badge {
      font-size: 0.75rem;
      padding: 0.25rem 0.6rem;
      border-radius: 12px;
      background: rgba(217, 119, 87, 0.2);
      color: var(--claude-accent);
      border: 1px solid rgba(217, 119, 87, 0.3);
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
      font-size: 0.82rem;
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
      <p class="subtitle">Conversational SST (Live Prompting), Full Voice Mode, &amp; Raw Dictation</p>
    </header>

    <div class="mode-selector">
      <button class="mode-btn active" id="btnSmartSst" onclick="switchMode('smart_sst')">
        &starf; Conversational SST (No Claude Answer)
      </button>
      <button class="mode-btn" id="btnVoice" onclick="switchMode('voice')">
        Conversational Audio (Claude Answers)
      </button>
      <button class="mode-btn" id="btnSst" onclick="switchMode('sst')">
        Raw Dictation SST (Deepgram Nova-3)
      </button>
    </div>

    <div class="card">
      <div class="card-title">
        <span id="activeModeTitle">Smart Conversational SST</span>
        <span class="mode-badge" id="modeBadge">Dynamic Prompting &bull; No Answer</span>
      </div>

      <div class="mic-controls">
        <button id="micBtn" class="mic-btn" title="Toggle Recording">
          <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
        </button>
        <div id="statusText" class="status-text">Click microphone to speak</div>
        <canvas id="viz" class="visualizer-canvas" height="40" width="850"></canvas>
      </div>

      <!-- Single Prompt View (Smart SST & Raw SST) -->
      <div id="singlePane">
        <div class="pane-header">
          <span class="prompt-accent" id="singlePaneTitle">Live Conversational Prompt</span>
          <span id="singleStatus"></span>
        </div>
        <div id="singleBox" class="pane-box">Speak to watch your prompt update dynamically in real time...</div>
      </div>

      <!-- Dual View (Conversational Voice Mode with Claude reply) -->
      <div class="dual-pane" id="dualPane" style="display: none;">
        <div>
          <div class="pane-header">
            <span class="prompt-accent">Live Prompt (You)</span>
          </div>
          <div id="promptBox" class="pane-box">Speak here...</div>
        </div>
        <div>
          <div class="pane-header">
            <span class="claude-accent">Claude Reply (Contextual)</span>
          </div>
          <div id="claudeBox" class="pane-box">Claude will answer here...</div>
        </div>
      </div>

      <div class="actions">
        <button class="btn" onclick="clearBoxes()">Clear</button>
        <button class="btn" onclick="copyResult()">Copy Prompt</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">WebSocket & REST Endpoints</div>
      <div class="code-snippet">
# 1. Non-streaming REST API (OpenAI & Sarvam compatible, returns complete response in memory):<br>
POST http://127.0.0.1:8790/v1/audio/transcriptions (or /speech-to-text)<br>
Returns: {"text": "...", "transcript": "...", "transcription": "..."}
      </div>
      <div class="code-snippet">
# 2. Smart Conversational SST WebSocket (Real-time dynamic prompt updates &bull; No Claude answer):<br>
ws://127.0.0.1:8790/ws/conversational_sst<br>
Stream 16kHz PCM &rarr; Receive {"type": "interim", "text": "..."} &amp; {"type": "final", "text": "..."}
      </div>
      <div class="code-snippet">
# 3. Full Conversational Audio Mode WebSocket (Claude answers in context):<br>
ws://127.0.0.1:8790/ws/voice?conversation_uuid=&lt;uuid&gt;<br>
Stream 16kHz PCM &rarr; Receive {"type": "user_interim", ...} &amp; {"type": "assistant_text_delta", ...}
      </div>
      <div class="code-snippet">
# 4. Raw Dictation SST WebSocket (Deepgram Nova-3):<br>
ws://127.0.0.1:8790/ws/sst (or /ws)<br>
Stream 16kHz PCM &rarr; Receive {"type": "final", "text": "..."}
      </div>
    </div>
  </div>

  <script>
    let currentMode = 'smart_sst';
    let ws = null;
    let audioContext = null;
    let micStream = null;
    let processor = null;
    let isRecording = false;

    const micBtn = document.getElementById('micBtn');
    const statusText = document.getElementById('statusText');
    const singlePane = document.getElementById('singlePane');
    const singleBox = document.getElementById('singleBox');
    const singlePaneTitle = document.getElementById('singlePaneTitle');
    const dualPane = document.getElementById('dualPane');
    const promptBox = document.getElementById('promptBox');
    const claudeBox = document.getElementById('claudeBox');
    const activeModeTitle = document.getElementById('activeModeTitle');
    const modeBadge = document.getElementById('modeBadge');
    const canvas = document.getElementById('viz');
    const canvasCtx = canvas.getContext('2d');

    function switchMode(mode) {
      if (isRecording) stopRecording();
      currentMode = mode;
      document.getElementById('btnSmartSst').classList.toggle('active', mode === 'smart_sst');
      document.getElementById('btnVoice').classList.toggle('active', mode === 'voice');
      document.getElementById('btnSst').classList.toggle('active', mode === 'sst');

      if (mode === 'smart_sst') {
        activeModeTitle.innerText = 'Smart Conversational SST';
        modeBadge.innerText = 'Dynamic Prompting • No Answer';
        singlePaneTitle.innerText = 'Live Conversational Prompt (Refined as you speak)';
        singlePane.style.display = 'block';
        dualPane.style.display = 'none';
      } else if (mode === 'voice') {
        activeModeTitle.innerText = 'Conversational Audio Mode';
        modeBadge.innerText = 'Full Turn Conversation';
        singlePane.style.display = 'none';
        dualPane.style.display = 'grid';
      } else {
        activeModeTitle.innerText = 'Raw Dictation SST';
        modeBadge.innerText = 'Deepgram Nova-3';
        singlePaneTitle.innerText = 'Transcribed Text';
        singlePane.style.display = 'block';
        dualPane.style.display = 'none';
      }
    }

    function clearBoxes() {
      singleBox.innerText = '';
      promptBox.innerText = '';
      claudeBox.innerText = '';
    }

    function copyResult() {
      const text = currentMode === 'voice'
        ? promptBox.innerText
        : singleBox.innerText;
      navigator.clipboard.writeText(text);
      alert('Prompt copied to clipboard!');
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
      canvasCtx.strokeStyle = '#d97757';
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
        let wsPath = '/ws/conversational_sst';
        if (currentMode === 'voice') wsPath = '/ws/voice';
        else if (currentMode === 'sst') wsPath = '/ws/sst';

        ws = new WebSocket(`${wsProto}//${window.location.host}${wsPath}`);

        ws.onopen = async () => {
          statusText.innerText = "Listening... Speak naturally";
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

            // Resample to 16000Hz PCM
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

          if (currentMode === 'smart_sst' || currentMode === 'sst') {
            if (t === 'interim' || t === 'final') {
              singleBox.innerText = data.text;
              singleBox.scrollTop = singleBox.scrollHeight;
            }
          } else if (currentMode === 'voice') {
            if (t === 'user_interim') {
              promptBox.innerText = data.text;
              promptBox.scrollTop = promptBox.scrollHeight;
            } else if (t === 'assistant_text_delta') {
              claudeBox.innerText += data.text;
              claudeBox.scrollTop = claudeBox.scrollHeight;
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
# WebSocket 1: Smart Conversational SST (Live Dynamic Prompt, NO Claude Answer)
# =====================================================================
@routes.get("/v1/audio/transcriptions/conversational")
@routes.get("/ws/conversational_sst")
@routes.get("/ws/smart_sst")
async def handle_ws_smart_sst(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    language = request.query.get("language", "en")
    logger.info("Client connected to Smart Conversational SST (suppress reply mode)")

    voice_session = ClaudeVoiceSession(
        language=language,
        suppress_reply=True,
        ephemeral=True,
    )

    try:
        await voice_session.connect()
    except Exception as e:
        logger.error(f"Failed to connect upstream to Claude Voice Mode: {e}")
        await ws.send_json({"type": "error", "error": f"Upstream connection failed: {e}"})
        await ws.close()
        return ws

    async def forward_audio():
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    await voice_session.send_audio(msg.data)
                elif msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get("action") == "stop":
                            await voice_session.send_interrupt()
                            break
                    except Exception:
                        pass
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    break
        except Exception as e:
            logger.debug(f"Smart SST forward error: {e}")

    async def forward_events():
        try:
            async for event in voice_session.receive_events():
                if ws.closed:
                    break
                await ws.send_json(event)
        except Exception as e:
            logger.debug(f"Smart SST event error: {e}")

    client_task = asyncio.create_task(forward_audio())
    server_task = asyncio.create_task(forward_events())

    done, pending = await asyncio.wait(
        [client_task, server_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

    await voice_session.close()
    if not ws.closed:
        await ws.close()

    logger.info("Smart Conversational SST session closed.")
    return ws


# =====================================================================
# WebSocket 2: Full Conversational Voice Mode (Context + Claude Answers)
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
    suppress_reply = request.query.get("suppress_reply", "false").lower() in ("true", "1")

    logger.info(f"Client connected to Conversational Voice Mode (conv: {conv_uuid}, suppress: {suppress_reply})")

    voice_session = ClaudeVoiceSession(
        conversation_uuid=conv_uuid,
        language=language,
        voice=voice,
        model=model,
        suppress_reply=suppress_reply,
    )

    try:
        await voice_session.connect()
    except Exception as e:
        logger.error(f"Failed to connect upstream to Claude Voice Mode: {e}")
        await ws.send_json({"type": "error", "error": f"Upstream connection failed: {e}"})
        await ws.close()
        return ws

    async def forward_client():
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

    async def forward_upstream():
        try:
            async for event in voice_session.receive_events():
                if ws.closed:
                    break
                await ws.send_json(event)
        except Exception as e:
            logger.debug(f"Voice downstream forward error: {e}")

    client_task = asyncio.create_task(forward_client())
    server_task = asyncio.create_task(forward_upstream())

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
# WebSocket 3: Raw Dictation SST (/ws, /ws/sst)
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
# WebSocket 4: OpenAI Realtime Compatible API (/v1/realtime, /realtime)
# Streams mic audio in real time as the user speaks, delivering instant transcript on commit
# =====================================================================
@routes.get("/v1/realtime")
@routes.get("/realtime")
async def handle_ws_realtime(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    language = request.query.get("language", "en")
    logger.info("Client connected to Realtime WebSocket (/v1/realtime)")

    voice_session: Optional[ClaudeVoiceSession] = None
    receiver_task: Optional[asyncio.Task] = None
    last_partial = ""
    lock = asyncio.Lock()

    async def ensure_voice_session():
        nonlocal voice_session, receiver_task, last_partial
        if voice_session and not voice_session._is_closed:
            return voice_session
        voice_session = ClaudeVoiceSession(
            language=language,
            suppress_reply=True,
            ephemeral=True,
        )
        await voice_session.connect()
        last_partial = ""

        async def receive_events_loop():
            nonlocal last_partial
            try:
                async for event in voice_session.receive_events():
                    if ws.closed:
                        break
                    if event.get("type") in ("interim", "user_interim"):
                        current_full = event.get("text", "")
                        if current_full:
                            if current_full.startswith(last_partial):
                                delta = current_full[len(last_partial):]
                            else:
                                delta = " " + current_full
                            last_partial = current_full
                            if delta:
                                await ws.send_json({
                                    "type": "conversation.item.input_audio_transcription.delta",
                                    "delta": delta,
                                })
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Realtime receiver loop error: {e}")

        receiver_task = asyncio.create_task(receive_events_loop())
        return voice_session

    async def close_voice_session():
        nonlocal voice_session, receiver_task
        if receiver_task:
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
            receiver_task = None
        if voice_session:
            await voice_session.close()
            voice_session = None

    try:
        session_id = str(uuid.uuid4())
        await ws.send_json({
            "type": "session.created",
            "session": {
                "id": session_id,
                "model": "claude-voice-conversational",
            }
        })

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue

                mtype = data.get("type")

                if mtype == "session.update":
                    sess = data.get("session", {})
                    audio_trans = sess.get("audio", {}).get("input", {}).get("transcription", {})
                    if audio_trans.get("language"):
                        language = audio_trans["language"]
                    await ws.send_json({
                        "type": "session.updated",
                        "session": sess,
                    })

                elif mtype == "input_audio_buffer.clear":
                    async with lock:
                        await close_voice_session()
                        await ensure_voice_session()
                    await ws.send_json({"type": "input_audio_buffer.cleared"})

                elif mtype == "input_audio_buffer.append":
                    b64_audio = data.get("audio", "")
                    if b64_audio:
                        raw_bytes = base64.b64decode(b64_audio)
                        if raw_bytes:
                            # Hyprwhspr sends 24kHz 16-bit mono PCM. Resample to 16kHz PCM for Claude.
                            pcm_in = np.frombuffer(raw_bytes, dtype=np.int16)
                            if len(pcm_in) > 0:
                                out_len = int(len(pcm_in) * 16000 / 24000)
                                if out_len > 0:
                                    orig_indices = np.linspace(0, len(pcm_in) - 1, out_len, endpoint=False)
                                    pcm_16k = np.interp(orig_indices, np.arange(len(pcm_in)), pcm_in).astype(np.int16).tobytes()
                                    async with lock:
                                        curr_session = await ensure_voice_session()
                                        await curr_session.send_audio(pcm_16k)

                elif mtype == "input_audio_buffer.commit":
                    # Mic released / recording stopped!
                    async with lock:
                        if voice_session and not voice_session._is_closed:
                            for _ in range(4):
                                await voice_session.send_audio(bytes(3200))
                                await asyncio.sleep(0.01)
                            await asyncio.sleep(0.12)
                            await voice_session.send_interrupt()

                            final_transcript = voice_session.accumulated_text
                        else:
                            final_transcript = ""

                    await ws.send_json({"type": "input_audio_buffer.committed"})
                    await ws.send_json({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": final_transcript,
                    })
                    logger.info(f"Realtime transcription committed ({len(final_transcript)} chars): {final_transcript[:80]}")

            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                break
    except Exception as e:
        logger.error(f"Realtime WebSocket handler error: {e}")
    finally:
        await close_voice_session()
        if not ws.closed:
            await ws.close()
        logger.info("Realtime WebSocket connection closed.")

    return ws


# =====================================================================
# REST: Non-Streaming Transcription API (OpenAI & Sarvam Compatible)
# Keeps entire output in memory and returns as one complete JSON response
# =====================================================================
@routes.post("/v1/audio/transcriptions")
@routes.post("/speech-to-text")
@routes.post("/transcribe")
async def handle_transcriptions(request: web.Request) -> web.Response:
    reader = await request.multipart()
    audio_bytes = None
    language = "en"
    engine = "conversational"  # default to high quality conversational engine

    while True:
        field = await reader.next()
        if field is None:
            break
        if field.name in ("file", "audio"):
            audio_bytes = await field.read()
        elif field.name in ("language", "language_code"):
            val = (await field.read()).decode("utf-8", errors="ignore").strip()
            if val and val != "unknown":
                language = val
        elif field.name in ("engine", "model"):
            val = (await field.read()).decode("utf-8", errors="ignore").strip()
            if val:
                engine = val
        else:
            await field.read()

    if not audio_bytes:
        return web.json_response({"error": "No audio file provided in request"}, status=400)

    # Transcode incoming audio (WAV, MP3, OGG, WebM, etc.) to 16kHz 16-bit mono PCM via ffmpeg
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
        pcm_data, err = proc.communicate(input=audio_bytes, timeout=45)
        if proc.returncode != 0:
            logger.error(f"FFmpeg error: {err.decode('utf-8', errors='ignore')}")
            return web.json_response({"error": "Audio transcoding failed"}, status=400)
    except Exception as e:
        logger.error(f"Failed to transcode audio: {e}")
        return web.json_response({"error": f"Transcoder error: {e}"}, status=500)

    final_text = ""

    # Choose between conversational engine and raw dictation engine
    if engine in ("sst", "dictation", "nova", "nova-3", "deepgram"):
        claude_session = ClaudeSSTSession(language=language)
        try:
            await claude_session.connect()

            async def send_audio_chunks():
                chunk_size = 3200
                for i in range(0, len(pcm_data), chunk_size):
                    await claude_session.send_audio(pcm_data[i:i+chunk_size])
                    await asyncio.sleep(0.02)
                for _ in range(8):
                    await claude_session.send_audio(b"\x00" * 3200)
                    await asyncio.sleep(0.02)
                await asyncio.sleep(0.5)
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
    else:
        # Default: Smart Conversational Engine (context-aware prompt builder across all pauses)
        voice_session = ClaudeVoiceSession(language=language, suppress_reply=True, ephemeral=True)
        try:
            await voice_session.connect()
            finished_sending = asyncio.Event()

            async def send_audio():
                chunk_size = 3200
                for i in range(0, len(pcm_data), chunk_size):
                    await voice_session.send_audio(pcm_data[i:i+chunk_size])
                    await asyncio.sleep(0.02)
                # Send silence frames to trigger turn completion for the final utterance
                for _ in range(16):
                    await voice_session.send_audio(b"\x00" * 3200)
                    await asyncio.sleep(0.02)
                finished_sending.set()

            sender = asyncio.create_task(send_audio())

            async def receive():
                try:
                    async for event in voice_session.receive_events():
                        pass
                except asyncio.CancelledError:
                    pass

            receiver = asyncio.create_task(receive())

            await finished_sending.wait()
            # Allow Claude to finalize the last utterance
            await asyncio.sleep(1.2)
            await voice_session.send_interrupt()
            receiver.cancel()

            final_text = voice_session.accumulated_text
        except Exception as e:
            logger.error(f"Error during conversational transcription: {e}")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            await voice_session.close()

    final_text = final_text.strip()
    logger.info(f"Completed non-streaming transcription ({len(final_text)} chars): {final_text[:60]}...")

    # Return universal response compatible with OpenAI, Sarvam, and generic clients
    return web.json_response({
        "text": final_text,
        "transcript": final_text,
        "transcription": final_text,
    })


def create_app() -> web.Application:
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.add_routes(routes)
    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", "8790"))
    host = os.getenv("HOST", "127.0.0.1")
    logger.info(f"Starting Claude Audio & SST Proxy on http://{host}:{port}")
    web.run_app(app, host=host, port=port)
