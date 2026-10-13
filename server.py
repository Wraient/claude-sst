"""Local Proxy Server for Claude Web Speech-to-Text (SST).

Provides:
1. Bidirectional streaming WebSocket API (/v1/audio/transcriptions/stream or /ws)
2. OpenAI-compatible REST API (POST /v1/audio/transcriptions)
3. Built-in interactive Web UI with live microphone streaming (GET /)
4. Health check endpoint (GET /health)
"""

import os
import io
import json
import logging
import asyncio
import subprocess
from typing import Optional
from aiohttp import web

from auth import auth_manager
from claude_sst_client import ClaudeSSTSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("claude_sst_proxy")

routes = web.RouteTableDef()


# =====================================================================
# HTML / Web UI
# =====================================================================
HTML_INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Claude Web SST &bull; Local Proxy</title>
  <style>
    :root {
      --bg: #18181b;
      --card-bg: #27272a;
      --border: #3f3f46;
      --primary: #d97706;
      --primary-hover: #b45309;
      --claude-accent: #cc785c;
      --text: #f4f4f5;
      --text-muted: #a1a1aa;
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
      max-width: 800px;
      width: 100%;
    }
    header {
      margin-bottom: 2rem;
      text-align: center;
    }
    h1 {
      font-size: 2rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }
    h1 span { color: var(--claude-accent); }
    p.subtitle {
      color: var(--text-muted);
      margin-top: 0.5rem;
      font-size: 0.95rem;
    }
    .badge {
      display: inline-block;
      padding: 0.25rem 0.6rem;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 9999px;
      margin-top: 0.5rem;
    }
    .badge-online { background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .card-title {
      font-size: 1.1rem;
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
      gap: 1.25rem;
      padding: 1rem 0;
    }
    .mic-btn {
      width: 80px;
      height: 80px;
      border-radius: 50%;
      background: var(--claude-accent);
      border: none;
      color: white;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
      box-shadow: 0 0 20px rgba(204, 120, 92, 0.4);
    }
    .mic-btn:hover {
      transform: scale(1.05);
      background: #b5674c;
    }
    .mic-btn.recording {
      background: var(--danger);
      animation: pulse 1.5s infinite;
      box-shadow: 0 0 25px rgba(239, 68, 68, 0.6);
    }
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.1); }
      100% { transform: scale(1); }
    }
    .mic-btn svg { width: 36px; height: 36px; fill: currentColor; }
    .status-text {
      font-size: 0.95rem;
      color: var(--text-muted);
      font-weight: 500;
    }
    .visualizer-canvas {
      width: 100%;
      height: 48px;
      border-radius: 6px;
      background: rgba(0,0,0,0.2);
    }
    .transcript-box {
      min-height: 180px;
      max-height: 350px;
      overflow-y: auto;
      background: #1e1e21;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      font-size: 1.15rem;
      line-height: 1.6;
      color: #fafafa;
      word-wrap: break-word;
      white-space: pre-wrap;
    }
    .transcript-box:empty:before {
      content: "Transcript will appear here as you speak...";
      color: #666;
      font-style: italic;
    }
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
      background: #333338;
      color: #eee;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 500;
      transition: background 0.15s;
    }
    button.btn:hover { background: #44444c; }
    .file-dropzone {
      border: 2px dashed var(--border);
      border-radius: 8px;
      padding: 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s;
    }
    .file-dropzone:hover { border-color: var(--claude-accent); }
    .code-snippet {
      background: #111113;
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
      <h1><span>Claude Web SST</span> Local Proxy</h1>
      <p class="subtitle">Direct Bidirectional Streaming Speech-to-Text Gateway</p>
      <div id="conn-badge" class="badge badge-online">Proxy Active &bull; ws://localhost:8000/ws</div>
    </header>

    <div class="card">
      <div class="card-title">
        <span>Live Microphone Dictation</span>
        <span id="timing-tag" style="font-size: 0.8rem; color: var(--text-muted);">Latency: ~150ms</span>
      </div>
      
      <div class="mic-controls">
        <button id="micBtn" class="mic-btn" title="Toggle Recording">
          <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
        </button>
        <div id="statusText" class="status-text">Click microphone to start talking</div>
        <canvas id="viz" class="visualizer-canvas" height="48" width="700"></canvas>
      </div>

      <div id="transcriptBox" class="transcript-box"></div>
      
      <div class="actions">
        <button class="btn" id="copyBtn">Copy Text</button>
        <button class="btn" id="clearBtn">Clear</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">File Transcription (REST API)</div>
      <div class="file-dropzone" id="dropzone">
        <input type="file" id="fileInput" accept="audio/*" style="display:none;">
        <p>Drop audio file here or <strong style="color:var(--claude-accent);">Browse</strong> (WAV, MP3, M4A, WebM)</p>
        <div id="fileUploadStatus" style="font-size:0.85rem; color:var(--text-muted); margin-top:0.5rem;"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">API Usage (WebSocket & REST)</div>
      <p style="font-size:0.9rem; color:var(--text-muted);">Connect directly to this proxy from any Python / Node.js / cURL application:</p>
      <div class="code-snippet">
# OpenAI-Compatible REST API:<br>
curl -X POST http://localhost:8000/v1/audio/transcriptions \\<br>
&nbsp;&nbsp;-F "file=@audio.mp3" \\<br>
&nbsp;&nbsp;-F "model=claude-sst"
      </div>
      <div class="code-snippet" style="margin-top: 0.5rem;">
# Bidirectional WebSocket API:<br>
ws://localhost:8000/v1/audio/transcriptions/stream<br>
Stream 16000Hz 16-bit mono PCM binary frames -> Receive JSON events {type, text}
      </div>
    </div>
  </div>

  <script>
    let ws = null;
    let audioContext = null;
    let micStream = null;
    let processor = null;
    let isRecording = false;

    const micBtn = document.getElementById('micBtn');
    const statusText = document.getElementById('statusText');
    const transcriptBox = document.getElementById('transcriptBox');
    const clearBtn = document.getElementById('clearBtn');
    const copyBtn = document.getElementById('copyBtn');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const fileUploadStatus = document.getElementById('fileUploadStatus');
    const canvas = document.getElementById('viz');
    const canvasCtx = canvas.getContext('2d');

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
      canvasCtx.strokeStyle = '#cc785c';
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
        ws = new WebSocket(`${wsProto}//${window.location.host}/ws`);

        ws.onopen = async () => {
          statusText.innerText = "Listening... Speak naturally";
          micBtn.classList.add('recording');
          isRecording = true;

          // Request mic access
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
          if (data.type === 'interim' || data.type === 'final') {
            transcriptBox.innerText = data.text;
            transcriptBox.scrollTop = transcriptBox.scrollHeight;
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
        console.error("Failed to start recording:", err);
        statusText.innerText = "Error accessing mic: " + err.message;
        stopRecording();
      }
    }

    function stopRecording() {
      if (!isRecording) return;
      isRecording = false;
      micBtn.classList.remove('recording');
      statusText.innerText = "Stopped. Click microphone to speak again.";

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
      if (isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    });

    clearBtn.addEventListener('click', () => {
      transcriptBox.innerText = '';
    });

    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(transcriptBox.innerText);
      const prev = copyBtn.innerText;
      copyBtn.innerText = 'Copied!';
      setTimeout(() => copyBtn.innerText = prev, 1500);
    });

    // File upload
    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async (e) => {
      if (!e.target.files.length) return;
      const file = e.target.files[0];
      fileUploadStatus.innerText = `Transcribing ${file.name}...`;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const resp = await fetch('/v1/audio/transcriptions', {
          method: 'POST',
          body: formData
        });
        const res = await resp.json();
        if (res.text) {
          transcriptBox.innerText = res.text;
          fileUploadStatus.innerText = `Transcription complete! (${file.name})`;
        } else {
          fileUploadStatus.innerText = `Error: ${res.error || 'Failed to transcribe'}`;
        }
      } catch (err) {
        fileUploadStatus.innerText = `Error: ${err.message}`;
      }
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
# WebSocket: Bidirectional Streaming SST
# =====================================================================
@routes.get("/v1/audio/transcriptions/stream")
@routes.get("/ws")
async def handle_ws_stream(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    language = request.query.get("language", "en")
    logger.info(f"Client connected to WebSocket stream (language: {language})")

    claude_session = ClaudeSSTSession(language=language)
    try:
        await claude_session.connect()
    except Exception as e:
        logger.error(f"Failed to connect upstream to Claude SST: {e}")
        await ws.send_json({"type": "error", "error": f"Upstream connection failed: {e}"})
        await ws.close()
        return ws

    async def forward_upstream():
        """Read binary audio chunks from client and forward to Claude."""
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
        """Read transcript events from Claude and stream back to client."""
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

    logger.info("WebSocket streaming session finished.")
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
            chunk_size = 3200  # 100ms
            for i in range(0, len(pcm_data), chunk_size):
                await claude_session.send_audio(pcm_data[i:i+chunk_size])
                await asyncio.sleep(0.04)  # 2.5x real-time streaming
            # Trailing silence to trigger endpointing
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
    logger.info(f"Starting Claude SST Proxy on http://127.0.0.1:{port}")
    web.run_app(app, host="127.0.0.1", port=port)
