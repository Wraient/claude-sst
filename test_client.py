"""Test script for Claude Local Gateway.

Tests:
1. Health check
2. OpenAI REST API with Conversational Engine (transcribe file)
3. Smart Conversational SST (live dynamic prompt, no Claude answer)
4. Raw Dictation SST
5. Full Conversational Audio Mode
"""

import os
import asyncio
import json
import uuid
import aiohttp
import requests

PORT = os.getenv("PORT", "8790")
SERVER_URL = f"http://127.0.0.1:{PORT}"
WS_SMART_SST = f"ws://127.0.0.1:{PORT}/ws/conversational_sst"
WS_SST_URL = f"ws://127.0.0.1:{PORT}/ws/sst"
WS_VOICE_URL = f"ws://127.0.0.1:{PORT}/ws/voice"
TEST_AUDIO = "/home/wraient/.config/BraveSoftware/Brave-Origin/Default/Extensions/hehggadaopoacecdllhhajmbjkdcmajg/1.26.901.11451_0/codex-sidepanel/assets/cove.en.9d1a95cf-BcLNoRkM.wav"


def test_health():
    print("\n=== 1. Health Check (GET /health) ===")
    resp = requests.get(f"{SERVER_URL}/health")
    print("Status:", resp.status_code)
    print("Response:", resp.json())
    assert resp.status_code == 200


def test_rest_transcription():
    print("\n=== 2. OpenAI / Sarvam REST API (POST /v1/audio/transcriptions) ===")
    with open(TEST_AUDIO, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {"model": "conversational", "language": "en"}
        resp = requests.post(f"{SERVER_URL}/v1/audio/transcriptions", files=files, data=data)
    print("Status:", resp.status_code)
    print("Result:", resp.json())
    assert resp.status_code == 200
    res = resp.json()
    assert "text" in res and "transcript" in res


async def test_smart_sst_streaming():
    print("\n=== 3. Smart Conversational SST (Streaming & Reply Suppressed) ===")
    import subprocess
    proc = subprocess.Popen(
        ["ffmpeg", "-i", TEST_AUDIO, "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    pcm_data, _ = proc.communicate()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(WS_SMART_SST) as ws:
            print("Connected to Smart Conversational SST WebSocket")

            async def send_audio():
                chunk_size = 3200
                for i in range(0, len(pcm_data), chunk_size):
                    await ws.send_bytes(pcm_data[i:i+chunk_size])
                    await asyncio.sleep(0.05)
                await ws.send_str(json.dumps({"action": "stop"}))

            asyncio.create_task(send_audio())

            received_prompt = False
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    mtype = data.get("type")
                    if mtype in ("interim", "final"):
                        print(f"[{mtype.upper()}] Prompt: {data.get('text')}")
                        received_prompt = True
                    elif mtype == "assistant_text_delta":
                        print("ERROR: Claude assistant replied when it should have been suppressed!")
                        assert False
                    if mtype == "final":
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    break

            assert received_prompt, "Did not receive prompt updates"
            print("Verified: Smart SST returned dynamic prompt and suppressed Claude answer!")


async def test_raw_sst_streaming():
    print("\n=== 4. Raw Dictation SST WebSocket (Deepgram Nova-3) ===")
    import subprocess
    proc = subprocess.Popen(
        ["ffmpeg", "-i", TEST_AUDIO, "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    pcm_data, _ = proc.communicate()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(WS_SST_URL) as ws:
            print("Connected to Raw Dictation SST WebSocket")

            async def send_audio():
                chunk_size = 3200
                for i in range(0, len(pcm_data), chunk_size):
                    await ws.send_bytes(pcm_data[i:i+chunk_size])
                    await asyncio.sleep(0.04)
                await ws.send_str(json.dumps({"action": "stop"}))

            asyncio.create_task(send_audio())

            received_text = False
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "final":
                        print(f"[FINAL] Raw SST: {data.get('text')}")
                        received_text = True
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    break

            assert received_text, "Did not receive dictation SST text"


async def main():
    test_health()
    test_rest_transcription()
    await test_smart_sst_streaming()
    await test_raw_sst_streaming()
    print("\nAll automated tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
