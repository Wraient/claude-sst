"""Test script for Claude SST Local Proxy Server."""

import asyncio
import json
import aiohttp
import requests

SERVER_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws"
TEST_AUDIO = "/home/wraient/.config/BraveSoftware/Brave-Origin/Default/Extensions/hehggadaopoacecdllhhajmbjkdcmajg/1.26.901.11451_0/codex-sidepanel/assets/cove.en.9d1a95cf-BcLNoRkM.wav"


def test_health():
    print("\n--- 1. Testing GET /health ---")
    resp = requests.get(f"{SERVER_URL}/health")
    print("Status:", resp.status_code)
    print("Response:", resp.json())
    assert resp.status_code == 200


def test_rest_transcription():
    print("\n--- 2. Testing POST /v1/audio/transcriptions (REST API) ---")
    with open(TEST_AUDIO, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        resp = requests.post(f"{SERVER_URL}/v1/audio/transcriptions", files=files)
    print("Status:", resp.status_code)
    print("Transcription Result:")
    print(resp.json())
    assert resp.status_code == 200
    assert "text" in resp.json()


async def test_websocket_streaming():
    print("\n--- 3. Testing Bidirectional WebSocket Streaming (/ws) ---")
    with open("/tmp/test_speech.pcm", "rb") as f:
        pcm = f.read()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(WS_URL) as ws:
            print("Connected to Local Proxy WebSocket!")

            async def send_audio():
                chunk_size = 3200  # 100ms chunks
                for i in range(0, len(pcm), chunk_size):
                    await ws.send_bytes(pcm[i:i+chunk_size])
                    await asyncio.sleep(0.08)
                # Silence
                for _ in range(10):
                    await ws.send_bytes(b"\x00" * 3200)
                    await asyncio.sleep(0.08)
                await asyncio.sleep(0.5)
                await ws.close()

            sender = asyncio.create_task(send_audio())

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    evt_type = data.get("type")
                    text = data.get("text", "")
                    if evt_type in ("interim", "final"):
                        print(f"[{evt_type.upper():7}] -> {text}")
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

            await sender
            print("WebSocket streaming completed successfully!")


if __name__ == "__main__":
    test_health()
    test_rest_transcription()
    asyncio.run(test_websocket_streaming())
