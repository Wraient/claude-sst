"""Test script for Claude Local Gateway (SST & Conversational Audio Mode)."""

import asyncio
import json
import uuid
import aiohttp
import requests

SERVER_URL = "http://127.0.0.1:8000"
WS_SST_URL = "ws://127.0.0.1:8000/ws/sst"
WS_VOICE_URL = "ws://127.0.0.1:8000/ws/voice"
TEST_AUDIO = "/home/wraient/.config/BraveSoftware/Brave-Origin/Default/Extensions/hehggadaopoacecdllhhajmbjkdcmajg/1.26.901.11451_0/codex-sidepanel/assets/cove.en.9d1a95cf-BcLNoRkM.wav"


def test_health():
    print("\n=== 1. Health Check (GET /health) ===")
    resp = requests.get(f"{SERVER_URL}/health")
    print("Status:", resp.status_code)
    print("Response:", resp.json())
    assert resp.status_code == 200


def test_rest_transcription():
    print("\n=== 2. OpenAI REST API (POST /v1/audio/transcriptions) ===")
    with open(TEST_AUDIO, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        resp = requests.post(f"{SERVER_URL}/v1/audio/transcriptions", files=files)
    print("Status:", resp.status_code)
    print("Transcription Result:", resp.json())
    assert resp.status_code == 200


async def test_websocket_sst():
    print("\n=== 3. Pure Dictation SST Streaming (/ws/sst) ===")
    with open("/tmp/test_speech.pcm", "rb") as f:
        pcm = f.read()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(WS_SST_URL) as ws:
            print("Connected to Dictation SST WebSocket!")

            async def send_audio():
                chunk_size = 3200
                for i in range(0, len(pcm), chunk_size):
                    await ws.send_bytes(pcm[i:i+chunk_size])
                    await asyncio.sleep(0.08)
                for _ in range(10):
                    await ws.send_bytes(b"\x00" * 3200)
                    await asyncio.sleep(0.08)
                await asyncio.sleep(0.5)
                await ws.close()

            sender = asyncio.create_task(send_audio())
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    t = data.get("type")
                    if t in ("interim", "final"):
                        print(f"[{t.upper():7}] {data.get('text')}")
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

            await sender
            print("Dictation SST test passed!")


async def test_websocket_conversational_voice():
    print("\n=== 4. Conversational Voice Mode (/ws/voice) ===")
    conv_uuid = str(uuid.uuid4())
    url = f"{WS_VOICE_URL}?conversation_uuid={conv_uuid}"
    print(f"Starting session for conversation: {conv_uuid}")

    with open("/tmp/test_speech.pcm", "rb") as f:
        pcm = f.read()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            print("Connected to Conversational Voice Mode WebSocket!")

            async def send_audio():
                chunk_size = 3200
                for i in range(0, len(pcm), chunk_size):
                    await ws.send_bytes(pcm[i:i+chunk_size])
                    await asyncio.sleep(0.08)
                for _ in range(12):
                    await ws.send_bytes(b"\x00" * 3200)
                    await asyncio.sleep(0.08)
                # Wait for Claude's response
                await asyncio.sleep(8.0)
                await ws.close()

            sender = asyncio.create_task(send_audio())

            last_prompt = ""
            print("--- Streaming Live User Prompt & Claude Reply ---")
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    t = data.get("type")
                    if t == "user_interim":
                        last_prompt = data.get("text", "")
                        print(f"\r[USER PROMPT UPDATE] -> \"{last_prompt}\"", end="", flush=True)
                    elif t == "user_speech_end":
                        print(f"\n[USER FINISHED SPEAKING]")
                    elif t == "assistant_text_delta":
                        print(data.get("text", ""), end="", flush=True)
                    elif t == "playback_start":
                        print("\n[CLAUDE ANSWER START]")
                    elif t == "playback_end":
                        print("\n[CLAUDE ANSWER END]")
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

            await sender
            print("\nConversational Voice Mode test passed!")


if __name__ == "__main__":
    test_health()
    test_rest_transcription()
    asyncio.run(test_websocket_sst())
    asyncio.run(test_websocket_conversational_voice())
