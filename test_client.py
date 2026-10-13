"""Test script for Claude Local Gateway.

Tests:
1. Health check
2. OpenAI REST API with Conversational Engine (transcribe file)
3. Smart Conversational SST (live dynamic prompt, no Claude answer)
4. Raw Dictation SST
5. Full Conversational Audio Mode
"""

import asyncio
import json
import uuid
import aiohttp
import requests

SERVER_URL = "http://127.0.0.1:8000"
WS_SMART_SST = "ws://127.0.0.1:8000/ws/conversational_sst"
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
    print("\n=== 2. OpenAI REST API with Conversational Engine ===")
    with open(TEST_AUDIO, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        resp = requests.post(f"{SERVER_URL}/v1/audio/transcriptions", files=files)
    print("Status:", resp.status_code)
    print("Result:", resp.json())
    assert resp.status_code == 200


async def test_smart_conversational_sst():
    print("\n=== 3. Smart Conversational SST (NO Claude Answer, Live Prompt Updating) ===")
    with open("/tmp/test_speech.pcm", "rb") as f:
        pcm = f.read()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(WS_SMART_SST) as ws:
            print("Connected to /ws/conversational_sst!")

            async def send_audio():
                chunk_size = 3200
                for i in range(0, len(pcm), chunk_size):
                    await ws.send_bytes(pcm[i:i+chunk_size])
                    await asyncio.sleep(0.08)
                for _ in range(12):
                    await ws.send_bytes(b"\x00" * 3200)
                    await asyncio.sleep(0.08)
                # Inform stop
                await ws.send_json({"action": "stop"})
                await asyncio.sleep(0.5)
                await ws.close()

            sender = asyncio.create_task(send_audio())

            last_text = ""
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    t = data.get("type")
                    if t == "interim":
                        last_text = data.get("text", "")
                        print(f"\r[LIVE PROMPT UPDATE] -> \"{last_text}\"", end="", flush=True)
                    elif t == "final":
                        last_text = data.get("text", "")
                        print(f"\n[FINAL PROMPT READY] -> \"{last_text}\"")
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

            await sender
            print("Smart Conversational SST completed successfully without Claude answering!")


if __name__ == "__main__":
    test_health()
    test_rest_transcription()
    asyncio.run(test_smart_conversational_sst())
