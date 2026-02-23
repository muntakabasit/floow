"""
FLOW — Live Bilingual Interpreter Prototype
Server: FastAPI + WebSocket proxy to OpenAI Realtime API

Architecture:
  Browser mic → WS to this server → WS to OpenAI Realtime API → translation text back

This is a throwaway prototype. Favor clarity over abstraction.
"""

import asyncio
import json
import os
import sys
import base64
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import websockets


# Force unbuffered stdout so logs appear immediately
def log(msg):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Local dev: load .env from current dir. Fly.io: secrets are real env vars.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env file.")

# OpenAI Realtime API config
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
OPENAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta": "realtime=v1",
}

# ---------------------------------------------------------------------------
# Interpreter system prompt — the soul of Flow
# ---------------------------------------------------------------------------

INTERPRETER_PROMPT = (
    "You are FLOW, a bilingual conversation interpreter for live human dialogue.\n"
    "Your only job is to translate what one person says from the source language "
    "to the target language.\n\n"
    "RULES — follow these exactly:\n"
    "- Detect the input language automatically.\n"
    "- If the speaker is speaking English, translate to Brazilian Portuguese (pt-BR).\n"
    "- If the speaker is speaking Portuguese, translate to English.\n"
    "- Translate ONLY what was said. Do not add anything.\n"
    "- Preserve intent, tone, uncertainty, hedging, and social meaning.\n"
    "- Preserve fragments, false starts, and interruptions.\n"
    "- Do not correct grammar or improve wording.\n"
    "- Do not add explanations, greetings, or extra politeness.\n"
    "- Do not expand, summarize, or paraphrase.\n"
    "- Do not answer questions — translate them.\n"
    "- Do not add commentary or context.\n"
    "- Match the length and directness of the source.\n"
    "- Return ONLY the translated text, suitable for immediate live delivery.\n"
    "- If you cannot determine the language, output the text unchanged.\n"
    "- Remain silent unless translating. Never speak on your own.\n"
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Flow Interpreter")

# Serve the browser client
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "flow-interpreter"}


# ---------------------------------------------------------------------------
# WebSocket proxy: browser ↔ this server ↔ OpenAI Realtime API
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_proxy(client_ws: WebSocket):
    """
    Bridge between the browser client and OpenAI's Realtime API.

    Flow:
    1. Browser connects via WS to /ws
    2. We open a WS to OpenAI Realtime API
    3. We send session.update with the interpreter prompt
    4. We relay audio from browser → OpenAI
    5. We relay text/audio responses from OpenAI → browser
    """
    await client_ws.accept()
    log("[flow] Client connected")

    openai_ws = None

    try:
        # Connect to OpenAI Realtime API
        openai_ws = await websockets.connect(
            OPENAI_REALTIME_URL,
            additional_headers=OPENAI_HEADERS,
            max_size=2**24,  # 16MB max message
        )
        log("[flow] Connected to OpenAI Realtime API")

        # Wait for session.created from OpenAI
        init_msg = await asyncio.wait_for(openai_ws.recv(), timeout=10)
        init_data = json.loads(init_msg)
        log(f"[flow] OpenAI session: {init_data.get('type', 'unknown')}")

        # Configure the session for interpretation
        session_config = {
            "type": "session.update",
            "session": {
                "instructions": INTERPRETER_PROMPT,
                "modalities": ["text", "audio"],
                "voice": "shimmer",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gpt-4o-mini-transcribe",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700,
                },
            },
        }
        await openai_ws.send(json.dumps(session_config))
        log("[flow] Session configured for interpretation")

        # Wait for session.updated confirmation
        update_resp = await asyncio.wait_for(openai_ws.recv(), timeout=10)
        update_data = json.loads(update_resp)
        print(f"[flow] Session update: {update_data.get('type', 'unknown')}")

        # Notify browser that we're ready
        await client_ws.send_json({
            "type": "flow.ready",
            "message": "Interpreter active. Speak naturally.",
        })

        # ----- Two concurrent tasks: relay in both directions -----

        audio_chunks_relayed = 0

        async def relay_client_to_openai():
            """Forward audio from browser to OpenAI."""
            nonlocal audio_chunks_relayed
            try:
                while True:
                    data = await client_ws.receive_text()
                    msg = json.loads(data)

                    if msg.get("type") == "audio":
                        # Browser sends base64 PCM16 audio chunks
                        audio_b64 = msg["audio"]
                        audio_event = {
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,
                        }
                        await openai_ws.send(json.dumps(audio_event))
                        audio_chunks_relayed += 1

                        if audio_chunks_relayed == 1:
                            # Log first chunk details for debugging
                            raw_bytes = base64.b64decode(audio_b64)
                            print(f"[flow] First audio chunk relayed: {len(raw_bytes)} bytes ({len(raw_bytes)//2} PCM16 samples)")

                        if audio_chunks_relayed % 50 == 0:
                            print(f"[flow] Audio chunks relayed: {audio_chunks_relayed}")

                    elif msg.get("type") == "language_override":
                        # Allow the client to switch target language direction
                        pass  # Reserved for future use

            except WebSocketDisconnect:
                log("[flow] Client disconnected")
            except Exception as e:
                print(f"[flow] Client relay error: {e}")

        async def relay_openai_to_client():
            """Forward translation results from OpenAI to browser."""
            try:
                async for raw_msg in openai_ws:
                    event = json.loads(raw_msg)
                    event_type = event.get("type", "")

                    # Log all events for debugging
                    if event_type == "response.audio.delta":
                        # Count audio deltas instead of logging each one
                        if not hasattr(relay_openai_to_client, '_audio_delta_count'):
                            relay_openai_to_client._audio_delta_count = 0
                        relay_openai_to_client._audio_delta_count += 1
                        if relay_openai_to_client._audio_delta_count % 20 == 1:
                            log(f"[flow] OpenAI → response.audio.delta (#{relay_openai_to_client._audio_delta_count})")
                    else:
                        detail = ""
                        if "transcript" in event:
                            detail = f" transcript={event['transcript'][:120]}"
                        elif "delta" in event:
                            detail = f" delta={event['delta'][:120]}"
                        elif "error" in event:
                            detail = f" error={event['error']}"
                        log(f"[flow] OpenAI → {event_type}{detail}")

                    # --- Input transcription (what the speaker said) ---
                    if event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "").strip()
                        if transcript:
                            await client_ws.send_json({
                                "type": "source_transcript",
                                "text": transcript,
                            })
                            print(f"[flow] Source: {transcript}")

                    # --- Translation text streaming (partial) ---
                    elif event_type == "response.audio_transcript.delta":
                        delta = event.get("delta", "")
                        if delta:
                            await client_ws.send_json({
                                "type": "translation_delta",
                                "text": delta,
                            })

                    # --- Translation text complete ---
                    elif event_type == "response.audio_transcript.done":
                        transcript = event.get("transcript", "").strip()
                        if transcript:
                            await client_ws.send_json({
                                "type": "translation_done",
                                "text": transcript,
                            })
                            print(f"[flow] Translation: {transcript}")

                    # --- Translation audio streaming ---
                    elif event_type == "response.audio.delta":
                        audio_b64 = event.get("delta", "")
                        if audio_b64:
                            await client_ws.send_json({
                                "type": "audio_delta",
                                "audio": audio_b64,
                            })

                    # --- Response complete ---
                    elif event_type == "response.done":
                        await client_ws.send_json({
                            "type": "turn_complete",
                        })

                    # --- Text-only response delta (fallback for text modality) ---
                    elif event_type == "response.text.delta":
                        delta = event.get("delta", "")
                        if delta:
                            await client_ws.send_json({
                                "type": "translation_delta",
                                "text": delta,
                            })

                    elif event_type == "response.text.done":
                        text = event.get("text", "").strip()
                        if text:
                            await client_ws.send_json({
                                "type": "translation_done",
                                "text": text,
                            })
                            print(f"[flow] Translation: {text}")

                    # --- Error handling ---
                    elif event_type == "error":
                        error_info = event.get("error", {})
                        print(f"[flow] OpenAI error: {error_info}")
                        await client_ws.send_json({
                            "type": "error",
                            "message": error_info.get("message", "Unknown error"),
                        })

                    # --- Speech started (user is speaking) ---
                    elif event_type == "input_audio_buffer.speech_started":
                        await client_ws.send_json({
                            "type": "speech_started",
                        })

                    # --- Speech stopped (VAD detected end of speech) ---
                    elif event_type == "input_audio_buffer.speech_stopped":
                        await client_ws.send_json({
                            "type": "speech_stopped",
                        })

            except websockets.exceptions.ConnectionClosed:
                log("[flow] OpenAI connection closed")
            except Exception as e:
                print(f"[flow] OpenAI relay error: {e}")

        # Run both relays concurrently
        await asyncio.gather(
            relay_client_to_openai(),
            relay_openai_to_client(),
        )

    except Exception as e:
        log(f"[flow] Connection error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            await client_ws.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass

    finally:
        if openai_ws and not openai_ws.closed:
            await openai_ws.close()
        log("[flow] Session ended")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   FLOW — Live Bilingual Interpreter  ║")
    print("  ║   English ↔ Brazilian Portuguese      ║")
    print("  ║   http://localhost:8765               ║")
    print("  ╚══════════════════════════════════════╝\n")
    uvicorn.run(app, host="0.0.0.0", port=8765)
