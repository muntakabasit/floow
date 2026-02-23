"""
FLOW LOCAL — Live Bilingual Interpreter (100% Local)
No cloud calls. Whisper + Ollama + Piper TTS on-device.

Streaming pipeline (sentence-level overlap):
  Browser mic → WebSocket → Silero VAD → faster-whisper STT
    → Ollama LLM streams tokens → detect sentence boundary
    → Piper TTS per sentence → audio_delta sent immediately
    (LLM continues generating while TTS plays previous sentence)

Key optimizations:
  - Single-pass STT with stable_lang hint (skips 3× dual-transcription)
  - Streaming LLM → sentence-level TTS (overlaps generation with synthesis)
  - Persistent httpx connection pooling for Ollama
  - Barge-in cancels remaining TTS sentences
  - Energy pre-filter + hallucination guard
  - Echo suppression (post-TTS 600ms silence window + VAD reset)
  - Per-turn METRICS logging (tts_first_audio_ms for P50/P95)
"""

import asyncio
import json
import os
import sys
import io
import wave
import base64
import time
import traceback
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import numpy as np
import httpx
import onnxruntime
from faster_whisper import WhisperModel
from piper import PiperVoice
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# Error codes for structured error handling
class ErrorCode(str, Enum):
    STT_FAILED = "STT_FAILED"
    STT_TIMEOUT = "STT_TIMEOUT"
    LLM_FAILED = "LLM_FAILED"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    TTS_FAILED = "TTS_FAILED"
    AUDIO_ENCODING_ERROR = "AUDIO_ENCODING_ERROR"
    VAD_ERROR = "VAD_ERROR"
    UNKNOWN = "UNKNOWN"

    def user_message(self):
        messages = {
            "STT_FAILED": "Could not understand speech. Please try again.",
            "STT_TIMEOUT": "Speech processing took too long. Try speaking shorter phrases.",
            "LLM_FAILED": "Translation service encountered an error. Retrying...",
            "LLM_TIMEOUT": "Translation is slow. Please wait or try a shorter phrase.",
            "LLM_UNAVAILABLE": "Translator offline. Make sure Ollama is running.",
            "TTS_FAILED": "Could not generate audio response.",
            "AUDIO_ENCODING_ERROR": "Audio encoding failed. Check microphone.",
            "VAD_ERROR": "Speech detection error.",
            "UNKNOWN": "Unknown error occurred.",
        }
        return messages.get(self.value, messages["UNKNOWN"])


# ---------------------------------------------------------------------------
# Logging (unbuffered)
# ---------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FLOW_DIR = Path(__file__).resolve().parent
PIPER_MODELS_DIR = FLOW_DIR / "models" / "piper"

# Whisper
WHISPER_MODEL_SIZE = "small"      # "small" = 244M params, much better accent/diacritic accuracy (already cached)
WHISPER_DEVICE = "cpu"            # ctranslate2 on macOS = cpu
WHISPER_COMPUTE_TYPE = "int8"     # fastest on cpu

# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_TIMEOUT = 30.0             # seconds per translation request (increased for stability)
OLLAMA_RETRIES = 3                # retry count on failure (increased for resilience)

# Audio
INPUT_SAMPLE_RATE = 24000         # from browser
VAD_SAMPLE_RATE = 16000           # silero requires 16kHz
WHISPER_SAMPLE_RATE = 16000       # whisper requires 16kHz

# VAD
VAD_THRESHOLD = 0.5
VAD_NEG_THRESHOLD = 0.35
SILENCE_DURATION_MS = 700         # enough to finish a thought without cutting off mid-sentence
MIN_SPEECH_MS = 350               # allow shorter natural turns without feeling delayed
MAX_SPEECH_S = 30                 # force-stop segment after 30s (prevents runaway)
AUDIO_TIMEOUT_MS = 10000          # 10 second timeout for audio - finalize if no audio received (prevents hanging)
VAD_WINDOW = 512                  # silero window size in samples
VAD_CONTEXT = 64                  # silero context size

# Energy pre-filter: skip segments quieter than this RMS
MIN_ENERGY_RMS = 0.002            # below this = silence/ambient noise (lowered to not clip quiet speech)

# ---------------------------------------------------------------------------
# Language Stability Contract (CRITICAL)
# ---------------------------------------------------------------------------

ALLOWED_LANGS = ["en", "pt"]      # ONLY English and Portuguese (PT includes all pt variants)
LANGUAGE_SWITCH_HYSTERESIS = 1    # trust each turn's detected language (live interpreter = alternating langs)
LANGUAGE_SWITCH_COOLDOWN = 0      # no cooldown — interpreter must switch every turn
MIN_CONFIDENCE_STT = 0.55         # skip translation if STT confidence below threshold
MIN_CONFIDENCE_SWITCH = 0.75      # require 0.75+ confidence to switch languages

# Reliability Modes — Default Configuration
DEFAULT_RELIABILITY_MODE = "stable"   # reliability-first baseline

# Mode-specific parameters (per-session, not hardcoded)
MODE_CONFIG = {
    "stable": {
        "SILENCE_DURATION_MS": 1300,    # 1.3s (longer to ensure complete utterance)
        "KEEPALIVE_INTERVAL": 20,       # seconds
        "KEEPALIVE_TIMEOUT": 90000,     # 90s timeout
    },
    "fast": {
        "SILENCE_DURATION_MS": 650,     # responsive finalization for live conversation
        "KEEPALIVE_INTERVAL": 20,       # seconds
        "KEEPALIVE_TIMEOUT": 45000,     # 45s timeout
    }
}

# Default values for legacy/fallback (SILENCE_DURATION_MS kept at 400ms from line 103)
# No override here - let line 103 value stick for fast default
KEEPALIVE_INTERVAL = MODE_CONFIG[DEFAULT_RELIABILITY_MODE]["KEEPALIVE_INTERVAL"]
KEEPALIVE_TIMEOUT = MODE_CONFIG[DEFAULT_RELIABILITY_MODE]["KEEPALIVE_TIMEOUT"]

# Whisper hallucination guard
HALLUCINATION_PATTERNS = {
    # Common faster-whisper hallucinations on silence/noise
    # English — YouTube-isms
    "thank you", "thanks for watching", "subscribe", "like and subscribe",
    "thanks for listening", "bye", "subtitles by", "thank you for watching",
    "thanks for viewing", "if you like this",
    # English — short phantom phrases from silence/breathing
    "hello", "ah hello", "oh hello", "hey", "hi",
    "how are you", "i'll call you", "i will call you",
    "you", "oh", "yes", "no", "okay", "so", "um", "uh", "hmm", "ah",
    "good", "good night", "good morning", "good afternoon",
    "one", "two", "three",
    "the end", "all right",
    # Portuguese — YouTube-isms
    "obrigado por assistir", "se inscreva", "legendas pela comunidade", "amara.org",
    "obrigado por ver", "obrigado por ouvir", "legendas pela comunidade amara.org",
    # Portuguese — short phantom phrases
    "olá", "oi", "tchau", "sim", "não",
    "obrigado", "obrigada", "por favor",
}
MIN_TRANSCRIPTION_CHARS = 2       # ignore 1-char transcriptions
MIN_SPEECH_SEGMENT_MS = 350       # skip micro-segments/noise before STT

# Thread pool for blocking inference calls (3 workers: STT + TTS can overlap)
EXECUTOR = ThreadPoolExecutor(max_workers=3)

# Persistent httpx client for Ollama (connection pooling, avoids TCP reconnect)
_ollama_client: httpx.AsyncClient | None = None

async def get_ollama_client() -> httpx.AsyncClient:
    global _ollama_client
    if _ollama_client is None or _ollama_client.is_closed:
        _ollama_client = httpx.AsyncClient(
            timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=5.0),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )
    return _ollama_client

# Sentence boundary for streaming TTS — split on .!?; followed by whitespace or end-of-string
SENTENCE_BOUNDARY_RE = re.compile(r'[.!?;]\s+|[.!?;]$')

# ---------------------------------------------------------------------------
# System prompt — interpreter behavior (text mode)
# ---------------------------------------------------------------------------

INTERPRETER_PROMPT = (
    "You are FLOW, a translation robot. You ONLY translate text.\n"
    "You have NO other purpose. You do not chat or explain.\n\n"
    "Your ONLY task:\n"
    "1. Read the [LANGUAGE → LANGUAGE] direction\n"
    "2. Translate ONLY the text after the direction indicator\n"
    "3. Return the translation EXACTLY as specified\n"
    "4. Return ONLY the translation. NOTHING ELSE.\n\n"
    "TRANSLATION RULES:\n"
    "- Preserve tone and intent exactly\n"
    "- Do NOT correct, rephrase, or explain\n"
    "- Do NOT add greetings, politeness, or commentary\n"
    "- Return EXACTLY one translation, no more\n"
)


# ---------------------------------------------------------------------------
# Model loading (at startup)
# ---------------------------------------------------------------------------

log("[flow-local] Loading Whisper model...")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)
log(f"[flow-local] Whisper '{WHISPER_MODEL_SIZE}' loaded")

log("[flow-local] Loading Piper voices...")
piper_voice_en = PiperVoice.load(str(PIPER_MODELS_DIR / "en_US-lessac-high.onnx"))
piper_voice_pt = PiperVoice.load(str(PIPER_MODELS_DIR / "pt_BR-faber-medium.onnx"))
PIPER_RATE_EN = piper_voice_en.config.sample_rate
PIPER_RATE_PT = piper_voice_pt.config.sample_rate
log(f"[flow-local] Piper voices loaded (EN={PIPER_RATE_EN}Hz, PT={PIPER_RATE_PT}Hz)")

log("[flow-local] Loading Silero VAD...")
VAD_ONNX_PATH = str(
    Path(onnxruntime.__file__).parent.parent
    / "faster_whisper" / "assets" / "silero_vad_v6.onnx"
)
# Verify it exists, fall back to finding it
if not Path(VAD_ONNX_PATH).exists():
    from faster_whisper.vad import get_assets_path
    VAD_ONNX_PATH = str(Path(get_assets_path()) / "silero_vad_v6.onnx")
log(f"[flow-local] Silero VAD at: {VAD_ONNX_PATH}")


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def resample(audio, input_rate, output_rate):
    """Linear interpolation resampler."""
    if input_rate == output_rate:
        return audio
    ratio = input_rate / output_rate
    output_len = int(len(audio) / ratio)
    indices = np.arange(output_len) * ratio
    low = np.floor(indices).astype(int)
    high = np.minimum(low + 1, len(audio) - 1)
    frac = (indices - low).astype(np.float32)
    return audio[low] * (1 - frac) + audio[high] * frac


def decode_browser_audio(b64_pcm16):
    """Decode base64 PCM16 24kHz from browser → float32 numpy array."""
    raw = base64.b64decode(b64_pcm16)
    int16 = np.frombuffer(raw, dtype=np.int16)
    return int16.astype(np.float32) / 32768.0


def float32_to_pcm16_b64(audio_float32):
    """Float32 numpy → PCM16 → base64 string."""
    int16 = np.clip(audio_float32 * 32767, -32768, 32767).astype(np.int16)
    raw = int16.tobytes()
    return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Streaming Silero VAD
# ---------------------------------------------------------------------------

class StreamingVAD:
    """
    Processes audio in real-time, detects speech boundaries.
    Calls ONNX Silero VAD directly with persistent LSTM state.
    """

    def __init__(
        self,
        threshold=VAD_THRESHOLD,
        neg_threshold=VAD_NEG_THRESHOLD,
        silence_ms=SILENCE_DURATION_MS,
        min_speech_ms=MIN_SPEECH_MS,
        max_speech_s=MAX_SPEECH_S,
        sample_rate=VAD_SAMPLE_RATE,
    ):
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.silence_samples = int(silence_ms * sample_rate / 1000)
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.max_speech_samples = int(max_speech_s * sample_rate)
        self.sample_rate = sample_rate

        # ONNX session
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            VAD_ONNX_PATH,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

        # Persistent LSTM state
        self.h = np.zeros((1, 1, 128), dtype="float32")
        self.c = np.zeros((1, 1, 128), dtype="float32")
        self.context = np.zeros(VAD_CONTEXT, dtype="float32")

        # State machine
        self.is_speaking = False
        self.silence_counter = 0
        self.speech_buffer = []  # list of float32 arrays
        self.speech_samples_count = 0
        self.pending = np.array([], dtype="float32")  # incomplete window buffer

    def reset_state(self):
        """Reset LSTM state for new session."""
        self.h = np.zeros((1, 1, 128), dtype="float32")
        self.c = np.zeros((1, 1, 128), dtype="float32")
        self.context = np.zeros(VAD_CONTEXT, dtype="float32")

    def reset_full(self):
        """Full reset: LSTM state + speech detection state machine.
        Call after TTS playback to prevent echo-triggered false speech detection."""
        self.reset_state()
        self.is_speaking = False
        self.silence_counter = 0
        self.speech_buffer = []
        self.speech_samples_count = 0
        self.pending = np.array([], dtype="float32")

    def _run_vad_window(self, window_512):
        """Run VAD on a single 512-sample window. Returns speech probability."""
        # Prepend context
        input_frame = np.concatenate([self.context, window_512])
        input_frame = input_frame.reshape(1, -1)  # (1, 576)

        output, self.h, self.c = self.session.run(
            None,
            {"input": input_frame, "h": self.h, "c": self.c},
        )

        # Update context
        self.context = window_512[-VAD_CONTEXT:]

        # output shape is (1,) — single probability value
        return float(output[0])

    def process_chunk(self, audio_16k):
        """
        Process a chunk of 16kHz float32 audio.
        Returns list of events: [("speech_started",), ("speech_stopped", audio_segment)]
        """
        events = []

        # Append to pending buffer
        self.pending = np.concatenate([self.pending, audio_16k])

        # Process all complete 512-sample windows
        while len(self.pending) >= VAD_WINDOW:
            window = self.pending[:VAD_WINDOW]
            self.pending = self.pending[VAD_WINDOW:]

            prob = self._run_vad_window(window)

            if not self.is_speaking:
                # Currently idle — check if speech starts
                if prob >= self.threshold:
                    self.is_speaking = True
                    self.silence_counter = 0
                    self.speech_buffer = [window]
                    self.speech_samples_count = VAD_WINDOW
                    events.append(("speech_started",))
            else:
                # Currently speaking — accumulate audio
                self.speech_buffer.append(window)
                self.speech_samples_count += VAD_WINDOW

                if prob < self.neg_threshold:
                    self.silence_counter += VAD_WINDOW
                else:
                    self.silence_counter = 0

                # Check if silence exceeded threshold OR max duration hit
                force_stop = self.speech_samples_count >= self.max_speech_samples
                if force_stop:
                    log(f"[flow-local] VAD: max duration ({MAX_SPEECH_S}s) hit, forcing segment end")

                if self.silence_counter >= self.silence_samples or force_stop:
                    self.is_speaking = False

                    # Ignore very short sounds (clicks, pops)
                    if self.speech_samples_count >= self.min_speech_samples:
                        # Trim trailing silence from the speech buffer
                        if not force_stop:
                            trim_windows = self.silence_counter // VAD_WINDOW
                            if trim_windows > 0 and trim_windows < len(self.speech_buffer):
                                self.speech_buffer = self.speech_buffer[:-trim_windows]

                        segment = np.concatenate(self.speech_buffer)
                        events.append(("speech_stopped", segment))
                    else:
                        events.append(("speech_stopped_short",))

                    self.speech_buffer = []
                    self.speech_samples_count = 0
                    self.silence_counter = 0

        return events


# ---------------------------------------------------------------------------
# STT: faster-whisper transcription
# ---------------------------------------------------------------------------

def compute_rms(audio):
    """Compute RMS energy of an audio segment."""
    return float(np.sqrt(np.mean(audio ** 2)))


def normalize_lang(raw_lang):
    """
    Normalize detected language to canonical form.

    Returns:
        "en" or "pt-BR" (normalized)
        Converts pt, pt-pt, pt-br variants → pt-BR
        Converts other languages to last stable lang fallback
    """
    if not raw_lang:
        return None

    raw_lower = raw_lang.lower().strip()

    # English
    if raw_lower == "en" or raw_lower.startswith("en-"):
        return "en"

    # Portuguese (all variants normalize to pt-BR)
    if raw_lower.startswith("pt"):
        return "pt-BR"

    # Unsupported language detected (ru, it, sq, etc.)
    return None


def is_gibberish(text):
    """Detect noise/gibberish: repeated chars, no vowels, too short, unbalanced tokens."""
    t = text.strip()
    if len(t) < 3:
        return True

    # Check for excessive repeated characters (aaaaaa, bbbbb = noise)
    for char in set(t.lower()):
        if char.isalpha() and t.lower().count(char) > len(t) * 0.5:  # >50% same char
            return True

    # Check vowel/consonant balance (gibberish has almost no vowels)
    vowels = sum(1 for c in t.lower() if c in 'aeiouáéíóú')
    if len(t) > 5 and vowels < len(t) * 0.2:  # <20% vowels = unnatural
        return True

    return False

def is_hallucination(text):
    """Check if Whisper output is a known hallucination pattern."""
    # First check for gibberish
    if is_gibberish(text):
        return True

    t = text.strip().lower().rstrip(".").rstrip("!")
    if len(t) < MIN_TRANSCRIPTION_CHARS:
        return True
    # Exact match for known patterns
    if t in HALLUCINATION_PATTERNS:
        return True
    # Partial match for obvious cases (subscribe, thank you, bye, etc.)
    hallucination_keywords = ["subscribe", "thank you", "bye", "like and subscribe", "for watching"]
    if any(keyword in t for keyword in hallucination_keywords):
        # But allow if it's clearly part of a longer meaningful phrase
        if len(t) > len(hallucination_keywords[0]) + 5:
            return False  # Likely part of a longer phrase
        return True
    return False


def transcribe_segment(audio_16k, forced_source_language=None, skip_dual=False):
    """
    Transcribe a speech segment using faster-whisper.
    Returns (text, language, confidence).

    Args:
        forced_source_language: Manual user preference only (from UI settings).
            NOT stable_lang — that would force Whisper to transcribe echo as wrong language.
        skip_dual: When True, skip the expensive 3-pass dual-transcription retry.
            Set to True when stable_lang is known (language already established).

    Robust strategy:
    - Use optional source-language hint when provided.
    - If auto-detect returns unsupported/misdetected language, retry with PT and EN forced.
    - Prefer non-hallucinated, longer candidate text.
    """
    # Energy pre-filter: skip near-silent segments
    rms = compute_rms(audio_16k)
    if rms < MIN_ENERGY_RMS:
        log(f"[flow-local] Energy too low (RMS={rms:.5f}), skipping")
        return "", "unknown", 0.0

    def transcribe_once(lang_hint=None):
        # initial_prompt biases Whisper toward correct transcription:
        # - Portuguese: seeds diacritics (ã, ç, é, ô) so decoder favors accented chars
        # - English: seeds diverse accent awareness for African/non-native speakers
        if lang_hint == "pt":
            prompt = "Transcrição de áudio em português brasileiro. Atenção às acentuações: ã, õ, ç, é, ê, ó, ô, á, à, ú, í."
        elif lang_hint == "en":
            prompt = "Transcription of spoken English with diverse accents. Listen carefully to non-native speakers."
        else:
            prompt = None
        segments, info = whisper_model.transcribe(
            audio_16k,
            beam_size=3,
            best_of=1,
            language=lang_hint,
            initial_prompt=prompt,
            vad_filter=False,
            without_timestamps=True,
        )
        raw_lang = info.language if info.language else (lang_hint or "unknown")
        conf = 1.0 - (info.no_speech_prob if hasattr(info, 'no_speech_prob') else 0.1)
        conf = max(0.0, min(1.0, conf))
        txt = " ".join([seg.text for seg in segments]).strip()
        return txt, raw_lang, conf

    # Optional source-language hint from client settings
    whisper_lang = None
    if forced_source_language:
        f = forced_source_language.lower()
        if f.startswith("pt"):
            whisper_lang = "pt"
        elif f.startswith("en"):
            whisper_lang = "en"

    text, raw_lang, confidence = transcribe_once(whisper_lang)

    # Dual-transcription for short segments OR unsupported languages:
    # Short Portuguese words ("Olá", "Obrigado") often get misdetected as other languages.
    # Always try both PT and EN forced transcription and pick the best result.
    duration_s = len(audio_16k) / 16000
    supported = (raw_lang == "en" or str(raw_lang).startswith("pt"))
    needs_retry = (whisper_lang is None) and not skip_dual and (not supported or duration_s < 2.0)

    if needs_retry:
        reason = "unsupported_lang" if not supported else "short_segment"
        log(f"[flow-local] Dual-transcription ({reason}): auto='{raw_lang}' text='{text}' dur={duration_s:.1f}s")
        candidates = []
        # Keep auto-detect result if it's supported and non-hallucinated
        if supported and text and not is_hallucination(text):
            score = (len(text) * 0.02) + confidence
            candidates.append((score, text, raw_lang, confidence))
        # Try forced PT and EN
        for hint, canon in (("pt", "pt"), ("en", "en")):
            t2, l2, c2 = transcribe_once(hint)
            if t2 and not is_hallucination(t2):
                score = (len(t2) * 0.02) + c2
                candidates.append((score, t2, l2 or canon, c2))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, text, raw_lang, confidence = candidates[0]
            log(f"[flow-local] Dual-transcription picked: lang={raw_lang} conf={confidence:.2f} text='{text}'")
        elif not supported:
            return "", raw_lang, confidence

    # If forced source language is set, trust it as final language family
    if whisper_lang == "pt":
        lang = "pt-BR"
    elif whisper_lang == "en":
        lang = "en"
    elif raw_lang == "en" or str(raw_lang).startswith("pt"):
        lang = raw_lang
    else:
        return "", raw_lang, confidence

    # Hallucination guard
    if is_hallucination(text):
        log(f"[flow-local] Hallucination filtered: '{text}'")
        return "", lang, confidence

    return text, lang, confidence


# ---------------------------------------------------------------------------
# Translation: Ollama streaming
# ---------------------------------------------------------------------------

def _norm_lang(lang):
    if not lang:
        return None
    l = str(lang).lower()
    if l.startswith("pt"):
        return "pt-BR"
    if l.startswith("en"):
        return "en"
    return l


def _choose_translation_direction(source_language, forced_target_language=None):
    """Return (source_norm, target_lang, direction_hint, no_op)."""
    source_norm = _norm_lang(source_language) or "en"
    forced_norm = _norm_lang(forced_target_language)

    # Pair mode default: auto opposite target
    auto_target = "en" if source_norm.startswith("pt") else "pt-BR"
    chosen_target = forced_norm or auto_target

    # Guard: never translate source->same source when target is forced same
    if chosen_target == source_norm:
        return source_norm, chosen_target, None, True

    if source_norm.startswith("pt") and chosen_target == "en":
        return source_norm, chosen_target, "[Portuguese → English] ", False

    if source_norm == "en" and chosen_target.startswith("pt"):
        return source_norm, chosen_target, "[English → Brazilian Portuguese] ", False

    # Fallback to auto opposite target if forced target contradicts source
    chosen_target = auto_target
    if source_norm.startswith("pt"):
        return source_norm, chosen_target, "[Portuguese → English] ", False
    return source_norm, chosen_target, "[English → Brazilian Portuguese] ", False


async def translate_text(text, source_language, client_ws, forced_target_language=None):
    """
    Translate text via Ollama (streaming) with retry logic.
    Sends translation_delta events as tokens arrive.
    Returns (full_translation, target_language).
    """
    source_norm, target_lang, direction_hint, no_op = _choose_translation_direction(source_language, forced_target_language)

    # Diagnostics
    log(f"[flow-local] Direction choose: source={source_norm} forced_target={_norm_lang(forced_target_language)} chosen_target={target_lang} hint={direction_hint} no_op={no_op}")

    # No-op when target == source (forced same-language): bypass translation
    if no_op:
        await client_ws.send_json({"type": "translation_done", "text": text})
        return text, target_lang

    # DEBUG: Log what we're translating
    log(f"[flow-local] Translating: source_lang={source_norm}, target={target_lang}, text='{text}'")

    full_text = ""
    last_error = None

    for attempt in range(1, OLLAMA_RETRIES + 1):
        full_text = ""
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": INTERPRETER_PROMPT},
                            {"role": "user", "content": direction_hint + text},
                        ],
                        "stream": True,
                        "options": {
                            "temperature": 0.1,      # very low creativity for consistent translations
                            "num_predict": 200,       # cap output length
                        },
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                            delta = chunk.get("message", {}).get("content", "")
                            if delta:
                                full_text += delta
                                # Voice-first: collect text silently, don't stream deltas
                                # Text will be sent AFTER TTS starts playing
                        except json.JSONDecodeError:
                            continue

            # DEBUG: Log raw translation before stripping
            log(f"[flow-local] LLM raw response: '{full_text}' (stripped: '{full_text.strip()}')")

            # Voice-first: DON'T send translation_done here
            # Caller will send it after TTS starts, so user hears before reading

            # Success — break out of retry loop
            return full_text.strip(), target_lang

        except httpx.ConnectError:
            last_error = ErrorCode.LLM_UNAVAILABLE
            log(f"[flow-local] Ollama connect failed (attempt {attempt}/{OLLAMA_RETRIES})")
        except (httpx.ReadTimeout, httpx.WriteTimeout, asyncio.TimeoutError):
            last_error = ErrorCode.LLM_TIMEOUT
            log(f"[flow-local] Ollama timeout (attempt {attempt}/{OLLAMA_RETRIES})")
        except Exception as e:
            last_error = ErrorCode.LLM_FAILED
            log(f"[flow-local] Translation error (attempt {attempt}/{OLLAMA_RETRIES}): {e}")

        # Brief pause before retry
        if attempt < OLLAMA_RETRIES:
            await asyncio.sleep(0.5)

    # All retries exhausted
    log(f"[flow-local] Translation failed after {OLLAMA_RETRIES} attempts: {last_error}")
    await client_ws.send_json({
        "type": "error",
        "error_code": last_error.value,
        "message": last_error.user_message(),
    })

    return full_text.strip(), target_lang


# ---------------------------------------------------------------------------
# TTS: Piper synthesis → PCM16 24kHz → base64 chunks
# ---------------------------------------------------------------------------

def synthesize_audio(text, target_lang):
    """
    Synthesize text to PCM16 24kHz float32 array.
    Runs in thread pool (blocking).
    Returns float32 numpy array at 24kHz, or None on failure.
    """
    try:
        voice = piper_voice_pt if target_lang == "pt-BR" else piper_voice_en
        native_rate = PIPER_RATE_PT if target_lang == "pt-BR" else PIPER_RATE_EN

        # Synthesize — returns AudioChunk objects (one per sentence)
        chunks = []
        for audio_chunk in voice.synthesize(text):
            chunks.append(audio_chunk.audio_float_array)

        if not chunks:
            return None

        # Concatenate all sentence audio
        audio = np.concatenate(chunks)

        # Resample to 24kHz if needed
        if native_rate != INPUT_SAMPLE_RATE:
            audio = resample(audio, native_rate, INPUT_SAMPLE_RATE)

        return audio

    except Exception as e:
        log(f"[flow-local] TTS error: {e}")
        traceback.print_exc()
        return None


async def synthesize_and_send(text, target_lang, client_ws):
    """Synthesize and stream audio chunks to browser."""
    loop = asyncio.get_event_loop()

    try:
        # Run TTS in thread pool to avoid blocking
        audio = await loop.run_in_executor(
            EXECUTOR,
            synthesize_audio,
            text,
            target_lang,
        )

        if audio is None or len(audio) == 0:
            log("[flow-local] ❌ TTS returned empty audio")
            await client_ws.send_json({"type": "error", "code": "TTS_FAILED", "message": "Could not generate audio response"})
            return

        # Chunk into ~2048 samples (~85ms at 24kHz) and send - reduced for lower perceived latency
        chunk_size = 2048
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            b64 = float32_to_pcm16_b64(chunk)
            await client_ws.send_json({
                "type": "audio_delta",
                "audio": b64,
            })
    except Exception as e:
        log(f"[flow-local] ❌ TTS EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        log(f"[flow-local] TTS Traceback:\n{traceback.format_exc()}")
        await client_ws.send_json({"type": "error", "code": "TTS_FAILED", "message": "Could not generate audio response"})


# ---------------------------------------------------------------------------
# Streaming pipeline: LLM translation → sentence-level TTS → audio chunks
# ---------------------------------------------------------------------------

async def translate_and_stream_tts(
    text: str,
    source_language: str,
    target_lang_override: str | None,
    client_ws: WebSocket,
    loop: asyncio.AbstractEventLoop,
    turn_id: int,
    barge_in_event: asyncio.Event,
) -> tuple[str, str, float, float, float]:
    """
    Streaming pipeline: LLM translation → sentence-level TTS → audio chunks.

    As the LLM generates tokens, we detect sentence boundaries and immediately
    synthesize + send each sentence. This overlaps LLM generation with TTS.

    Returns: (full_translation, target_lang, llm_ms, tts_first_audio_ms, tts_total_ms)
    """
    source_norm, target_lang, direction_hint, no_op = _choose_translation_direction(
        source_language, target_lang_override
    )

    log(f"[flow-local] Streaming pipeline: source={source_norm} target={target_lang} hint={direction_hint} no_op={no_op}")

    if no_op:
        await client_ws.send_json({"type": "translation_done", "text": text})
        return text, target_lang, 0.0, 0.0, 0.0

    log(f"[flow-local] Translating: '{text}'")

    llm_start = time.monotonic()
    tts_first_audio_time = None
    tts_total_start = None
    full_text = ""
    pending_sentence = ""
    sentence_queue: asyncio.Queue = asyncio.Queue()
    tts_started = False
    sentences_sent = 0

    # Background task: consume sentence_queue, synthesize, send audio
    async def tts_consumer():
        nonlocal tts_first_audio_time, tts_total_start, tts_started, sentences_sent
        while True:
            item = await sentence_queue.get()
            if item is None:  # sentinel: LLM done
                break

            sentence_text, sentence_idx = item
            if not sentence_text.strip():
                continue

            # Check barge-in before each sentence
            if barge_in_event.is_set():
                log(f"[flow-local] Barge-in: skipping remaining TTS sentences")
                # Drain remaining queue
                while not sentence_queue.empty():
                    try:
                        sentence_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break

            if tts_total_start is None:
                tts_total_start = time.monotonic()

            # Synthesize in thread pool
            try:
                audio = await loop.run_in_executor(
                    EXECUTOR, synthesize_audio, sentence_text, target_lang,
                )
            except Exception as e:
                log(f"[flow-local] TTS sentence {sentence_idx} error: {e}")
                continue

            if audio is None or len(audio) == 0:
                log(f"[flow-local] TTS sentence {sentence_idx} empty, skipping")
                continue

            # Send tts_start before first audio
            if not tts_started:
                tts_started = True
                await client_ws.send_json({"type": "tts_start"})
                tts_first_audio_time = time.monotonic()

            # Stream audio chunks (~85ms each at 24kHz)
            chunk_size = 2048
            for i in range(0, len(audio), chunk_size):
                if barge_in_event.is_set():
                    break
                chunk = audio[i : i + chunk_size]
                b64 = float32_to_pcm16_b64(chunk)
                await client_ws.send_json({"type": "audio_delta", "audio": b64})

            sentences_sent += 1

    # Start TTS consumer as background task
    tts_task = asyncio.create_task(tts_consumer())
    sentence_idx = 0

    # Stream LLM translation
    last_error = None
    for attempt in range(1, OLLAMA_RETRIES + 1):
        full_text = ""
        pending_sentence = ""
        try:
            client = await get_ollama_client()
            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": INTERPRETER_PROMPT},
                        {"role": "user", "content": direction_hint + text},
                    ],
                    "stream": True,
                    "options": {"temperature": 0.1, "num_predict": 200},
                },
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        delta = chunk.get("message", {}).get("content", "")
                        if delta:
                            full_text += delta
                            pending_sentence += delta

                            # Check for sentence boundary
                            match = SENTENCE_BOUNDARY_RE.search(pending_sentence)
                            if match:
                                end_pos = match.end()
                                complete = pending_sentence[:end_pos].strip()
                                pending_sentence = pending_sentence[end_pos:]

                                if complete:
                                    # Stream text to client for progressive display
                                    await client_ws.send_json({
                                        "type": "translation_delta",
                                        "text": complete + " ",
                                    })
                                    # Queue for TTS
                                    await sentence_queue.put((complete, sentence_idx))
                                    sentence_idx += 1
                    except json.JSONDecodeError:
                        continue

            # LLM done — flush remaining text
            if pending_sentence.strip():
                await client_ws.send_json({
                    "type": "translation_delta",
                    "text": pending_sentence.strip(),
                })
                await sentence_queue.put((pending_sentence.strip(), sentence_idx))
                sentence_idx += 1

            # Signal TTS consumer to stop
            await sentence_queue.put(None)
            llm_ms = (time.monotonic() - llm_start) * 1000

            # Wait for TTS to finish
            await tts_task

            tts_first_ms = ((tts_first_audio_time - llm_start) * 1000) if tts_first_audio_time else 0
            tts_total_ms = ((time.monotonic() - tts_total_start) * 1000) if tts_total_start else 0

            log(f"[flow-local] LLM raw: '{full_text.strip()}'")
            log(f"[flow-local] Streaming done: LLM={llm_ms:.0f}ms first_audio={tts_first_ms:.0f}ms tts_total={tts_total_ms:.0f}ms sentences={sentences_sent}")

            return full_text.strip(), target_lang, llm_ms, tts_first_ms, tts_total_ms

        except httpx.ConnectError:
            last_error = ErrorCode.LLM_UNAVAILABLE
            log(f"[flow-local] Ollama connect failed (attempt {attempt}/{OLLAMA_RETRIES})")
        except (httpx.ReadTimeout, httpx.WriteTimeout, asyncio.TimeoutError):
            last_error = ErrorCode.LLM_TIMEOUT
            log(f"[flow-local] Ollama timeout (attempt {attempt}/{OLLAMA_RETRIES})")
        except Exception as e:
            last_error = ErrorCode.LLM_FAILED
            log(f"[flow-local] Translation error (attempt {attempt}/{OLLAMA_RETRIES}): {e}")

        if attempt < OLLAMA_RETRIES:
            await asyncio.sleep(0.5)

    # All retries failed — clean up TTS task
    await sentence_queue.put(None)
    await tts_task

    await client_ws.send_json({
        "type": "error",
        "error_code": last_error.value,
        "message": last_error.user_message(),
    })
    return full_text.strip(), target_lang, 0, 0, 0


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Flow Local Interpreter")

STATIC_DIR = FLOW_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "flow-local-interpreter",
        "whisper": WHISPER_MODEL_SIZE,
        "ollama_model": OLLAMA_MODEL,
        "tts": "piper",
    }


# ---------------------------------------------------------------------------
# WebSocket handler — the main pipeline
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_handler(client_ws: WebSocket):
    await client_ws.accept()
    log("[flow-local] Client connected")

    # Per-session mode configuration (FIRST — before VAD creation)
    session_reliability_mode = DEFAULT_RELIABILITY_MODE  # will be overridden by client
    session_config = MODE_CONFIG[session_reliability_mode]
    session_silence_duration_ms = session_config["SILENCE_DURATION_MS"]
    session_keepalive_interval = session_config["KEEPALIVE_INTERVAL"]
    session_keepalive_timeout = session_config["KEEPALIVE_TIMEOUT"]

    # Per-session state (VAD created AFTER session config is set)
    vad = StreamingVAD(silence_ms=session_silence_duration_ms)  # Pass session-specific silence duration
    chunks_received = 0
    loop = asyncio.get_event_loop()
    is_playing_tts = False          # echo suppression flag
    tts_done_time = 0.0             # monotonic timestamp when TTS playback ended
    POST_TTS_SILENCE_MS = 1200      # discard mic audio for 1200ms after TTS ends (room echo can persist 600-1000ms)
    turn_count = 0                  # for latency logging
    keepalive_task = None           # background keepalive ping
    last_audio_time = time.monotonic()  # Track audio timeout
    barge_in_event = asyncio.Event()    # signal streaming TTS to stop on barge-in

    # Language stability state
    stable_lang = None              # current stable language (None = not yet detected, normalized to "en" or "pt-BR")
    lang_switch_counter = 0         # consecutive detections of a candidate language
    candidate_lang = None           # language we're considering switching to
    turns_since_switch = 0          # cooldown counter after language switch

    # Optional language preferences from client UI
    preferred_source_lang = None    # "en" | "pt-BR" | None(auto)
    preferred_target_lang = None    # "en" | "pt-BR" | None(auto)
    lock_target_lang = False        # when true, force target language; default pair-mode auto

    # Send ready (wait for client to send mode preference)
    await client_ws.send_json({
        "type": "flow.ready",
        "message": "Local interpreter active. Speak naturally.",
        "reliability_mode": session_reliability_mode,
        "keepalive_timeout_ms": session_keepalive_timeout,
    })

    # Keepalive: ping at interval to detect stale connections
    async def keepalive():
        try:
            ping_count = 0
            while True:
                # Use session-specific interval (20-25s per mode config)
                await asyncio.sleep(session_keepalive_interval)
                if client_ws.application_state.value == "connected":
                    try:
                        ping_count += 1
                        await client_ws.send_json({"type": "ping"})
                        if ping_count % 3 == 0:
                            log(f"[flow-local] Keepalive ping #{ping_count} sent (mode: {session_reliability_mode})")
                    except Exception as e:
                        log(f"[flow-local] Keepalive ping failed: {e}")
                        break
        except asyncio.CancelledError:
            log("[flow-local] Keepalive task cancelled")
            pass

    keepalive_task = asyncio.create_task(keepalive())

    try:
        while True:
            data = await client_ws.receive_text()
            msg = json.loads(data)

            msg_type = msg.get("type")

            # Mode preference: client requests reliability mode
            if msg_type == "mode_preference":
                requested_mode = msg.get("mode", DEFAULT_RELIABILITY_MODE)
                if requested_mode in MODE_CONFIG:
                    session_reliability_mode = requested_mode
                    session_config = MODE_CONFIG[session_reliability_mode]
                    session_silence_duration_ms = session_config["SILENCE_DURATION_MS"]
                    session_keepalive_interval = session_config["KEEPALIVE_INTERVAL"]
                    session_keepalive_timeout = session_config["KEEPALIVE_TIMEOUT"]

                    # CRITICAL: Recreate VAD with new silence duration
                    vad = StreamingVAD(silence_ms=session_silence_duration_ms)

                    log(f"[flow-local] Mode switched to {session_reliability_mode} (silence: {session_silence_duration_ms}ms, keepalive: {session_keepalive_timeout}ms, VAD recreated)")
                    # Echo back confirmation
                    await client_ws.send_json({
                        "type": "mode_confirmed",
                        "reliability_mode": session_reliability_mode,
                        "keepalive_timeout_ms": session_keepalive_timeout,
                    })
                continue

            # Language preference from client settings (optional override)
            if msg_type == "language_config":
                src = (msg.get("source_language") or "").strip().lower()
                tgt = (msg.get("target_language") or "").strip().lower()

                preferred_source_lang = _norm_lang(src)
                preferred_target_lang = _norm_lang(tgt)

                # Pair mode default: target auto-switches opposite of detected source.
                # Only force target when explicit lock_target=true is provided.
                lock_target_lang = bool(msg.get("lock_target", False))

                log(
                    f"[flow-local] Language config updated: "
                    f"source={preferred_source_lang or 'auto'} "
                    f"target={preferred_target_lang or 'auto'} "
                    f"lock_target={lock_target_lang}"
                )
                continue

            # Echo suppression: browser tells us when TTS playback ends
            if msg_type == "tts_playback_done":
                is_playing_tts = False
                tts_done_time = time.monotonic()
                barge_in_event.set()  # signal streaming TTS to stop
                vad.reset_full()  # Clear LSTM state that accumulated TTS energy
                log(f"[flow-local] TTS done → VAD reset, {POST_TTS_SILENCE_MS}ms silence window started")
                continue

            # Keepalive messages: respond immediately
            if msg_type == "keepalive_ping":
                try:
                    await client_ws.send_json({"type": "keepalive_pong"})
                except Exception:
                    pass
                continue

            if msg_type == "pong":
                # client responded to our ping, no action needed
                continue

            if msg_type != "audio":
                continue

            # Update audio timeout counter
            last_audio_time = time.monotonic()

            chunks_received += 1
            if chunks_received == 1:
                log("[flow-local] First audio chunk received")

            # Echo suppression: discard mic audio while TTS is playing
            if is_playing_tts:
                continue
            # Post-TTS cooldown: discard mic audio for 600ms after TTS ends
            # Prevents echo loop (mic picks up lingering TTS from speaker)
            elapsed_since_tts = (time.monotonic() - tts_done_time) * 1000
            if elapsed_since_tts < POST_TTS_SILENCE_MS:
                continue

            # Decode browser audio (24kHz float32)
            audio_24k = decode_browser_audio(msg["audio"])

            # Resample to 16kHz for VAD + Whisper
            audio_16k = resample(audio_24k, INPUT_SAMPLE_RATE, VAD_SAMPLE_RATE)

            # Run VAD
            events = vad.process_chunk(audio_16k)

            for event in events:
                if event[0] == "speech_started":
                    speech_start_time = time.monotonic()
                    log("[flow-local] Speech started")
                    await client_ws.send_json({"type": "speech_started"})

                elif event[0] == "speech_stopped":
                    speech_stop_time = time.monotonic()
                    # CRITICAL: VAD firing latency (silence after last word → VAD detection)
                    vad_delta_ms = (speech_stop_time - speech_start_time) * 1000 if 'speech_start_time' in locals() else 0
                    log(f"[flow-local] Speech stopped — VAD fired after {vad_delta_ms:.0f}ms")
                    await client_ws.send_json({"type": "speech_stopped", "vad_delta_ms": vad_delta_ms})
                    turn_start = time.monotonic()

                    speech_audio = event[1]  # float32 16kHz
                    duration_s = len(speech_audio) / VAD_SAMPLE_RATE
                    segment_ms = int(duration_s * 1000)
                    log(f"[flow-local] Speech segment: {duration_s:.1f}s ({len(speech_audio)} samples)")

                    # Guard: skip tiny segments that are usually noise/trailing breaths
                    if segment_ms < MIN_SPEECH_SEGMENT_MS:
                        log(f"[flow-local] Turn skipped: short_segment ({segment_ms}ms < {MIN_SPEECH_SEGMENT_MS}ms)")
                        continue

                    # 1. Transcribe (in thread pool — blocking)
                    #    Only pass manual user preference as Whisper language hint.
                    #    Do NOT pass stable_lang — it would force-transcribe echo as wrong language.
                    #    Instead, pass skip_dual=True when language is established to skip 3-pass retry.
                    stt_start = time.monotonic()
                    try:
                        text, detected_lang, stt_confidence = await loop.run_in_executor(
                            EXECUTOR,
                            transcribe_segment,
                            speech_audio,
                            preferred_source_lang,       # manual preference only (or None)
                            stable_lang is not None,     # skip_dual: skip 3-pass when lang is known
                        )
                        stt_ms = (time.monotonic() - stt_start) * 1000
                        log(f"[flow-local] STT ({stt_ms:.0f}ms): [{detected_lang}] confidence={stt_confidence:.2f} text='{text}'")
                    except Exception as e:
                        stt_ms = (time.monotonic() - stt_start) * 1000
                        log(f"[flow-local] ❌ STT EXCEPTION ({stt_ms:.0f}ms): {type(e).__name__}: {e}")
                        import traceback
                        log(f"[flow-local] STT Traceback:\n{traceback.format_exc()}")
                        await client_ws.send_json({"type": "error", "code": "STT_FAILED", "message": "Could not understand speech"})
                        continue

                    # WEB=iOS CONTRACT: Item 4 - STT guardrails
                    # GUARD: Empty transcript — skip translation
                    if not text.strip():
                        log("[flow-local] Turn skipped: empty_transcript")
                        await client_ws.send_json({"type": "turn_complete", "skip_reason": "empty_transcript"})
                        turns_since_switch += 1
                        continue

                    # GUARD: Low STT confidence — skip translation
                    if stt_confidence < MIN_CONFIDENCE_STT:
                        log(f"[flow-local] Turn skipped: low_confidence ({stt_confidence:.2f} < {MIN_CONFIDENCE_STT})")
                        await client_ws.send_json({"type": "turn_complete", "skip_reason": "low_confidence"})
                        turns_since_switch += 1
                        continue

                    # GUARD: Gibberish detection — skip translation
                    if is_gibberish(text):
                        log(f"[flow-local] Turn skipped: gibberish detected ('{text}')")
                        await client_ws.send_json({"type": "turn_complete", "skip_reason": "gibberish"})
                        turns_since_switch += 1
                        continue

                    # LANGUAGE STABILITY: Apply hysteresis and cooldown
                    # Normalize detected language to canonical form (en or pt-BR)
                    normalized_lang = normalize_lang(detected_lang)

                    switch_reason = None
                    active_lang = stable_lang if stable_lang else normalized_lang

                    # First detection: initialize stable language (normalized)
                    if stable_lang is None:
                        if normalized_lang:  # Only set if it's a supported language
                            stable_lang = normalized_lang
                            active_lang = normalized_lang
                            switch_reason = "initial_detection"
                            log(f"[flow-local] Language initialized: {stable_lang} (detected: {detected_lang})")
                        else:
                            # Unsupported language on first detection
                            log(f"[flow-local] Unsupported language on first detection: {detected_lang}, waiting for supported lang")
                            switch_reason = "unsupported_initial"
                            active_lang = normalized_lang or "pt-BR"  # fallback
                            await client_ws.send_json({"type": "turn_complete"})
                            turns_since_switch += 1
                            continue

                    # Language switch logic with hysteresis (only for supported languages)
                    elif normalized_lang and normalized_lang != stable_lang:
                        # Reset hysteresis if candidate language changes
                        if normalized_lang != candidate_lang:
                            candidate_lang = normalized_lang
                            lang_switch_counter = 1
                            active_lang = normalized_lang  # Use detected language immediately for responsiveness
                            switch_reason = "language_candidate_detected"
                            log(f"[flow-local] Language candidate change: {candidate_lang} (detected as {detected_lang}, count=1)")
                        else:
                            lang_switch_counter += 1
                            log(f"[flow-local] Language candidate {candidate_lang}: count={lang_switch_counter}/{LANGUAGE_SWITCH_HYSTERESIS}")

                            # Check if we have enough consecutive detections to switch
                            if lang_switch_counter >= LANGUAGE_SWITCH_HYSTERESIS:
                                # Check cooldown: allow very high confidence to override
                                if turns_since_switch >= LANGUAGE_SWITCH_COOLDOWN or stt_confidence >= 0.95:
                                    stable_lang = normalized_lang
                                    active_lang = normalized_lang
                                    candidate_lang = None
                                    lang_switch_counter = 0
                                    turns_since_switch = 0
                                    switch_reason = "hysteresis_satisfied"
                                    log(f"[flow-local] Language switched to {stable_lang} (cooldown ok)")
                                else:
                                    # Still in cooldown period
                                    active_lang = stable_lang
                                    switch_reason = f"cooldown_active ({turns_since_switch}/{LANGUAGE_SWITCH_COOLDOWN})"
                                    log(f"[flow-local] Language switch blocked by cooldown: {switch_reason}")
                            else:
                                # Not enough consecutive detections yet
                                # BUT: Use the detected language for translation (for real-time responsiveness)
                                # Only update stable_lang after hysteresis is satisfied
                                active_lang = normalized_lang
                                switch_reason = f"hysteresis_pending ({lang_switch_counter}/{LANGUAGE_SWITCH_HYSTERESIS})"
                    elif normalized_lang and normalized_lang == stable_lang:
                        # Same language detected again — reset hysteresis
                        candidate_lang = None
                        lang_switch_counter = 0
                        active_lang = stable_lang
                        switch_reason = "confirmed_language"
                    else:
                        # Unsupported language detected while stable lang exists
                        active_lang = stable_lang  # Ignore unsupported detection
                        switch_reason = f"unsupported_lang_ignored ({detected_lang})"
                        log(f"[flow-local] Unsupported language detected: {detected_lang}, keeping {stable_lang}")

                    # Optional manual source override from client settings
                    if preferred_source_lang:
                        active_lang = preferred_source_lang
                        switch_reason = f"manual_source_override ({preferred_source_lang})"

                    # Send source transcript with language diagnostics
                    await client_ws.send_json({
                        "type": "source_transcript",
                        "text": text,
                        "diagnostics": {
                            "detected_lang": detected_lang,
                            "stt_confidence": stt_confidence,
                            "stable_lang": stable_lang,
                            "active_lang": active_lang,
                            "switch_reason": switch_reason,
                            "segment_ms": segment_ms,
                        }
                    })

                    # 2+3. Streaming translate + TTS (overlapped)
                    barge_in_event.clear()
                    is_playing_tts = True
                    full_translation, target_lang, llm_ms, tts_first_ms, tts_total_ms = \
                        await translate_and_stream_tts(
                            text, active_lang,
                            preferred_target_lang if lock_target_lang else None,
                            client_ws, loop, turn_count, barge_in_event,
                        )

                    # 4. Send final text (voice-first: text after audio started)
                    if full_translation.strip():
                        await client_ws.send_json({
                            "type": "translation_done",
                            "text": full_translation.strip(),
                        })

                    # Turn complete — log with streaming metrics
                    turn_ms = (time.monotonic() - turn_start) * 1000
                    turn_count += 1
                    turns_since_switch += 1
                    await client_ws.send_json({"type": "turn_complete"})

                    # Per-turn instrumentation
                    metrics = {
                        "turn_id": turn_count,
                        "speech_duration_ms": round(segment_ms),
                        "stt_ms": round(stt_ms),
                        "llm_ms": round(llm_ms),
                        "tts_first_audio_ms": round(tts_first_ms),
                        "tts_total_ms": round(tts_total_ms),
                        "total_ms": round(turn_ms),
                        "source_lang": active_lang,
                        "target_lang": target_lang,
                    }
                    log(f"[flow-local] METRICS: {json.dumps(metrics)}")
                    log(f"[flow-local] Turn #{turn_count} complete — {switch_reason} | total {turn_ms:.0f}ms (STT:{stt_ms:.0f} LLM:{llm_ms:.0f} TTS-first:{tts_first_ms:.0f} TTS-total:{tts_total_ms:.0f}ms)")

                elif event[0] == "speech_stopped_short":
                    log("[flow-local] Short sound ignored")

            if chunks_received % 100 == 0:
                log(f"[flow-local] Audio chunks: {chunks_received}")

            # Audio timeout check: if no audio for 10 seconds, something's wrong
            time_since_audio = (time.monotonic() - last_audio_time) * 1000
            if time_since_audio > AUDIO_TIMEOUT_MS:
                log(f"[flow-local] Audio timeout ({time_since_audio:.0f}ms) - client may have disconnected")
                break

    except WebSocketDisconnect:
        log("[flow-local] Client disconnected")
    except Exception as e:
        log(f"[flow-local] ❌ UNHANDLED EXCEPTION: {type(e).__name__}: {e}")
        log(f"[flow-local] Full traceback:\n{traceback.format_exc()}")
        try:
            await client_ws.send_json({
                "type": "error",
                "code": "UNKNOWN",
                "message": f"Server error: {type(e).__name__}",
            })
        except Exception as close_err:
            log(f"[flow-local] Could not send error message: {close_err}")
    finally:
        if keepalive_task:
            keepalive_task.cancel()
    log("[flow-local] Session ended")


# ---------------------------------------------------------------------------
# Lightweight regression tests (direction logic)
# ---------------------------------------------------------------------------

def _run_direction_logic_tests():
    cases = [
        ("pt-BR", "pt-BR", True, "pt-BR"),  # pt speech + forced pt -> no-op
        ("pt-BR", "en", False, "en"),       # pt speech + forced en -> pt->en
        ("en", "pt-BR", False, "pt-BR"),   # en speech + forced pt -> en->pt
        ("en", "en", True, "en"),          # en speech + forced en -> no-op
    ]
    for src, forced, expect_noop, expect_target in cases:
        s, t, _, no_op = _choose_translation_direction(src, forced)
        assert no_op == expect_noop, f"no_op mismatch for src={src} forced={forced}"
        assert t == expect_target, f"target mismatch for src={src} forced={forced}: got {t}"


# ---------------------------------------------------------------------------
# Ollama warmup
# ---------------------------------------------------------------------------

async def warmup_ollama():
    """Send a warmup request so the model is loaded in memory."""
    log(f"[flow-local] Warming up Ollama ({OLLAMA_MODEL})...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "user", "content": "Translate: hello"},
                    ],
                    "stream": False,
                    "options": {"num_predict": 5},
                },
            )
            data = resp.json()
            result = data.get("message", {}).get("content", "")
            log(f"[flow-local] Ollama warm: '{result}'")
    except Exception as e:
        log(f"[flow-local] Ollama warmup failed: {e}")
        log("[flow-local] Make sure Ollama is running: ollama serve")


@app.on_event("startup")
async def startup():
    _run_direction_logic_tests()
    await warmup_ollama()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    log("")
    log("  ╔══════════════════════════════════════╗")
    log("  ║   FLOW LOCAL — Bilingual Interpreter ║")
    log("  ║   English ↔ Brazilian Portuguese      ║")
    log("  ║   100% LOCAL — No cloud calls         ║")
    log("  ║   http://localhost:8765               ║")
    log("  ╚══════════════════════════════════════╝")
    log("")
    uvicorn.run(app, host="0.0.0.0", port=8765)
