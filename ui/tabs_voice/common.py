"""Shared in-memory helpers for Streamlit voice tabs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from modules.voice.audio_loader import encode_wav_bytes
from modules.voice.device_manager import resolve_execution_device


def uploaded_audio_name(uploaded_file: Any) -> str:
    if isinstance(uploaded_file, dict):
        return str(uploaded_file.get("filename") or uploaded_file.get("name") or "audio.wav")
    return str(getattr(uploaded_file, "name", "audio.wav"))


def uploaded_audio_bytes(uploaded_file: Any) -> bytes:
    if isinstance(uploaded_file, dict):
        payload = uploaded_file.get("bytes", b"")
    elif isinstance(uploaded_file, (bytes, bytearray, memoryview)):
        payload = uploaded_file
    elif hasattr(uploaded_file, "getvalue"):
        payload = uploaded_file.getvalue()
    else:
        raise TypeError("Unsupported uploaded audio object.")
    return bytes(payload)


def audio_fingerprint(uploaded_file: Any) -> str:
    payload = uploaded_audio_bytes(uploaded_file)
    name = uploaded_audio_name(uploaded_file)
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8", errors="replace"))
    digest.update(payload)
    return digest.hexdigest()


@st.cache_resource(show_spinner=False)
def cached_device_context() -> dict:
    return resolve_execution_device()


def player_wav(audio_signal: np.ndarray, sample_rate: int) -> bytes:
    return encode_wav_bytes(audio_signal, sample_rate, bit_depth=16)


def safe_export_stem(filename: str) -> str:
    stem = Path(filename).stem
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    return safe.strip("_") or "enhanced_voice"
