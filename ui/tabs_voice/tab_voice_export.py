"""In-memory enhanced voice export controls."""

from __future__ import annotations

import numpy as np
import streamlit as st

from modules.voice.audio_loader import (
    encode_flac_bytes,
    encode_mp3_bytes,
    encode_ogg_bytes,
    encode_wav_bytes,
    resample_audio,
)
from modules.voice.booster import apply_true_peak_limiter
from ui.tabs_voice.common import safe_export_stem
from ui.utils_ux import render_tab_guidance_header


def render_tab_voice_export(
    audio_signal: np.ndarray,
    sample_rate: int = 16_000,
    source_name: str = "audio.wav",
    *,
    is_enhanced: bool = True,
) -> None:
    """Render target format controls and an in-memory download button."""
    render_tab_guidance_header(
        title="Voice Export Studio",
        objective="Export the current voice track in a lossless format or a smaller lossy file.",
        methodology=(
            "Choose 16, 44.1, or 48 kHz. After resampling, a true-peak limiter leaves "
            "headroom before encoding: -0.5 dBFS for WAV/FLAC or -3.0 dBFS for MP3/OGG."
        ),
        deliverable="A 16/24-bit WAV, FLAC, OGG Vorbis, or MP3 download. MP3 is 160 kbps at 16 kHz and 320 kbps otherwise.",
    )
    if not is_enhanced:
        st.info("No enhanced output exists yet; export currently uses decoded mono audio.")
    format_column, rate_column = st.columns(2)
    with format_column:
        output_format = st.selectbox(
            "Multi-Format Export",
            options=[
                ".WAV (16-bit PCM)",
                ".WAV (24-bit PCM)",
                ".MP3 (320kbps)",
                ".FLAC",
                ".OGG",
            ],
            key="voice_export_format",
        )
    with rate_column:
        target_sample_rate = st.selectbox(
            "Target Sample Rate",
            options=[16_000, 44_100, 48_000],
            format_func=lambda value: f"{value / 1000:g} kHz",
            index=2 if int(sample_rate) == 48_000 else 0,
            key="voice_export_sample_rate",
        )
    export_audio = resample_audio(audio_signal, int(sample_rate), int(target_sample_rate))
    lossy_export = output_format.startswith(".MP3") or output_format == ".OGG"
    export_peak_ceiling = -3.0 if lossy_export else -0.5
    export_audio = apply_true_peak_limiter(
        export_audio,
        int(target_sample_rate),
        target_peak_dbfs=export_peak_ceiling,
        lookahead_ms=1.0,
    )
    stem = safe_export_stem(source_name)
    asset_suffix = "enhanced" if is_enhanced else "decoded"
    try:
        if output_format.startswith(".WAV"):
            bit_depth = 24 if "24-bit" in output_format else 16
            payload = encode_wav_bytes(
                export_audio,
                int(target_sample_rate),
                bit_depth=int(bit_depth),
            )
            mime = "audio/wav"
            filename = f"{stem}_{asset_suffix}.wav"
            specification = f"PCM {bit_depth}-bit"
        elif output_format.startswith(".MP3"):
            effective_bitrate = 320_000 if target_sample_rate >= 32_000 else 160_000
            payload = encode_mp3_bytes(
                export_audio,
                int(target_sample_rate),
                bit_rate=effective_bitrate,
            )
            mime = "audio/mpeg"
            filename = f"{stem}_{asset_suffix}.mp3"
            specification = (
                "MP3 320 kbps"
                if effective_bitrate == 320_000
                else "MP3 160 kbps (16 kHz MPEG-2 limit)"
            )
        elif output_format == ".FLAC":
            payload = encode_flac_bytes(export_audio, int(target_sample_rate))
            mime = "audio/flac"
            filename = f"{stem}_{asset_suffix}.flac"
            specification = "FLAC lossless 24-bit"
        else:
            payload = encode_ogg_bytes(export_audio, int(target_sample_rate))
            mime = "audio/ogg"
            filename = f"{stem}_{asset_suffix}.ogg"
            specification = "OGG Vorbis"
    except RuntimeError as exc:
        st.error(str(exc))
        return

    st.markdown(
        f"**Export asset:** {target_sample_rate:,} Hz · {specification} · "
        f"Pre-encode True-Peak ≤ {export_peak_ceiling:.1f} dBFS · "
        f"{len(payload) / 1024:.1f} KiB"
    )
    st.download_button(
        "Download Enhanced Audio" if is_enhanced else "Download Decoded Audio",
        data=payload,
        file_name=filename,
        mime=mime,
        width="stretch",
        key="voice_download_audio",
    )


render_voice_export_tab = render_tab_voice_export
