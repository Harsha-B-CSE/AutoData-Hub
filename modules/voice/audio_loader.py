"""In-memory audio decoding, downmixing, resampling, and PCM helpers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import io
from pathlib import Path
from typing import Any
import wave

import numpy as np
from scipy.signal import resample_poly


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_AUDIO_FILE_BYTES = 512 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 30 * 60
MAX_AUDIO_CHANNELS = 32
MAX_AUDIO_SAMPLE_RATE = 384_000
MAX_DECODED_FLOAT_VALUES = 20_000_000


@dataclass(frozen=True)
class LoadedAudio:
    waveform: np.ndarray
    sample_rate: int
    source_sample_rate: int
    source_channels: int
    decoder: str
    filename: str


def _sanitize_waveform(audio_signal: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_signal, dtype=np.float32)
    return np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)


def downmix_to_mono(
    audio_signal: np.ndarray,
    *,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Average channels according to y[n] = (1/C) * sum(y_c[n])."""

    audio = _sanitize_waveform(audio_signal)
    if audio.ndim == 0:
        return audio.reshape(1)
    if audio.ndim == 1:
        return np.ascontiguousarray(audio)
    if audio.ndim != 2:
        raise ValueError("Audio must be a one- or two-dimensional array.")

    if channel_axis is None:
        if audio.shape[0] <= MAX_AUDIO_CHANNELS and audio.shape[1] > audio.shape[0]:
            channel_axis = 0
        elif audio.shape[1] <= MAX_AUDIO_CHANNELS:
            channel_axis = 1
        else:
            raise ValueError("Could not infer the channel axis for the audio array.")
    if channel_axis not in (0, 1):
        raise ValueError("channel_axis must be 0 or 1.")
    if audio.shape[channel_axis] > MAX_AUDIO_CHANNELS:
        raise ValueError(f"Audio exceeds the {MAX_AUDIO_CHANNELS}-channel limit.")
    return np.ascontiguousarray(np.mean(audio, axis=channel_axis, dtype=np.float64), dtype=np.float32)


def resample_audio(
    audio_signal: np.ndarray,
    source_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample mono audio with TorchAudio's polyphase path or SciPy fallback."""

    source_sr = int(source_sr)
    target_sr = int(target_sr)
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive integers.")
    audio = downmix_to_mono(audio_signal)
    if source_sr == target_sr or audio.size == 0:
        return audio.copy()

    expected_samples = max(1, int(round(audio.size * target_sr / source_sr)))
    try:
        import torch  # type: ignore
        import torchaudio  # type: ignore

        tensor = torch.from_numpy(audio).unsqueeze(0)
        transform = torchaudio.transforms.Resample(
            orig_freq=source_sr,
            new_freq=target_sr,
            dtype=tensor.dtype,
        )
        result = transform(tensor).squeeze(0).detach().cpu().numpy()
    except Exception:
        ratio = Fraction(target_sr, source_sr)
        result = resample_poly(audio, ratio.numerator, ratio.denominator)

    result = np.asarray(result, dtype=np.float32).reshape(-1)
    if result.size > expected_samples:
        result = result[:expected_samples]
    elif result.size < expected_samples:
        result = np.pad(result, (0, expected_samples - result.size))
    return np.ascontiguousarray(result)


def _uploaded_name(source: Any) -> str:
    if isinstance(source, dict):
        return str(source.get("filename") or source.get("name") or "uploaded.wav")
    if isinstance(source, (str, Path)):
        return Path(source).name
    return str(getattr(source, "name", "uploaded.wav"))


def _uploaded_bytes(source: Any) -> bytes:
    if isinstance(source, dict):
        payload = source.get("bytes", b"")
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > MAX_AUDIO_FILE_BYTES:
            raise ValueError("Audio file exceeds the 512 MiB safety limit.")
        payload = path.read_bytes()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
    elif hasattr(source, "getvalue"):
        payload = source.getvalue()
    elif hasattr(source, "read"):
        payload = source.read()
    else:
        raise TypeError("Unsupported audio source object.")
    payload = bytes(payload)
    if not payload:
        raise ValueError("Audio payload is empty.")
    if len(payload) > MAX_AUDIO_FILE_BYTES:
        raise ValueError("Audio file exceeds the 512 MiB safety limit.")
    return payload


def _decode_pcm_samples(payload: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        values = np.frombuffer(payload, dtype=np.uint8)
        if values.size % 3:
            raise ValueError("24-bit PCM payload has an incomplete sample.")
        triples = values.reshape(-1, 3).astype(np.int32)
        signed = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        return signed.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes.")


def _decode_wav(payload: bytes) -> tuple[np.ndarray, int, int]:
    with wave.open(io.BytesIO(payload), "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        sample_rate = int(wav_file.getframerate())
        sample_width = int(wav_file.getsampwidth())
        frame_count = int(wav_file.getnframes())
        if channels <= 0 or channels > MAX_AUDIO_CHANNELS:
            raise ValueError("WAV channel count exceeds the safety limit.")
        if sample_rate <= 0 or sample_rate > MAX_AUDIO_SAMPLE_RATE:
            raise ValueError("WAV sample rate exceeds the safety limit.")
        if frame_count * channels > MAX_DECODED_FLOAT_VALUES:
            raise ValueError("Decoded audio exceeds the in-memory sample limit.")
        if frame_count / sample_rate > MAX_AUDIO_DURATION_SECONDS:
            raise ValueError("Decoded audio exceeds the 30-minute duration limit.")
        if wav_file.getcomptype() != "NONE":
            raise ValueError("Compressed WAV data requires an external decoder.")
        raw = wav_file.readframes(frame_count)
    samples = _decode_pcm_samples(raw, sample_width)
    if samples.size % channels:
        raise ValueError("WAV payload contains incomplete channel frames.")
    return samples.reshape(-1, channels), sample_rate, channels


def _decode_soundfile(payload: bytes) -> tuple[np.ndarray, int, int]:
    import soundfile as sf  # type: ignore

    with sf.SoundFile(io.BytesIO(payload)) as audio_file:
        sample_rate = int(audio_file.samplerate)
        channels = int(audio_file.channels)
        frames = int(len(audio_file))
        if channels <= 0 or channels > MAX_AUDIO_CHANNELS:
            raise ValueError("Audio channel count exceeds the safety limit.")
        if sample_rate <= 0 or sample_rate > MAX_AUDIO_SAMPLE_RATE:
            raise ValueError("Audio sample rate exceeds the safety limit.")
        if frames * channels > MAX_DECODED_FLOAT_VALUES:
            raise ValueError("Decoded audio exceeds the in-memory sample limit.")
        if frames / sample_rate > MAX_AUDIO_DURATION_SECONDS:
            raise ValueError("Decoded audio exceeds the 30-minute duration limit.")
        audio = audio_file.read(dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), sample_rate, channels


def _decode_pyav(payload: bytes) -> tuple[np.ndarray, int, int]:
    import av  # type: ignore

    chunks: list[np.ndarray] = []
    sample_rate = 0
    channels = 0
    decoded_values = 0
    with av.open(io.BytesIO(payload), mode="r") as container:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if not streams:
            raise ValueError("The media container has no audio stream.")
        stream = streams[0]
        resampler = av.AudioResampler(format="fltp")

        def append_frame(converted: Any) -> None:
            nonlocal channels, decoded_values, sample_rate
            array = np.asarray(converted.to_ndarray(), dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            frame_channels = int(array.shape[0])
            if frame_channels <= 0 or frame_channels > MAX_AUDIO_CHANNELS:
                raise ValueError("Audio channel count exceeds the safety limit.")
            frame_rate = int(converted.sample_rate or stream.rate or 0)
            if frame_rate <= 0 or frame_rate > MAX_AUDIO_SAMPLE_RATE:
                raise ValueError("Audio sample rate exceeds the safety limit.")
            decoded_values += int(array.size)
            if decoded_values > MAX_DECODED_FLOAT_VALUES:
                raise ValueError("Decoded audio exceeds the in-memory sample limit.")
            if decoded_values / (frame_channels * frame_rate) > MAX_AUDIO_DURATION_SECONDS:
                raise ValueError("Decoded audio exceeds the 30-minute duration limit.")
            channels = frame_channels
            sample_rate = frame_rate
            chunks.append(array)

        for frame in container.decode(stream):
            for converted in resampler.resample(frame) or []:
                append_frame(converted)
        for converted in resampler.resample(None) or []:
            append_frame(converted)
    if not chunks:
        raise ValueError("The media container contains no decodable audio frames.")
    return np.concatenate(chunks, axis=1).T, sample_rate, channels


def load_audio_with_metadata(
    source: Any,
    target_sr: int = 16_000,
    *,
    sample_rate: int | None = None,
) -> LoadedAudio:
    """Decode a supported static source, downmix it, and resample it."""

    filename = _uploaded_name(source)
    if isinstance(source, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required for NumPy audio input.")
        source_audio = np.asarray(source)
        source_channels = 1 if source_audio.ndim == 1 else min(source_audio.shape)
        mono = downmix_to_mono(source_audio)
        decoded_sr = int(sample_rate)
        decoder = "numpy"
    else:
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(
                "Unsupported audio format. Expected WAV, MP3, FLAC, OGG, or M4A."
            )
        payload = _uploaded_bytes(source)
        decoders = []
        if extension == ".wav":
            decoders.append(("wave", _decode_wav))
        decoders.extend(
            [
                ("soundfile", _decode_soundfile),
                ("pyav", _decode_pyav),
            ]
        )
        errors = []
        decoded = None
        decoder = ""
        for decoder_name, decode in decoders:
            try:
                decoded = decode(payload)
                decoder = decoder_name
                break
            except Exception as exc:
                errors.append(f"{decoder_name}: {type(exc).__name__}")
        if decoded is None:
            raise ValueError(
                f"Could not decode {extension} audio ({', '.join(errors)}). "
                "Install the audio dependencies declared in requirements.txt."
            )
        channel_audio, decoded_sr, source_channels = decoded
        mono = downmix_to_mono(channel_audio, channel_axis=1)

    if decoded_sr <= 0 or decoded_sr > MAX_AUDIO_SAMPLE_RATE:
        raise ValueError("Decoded audio has an invalid or unsafe sample rate.")
    if source_channels <= 0 or source_channels > MAX_AUDIO_CHANNELS:
        raise ValueError("Decoded audio has an invalid or unsafe channel count.")
    duration = mono.size / decoded_sr
    if duration > MAX_AUDIO_DURATION_SECONDS:
        raise ValueError("Decoded audio exceeds the 30-minute duration limit.")
    waveform = resample_audio(mono, decoded_sr, int(target_sr))
    return LoadedAudio(
        waveform=np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False),
        sample_rate=int(target_sr),
        source_sample_rate=decoded_sr,
        source_channels=source_channels,
        decoder=decoder,
        filename=filename,
    )


def load_audio(
    source: Any,
    target_sr: int = 16_000,
    *,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Compatibility tuple API returning ``(mono_waveform, sample_rate)``."""

    loaded = load_audio_with_metadata(source, target_sr, sample_rate=sample_rate)
    return loaded.waveform, loaded.sample_rate


def pcm_bytes_to_float32(
    payload: bytes,
    *,
    sample_width: int = 2,
    channels: int = 1,
) -> np.ndarray:
    """Decode little-endian interleaved PCM bytes to a mono float32 chunk."""

    if channels <= 0 or channels > MAX_AUDIO_CHANNELS:
        raise ValueError("channels is outside the supported range.")
    samples = _decode_pcm_samples(bytes(payload), int(sample_width))
    if samples.size % channels:
        raise ValueError("PCM chunk ends with an incomplete channel frame.")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def float32_to_pcm_bytes(audio_signal: np.ndarray, *, bit_depth: int = 16) -> bytes:
    """Encode mono float audio as little-endian integer PCM bytes."""

    audio = np.clip(downmix_to_mono(audio_signal), -1.0, 1.0)
    if bit_depth == 8:
        return np.round((audio + 1.0) * 127.5).astype(np.uint8).tobytes()
    if bit_depth == 16:
        return np.round(audio * 32767.0).astype("<i2").tobytes()
    if bit_depth == 24:
        integers = np.round(audio * 8388607.0).astype(np.int32)
        unsigned = integers & 0xFFFFFF
        packed = np.empty((unsigned.size, 3), dtype=np.uint8)
        packed[:, 0] = unsigned & 0xFF
        packed[:, 1] = (unsigned >> 8) & 0xFF
        packed[:, 2] = (unsigned >> 16) & 0xFF
        return packed.tobytes()
    if bit_depth == 32:
        return np.round(audio * 2147483647.0).astype("<i4").tobytes()
    raise ValueError("PCM bit depth must be 8, 16, 24, or 32.")


def encode_wav_bytes(
    audio_signal: np.ndarray,
    sample_rate: int,
    *,
    bit_depth: int = 16,
) -> bytes:
    """Encode a mono PCM WAV entirely in memory."""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(float32_to_pcm_bytes(audio_signal, bit_depth=bit_depth))
    return buffer.getvalue()


def _encode_pyav_bytes(
    audio_signal: np.ndarray,
    sample_rate: int,
    *,
    container_format: str,
    codecs: tuple[str, ...],
    bit_rate: int | None = None,
) -> bytes:
    try:
        import av  # type: ignore
    except Exception as exc:
        raise RuntimeError("Compressed export requires the PyAV package.") from exc

    audio = np.clip(downmix_to_mono(audio_signal), -1.0, 1.0)
    pcm = np.round(audio * 32767.0).astype(np.int16).reshape(1, -1)
    last_error: Exception | None = None
    for codec in codecs:
        output = io.BytesIO()
        try:
            with av.open(output, mode="w", format=container_format) as container:
                stream = container.add_stream(codec, rate=int(sample_rate))
                stream.layout = "mono"
                if bit_rate is not None:
                    stream.bit_rate = int(bit_rate)
                frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
                frame.sample_rate = int(sample_rate)
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            return output.getvalue()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"The local FFmpeg build cannot encode {container_format.upper()} audio."
    ) from last_error


def encode_mp3_bytes(
    audio_signal: np.ndarray,
    sample_rate: int,
    *,
    bit_rate: int = 320_000,
) -> bytes:
    """Encode mono MP3 through PyAV when its FFmpeg codecs are available."""

    return _encode_pyav_bytes(
        audio_signal,
        sample_rate,
        container_format="mp3",
        codecs=("libmp3lame", "mp3"),
        bit_rate=bit_rate,
    )


def _encode_soundfile_bytes(
    audio_signal: np.ndarray,
    sample_rate: int,
    *,
    format_name: str,
    subtype: str,
) -> bytes:
    import soundfile as sf  # type: ignore

    output = io.BytesIO()
    sf.write(
        output,
        downmix_to_mono(audio_signal),
        int(sample_rate),
        format=format_name,
        subtype=subtype,
    )
    return output.getvalue()


def encode_flac_bytes(audio_signal: np.ndarray, sample_rate: int) -> bytes:
    """Encode lossless 24-bit FLAC in memory with SoundFile or PyAV."""

    try:
        return _encode_soundfile_bytes(
            audio_signal,
            sample_rate,
            format_name="FLAC",
            subtype="PCM_24",
        )
    except Exception:
        return _encode_pyav_bytes(
            audio_signal,
            sample_rate,
            container_format="flac",
            codecs=("flac",),
        )


def encode_ogg_bytes(audio_signal: np.ndarray, sample_rate: int) -> bytes:
    """Encode OGG Vorbis in memory with SoundFile or PyAV."""

    try:
        return _encode_soundfile_bytes(
            audio_signal,
            sample_rate,
            format_name="OGG",
            subtype="VORBIS",
        )
    except Exception:
        return _encode_pyav_bytes(
            audio_signal,
            sample_rate,
            container_format="ogg",
            codecs=("libvorbis", "vorbis"),
        )
