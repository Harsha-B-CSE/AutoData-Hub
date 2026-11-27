"""Live PCM regressions. Run `python -m tests.test_live_voice_deadline` for timings."""

from fractions import Fraction
from itertools import product
import subprocess
import sys
from time import perf_counter_ns, sleep
from unittest.mock import patch

import av
import numpy as np
import pytest

from modules.voice import booster
from ui.tabs_voice import tab_live_webrtc as live


def _frame(channels, rate, offset, planar):
    pcm = channels if planar else channels.T.reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(pcm), format="s16p" if planar else "s16", layout="stereo"
    )
    frame.sample_rate = rate
    frame.time_base = Fraction(1, 2 * rate)
    frame.pts = 71 + 2 * offset
    return frame


def _signal(rate, offset, n, hot):
    if not hot:
        return np.zeros((2, n), dtype=np.int16)
    i = offset + np.arange(n)
    t = (i % (rate // 8)) / rate
    left = np.sin(2 * np.pi * (240 * t + 16_000 * t**2) + 0.3)
    # A repeated chirp and full-scale square bursts create real inter-sample peaks.
    left = np.where(i // (rate // 200) % 13 == 0, 2 * (i // 3 % 2) - 1, left)
    right = 0.88 * left + 0.12 * np.sin(2 * np.pi * 731 * i / rate)
    return np.rint(32767 * np.stack((left, right))).astype(np.int16)


def _returned_pcm(output, source):
    assert isinstance(output, av.AudioFrame)
    assert (output.samples, output.sample_rate) == (source.samples, source.sample_rate)
    assert (output.pts, output.time_base) == (source.pts, source.time_base)
    assert (output.format.name, output.layout.name) == ("s16", "mono")
    return output.to_ndarray().ravel()


@pytest.mark.parametrize("rate", [16_000, 48_000])
def test_live_guard_skips_work_reprobes_and_keeps_sample_alignment(monkeypatch, rate):
    processor = live.LiveAudioDenoiseProcessor(0, 0)
    now = 0
    gate = live.StreamingSpectralGate.process
    # An isolated miss, exactly-on-deadline misses, a failed probe, then recovery.
    offsets = iter([0, -1, 0, 0, 0, 1, -1, -1])

    def timed_gate(self, mono):
        nonlocal now
        now += mono.size * 1_000_000_000 // rate + next(offsets)
        return gate(self, mono)

    monkeypatch.setattr(live, "perf_counter_ns", lambda: now)
    monkeypatch.setattr(live.StreamingSpectralGate, "process", timed_gate)
    guarded = [False] * 5 + [True] * 32 + [False] + [True] * 32 + [False] * 2
    rng = np.random.default_rng(17)
    sent, received = [], []
    offset = 0
    with patch.object(live, "apply_quality_booster", wraps=live.apply_quality_booster) as boost:
        for i, expected in enumerate(guarded):
            n = rate // (50 if i % 2 else 100)
            # At this level both paths are transparent to PCM16 precision.
            marker = rng.integers(-300, 300, n, dtype=np.int16)
            frame = _frame(np.stack((marker, marker)), rate, offset, i % 2)
            received.append(_returned_pcm(processor.recv(frame), frame))
            sent.append(marker)
            offset += n
            stats = processor.telemetry
            assert stats["last_degraded"] is expected
            assert stats["latency_budget_ms"] == n * 1000 / rate
            if i == 0:
                assert stats["latency_budget_met"] is False
        assert boost.call_count == 8
    assert stats["deadline_misses"] == 5
    assert stats["degraded_frames"] == 64 and stats["frames_failed"] == 0
    assert stats["degraded_remaining"] == 0 and stats["latency_budget_met"]
    delay = rate * 4 // 1000
    np.testing.assert_array_equal(
        np.concatenate(received), np.pad(np.concatenate(sent), (delay, 0))[:-delay]
    )


@pytest.mark.parametrize("rate,ms,planar", list(product((16_000, 48_000), (10, 20), (False, True))))
def test_live_hot_pcm_peak_across_guard_entry_and_exit(monkeypatch, rate, ms, planar):
    n = rate * ms // 1000
    processor = live.LiveAudioDenoiseProcessor(12, 6)
    # The first three callbacks miss; 32 guarded callbacks and a full probe follow.
    ticks = []
    for i in range(45):
        start = i * 30_000_000
        ticks.extend((start, start + (ms * 1_000_000 if i < 3 else 100_000)))
    monkeypatch.setattr(live, "perf_counter_ns", iter(ticks).__next__)
    received = []
    with patch.object(booster, "upfirdn", wraps=booster.upfirdn) as fir:
        for i in range(45):
            pcm = _signal(rate, i * n, n, True)
            frame = _frame(pcm, rate, i * n, planar)
            if i == 0:
                mono, sr = live._audio_frame_to_mono_float32(frame)
                np.testing.assert_array_equal(mono, pcm.mean(axis=0) / 32768)
                assert sr == rate
            before = fir.call_count
            received.append(_returned_pcm(processor.recv(frame), frame))
            if 4 <= i < 35:  # past the entry crossfade and lookahead
                assert fir.call_count == before
        assert fir.call_count > 0  # healthy, hot PCM exercises the oversampled detector
    audio = np.concatenate(received).astype(np.float32) / 32767
    assert booster.calculate_true_peak_dbfs(audio, rate) <= -0.5
    assert booster.calculate_true_peak_dbfs(audio, rate, oversample_factor=32) <= -0.5
    assert np.min(np.max(np.abs(audio.reshape(-1, n)), axis=1)) > 0.15
    assert processor.telemetry["degraded_frames"] == 32
    assert processor.telemetry["frames_failed"] == 0


@pytest.mark.parametrize("rate", [16_000, 48_000])
def test_streaming_pcm_limiter_is_independent_of_chunk_boundaries(rate):
    t = np.arange(2 * rate + 13) / rate
    audio = np.sin(2 * np.pi * (632 + 2130 * t) * t + 0.8).astype(np.float32)
    audio[(t > 0.7) & (t < 1.0)] *= 0.01
    audio[t > 1.5] = np.random.default_rng(3).uniform(-1, 1, np.count_nonzero(t > 1.5))
    limiter = booster.StreamingTruePeakLimiter(rate)
    whole = np.concatenate((limiter.process(audio), limiter.flush()))
    cuts = np.unique(np.random.default_rng(7).integers(1, audio.size, 370))
    chunks = [limiter.process(chunk) for chunk in np.split(audio, cuts)]
    assert sum(chunk.size for chunk in chunks) == audio.size - rate * 4 // 1000
    np.testing.assert_array_equal(np.concatenate([*chunks, limiter.flush()]), whole)
    assert limiter.flush().size == 0
    assert whole.size == audio.size and np.max(np.abs(whole)) > 0.75
    pcm = np.rint(whole * 32767) / 32767
    assert booster.calculate_true_peak_dbfs(pcm, rate, oversample_factor=32) <= -0.5


def test_live_guard_counts_misses_even_when_mandatory_work_is_late(monkeypatch):
    processor = live.LiveAudioDenoiseProcessor()
    # A 30 ms packet still has a strict 20 ms deadline, even while guarded.
    ticks = iter(range(0, 100 * 40_000_000, 20_000_000))
    monkeypatch.setattr(live, "perf_counter_ns", ticks.__next__)
    n = 480
    for i in range(36):
        frame = _frame(_signal(16_000, i * n, n, True), 16_000, i * n, True)
        assert np.any(_returned_pcm(processor.recv(frame), frame))
    stats = processor.telemetry
    assert stats["deadline_misses"] == 36 and not stats["latency_budget_met"]
    assert stats["degraded_frames"] == 32 and stats["degraded_remaining"] == 32
    assert stats["latency_budget_ms"] == 20 and stats["frames_failed"] == 0


def test_live_pcm_failure_mutes_instead_of_returning_hot_input():
    processor = live.LiveAudioDenoiseProcessor(0, 6)
    hot = _signal(48_000, 0, 480, True)
    processor.recv(_frame(hot, 48_000, 0, False))
    with patch.object(booster.StreamingTruePeakLimiter, "process", side_effect=ValueError):
        frame = _frame(hot, 48_000, 480, True)
        assert not np.any(_returned_pcm(processor.recv(frame), frame))
    silence = _frame(np.zeros_like(hot), 48_000, 960, False)
    assert not np.any(_returned_pcm(processor.recv(silence), silence))
    assert processor.telemetry["frames_failed"] == 1
    assert processor.telemetry["last_degraded"]


def test_file_route_does_not_load_live_extras():
    # Test in a fresh interpreter: other voice tests have already imported PyAV.
    code = """
import sys
sys.modules['av'] = sys.modules['streamlit_webrtc'] = None
from streamlit.testing.v1 import AppTest
app = AppTest.from_string('''
import numpy as np
from modules.voice.audio_loader import encode_wav_bytes
from ui.tabs_voice.main_layout import render_voice_app
render_voice_app({'filename':'test.wav', 'bytes':encode_wav_bytes(np.zeros(1600), 16000)})
''').run(timeout=60)
assert not app.exception and len(app.tabs) == 4
assert app.session_state['voice_execution_mode'] == 'file_offline'
assert 'ui.tabs_voice.tab_live_webrtc' not in sys.modules
selector = app.radio(key='voice_input_source')
selector.set_value(selector.options[1]).run(timeout=60)
assert not app.exception and app.session_state['voice_execution_mode'] == 'live_stream'
assert any('Live microphone support is unavailable' in e.value for e in app.error)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=100)


def _benchmark(rate, ms, planar, hot):
    n = rate * ms // 1000
    processor = live.LiveAudioDenoiseProcessor(12, 6)
    times, outputs, guarded = [], [], []
    for i in range(1025):
        frame = _frame(_signal(rate, i * n, n, hot), rate, i * n, planar)
        start = perf_counter_ns()
        out = processor.recv(frame)
        times.append((perf_counter_ns() - start) / 1e6)
        outputs.append(_returned_pcm(out, frame))
        if processor.telemetry["last_degraded"]:
            guarded.append(times[-1])
    steady = np.array(times[1:])
    audio = np.concatenate(outputs).astype(np.float32) / 32767
    peak = booster.calculate_true_peak_dbfs(audio, rate, oversample_factor=32)
    assert processor.telemetry["frames_failed"] == 0 and peak <= -0.5
    assert (peak > -12) if hot else (peak == -120)
    print(rate, ms, "s16p" if planar else "s16", "hot" if hot else "silence",
          *(f"{v:.3f}" for v in (times[0], *np.percentile(steady, [50, 95, 99, 100]))),
          int(times[0] >= ms), np.count_nonzero(steady >= ms), len(guarded), f"{peak:.5f}")
    if guarded:
        print("guarded only: p50/p95/p99/max ms", np.round(np.percentile(guarded, [50, 95, 99, 100]), 3),
              "misses", np.count_nonzero(np.array(guarded) >= ms))


if __name__ == "__main__":
    print("recv compute ms; fresh processor per case, imports loaded; 1 cold + 1024 steady")
    print("Hz ms format signal cold p50 p95 p99 max cold_miss steady_misses guarded 32x_dBTP")
    for case in product((48_000, 16_000), (10, 20), (False, True), (False, True)):
        _benchmark(*case)

    gate = live.StreamingSpectralGate.process

    def stalled_gate(self, mono):
        sleep(0.025)
        return gate(self, mono)

    print("Separate overload run: injected 25 ms stall in full-mode spectral gate")
    with patch.object(live.StreamingSpectralGate, "process", stalled_gate):
        _benchmark(48_000, 10, False, True)
