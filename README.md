# AutoData-Hub

[![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.x%2F13.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**AutoData-Hub** is a local-first, privacy-preserving multi-modal MLOps preprocessing and enhancement platform spanning **Tabular Datasets**, **Image Datasets**, **Voice / Audio**, and **Video Media**. It unifies `modules/` as the canonical single source of truth and processing core consumed by the interactive Streamlit workspace (`ui/`) and the FastAPI + React service layer (`backend/api/` + `frontend/`).

> **Security & Runtime Boundary:** AutoData-Hub is designed for local single-host evaluation and controlled development workflows. Bind services to loopback (`127.0.0.1`) unless fronted by an authenticated reverse proxy.

---

## 1. System Architecture Diagram

```text
+-----------------------------------------------------------------------------------+
|                        MULTIMODAL INGESTION & ROUTER (app.py)                     |
|   Tabular (.csv/.parquet/.xlsx/.json/SQL)  |  Image (.png/.jpg/.webp/.zip)        |
|   Voice (.wav/.mp3/.flac/.ogg/.m4a/WebRTC) |  Video (.mp4/.avi/.mov/.mkv/.webm)   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               SINGLETON HARDWARE ENGINE (modules/image/gpu_manager.py)            |
|   @st.cache_resource -> get_cached_hardware_context()                             |
|   - Queries torch.cuda.is_available() & onnxruntime.get_available_providers()     |
|   - Preloads CUDAExecutionProvider once per lifecycle (prevents WDDM TDR resets)  |
|   - Bounds CPU threads: max(1, min(os.cpu_count() - 1, 4))                        |
+------+------------------------+-------------------------+-------------------------+
       |                        |                         |                         |
       v                        v                         v                         v
+--------------+      +------------------+      +------------------+      +------------------+
| TABULAR CORE |      |   VISION CORE    |      |    VOICE CORE    |      |    VIDEO CORE    |
| modules/     |      | modules/image/   |      | modules/voice/   |      | modules/video/   |
| tabular/     |      | - Laplacian/Tenen|      | - SNR & SI-SDR   |      | - Bounded Keyfrm |
| - Health Q   |      | - LAB CLAHE      |      | - DeepFilterNet3 |      | - HSV Scene Cuts |
| - PII Luhn   |      | - NAFNet/Restorm |      | - VoiceFixer 48k |      | - Farneback Flow |
| - KNN/MICE   |      | - Real-ESRGAN    |      | - 5-Band SOS EQ  |      | - EMA Color Smooth|
| - IF/LOF/SVM |      | - DarkIR/SwinIR  |      | - DeEsser & VCA  |      |   (alpha = 0.15) |
| - AST Export |      | - ASC CDL Color  |      | - WebRTC <20ms   |      | - pipeline_video |
+------+-------+      +--------+---------+      +--------+---------+      +--------+---------+
       |                       |                         |                         |
       +-----------------------+-------------------------+-------------------------+
                               |
                               v
+-----------------------------------------------------------------------------------+
|                 UNIFIED STATE OUTPUT & REPRODUCIBLE DELIVERABLES                  |
|   - Standardized @dataclass Contracts (TransformResult / ModuleResult)            |
|   - Versioned Lineage (v0_raw -> v1_privacy -> v2_imputed -> v3_anomaly)          |
|   - 18 Guided UI Tabs (ui/utils_ux.py -> render_tab_guidance_header)              |
|   - Exports: Parquet/CSV/XLSX, 24-bit WAV/MP3/FLAC/OGG, PNG/ZIP, PDF & pipeline.py|
+-----------------------------------------------------------------------------------+
```

### Engine Consolidation & Current Capability Matrix

| Subsystem | Location | Status | Notes |
| --- | --- | --- | --- |
| **Unified Processing Core** | `modules/` | Implemented | Single source of truth consumed across `ui/` (Streamlit) and `backend/api/` (FastAPI) |
| **Tabular Hygiene Core** | `modules/tabular/`, `modules/health_audit/`, `modules/privacy/`, `modules/imputation/`, `modules/anomaly/` | Implemented | Non-mutating lineage (`v0_raw` → `v3_anomaly`), CleanML $Q$-Score, Luhn PII, MICE/KNN, IF/LOF/OC-SVM |
| **AST Script Synthesizer** | `modules/tabular/synthesizer.py`, `modules/script_gen/exporter.py` | Full Multi-Modal AST Pipeline Exporter | Generates standalone validated `pipeline.py` scripts and ReportLab `audit_report.pdf` |
| **Vision Core** | `modules/image/`, `ui/tabs_image/` | Implemented | Classical OpenCV + ONNX Runtime (`NAFNet`, `Restormer`, `Real-ESRGAN`, `SwinIR`, `RealPLKSR`, `DarkIR`) + ASC CDL Color Engine |
| **Voice & WebRTC Core** | `modules/voice/`, `ui/tabs_voice/` | Implemented | Offline 48 kHz mastering and live PCM denoising/peak limiting with a per-frame DSP budget; network and browser latency are separate |
| **Video Temporal Core** | `modules/video/`, `ui/tabs_video/` | Implemented | Budgeted `cv2.VideoCapture` decoding, 3D HSV scene cuts, Farneback vs. defocus blur, and EMA ($\alpha=0.15$) color grading |
| **In-Tab UX Guidance** | `ui/utils_ux.py` | Implemented | Capability guides across 18 primary workspace views and their input/result subtabs |
| **FastAPI & React Dashboard** | `backend/`, `frontend/` | Implemented (local dev) | REST API on `127.0.0.1:8000` with SQLite metadata and partitioned `storage/` artifacts |

---

## 2. Hardware Acceleration & GPU/CPU Conflict Resolution Policy

AutoData-Hub coordinates PyTorch and ONNX Runtime within the same Python process using a strict singleton policy in [`modules/image/gpu_manager.py`](modules/image/gpu_manager.py) and [`modules/voice/device_manager.py`](modules/voice/device_manager.py):

| Conflict / Risk | Root Cause | Mitigation Policy |
| --- | --- | --- |
| **Streamlit Rerun Thrashing** | Repeated framework imports, provider discovery, and model initialization delay interactive reruns. | Cache discovery per Python process and reuse ONNX sessions by model and provider. Classical OpenCV work does not require initializing CUDA. Separate server workers do not share a GPU context. |
| **Windows WDDM TDR Driver Resets** | Repeated main-thread CUDA context allocations and DLL loading can trigger Windows Display Driver Model Timeout Detection & Recovery (TDR) resets. | `torch` is imported prior to `onnxruntime`, `ort.preload_dlls()` executes once inside the cached singleton, and `InferenceSession` objects are reused (`max_entries=2`). |
| **CPU Thread Starvation** | PyTorch/ONNX unbounded OpenMP thread pools contend with Uvicorn ASGI workers and WebRTC audio threads. | Enforces $\text{NumThreads} = \max(1, \min(\text{CPU\_Count} - 1, 4))$ across both `torch.set_num_threads` and `ort.SessionOptions`. |
| **ONNX Provider Fallback Chain** | GPU environments may lack compatible CUDA/cuDNN DLLs or run on CPU-only hosts. | Registers providers sequentially as `["CUDAExecutionProvider", "CPUExecutionProvider"]`, cleanly degrading to `["CPUExecutionProvider"]` or OpenCV/SciPy fallbacks. |

```python
# modules/image/gpu_manager.py
@st.cache_resource(show_spinner=False)
def get_cached_hardware_context() -> Dict[str, Any]:
    """Cache GPU discovery and CUDA provider loading across Streamlit reruns."""
    return _probe_vision_execution_context()

def run_pytorch_vision_forward(model_or_fn, *args, execution_context=None, **kwargs):
    """Execute a PyTorch vision forward pass inside FP16 autocast when CUDA is active."""
    with vision_autocast_context(execution_context):
        return model_or_fn(*args, **kwargs)
```

---

## 3. Multi-Modal Subsystem Deep Dive

### A. Tabular Hygiene Core (`modules/tabular/`, `ui/tabs/`)
- **Health Audit (`modules/health_audit/auditor.py`)**: Computes composite Quality Score $Q$, vectorized missingness, JSON-safe duplicate detection, univariate IQR outlier bounds, and mixed-type mismatches.
- **Privacy & PII Firewall (`modules/privacy/`)**: Combines semantic header inspection with regex + **Luhn (Mod-10) checksum validation** for credit cards, applying deterministic Redaction, Salted SHA-256 Hashing, or Anonymization (`v1_privacy`).
- **Missing Value Imputation (`modules/imputation/`)**: Evaluates `auto_cv` RMSE across `KNNImputer`, `IterativeImputer` (MICE BayesianRidge and RandomForest/MissForest-style non-linear regressors), Mean, Median, and Mode, automatically injecting binary `_is_missing` MNAR indicators when missingness exceeds 15% (`v2_imputed`).
- **Anomaly Ensemble (`modules/anomaly/`)**: Combines **Isolation Forest**, **Local Outlier Factor (LOF)**, and **One-Class SVM** over `RobustScaler`-standardized features with Adaptive 3-Sigma EVT or fixed quantile cutoffs and Winsorization/Pruning mitigation (`v3_anomaly`).
- **AST Pipeline Synthesizer (`modules/tabular/synthesizer.py`, `modules/script_gen/exporter.py`)**: Compiles immutable `TransformationAction` records into AST-validated standalone `pipeline.py` scripts and SHA-256-verified PDF audit reports.

### B. Vision Core (`modules/image/`, `ui/tabs_image/`)
- **Focus & Sharpness (`sharpness_evaluator.py`)**: Computes spatial Laplacian variance $\text{Var}(\nabla^2 I)$, Sobel Tenengrad gradient energy, and mean luminance with auto-degradation alerts.
- **Dual-Tier Restoration (`denoiser.py`, `clahe_contrast.py`, `neural_restoration.py`)**:
  - *Tier-1*: Bilateral / Gaussian denoising + LAB luminance-only CLAHE (preserving chrominance).
  - *Tier-2*: Checksum-verified ONNX Runtime models (`NAFNet`, `Restormer`, `Real-ESRGAN x4`, `SwinIR`, `RealPLKSR`, `UltraSharp V2`, `DAT Nomos8k`) with overlap tiling and audited OpenCV fallback when weights are absent. DarkIR inference requires a separately supplied ONNX export; the published DarkIR checkpoint is PyTorch.
- **Color Balancing & Grading (`color_engine.py`, `tab_color_grading.py`)**: Each edited ZIP member has its own `ImagePipelineBuffer`. Sliders grade a copy of the latest restoration (or raw upload); rerunning restoration reapplies the saved settings. Reset Color removes only the grade. Single and ZIP exports use the same final pixels.
  - `apply_auto_white_balance(image_bgr, method=...)`: Supports `gray_world` ($\mu_{\text{gray}} / \mu_C$) and `max_rgb` ($1.0 / \max I_C$) chromatic scaling.
  - `apply_color_grading(...)`: Non-linear ASC CDL Lift-Gamma-Gain transfer:
    $$I_{\text{out}} = \text{clip}\left((I \cdot (1 - L) + L)^{1 / \Gamma} \cdot G,\ 0,\ 1\right)$$
    with Rec.709 saturation scaling and presets `🎬 Teal & Orange`, `📻 Vintage Film`, `❄️ Cold Cyberpunk`, and `Custom`.

### C. Voice & Speech Core (`modules/voice/`, `ui/tabs_voice/`)
- **Acoustic Telemetry (`audio_metrics.py`)**: Evaluates SNR, zero-mean SI-SDR, estimated PESQ/pystoi STOI, crest factor, RMS/true-peak dBFS, EBU R128 LUFS, and FFT spectrograms.
- **Dual-Tier Denoising & 48 kHz BWE (`denoiser_sota.py`, `bandwidth_extension.py`)**: Runs `DeepFilterNet3` ONNX/TorchScript (with STFT spectral-gate fallback) and `VoiceFixer` / `MP-SENet` 48 kHz bandwidth extension.
- **Studio Mastering DSP (`studio_tuner.py`, `tuner.py`, `booster.py`)**: Linked-channel 4th-order Butterworth highpass, 5-band SOS biquad parametric EQ, dynamic sidechain De-Esser ($5\text{–}9\text{ kHz}$), soft-knee VCA compressor, tube/tape harmonic excitation, and 4x oversampled lookahead True-Peak Limiter ($-0.5\text{ dBFS}$ lossless / $-3.0\text{ dBFS}$ lossy).
- **Live WebRTC Microphone Stream (`ui/tabs_voice/main_layout.py`, `tab_live_webrtc.py`)**: Sidebar selector toggles between `📁 Audio File Upload` (4 mastering tabs) and `🎤 Live Microphone Stream (WebRTC)`, processing 10–20 ms `av.AudioFrame` PCM buffers via `streamlit-webrtc` with `<20 ms` DSP budget telemetry.

### D. Video Temporal Core (`modules/video/`, `ui/tabs_video/`)
- **Streamed Keyframe Extraction (`keyframe_extractor.py`)**: Sequential `cv2.VideoCapture` decoding with a per-job working-memory budget (`MAX_MEMORY_BUDGET_BYTES`). Native codec allocations, the Python runtime, and concurrent sessions are not an OS-enforced process memory limit.
- **Scene Cut Detection (`scene_detector.py`)**: L1-normalized 3D HSV color histograms compared via Pearson correlation; flags scene transitions at $d_{\text{corr}}(H_t, H_{t-1}) < 0.65$.
- **Motion vs. Defocus Blur (`temporal_blur.py`)**: Dual-metric classifier combining Laplacian variance $\text{Var}(\nabla^2 I)$ and dense Farneback optical flow magnitude $M_{\text{flow}}$ to distinguish `motion_blur` (low sharpness + high flow) from `defocus_blur` (low sharpness + low flow) and `sharp` frames.
- **EMA Temporal Color Stabilization (`color_engine_video.py`)**: Smooths chromatic gains across frames via Exponential Moving Average ($\alpha = 0.15$, reset at detected scene cuts) to eliminate frame-to-frame flicker.
- **Master Orchestrator (`pipeline_video.py`)**: `process_video_pipeline()` returns standardized `@dataclass` contracts (`VideoTransformResult(TransformResult)` and `ModuleResult`).

### E. Standardized In-Tab UX Guidance (`ui/utils_ux.py`)
Every single tab across Tabular (5 tabs), Vision (4 tabs), Voice (5 tabs), and Video (4 tabs) mounts `render_tab_guidance_header(title, objective, methodology, deliverable)` via `st.expander` so analysts see the objective, algorithm, and expected deliverable directly inside each workspace tab.

---

## 4. Mathematical Engine Formulas

1. **CleanML Composite Quality Score ($Q \in [0, 100]$)**:
   $$Q = \max\left(0,\ 100 - \left(w_1 M + w_2 D + w_3 O + w_4 T\right) \cdot 100\right)$$
   with default weights $(w_1, w_2, w_3, w_4) = (0.35, 0.20, 0.25, 0.20)$ and denominator stabilization $\epsilon = 10^{-9}$ (`_safe_ratio` in `modules/health_audit/auditor.py`) so 0-row and 0-column DataFrames yield finite scores without `ZeroDivisionError`.

2. **Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)**:
   $$\tilde{s} = s - \mu_s, \quad \tilde{\hat{s}} = \hat{s} - \mu_{\hat{s}}, \quad s_{\text{target}} = \frac{\langle \tilde{\hat{s}}, \tilde{s} \rangle}{\max(\|\tilde{s}\|_2^2,\ \epsilon_1)} \tilde{s}$$
   $$\text{SI-SDR} = 10 \log_{10} \frac{\max(\|s_{\text{target}}\|_2^2,\ \epsilon_1)}{\max(\|\tilde{\hat{s}} - s_{\text{target}}\|_2^2,\ \epsilon_1)}$$
   using mandatory zero-mean centering and epsilon energy floor $\epsilon_1 = 10^{-10}$ in `modules/voice/audio_metrics.py` (`_EPSILON = 1e-10`).

3. **Calibrated Outlier Consensus ($V \ge 2/3$)**:
   Each detector $m \in \{\text{IF}, \text{LOF}, \text{OC-SVM}\}$ emits oriented scores normalized to $P_m(x_i) \in [0, 1]$, averaged into ensemble score $\bar{P}(x_i) = \frac{1}{M}\sum_{m=1}^M P_m(x_i)$. A record is flagged when both the ensemble threshold and majority consensus hold:
   $$\text{Flag}(x_i) = \mathbb{I}\!\left(\bar{P}(x_i) \ge \tau\right) \land \mathbb{I}\!\left(\sum_{m=1}^{M} \mathbb{I}\!\left(P_m(x_i) \ge 0.5\right) \ge \lceil 0.66 \cdot M \rceil\right)$$

4. **HSV Histogram Scene Cut Correlation**:
   $$d_{\text{corr}}(H_t, H_{t-1}) = \frac{\langle H_t - \bar{H}_t,\ H_{t-1} - \bar{H}_{t-1} \rangle}{\|H_t - \bar{H}_t\|_2 \|H_{t-1} - \bar{H}_{t-1}\|_2} < 0.65 \implies \text{Scene Cut}$$

---

## 5. Quickstart & Installation Guide

> **Environment Rule:** Never install `onnxruntime` (CPU) and `onnxruntime-gpu` (CUDA) in the same virtual environment.

### Prerequisites
- 64-bit **Python 3.11–3.13** (Python 3.12 recommended).
- **Node.js 20.19+** (optional, for the React dashboard in `frontend/`).

### Option A: CPU Mode (`requirements-cpu.txt`)

```powershell
# Windows PowerShell
py -3.12 -m venv .venv-cpu
.\.venv-cpu\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-cpu\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

```bash
# Linux / macOS
python3 -m venv .venv-cpu
./.venv-cpu/bin/python -m pip install --upgrade pip setuptools wheel
./.venv-cpu/bin/python -m pip install -r requirements-cpu.txt
```

### Option B: NVIDIA CUDA GPU Mode (`requirements-gpu.txt`)

```powershell
# Windows PowerShell (CUDA 12.x/13.x + cuDNN 9)
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

### Launch Services

1. **Streamlit Multi-Modal Workspace**:
   ```bash
   python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
   ```
2. **FastAPI Backend + React Frontend (Optional)**:
   ```bash
   python -m uvicorn backend.main:app --env-file .env --host 127.0.0.1 --port 8000
   npm --prefix ./frontend ci && npm --prefix ./frontend run dev -- --host 127.0.0.1
   ```
3. **Provision Optional Vision Models**:
   ```bash
   python -m utils.setup_vision_models
   python -m utils.setup_vision_models --model darkir
   ```
   `--models` is an alias for `--model`; either accepts one or more IDs. DarkIR is opt-in. It downloads the authors' checksum-verified [PyTorch checkpoint](https://huggingface.co/Cidaut/DarkIR) (`DarkIR_384.pt`, CC-BY-4.0). It is not an ONNX export: the ONNX restoration path still requires a separately supplied `darkir.onnx` and otherwise uses the OpenCV fallback.

---

## 6. FastAPI REST & Real-Time Signaling Reference

Authoritative OpenAPI specification is served live at `http://127.0.0.1:8000/docs` and `/openapi.json`. Live browser audio signaling runs over WebRTC via `streamlit-webrtc` (`WebRtcMode.SENDRECV`) in the Streamlit Voice workspace rather than FastAPI WebSockets.

| Protocol / Method | Endpoint / Channel | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Service health, environment, worker allocation, and upload limit |
| `POST` | `/api/v1/upload` | Upload and classify a tabular or vision dataset |
| `GET` | `/api/v1/datasets` | List uploaded datasets |
| `GET` | `/api/v1/datasets/{dataset_id}` | Retrieve dataset metadata |
| `GET` | `/api/v1/datasets/{dataset_id}/preview` | Paginated tabular preview or vision thumbnail metadata |
| `GET` | `/api/v1/datasets/{dataset_id}/image/{image_name}` | Stream a raw uploaded image asset |
| `GET` | `/api/v1/config/presets` | List built-in and saved processing presets |
| `POST` | `/api/v1/config/presets` | Create a custom processing preset |
| `GET` | `/api/v1/config/presets/{config_id}` | Fetch a specific preset configuration |
| `POST` | `/api/v1/process` | Dispatch a processing job |
| `GET` | `/api/v1/jobs` | List processing jobs |
| `GET` | `/api/v1/jobs/{job_id}` | Poll job lifecycle status and stage progress |
| `GET` | `/api/v1/jobs/{job_id}/benchmarks` | Retrieve stage latency and throughput metrics |
| `GET` | `/api/v1/jobs/{job_id}/quality` | Retrieve pre/post quality comparison report |
| `GET` | `/api/v1/jobs/{job_id}/diff` | Retrieve tabular or visual before/after diff payload |
| `GET` | `/api/v1/jobs/{job_id}/image/{image_name}` | Stream a processed image asset |
| `GET` | `/api/v1/jobs/{job_id}/export` | Export job output (`csv`, `xlsx`, `pdf`, `json`, `png`, `jpg`, `zip`) |
| `GET` | `/api/v1/jobs/{job_id}/download-image` | Download processed vision image |
| `WebRTC` | `autodata-voice-live-webrtc` (`SENDRECV`) | Low-latency (`<20 ms`) live microphone PCM frame denoising & peak limiting |

---

## 7. Testing & Verification

Run the automated architecture/mathematical baseline audit and the full test suite:

```bash
python verify_codebase.py
python -m pytest tests/ -v
```

---

## License

Licensed under the [Apache License 2.0](LICENSE).
