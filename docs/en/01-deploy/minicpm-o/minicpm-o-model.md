## MiniCPM-o 4.5 Model Introduction

This page introduces the **MiniCPM-o 4.5** omni-modal architecture, sub-model breakdown, and GGUF file checklist to help you prepare before deployment.

> Prerequisite: Recommended to complete [ROCm Environment Setup](/00-environment/) first.

---

### 1. What is MiniCPM-o

MiniCPM-o (Omni) is an end-to-end omni-modal model series from OpenBMB. Version 4.5 has approximately **9B total parameters**, composed of four backbone modules:

| Backbone Module | Base Model |
|-----------------|------------|
| LLM backbone | **Qwen3-8B** |
| Vision encoder | **SigLIP2** |
| Audio encoder | **Whisper-medium** |
| Speech synthesis (TTS) | **CosyVoice2** |

> The model is **built in an end-to-end fashion** — the modality encoders/decoders and LLM are densely connected via hidden states, enabling better information flow and control, and facilitating full exploitation of rich multimodal knowledge during training.

Based on this architecture, MiniCPM-o 4.5 integrates the following capabilities:

| Capability | Description |
|------------|-------------|
| **Text chat** | Same as a regular LLM, supports multi-turn conversation and system prompts, 30+ languages |
| **Voice input** | Real-time audio stream encoding, understands spoken input (bilingual real-time speech in Chinese & English) |
| **Image / Video understanding** | High-resolution images (up to 1.8M pixels) and high-FPS video understanding (up to 10fps) |
| **Voice output (TTS)** | Natural speech synthesis, supports voice cloning, emotion control, and long speech generation (>1 min) |
| **Full-duplex conversation** | Simultaneously listens to the microphone and generates voice responses, input and output streams do not block each other |
| **Proactive interaction** | Monitors input streams at ~1Hz frequency, can initiate unprompted responses based on continuous scene understanding |

Compared to a pure text model of the same parameter count, MiniCPM-o 4.5 achieves a full-duplex response latency of approximately **800 ms**, enabling near-real-time voice interaction on edge devices.

---

### 2. Model Architecture

MiniCPM-o 4.5 consists of multiple independent modules, each loaded as a separate GGUF file by `llama.cpp-omni`:

```
MiniCPM-o 4.5 (full omni inference)
├── LLM (Qwen3-8B)             MiniCPM-o-4_5-Q4_K_M.gguf             ~4.9 GB
├── Vision encoder (SigLIP2)    vision/MiniCPM-o-4_5-vision-F16.gguf  ~0.9 GB
├── Audio encoder               audio/MiniCPM-o-4_5-audio-F16.gguf ~1.2 GB
├── TTS language model          tts/MiniCPM-o-4_5-tts-F16.gguf   ~0.5 GB
├── TTS projector               tts/MiniCPM-o-4_5-projector-F16.gguf  ~0.1 GB
└── Token-to-Wav (vocoder)
    ├── token2wav-gguf/encoder.gguf
    ├── token2wav-gguf/flow_matching.gguf
    ├── token2wav-gguf/flow_extra.gguf
    ├── token2wav-gguf/hifigan2.gguf
    └── token2wav-gguf/prompt_cache.gguf    total ~0.7 GB
```

**All 10 GGUF files total approximately 8.3 GB** (Q4_K_M quantization).

---

### 3. Available Versions and Quantization

This tutorial uses **Q4_K_M quantization** for the LLM backbone, which balances precision and VRAM usage well for consumer AMD GPUs and APUs.

| File | Quantization | Purpose |
|------|-------------|---------|
| `MiniCPM-o-4_5-Q4_K_M.gguf` | Q4_K_M | LLM backbone (text generation) |
| `vision/MiniCPM-o-4_5-vision-F16.gguf` | F16 | Image understanding |
| `audio/MiniCPM-o-4_5-audio-F16.gguf` | F16 | Audio input understanding |
| `tts/MiniCPM-o-4_5-tts-F16.gguf` | F16 | Speech synthesis (LM part) |
| `tts/MiniCPM-o-4_5-projector-F16.gguf` | F16 | Speech synthesis (projection) |
| `token2wav-gguf/*.gguf` (5 files) | — | Vocoder (audio waveform generation) |

> Sub-models (vision, audio, TTS) are currently only available in F16. Even with Q4 quantization on the LLM backbone, total VRAM usage remains ~8.3 GB. Future Q8/Q4 sub-model variants may reduce this further.

---

### 4. VRAM Estimation

| Mode | Modules loaded | Approx. VRAM |
|------|----------------|--------------|
| Text only | LLM | ~5 GB |
| Text + image | LLM + vision encoder | ~6 GB |
| Text + voice input | LLM + audio encoder | ~6.5 GB |
| Full omni (voice in + TTS out) | LLM + vision + audio + TTS + Token2Wav | **~9 GB** |

> **Note**: Estimates assume a 4096-token context. Using an 8192-token context adds ~1–2 GB for KV cache.
>
> AMD Ryzen AI MAX+ 395 APUs with 64 GB unified memory (all usable as VRAM) can comfortably run all modules with room for large contexts.

---

### 5. Conversation Modes

Both `llama.cpp-omni` and MiniCPM-o-Demo support 4 conversation modes:

| Mode | Input | Output | Notes |
|------|-------|--------|-------|
| **Turn-based** | Audio file / text | Text + TTS | Most stable; good for initial validation |
| **Half-Duplex** | Live microphone | Text + TTS | Speaks after user finishes; no interruption |
| **Omni (full-duplex)** | Live mic + camera | Real-time audio stream | Simultaneous listen + speak with interruption |
| **Audio Duplex** | Live microphone | Real-time audio stream | Same as Omni but camera not required |

CLI deployment (`llama-omni-cli`) is suited to validating Turn-based mode. The Web Demo supports all 4 modes, enabling real Omni full-duplex via a browser with camera and microphone.

---

### 6. Where to Download Models

GGUF files are available from **ModelScope** (recommended for mainland China, faster download) or **Hugging Face**:

| Source | URL |
|--------|-----|
| Hugging Face | [OpenBMB/MiniCPM-o-4_5-gguf](https://huggingface.co/OpenBMB/MiniCPM-o-4_5-gguf) |
| ModelScope | [OpenBMB/MiniCPM-o-4_5-gguf](https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf) |

---

### References

- [MiniCPM-o Official Repository](https://github.com/OpenBMB/MiniCPM-o)
- [MiniCPM-o Technical Report (arxiv)](https://arxiv.org/abs/2408.01800)
- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)

---

> After reading about the model architecture, proceed to [llama.cpp-omni CLI Deployment](./llamacpp-omni-rocm7-deploy.md) or [Web Demo Full-Duplex Deployment](./webdemo-rocm7-deploy.md).
