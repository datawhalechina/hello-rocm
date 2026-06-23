## llama.cpp Deployment of MiniCPM-V (Ubuntu 24.04 + ROCm 7+)

This section explains how to deploy the **multimodal** model **MiniCPM-V 4.6** for
local image + text inference with **llama.cpp** on an AMD GPU (Ubuntu 24.04 + ROCm 7+).

It follows the same prebuilt-binary flow as the `qwen3/llamacpp-rocm7-deploy.md` example,
with one key difference: **MiniCPM-V is a vision-language model**, so in addition to the GGUF
weights you must also load an **`mmproj` multimodal projector** file, and you use
`llama-mtmd-cli` / `llama-server` with `--mmproj`.

The example model is **MiniCPM-V 4.6 Q4_K_M (GGUF)** — a 1.3B vision-language model, very
friendly for single-GPU / edge deployment.

> Prerequisite: ROCm 7+ system installation and verification is complete
> (see `env-prepare-ubuntu24-rocm7.md`). Verified on **AMD Ryzen AI MAX+ 395 (Radeon 8060S,
> gfx1151), ROCm 7.13**.

---

### Method 1 (Recommended): Prebuilt Executables

#### 1. Download the Prebuilt Version

Use the prebuilt llama.cpp provided by Lemonade, where:

- **370** corresponds to the **gfx1150** architecture
- **395** corresponds to the **gfx1151** architecture

Related links:

- https://github.com/lemonade-sdk/llamacpp-rocm
- https://github.com/lemonade-sdk/llamacpp-rocm/releases

```bash
mkdir -p ~/minicpmv-rocm && cd ~/minicpmv-rocm
# Pick the asset matching your architecture (gfx1151 shown here)
curl -L -o llama-rocm-gfx1151.zip \
  https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/b1292/llama-b1292-ubuntu-rocm-gfx1151-x64.zip
mkdir -p llama-bin && unzip -q llama-rocm-gfx1151.zip -d llama-bin
```

---

#### 2. Verify ROCm 7+ Installation (Must Be System-level ROCm)

```bash
amd-smi
```

You should see your GPU model, driver, and ROCm version, e.g.:

```
MARKET_NAME: Radeon 8060S Graphics
TARGET_GRAPHICS_VERSION: gfx1151
ROCm version: 7.13.0
```

Confirm llama.cpp sees the GPU through ROCm:

```bash
cd ~/minicpmv-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
./llama-cli --list-devices
# Available devices:
#   ROCm0: Radeon 8060S Graphics (65536 MiB, ... free)
```

---

#### 3. Enter the llama Backend Directory and Set Permissions / Environment Variables

```bash
cd ~/minicpmv-rocm/llama-bin
chmod +x llama-cli llama-server llama-mtmd-cli
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
```

> Note: the Lemonade build bundles its own ROCm runtime libraries next to the binaries, so put
> the backend directory itself on `LD_LIBRARY_PATH` (the `$PWD` above) in addition to `/opt/rocm/lib`.

---

#### 4. Download MiniCPM-V 4.6 GGUF + mmproj Projector

llama.cpp uses the **GGUF model format**. For a multimodal model you need **two** files:

- the quantized LLM weights (`*Q4_K_M*.gguf`)
- the vision projector (`mmproj-*.gguf`)

Using the Chinese Hugging Face mirror `https://hf-mirror.com/`:

```bash
mkdir -p ~/models/MiniCPM-V-4_6-gguf && cd ~/models/MiniCPM-V-4_6-gguf
export HF_ENDPOINT=https://hf-mirror.com

# Model weights (Q4_K_M, ~505 MB) and the multimodal projector (~1.1 GB)
for f in MiniCPM-V-4_6-Q4_K_M.gguf mmproj-model-f16.gguf; do
  curl -L --fail -o "$f" \
    "https://hf-mirror.com/openbmb/MiniCPM-V-4_6-gguf/resolve/main/$f"
done
```

> Tip: you can also use `hfd.sh` + `aria2` (as in the Qwen3 example) for faster, resumable
> multi-connection downloads. GGUF repo / file names may change with upstream updates — search
> `MiniCPM-V-4_6-gguf` on Hugging Face for the latest trusted repository.

---

#### 5. CLI Multimodal Test (`llama-mtmd-cli`)

This is the quickest way to confirm the vision pipeline works. `llama-mtmd-cli` is the
multimodal CLI; pass the projector with `--mmproj` and an image with `--image`:

```bash
cd ~/minicpmv-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH

./llama-mtmd-cli \
  -m ~/models/MiniCPM-V-4_6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf \
  --mmproj ~/models/MiniCPM-V-4_6-gguf/mmproj-model-f16.gguf \
  -ngl 99 -c 4096 --temp 0.2 \
  --image /path/to/image.jpeg \
  -p "Describe this image in detail."
```

Expected: a coherent natural-language description of the image, generated on `ROCm0`.

---

#### 6. Start llama-server (OpenAI-compatible API)

```bash
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
cd ~/minicpmv-rocm/llama-bin

./llama-server \
  -m ~/models/MiniCPM-V-4_6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf \
  --mmproj ~/models/MiniCPM-V-4_6-gguf/mmproj-model-f16.gguf \
  -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
```

---

#### 7. Test the API (text + image, with tokens/s)

**Text completion** (same formula as the Qwen3 example,
`completion_tokens / (timings.predicted_ms / 1000)`):

```bash
curl -s -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "MiniCPM-V-4_6",
  "prompt": "Explain large language models in one sentence",
  "max_tokens": 128
}' | jq -r '
.choices[0].text as $txt |
(.usage.completion_tokens / (.timings.predicted_ms / 1000)) as $tps |
"Generated text:\n\($txt)\n\ntokens/s: \($tps|tostring)"
'
```

**Multimodal chat** (image via base64 data URL on the OpenAI `chat/completions` endpoint):

```bash
IMG_B64=$(base64 -w0 /path/to/image.jpeg)
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "MiniCPM-V-4_6",
  "max_tokens": 128,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image in one sentence."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG_B64"'"}}
  ]}]
}' | jq -r '.choices[0].message.content'
```

Reference results on **Radeon 8060S (gfx1151), ROCm 7.13, ctx=4096**:

- Text decode: **~190 tokens/s** (MiniCPM-V 4.6 is only 1.3B, so much faster than an 8B model)
- Multimodal chat: **~190 tokens/s** decode (plus image-encode time on the first turn)
- **tokens/s depends on your actual hardware.**

---

### Method 2: Docker Method (Official ROCm llama.cpp Image)

If you prefer Docker, follow the official documentation and build from source inside the
container, exactly as in the Qwen3 / Gemma4 examples — the only MiniCPM-V-specific change is
to launch with both `-m <gguf>` and `--mmproj <mmproj-gguf>`:

- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/llama-cpp-install.html

```bash
./build/bin/llama-mtmd-cli \
  -m /data/MiniCPM-V-4_6-gguf/MiniCPM-V-4_6-Q4_K_M.gguf \
  --mmproj /data/MiniCPM-V-4_6-gguf/mmproj-model-f16.gguf \
  -ngl 99 -c 4096 \
  --image /data/image.jpeg \
  -p "What is in this image?"
```

> Note: building llama.cpp from source for ROCm (`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`) is
> covered step-by-step in `qwen3/llamacpp-rocm7-deploy.md`; multimodal support and the
> `mmproj` file are the only additions needed for MiniCPM-V.
