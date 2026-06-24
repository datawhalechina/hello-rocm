## llama.cpp Deployment of MiniCPM (Ubuntu 24.04 + ROCm 7+)

This section explains how to deploy the **text** LLM **MiniCPM5-1B** for local inference with
**llama.cpp** on an AMD GPU (Ubuntu 24.04 + ROCm 7+).

It follows the same prebuilt-binary flow as the `qwen3/llamacpp-rocm7-deploy.md` example.
Unlike the multimodal `minicpmv/` guide, MiniCPM is **text-only**, so there is **no `mmproj`
projector** — you load a single GGUF and use the standard `llama-cli` / `llama-server`.

The example model is **MiniCPM5-1B Q4_K_M (GGUF)** — a compact 1B dense model built for
on-device / edge deployment.

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
mkdir -p ~/minicpm-rocm && cd ~/minicpm-rocm
# Pick the asset matching your architecture (gfx1151 shown here)
curl -L -o llama-rocm-gfx1151.zip \
  https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/b1292/llama-b1292-ubuntu-rocm-gfx1151-x64.zip
mkdir -p llama-bin && unzip -q llama-rocm-gfx1151.zip -d llama-bin
```

> If you already set up llama.cpp for the `minicpmv/` (MiniCPM-V) guide, you can reuse the same
> `llama-bin/` directory here — the binaries are identical; only the model differs.

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
cd ~/minicpm-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
./llama-cli --list-devices
# Available devices:
#   ROCm0: Radeon 8060S Graphics (65536 MiB, ... free)
```

---

#### 3. Enter the llama Backend Directory and Set Permissions / Environment Variables

```bash
cd ~/minicpm-rocm/llama-bin
chmod +x llama-cli llama-server
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
```

> Note: the Lemonade build bundles its own ROCm runtime libraries next to the binaries, so put
> the backend directory itself on `LD_LIBRARY_PATH` (the `$PWD` above) in addition to `/opt/rocm/lib`.

---

#### 4. Download the MiniCPM5-1B GGUF

llama.cpp uses the **GGUF model format**. MiniCPM5-1B ships ready-made GGUFs:

| File | Size | Use case |
| --- | --- | --- |
| `MiniCPM5-1B-F16.gguf` | 2.1 GB | reference quality |
| `MiniCPM5-1B-Q8_0.gguf` | 1.1 GB | tiny quality drop vs F16 |
| `MiniCPM5-1B-Q4_K_M.gguf` | 657 MB | edge / minimal VRAM |

Using the Chinese Hugging Face mirror `https://hf-mirror.com/`:

```bash
mkdir -p ~/models/MiniCPM5-1B-GGUF && cd ~/models/MiniCPM5-1B-GGUF
export HF_ENDPOINT=https://hf-mirror.com

curl -L --fail -o MiniCPM5-1B-Q4_K_M.gguf \
  "https://hf-mirror.com/openbmb/MiniCPM5-1B-GGUF/resolve/main/MiniCPM5-1B-Q4_K_M.gguf"
```

> Tip: you can also use `huggingface-cli download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q4_K_M.gguf`,
> or `hfd.sh` + `aria2` for faster, resumable downloads.

---

#### 5. CLI Text Test (`llama-cli`)

```bash
cd ~/minicpm-rocm/llama-bin
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH

# Interactive chat (auto-applies the chat template)
./llama-cli \
  -m ~/models/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 -c 4096 --temp 0.7 --top-p 0.95 -n 2048
```

Expected: coherent generation on `ROCm0`. MiniCPM5-1B is a reasoning model and may emit a
`[Start thinking]` block before its final answer.

---

#### 6. Start llama-server (OpenAI-compatible API)

```bash
export LD_LIBRARY_PATH=$PWD:/opt/rocm/lib:$LD_LIBRARY_PATH
cd ~/minicpm-rocm/llama-bin

./llama-server \
  -m ~/models/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 -c 8192 --jinja --host 127.0.0.1 --port 8080
```

> `--jinja` enables the model's bundled chat template (recommended for MiniCPM5-1B).

---

#### 7. Test the API (chat + tokens/s)

```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "MiniCPM5-1B",
  "messages": [{"role": "user", "content": "1+1=? Then explain in one short sentence."}],
  "temperature": 0.7, "top_p": 0.95, "max_tokens": 256
}' | jq -r '
.choices[0].message.content as $txt |
(.usage.completion_tokens / (.timings.predicted_ms / 1000)) as $tps |
"Answer:\n\($txt)\n\ntokens/s: \($tps|tostring)"
'
```

Reference result on **Radeon 8060S (gfx1151), ROCm 7.13, ctx=8192**:

- Decode: **~185 tokens/s** (MiniCPM5-1B is a 1B model, so much faster than an 8B model)
- **tokens/s depends on your actual hardware.**

#### Generation parameters

| Mode | `--temp` | `--top-p` | When to use |
| --- | --- | --- | --- |
| Think | 0.9 | 0.95 | reasoning, math, code, multi-step |
| No-think | 0.7 | 0.95 | fast assistant, latency-bound |

---

### Method 2: Docker Method / Build from Source

If you prefer Docker or want to build llama.cpp from source for ROCm
(`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`), the steps are identical to the Qwen3 guide
(`qwen3/llamacpp-rocm7-deploy.md`); MiniCPM is text-only, so no extra flags are needed at runtime:

```bash
./build/bin/llama-cli \
  -m /data/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf \
  -ngl 99 -c 4096 -p "Explain AMD ROCm in two sentences."
```

#### Build a GGUF from your own checkpoint

If you fine-tuned your own MiniCPM5-1B variant and want a GGUF, convert + quantize with llama.cpp:

```bash
python ./convert_hf_to_gguf.py /path/to/your-MiniCPM5-fp16-hf --outfile F16.gguf --outtype f16
./build/bin/llama-quantize F16.gguf MiniCPM5-1B-Q4_K_M.gguf Q4_K_M
```

---

### See also

- `minicpmv/llamacpp-rocm7-deploy.md` — the **multimodal** MiniCPM-V on llama.cpp (adds an `mmproj`
  projector + `llama-mtmd-cli` for image input).
- `qwen3/llamacpp-rocm7-deploy.md` — the reference text-model llama.cpp flow this guide mirrors.
