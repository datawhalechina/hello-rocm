# 1.5 标准引擎三：`vLLM` 工业级高并发部署、三大引擎横评与 `self-llm` 导流（端口 `8000`）

> **本节定位（Linux 大显存工作站 & 云算力并发王者！）**：如果你在企业招聘 JD 里看到“熟悉大模型推理部署”，90% 指的就是 **[`vllm-project/vllm`](https://github.com/vllm-project/vllm)**！本节带你在 ROCm 上跑通工业界标杆引擎 `vLLM`，完成本章 **三大标准引擎全景选型对比** 与 **作品集第 1 张测速卡**，并对接 Datawhale `self-llm` 专精课。

如果说 `llama-server` 是精致灵巧的“个人超跑”，那么 **`vLLM`** 就是能够同时拉着几十位乘客风驰电掣的 **“复兴号高铁”**：
- 它首创了著名的 **PagedAttention（分页注意力显存管理）** 与 **Continuous Batching（连续动态批处理）**——当你在第 4 章同时跑多个子智能体、在第 5 章批量给 200 条测试集打分、或给全实验室同学共享 API 时，`vLLM` 的并发吞吐量（Total `tok/s`）相比单线程引擎可飙升 **3～8 倍**！
- 同时，它也是第 5 章挂载 **LoRA 微调适配器（`--enable-lora`）** 与第 6 章加载 **AMD Quark `INT4-AWQ / FP8` 量化模型**、开启 **`AITER` 融合算子加速** 的主力舞台。

---

## Step 1：准备 `vLLM` ROCm 运行环境（Ubuntu 24.04 / 云算力推荐）

> [!NOTE]
> **系统提示**：`vLLM` 原生面向 Linux（Ubuntu 24.04）与云端服务器设计。如果你目前使用 Windows 11 个人电脑，日常单人办公与编程用前两节的 **`llama-server`（`8080`）** 或 **`lemonade-server`（`13305`）** 体验更轻快；想体验 `vLLM` 可使用 WSL2 或免费的 **[AMD Radeon Cloud](../cloud/amd-radeon-cloud.md)**！

在 Ubuntu 24.04 或 AMD Radeon Cloud 上，最省心、零环境冲突的启动方式是使用 **AMD 官方预构建的 `vLLM` ROCm 容器镜像**（或者在 `1.2` 节已装好 ROCm 10 PyTorch 的虚拟环境中运行）：

### 方式 A（最稳零冲突）：拉取 AMD 官方 `rocm/vllm` Docker 镜像

```bash
# 拉取 AMD ROCm 官方验证好的 vLLM 镜像（已内置 ROCm 算子库、FlashAttention 与 AITER 支持）
docker pull rocm/vllm:latest
```

### 方式 B（免 Docker 原生环境）：在云平台预置环境或本地 `.venv` 中确认 `vllm`

如果你在 AMD Radeon Cloud 选择了带 `vLLM` 的模板，直接在终端运行 `vllm --version` 即可看到版本号！

---

## Step 2：从魔搭（ModelScope）拉取原版或 AWQ/FP8 模型权重

`vLLM` 最擅长直接读取 HuggingFace / ModelScope 标准的 **`.safetensors`** 模型目录（而且从魔搭下载 `google/gemma-4-E4B-it` 或 `google/gemma-4-12B-it` **完全不需要去海外网站申请 Gated Token 审批**，国内直接满速拉取！）：

```bash
# 🔵 16G～24G 独显 / 云算力推荐主力：下载 Qwen3.5-9B（若显存仅 12G～16G，推荐下载 Qwen/Qwen3.5-9B-AWQ 4bit 版！）
modelscope download --model Qwen/Qwen3.5-9B --local_dir ~/ai_workspace/models/Qwen3.5-9B

# （可选）🟢 12G～16G 体验原版多模态小钢炮：Gemma 4 E4B
# modelscope download --model google/gemma-4-E4B-it --local_dir ~/ai_workspace/models/gemma-4-E4B-it

# （可选）🔵 24G 独显 / 云算力代码推理利器：Gemma 4 12B
# modelscope download --model google/gemma-4-12B-it --local_dir ~/ai_workspace/models/gemma-4-12B-it
```

---

## Step 3：在 `8000` 端口启动工业级 `vLLM` OpenAI 服务

### 使用 Docker 一键启动（挂载本地模型目录，暴露 `8000` 端口）

```bash
docker run -it --rm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 16G \
  -v ~/ai_workspace/models:/models \
  -e VLLM_ROCM_USE_AITER=1 \
  rocm/vllm:latest \
  vllm serve /models/Qwen3.5-9B \
    --served-model-name qwen3.5-9b \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.88 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

*(若你在已安装好 `vllm` 的云平台原生终端里运行，只需直接复制最后 `vllm serve ...` 那几行命令并把 `/models/Qwen3.5-9B` 指向 `~/ai_workspace/models/Qwen3.5-9B` 即可！)*

### 📖 面试常考的 5 个 `vLLM` 核心参数秒懂指南

1. **`--served-model-name qwen3.5-9b`**：对外暴露简洁统一的模型名（等价于 `llama-server` 的 `--alias`）；
2. **`--gpu-memory-utilization 0.88`**：告诉 `vLLM` 预先接管 **88%** 的空闲显存用来建立 PagedAttention 的 KV Cache 高速内存池（如果在个人桌面系统同时开着浏览器，建议调低到 **`0.80`** 防冲突）；
3. **`--max-model-len 32768`**：限制单次会话最大上下文长度为 `32K`（如果不限制，像 `Qwen3.5-9B` 默认想开 `262K` 超长窗口，16G/24G 卡启动时就会提示 KV Cache 预分配空间不够！）；
4. **`--enable-auto-tool-choice --tool-call-parser hermes`**：**Agent 灵魂开关**！让 `vLLM` 自动把模型生成的函数调用解析成标准 OpenAI `tool_calls` JSON 结构（Gemma 4 系列模型可将解析器设为对应支持的 parser）；
5. **`-e VLLM_ROCM_USE_AITER=1` & `--enable-prefix-caching`**：提前为你点亮 AMD 官方 **[`AITER`](https://github.com/ROCm/aiter)** 高性能融合算子库与前缀缓存（我们在第 6 章会专门实测它带来的翻倍加速威力！）。

---

## Step 4：并发威力验收！5 路请求同时轰炸测试

当终端出现 `Application startup complete.` 后，新开一个终端，用下面这段 20 行异步并发脚本同时向 `vLLM` 发出 **5 个并行提问**，感受什么叫工业级连续批处理（Continuous Batching）：

```bash
python -c "
import asyncio, time
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url='http://127.0.0.1:8000/v1', api_key='local-rocm')
prompts = [
    '用一句话解释什么是 PagedAttention？',
    '用一句话解释什么是 LoRA 微调？',
    '用一句话解释什么是 RAG 检索增强生成？',
    '用一句话解释什么是 MCP 模型上下文协议？',
    '用一句话解释为什么本地大模型保护隐私？',
]

async def ask(i, p):
    r = await client.chat.completions.create(
        model='qwen3.5-9b',
        messages=[{'role': 'user', 'content': p}],
        max_tokens=80,
    )
    print(f'✅ [并发请求 #{i+1}] {r.choices[0].message.content.strip()}')

async def main():
    t0 = time.perf_counter()
    await asyncio.gather(*(ask(i, p) for i, p in enumerate(prompts)))
    print(f'\n🚀 5 路并发请求全部完成！总墙钟耗时仅 {time.perf_counter()-t0:.2f} 秒！')

asyncio.run(main())
"
```

你会惊喜地发现：**同时回答 5 个问题花的总时间，几乎和单独回答 1 个问题差不多快！** 这就是 `vLLM` 在团队共享与多智能体并发场景下不可替代的原因。

---

## 📊 本章总复盘：三大标准推理引擎一表选型决策

学到这里，你已经完整认识了 `hello-rocm 2.0` 的三大标准动力源。今后在任何项目里，按这张表 3 秒钟就能做出最专业的架构选型：

| 对比维度 | 🟢 `llama-server` (`:8080`) | 🟡 `lemonade-server` (`:13305`) | 🔵 `vLLM` (`:8000`) |
|:---|:---|:---|:---|
| **安装复杂度** | ⭐ 极简（单压缩包解压即跑） | ⭐ 极简（Win 安装包 / Ubuntu PPA） | ⭐⭐⭐ 中等（需 Linux / Docker + 大显存） |
| **最佳操作系统** | Windows 11 / Ubuntu / macOS 全通 | Windows 11 / Ubuntu（尤其锐龙 AI PC） | Ubuntu 24.04 / AMD Radeon Cloud |
| **支持硬件范围** | 所有 AMD 独显、核显（ROCm / Vulkan） | Radeon GPU + **独家支持 XDNA 2 NPU** | RDNA 3 / 3.5 / 4 及 Instinct 计算卡 |
| **模型文件格式** | `.gguf`（单文件极易搬运分享） | `.gguf` + `FLM` NPU 专用格式 | `.safetensors`（原版 / Quark AWQ / FP8） |
| **单人首字与功耗** | **极快、显存极省** | **极省电（NPU 后台值守神器）** | 常驻显存占用高（预分配 KV 池） |
| **多人/多 Agent 并发** | 轻量队列（适合 1～2 并发） | 轻量个人桌面级 | **👑 工业级霸主（数十并发高吞吐）** |
| **本书后续推荐场景** | **第 2～4 章个人日常办公与 IDE 首选** | **锐龙 AI 本 & Max+ 395 跑 35B 首选** | **第 5 章评测、第 6 章 Quark/AITER 压测** |

---

## 🏆 第 1 章通关成果：填写你的《作品集素材 #1：本机部署与测速记录卡》

恭喜你正式解锁 **`个人学习助手 v0.1`**！请顺手把刚才实测出来的真实数据抄在你的学习笔记里（第 7 章写简历和作品集时直接拿来用，真实数字最具说服力！）：

```markdown
### 📋 我的 AMD 本地大模型首航测速卡
- **运行硬件 / 云实例**：________________________（可用显存/统一内存：____ GB，架构：`gfx____`）
- **操作系统与引擎**：[ ] llama-server (:8080)  [ ] lemonade-server (:13305)  [ ] vLLM (:8000)
- **点亮的主体模型**：________________________（量化规格：________，上下文长度：`32768`）
- **实测显存峰值占用**：____.__ GB（空载显存：____.__ GB）
- **首字响应延迟（TTFT）**：______ ms ｜ **生成吐字速度**：____.__ tok/s
```

---

## 🔗 本章课程引用与进阶导流：直通 `self-llm`

在本章中，我们聚焦于在 AMD 平台上跑通 **4 款最具代表性的主体大模型** 与 **三大标准 OpenAI 服务引擎**。

如果你在点亮第一颗模型后意犹未尽，想要进一步探索：
- **百花齐放的百款开源模型生态**（如 DeepSeek 全系列、InternLM 书生·浦语、GLM-4、MiniCPM-V/o 视觉语音全双工多模态模型、嵌入向量与重排模型的逐一部署细节）；
- **不同模型家族各自特有的推理参数、WebDemo 界面搭建与 FastAPI 封装技巧**；

欢迎随时拓展阅读 Datawhale 累计破万 Star 的王牌开源教程 —— **[`datawhalechina/self-llm`（《开源大模型食用指南》）](https://github.com/datawhalechina/self-llm)**！你在本章打下的 ROCm 10 环境与三大引擎功底，可以让你无缝复现 `self-llm` 中的绝大多数开源模型教程！

---

## 👉 下一站：零代码收获第一波职场与学习生产力暴击！

现在，请**保持你刚刚启动的 `8080`（或 `13305` / `8000`）服务窗口不要关**！  
接下来我们**一行代码都不用写**，直接把这个热乎的本地大脑插进桌面神器 **WorkBuddy**，亲手用 **`dashi-ppt-skill`（大师 PPT）** 3 分钟生成一份高颜值可编辑 PPT、自动汇总报销 Excel，并配上每天自动干活的 AI 定时任务：

➡️ **[进入第 2 章｜零代码办公先行：接入 WorkBuddy（大师 PPT / 自动填表 / 定时任务）与 `Qoder` 编程助手](../02-connect-tools/index.md)**
