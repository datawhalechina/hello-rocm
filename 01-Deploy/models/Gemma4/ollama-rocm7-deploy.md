## Ollama 零基础环境部署（Ubuntu 24.04 + ROCm 7+）

本节介绍在 Ubuntu 24.04 + ROCm 7+ 环境下，如何安装和使用 **Ollama（ROCm 版 llama.cpp 后端）**，并以 Qwen3-8B Q4_K_M 为例进行性能测试。

> 前置条件：已经完成 ROCm 7.1.0 环境准备（参考 `env-prepare-ubuntu24-rocm7.md`）。

---

### 1. 安装 Ollama（系统服务）

第一步：一键安装到 `systemctl` 中管理，会占用本地 `11434` 端口启动服务。  
更多信息参考官方文档：https://docs.ollama.com/linux

安装命令：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

### 2. 验证服务连通性

安装完成后，可以用以下命令验证服务是否正常运行：

```bash
curl http://localhost:11434
```

若返回 JSON 信息（如版本号等），说明服务启动成功。

---

### 3. 基本使用命令

常用基础命令如下：

```bash
# 列出所有模型
ollama list

# 下载模型
ollama pull qwen3:8b-q4_K_M

# 测试模型运行（交互式）
ollama run qwen3:8b-q4_K_M
```

---

### 4. 使用 curl 测速（计算 tokens/s）

下面的命令示例将调用 Ollama 的 REST 接口，并使用 `jq` 解析返回中的评估信息，计算推理速度：

```bash
curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
  "model": "qwen3:8b-q4_K_M",
  "prompt": "用一句话解释什么是大语言模型",
  "stream": false
}' | jq '.eval_count, .eval_duration' | \
awk 'NR==1{count=$1} NR==2{duration=$1/1e9} END{printf "tokens/s: %.2f\n", count/duration}'
```

- `eval_count`：推理产生的 token 数量  
- `eval_duration`：推理耗时（单位：纳秒）  
- `tokens/s`：通过 `count / (duration / 1e9)` 计算得到

---

### 5. Qwen3-8B Q4_K_M 性能示例

在上述环境中，对 **Qwen3-8B Q4_K_M** 模型进行测试（上下文长度 4096）得到示例结果：

- **约 37.50 tokens/s**

截图示例：

<img src="./images/media/image6.png" style="width:5.75in;height:3.25in" />


