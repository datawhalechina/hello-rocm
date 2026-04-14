**Ubuntu24.04(Linux) 部署支持 rocm7+ 的推理框架部署指南**

**一、删除目前环境中所有 rocm 相关软件**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">PowerShell<br />
shell<br />
sudo apt remove rocm*<br />
sudo apt remove amd*</td>
</tr>
</tbody>
</table>

**二、运行脚本安装 rocm7.1.0 到系统中**

|  |
|:---|
| 如果使用docker，需要安装 amdgpu-dkms：https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html 下方脚本以包含，如果不执行下方脚本安装，需要自己手动安装 |

下载安装脚本到本地：https://github.com/amdjiahangpan/rocm-install-script/blob/ROCm_7.1.0_ubuntu_24.04/2-install.sh

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">PowerShell<br />
shell<br />
sudo apt update<br />
# 更新内核<br />
sudo apt upgrade -y<br />
sudo bash 2-install.sh<br />
# 验证命令，均有显卡内容输出<br />
rocminfo<br />
rocm-smi<br />
amd-smi</td>
</tr>
</tbody>
</table>

|  |
|:---|
| 参考指导：https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html |

**三、安装推理框架并调用rocm推理**

1\. **使用 lm-studio (选择 rocm 版本 llama.cpp 后端推理)**

首先需要从官方 https://lmstudio.ai/ 下载最新的APPImage文件

<img src="./images/media/image1.png"
style="width:5.75in;height:2.27083in" />

提取AppImage内容解压到 squashfs-root 目录中：

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">PowerShell<br />
shell<br />
chmod u+x LM-Studio-*.AppImage<br />
./LM-Studio-*.AppImage --appimage-extract</td>
</tr>
</tbody>
</table>

进入squashfs-root目录中并为chrome-sandbox文件设置适当的权限，该文件是应用程序安全的运行所需二进制文件

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">PowerShell<br />
cd squashfs-root<br />
sudo chown root:root chrome-sandbox<br />
sudo chmod 4755 chrome-sandbox</td>
</tr>
</tbody>
</table>

在当前文件夹下启动 LM Studio 应用程序：

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">PowerShell<br />
shell<br />
./lm-studio</td>
</tr>
</tbody>
</table>

安装rocm 版本 llama.cpp 后端推理

<img src="./images/media/image2.png"
style="width:5.75in;height:3.8125in" />

需要注意目前lm-studio提供的rocm版本llama.cpp支持的架构列表

<img src="./images/media/image3.png"
style="width:5.75in;height:3.8125in" />

<img src="./images/media/image4.png"
style="width:5.75in;height:3.8125in" />

测试 Qwen3-8B 量化 Q4_K_M 上下文4096 -\>36 token/s

<img src="./images/media/image5.png"
style="width:5.75in;height:3.8125in" />

2\. **使用 Ollma (选择 rocm 版本 llama.cpp 后端推理)**

第一步骤：一键安装到 systemctl 中管理，会占用本地 11434
端口启动服务，更多信息参考官方 https://docs.ollama.com/linux

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
shell<br />
curl -fsSL https://ollama.com/install.sh | sh</td>
</tr>
</tbody>
</table>

验证连通性：

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
shell<br />
curl http://localhost:11434</td>
</tr>
</tbody>
</table>

基本命令

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">C++<br />
shell<br />
# 列出所有模型<br />
ollama list<br />
# 下载模型<br />
ollama pull qwen3:8b-q4_K_M<br />
# 测试模型运行<br />
ollama run qwen3:8b-q4_K_M<br />
# curl 测速<br />
curl -s -X POST http://localhost:11434/api/generate \<br />
-H "Content-Type: application/json" \<br />
-d '{<br />
"model": "qwen3:8b-q4_K_M",<br />
"prompt": "用一句话解释什么是大语言模型",<br />
"stream": false<br />
}' | jq '.eval_count, .eval_duration' | \<br />
awk 'NR==1{count=$1} NR==2{duration=$1/1e9} END{printf "tokens/s:
%.2f\n", count/duration}'</td>
</tr>
</tbody>
</table>

测试 Qwen3-8B 量化 Q4_K_M 上下文4096 -\>37.50 token/s

<img src="./images/media/image6.png"
style="width:5.75in;height:3.25in" />

3\. **使用 llama.cpp**

**方式一 （推荐）: 预构建的可执行文件**

使用lenmonade提供的预构建版本，其中 **370** 为 **gfx1150** 架构，**395**
为 **gfx1151** 架构

https://github.com/lemonade-sdk/llamacpp-rocm

https://github.com/lemonade-sdk/llamacpp-rocm/releases

<img src="./images/media/image7.png"
style="width:5.75in;height:5.10417in" />

**1️⃣ 确认 ROCm 7+ 已安装（必须安装系统版本的 ROCm ）**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
amd-smi</td>
</tr>
</tbody>
</table>

<img src="./images/media/image8.png"
style="width:5.75in;height:4.01042in" />

可以看到 GPU 型号、驱动、ROCm 版本

如果没问题，就可以用 GPU 推理

**2️⃣ 进入 llama 后端目录并设置权限**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
cd llama-*x64/<br />
sudo chmod +x *<br />
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH</td>
</tr>
</tbody>
</table>

**3️⃣ 下载 Qwen3 8B Q4_K_M 模型**

Ollama / llama-server 通常使用 **GGUF 模型格式**。

使用国内 huggingface 镜像 https://hf-mirror.com/ 需要登陆
[huggingface](https://huggingface.co/) 获取 用户名 和 token\
你可以用官方方式下载：

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Markdown<br />
shell<br />
# 创建模型存放目录<br />
mkdir -p ~/models<br />
cd ~/models<br />
<br />
# 下载 GGUF 文件（示例，需替换成官方/开源下载地址）<br />
wget https://hf-mirror.com/hfd/hfd.sh<br />
chmod a+x hfd.sh<br />
export HF_ENDPOINT=https://hf-mirror.com<br />
sudo apt update<br />
sudo apt install aria2<br />
# 需要登陆 huggingface 获取你的 用户名 和 token<br />
./hfd.sh netrunnerllm/Qwen3-8B-Q4_K_M-GGUF --hf_username
&lt;USERNAME&gt; --hf_token hf_***</td>
</tr>
</tbody>
</table>

**4️⃣ 启动服务器**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
shell<br />
# rocm 驱动库链接<br />
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH<br />
cd llama-*x64/<br />
./llama-server -m ~/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf
-ngl 99</td>
</tr>
</tbody>
</table>

**5️⃣ 测试接口**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
shell<br />
curl -s -X POST http://127.0.0.1:8080/v1/completions \<br />
-H "Content-Type: application/json" \<br />
-d '{<br />
"model": "qwen3-8b-q4_k_m",<br />
"prompt": "用一句话解释大语言模型",<br />
"max_tokens": 128<br />
}' | jq -r '<br />
# 打印生成文本<br />
.choices[0].text as $txt |<br />
# 计算 token/s<br />
(.usage.completion_tokens / (.timings.predicted_ms / 1000)) as $tps
|<br />
"生成文本:\n\($txt)\n\ntokens/s: \($tps|tostring)"<br />
'</td>
</tr>
</tbody>
</table>

<img src="./images/media/image9.png"
style="width:5.75in;height:3.25in" />

测试 Qwen3-8B 量化 Q4_K_M 上下文 4096 -\> 40.71 token/s

**方式二 Docker：**

如果使用docker，需要安装
amdgpu-dkms：https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html
第二步骤脚本已包含，如果不执行脚本安装，需要自己手动安装

Docker
参考：https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/llama-cpp-install.html

**1️⃣ 下载容器**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
shell<br />
export MODEL_PATH='~/models'<br />
<br />
<br />
sudo docker run -it \<br />
--name=$(whoami)_llamacpp \<br />
--privileged --network=host \<br />
--device=/dev/kfd --device=/dev/dri \<br />
--group-add video --cap-add=SYS_PTRACE \<br />
--security-opt seccomp=unconfined \<br />
--ipc=host --shm-size 16G \<br />
-v $MODEL_PATH:/data \<br />
rocm/dev-ubuntu-24.04:7.0-complete</td>
</tr>
</tbody>
</table>

**2️⃣ 容器内，设置你的工作区：**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
shell<br />
apt-get update &amp;&amp; apt-get install -y nano libcurl4-openssl-dev
cmake git<br />
mkdir -p /workspace &amp;&amp; cd /workspace</td>
</tr>
</tbody>
</table>

**3️⃣ 容器内 克隆
[<u>ROCm/llama.cpp</u>](https://github.com/ROCm/llama.cpp) 仓库：**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
shell <br />
git clone https://github.com/ROCm/llama.cpp<br />
cd llama.cpp</td>
</tr>
</tbody>
</table>

**4️⃣ 设定你的 ROCm 架构，以 AI MAX 395 为例：**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
shell<br />
export LLAMACPP_ROCM_ARCH=gfx1151</td>
</tr>
</tbody>
</table>

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;"><p>要为多种微架构编译，运行：</p>
<table style="width:86%;">
<colgroup>
<col style="width: 85%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
shell <br />
export
LLAMACPP_ROCM_ARCH=gfx803,gfx900,gfx906,gfx908,gfx90a,gfx942,gfx1010,gfx1030,gfx1032,gfx1100,gfx1101,gfx1102,gfx1150,gfx1151</td>
</tr>
</tbody>
</table></td>
</tr>
</tbody>
</table>

**5️⃣ 编译与安装 llama.cpp：**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
shell<br />
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B
build -DGGML_HIP=ON -DAMDGPU_TARGETS=$LLAMACPP_ROCM_ARCH
-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON &amp;&amp; cmake --build
build --config Release -j$(nproc)</td>
</tr>
</tbody>
</table>

测试安装

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
shell<br />
cd /workspace/llama.cpp<br />
./build/bin/test-backend-ops</td>
</tr>
</tbody>
</table>

**运行测试 Qwen3-8B 量化 Q4_K_M 上下文4096 -\>39.60 token/s：**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Plain Text<br />
./build/bin/llama-cli -m /data/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf
-ngl 99</td>
</tr>
</tbody>
</table>

<img src="./images/media/image10.png"
style="width:5.75in;height:3.25in" />

4\. **使用 vllm**

**方式一 Docker：**

参考官方链接：https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

如果使用docker，需要安装
amdgpu-dkms：https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html
第二步骤脚本已包含，如果不执行脚本安装，需要自己手动安装

**启动容器**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
shell<br />
sudo docker pull rocm/vllm-dev:nightly # to get the latest image<br />
sudo docker run -it --rm \<br />
--network=host \<br />
--cpus="16" \<br />
--group-add=video \<br />
--ipc=host \<br />
--cap-add=SYS_PTRACE \<br />
--security-opt seccomp=unconfined \<br />
--device /dev/kfd \<br />
--device /dev/dri \<br />
-v ~/models:/app/models \<br />
-e HF_HOME="/app/models" \<br />
rocm/vllm-dev:nightly</td>
</tr>
</tbody>
</table>

**容器内启动模型：**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
# 在容器内运行<br />
vllm serve /app/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf --dtype
float16 --max-model-len 4096 --max-num-seqs 32 --tokenizer
qwen/Qwen3-8B<br />
# 快速启动 --enforce-eager：禁用 CUDA
graph，启动速度极快，但推理速度会稍微慢一点（通常慢 10-20%）。对于测试
GGUF 来说，这通常是值得的。<br />
vllm serve /app/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf
--tokenizer qwen/Qwen3-8B --dtype float16 --enforce-eager --max-num-seqs
32 --max-model-len 4096</td>
</tr>
</tbody>
</table>

**运行测试 Qwen3-8B 量化 Q4_K_M 上下文 4096 -\>30.21 token/s：**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">SQL<br />
# 1. 准备随机 Prompt<br />
RAND_PROMPT="随机码$(date +%N):
请详细介绍量子计算的未来，要求内容丰富，不要重复。"<br />
<br />
# 2. 记录精确开始时间<br />
start=$(date +%s.%N)<br />
<br />
# 3. 发起请求并存入变量<br />
response=$(curl -s -X POST http://127.0.0.1:8000/v1/completions \<br />
-H "Content-Type: application/json" \<br />
-d "{<br />
\"model\":
\"/app/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf\",<br />
\"prompt\": \"$RAND_PROMPT\",<br />
\"max_tokens\": 512,<br />
\"temperature\": 0.8<br />
}")<br />
<br />
# 4. 记录结束时间<br />
end=$(date +%s.%N)<br />
<br />
# 5. 解析内容<br />
# 提取生成文本原始内容<br />
content=$(echo "$response" | jq -r '.choices[0].text')<br />
# 提取 Token 数<br />
tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')<br />
# 计算耗时<br />
duration=$(echo "$end - $start" | bc)<br />
<br />
# 6. 打印输出<br />
echo "==================== 原始内容 ===================="<br />
echo "$content"<br />
echo "=================================================="<br />
<br />
if (( $(echo "$duration &lt; 0.05" | bc -l) )); then<br />
echo "检测到异常极速响应 ($duration 秒)，可能命中了缓存。"<br />
else<br />
tps=$(echo "scale=2; $tokens / $duration" | bc)<br />
echo "生成 Token 数: $tokens"<br />
echo "实际总耗时: $duration 秒"<br />
echo "真实推理速度: $tps tokens/s"<br />
fi<br />
echo "=================================================="</td>
</tr>
</tbody>
</table>

开启 --enforce-eager

<img src="./images/media/image11.png"
style="width:5.75in;height:3.25in" />

关闭

<img src="./images/media/image12.png"
style="width:5.75in;height:3.25in" />

**方式二 手动构建（AI MAX 395 为例）：**

参考官方链接：https://docs.vllm.ai/en/latest/getting_started/quickstart/#installation

VLLM 0.13.0

GPU: MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900
series (gfx1100/1101), Radeon RX 9000 series (gfx1200/1201), Ryzen AI
MAX / AI 300 Series (gfx1151/1150)

ROCm 6.3 or above

MI350 requires ROCm 7.0 or above

Ryzen AI MAX / AI 300 Series requires ROCm 7.0.2 or above

**使用 uv 构建虚拟环境**

uv相关参考：https://www.runoob.com/python3/uv-tutorial.html

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
shell<br />
# 安装 uv<br />
curl -LsSf https://astral.sh/uv/install.sh | sh<br />
# 创建并激活虚拟环境<br />
uv venv --python 3.12 --seed<br />
source .venv/bin/activate</td>
</tr>
</tbody>
</table>

**安装rocm7+支持的torch**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
# Install PyTorch<br />
uv pip uninstall torch<br />
uv pip install --no-cache-dir torch torchvision --index-url
https://download.pytorch.org/whl/nightly/rocm7.0</td>
</tr>
</tbody>
</table>

**为 [ROCm 安装 Triton](https://github.com/ROCm/triton.git)**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
shell<br />
uv pip install ninja cmake wheel pybind11<br />
uv pip uninstall triton<br />
git clone https://github.com/ROCm/triton.git<br />
cd triton<br />
# git checkout $TRITON_BRANCH<br />
git checkout f9e5bf54<br />
# 启动 16 核并行编译，并显示实时输出<br />
# --no-build-isolation 确保使用你当前环境已有的 ninja, cmake<br />
if [ ! -f setup.py ]; then cd python; fi<br />
MAX_JOBS=16 uv pip install --no-build-isolation -e .<br />
cd ..</td>
</tr>
</tbody>
</table>

下载triton各种编译依赖文件 10+ GB（Preparing
packages...）需要开启代理，下载时间大概需要 3小时 起步

编译大概5分钟

**构建 flash_attention**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
git clone https://github.com/ROCm/flash-attention.git<br />
cd flash-attention<br />
git checkout origin/main_perf<br />
git submodule update --init<br />
<br />
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"<br />
export TORCH_USE_HIP_DSA=1<br />
python setup.py bdist_wheel --dist-dir=dist<br />
uv pip install dist/*.whl</td>
</tr>
</tbody>
</table>

编译完后进入 benchmarks 目录跑测试：

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"<br />
export TORCH_USE_HIP_DSA=1<br />
<br />
cd benchmarks/<br />
python benchmark_flash_attention.py</td>
</tr>
</tbody>
</table>

**构建vllm-rocm7.0+**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
shell<br />
# 1. 更新 uv 自身 (可选，uv 建议保持最新)<br />
uv self update<br />
<br />
# 2. 安装 AMD SMI (直接指向本地路径，uv 同样支持) # uv pip install
/opt/rocm/share/amd_smi<br />
# 权限问题<br />
# 1. 拷贝目录<br />
cp -r /opt/rocm/share/amd_smi ./amdsmi_src<br />
# 2. 进入拷贝后的目录<br />
cd ./amdsmi_src<br />
# 3. 使用 uv 安装<br />
uv pip install .<br />
cd ..<br />
<br />
<br />
# 3. 安装核心编译依赖<br />
uv pip install --upgrade \<br />
numba \<br />
scipy \<br />
"huggingface-hub[cli,hf_transfer]" \<br />
setuptools_scm<br />
<br />
git clone https://github.com/vllm-project/vllm.git<br />
cd vllm<br />
# 4. 安装 vLLM ROCm 专用依赖项<br />
uv pip install -r requirements/rocm.txt<br />
uv pip install numpy setuptools wheel<br />
<br />
# 5. 设置针对 gfx1151 的架构变量<br />
export VLLM_TARGET_DEVICE="rocm"<br />
export PYTORCH_ROCM_ARCH="gfx1151"<br />
export ROCM_HOME="/opt/rocm"<br />
<br />
# 6. 以可编辑模式编译并安装 vLLM<br />
MAX_JOBS=16 uv pip install -e . --no-build-isolation</td>
</tr>
</tbody>
</table>

**虚拟环境中启动模型：**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
# 启用 Flash_attention<br />
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"<br />
export TORCH_USE_HIP_DSA=1<br />
# 在容器内运行<br />
vllm serve ~/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf --dtype
float16 --max-model-len 4096 --max-num-seqs 32 --tokenizer
qwen/Qwen3-8B<br />
# 快速启动 --enforce-eager：禁用 CUDA
graph，启动速度极快，但推理速度会稍微慢一点（通常慢 10-20%）。对于测试
GGUF 来说，这通常是值得的。<br />
vllm serve ~/models/Qwen3-8B-Q4_K_M-GGUF/qwen3-8b-q4_k_m.gguf
--tokenizer qwen/Qwen3-8B --dtype float16 --enforce-eager --max-num-seqs
32 --max-model-len 4096</td>
</tr>
</tbody>
</table>

**运行测试 Qwen3-8B 量化 Q4_K_M 上下文 4096 -\> 29.79 token/s：**

<table style="width:88%;">
<colgroup>
<col style="width: 88%" />
</colgroup>
<tbody>
<tr>
<td style="text-align: left;">Bash<br />
# 1. 自动获取模型 ID (防止 404)<br />
MODEL_ID=$(curl -s http://127.0.0.1:8000/v1/models | jq -r
'.data[0].id')<br />
echo "探测到模型 ID: $MODEL_ID"<br />
<br />
# 2. 准备随机 Prompt<br />
RAND_PROMPT="随机码$(date +%N):
请详细介绍量子计算的未来，不少于500字。"<br />
<br />
# 3. 记录精确开始时间<br />
start=$(date +%s.%N)<br />
<br />
# 4. 发起请求<br />
response=$(curl -s -X POST http://127.0.0.1:8000/v1/completions \<br />
-H "Content-Type: application/json" \<br />
-d "{<br />
\"model\": \"$MODEL_ID\",<br />
\"prompt\": \"$RAND_PROMPT\",<br />
\"max_tokens\": 512,<br />
\"temperature\": 0.8<br />
}")<br />
<br />
# 5. 记录结束时间<br />
end=$(date +%s.%N)<br />
<br />
# 6. 解析内容<br />
content=$(echo "$response" | jq -r '.choices[0].text // "错误:
未获取到文本内容，请检查输出"')<br />
tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')<br />
duration=$(echo "$end - $start" | bc)<br />
<br />
# 7. 打印输出<br />
echo "==================== 原始内容 ===================="<br />
echo "$content"<br />
echo "=================================================="<br />
<br />
if (( $(echo "$duration &lt; 0.05" | bc -l) )); then<br />
echo "响应过快 ($duration 秒)，可能是 404 报错或缓存。"<br />
echo "完整响应体: $response"<br />
else<br />
tps=$(echo "scale=2; $tokens / $duration" | bc)<br />
echo "生成 Token 数: $tokens"<br />
echo "实际总耗时: $duration 秒"<br />
echo "真实推理速度: $tps tokens/s"<br />
fi<br />
echo "=================================================="</td>
</tr>
</tbody>
</table>

未编译 Flash_attention 使用 --enforce-eager

<img src="./images/media/image13.png"
style="width:5.75in;height:3.25in" />

编译 Flash_attention 后 --enforce-eager

<img src="./images/media/image14.png"
style="width:5.75in;height:3.25in" />

编译 Flash_attention 后

<img src="./images/media/image15.png"
style="width:5.75in;height:3.25in" />
