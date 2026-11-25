# 7.2 个人 AI 作品集「五件套」标准仓库打包指南

> **本节目标**：把你在前 6 章产出的全部代码、配置、曲线图与评测卡，整理成一个让技术面试官点开 **30 秒内就忍不住给你打高分** 的独立 GitHub / Gitee 置顶作品集仓库 —— **`my-amd-ai-assistant` 标准五件套**！

为什么很多同学简历上写了 GitHub 链接，面试官点进去看一眼却直接关掉了？  
因为大多数新手仓库要么只有乱七八糟未整理的实验草稿，要么直接 Fork 了一个知名大项目却没写清楚**哪一行是自己真正动手做的**，甚至连一张运行截图和评测表都没有！

下面这套经过大量一线技术面试官验证的 **「个人 AI 作品集五件套」标准结构**，能全方位证明你的 **工程落地力、科学评测观与代码审美**。

---

## 一、一目了然的「五件套」黄金目录结构

建议在你的个人 GitHub 主页新建一个公开仓库（例如命名为 `amd-local-ai-assistant` 并设为 **Pinned 置顶仓库**），目录按以下 5 大模块清爽排布：

```text
amd-local-ai-assistant/
├── README.md                        # 🌟 门面担当：架构图 + 60秒演示GIF + 核心评测涨分表置顶！
├── 01_configs_and_deploy/           # 【第①件：服务与多端配置】
│   ├── start_engine.sh              #   你验证调优后的 llama-server / lemonade / vLLM 启动脚本
│   ├── workbuddy-models.json        #   脱敏后的 WorkBuddy / Qoder / Codex 接入配置
│   └── env.example                  #   环境变量示例（严禁提交真实含密钥的 .env！）
├── 02_mcp_and_skills/               # 【第①件续：智能体工具与技能源码】
│   ├── knowledge_mcp_server.py      #   你的可共享「知识管理 MCP」服务端（含你自己的扩展函数！）
│   ├── test_mcp_server.py           #   配套边界与防路径穿越单元测试
│   └── skills/                      #   你编写的自定义办公/学习 SKILL.md
├── 03_finetune_zh/                  # 【第③件：中文微调配方与训练实录】
│   ├── prepare_and_desensitize.py   #   中文语料正则隐私脱敏与训练/测试集切分脚本
│   ├── lora_sft_config.yaml         #   可一键复现的 LLaMA-Factory 训练配置
│   └── swanlab_loss_curve.png       #   导出的真实 SwanLab 训练 Loss 收敛曲线图
├── 04_eval_and_benchmarks/          # 【第④件：客观效果评测与量化加速报告（含金量核心！）】
│   ├── eval_before_after_zh.py      #   客观评测脚本与脱敏后的小规模测试样例集
│   ├── quark_quant_bench.py         #   AMD Quark 量化与前后对比脚本
│   └── BENCHMARK_REPORT.md          #   汇总第1/3/5/6章四张真实测试数据表的总报告！
├── assets/                          # 【第②件：高光交互视觉实录】
│   ├── demo_workflow.gif            #   60 秒端到端丝滑录屏动图
│   └── architecture.png             #   系统分层架构图
└── CONTRIBUTING_NOTES.md            # 【第⑤件：个人独立贡献与工程边界声明（极度加分！）】
```

### ⚡ 10 秒钟一键生成你的作品集骨架目录

在终端复制运行下面这段初始化命令，瞬间把你前几章的成果归位：

```bash
PORTFOLIO=~/ai_workspace/amd-local-ai-assistant
mkdir -p "$PORTFOLIO"/{01_configs_and_deploy,02_mcp_and_skills/skills,03_finetune_zh,04_eval_and_benchmarks,assets}
echo -e ".env\n__pycache__/\n*.gguf\n*.safetensors\nsaves/\n" > "$PORTFOLIO/.gitignore"
echo "✅ 作品集仓库骨架与安全 .gitignore 创建完成：$PORTFOLIO"
```

---

## 二、逐件打磨秘籍：怎样让每一件都透出专业感？

### 🎬 第 ② 件点睛：用 60 秒录出让面试官过目不忘的 `demo_workflow.gif`！

在仓库 `README.md` 标题正下方放一张 **45～60 秒的高清动图（GIF）或短视频**，胜过三千字苍白描述！推荐直接按下面这个 **「黄金三镜头分镜脚本」** 录屏（Windows 可用免费的 `ScreenToGif`，Linux/macOS 可用 `OBS` 或 `peek`）：
- **镜头 1（0～15 秒 · 算力底座亮剑）**：屏幕左半边展示 `amd-smi monitor`（或任务管理器 AMD GPU 显存稳稳跑在本地），右半边展示终端里本地 `/v1` 服务在线；
- **镜头 2（15～40 秒 · 跨软件知识库 MCP 联动）**：在 `Qoder` 或 WorkBuddy 里发出一句指令，清晰拍到模型自动调用 `search_notes` 查出你的私人笔记并带 `[[双链角标]]` 回答，紧接着自动调用 `create_note_card` 把总结写回 Obsidian 图谱；
- **镜头 3（40～60 秒 · 一键交付办公成果）**：调用 `dashi-ppt-skill` 弹开预览控制台，双击打开导出的 `.pptx` 随便改动一行标题文字，证明 100% 原生可编辑！

---

### 📊 第 ④ 件核心：在 `BENCHMARK_REPORT.md` 集齐你的「四张硬核实测表」

把你在前面四个章节亲手测出来的真实数字汇总进同一份报告里，形成无懈可击的证据链：
1. **表 1（来自第 1.5 节）**：本机硬件环境与三大引擎首航基准测速卡（显存占用 / `TTFT` / `tok/s`）；
2. **表 2（来自第 3.4 节）**：私人笔记库「6+2+2」10 道开卷防幻觉测试得分表（尤其突出超纲陷阱题 **100% 诚实拒答**）；
3. **表 3（来自第 5.4 节）**：中文群通知 JSON 抽取 LoRA 微调前后 Hold-out 测试集准确率跃升表（或 `WeClone`/小红书客服破功词清零对比表）；
4. **表 4（来自第 6.2 & 6.4 节）**：单模型 **`AMD Quark` 量化前后体积/显存/速度对比表** + 开启 **Prefix Caching** 前后次轮 `TTFT` 对比表！

---

### 🤝 第 ⑤ 件王牌：写好 `CONTRIBUTING_NOTES.md`（诚实是最顶级的聪明！）

很多新手怕面试官发现自己跟着开源教程做过项目，反而遮遮掩掩；其实资深面试官一眼就能看出底座来源——**真正让面试官刮目相看的，是你大方、清晰地划清「开源底座复用」与「本人独立工程增量」的边界！**

直接在 `CONTRIBUTING_NOTES.md` 中参考以下模板撰写：

```markdown
# 项目架构演进与个人独立工程贡献说明

本项目底层推理引擎基于开源 `llama.cpp` / `lemonade` / `vLLM` 构建，学习路径受 Datawhale `hello-rocm 2.0` 启发。在此基础之上，本人独立完成了以下垂直场景定制与评测闭环：

1. **私有知识库重构与 MCP 扩展开发**：
   - 针对本人【填写你的专业/业务，如：嵌入式课程实验与排障记录】整理了 XX 篇结构化双链笔记；
   - 在基础 `FastMCP` 服务之上，本人独立新增了 `【填写你扩展的功能，如：按日期范围统计复习卡片 / 自动提取错题标签】` 工具函数，并补充了对应的边界单元测试。
2. **自有场景中文数据集构造与微调对比**：
   - 独立收集并编写正则脱敏脚本清洗了 XXX 条【校园教务通知 / 社团财务单据 / 个人聊天】语料，构建了包含 XX 条样本的物理隔离测试集；
   - 独立对比了 `r=8` 与 `r=16` 两组超参在 `SwanLab` 上的收敛表现，定位并修复了缺失字段误填问题。
3. **本机硬件全链路压测与调优落地**：
   - 在本人【具体显卡型号】上独立完成了 `AMD Quark` 量化前后及 Prefix Caching 开启前后的控制变量压测，产出全部实测数据图表。
```

> [!TIP]
> **点睛建议**：花 20 分钟在 `knowledge_mcp_server.py` 里多写 **1 个属于你自己的小工具函数**（比如 `get_recent_notes(days=7)` 查看最近一周新写的笔记），再把数据集换成你身边真实的专业领域文本——你的项目立刻就从“课程作业”升华成了**全世界独一无二的原创工程作品**！

---

## 👉 下一节：把这五件套浓缩成简历黄金 5 行字，稳稳拿下技术面试！

仓库打包得漂漂亮亮之后，在寸土寸金的一页纸简历上该怎么写？面试官看到这段经历后最爱连环追问哪 6 个刁钻问题？

马上进入本章压轴宝典：**[7.3 STAR 法则高分简历项目模板、6 道高频面试追问满分拆解与毕业深造驿站](./03-star-resume-interview.md)**！
