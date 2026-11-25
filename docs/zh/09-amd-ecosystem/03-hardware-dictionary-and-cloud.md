# 9.3 自包含权威速查百宝箱：全系 AMD `gfx` 硬件大字典、云算力与开源共建

> **本节定位（永久收藏级速查页）**：将散落在各处的 **AMD 全产品线（桌面独显 / 锐龙 AI 笔记本核显与 NPU / 工作站专业卡 / Instinct 数据中心卡）`gfx` 架构编号、ROCm 10 `pip` 安装标识、免费云算力申领通道与全书配置模板索引** 浓缩成一页自包含工具字典，方便你今后装新机、查参数时 10 秒钟一查即得！

---

## 一、📋 全系 AMD GPU / APU 架构代号与 ROCm 10 `pip` 索引大字典

在使用 ROCm 10 模块化安装命令（`uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfxXXXX]==2.13.0+rocm10.0.0"`）时，请直接在下表查找你的精确硬件型号：

### 1. 消费级桌面独显与专业工作站显卡（RDNA 4 / RDNA 3 / RDNA 2）

| 架构代际 | 代表显卡具体型号 | 显存规格 | `gfx` Target 代号 | ROCm 10 PyTorch Wheel Extra 写法与部署建议 |
|:---|:---|:---:|:---:|:---|
| **RDNA 4** | Radeon RX 9070 XT / RX 9070 | 16 GB | **`gfx1201`** | `"torch[device-gfx1201]"`；原生硬件支持 **FP8** 加速，16G 甜点王牌！ |
| **RDNA 4** | Radeon RX 9060 XT | 8G / 16G | **`gfx1200`** | `"torch[device-gfx1200]"`；主流高性价比甜点/入门卡 |
| **RDNA 3** | Radeon RX 7900 XTX / 7900 XT / 7900 GRE | 24G / 20G / 16G | **`gfx1100`** | `"torch[device-gfx1100]"`；大显存本地 `vLLM` 与中文 LoRA 微调主力！ |
| **RDNA 3** | Radeon PRO W7900 / W7800 | 48G / 32G | **`gfx1100`** | `"torch[device-gfx1100]"`；单卡超大显存专业工作站神器 |
| **RDNA 3** | Radeon RX 7800 XT / 7700 XT / PRO W7700 | 16G / 12G | **`gfx1101`** | `"torch[device-gfx1101]"`；跑 `Qwen3.5-9B` 与 `Gemma 4 12B` 极稳 |
| **RDNA 3** | Radeon RX 7600 XT / RX 7600 | 16G / 8G | **`gfx1102`** | `"torch[device-gfx1102]"`；入门高性价比卡，8G 版首推 `Gemma 4 E4B` |
| **RDNA 2** | Radeon RX 6950 XT / 6900 XT / 6800 XT / 6800 | 16 GB | **`gfx1030`** | 日常推理首推 **`llama-server` / `lemonade-server` Vulkan 后端**零门槛满速运行！ |

---

### 2. 锐龙 AI 移动处理器与统一内存迷你主机（RDNA 3.5 + XDNA 2 NPU）

| 处理器家族代号 | 代表处理器与内置集显型号 | NPU AI 算力 | GPU `gfx` 代号 | 推荐引擎与部署黄金组合 |
|:---|:---|:---:|:---:|:---|
| **Strix Halo**<br/>（旗舰统一内存 AI PC） | **Ryzen AI Max+ 395**（Radeon 8060S 40CU）<br/>**Ryzen AI Max 390**（Radeon 8050S 32CU） | **50 TOPS**<br/>（XDNA 2） | **`gfx1151`** | `"torch[device-gfx1151]"` + **`lemonade-server` / `llama-server`**；64G/128G 统一内存闭眼直上旗舰 **`Qwen3.6-35B-A3B`**！ |
| **Strix Point**<br/>（高端全能轻薄本） | **Ryzen AI 9 HX 370**（Radeon 890M 16CU）<br/>**Ryzen AI 9 365**（Radeon 880M 12CU） | **50 TOPS**<br/>（XDNA 2） | **`gfx1150`** | `"torch[device-gfx1150]"` + **`lemonade-server`**（插电用 GPU 跑 9B，离电用 XDNA 2 NPU 跑 `Gemma 4 E4B` 极省电！） |
| **Krackan Point**<br/>（主流锐龙 AI 轻薄本） | **Ryzen AI 7 350 / AI 5 340**（Radeon 860M / 840M） | **50 TOPS**<br/>（XDNA 2） | **`gfx1152`** | `"torch[device-gfx1152]"` + **`lemonade-server`**（NPU + iGPU 异构运行 `Gemma 4 E4B`） |
| **Phoenix / Hawk Point**<br/>（锐龙 7040 / 8040 系列） | Ryzen 7 7840HS / 8845HS（Radeon 780M） | 10～16 TOPS<br/>（XDNA 1） | **`gfx1103`** | 内存 ≥24G/32G 下使用 **`llama-server` / `lemonade-server` Vulkan 后端** 流畅运行 `Gemma 4 E4B` / `Qwen3.5-9B` |

---

### 3. 数据中心与云平台计算加速卡（CDNA 3 / CDNA 2 —— Radeon Cloud 常见）

| 架构代际 | 代表计算卡型号 | HBM 高带宽显存 | `gfx` Target 代号 | 典型应用场景 |
|:---|:---|:---:|:---:|:---|
| **CDNA 3** | **AMD Instinct MI300X / MI325X** | **192 GB / 256 GB HBM3(e)** | **`gfx942`** | `"torch[device-gfx942]"`；[AMD Radeon Cloud](../cloud/amd-radeon-cloud.md) 旗舰云算力，原生 FP8、`vLLM` 百并发与百亿大模型全参/LoRA 训练怪兽！ |
| **CDNA 3** | AMD Instinct MI300A（APU 统一封装） | 128 GB HBM3 | **`gfx942`** | 超算与融合科学计算 + 大模型训练推理一体 |
| **CDNA 2** | AMD Instinct MI250X / MI250 / MI210 | 128 GB / 64 GB HBM2e | **`gfx90a`** | `"torch[device-gfx90a]"`；经典数据中心多卡大模型训练与科学仿真 |

---

## 二、🧰 官方文档、交互图谱与免费云算力总入口

1. **免费云端与高校算力福利**：
   - ☁️ **[AMD Radeon Cloud 免费 100～1000+ 小时算力申领与 SSH 隧道指南](../cloud/amd-radeon-cloud.md)**
   - 🎓 **[AUP Learning Cloud（AMD 大学计划）实训算力指南](../cloud/aup-learning-cloud.md)**
2. **全景图谱与权威官方站**：
   - 🗺️ **[ROCm 软件栈交互式全景图谱（点击在线探索）](/atlas/rocm/index.html)**
   - 📖 **[AMD ROCm 官方文档中心（Latest / 10.0.0）](https://rocm.docs.amd.com/en/latest/)**
   - 📰 **[AMD ROCm 官方深度技术博客（前沿模型首日适配与调优战报）](https://rocm.blogs.amd.com/)**
   - 🏆 **[Strix Arena 社区真机性能排行榜](https://strix-arena.web.app/)**
3. **本仓库配套资产一键索引**：
   - ⚙️ **已验证客户端标准配置模板库**：[`src/configs/`](https://github.com/datawhalechina/hello-rocm/tree/dev_1.0/src/configs)（WorkBuddy、`Qoder`、`Codex`）
   - 🤖 **内置 AI 伴学助教技能包**：[`src/hello-rocm-skill/`](https://github.com/datawhalechina/hello-rocm/tree/dev_1.0/src/hello-rocm-skill)
   - 🧩 **主线实战与自动化单测源码**：[`src/main-assistant/`](https://github.com/datawhalechina/hello-rocm/tree/dev_1.0/src/main-assistant) & [`tests/test_course_v2.py`](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/tests/test_course_v2.py)

---

## 三、🤝 终点亦是新起点：加入 `hello-rocm` 与 Datawhale 开源共建！

开源社区最迷人的地方在于：**今天你是照着教程点亮第一颗模型的初学者，明天你踩过的一个小坑、写出的一个趣味案例，就能照亮成千上万后来者的路！**

我们真诚邀请通关至此的你，用任意一种方式留下你的名字：
1. **点亮一颗 Star ⭐ 鼓励开源创作**：前往 **[`github.com/datawhalechina/hello-rocm`](https://github.com/datawhalechina/hello-rocm)** 点亮右上角的小星星，把这本零基础手册分享给身边买了 AMD 电脑却只用来打游戏的朋友；
2. **成为「新手试学官」或贡献第 16 个案例**：
   - 发现任何表达不够直白或新版本小变化？直接提一个 **[新手试学反馈 Issue](https://github.com/datawhalechina/hello-rocm/issues/new/choose)**；
   - 想分享你独创的本地 AI 玩法？复制 **[五要素案例模板 `docs/zh/08-fun-cases/_CASE_TEMPLATE.md`](../08-fun-cases/_CASE_TEMPLATE.md)**，遵循 **[内容规范指南](https://github.com/datawhalechina/hello-rocm/blob/dev_1.0/CONTENT_GUIDE.md)** 提交 PR（本地跑通 `PYTHONPATH=. python3 tests/test_course_v2.py` 即可丝滑合入）！
3. **结伴同行不孤单**：
   - 查看 **[hello-rocm 组队学习与 20 天打卡计划](../learning/index.md)**；
   - 加入 **[Datawhale × hello-rocm 飞书共学社区](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)**，晒出你的《成功故事》与五件套作品集，参评优秀学习者证书并直通 **AMD 开发者布道师计划**！

---

<div align="center" style="margin: 32px 0;">

### 🎉 祝贺你圆满通关 `hello-rocm 2.0` 全书十章！
**愿你手中的每一份算力，都化作探索未知与创造美好的自由羽翼。我们下一门专精课再见！🚀**

</div>
