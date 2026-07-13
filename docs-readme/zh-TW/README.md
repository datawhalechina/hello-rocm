<div align=center>
  <img src="../../docs/public/images/head.png" >
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*開源 · 社群驅動 · 讓 AMD AI 生態更好用*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="README.md"><img alt="簡體中文" src="https://img.shields.io/badge/簡體中文-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="../ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>
<div align="center">

<a href="https://datawhalechina.github.io/hello-rocm/"><img src="https://img.shields.io/badge/hello--rocm_完整教學-線上體驗-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="25"></a>

</div>

&emsp;&emsp;自 **ROCm 7.10.0**（2025 年 12 月 11 日釋出）起，ROCm 已可像 CUDA 一樣在 Python 虛擬環境中無縫安裝，並正式支援 **Linux 與 Windows** 雙平台。這是 AMD 在 AI 領域的一大突破——學習者與大型語言模型愛好者在硬體選擇上不再只有 NVIDIA，AMD GPU 正成為一個有力的競爭選項。

&emsp;&emsp;然而，**硬體門檻降低並不代表學習路徑自動變得清晰**。對於已經有大型語言模型基礎、想要在 AMD GPU 上實際動手的學習者來說，真正的挑戰才剛開始：怎麼在 AMD GPU 上把一個模型跑起來？怎麼在這個基礎上做微調與訓練？怎麼理解 ROCm 的 GPU 程式設計體系，完成從 CUDA 到 ROCm 的轉移？最終，這些能力又如何匯聚成一個可以實際上線的 AI 應用？

&emsp;&emsp;**hello-rocm** 正是為這條路徑而生。本專案系統性地涵蓋 AMD ROCm 平台上大型語言模型的完整使用流程，帶你從**把第一個模型跑起來**，走到**在 AMD GPU 上打造真實 AI 應用**，中間經過微調訓練與 GPU 程式設計的每一個關鍵環節——讓 AMD GPU 不只是一張顯示卡，而是你進入 AI 開發世界的真實起點。

&emsp;&emsp;**這個專案的核心內容就是教學，讓更多學生和未來的從業者了解並熟悉 AMD ROCm 的使用方式！任何人都可以提出 issue 或送出 PR，一起共建維護這個專案。**

> &emsp;&emsp;***學習建議：建議先完成 [00-Environment](../../docs/zh/00-environment/index.md) 中的環境安裝（ROCm + PyTorch + uv），再學習部署與微調，最後探索算子最佳化與 GPU 程式設計。初學者可在環境就緒後從 LM Studio 或 vLLM 部署開始。***

### hello-rocm Skill：把本專案裝進你的 AI 助手

&emsp;&emsp;如果你使用支援 Skills、Rules 或 Agent 設定的 AI 程式開發工具，可以直接使用本專案內建的 **hello-rocm Skill**。它會根據本儲存庫的目錄結構、Reference 索引、GPU 架構表、部署教學和疑難排解清單，為你定位到具體文件與官方連結。

```text
請使用當前儲存庫的 src/hello-rocm-skill 作為 hello-rocm Skill；如果你的工具支援 Skills、Rules 或 Agent 設定，請把它安裝或載入到合適位置（例如 .claude/skills、.cursor/skills 或 .agents/skills），然後根據該 Skill 幫我學習、部署和排查 AMD ROCm。
```

&emsp;&emsp;你可以這樣問：我的 AMD GPU 能不能跑 ROCm？我想最快跑通一個本地大型語言模型應該看哪篇？vLLM / Ollama / llama.cpp 在 ROCm 上怎麼裝？`torch.cuda.is_available()` 回傳 False 怎麼排查？更多說明見 [hello-rocm Skill 使用指南](../../docs/zh/04-references/index.md#hello-rocm-skill)。

### 最新動態

- *2026.5.15:* [*ROCm 7.13.0 Release Notes*](https://rocm.docs.amd.com/en/7.13.0-preview/index.html)

- *2026.3.11:* [*ROCm 7.12.0 Release Notes*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *2025.12.11:* [*ROCm 7.10.0 Release Notes*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### 已支援模型

<p align="center">
  <strong>✨ 主流大型語言模型：環境設定 · 多框架推論 · 微調實作 ✨</strong><br>
  <em>ROCm 統一環境安裝（Windows / Ubuntu）+ ROCm 7+ · 按模型分目錄教學（持續擴充）</em><br>
 <a href="../../docs/zh/00-environment/index.md">00-環境安裝教學</a> 
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">

  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Qwen3</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/01-deploy/qwen3/lm-studio-rocm7-deploy.md">LM Studio 部署</a><br>
      • <a href="../../docs/zh/01-deploy/qwen3/vllm-rocm7-deploy.md">vLLM 部署</a><br>
      • <a href="../../docs/zh/01-deploy/qwen3/ollama-rocm7-deploy.md">Ollama 部署</a><br>
      • <a href="../../docs/zh/01-deploy/qwen3/llamacpp-rocm7-deploy.md">llama.cpp 部署</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/02-fine-tune/qwen3/qwen3-0.6b-lora-swanlab.md">Qwen3-0.6B LoRA 微調</a><br>
      • <a href="../../src/fine-tune/models/qwen3/01-Qwen3-8B-LoRA.ipynb">Qwen3-8B LoRA 微調</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Qwen3.5</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/01-deploy/qwen3.5/lm-studio-rocm7-deploy.md">LM Studio 部署</a><br>
      • <a href="../../docs/zh/01-deploy/qwen3.5/vllm-rocm7-deploy.md">vLLM 部署</a><br>
      • <a href="../../docs/zh/01-deploy/qwen3.5/ollama-rocm7-deploy.md">Ollama 部署</a><br>
      • <a href="../../docs/zh/01-deploy/qwen3.5/llamacpp-rocm7-deploy.md">llama.cpp 部署</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/02-fine-tune/qwen3.5/qwen3.5-4b-lora-swanlab.md">Qwen3.5-4B LoRA 微調</a><br>
      • <a href="../../src/fine-tune/models/qwen3.5/Qwen3.5-4B-LoRA.ipynb">Qwen3.5-4B LoRA 微調 Notebook</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Gemma4</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/01-deploy/gemma4/gemma4_model.md">Gemma 4 模型介紹</a><br>
      • <a href="../../docs/zh/01-deploy/gemma4/lm-studio-rocm7-deploy.md">LM Studio 部署</a><br>
      • <a href="../../docs/zh/01-deploy/gemma4/vllm-rocm7-deploy.md">vLLM 部署</a><br>
      • <a href="../../docs/zh/01-deploy/gemma4/ollama-rocm7-deploy.md">Ollama 部署</a><br>
      • <a href="../../docs/zh/01-deploy/gemma4/llamacpp-rocm7-deploy.md">llama.cpp 部署</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb">Gemma4 - E4B LoRA 微調（基於 TRL）</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>MiniCPM5</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/01-deploy/minicpm/llamacpp-rocm7-deploy.md">llama.cpp 部署</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>MiniCPM-V 4.6</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/01-deploy/minicpmv/llamacpp-rocm7-deploy.md">llama.cpp 部署</a><br>
      • <a href="../../docs/zh/01-deploy/minicpmv/vllm-rocm7-deploy.md">vLLM 部署</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>MiniCPM-o 4.5</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/01-deploy/minicpm-o/minicpm-o-model.md">MiniCPM-o 4.5 模型介紹</a><br>
      • <a href="../../docs/zh/01-deploy/minicpm-o/llamacpp-omni-rocm7-deploy.md">llama.cpp-omni 部署（語音+視覺+TTS）</a><br>
      • <a href="../../docs/zh/01-deploy/minicpm-o/webdemo-rocm7-deploy.md">Web Demo 全雙工部署</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>具身智能策略（ACT / SmolVLA / Pi0 / Pi0.5）</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/05-amd-yes/every-embodied.md">ROCm 具身智能策略複刻案例</a><br>
      • <a href="https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/08_act_training_rocm.ipynb">ACT ROCm 訓練 Notebook</a><br>
      • <a href="https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/09_smolvla_training_rocm.ipynb">SmolVLA ROCm 訓練 Notebook</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/10_pi0_training_rocm.ipynb">Pi0 ROCm 訓練 Notebook</a><br>
      • <a href="https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/12_pi0_strict_input_end_to_end.ipynb">Pi0 嚴格端到端診斷 Notebook</a><br>
      • <a href="https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/13_pi05_random_position_eef_delta.ipynb">Pi0.5 EEF-delta 訓練 Notebook</a><br>
    </td>
  </tr>
</table>

## 專案意義

&emsp;&emsp;什麼是 ROCm？

> ROCm（Radeon Open Compute）是 AMD 推出的開源 GPU 運算平台，為高效能運算和機器學習提供開放的軟體堆疊。它讓 AMD GPU 也能跑平行運算，是 CUDA 在 AMD 平台上的替代方案。

&emsp;&emsp;百模大戰正值火熱，開源大型語言模型層出不窮。然而，目前大多數大型語言模型教學和開發工具都建立在 NVIDIA CUDA 生態之上。對於想要使用 AMD GPU 的開發者來說，缺乏系統性的學習資源一直是痛點。

&emsp;&emsp;自 ROCm 7.10.0（2025 年 12 月 11 日）起，AMD 透過 TheRock 專案對 ROCm 底層架構進行了重構，將運算執行環境與作業系統脫鉤，使同一套 ROCm 上層介面可以同時運行在 Linux 與 Windows 上，並支援像 CUDA 一樣直接安裝到 Python 虛擬環境中使用。這意味著 ROCm 不再是只面向 Linux 的「工程工具」，而是升級為一個真正面向 AI 學習者與開發者的跨平台 GPU 運算平台——不論使用 Windows 還是 Linux，使用者都可以更低門檻地使用 AMD GPU 進行訓練和推論，大型語言模型與 AI 玩家在硬體選擇上不再被 NVIDIA 單一生態綁定，AMD GPU 正逐步成為一個可以被一般使用者真正用起來的 AI 運算平台。

&emsp;&emsp;本專案旨在基於核心貢獻者的經驗，提供 AMD ROCm 平台上大型語言模型部署、微調、訓練的完整教學；我們希望充分匯聚共創者，一起豐富 AMD AI 生態。

&emsp;&emsp;***我們希望成為 AMD GPU 與廣大使用者之間的橋樑，以自由、平等的開源精神，擁抱更遼闊的 AI 世界。***

## 專案受眾

&emsp;&emsp;本專案適合以下學習者：


* 手邊有一張 AMD 顯示卡，想體驗一下大型語言模型在本機運行；
* 想要使用 AMD GPU 進行大型語言模型開發，但找不到系統性的教學；
* 希望以低成本、高 CP 值的方式部署和運行大型語言模型；
* 對 ROCm 生態感興趣，想要親自上手實作；

## 專案規劃及進展

&emsp;&emsp;本專案擬圍繞 ROCm 大型語言模型應用全流程組織，包括統一環境基線（00-Environment）、部署應用、微調訓練、算子最佳化等：


### 專案結構

```
hello-rocm/
├── docs/                   # VitePress 文件源碼
│   ├── zh/                 # 中文文件
│   │   ├── 00-environment/ # ROCm 基礎環境安裝與設定
│   │   ├── 01-deploy/      # ROCm 大型語言模型部署實作
│   │   ├── 02-fine-tune/   # ROCm 大型語言模型微調實作
│   │   ├── 03-infra/       # ROCm 算子最佳化實作
│   │   ├── 04-references/  # ROCm 優質參考資料
│   │   └── 05-amd-yes/     # AMD 實作案例集合
│   └── en/                 # English docs
├── src/                    # 原始碼與腳本
└── assets/                 # 公用資源
```

### 00. Environment - ROCm 基礎環境

<p align="center">
  <strong>🛠️ ROCm 基礎環境安裝與設定</strong><br>
  <em>統一環境基線 · ROCm 7.12.0 · Windows / Ubuntu · uv + PyTorch</em><br>
  📖 <strong><a href="../../docs/zh/00-environment/index.md">Getting Started with ROCm Environment</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="../../docs/zh/00-environment/rocm-gpu-architecture-table.md">GPU 架構與 pip 索引對照表</a><br>
      • Windows 11 安裝、驅動與安全性前置說明<br>
      • Ubuntu 24.04 安裝（uv 方式與備選一鍵腳本）<br>
      • 安裝驗證、移除與切換其他 GPU 架構
    </td>
  </tr>
</table>

### 01. Deploy - ROCm 大型語言模型部署

<p align="center">
  <strong>🚀 ROCm 大型語言模型部署實作</strong><br>
  <em>零基礎快速上手 AMD GPU 大型語言模型部署</em><br>
  📖 <strong><a href="../../docs/zh/01-deploy/index.md">Getting Started with ROCm Deploy</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • LM Studio 零基礎大型語言模型部署<br>
      • vLLM 零基礎大型語言模型部署<br>
      • Ollama 零基礎大型語言模型部署<br>
      • llama.cpp 零基礎大型語言模型部署<br>
      • ATOM 零基礎大型語言模型部署
    </td>
  </tr>
</table>

### 02. Fine-tune - ROCm 大型語言模型微調

<p align="center">
  <strong>🔧 ROCm 大型語言模型微調實作</strong><br>
  <em>在 AMD GPU 上進行高效模型微調</em><br>
  📖 <strong><a href="../../docs/zh/02-fine-tune/index.md">Getting Started with ROCm Fine-tune</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • 大型語言模型零基礎微調教學<br>
      • 大型語言模型單機微調腳本<br>
      • 大型語言模型多機多卡微調教學
    </td>
  </tr>
</table>

### 03. Infra - ROCm 算子最佳化與 GPU 程式設計

<p align="center">
  <strong>⚙️ ROCm 算子最佳化與 GPU 程式設計</strong><br>
  <em>從 AMD AI 硬體全景到 HIP 算子與效能分析</em><br>
  📖 <strong><a href="../../docs/zh/03-infra/index.md">開始學習 ROCm 算子最佳化與 GPU 程式設計</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • AMD AI 硬體全景與 ROCm 生態<br>
      • GPU 軟體堆疊與硬體架構深度解析<br>
      • HIP 程式設計入門與手寫 Kernel 實戰<br>
      • 自訂 PyTorch 算子與 Autograd 整合
    </td>
  </tr>
</table>

### 04. References - ROCm 優質參考資料

<p align="center">
  <strong>📚 ROCm 優質參考資料</strong><br>
  <em>精選的 AMD 官方與社群資源</em><br>
  📖 <strong><a href="../../docs/zh/04-references/index.md">ROCm References</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">ROCm 官方文件</a><br>
      • <a href="https://github.com/amd">AMD GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">ROCm Release Notes</a><br>
      • <a href="../../docs/zh/04-references/index.md#amd-gpu-架构白皮书">AMD GPU 架構白皮書（CDNA / RDNA）</a><br>
      • <a href="../../docs/zh/04-references/index.md#框架与推理服务rocm-快速安装入口">框架與推論服務 ROCm 快速安裝入口</a><br>
      • 相關新聞
    </td>
  </tr>
</table>

### 05. AMD-YES - AMD 實作案例集合

<p align="center">
  <strong>✨ AMD 實作案例集合</strong><br>
  <em>社群驅動的 AMD GPU 專案實作</em><br>
  📖 <strong><a href="../../docs/zh/05-amd-yes/index.md">Getting Started with ROCm AMD-YES</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • toy-cli - LLM 輕量化終端助手<br>
      • Minesweeper Agent - 用本地大模型玩掃雷（Agent 循環實踐）<br>
      • YOLOv10 微信跳一跳 - 遊戲 AI 實戰（在 ROCm 環境下訓練並使用 yolov10）<br>
      • Chat-甄嬛 - 古風對話大型語言模型<br>
      • 智能旅行規劃助手 - HelloAgents Agent 實戰<br>
      • Torch-RecHub - 推薦系統實戰（CTR、召回、多任務、ONNX 導出）<br>
      • happy-llm - 分散式大型語言模型訓練<br>
      • <a href="../../docs/zh/05-amd-yes/every-embodied.md">Every Embodied - ROCm 具身智能策略複刻</a><br>
    </td>
  </tr>
</table>

## 貢獻指南

&emsp;&emsp;我們歡迎所有形式的貢獻！不論是：

* 完善或新增教學
* 修復錯誤與 Bug
* 分享你的 AMD 專案
* 提出建議與想法

&emsp;&emsp;參與前請先閱讀 **[Content Guide](../../CONTENT_GUIDE_en.md)**（目錄、命名、配圖與文件結構與 **Qwen3 / Qwen3.5** 等教學對齊），再閱讀 **[CONTRIBUTING_en.md](../../CONTRIBUTING_en.md)**（Issue / PR 流程與模型專項目錄約定）。

&emsp;&emsp;如果你在使用 ROCm、部署模型或閱讀教學時遇到疑難排解與常見問題，也歡迎加入我們的 **[社群討論](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)**，和社群一起補充經驗、回報問題、完善教學。

&emsp;&emsp;想要深度參與的同學可以聯繫我們，我們會將你加入到專案的維護者中。

## 致謝
### 核心貢獻者


- [宋志學(不要蔥薑蒜)-專案負責人](https://github.com/KMnO4-zx) （Datawhale 成員，self-llm, happy-llm 專案負責人）
- [陳榆-專案負責人](https://github.com/lucachen) （內容創作者-Google 開發者機器學習技術專家）
- [陳思州-專案成員](https://github.com/jjyaoao) (Datawhale 成員, hello-agents 專案負責人)
- [潘嘉航-專案成員](https://github.com/amdjiahangpan) （內容創作者-AMD 軟體工程師）
- [劉偉鴻-專案成員](https://github.com/Weihong-Liu) （Datawhale 成員）
- [郝東波-專案成員](https://github.com/wlkq151172) （內容創作者-江南大學研究生）
- [柯慕靈-專案成員](https://github.com/1985312383)（Datawhale 成員）
> 註：歡迎更多貢獻者加入！

### 其他

- 如果有任何想法可以聯繫我們，也歡迎大家多多提出 issue
- 特別感謝以下為教學做出貢獻的同學！
- 感謝 AMD University Program 對本專案的支持！！

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>


## License

[MIT License](../../LICENSE)

---

<div align="center">

**讓我們一起打造 AMD AI 的未來！** 💪

Made with ❤️ by the hello-rocm community

</div>
