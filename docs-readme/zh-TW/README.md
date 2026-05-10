<div align=center>
  <img src="../../docs/public/images/head.png" >
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*開源 · 社區驅動 · 讓 AMD AI 生態更易用*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="簡體中文" src="https://img.shields.io/badge/簡體中文-d9d9d9"></a>
  <a href="README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="../ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>


&emsp;&emsp;自 **ROCm 7.10.0** (2025年12月11日發布) 以來，ROCm 已支持像 CUDA 一樣在 Python 虛擬環境中無縫安裝，並正式支持 **Linux 和 Windows** 雙系統。這標誌著 AMD 在 AI 領域的重大突破——學習者與大模型愛好者在硬體選擇上不再局限於 NVIDIA，AMD GPU 正成為一個強有力的競爭選擇。

&emsp;&emsp;蘇媽在發布會上宣布 ROCm 將保持 **每 6 周一個新版本** 的迭代節奏，並全力轉向 AI 領域。前景令人振奮！

&emsp;&emsp;然而，目前全球範圍內缺乏系統的 ROCm 大模型推理、部署、訓練、微調及 Infra 的學習教程。**hello-rocm** 應運而生，旨在填補這一空白。

&emsp;&emsp;**項目的主要內容就是教程，讓更多的學生和未來的從業者了解和熟悉 AMD ROCm 的使用方法！任何人都可以提出 issue 或是提交 PR，共同構建維護這個項目。**

> &emsp;&emsp;***學習建議：建議先完成 [00-Environment](../../docs/zh/00-environment/index.md) 中的環境安裝（ROCm + PyTorch + uv），再學習部署與微調，最後探索 Infra 算子優化。初學者可在環境就緒後從 LM Studio 或 vLLM 部署開始。***

### 最新動態

- *2026.3.11:* [*ROCm 7.12.0 Release Notes*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *2025.12.11:* [*ROCm 7.10.0 Release Notes*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### 已支持模型

<p align="center">
  <strong>✨ 主流大模型：環境配置 · 多框架推理 · 微調實踐 ✨</strong><br>
  <em>ROCm 統一環境安裝（Windows / Ubuntu）+ ROCm 7+ · 按模型分目錄教程（持續擴充）</em><br>
 <a href="../../docs/zh/00-environment/index.md">00-環境安裝教程</a> 
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">

  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Qwen3</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/01-deploy/qwen3/lm-studio-rocm7-deploy.md">LM Studio部署</a><br>
      • <a href="../../docs/zh/01-deploy/qwen3/vllm-rocm7-deploy.md">vLLM部署</a><br>
      • <a href="../../docs/zh/01-deploy/qwen3/ollama-rocm7-deploy.md">Ollama部署</a><br>
      • <a href="../../docs/zh/01-deploy/qwen3/llamacpp-rocm7-deploy.md">llama.cpp部署</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/02-fine-tune/qwen3/qwen3-0.6b-lora-swanlab.md">Qwen3-0.6B LoRA微調</a><br>
      • <a href="../../src/fine-tune/models/qwen3/01-Qwen3-8B-LoRA.ipynb">Qwen3-8B LoRA微調</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Gemma4</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/zh/01-deploy/gemma4/gemma4_model.md">Gemma 4 模型介紹</a><br>
      • <a href="../../docs/zh/01-deploy/gemma4/lm-studio-rocm7-deploy.md">LM Studio部署</a><br>
      • <a href="../../docs/zh/01-deploy/gemma4/vllm-rocm7-deploy.md">vLLM部署</a><br>
      • <a href="../../docs/zh/01-deploy/gemma4/ollama-rocm7-deploy.md">Ollama部署</a><br>
      • <a href="../../docs/zh/01-deploy/gemma4/llamacpp-rocm7-deploy.md">llama.cpp部署</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../src/fine-tune/models/gemma4/01-Gemma4-E4B-LoRA及SwanLab可视化记录.ipynb">Gemma4 - E4B LoRA微調（基於TRL）</a><br>
    </td>
  </tr>
</table>

## 項目意義

&emsp;&emsp;什麼是 ROCm？

> ROCm（Radeon Open Compute）是 AMD 推出的開源 GPU 計算平臺，旨在為高性能計算和機器學習提供開放的軟體棧。它支持 AMD GPU 進行並行計算，是 CUDA 在 AMD 平臺上的替代方案。

&emsp;&emsp;百模大戰正值火熱，開源 LLM 層出不窮。然而，目前大多數大模型教程和開發工具都基於 NVIDIA CUDA 生態。對於想要使用 AMD GPU 的開發者來說，缺乏系統性的學習資源是一個痛點。

&emsp;&emsp;自 ROCm 7.10.0（2025 年 12 月 11 日） 起，AMD 通過 TheRock 項目對 ROCm 底層架構進行了重構，將計算運行時與作業系統解耦，使同一套 ROCm 上層接口可以同時運行在 Linux 與 Windows 上，並支持像 CUDA 一樣直接安裝到 Python 虛擬環境中使用。這意味著 ROCm 不再是只面向 Linux 的「工程工具」，而是升級為一個真正面向 AI 學習者與開發者的跨平臺 GPU 計算平臺——無論使用 Windows 還是 Linux，用戶都可以更低門檻地使用 AMD GPU 進行訓練和推理，大模型與 AI 玩家在硬體選擇上不再被 NVIDIA 單一生態所綁定，AMD GPU 正逐步成為一個可以被普通用戶真實使用的 AI 計算平臺。

&emsp;&emsp;本項目旨在基於核心貢獻者的經驗，提供 AMD ROCm 平臺上大模型部署、微調、訓練的完整教程；我們希望充分聚集共創者，一起豐富 AMD AI 生態。

&emsp;&emsp;***我們希望成為 AMD GPU 與普羅大眾的橋梁，以自由、平等的開源精神，擁抱更恢弘而遼闊的 AI 世界。***

## 項目受眾

&emsp;&emsp;本項目適合以下學習者：


* 手頭有一張AMD顯卡，想體驗一下大模型本地運行;
* 想要使用 AMD GPU 進行大模型開發，但找不到系統教程；
* 希望低成本、高性價比地部署和運行大模型；
* 對 ROCm 生態感興趣，想要親自上手實踐；

## 項目規劃及進展

&emsp;&emsp;本項目擬圍繞 ROCm 大模型應用全流程組織，包括統一環境基線（00-Environment）、部署應用、微調訓練、算子優化等：


### 項目結構

```
hello-rocm/
├── 00-Environment/         # ROCm 基礎環境安裝與配置（統一基線）
├── 01-Deploy/              # ROCm 大模型部署實踐
├── 02-Fine-tune/           # ROCm 大模型微調實踐
├── 03-Infra/               # ROCm 算子優化實踐
├── 04-References/          # ROCm 優質參考資料
└── 05-AMD-YES/             # AMD 實踐案例集合
```

### 00. Environment - ROCm 基礎環境

<p align="center">
  <strong>🛠️ ROCm 基礎環境安裝與配置</strong><br>
  <em>統一環境基線 · ROCm 7.12.0 · Windows / Ubuntu · uv + PyTorch</em><br>
  📖 <strong><a href="../../docs/zh/00-environment/index.md">Getting Started with ROCm Environment</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="../../docs/zh/00-environment/rocm-gpu-architecture-table.md">GPU 架構與 pip 索引對照表</a><br>
      • Windows 11 安裝、驅動與安全項前置說明<br>
      • Ubuntu 24.04 安裝（uv 方式與備選一鍵腳本）<br>
      • 安裝校驗、卸載與切換其他 GPU 架構
    </td>
  </tr>
</table>

### 01. Deploy - ROCm 大模型部署

<p align="center">
  <strong>🚀 ROCm 大模型部署實踐</strong><br>
  <em>零基礎快速上手 AMD GPU 大模型部署</em><br>
  📖 <strong><a href="../../docs/zh/01-deploy/index.md">Getting Started with ROCm Deploy</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • LM Studio 零基礎大模型部署<br>
      • vLLM 零基礎大模型部署<br>
      • Ollama 零基礎大模型部署<br>
      • llama.cpp 零基礎大模型部署<br>
      • ATOM 零基礎大模型部署
    </td>
  </tr>
</table>

### 02. Fine-tune - ROCm 大模型微調

<p align="center">
  <strong>🔧 ROCm 大模型微調實踐</strong><br>
  <em>在 AMD GPU 上進行高效模型微調</em><br>
  📖 <strong><a href="../../docs/zh/02-fine-tune/index.md">Getting Started with ROCm Fine-tune</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • 大模型零基礎微調教程<br>
      • 大模型單機微調腳本<br>
      • 大模型多機多卡微調教程
    </td>
  </tr>
</table>

### 03. Infra - ROCm 算子優化

<p align="center">
  <strong>⚙️ ROCm 算子優化實踐</strong><br>
  <em>CUDA 到 ROCm 的遷移與優化指南</em><br>
  📖 <strong><a href="../../docs/zh/03-infra/index.md">Getting Started with ROCm Infra</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • HIPify 自動化遷移實戰<br>
      • BLAS 與 DNN 的無縫切換<br>
      • NCCL 到 RCCL 的遷移<br>
      • Nsight 到 Rocprof 的映射
    </td>
  </tr>
</table>

### 04. References - ROCm 優質參考資料

<p align="center">
  <strong>📚 ROCm 優質參考資料</strong><br>
  <em>精選的 AMD 官方與社區資源</em><br>
  📖 <strong><a href="../../docs/zh/04-references/references.md">ROCm References</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">ROCm 官方文檔</a><br>
      • <a href="https://github.com/amd">AMD GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">ROCm Release Notes</a><br>
      • 相關新聞
    </td>
  </tr>
</table>

### 05. AMD-YES - AMD 實踐案例集合

<p align="center">
  <strong>✨ AMD 實踐案例集合</strong><br>
  <em>社區驅動的 AMD GPU 項目實踐</em><br>
  📖 <strong><a href="../../docs/zh/05-amd-yes/index.md">Getting Started with ROCm AMD-YES</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • toy-cli - LLM 輕量化終端助手<br>
      • YOLOv10 微信跳一跳 - 遊戲 AI 實戰<br>
      • Chat-甄嬛 - 古風對話大模型<br>
      • 智能旅行規劃助手 - HelloAgents Agent 實戰<br>
      • happy-llm - 分布式大模型訓練
    </td>
  </tr>
</table>

## 貢獻指南

&emsp;&emsp;我們歡迎所有形式的貢獻！無論是：

* 完善或新增教程
* 修復錯誤與 Bug
* 分享你的 AMD 項目
* 提出建議與想法

&emsp;&emsp;參與前請先閱讀 **[規範指南](../../規範指南.md)**（目錄、命名、配圖與文檔結構與 **Qwen3** 等教程對齊），再閱讀 **[CONTRIBUTING.md](../../CONTRIBUTING.md)**（Issue / PR 流程與模型專項目錄約定）。

&emsp;&emsp;想要深度參與的同學可以聯繫我們，我們會將你加入到項目的維護者中。

## 致謝
### 核心貢獻者


- [宋志學(不要蔥姜蒜)-項目負責人](https://github.com/KMnO4-zx) （Datawhale成員）
- [陳榆-項目負責人](https://github.com/lucachen) （內容創作者-谷歌開發者機器學習技術專家）
- [潘嘉航-項目成員](https://github.com/amdjiahangpan) （內容創作者-AMD軟體工程師）
- [劉偉鴻-項目成員](https://github.com/Weihong-Liu) （Datawhale成員）
> 註：歡迎更多貢獻者加入！

### 其他

- 如果有任何想法可以聯繫我們，也歡迎大家多多提出 issue
- 特別感謝以下為教程做出貢獻的同學！
- 感謝 AMD University Program 對本項目的支持！！

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>


## License

[MIT License](../../LICENSE)

---

<div align="center">

**讓我們一起構建 AMD AI 的未來！** 💪

Made with ❤️ by the hello-rocm community

</div>

