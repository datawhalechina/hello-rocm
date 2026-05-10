<div align=center>
  <img src="../../docs/public/images/head.png" >
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*オープンソース · コミュニティ主導 · AMD AIエコシステムをもっと身近に*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="../ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>

&emsp;&emsp;**ROCm 7.10.0**（2025年12月11日リリース）以降、ROCmはCUDAと同様にPython仮想環境にシームレスにインストールでき、**LinuxとWindowsの両方**を公式サポートしています。これはAMDにとってAI分野における大きな一歩です。学習者やLLM愛好家はもはやNVIDIAハードウェアに制限されることなく、AMD GPUが強力で実用的な選択肢となります。

&emsp;&emsp;AMDは**約6週間**のROCmリリースサイクルを約束しており、AIに重点を置いています。ロードマップは非常にエキサイティングです。

&emsp;&emsp;しかし、世界中でROCmを用いたLLMの推論、デプロイ、トレーニング、ファインチューニング、インフラに関する体系的なチュートリアルは依然として不足しています。**hello-rocm**はそのギャップを埋めるために存在します。

&emsp;&emsp;**このプロジェクトは主にチュートリアル**であり、学生や将来の実務者がAMD ROCmを体系的に学べるようにするものです。**どなたでもIssueの作成やプルリクエストの送信を歓迎します**。プロジェクトを共に成長させ、維持していきましょう。

> &emsp;&emsp;***学習パス: まず[00-環境設定](../../docs/en/00-environment/index.md)（ROCm + PyTorch + **uv**）を完了し、次にデプロイとファインチューニング、最後にインフラ/オペレーターレベルのトピックに進みます。環境が動作したら、LM StudioまたはvLLMから始めるのが良いでしょう。***

### hello-rocm Skill：AIアシスタントで本プロジェクトを使う

&emsp;&emsp;Skills、Rules、Agent設定をサポートするAIコーディングツールを使っている場合は、内蔵の **hello-rocm Skill** を利用できます。このSkillは、リポジトリ構成、Reference索引、GPUアーキテクチャ表、デプロイ手順、トラブルシューティング一覧をもとに、適切なドキュメントと公式リンクへ案内します。

```text
Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place, such as .claude/skills, .cursor/skills, or .agents/skills, then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.
```

&emsp;&emsp;詳しくは [hello-rocm Skill guide](../../docs/en/04-references/index.md#hello-rocm-skill) を参照してください。

### 最新情報

- *2026年3月11日:* [*ROCm 7.12.0 リリースノート*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *2025年12月11日:* [*ROCm 7.10.0 リリースノート*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### サポート対象モデル & チュートリアル

<p align="center">
  <strong>✨ 主要LLM: 環境設定 · マルチフレームワーク推論 · ファインチューニング ✨</strong><br>
  <em>統一ROCmセットアップ（Windows / Ubuntu）+ ROCm 7+ · モデル別チュートリアル（随時追加）</em><br>
 <a href="../../docs/en/00-environment/index.md">00 — 環境設定</a>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">

  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Qwen3</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/qwen3/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/qwen3/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/qwen3/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/qwen3/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/02-fine-tune/qwen3/qwen3-0.6b-lora-swanlab.md">Qwen3-0.6B LoRA + SwanLab</a><br>
      • <a href="../../src/fine-tune/models/qwen3/01-Qwen3-8B-LoRA.ipynb">Qwen3-8B LoRA (Notebook)</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Gemma4</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/gemma4/gemma4_model.md">Gemma 4 モデル概要</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../src/fine-tune/models/gemma4/01-Gemma4-E4B-LoRA及SwanLab可视化记录.ipynb">Gemma4 E4B LoRA ファインチューニング (TRL, Notebook)</a><br>
    </td>
  </tr>
</table>

## このプロジェクトの目的

&emsp;&emsp;ROCmとは何か？

> ROCm（Radeon Open Compute）は、AMDのオープンGPUコンピューティングスタックであり、HPCおよび機械学習向けに設計されています。AMD GPU上で並列ワークロードを実行でき、AMDハードウェアにおける主要なCUDA代替パスです。

&emsp;&emsp;オープンLLMはあらゆる場所で利用可能ですが、ほとんどのチュートリアルやツールはNVIDIA CUDAスタックを前提としています。AMDを選択する開発者は、エンドツーエンドのROCmネイティブな学習教材を欠いていることがよくあります。
&emsp;&emsp;**ROCm 7.10.0**（2025年12月11日）以降、AMDの**TheRock**プロジェクトにより、コンピュートランタイムがOSから分離され、同じROCmインターフェースが**LinuxとWindows**の両方で動作するようになりました。また、ROCmはCUDAと同様にPython環境にインストールできるようになりました。ROCmはもはや「Linux専用の配管」ではなく、クロスプラットフォームなAIコンピュートプラットフォームです。**hello-rocm**は、より多くの人が実際にAMD GPUをトレーニングや推論に活用できるよう、実践的なガイドをまとめています。

&emsp;&emsp;***私たちは、AMD GPUと日々の開発者をつなぐ架け橋となることを目指しています——オープンで、包括的で、より広いAIの未来を志向して。***

## 対象者

&emsp;&emsp;以下のような方に、このプロジェクトは役立つかもしれません：

* AMD GPUを所有しており、ローカルでLLMを実行したい方
* AMD上で開発したいが、体系化されたROCmカリキュラムがない方
* コスト効率の良いデプロイと推論に関心がある方
* ROCmに興味があり、実践的に学びたい方

## ロードマップと構成

&emsp;&emsp;このリポジトリは、ROCmを用いたLLMワークフローの全体像に沿っています：**統合ベースライン（00-Environment）**、デプロイ、ファインチューニング、そしてインフラ関連のトピックです。

### リポジトリ構成

```
hello-rocm/
├── 00-Environment/         # ROCmベースラインのインストールと設定
├── 01-Deploy/              # ROCm上でのLLMデプロイ
├── 02-Fine-tune/           # ROCm上でのLLMファインチューニング
├── 03-Infra/               # ROCm上のインフラ/オペレーター
├── 04-References/          # 厳選されたROCmリファレンス
└── 05-AMD-YES/             # コミュニティAMDプロジェクトの紹介
```

### 00. 環境 — ROCmベースライン

<p align="center">
  <strong>🛠️ ROCm環境のインストールと設定</strong><br>
  <em>単一ベースライン · ROCm 7.12.0 · Windows / Ubuntu · uv + PyTorch</em><br>
  📖 <strong><a href="../../docs/en/00-environment/index.md">ROCm環境入門</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="../../docs/en/00-environment/rocm-gpu-architecture-table.md">GPUアーキテクチャとpipインデックス対応表</a><br>
      • Windows 11：ドライバ、セキュリティ前提条件、インストールフロー<br>
      • Ubuntu 24.04：uvベースのインストールとオプションの統合インストーラスクリプト<br>
      • 検証、アンインストール、GPUターゲットの切り替え
    </td>
  </tr>
</table>

### 01. デプロイ — ROCm上でのLLMデプロイ

<p align="center">
  <strong>🚀 ROCm LLMデプロイ</strong><br>
  <em>ゼロからAMD GPU上でモデルを実行するまで</em><br>
  📖 <strong><a href="../../docs/en/01-deploy/index.md">ROCmデプロイ入門</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • LM Studio をゼロから<br>
      • vLLM をゼロから<br>
      • Ollama をゼロから<br>
      • llama.cpp をゼロから<br>
      • ATOM をゼロから
    </td>
  </tr>
</table>

### 02. ファインチューニング — ROCm上でのLLMファインチューニング

<p align="center">
  <strong>🔧 ROCm LLMファインチューニング</strong><br>
  <em>AMD GPU上での効率的なファインチューニング</em><br>
  📖 <strong><a href="../../docs/en/02-fine-tune/index.md">ROCmファインチューニング入門</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • ゼロからのファインチューニングチュートリアル<br>
      • シングルマシンファインチューニングスクリプト<br>
      • マルチノード・マルチGPUファインチューニング
    </td>
  </tr>
</table>

### 03. インフラ — オペレーターとスタックの深掘り

<p align="center">
  <strong>⚙️ ROCmインフラとオペレーター</strong><br>
  <em>ハードウェア/ソフトウェアスタックからHIPレベルの実践まで</em><br>
  📖 <strong><a href="../../docs/en/03-infra/index.md">ROCmインフラ入門</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • HIPifyによる自動マイグレーション<br>
      • BLAS / DNNライブラリのマイグレーション（rocBLAS、MIOpen、…）<br>
      • NCCL → RCCL<br>
      • Nsight → rocprof マッピング
    </td>
  </tr>
</table>

### 04. リファレンス

<p align="center">
  <strong>📚 ROCmリファレンス</strong><br>
  <em>公式およびコミュニティリソース</em><br>
  📖 <strong><a href="../../docs/en/04-references/index.md">ROCmリファレンス</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">ROCm公式ドキュメント</a><br>
      • <a href="https://github.com/amd">AMD on GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">ROCmリリースノート</a><br>
      • <a href="../../docs/en/04-references/index.md#amd-gpu-architecture-whitepapers">AMD GPUアーキテクチャ白書（CDNA / RDNA）</a><br>
      • <a href="../../docs/en/04-references/index.md#frameworks-and-inference-services-rocm-quick-install-links">フレームワークと推論サービスのROCmクイックインストールリンク</a><br>
      • 関連ニュース
    </td>
  </tr>
</table>

### 05. AMD-YES — コミュニティプロジェクトの紹介

<p align="center">
  <strong>✨ AMDプロジェクト紹介</strong><br>
  <em>コミュニティ主導のAMD GPU活用例</em><br>
  📖 <strong><a href="../../docs/en/05-amd-yes/index.md">ROCm AMD-YES入門</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
• toy-cli — 軽量ターミナルLLMアシスタント<br>
      • WeChat「ジャンプジャンプ」とYOLOv10 — ゲームAI実戦（ROCm環境でyolov10をトレーニング・使用）<br>
      • Chat-甄嬛 — 時代劇風対話モデル<br>
      • 旅行プランナー — HelloAgentsエージェントデモ<br>
      • happy-llm — 分散LLMトレーニング
    </td>
  </tr>
</table>

## コントリビューション

&emsp;&emsp;あらゆる種類のコントリビューションを歓迎します：

* チュートリアルの改善や追加
* エラーやバグの修正
* AMDプロジェクトの共有
* アイデアや方向性の提案

&emsp;&emsp;まずは **[規範指南](../../規範指南.md)**（構造、命名、画像 — Qwen3などのチュートリアルに準拠）をお読みいただき、次に **[CONTRIBUTING.md](../../CONTRIBUTING.md)**（Issue、PR、モデルごとのディレクトリ規約）をご確認ください。

&emsp;&emsp;ROCm の使用、モデルのデプロイ、チュートリアルの閲覧中にトラブルシューティングやFAQに関する問題があれば、**[コミュニティディスカッション](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)** にもぜひ参加してください。経験の共有、問題の報告、チュートリアル改善をコミュニティと一緒に進められます。

&emsp;&emsp;長期的にリポジトリのメンテナンスにご協力いただける方は、ご連絡ください。メンテナーとして追加いたします。

## 謝辞

### 主要コントリビューター

- [Zhixue Song (不要葱姜蒜) — プロジェクトリーダー](https://github.com/KMnO4-zx) (Datawhaleメンバー、self-llm / happy-llm プロジェクトリーダー)
- [Yu Chen — プロジェクトリーダー](https://github.com/lucachen) (コンテンツクリエイター、Google Developer Expert in Machine Learning)
- [Sizhou Chen — コントリビューター](https://github.com/jjyaoao) (Datawhaleメンバー、hello-agents プロジェクトリーダー)
- [Jiahang Pan — コントリビューター](https://github.com/amdjiahangpan) (コンテンツクリエイター、AMDソフトウェアエンジニア)
- [Weihong Liu — コントリビューター](https://github.com/Weihong-Liu) (Datawhaleメンバー)
- [Dongbo Hao — コントリビューター](https://github.com/wlkq151172) (Datawhaleメンバー)
- [Muling Ke — コントリビューター](https://github.com/1985312383) (Datawhaleメンバー)

> さらなるコントリビューターを常に歓迎します。

### その他

- アイデアやフィードバックは歓迎します — Issueを開いてください。
- チュートリアルに貢献してくださった皆様に感謝します。
- 本プロジェクトを支援してくださった **AMD University Program** に感謝します。

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>

## ライセンス

[MITライセンス](../../LICENSE)

---

<div align="center">

**AMD AIの未来を共に築きましょう。** 💪

hello-rocmコミュニティより愛を込めて ❤️

</div>
