<div align=center>
  <img src="../../docs/public/images/head.png" >
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*오픈소스 · 커뮤니티 주도 · AMD AI 생태계를 더욱 접근하기 쉽게*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>
<div align="center">

<a href="https://datawhalechina.github.io/hello-rocm/"><img src="https://img.shields.io/badge/hello--rocm_전체_튜토리얼-온라인_체험-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="25"></a>

</div>

&emsp;&emsp;**ROCm 7.10.0** (2025년 12월 11일 출시)부터 ROCm은 CUDA처럼 Python 가상 환경에 원활하게 설치할 수 있으며, **Linux와 Windows** 모두 공식 지원합니다. 이는 AMD의 AI 분야에서 중요한 진전입니다. 학습자와 LLM 애호가들은 더 이상 NVIDIA 하드웨어에 제한되지 않으며, AMD GPU는 강력하고 실용적인 선택지가 되었습니다.

&emsp;&emsp;그러나 하드웨어 진입 장벽이 낮아졌다고 해서 학습 경로가 자동으로 명확해지는 것은 아닙니다. 이미 대규모 모델의 기초를 갖추고 AMD GPU에서 실천하고자 하는 학습자에게 진짜 문제는 이제 시작입니다: AMD GPU에서 모델을 어떻게 배포할 것인가? 그 위에서 파인튜닝과 훈련을 어떻게 할 것인가? ROCm의 GPU 프로그래밍 체계를 어떻게 이해하고 CUDA에서 ROCm으로의 마이그레이션을 어떻게 완료할 것인가? 최종적으로 이러한 역량을 어떻게 모아서 실제로 작동하는 AI 애플리케이션으로 만들 것인가?

&emsp;&emsp;**hello-rocm**은 바로 이 경로를 위해 탄생했습니다. 본 프로젝트는 AMD ROCm 플랫폼에서의 대규모 모델 전체 사용 체인을 체계적으로 다루며, **첫 번째 모델을 실행하는 것**부터 **AMD GPU에서 실제 AI 애플리케이션을 구축하는 것**까지, 파인튜닝·훈련부터 GPU 프로그래밍까지의 모든 핵심 단계를 안내합니다—AMD GPU를 단순한 그래픽 카드가 아닌 AI 개발 세계로의 진정한 출발점으로 만듭니다.

&emsp;&emsp;**이 프로젝트는 주로 튜토리얼**로 구성되어 있어 학생과 미래 실무자들이 AMD ROCm을 체계적으로 학습할 수 있습니다. **누구든지 이슈를 열거나 풀 리퀘스트를 제출**하여 프로젝트를 함께 성장시키고 유지할 수 있습니다.

> &emsp;&emsp;***학습 경로: [00-환경설정](../../docs/en/00-environment/index.md)을 먼저 완료하세요 (ROCm + PyTorch + **uv**), 그 다음 배포와 파인튜닝, 마지막으로 오퍼레이터 최적화와 GPU 프로그래밍 주제를 다루세요. 환경이 준비되면 LM Studio나 vLLM이 시작하기 좋은 곳입니다.***

### hello-rocm Skill: AI 도우미에서 이 프로젝트 사용하기

&emsp;&emsp;Skills, Rules 또는 Agent 설정을 지원하는 AI 코딩 도구를 사용한다면 내장된 **hello-rocm Skill** 을 사용할 수 있습니다. 이 Skill은 저장소 구조, Reference 인덱스, GPU 아키텍처 표, 배포 튜토리얼, 문제 해결 목록을 바탕으로 적절한 문서와 공식 링크를 안내합니다.

```text
Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place, such as .claude/skills, .cursor/skills, or .agents/skills, then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.
```

&emsp;&emsp;이렇게 물어보세요: 제 AMD GPU가 ROCm을 지원하나요? 로컬 LLM을 가장 빨리 실행하려면 어떤 글을 봐야 하나요? ROCm에서 vLLM / Ollama / llama.cpp를 어떻게 설치하나요? `torch.cuda.is_available()`이 False를 반환할 때 어떻게 디버깅하나요? 자세한 내용은 [hello-rocm Skill 가이드](../../docs/en/04-references/index.md#hello-rocm-skill)를 참조하세요.

### 최신 업데이트

- *2026.5.15:* [*ROCm 7.13.0 릴리스 노트*](https://rocm.docs.amd.com/en/7.13.0-preview/index.html)

- *2026.3.11:* [*ROCm 7.12.0 릴리스 노트*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *2025.12.11:* [*ROCm 7.10.0 릴리스 노트*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### 지원 모델 및 튜토리얼

<p align="center">
  <strong>✨ 주요 LLM: 환경 · 멀티 프레임워크 추론 · 파인튜닝 ✨</strong><br>
  <em>통합 ROCm 설정 (Windows / Ubuntu) + ROCm 7+ · 모델별 튜토리얼 (계속 확장 중)</em><br>
 <a href="../../docs/en/00-environment/index.md">00 — 환경 설정</a>
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
    <td colspan="2" align="center" style="border: none !important;"><strong>Qwen3.5</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/qwen3.5/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/qwen3.5/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/qwen3.5/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/qwen3.5/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/02-fine-tune/qwen3.5/qwen3.5-4b-lora-swanlab.md">Qwen3.5-4B LoRA + SwanLab</a><br>
      • <a href="../../src/fine-tune/models/qwen3.5/Qwen3.5-4B-LoRA.ipynb">Qwen3.5-4B LoRA (Notebook)</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Gemma4</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/gemma4/gemma4_model.md">Gemma 4 모델 개요</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb">Gemma4 E4B LoRA 파인튜닝 (TRL, Notebook)</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>MiniCPM5</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/minicpm/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>MiniCPM-V 4.6</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/minicpmv/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
      • <a href="../../docs/en/01-deploy/minicpmv/vllm-rocm7-deploy.md">vLLM</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>MiniCPM-o 4.5</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/minicpm-o/minicpm-o-model.md">MiniCPM-o 4.5 모델 소개</a><br>
      • <a href="../../docs/en/01-deploy/minicpm-o/llamacpp-omni-rocm7-deploy.md">llama.cpp-omni 배포 (음성+비전+TTS)</a><br>
      • <a href="../../docs/en/01-deploy/minicpm-o/webdemo-rocm7-deploy.md">Web Demo 전이중 배포</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
    </td>
  </tr>
</table>

## 이 프로젝트의 목적

&emsp;&emsp;ROCm이란 무엇인가요?

> ROCm(Radeon Open Compute)은 HPC 및 머신러닝을 위한 AMD의 오픈 GPU 컴퓨팅 스택입니다. AMD GPU에서 병렬 워크로드를 실행할 수 있게 해주며, AMD 하드웨어에서 주요 CUDA 대안 경로입니다.

&emsp;&emsp;오픈 LLM은 어디에나 있지만, 대부분의 튜토리얼과 도구는 NVIDIA CUDA 스택을 가정합니다. AMD를 선택하는 개발자들은 종종 종단 간 ROCm 네이티브 학습 자료가 부족합니다.
&emsp;&emsp;**ROCm 7.10.0**(2025년 12월 11일)부터 AMD의 **TheRock** 작업은 OS로부터 컴퓨트 런타임을 분리하여 동일한 ROCm 인터페이스가 **Linux와 Windows**에서 실행되고, ROCm을 CUDA와 유사하게 Python 환경에 설치할 수 있게 합니다. ROCm은 더 이상 "Linux 전용 인프라"가 아니라, 크로스 플랫폼 AI 컴퓨트 플랫폼입니다. **hello-rocm**은 더 많은 사람들이 실제로 AMD GPU를 훈련 및 추론에 사용할 수 있도록 실용적인 가이드를 모아 놓았습니다.

&emsp;&emsp;***우리는 AMD GPU와 일상적인 개발자 사이의 다리가 되고자 합니다—개방적이고 포용적이며, 더 넓은 AI 미래를 지향합니다.***

## 대상 독자

&emsp;&emsp;다음과 같은 경우 이 프로젝트가 유용할 수 있습니다:

* AMD GPU를 보유하고 있으며 로컬에서 LLM을 실행하고 싶은 경우
* AMD 기반으로 구축하고 싶지만 체계적인 ROCm 커리큘럼이 부족한 경우
* 비용 효율적인 배포 및 추론에 관심이 있는 경우
* ROCm에 대해 궁금하고 실습을 선호하는 경우

## 로드맵 및 구조

&emsp;&emsp;이 저장소는 전체 ROCm LLM 워크플로우를 따릅니다: **통합 베이스라인(00-환경)**, 배포, 미세 조정, 인프라 스타일 주제:

### 저장소 구조

```
hello-rocm/
├── docs/                   # VitePress 문서 소스
│   ├── zh/                 # 중국어 문서
│   │   ├── 00-environment/ # ROCm 기본 환경 설치 및 설정
│   │   ├── 01-deploy/      # ROCm LLM 배포
│   │   ├── 02-fine-tune/   # ROCm LLM 파인튜닝
│   │   ├── 03-infra/       # 오퍼레이터 최적화 & GPU 프로그래밍
│   │   ├── 04-references/  # 엄선된 ROCm 참고 자료
│   │   └── 05-amd-yes/     # 커뮤니티 AMD 프로젝트 쇼케이스
│   └── en/                 # English docs
├── src/                    # 소스 코드 및 스크립트
└── assets/                 # 공유 에셋
```

### 00. 환경 — ROCm 베이스라인

<p align="center">
  <strong>🛠️ ROCm 환경 설치 및 설정</strong><br>
  <em>단일 베이스라인 · ROCm 7.12.0 · Windows / Ubuntu · uv + PyTorch</em><br>
  📖 <strong><a href="../../docs/en/00-environment/index.md">ROCm 환경 시작하기</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="../../docs/en/00-environment/rocm-gpu-architecture-table.md">GPU 아키텍처 및 pip 인덱스 맵</a><br>
      • Windows 11: 드라이버, 보안 전제 조건, 설치 흐름<br>
      • Ubuntu 24.04: uv 기반 설치 및 선택적 통합 설치 프로그램 스크립트<br>
      • 확인, 제거 및 GPU 대상 전환
    </td>
  </tr>
</table>

### 01. 배포 — ROCm에서의 LLM 배포

<p align="center">
  <strong>🚀 ROCm LLM 배포</strong><br>
  <em>AMD GPU에서 모델 실행까지 처음부터</em><br>
  📖 <strong><a href="../../docs/en/01-deploy/index.md">ROCm 배포 시작하기</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • LM Studio 처음부터 시작하기<br>
      • vLLM 처음부터 시작하기<br>
      • Ollama 처음부터 시작하기<br>
      • llama.cpp 처음부터 시작하기<br>
      • ATOM 처음부터 시작하기
    </td>
  </tr>
</table>

### 02. 미세 조정 — ROCm에서의 LLM 미세 조정

<p align="center">
  <strong>🔧 ROCm LLM 미세 조정</strong><br>
  <em>AMD GPU에서의 효율적인 미세 조정</em><br>
  📖 <strong><a href="../../docs/en/02-fine-tune/index.md">ROCm 미세 조정 시작하기</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • 처음부터 시작하는 미세 조정 튜토리얼<br>
      • 단일 머신 미세 조정 스크립트<br>
      • 다중 노드 다중 GPU 미세 조정
    </td>
  </tr>
</table>

### 03. 인프라 — 연산자 최적화 & GPU 프로그래밍

<p align="center">
  <strong>⚙️ ROCm 연산자 최적화 & GPU 프로그래밍</strong><br>
  <em>AMD AI 하드웨어 전경부터 HIP 연산자 및 성능 분석까지</em><br>
  📖 <strong><a href="../../docs/en/03-infra/index.md">ROCm 연산자 최적화 시작하기</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • AMD AI 하드웨어 전경과 ROCm 생태계<br>
      • GPU 소프트웨어 스택과 하드웨어 아키텍처 심층 분석<br>
      • HIP 프로그래밍 입문과 직접 작성한 커널 실습<br>
      • 커스텀 PyTorch 연산자와 Autograd 통합
    </td>
  </tr>
</table>

### 04. 참고 자료

<p align="center">
  <strong>📚 ROCm 참고 자료</strong><br>
  <em>공식 및 커뮤니티 리소스</em><br>
  📖 <strong><a href="../../docs/en/04-references/index.md">ROCm 참고 자료</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">ROCm 공식 문서</a><br>
      • <a href="https://github.com/amd">AMD on GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">ROCm 릴리스 노트</a><br>
      • <a href="../../docs/en/04-references/index.md#amd-gpu-architecture-whitepapers">AMD GPU 아키텍처 백서(CDNA / RDNA)</a><br>
      • <a href="../../docs/en/04-references/index.md#frameworks-and-inference-services-rocm-quick-install-links">프레임워크 및 추론 서비스 ROCm 빠른 설치 링크</a><br>
      • 관련 뉴스
    </td>
  </tr>
</table>

### 05. AMD-YES — 커뮤니티 쇼케이스

<p align="center">
  <strong>✨ AMD 프로젝트 쇼케이스</strong><br>
  <em>AMD GPU에서의 커뮤니티 주도 예제</em><br>
  📖 <strong><a href="../../docs/en/05-amd-yes/index.md">ROCm AMD-YES 시작하기</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
• toy-cli — 경량 터미널 LLM 어시스턴트<br>
      • Minesweeper Agent — 로컬 LLM으로 지뢰찾기 (Agent 루프 실습)<br>
      • WeChat “Jump Jump” with YOLOv10 — 게임 AI 실전 (ROCm 환경에서 yolov10 훈련 및 사용)<br>
      • Chat-甄嬛 — 시대극 스타일 대화 모델<br>
      • Travel planner — HelloAgents 에이전트 데모<br>
      • Torch-RecHub — 추천 시스템 실전 (CTR, 리콜, 멀티태스크, ONNX 내보내기)<br>
      • happy-llm — 분산 LLM 학습
    </td>
  </tr>
</table>

## 기여하기

&emsp;&emsp;모든 종류의 기여를 환영합니다:

* 튜토리얼 개선 또는 추가
* 오류 및 버그 수정
* AMD 프로젝트 공유
* 아이디어 및 방향 제안

&emsp;&emsp;먼저 **[Content Guide](../../CONTENT_GUIDE_en.md)** (구조, 명명, 이미지 — Qwen3 등 튜토리얼과 일관성 유지)를 읽어주시고, 이후 **[CONTRIBUTING_en.md](../../CONTRIBUTING_en.md)** (이슈, PR 및 모델별 디렉토리 규칙)를 참고해 주세요.

&emsp;&emsp;ROCm 사용, 모델 배포 또는 튜토리얼 학습 중 문제 해결이나 FAQ 관련 이슈가 있다면 **[커뮤니티 토론](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)** 에도 참여해 주세요. 커뮤니티와 함께 경험을 공유하고, 문제를 제보하며, 튜토리얼을 개선할 수 있습니다.

&emsp;&emsp;장기적으로 저장소 유지 관리를 도와주실 분은 연락 주세요. 관리자로 추가해 드리겠습니다.

## 감사의 말

### 핵심 기여자

- [Zhixue Song (不要葱姜蒜) — 프로젝트 리드](https://github.com/KMnO4-zx) (Datawhale 멤버, self-llm 및 happy-llm 프로젝트 리드)
- [Yu Chen — 프로젝트 리드](https://github.com/lucachen) (콘텐츠 크리에이터, Google Developer Expert in Machine Learning)
- [Sizhou Chen — 기여자](https://github.com/jjyaoao) (Datawhale 멤버, hello-agents 프로젝트 리드)
- [Jiahang Pan — 기여자](https://github.com/amdjiahangpan) (콘텐츠 크리에이터, AMD 소프트웨어 엔지니어)
- [Weihong Liu — 기여자](https://github.com/Weihong-Liu) (Datawhale 멤버)
- [Dongbo Hao — 기여자](https://github.com/wlkq151172) (콘텐츠 크리에이터; 장난대학 대학원생)
- [Muling Ke — 기여자](https://github.com/1985312383) (Datawhale 멤버)

> 더 많은 기여자는 언제나 환영합니다.

### 기타

- 아이디어와 피드백은 언제든지 이슈를 열어 주세요.
- 튜토리얼을 기여해 주신 모든 분들께 감사드립니다.
- 이 프로젝트를 지원해 주신 **AMD University Program**에 감사드립니다.

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>


## 라이선스

[MIT License](../../LICENSE)

---

<div align="center">

**AMD AI의 미래를 함께 만들어 갑시다.** 💪

hello-rocm 커뮤니티가 ❤️로 만들었습니다.

</div>
