<div align=center>
  <img src="../../docs/public/images/head.png" >
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*Open Source · Community Driven · Das AMD-KI-Ökosystem zugänglicher machen*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="../ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>
<div align="center">

<a href="https://datawhalechina.github.io/hello-rocm/"><img src="https://img.shields.io/badge/hello--rocm_Vollständiges_Tutorial-Online_testen-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="25"></a>

</div>

&emsp;&emsp;Seit **ROCm 7.10.0** (veröffentlicht am 11. Dezember 2025) kann ROCm nahtlos in Python-Virtual-Umgebungen installiert werden, ähnlich wie CUDA, mit offizieller Unterstützung für **Linux und Windows**. Dies ist ein großer Schritt für AMD im Bereich KI: Lernende und LLM-Enthusiasten sind nicht länger auf NVIDIA-Hardware beschränkt – AMD-GPUs sind eine starke, praktische Wahl.

&emsp;&emsp;Jedoch bedeutet **die Senkung der Hardware-Barriere nicht automatisch, dass der Lernpfad klar wird**. Für Lernende, die bereits LLM-Grundlagen haben und diese auf AMD-GPUs in die Praxis umsetzen möchten, beginnen die wahren Herausforderungen erst: Wie deployt man ein Modell auf AMD-GPU? Wie führt man darauf aufbauend Feinabstimmung und Training durch? Wie versteht man das GPU-Programmiersystem von ROCm und vollzieht die Migration von CUDA zu ROCm? Und schließlich, wie bringt man all diese Fähigkeiten in einer echten, funktionierenden KI-Anwendung zusammen?

&emsp;&emsp;**hello-rocm** wurde genau für diesen Pfad geschaffen. Dieses Projekt deckt systematisch die vollständige Nutzungskette großer Modelle auf der AMD ROCm-Plattform ab und führt Sie von **der Ausführung Ihres ersten Modells** bis zum **Aufbau echter KI-Anwendungen auf AMD-GPUs**, durch jeden wichtigen Schritt einschließlich Feinabstimmung, Training und GPU-Programmierung, wodurch die AMD-GPU nicht nur eine Grafikkarte ist, sondern Ihr echter Einstiegspunkt in die Welt der KI-Entwicklung.

&emsp;&emsp;**Dieses Projekt besteht hauptsächlich aus Tutorials**, damit Studierende und zukünftige Praktiker AMD ROCm strukturiert erlernen können. **Jeder ist herzlich eingeladen, Issues zu eröffnen oder Pull Requests einzureichen**, um das Projekt gemeinsam zu erweitern und zu pflegen.

> &emsp;&emsp;***Lernpfad: Zuerst [00-Umgebung](../../docs/en/00-environment/index.md) abschließen (ROCm + PyTorch + **uv**), dann Bereitstellung und Feinabstimmung, und schließlich Operatoroptimierung und GPU-Programmierung. Sobald Ihre Umgebung funktioniert, ist LM Studio oder vLLM ein guter Startpunkt.***

### hello-rocm Skill: dieses Projekt im KI-Assistenten nutzen

&emsp;&emsp;Wenn du ein KI-Coding-Tool verwendest, das Skills, Rules oder Agent-Konfigurationen unterstützt, kannst du den integrierten **hello-rocm Skill** nutzen. Der Skill verwendet die Repository-Struktur, den Reference-Index, die GPU-Architekturtabelle, Deployment-Tutorials und die Troubleshooting-Liste, um dich zum passenden Dokument und offiziellen Link zu führen.

```text
Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place, such as .claude/skills, .cursor/skills, or .agents/skills, then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.
```

&emsp;&emsp;Versuchen Sie zu fragen: Unterstützt meine AMD-GPU ROCm? Was ist der schnellste Weg, mein erstes lokales LLM auszuführen? Wie installiere ich vLLM/Ollama/llama.cpp auf ROCm? Wie debugge ich, wenn torch.cuda.is_available() False zurückgibt? Siehe den [hello-rocm Skill guide](../../docs/en/04-references/index.md#hello-rocm-skill).

### Neueste Updates

- *2026.5.15:* [*ROCm 7.13.0 Versionshinweise*](https://rocm.docs.amd.com/en/7.13.0-preview/index.html)

- *2026.3.11:* [*ROCm 7.12.0 Versionshinweise*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *2025.12.11:* [*ROCm 7.10.0 Versionshinweise*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### Unterstützte Modelle & Tutorials

<p align="center">
  <strong>✨ Mainstream-LLMs: Umgebung · Multi-Framework-Inferenz · Feinabstimmung ✨</strong><br>
  <em>Einheitliche ROCm-Einrichtung (Windows / Ubuntu) + ROCm 7+ · Modellspezifische Tutorials (wachsend)</em><br>
 <a href="../../docs/en/00-environment/index.md">00 — Umgebungseinrichtung</a>
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
      • <a href="../../docs/en/01-deploy/gemma4/gemma4_model.md">Gemma 4 Modellübersicht</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb">Gemma4 E4B LoRA-Feinabstimmung (TRL, Notebook)</a><br>
    </td>
  </tr>
</table>

## Warum dieses Projekt

&emsp;&emsp;Was ist ROCm?

> ROCm (Radeon Open Compute) ist AMDs offener GPU-Computing-Stack für HPC und maschinelles Lernen. Er ermöglicht die Ausführung paralleler Arbeitslasten auf AMD-GPUs und ist der primäre CUDA-Alternativpfad auf AMD-Hardware.

&emsp;&emsp;Offene LLMs sind allgegenwärtig, doch die meisten Tutorials und Tools setzen den NVIDIA-CUDA-Stack voraus. Entwickler, die AMD wählen, haben oft kein durchgängiges, ROCm-natives Lernmaterial.
&emsp;&emsp;Ab **ROCm 7.10.0** (11. Dezember 2025) entkoppelt AMDs **TheRock**-Arbeit die Compute-Laufzeitumgebung vom Betriebssystem, sodass dieselben ROCm-Schnittstellen auf **Linux und Windows** laufen und ROCm ähnlich wie CUDA in Python-Umgebungen installiert werden kann. ROCm ist nicht länger eine „nur für Linux geeignete Infrastruktur“ – es ist eine plattformübergreifende KI-Compute-Plattform. **hello-rocm** sammelt praktische Anleitungen, damit mehr Menschen AMD-GPUs tatsächlich für Training und Inferenz nutzen können.

&emsp;&emsp;***Wir hoffen, eine Brücke zwischen AMD-GPUs und alltäglichen Entwicklern zu sein – offen, inklusiv und auf eine breitere KI-Zukunft ausgerichtet.***

## Für wen ist es gedacht

&emsp;&emsp;Dieses Projekt könnte für Sie nützlich sein, wenn:

* Sie eine AMD-GPU besitzen und LLMs lokal ausführen möchten;
* Sie auf AMD aufbauen möchten, aber keinen strukturierten ROCm-Lehrplan haben;
* Sie sich für kosteneffizientes Deployment und Inferenz interessieren;
* Sie neugierig auf ROCm sind und praxisorientiertes Lernen bevorzugen.

## Roadmap und Struktur

&emsp;&emsp;Das Repository folgt dem vollständigen ROCm-LLM-Workflow: **einheitliche Basis (00-Environment)**, Deployment, Feintuning und Infra-Themen:

### Repository-Aufbau

```
hello-rocm/
├── docs/                   # VitePress-Dokumentationsquelle
│   ├── zh/                 # Chinesische Dokumentation
│   │   ├── 00-environment/ # ROCm-Basisinstallation & -Konfiguration
│   │   ├── 01-deploy/      # LLM-Deployment auf ROCm
│   │   ├── 02-fine-tune/   # LLM-Feintuning auf ROCm
│   │   ├── 03-infra/       # Operatoroptimierung & GPU-Programmierung
│   │   ├── 04-references/  # Kuratierte ROCm-Referenzen
│   │   └── 05-amd-yes/     # Community-AMD-Projektvorstellungen
│   └── en/                 # Englische Dokumentation
├── src/                    # Quellcode & Skripte
└── assets/                 # Gemeinsame Ressourcen
```

### 00. Umgebung – ROCm-Basis

<p align="center">
  <strong>🛠️ ROCm-Umgebungsinstallation & -Konfiguration</strong><br>
  <em>Einheitliche Basis · ROCm 7.12.0 · Windows / Ubuntu · uv + PyTorch</em><br>
  📖 <strong><a href="../../docs/en/00-environment/index.md">Erste Schritte mit der ROCm-Umgebung</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="../../docs/en/00-environment/rocm-gpu-architecture-table.md">GPU-Architektur & pip-Index-Zuordnung</a><br>
      • Windows 11: Treiber, Sicherheitsvoraussetzungen, Installationsablauf<br>
      • Ubuntu 24.04: uv-basierte Installation und optionales einheitliches Installationsskript<br>
      • Überprüfung, Deinstallation und Wechsel der GPU-Ziele
    </td>
  </tr>
</table>

### 01. Deployment – LLM-Deployment auf ROCm

<p align="center">
  <strong>🚀 ROCm-LLM-Deployment</strong><br>
  <em>Von null zu einem laufenden Modell auf AMD-GPUs</em><br>
  📖 <strong><a href="../../docs/en/01-deploy/index.md">Erste Schritte mit ROCm-Deployment</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • LM Studio von Grund auf<br>
      • vLLM von Grund auf<br>
      • Ollama von Grund auf<br>
      • llama.cpp von Grund auf<br>
      • ATOM von Grund auf
    </td>
  </tr>
</table>

### 02. Feintuning – LLM-Feintuning auf ROCm

<p align="center">
  <strong>🔧 ROCm-LLM-Feintuning</strong><br>
  <em>Effizientes Feintuning auf AMD-GPUs</em><br>
  📖 <strong><a href="../../docs/en/02-fine-tune/index.md">Erste Schritte mit ROCm-Feintuning</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • Feintuning-Tutorials von Grund auf<br>
      • Feintuning-Skripte für einzelne Maschinen<br>
      • Multi-Node-Multi-GPU-Feintuning
    </td>
  </tr>
</table>

### 03. Infra – Operatoroptimierung & GPU-Programmierung

<p align="center">
  <strong>⚙️ ROCm-Operatoroptimierung & GPU-Programmierung</strong><br>
  <em>Von der AMD-AI-Hardwarepanorama bis zu HIP-Operatoren und Leistungsanalyse</em><br>
  📖 <strong><a href="../../docs/en/03-infra/index.md">Erste Schritte mit ROCm-Operatoroptimierung</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • AMD-AI-Hardwarepanorama & ROCm-Ökosystem<br>
      • GPU-Software-Stack & Hardware-Architektur tiefgehend analysiert<br>
      • HIP-Programmierung & handgeschriebene Kernel-Praxis<br>
      • Benutzerdefinierte PyTorch-Operatoren & Autograd-Integration
    </td>
  </tr>
</table>

### 04. Referenzen

<p align="center">
  <strong>📚 ROCm-Referenzen</strong><br>
  <em>Offizielle und Community-Ressourcen</em><br>
  📖 <strong><a href="../../docs/en/04-references/index.md">ROCm-Referenzen</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">Offizielle ROCm-Dokumentation</a><br>
      • <a href="https://github.com/amd">AMD auf GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">ROCm-Versionshinweise</a><br>
      • <a href="../../docs/en/04-references/index.md#amd-gpu-architecture-whitepapers">AMD GPU-Architektur-Whitepapers (CDNA / RDNA)</a><br>
      • <a href="../../docs/en/04-references/index.md#frameworks-and-inference-services-rocm-quick-install-links">ROCm-Schnellinstallationslinks für Frameworks und Inferenzdienste</a><br>
      • Verwandte Neuigkeiten
    </td>
  </tr>
</table>

### 05. AMD-YES – Community-Vorstellungen

<p align="center">
  <strong>✨ AMD-Projektvorstellungen</strong><br>
  <em>Community-getriebene Beispiele auf AMD-GPUs</em><br>
  📖 <strong><a href="../../docs/en/05-amd-yes/index.md">Erste Schritte mit ROCm AMD-YES</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
• toy-cli — leichter terminalbasierter LLM-Assistent<br>
      • Minesweeper Agent — Minesweeper mit lokalem LLM spielen (Agent-Loop-Praxis)<br>
      • WeChat „Jump Jump“ mit YOLOv10 — Game-KI in Aktion (YOLOv10 unter ROCm trainieren und verwenden)<br>
      • Chat-甄嬛 — Dialogmodell im historischen Stil<br>
      • Reiseplaner — HelloAgents-Agenten-Demo<br>
      • happy-llm — verteiltes LLM-Training
    </td>
  </tr>
</table>

## Mitwirken

&emsp;&emsp;Wir begrüßen Beiträge aller Art:

* Verbessern oder ergänzen von Tutorials
* Korrigieren von Fehlern und Bugs
* Teilen Ihrer AMD-Projekte
* Vorschlagen von Ideen und Richtungen

&emsp;&emsp;Bitte lesen Sie den **[Content Guide](../../CONTENT_GUIDE_en.md)** (Struktur, Benennung, Bilder – abgestimmt auf Tutorials wie Qwen3) und dann **[CONTRIBUTING_en.md](../../CONTRIBUTING_en.md)** (Issues, PRs und modellspezifische Verzeichniskonventionen).

&emsp;&emsp;Wenn Sie bei der Nutzung von ROCm, beim Deployment von Modellen oder beim Lesen der Tutorials auf Fehlerbehebungs- oder FAQ-Fragen stoßen, können Sie auch an der **[Community-Diskussion](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)** teilnehmen, um Erfahrungen zu teilen, Probleme zu melden und die Tutorials gemeinsam mit der Community zu verbessern.

&emsp;&emsp;Wenn Sie langfristig zur Pflege des Repos beitragen möchten, melden Sie sich – wir können Sie als Maintainer hinzufügen.

## Danksagungen

### Hauptbeitragende

- [Zhixue Song (不要葱姜蒜) — Projektleitung](https://github.com/KMnO4-zx) (Datawhale-Mitglied; Projektleitung von self-llm und happy-llm)
- [Yu Chen — Projektleitung](https://github.com/lucachen) (Content Creator; Google Developer Expert für Maschinelles Lernen)
- [Sizhou Chen — Beitragender](https://github.com/jjyaoao) (Datawhale-Mitglied; Projektleitung von hello-agents)
- [Jiahang Pan — Beitragender](https://github.com/amdjiahangpan) (Content Creator; AMD-Softwareentwickler)
- [Weihong Liu — Beitragender](https://github.com/Weihong-Liu) (Datawhale-Mitglied)
- [Dongbo Hao — Beitragender](https://github.com/wlkq151172) (Inhaltsersteller; Doktorand der Jiangnan-Universität)
- [Muling Ke — Beitragender](https://github.com/1985312383) (Datawhale-Mitglied)

> Weitere Beitragende sind jederzeit willkommen.

### Sonstige

- Ideen und Feedback sind willkommen – bitte eröffnen Sie Issues.
- Danke an alle, die Tutorials beigetragen haben.
- Danke an das **AMD University Program** für die Unterstützung dieses Projekts.

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>


## Lizenz

[MIT-Lizenz](../../LICENSE)

---

<div align="center">

**Lasst uns gemeinsam die Zukunft der AMD-KI gestalten.** 💪

Mit ❤️ erstellt von der hello-rocm-Community

</div>
