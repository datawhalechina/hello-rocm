<div align=center>
  <img src="../../docs/public/images/head.png" >
  <strong>AMD YES ! 🚀</strong>
</div>

<div align="center">

*Open Source · Piloté par la communauté · Rendre l'écosystème IA d'AMD plus accessible*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="../ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>
<div align="center">

<a href="https://datawhalechina.github.io/hello-rocm/"><img src="https://img.shields.io/badge/hello--rocm_Tutoriel_complet-Essayer_en_ligne-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="25"></a>

</div>

&emsp;&emsp;Depuis **ROCm 7.10.0** (publiée le 11 décembre 2025), ROCm peut être installé de manière transparente dans des environnements virtuels Python, un peu comme CUDA, avec un support officiel pour **Linux et Windows**. C'est un grand pas en avant pour AMD dans le domaine de l'IA : les apprenants et les passionnés de LLM ne sont plus limités au matériel NVIDIA — les GPU AMD sont un choix solide et pratique.

&emsp;&emsp;Cependant, **abaisser la barrière matérielle ne clarifie pas automatiquement le parcours d'apprentissage**. Pour les apprenants qui ont déjà des bases en LLM et souhaitent les mettre en pratique sur des GPU AMD, les vrais défis ne font que commencer : Comment déployer un modèle sur GPU AMD ? Comment effectuer du fine-tuning et de l'entraînement sur cette base ? Comment comprendre le système de programmation GPU de ROCm et réaliser la migration de CUDA vers ROCm ? Et finalement, comment rassembler toutes ces capacités dans une application IA réelle et fonctionnelle ?

&emsp;&emsp;**hello-rocm** est né précisément pour ce parcours. Ce projet couvre systématiquement la chaîne d'utilisation complète des grands modèles sur la plateforme AMD ROCm, vous emmenant depuis **l'exécution de votre premier modèle** jusqu'à **la construction d'applications IA réelles sur GPU AMD**, en passant par chaque étape clé incluant le fine-tuning, l'entraînement et la programmation GPU, faisant du GPU AMD non pas seulement une carte graphique, mais votre véritable point de départ dans le monde du développement IA.

&emsp;&emsp;**Ce projet est principalement un ensemble de tutoriels** afin que les étudiants et les futurs praticiens puissent apprendre AMD ROCm de manière structurée. **Toute personne est la bienvenue pour ouvrir des issues ou soumettre des pull requests** afin de développer et maintenir le projet ensemble.

> &emsp;&emsp;***Parcours d'apprentissage : Commencez par [00-Environnement](../../docs/en/00-environment/index.md) (ROCm + PyTorch + **uv**), puis le déploiement et le fine-tuning, et enfin l'optimisation d'opérateurs et la programmation GPU. Une fois votre environnement opérationnel, LM Studio ou vLLM est un bon point de départ.***

### hello-rocm Skill : utiliser ce projet dans votre assistant IA

&emsp;&emsp;Si vous utilisez un outil de codage IA compatible avec Skills, Rules ou une configuration Agent, vous pouvez utiliser le **hello-rocm Skill** intégré. Il s’appuie sur la structure du dépôt, l’index des références, la table d’architecture GPU, les tutoriels de déploiement et la liste de dépannage pour vous orienter vers le bon document et le bon lien officiel.

```text
Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place, such as .claude/skills, .cursor/skills, or .agents/skills, then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.
```

&emsp;&emsp;Essayez de demander : Mon GPU AMD supporte-t-il ROCm ? Quel est le chemin le plus rapide pour exécuter mon premier LLM local ? Comment installer vLLM/Ollama/llama.cpp sur ROCm ? Comment déboguer quand torch.cuda.is_available() renvoie False ? Voir le [guide hello-rocm Skill](../../docs/en/04-references/index.md#hello-rocm-skill).

### Dernières mises à jour

- *15 mai 2026 :* [*Notes de version de ROCm 7.13.0*](https://rocm.docs.amd.com/en/7.13.0-preview/index.html)

- *11 mars 2026 :* [*Notes de version de ROCm 7.12.0*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *11 décembre 2025 :* [*Notes de version de ROCm 7.10.0*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### Modèles et tutoriels pris en charge

<p align="center">
  <strong>✨ LLM grand public : environnement · inférence multi-cadres · fine-tuning ✨</strong><br>
  <em>Configuration ROCm unifiée (Windows / Ubuntu) + ROCm 7+ · tutoriels par modèle (en croissance)</em><br>
 <a href="../../docs/en/00-environment/index.md">00 — Configuration de l'environnement</a>
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
      • <a href="../../docs/en/01-deploy/gemma4/gemma4_model.md">Présentation du modèle Gemma 4</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb">Fine-tuning LoRA de Gemma4 E4B (TRL, Notebook)</a><br>
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
      • <a href="../../docs/en/01-deploy/minicpm-o/minicpm-o-model.md">Présentation du modèle MiniCPM-o 4.5</a><br>
      • <a href="../../docs/en/01-deploy/minicpm-o/llamacpp-omni-rocm7-deploy.md">Déploiement llama.cpp-omni (voix + vision + TTS)</a><br>
      • <a href="../../docs/en/01-deploy/minicpm-o/webdemo-rocm7-deploy.md">Déploiement Web Demo full-duplex</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
    </td>
  </tr>
</table>

## Pourquoi ce projet

&emsp;&emsp;Qu'est-ce que ROCm ?

> ROCm (Radeon Open Compute) est la pile de calcul GPU ouverte d'AMD pour le HPC et l'apprentissage automatique. Elle permet d'exécuter des charges de travail parallèles sur les GPU AMD et constitue la principale alternative à CUDA sur le matériel AMD.

&emsp;&emsp;Les LLM ouverts sont partout, mais la plupart des tutoriels et des outils supposent l'utilisation de la pile NVIDIA CUDA. Les développeurs qui choisissent AMD manquent souvent de matériel d'apprentissage complet et natif ROCm.
&emsp;&emsp;À partir de **ROCm 7.10.0** (11 décembre 2025), le projet **TheRock** d’AMD dissocie l’environnement d’exécution de calcul du système d’exploitation, de sorte que les mêmes interfaces ROCm fonctionnent sous **Linux et Windows**, et que ROCm puisse être installé dans des environnements Python de la même manière que CUDA. ROCm n’est plus une « simple tuyauterie Linux » — c’est une plateforme de calcul IA multiplateforme. **hello-rocm** rassemble des guides pratiques pour permettre à davantage de personnes d’utiliser les GPU AMD pour l’entraînement et l’inférence.

&emsp;&emsp;***Nous espérons être un pont entre les GPU AMD et les constructeurs du quotidien — ouvert, inclusif et tourné vers un avenir de l’IA plus large.***

## À qui s’adresse ce projet

&emsp;&emsp;Ce projet peut vous être utile si vous :

* Possédez un GPU AMD et souhaitez exécuter des LLM localement ;
* Souhaitez construire sur AMD mais manquez d’un cursus ROCm structuré ;
* Vous intéressez au déploiement et à l’inférence économiques ;
* Êtes curieux à propos de ROCm et préférez l’apprentissage pratique.

## Feuille de route et structure

&emsp;&emsp;Le dépôt suit le workflow complet ROCm pour les LLM : **base unifiée (00-Environment)**, déploiement, fine-tuning et sujets de type Infra :

### Organisation du dépôt

```
hello-rocm/
├── docs/                   # Source de documentation VitePress
│   ├── zh/                 # Documentation en chinois
│   │   ├── 00-environment/ # Installation et configuration de base ROCm
│   │   ├── 01-deploy/      # Déploiement de LLM sur ROCm
│   │   ├── 02-fine-tune/   # Fine-tuning de LLM sur ROCm
│   │   ├── 03-infra/       # Optimisation d'opérateurs et programmation GPU
│   │   ├── 04-references/  # Références ROCm sélectionnées
│   │   └── 05-amd-yes/     # Projets communautaires AMD
│   └── en/                 # Documentation en anglais
├── src/                    # Code source et scripts
└── assets/                 # Ressources partagées
```

### 00. Environnement — Base ROCm

<p align="center">
  <strong>🛠️ Installation et configuration de l’environnement ROCm</strong><br>
  <em>Base unique · ROCm 7.12.0 · Windows / Ubuntu · uv + PyTorch</em><br>
  📖 <strong><a href="../../docs/en/00-environment/index.md">Démarrer avec l’environnement ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="../../docs/en/00-environment/rocm-gpu-architecture-table.md">Tableau des architectures GPU et correspondance pip</a><br>
      • Windows 11 : pilotes, prérequis de sécurité, procédure d’installation<br>
      • Ubuntu 24.04 : installation basée sur uv et script d’installation unifié optionnel<br>
      • Vérification, désinstallation et changement de cible GPU
    </td>
  </tr>
</table>

### 01. Déploiement — Déploiement de LLM sur ROCm

<p align="center">
  <strong>🚀 Déploiement de LLM sur ROCm</strong><br>
  <em>Du zéro à un modèle en fonctionnement sur GPU AMD</em><br>
  📖 <strong><a href="../../docs/en/01-deploy/index.md">Démarrer avec le déploiement ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • LM Studio à partir de zéro<br>
      • vLLM à partir de zéro<br>
      • Ollama à partir de zéro<br>
      • llama.cpp à partir de zéro<br>
      • ATOM à partir de zéro
    </td>
  </tr>
</table>

### 02. Fine-tuning — Fine-tuning de LLM sur ROCm

<p align="center">
  <strong>🔧 Fine-tuning de LLM sur ROCm</strong><br>
  <em>Fine-tuning efficace sur GPU AMD</em><br>
  📖 <strong><a href="../../docs/en/02-fine-tune/index.md">Démarrer avec le fine-tuning ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • Tutoriels de fine-tuning à partir de zéro<br>
      • Scripts de fine-tuning sur machine unique<br>
      • Fine-tuning multi-nœuds et multi-GPU
    </td>
  </tr>
</table>

### 03. Infra — Optimisation d’opérateurs et programmation GPU

<p align="center">
  <strong>⚙️ Optimisation d’opérateurs ROCm et programmation GPU</strong><br>
  <em>Du panorama matériel AMD AI aux opérateurs HIP et à l’analyse de performances</em><br>
  📖 <strong><a href="../../docs/en/03-infra/index.md">Démarrer avec l’optimisation d’opérateurs ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • Panorama matériel AMD AI et écosystème ROCm<br>
      • Analyse approfondie de la pile logicielle GPU et de l’architecture matérielle<br>
      • Introduction à la programmation HIP et pratique de noyaux écrits à la main<br>
      • Opérateurs PyTorch personnalisés et intégration Autograd
    </td>
  </tr>
</table>

### 04. Références

<p align="center">
  <strong>📚 Références ROCm</strong><br>
  <em>Ressources officielles et communautaires</em><br>
  📖 <strong><a href="../../docs/en/04-references/index.md">Références ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">Documentation officielle ROCm</a><br>
      • <a href="https://github.com/amd">AMD sur GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">Notes de version ROCm</a><br>
      • <a href="../../docs/en/04-references/index.md#amd-gpu-architecture-whitepapers">Livres blancs sur l’architecture GPU AMD (CDNA / RDNA)</a><br>
      • <a href="../../docs/en/04-references/index.md#frameworks-and-inference-services-rocm-quick-install-links">Liens rapides d’installation ROCm pour frameworks et services d’inférence</a><br>
      • Actualités connexes
    </td>
  </tr>
</table>

### 05. AMD-YES — Vitrines communautaires

<p align="center">
  <strong>✨ Vitrines de projets AMD</strong><br>
  <em>Exemples pilotés par la communauté sur GPU AMD</em><br>
  📖 <strong><a href="../../docs/en/05-amd-yes/index.md">Démarrer avec AMD-YES sur ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
• toy-cli — assistant LLM léger en terminal<br>
      • Minesweeper Agent — jouer au démineur avec un LLM local (pratique boucle Agent)<br>
      • « Jump Jump » de WeChat avec YOLOv10 — IA de jeu en action (Entraîner et utiliser yolov10 sous ROCm)<br>
      • Chat-甄嬛 — modèle de dialogue de style historique<br>
      • Planificateur de voyage — démo d’agent HelloAgents<br>
      • Torch-RecHub — systèmes de recommandation (CTR, rappel, multi-tâches, export ONNX)<br>
      • happy-llm — apprentissage distribué de LLM
    </td>
  </tr>
</table>

## Contribution

&emsp;&emsp;Nous accueillons toutes sortes de contributions :

* Améliorer ou ajouter des tutoriels
* Corriger des erreurs et des bugs
* Partager vos projets AMD
* Suggérer des idées et des orientations

&emsp;&emsp;Veuillez lire le **[Content Guide](../../CONTENT_GUIDE_en.md)** (structure, nommage, images — aligné sur les tutoriels comme Qwen3), puis **[CONTRIBUTING_en.md](../../CONTRIBUTING_en.md)** (issues, PRs et conventions par répertoire de modèle).

&emsp;&emsp;Si vous rencontrez des questions de dépannage ou de FAQ en utilisant ROCm, en déployant des modèles ou en lisant les tutoriels, vous pouvez aussi rejoindre la **[discussion communautaire](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)** afin de partager vos retours, signaler des problèmes et améliorer les tutoriels avec la communauté.

&emsp;&emsp;Si vous souhaitez aider à maintenir le dépôt à long terme, contactez-nous — nous pouvons vous ajouter en tant que mainteneur.

## Remerciements

### Contributeurs principaux

- [Zhixue Song (不要葱姜蒜) — responsable du projet](https://github.com/KMnO4-zx) (membre Datawhale ; responsable des projets self-llm et happy-llm)
- [Yu Chen — responsable du projet](https://github.com/lucachen) (créateur de contenu ; Google Developer Expert en Machine Learning)
- [Sizhou Chen — contributeur](https://github.com/jjyaoao) (membre Datawhale ; responsable du projet hello-agents)
- [Jiahang Pan — contributeur](https://github.com/amdjiahangpan) (créateur de contenu ; ingénieur logiciel AMD)
- [Weihong Liu — contributeur](https://github.com/Weihong-Liu) (membre Datawhale)
- [Dongbo Hao — contributeur](https://github.com/wlkq151172) (créateur de contenu ; doctorant à l'Université Jiangnan)
- [Muling Ke — contributeur](https://github.com/1985312383) (membre Datawhale)

> D’autres contributeurs sont toujours les bienvenus.

### Autres

- Les idées et retours sont les bienvenus — veuillez ouvrir des issues.
- Merci à tous ceux qui ont contribué aux tutoriels.
- Merci au **Programme universitaire AMD** pour son soutien à ce projet.

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>

## Licence

[Licence MIT](../../LICENSE)

---

<div align="center">

**Construisons ensemble l’avenir de l’IA AMD.** 💪

Fait avec ❤️ par la communauté hello-rocm

</div>
