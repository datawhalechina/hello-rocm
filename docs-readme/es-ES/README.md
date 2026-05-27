<div align=center>
  <img src="../../docs/public/images/head.png" >
  <strong>AMD ¡SÍ! 🚀</strong>
</div>

<div align="center">

*Código Abierto · Impulsado por la Comunidad · Haciendo el Ecosistema de IA de AMD Más Accesible*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="../ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>
<div align="center">

<a href="https://datawhalechina.github.io/hello-rocm/"><img src="https://img.shields.io/badge/hello--rocm_Tutorial_completo-Probar_en_línea-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="25"></a>

</div>

&emsp;&emsp;Desde **ROCm 7.10.0** (lanzado el 11 de diciembre de 2025), ROCm se puede instalar sin problemas en entornos virtuales de Python, de manera similar a CUDA, con soporte oficial tanto para **Linux como para Windows**. Este es un gran paso para AMD en IA: los estudiantes y entusiastas de LLM ya no están limitados al hardware de NVIDIA: las GPU de AMD son una opción sólida y práctica.

&emsp;&emsp;Sin embargo, **reducir la barrera de hardware no significa que el camino de aprendizaje se aclare automáticamente**. Para los estudiantes que ya tienen fundamentos de LLM y quieren ponerlos en práctica en GPUs AMD, los verdaderos desafíos apenas comienzan: ¿Cómo desplegar un modelo en GPU AMD? ¿Cómo hacer ajuste fino y entrenamiento sobre esa base? ¿Cómo entender el sistema de programación GPU de ROCm y completar la migración de CUDA a ROCm? Y finalmente, ¿cómo reunir todas estas capacidades en una aplicación de IA real y funcional?

&emsp;&emsp;**hello-rocm** nació exactamente para este camino. Este proyecto cubre sistemáticamente la cadena completa de uso de modelos grandes en la plataforma AMD ROCm, llevándote desde **ejecutar tu primer modelo** hasta **construir aplicaciones de IA reales en GPUs AMD**, pasando por cada paso clave incluyendo ajuste fino, entrenamiento y programación GPU, haciendo que la GPU AMD no sea solo una tarjeta gráfica, sino tu verdadero punto de partida hacia el mundo del desarrollo de IA.

&emsp;&emsp;**Este proyecto es principalmente tutoriales** para que estudiantes y futuros profesionales puedan aprender AMD ROCm de manera estructurada. **Cualquier persona es bienvenida a abrir issues o enviar pull requests** para crecer y mantener el proyecto juntos.

> &emsp;&emsp;***Ruta de aprendizaje: Completa primero [00-Environment](../../docs/en/00-environment/index.md) (ROCm + PyTorch + **uv**), luego implementación y ajuste fino, y finalmente optimización de operadores y programación GPU. Después de que tu entorno funcione, LM Studio o vLLM es un buen punto de partida.***

### hello-rocm Skill: usa este proyecto dentro de tu asistente de IA

&emsp;&emsp;Si usas una herramienta de programación con IA que admite Skills, Rules o configuración de Agent, puedes usar el **hello-rocm Skill** integrado. El Skill usa la estructura del repositorio, el índice de referencias, la tabla de arquitecturas GPU, los tutoriales de despliegue y la lista de solución de problemas para llevarte al documento y enlace oficial correctos.

```text
Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place, such as .claude/skills, .cursor/skills, or .agents/skills, then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.
```

&emsp;&emsp;Prueba preguntando: ¿Mi GPU AMD soporta ROCm? ¿Cuál es el camino más rápido para ejecutar mi primer LLM local? ¿Cómo instalar vLLM/Ollama/llama.cpp en ROCm? ¿Cómo depurar cuando torch.cuda.is_available() devuelve False? Consulta la [guía de hello-rocm Skill](../../docs/en/04-references/index.md#hello-rocm-skill).

### Últimas actualizaciones

- *2026.5.15:* [*Notas de la versión de ROCm 7.13.0*](https://rocm.docs.amd.com/en/7.13.0-preview/index.html)

- *2026.3.11:* [*Notas de la versión de ROCm 7.12.0*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *2025.12.11:* [*Notas de la versión de ROCm 7.10.0*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### Modelos y tutoriales compatibles

<p align="center">
  <strong>✨ LLMs principales: entorno · inferencia multi-marco · ajuste fino ✨</strong><br>
  <em>Configuración unificada de ROCm (Windows / Ubuntu) + ROCm 7+ · tutoriales por modelo (en crecimiento)</em><br>
 <a href="../../docs/en/00-environment/index.md">00 — Configuración del entorno</a>
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
      • <a href="../../docs/en/01-deploy/gemma4/gemma4_model.md">Resumen del modelo Gemma 4</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb">Ajuste fino LoRA de Gemma4 E4B (TRL, Notebook)</a><br>
    </td>
  </tr>
</table>

## Por qué este proyecto

&emsp;&emsp;¿Qué es ROCm?

> ROCm (Radeon Open Compute) es la pila de computación abierta de GPU de AMD para HPC y aprendizaje automático. Permite ejecutar cargas de trabajo paralelas en GPU de AMD y es la principal ruta alternativa a CUDA en hardware AMD.

&emsp;&emsp;Los LLM abiertos están por todas partes, pero la mayoría de los tutoriales y herramientas asumen la pila NVIDIA CUDA. Los desarrolladores que eligen AMD a menudo carecen de material de aprendizaje integral y nativo de ROCm.
&emsp;&emsp;A partir de **ROCm 7.10.0** (11 de diciembre de 2025), el trabajo **TheRock** de AMD desacopla el runtime de cómputo del sistema operativo, de modo que las mismas interfaces de ROCm funcionan en **Linux y Windows**, y ROCm se puede instalar en entornos Python de manera similar a CUDA. ROCm ya no es una "tubería solo para Linux", sino una plataforma de cómputo de IA multiplataforma. **hello-rocm** recopila guías prácticas para que más personas puedan usar realmente las GPU de AMD para entrenamiento e inferencia.

&emsp;&emsp;***Esperamos ser un puente entre las GPU de AMD y los constructores cotidianos—abierto, inclusivo y orientado a un futuro de IA más amplio.***

## Para quién es

&emsp;&emsp;Puede encontrar este proyecto útil si:

* Tiene una GPU AMD y desea ejecutar LLMs localmente;
* Desea construir sobre AMD pero carece de un plan de estudios estructurado de ROCm;
* Le importa el despliegue y la inferencia rentables;
* Siente curiosidad por ROCm y prefiere el aprendizaje práctico.

## Hoja de ruta y estructura

&emsp;&emsp;El repositorio sigue el flujo de trabajo completo de LLM en ROCm: **línea base unificada (00-Environment)**, despliegue, ajuste fino y temas de tipo Infraestructura:

### Estructura del repositorio

```
hello-rocm/
├── docs/                   # Fuente de documentación VitePress
│   ├── zh/                 # Documentación en chino
│   │   ├── 00-environment/ # Instalación y configuración base de ROCm
│   │   ├── 01-deploy/      # Despliegue de LLM en ROCm
│   │   ├── 02-fine-tune/   # Ajuste fino de LLM en ROCm
│   │   ├── 03-infra/       # Optimización de operadores y programación GPU
│   │   ├── 04-references/  # Referencias seleccionadas de ROCm
│   │   └── 05-amd-yes/     # Muestras de proyectos comunitarios AMD
│   └── en/                 # Documentación en inglés
├── src/                    # Código fuente y scripts
└── assets/                 # Recursos compartidos
```

### 00. Environment — Línea base de ROCm

<p align="center">
  <strong>🛠️ Instalación y configuración del entorno ROCm</strong><br>
  <em>Línea base única · ROCm 7.12.0 · Windows / Ubuntu · uv + PyTorch</em><br>
  📖 <strong><a href="../../docs/en/00-environment/index.md">Primeros pasos con el entorno ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="../../docs/en/00-environment/rocm-gpu-architecture-table.md">Arquitectura de GPU y mapa de índice pip</a><br>
      • Windows 11: controladores, requisitos de seguridad, flujo de instalación<br>
      • Ubuntu 24.04: instalación basada en uv y script de instalación unificado opcional<br>
      • Verificación, desinstalación y cambio de objetivos de GPU
    </td>
  </tr>
</table>

### 01. Deploy — Despliegue de LLM en ROCm

<p align="center">
  <strong>🚀 Despliegue de LLM en ROCm</strong><br>
  <em>De cero a un modelo en ejecución en GPU AMD</em><br>
  📖 <strong><a href="../../docs/en/01-deploy/index.md">Primeros pasos con el despliegue en ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • LM Studio desde cero<br>
      • vLLM desde cero<br>
      • Ollama desde cero<br>
      • llama.cpp desde cero<br>
      • ATOM desde cero
    </td>
  </tr>
</table>

### 02. Fine-tune — Ajuste fino de LLM en ROCm

<p align="center">
  <strong>🔧 Ajuste fino de LLM en ROCm</strong><br>
  <em>Ajuste fino eficiente en GPU AMD</em><br>
  📖 <strong><a href="../../docs/en/02-fine-tune/index.md">Primeros pasos con el ajuste fino en ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • Tutoriales de ajuste fino desde cero<br>
      • Scripts de ajuste fino en una sola máquina<br>
      • Ajuste fino multi-nodo y multi-GPU
    </td>
  </tr>
</table>

### 03. Infra — Optimización de operadores y programación GPU

<p align="center">
  <strong>⚙️ Optimización de operadores ROCm y programación GPU</strong><br>
  <em>Del panorama de hardware AMD AI a operadores HIP y análisis de rendimiento</em><br>
  📖 <strong><a href="../../docs/en/03-infra/index.md">Primeros pasos con la optimización de operadores ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • Panorama de hardware AMD AI y ecosistema ROCm<br>
      • Análisis profundo del stack de software GPU y arquitectura de hardware<br>
      • Introducción a la programación HIP y práctica de kernels escritos a mano<br>
      • Operadores personalizados de PyTorch e integración con Autograd
    </td>
  </tr>
</table>

### 04. References

<p align="center">
  <strong>📚 Referencias de ROCm</strong><br>
  <em>Recursos oficiales y de la comunidad</em><br>
  📖 <strong><a href="../../docs/en/04-references/index.md">Referencias de ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">Documentación oficial de ROCm</a><br>
      • <a href="https://github.com/amd">AMD en GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">Notas de la versión de ROCm</a><br>
      • <a href="../../docs/en/04-references/index.md#amd-gpu-architecture-whitepapers">Whitepapers de arquitectura GPU AMD (CDNA / RDNA)</a><br>
      • <a href="../../docs/en/04-references/index.md#frameworks-and-inference-services-rocm-quick-install-links">Enlaces rápidos de instalación ROCm para frameworks y servicios de inferencia</a><br>
      • Noticias relacionadas
    </td>
  </tr>
</table>

### 05. AMD-YES — Muestras comunitarias

<p align="center">
  <strong>✨ Muestras de proyectos AMD</strong><br>
  <em>Ejemplos impulsados por la comunidad en GPU AMD</em><br>
  📖 <strong><a href="../../docs/en/05-amd-yes/index.md">Primeros pasos con AMD-YES en ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
• toy-cli — asistente ligero de LLM para terminal<br>
      • “Jump Jump” de WeChat con YOLOv10 — IA de juegos en acción (Entrenar y usar yolov10 en ROCm)<br>
      • Chat-甄嬛 — modelo de diálogo en estilo de época<br>
      • Planificador de viajes — demo del agente HelloAgents<br>
      • happy-llm — entrenamiento distribuido de LLM
    </td>
  </tr>
</table>

## Contribuciones

&emsp;&emsp;Aceptamos todo tipo de contribuciones:

* Mejorar o añadir tutoriales
* Corregir errores y fallos
* Compartir tus proyectos con AMD
* Sugerir ideas y direcciones

&emsp;&emsp;Por favor, lee **[规范指南](../../规范指南.md)** (estructura, nomenclatura, imágenes—alineado con tutoriales como Qwen3), y luego **[CONTRIBUTING.md](../../CONTRIBUTING.md)** (issues, PRs y convenciones de directorios por modelo).

&emsp;&emsp;Si encuentras problemas de solución de problemas o FAQ al usar ROCm, desplegar modelos o leer los tutoriales, también puedes unirte a la **[discusión comunitaria](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)** para compartir experiencias, reportar problemas y mejorar los tutoriales con la comunidad.

&emsp;&emsp;Si deseas ayudar a mantener el repositorio a largo plazo, contáctanos—podemos agregarte como mantenedor.

## Agradecimientos

### Contribuyentes principales

- [Zhixue Song (不要葱姜蒜) — líder del proyecto](https://github.com/KMnO4-zx) (miembro de Datawhale; líder de los proyectos self-llm y happy-llm)
- [Yu Chen — líder del proyecto](https://github.com/lucachen) (creador de contenido; Google Developer Expert en Machine Learning)
- [Sizhou Chen — contribuyente](https://github.com/jjyaoao) (miembro de Datawhale; líder del proyecto hello-agents)
- [Jiahang Pan — contribuyente](https://github.com/amdjiahangpan) (creador de contenido; ingeniero de software en AMD)
- [Weihong Liu — contribuyente](https://github.com/Weihong-Liu) (miembro de Datawhale)
- [Dongbo Hao — contribuyente](https://github.com/wlkq151172) (creador de contenido; estudiante de posgrado de la Universidad Jiangnan)
- [Muling Ke — contribuyente](https://github.com/1985312383) (miembro de Datawhale)

> Siempre son bienvenidos más contribuyentes.

### Otros

- Las ideas y comentarios son bienvenidos—por favor, abre issues.
- Gracias a todos los que han contribuido con tutoriales.
- Gracias al **Programa Universitario de AMD** por apoyar este proyecto.

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>

## Licencia

[Licencia MIT](../../LICENSE)

---

<div align="center">

**Construyamos juntos el futuro de la IA de AMD.** 💪

Hecho con ❤️ por la comunidad hello-rocm

</div>
