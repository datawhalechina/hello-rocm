<div align=center>
  <img src="../../docs/public/images/head.png" >
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*مفتوح المصدر · بقيادة المجتمع · جعل نظام AMD البيئي للذكاء الاصطناعي أكثر سهولة*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="../ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>
<div align="center">

<a href="https://datawhalechina.github.io/hello-rocm/"><img src="https://img.shields.io/badge/hello--rocm_الدليل_الكامل-تجربة_مباشرة-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="25"></a>

</div>

&emsp;&emsp;منذ **ROCm 7.10.0** (الذي صدر في 11 ديسمبر 2025)، يمكن تثبيت ROCm بسلاسة في البيئات الافتراضية Python تمامًا مثل CUDA، مع دعم رسمي لكل من **Linux وWindows**. هذه خطوة كبيرة من AMD في مجال الذكاء الاصطناعي: لم يعد المتعلمون وعشاق LLM مقيدين بأجهزة NVIDIA—بطاقات AMD GPU أصبحت خيارًا قويًا وعمليًا.

&emsp;&emsp;ومع ذلك، فإن خفض حاجز الأجهزة لا يعني تلقائيًا توضيح مسار التعلم. بالنسبة للمتعلمين الذين لديهم بالفعل أساسيات LLM ويريدون تطبيقها على معالجات AMD الرسومية، فإن التحديات الحقيقية تبدأ للتو: كيفية نشر نموذج على معالج AMD الرسومي؟ كيفية الضبط الدقيق والتدريب على ذلك؟ كيفية فهم نظام برمجة GPU الخاص بـ ROCm وإكمال الترحيل من CUDA إلى ROCm؟ وفي النهاية، كيفية جمع كل هذه القدرات معًا في تطبيق ذكاء اصطناعي حقيقي وعملي؟

&emsp;&emsp;ولد **hello-rocm** بالضبط لهذا المسار. يغطي هذا المشروع بشكل منهجي سلسلة الاستخدام الكاملة للنماذج الكبيرة على منصة AMD ROCm، ويأخذك من تشغيل نموذجك الأول إلى بناء تطبيقات ذكاء اصطناعي حقيقية على معالجات AMD الرسومية، من خلال كل خطوة رئيسية بما في ذلك الضبط الدقيق والتدريب وبرمجة GPU—مما يجعل معالج AMD الرسومي ليس مجرد بطاقة رسومات، بل نقطة انطلاقك الحقيقية إلى عالم تطوير الذكاء الاصطناعي.

&emsp;&emsp;**هذا المشروع هو في الأساس دروس تعليمية** حتى يتمكن الطلاب والممارسون المستقبليون من تعلم AMD ROCm بطريقة منظمة. **نرحب بأي شخص لفتح مشكلات أو تقديم طلبات سحب** للمساهمة في نمو المشروع وصيانته معًا.

> &emsp;&emsp;***مسار التعلم: أكمل [00-Environment](../../docs/en/00-environment/index.md) أولاً (ROCm + PyTorch + **uv**)، ثم النشر والضبط الدقيق، وأخيرًا تحسين المشغلات وبرمجة GPU. بعد أن تعمل بيئتك، يعد LM Studio أو vLLM نقطة بداية جيدة.***

### hello-rocm Skill: استخدم هذا المشروع داخل مساعد الذكاء الاصطناعي

&emsp;&emsp;إذا كنت تستخدم أداة برمجة بالذكاء الاصطناعي تدعم Skills أو Rules أو إعدادات Agent، يمكنك استخدام **hello-rocm Skill** المدمج. يعتمد هذا Skill على بنية المستودع وفهرس المراجع وجدول معماريات GPU ودروس النشر وقائمة استكشاف الأخطاء لتوجيهك إلى المستند والرابط الرسمي المناسب.

```text
Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place, such as .claude/skills, .cursor/skills, or .agents/skills, then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.
```

&emsp;&emsp;جرب السؤال: هل يدعم معالج AMD الرسومي الخاص بي ROCm؟ ما هو أسرع مسار لتشغيل أول LLM محلي لي؟ كيفية تثبيت vLLM/Ollama/llama.cpp على ROCm؟ كيفية تصحيح `torch.cuda.is_available()` الذي يعيد False؟ راجع [دليل hello-rocm Skill](../../docs/en/04-references/index.md#hello-rocm-skill).

### آخر التحديثات

- *2026.5.15:* [*ملاحظات إصدار ROCm 7.13.0*](https://rocm.docs.amd.com/en/7.13.0-preview/index.html)

- *2026.3.11:* [*ملاحظات إصدار ROCm 7.12.0*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *2025.12.11:* [*ملاحظات إصدار ROCm 7.10.0*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### النماذج والدروس المدعومة

<p align="center">
  <strong>✨ نماذج LLM الرئيسية: البيئة · الاستدلال متعدد الأطر · الضبط الدقيق ✨</strong><br>
  <em>إعداد ROCm الموحد (Windows / Ubuntu) + ROCm 7+ · دروس لكل نموذج (قيد النمو)</em><br>
 <a href="../../docs/en/00-environment/index.md">00 — إعداد البيئة</a>
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
      • <a href="../../src/fine-tune/models/qwen3/01-Qwen3-8B-LoRA.ipynb">Qwen3-8B LoRA (دفتر)</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Gemma4</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/gemma4/gemma4_model.md">نظرة عامة على نموذج Gemma 4</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb">الضبط الدقيق LoRA لنموذج Gemma4 E4B (TRL، دفتر)</a><br>
    </td>
  </tr>
</table>

## لماذا هذا المشروع

&emsp;&emsp;ما هو ROCm؟

> ROCm (Radeon Open Compute) هي مجموعة الحوسبة المفتوحة لوحدات معالجة الرسوميات من AMD للحوسبة عالية الأداء والتعلم الآلي. تتيح لك تشغيل أعباء العمل المتوازية على وحدات معالجة الرسوميات AMD وهي المسار البديل الرئيسي لـ CUDA على أجهزة AMD.

&emsp;&emsp;نماذج LLM المفتوحة في كل مكان، ومع ذلك تفترض معظم الدروس والأدوات مجموعة NVIDIA CUDA. غالبًا ما يفتقر المطورون الذين يختارون AMD إلى مواد تعليمية شاملة وموجهة لـ ROCm.
&emsp;&emsp;اعتبارًا من **ROCm 7.10.0** (11 ديسمبر 2025)، يعمل مشروع **TheRock** من AMD على فصل بيئة التشغيل الحاسوبية عن نظام التشغيل، مما يسمح لواجهات ROCm نفسها بالعمل على **لينكس وويندوز**، ويمكن تثبيت ROCm في بيئات بايثون على غرار CUDA. لم يعد ROCm مجرد "بنية تحتية خاصة بلينكس" بل أصبح منصة حوسبة ذكاء اصطناعي متعددة المنصات. يجمع **hello-rocm** أدلة عملية لتمكين المزيد من الأشخاص من استخدام معالجات AMD الرسومية في التدريب والاستدلال.

&emsp;&emsp;***نأمل أن نكون جسرًا بين معالجات AMD الرسومية والمطورين اليوميين—منفتحين، شاملين، وموجهين نحو مستقبل أوسع للذكاء الاصطناعي.***

## لمن هذا المشروع

&emsp;&emsp;قد تجد هذا المشروع مفيدًا إذا كنت:

* تمتلك معالج AMD رسوميًا وترغب في تشغيل نماذج اللغة الكبيرة محليًا؛
* ترغب في البناء على AMD ولكنك تفتقر إلى منهج ROCm منظم؛
* تهتم بالنشر والاستدلال الفعال من حيث التكلفة؛
* لديك فضول حول ROCm وتفضل التعلم العملي.

## خريطة الطريق والهيكل

&emsp;&emsp;يتبع المستودع سير عمل ROCm الكامل لنماذج اللغة الكبيرة: **أساس موحد (00-Environment)**، النشر، الضبط الدقيق، ومواضيع البنية التحتية:

### هيكل المستودع

```
hello-rocm/
├── docs/                   # مصدر وثائق VitePress
│   ├── zh/                 # الوثائق الصينية
│   │   ├── 00-environment/ # تثبيت وتكوين ROCm الأساسي
│   │   ├── 01-deploy/      # نشر LLM على ROCm
│   │   ├── 02-fine-tune/   # ضبط دقيق لـ LLM على ROCm
│   │   ├── 03-infra/       # تحسين المشغلات وبرمجة GPU
│   │   ├── 04-references/  # مراجع ROCm منسقة
│   │   └── 05-amd-yes/     # عروض مشاريع AMD المجتمعية
│   └── en/                 # الوثائق الإنجليزية
├── src/                    # الكود المصدري والنصوص البرمجية
└── assets/                 # الأصول المشتركة
```

### 00. البيئة — أساس ROCm

<p align="center">
  <strong>🛠️ تثبيت وتكوين بيئة ROCm</strong><br>
  <em>أساس واحد · ROCm 7.12.0 · ويندوز / أوبونتو · uv + PyTorch</em><br>
  📖 <strong><a href="../../docs/en/00-environment/index.md">بدء استخدام بيئة ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="../../docs/en/00-environment/rocm-gpu-architecture-table.md">خريطة بنية المعالج الرسومي وفهرس pip</a><br>
      • ويندوز 11: برامج التشغيل، المتطلبات الأمنية، خطوات التثبيت<br>
      • أوبونتو 24.04: تثبيت قائم على uv ونص تثبيت موحد اختياري<br>
      • التحقق، الإلغاء، وتبديل أهداف المعالج الرسومي
    </td>
  </tr>
</table>

### 01. النشر — نشر نماذج اللغة الكبيرة على ROCm

<p align="center">
  <strong>🚀 نشر نماذج اللغة الكبيرة على ROCm</strong><br>
  <em>من الصفر إلى نموذج قيد التشغيل على معالجات AMD الرسومية</em><br>
  📖 <strong><a href="../../docs/en/01-deploy/index.md">بدء استخدام نشر ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • LM Studio من الصفر<br>
      • vLLM من الصفر<br>
      • Ollama من الصفر<br>
      • llama.cpp من الصفر<br>
      • ATOM من الصفر
    </td>
  </tr>
</table>

### 02. الضبط الدقيق — ضبط دقيق لنماذج اللغة الكبيرة على ROCm

<p align="center">
  <strong>🔧 ضبط دقيق لنماذج اللغة الكبيرة على ROCm</strong><br>
  <em>ضبط دقيق فعال على معالجات AMD الرسومية</em><br>
  📖 <strong><a href="../../docs/en/02-fine-tune/index.md">بدء استخدام الضبط الدقيق على ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • دروس تعليمية للضبط الدقيق من الصفر<br>
      • نصوص ضبط دقيق لجهاز واحد<br>
      • ضبط دقيق متعدد العقد ومتعدد المعالجات الرسومية
    </td>
  </tr>
</table>

### 03. البنية التحتية — تحسين المشغلات وبرمجة GPU

<p align="center">
  <strong>⚙️ تحسين مشغلات ROCm وبرمجة GPU</strong><br>
  <em>من بانوراما أجهزة AMD AI إلى مشغلات HIP وتحليل الأداء</em><br>
  📖 <strong><a href="../../docs/en/03-infra/index.md">البدء مع تحسين مشغلات ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • بانوراما أجهزة AMD AI وبيئة ROCm<br>
      • تحليل معمق لمكدس برمجيات GPU وبنية الأجهزة<br>
      • مقدمة برمجة HIP وكتابة النوى يدويًا<br>
      • مشغلات PyTorch المخصصة وتكامل Autograd
    </td>
  </tr>
</table>

### 04. المراجع

<p align="center">
  <strong>📚 مراجع ROCm</strong><br>
  <em>موارد رسمية ومجتمعية</em><br>
  📖 <strong><a href="../../docs/en/04-references/index.md">مراجع ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">توثيق ROCm الرسمي</a><br>
      • <a href="https://github.com/amd">AMD على GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">ملاحظات إصدار ROCm</a><br>
      • <a href="../../docs/en/04-references/index.md#amd-gpu-architecture-whitepapers">الأوراق البيضاء لمعمارية AMD GPU ‏(CDNA / RDNA)</a><br>
      • <a href="../../docs/en/04-references/index.md#frameworks-and-inference-services-rocm-quick-install-links">روابط التثبيت السريع لـ ROCm للأطر وخدمات الاستدلال</a><br>
      • أخبار ذات صلة
    </td>
  </tr>
</table>

### 05. AMD-YES — عروض المجتمع

<p align="center">
  <strong>✨ عروض مشاريع AMD</strong><br>
  <em>أمثلة يقودها المجتمع على معالجات AMD الرسومية</em><br>
  📖 <strong><a href="../../docs/en/05-amd-yes/index.md">بدء استخدام AMD-YES على ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
• toy-cli — مساعد طرفية خفيف الوزن لنماذج اللغة<br>
      • Minesweeper Agent — لعب كاسحة الألغام مع نموذج لغة محلي (ممارسة حلقة Agent)<br>
      • WeChat “Jump Jump” مع YOLOv10 — ذكاء اصطناعي للألعاب عملياً (تدريب واستخدام yolov10 على ROCm)<br>
      • Chat-甄嬛 — نموذج حوار بأسلوب تاريخي<br>
      • مخطط السفر — عرض توضيحي لعامل HelloAgents<br>
      • happy-llm — تدريب موزع لنماذج اللغة الكبيرة
    </td>
  </tr>
</table>

## المساهمة

&emsp;&emsp;نرحب بجميع أنواع المساهمات:

* تحسين أو إضافة دروس تعليمية
* تصحيح الأخطاء والعلل
* مشاركة مشاريع AMD الخاصة بك
* اقتراح أفكار وتوجيهات

&emsp;&emsp;يرجى قراءة **[Content Guide](../../CONTENT_GUIDE_en.md)** (الهيكل، التسمية، الصور — متوافقة مع دروس مثل Qwen3)، ثم **[CONTRIBUTING_en.md](../../CONTRIBUTING_en.md)** (المشكلات، طلبات السحب، واتفاقيات الدليل لكل نموذج).

&emsp;&emsp;إذا واجهت مشكلات متعلقة باستكشاف الأخطاء أو الأسئلة الشائعة أثناء استخدام ROCm أو نشر النماذج أو قراءة الدروس، فنرحب بك أيضًا في **[نقاش المجتمع](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)** لمشاركة الخبرات والإبلاغ عن المشكلات وتحسين الدروس مع المجتمع.

&emsp;&emsp;إذا كنت ترغب في المساعدة في صيانة المستودع على المدى الطويل، تواصل معنا — يمكننا إضافتك كمشرف.

## شكر وتقدير

### المساهمون الأساسيون

- [Zhixue Song (不要葱姜蒜) — قائد المشروع](https://github.com/KMnO4-zx) (عضو Datawhale؛ قائد مشروعي self-llm و happy-llm)
- [Yu Chen — قائد المشروع](https://github.com/lucachen) (منشئ محتوى؛ خبير Google في التعلم الآلي)
- [Sizhou Chen — مساهم](https://github.com/jjyaoao) (عضو Datawhale؛ قائد مشروع hello-agents)
- [Jiahang Pan — مساهم](https://github.com/amdjiahangpan) (منشئ محتوى؛ مهندس برمجيات في AMD)
- [Weihong Liu — مساهم](https://github.com/Weihong-Liu) (عضو Datawhale)
- [Dongbo Hao — مساهم](https://github.com/wlkq151172) (صانع محتوى؛ طالب دراسات عليا في جامعة جيانغنان)
- [Muling Ke — مساهم](https://github.com/1985312383) (عضو Datawhale)

> المزيد من المساهمين مرحب بهم دائمًا.

### آخرون

- الأفكار والملاحظات مرحب بها — يرجى فتح مشكلات.
- شكرًا لكل من ساهم في الدروس التعليمية.
- شكرًا لـ **برنامج AMD الجامعي** لدعم هذا المشروع.

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>

## الترخيص

[رخصة MIT](../../LICENSE)

---

<div align="center">

**دعونا نبني مستقبل الذكاء الاصطناعي من AMD معًا.** 💪

صنع بحب ❤️ من مجتمع hello-rocm

</div>
