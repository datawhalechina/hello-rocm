import { defineConfig, type DefaultTheme } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

const zhNav: DefaultTheme.NavItem[] = [
  { text: '首页', link: '/zh/' },
  {
    text: '入门主线 (0~6章)',
    items: [
      { text: '第0章｜导读与体验旅程', link: '/zh/00-guide/' },
      { text: '第1章｜直觉认知与三大引擎部署', link: '/zh/01-first-model/' },
      { text: '第2章｜WorkBuddy办公三件套与Qoder', link: '/zh/02-connect-tools/' },
      { text: '第3章｜提示词闯关与个人知识库', link: '/zh/03-prompt-and-rag/' },
      { text: '第4章｜可共享知识管理MCP与Skills', link: '/zh/04-mcp-and-skills/' },
      { text: '第5章｜第一次中文模型微调', link: '/zh/05-fine-tune/' },
      { text: '第6章｜AMD量化对比与自动加速', link: '/zh/06-optimize-speed/' }
    ]
  },
  { text: '简历与作品 (第7章)', link: '/zh/07-portfolio-resume/' },
  { text: '15大趣味案例 (第8章)', link: '/zh/08-fun-cases/' },
  { text: 'AMD生态百宝箱 (第9章)', link: '/zh/09-amd-ecosystem/' },
  {
    text: '专项档案库',
    items: [
      { text: '00-基础环境详细手册', link: '/zh/00-environment/' },
      { text: '01-多模型部署档案', link: '/zh/01-deploy/' },
      { text: '02-单模型微调实录', link: '/zh/02-fine-tune/' },
      { text: '03-HIP算子与GPU编程', link: '/zh/03-infra/' },
      { text: '04-ROCm参考资料总表', link: '/zh/04-references/' },
      { text: '05-AMD-YES高阶项目集', link: '/zh/05-amd-yes/' }
    ]
  }
]

const enNav: DefaultTheme.NavItem[] = [
  { text: 'Home', link: '/' },
  { text: 'Environment', link: '/00-environment/' },
  { text: 'Deploy', link: '/01-deploy/' },
  { text: 'Fine-tune', link: '/02-fine-tune/' },
  { text: 'Infra', link: '/03-infra/' },
  { text: 'References', link: '/04-references/' },
  { text: 'AMD Practice', link: '/05-amd-yes/' }
]

const zhCourseSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '第 0 章｜导读与体验旅程',
    collapsed: false,
    items: [
      { text: '把你的 AMD 电脑变成 AI 助手', link: '/zh/00-guide/' }
    ]
  },
  {
    text: '第 1 章｜直觉认知与三大引擎部署',
    collapsed: false,
    items: [
      { text: '1.1 数据类型与显存“三本明白账”', link: '/zh/01-first-model/' },
      { text: '1.2 极简环境准备 (Win11 / Ubuntu / 云)', link: '/zh/01-first-model/quick-env' },
      { text: '1.3~1.5 三大标准引擎极速部署', link: '/zh/01-first-model/three-engines-deploy' }
    ]
  },
  {
    text: '第 2 章｜办公先行：WorkBuddy 与 Qoder',
    collapsed: false,
    items: [
      { text: '2.1 万能插座与三大安全铁律', link: '/zh/02-connect-tools/' },
      { text: '2.2 🌟 首发：WorkBuddy 办公三件套', link: '/zh/02-connect-tools/workbuddy-office-suite' },
      { text: '2.3~2.4 接入 Qoder 与 Codex CLI', link: '/zh/02-connect-tools/qoder-and-codex' }
    ]
  },
  {
    text: '第 3 章｜提示词闯关与个人知识库',
    collapsed: false,
    items: [
      { text: '3.1~3.2 提示词 6 关与菜谱切分实验', link: '/zh/03-prompt-and-rag/' },
      { text: '3.3~3.4 WorkBuddy + Obsidian 知识库', link: '/zh/03-prompt-and-rag/workbuddy-obsidian-rag' }
    ]
  },
  {
    text: '第 4 章｜知识管理 MCP 与 Skills',
    collapsed: false,
    items: [
      { text: '可共享「知识管理 MCP」与自定义 Skill', link: '/zh/04-mcp-and-skills/' }
    ]
  },
  {
    text: '第 5 章｜第一次中文模型微调',
    collapsed: false,
    items: [
      { text: '5.1~5.3 微调决策树与弱智吧 5 分钟破冰', link: '/zh/05-fine-tune/' },
      { text: '5.4~5.6 中文通知抽取与 WeClone 分身', link: '/zh/05-fine-tune/chinese-lora-and-persona' }
    ]
  },
  {
    text: '第 6 章｜AMD 量化对比与自动加速',
    collapsed: false,
    items: [
      { text: '6.1~6.3 AMD Quark 量化前后对比与融合算子', link: '/zh/06-optimize-speed/' },
      { text: '6.4~6.5 Prefix Caching 与 Hyperloom 寻优', link: '/zh/06-optimize-speed/agent-cache-and-hyperloom' }
    ]
  },
  {
    text: '第 7 章｜作品集与简历成果锻造',
    collapsed: false,
    items: [
      { text: '成功故事、五件套与简历面试指南', link: '/zh/07-portfolio-resume/' }
    ]
  },
  {
    text: '第 8 章｜15 个趣味实战案例库',
    collapsed: false,
    items: [
      { text: '全景矩阵与新手试学机制', link: '/zh/08-fun-cases/' },
      { text: '上辑 (Case 01~08)：办公、论文与知识库', link: '/zh/08-fun-cases/office-and-study-cases' },
      { text: '下辑 (Case 09~15)：语音、Qoder编程与游戏', link: '/zh/08-fun-cases/voice-code-agent-game-cases' },
      { text: '附：五要素标准案例模板', link: '/zh/08-fun-cases/_CASE_TEMPLATE' }
    ]
  },
  {
    text: '第 9 章｜AMD 优秀项目与资源百宝箱',
    collapsed: false,
    items: [
      { text: '官方明星项目、行业案例与算力字典', link: '/zh/09-amd-ecosystem/' }
    ]
  }
]

const zhSidebar: DefaultTheme.Sidebar = {
  '/zh/00-guide/': zhCourseSidebar,
  '/zh/01-first-model/': zhCourseSidebar,
  '/zh/02-connect-tools/': zhCourseSidebar,
  '/zh/03-prompt-and-rag/': zhCourseSidebar,
  '/zh/04-mcp-and-skills/': zhCourseSidebar,
  '/zh/05-fine-tune/': zhCourseSidebar,
  '/zh/06-optimize-speed/': zhCourseSidebar,
  '/zh/07-portfolio-resume/': zhCourseSidebar,
  '/zh/08-fun-cases/': zhCourseSidebar,
  '/zh/09-amd-ecosystem/': zhCourseSidebar,

  '/zh/00-environment/': [
    {
      text: 'Environment',
      items: [
        { text: 'ROCm 基础环境安装与配置', link: '/zh/00-environment/' },
        { text: 'ROCm 10.0.0 版本说明', link: '/zh/00-environment/rocm-10-0-0-release-notes' },
        { text: 'GPU 架构与 pip 索引对照表', link: '/zh/00-environment/rocm-gpu-architecture-table' },
        { text: 'Ryzen AI XDNA2 硬件对照', link: '/zh/00-environment/xdna2-npu' },
        { text: 'FastFlowLM 环境配置', link: '/zh/00-environment/fastflowlm' },
        { text: 'Lemonade 环境配置', link: '/zh/00-environment/lemonade' },
        { text: 'NPU / Lemonade 排错', link: '/zh/00-environment/npu-troubleshooting' }
      ]
    }
  ],

  '/zh/01-deploy/': [
    {
      text: 'Deploy',
      items: [
        { text: 'ROCm 大模型部署实践', link: '/zh/01-deploy/' },
        {
          text: 'models',
          collapsed: false,
          items: [
            {
              text: 'Qwen3',
              collapsed: false,
              items: [
                { text: 'Ubuntu 24.04 + ROCm 10 环境准备', link: '/zh/01-deploy/qwen3/env-prepare-ubuntu24-rocm7' },
                { text: 'LM Studio 零基础部署', link: '/zh/01-deploy/qwen3/lm-studio-rocm7-deploy' },
                { text: 'vLLM 零基础部署', link: '/zh/01-deploy/qwen3/vllm-rocm7-deploy' },
                { text: 'Ollama 零基础部署', link: '/zh/01-deploy/qwen3/ollama-rocm7-deploy' },
                { text: 'llama.cpp 零基础部署', link: '/zh/01-deploy/qwen3/llamacpp-rocm7-deploy' }
              ]
            },
            {
              text: 'Qwen3.5',
              collapsed: false,
              items: [
                { text: 'Ubuntu 24.04 + ROCm 10 环境准备', link: '/zh/01-deploy/qwen3.5/env-prepare-ubuntu24-rocm7' },
                { text: 'LM Studio 零基础部署', link: '/zh/01-deploy/qwen3.5/lm-studio-rocm7-deploy' },
                { text: 'vLLM 部署调用', link: '/zh/01-deploy/qwen3.5/vllm-rocm7-deploy' },
                { text: 'Ollama 零基础部署', link: '/zh/01-deploy/qwen3.5/ollama-rocm7-deploy' },
                { text: 'llama.cpp 零基础部署', link: '/zh/01-deploy/qwen3.5/llamacpp-rocm7-deploy' }
              ]
            },
            {
              text: 'Qwen3.6',
              collapsed: false,
              items: [
                { text: 'Qwen3.6 模型介绍', link: '/zh/01-deploy/qwen36/qwen36_model' },
                { text: 'FastFlowLM NPU 部署', link: '/zh/01-deploy/qwen36/fastflowlm-npu-deploy' },
                { text: 'Lemonade GPU 部署', link: '/zh/01-deploy/qwen36/lemonade-gpu-deploy' },
                { text: 'Lemonade NPU 部署', link: '/zh/01-deploy/qwen36/lemonade-npu-deploy' }
              ]
            },
            {
              text: 'Gemma4',
              collapsed: false,
              items: [
                { text: 'Ubuntu 24.04 + ROCm 10 环境准备', link: '/zh/01-deploy/gemma4/env-prepare-ubuntu24-rocm7' },
                { text: 'Gemma 4 模型介绍', link: '/zh/01-deploy/gemma4/gemma4_model' },
                { text: 'LM Studio 零基础部署', link: '/zh/01-deploy/gemma4/lm-studio-rocm7-deploy' },
                { text: 'vLLM 零基础部署', link: '/zh/01-deploy/gemma4/vllm-rocm7-deploy' },
                { text: 'Ollama 零基础部署', link: '/zh/01-deploy/gemma4/ollama-rocm7-deploy' },
                { text: 'llama.cpp 零基础部署', link: '/zh/01-deploy/gemma4/llamacpp-rocm7-deploy' },
                { text: 'FastFlowLM NPU 部署', link: '/zh/01-deploy/gemma4/fastflowlm-npu-deploy' },
                { text: 'Lemonade GPU 部署', link: '/zh/01-deploy/gemma4/lemonade-gpu-deploy' },
                { text: 'Lemonade NPU 部署', link: '/zh/01-deploy/gemma4/lemonade-npu-deploy' },
                { text: 'FastFlowLM NPU 实测', link: '/zh/01-deploy/gemma4/fastflowlm-npu-results' }
              ]
            },
            {
              text: 'MiniCPM',
              collapsed: false,
              items: [
                { text: 'llama.cpp 部署 MiniCPM', link: '/zh/01-deploy/minicpm/llamacpp-rocm7-deploy' }
              ]
            },
            {
              text: 'MiniCPM-V',
              collapsed: false,
              items: [
                { text: 'llama.cpp 部署 MiniCPM-V', link: '/zh/01-deploy/minicpmv/llamacpp-rocm7-deploy' },
                { text: 'vLLM 部署 MiniCPM-V', link: '/zh/01-deploy/minicpmv/vllm-rocm7-deploy' }
              ]
            },
            {
              text: 'MiniCPM-o',
              collapsed: false,
              items: [
                { text: 'MiniCPM-o 4.5 部署概览', link: '/zh/01-deploy/minicpm-o/' },
                { text: 'MiniCPM-o 4.5 模型介绍', link: '/zh/01-deploy/minicpm-o/minicpm-o-model' },
                { text: 'llama.cpp-omni 零基础部署', link: '/zh/01-deploy/minicpm-o/llamacpp-omni-rocm7-deploy' },
                { text: 'Web Demo 全双工部署', link: '/zh/01-deploy/minicpm-o/webdemo-rocm7-deploy' }
              ]
            }
          ]
        }
      ]
    }
  ],

  '/zh/02-fine-tune/': [
    {
      text: 'Fine-tune',
      items: [
        { text: 'ROCm 大模型微调实践', link: '/zh/02-fine-tune/' },
        {
          text: 'models',
          collapsed: false,
          items: [
            {
              text: 'Qwen3',
              items: [
                { text: 'Qwen3-0.6B LoRA 及 SwanLab 可视化记录', link: '/zh/02-fine-tune/qwen3/qwen3-0.6b-lora-swanlab' }
              ]
            },
            {
              text: 'Qwen3.5',
              items: [
                { text: 'Qwen3.5-4B LoRA 及 SwanLab 可视化记录', link: '/zh/02-fine-tune/qwen3.5/qwen3.5-4b-lora-swanlab' }
              ]
            },
            {
              text: 'Gemma4',
              items: [
                { text: 'Gemma4-E4B LoRA（ModelScope 单卡）', link: '/zh/02-fine-tune/gemma4/gemma4-e4b-lora-modelscope-single-gpu' }
              ]
            }
          ]
        }
      ]
    }
  ],

  '/zh/03-infra/': [
    {
      text: 'Infra',
      items: [
        { text: 'ROCm 算子优化实践', link: '/zh/03-infra/' },
        {
          text: 'embrace-amd-ai',
          items: [
            { text: '拥抱 AMD AI 算力新时代', link: '/zh/03-infra/embrace-amd-ai' }
          ]
        },
        {
          text: 'decode-ai-accelerator',
          items: [
            { text: '解密 AI 加速器：从软件栈到硬件架构', link: '/zh/03-infra/decode-ai-accelerator' }
          ]
        },
        {
          text: 'handwrite-rocm-operator',
          items: [
            { text: '迈入 ROCm 编程世界：手写 ROCm 算子', link: '/zh/03-infra/handwrite-rocm-operator' }
          ]
        },
        {
          text: 'custom-pytorch-operator',
          items: [
            { text: '为 PyTorch 编写自定义 ROCm 算子', link: '/zh/03-infra/custom-pytorch-operator' }
          ]
        },
        { text: 'ROCm 图谱', link: '/atlas/rocm/index.html', target: '_blank' }
      ]
    }
  ],

  '/zh/05-amd-yes/': [
    {
      text: 'AMD-YES',
      items: [
        { text: 'AMD 实践案例集合', link: '/zh/05-amd-yes/' },
        {
          text: 'toy-cli',
          items: [
            { text: 'LLM 轻量化终端助手', link: '/zh/05-amd-yes/toy-cli' }
          ]
        },
        {
          text: 'minesweeper-agent',
          items: [
            { text: 'Minesweeper Agent：用本地大模型玩扫雷', link: '/zh/05-amd-yes/minesweeper-agent' }
          ]
        },
        {
          text: 'wechat-jump',
          items: [
            { text: 'YOLOv10 微信跳一跳', link: '/zh/05-amd-yes/wechat-jump' }
          ]
        },
        {
          text: 'huanhuan-chat',
          items: [
            { text: 'Chat-甄嬛：后宫语言模型', link: '/zh/05-amd-yes/huanhuan-chat' }
          ]
        },
        {
          text: 'happy-llm',
          collapsed: false,
          items: [
            { text: 'Happy-LLM：从零训练大模型', link: '/zh/05-amd-yes/happy-llm/' },
            {
              text: 'chapter5',
              collapsed: false,
              items: [
                { text: '执行流程与脚本说明', link: '/zh/05-amd-yes/happy-llm/chapter5/' },
                { text: '第五章 动手搭建大模型', link: '/zh/05-amd-yes/happy-llm/chapter5/chapter5-hands-on-llm-building' }
              ]
            },
            {
              text: 'chapter6',
              collapsed: false,
              items: [
                { text: '执行流程与脚本说明', link: '/zh/05-amd-yes/happy-llm/chapter6/' },
                { text: '第六章 大模型训练流程实践', link: '/zh/05-amd-yes/happy-llm/chapter6/chapter6-llm-training-workflow-practice' },
                { text: '6.4 偏好对齐', link: '/zh/05-amd-yes/happy-llm/chapter6/chapter6-4-wip-preference-alignment' }
              ]
            }
          ]
        },
        {
          text: 'hello-agents',
          collapsed: false,
          items: [
            { text: 'Hello Agents 智能体实践', link: '/zh/05-amd-yes/hello-agents/' },
            {
              text: 'smart-travel-planner',
              items: [
                { text: '智能旅行规划助手', link: '/zh/05-amd-yes/hello-agents/smart-travel-planner/amd395-helloagents-smart-travel-planner' }
              ]
            }
          ]
        },
        {
          text: 'torch-rechub',
          collapsed: false,
          items: [
            { text: 'Torch-RecHub 推荐系统实战', link: '/zh/05-amd-yes/torch-rechub/' },
            { text: 'CTR 预测：DeepFM', link: '/zh/05-amd-yes/torch-rechub/00_QuickStart_CTR_DeepFM' },
            { text: '序列兴趣建模：DIN', link: '/zh/05-amd-yes/torch-rechub/01_Ranking_DIN' },
            { text: '匹配/召回：DSSM', link: '/zh/05-amd-yes/torch-rechub/02_Matching_DSSM' },
            { text: '多任务学习：MMOE', link: '/zh/05-amd-yes/torch-rechub/03_MultiTask_MMOE' },
            { text: '实验跟踪：model_logger', link: '/zh/05-amd-yes/torch-rechub/04_Experiment_Tracking_Light' },
            { text: '模型导出与推理验证：ONNX', link: '/zh/05-amd-yes/torch-rechub/05_Model_Export_and_Serving' }
          ]
        },
        {
          text: 'openclaw',
          items: [
            { text: 'OpenClaw 全隐私本地 AI 智能体平台', link: '/zh/05-amd-yes/openclaw' }
          ]
        },
        {
          text: 'every-embodied',
          items: [
            { text: 'Every Embodied 具身智能策略复刻', link: '/zh/05-amd-yes/every-embodied' }
          ]
        },
        {
          text: 'ds4',
          items: [
            { text: 'DeepSeek-V4-Flash 原生 HIP 部署', link: '/zh/05-amd-yes/ds4' }
          ]
        },
        {
          text: 'qwen3.8',
          items: [
            { text: 'Qwen3.8-Flash-CIRU 部署', link: '/zh/05-amd-yes/qwen3.8' }
          ]
        }
      ]
    }
  ],

  '/zh/04-references': [
    {
      text: 'References',
      items: [
        { text: 'ROCm 优质参考资料', link: '/zh/04-references/' }
      ]
    }
  ],

  '/zh/learning/': [
    {
      text: '组队学习',
      items: [
        { text: 'hello-rocm 组队学习', link: '/zh/learning/' },
        { text: '20 天纯云端组队学习计划', link: '/zh/learning/20-day-learning-plan' }
      ]
    }
  ],

  '/zh/cloud/': [
    {
      text: '云平台',
      items: [
        { text: 'AMD Radeon Cloud 使用指南', link: '/zh/cloud/amd-radeon-cloud' },
        { text: 'AUP Learning Cloud 使用指南', link: '/zh/cloud/aup-learning-cloud' }
      ]
    }
  ]
}

const enSidebar: DefaultTheme.Sidebar = {
  '/00-environment/': [
    {
      text: 'Environment',
      items: [
        { text: 'ROCm Environment Setup', link: '/00-environment/' },
        { text: 'ROCm 10.0.0 Release Notes', link: '/00-environment/rocm-10-0-0-release-notes' },
        { text: 'GPU Architecture and pip Index Table', link: '/00-environment/rocm-gpu-architecture-table' },
        { text: 'Ryzen AI XDNA2 Hardware', link: '/00-environment/xdna2-npu' },
        { text: 'FastFlowLM Environment', link: '/00-environment/fastflowlm' },
        { text: 'Lemonade Environment', link: '/00-environment/lemonade' },
        { text: 'NPU / Lemonade Troubleshooting', link: '/00-environment/npu-troubleshooting' }
      ]
    }
  ],

  '/01-deploy/': [
    {
      text: 'Deploy',
      items: [
        { text: 'LLM Deployment on ROCm', link: '/01-deploy/' },
        {
          text: 'models',
          collapsed: false,
          items: [
            {
              text: 'Qwen3',
              collapsed: false,
              items: [
                { text: 'Ubuntu 24.04 + ROCm 10 Environment Preparation', link: '/01-deploy/qwen3/env-prepare-ubuntu24-rocm7' },
                { text: 'LM Studio Deployment', link: '/01-deploy/qwen3/lm-studio-rocm7-deploy' },
                { text: 'vLLM Deployment', link: '/01-deploy/qwen3/vllm-rocm7-deploy' },
                { text: 'Ollama Deployment', link: '/01-deploy/qwen3/ollama-rocm7-deploy' },
                { text: 'llama.cpp Deployment', link: '/01-deploy/qwen3/llamacpp-rocm7-deploy' }
              ]
            },
            {
              text: 'Qwen3.5',
              collapsed: false,
              items: [
                { text: 'Ubuntu 24.04 + ROCm 10 Environment Preparation', link: '/01-deploy/qwen3.5/env-prepare-ubuntu24-rocm7' },
                { text: 'LM Studio Deployment', link: '/01-deploy/qwen3.5/lm-studio-rocm7-deploy' },
                { text: 'vLLM Deployment and API Calls', link: '/01-deploy/qwen3.5/vllm-rocm7-deploy' },
                { text: 'Ollama Deployment', link: '/01-deploy/qwen3.5/ollama-rocm7-deploy' },
                { text: 'llama.cpp Deployment', link: '/01-deploy/qwen3.5/llamacpp-rocm7-deploy' }
              ]
            },
            {
              text: 'Qwen3.6',
              collapsed: false,
              items: [
                { text: 'Qwen3.6 Model Introduction', link: '/01-deploy/qwen36/qwen36_model' },
                { text: 'FastFlowLM NPU Deployment', link: '/01-deploy/qwen36/fastflowlm-npu-deploy' },
                { text: 'Lemonade GPU Deployment', link: '/01-deploy/qwen36/lemonade-gpu-deploy' },
                { text: 'Lemonade NPU Deployment', link: '/01-deploy/qwen36/lemonade-npu-deploy' }
              ]
            },
            {
              text: 'Gemma4',
              collapsed: false,
              items: [
                { text: 'Ubuntu 24.04 + ROCm 10 Environment Preparation', link: '/01-deploy/gemma4/env-prepare-ubuntu24-rocm7' },
                { text: 'Gemma 4 Model Introduction', link: '/01-deploy/gemma4/gemma4_model' },
                { text: 'LM Studio Deployment', link: '/01-deploy/gemma4/lm-studio-rocm7-deploy' },
                { text: 'vLLM Deployment', link: '/01-deploy/gemma4/vllm-rocm7-deploy' },
                { text: 'Ollama Deployment', link: '/01-deploy/gemma4/ollama-rocm7-deploy' },
                { text: 'llama.cpp Deployment', link: '/01-deploy/gemma4/llamacpp-rocm7-deploy' },
                { text: 'FastFlowLM NPU Deployment', link: '/01-deploy/gemma4/fastflowlm-npu-deploy' },
                { text: 'Lemonade GPU Deployment', link: '/01-deploy/gemma4/lemonade-gpu-deploy' },
                { text: 'Lemonade NPU Deployment', link: '/01-deploy/gemma4/lemonade-npu-deploy' },
                { text: 'FastFlowLM NPU Results', link: '/01-deploy/gemma4/fastflowlm-npu-results' }
              ]
            },
            {
              text: 'MiniCPM',
              collapsed: false,
              items: [
                { text: 'llama.cpp Deployment', link: '/01-deploy/minicpm/llamacpp-rocm7-deploy' }
              ]
            },
            {
              text: 'MiniCPM-V',
              collapsed: false,
              items: [
                { text: 'llama.cpp Deployment', link: '/01-deploy/minicpmv/llamacpp-rocm7-deploy' },
                { text: 'vLLM Deployment', link: '/01-deploy/minicpmv/vllm-rocm7-deploy' }
              ]
            },
            {
              text: 'MiniCPM-o',
              collapsed: false,
              items: [
                { text: 'MiniCPM-o 4.5 Deployment Overview', link: '/01-deploy/minicpm-o/' },
                { text: 'MiniCPM-o 4.5 Model Introduction', link: '/01-deploy/minicpm-o/minicpm-o-model' },
                { text: 'llama.cpp-omni Deployment', link: '/01-deploy/minicpm-o/llamacpp-omni-rocm7-deploy' },
                { text: 'Web Demo Full-Duplex Deployment', link: '/01-deploy/minicpm-o/webdemo-rocm7-deploy' }
              ]
            }
          ]
        }
      ]
    }
  ],

  '/02-fine-tune/': [
    {
      text: 'Fine-tune',
      items: [
        { text: 'LLM Fine-tuning on ROCm', link: '/02-fine-tune/' },
        {
          text: 'models',
          collapsed: false,
          items: [
            {
              text: 'Qwen3',
              items: [
                { text: 'Qwen3-0.6B LoRA and SwanLab Records', link: '/02-fine-tune/qwen3/qwen3-0.6b-lora-swanlab' }
              ]
            },
            {
              text: 'Qwen3.5',
              items: [
                { text: 'Qwen3.5-4B LoRA and SwanLab Records', link: '/02-fine-tune/qwen3.5/qwen3.5-4b-lora-swanlab' }
              ]
            },
            {
              text: 'Gemma4',
              items: [
                { text: 'Gemma4-E4B LoRA (ModelScope, single GPU)', link: '/02-fine-tune/gemma4/gemma4-e4b-lora-modelscope-single-gpu' }
              ]
            }
          ]
        }
      ]
    }
  ],

  '/03-infra/': [
    {
      text: 'Infra',
      items: [
        { text: 'ROCm Operator Optimization Practice', link: '/03-infra/' },
        {
          text: 'embrace-amd-ai',
          items: [
            { text: 'Embrace the AMD AI Computing Era', link: '/03-infra/embrace-amd-ai' }
          ]
        },
        {
          text: 'decode-ai-accelerator',
          items: [
            { text: 'Decode AI Accelerators', link: '/03-infra/decode-ai-accelerator' }
          ]
        },
        {
          text: 'handwrite-rocm-operator',
          items: [
            { text: 'Handwrite ROCm Operators', link: '/03-infra/handwrite-rocm-operator' }
          ]
        },
        {
          text: 'custom-pytorch-operator',
          items: [
            { text: 'Custom ROCm Operators for PyTorch', link: '/03-infra/custom-pytorch-operator' }
          ]
        },
        { text: 'ROCm Atlas', link: '/atlas/rocm/index.html', target: '_blank' }
      ]
    }
  ],

  '/05-amd-yes/': [
    {
      text: 'AMD-YES',
      items: [
        { text: 'AMD Practice Showcases', link: '/05-amd-yes/' },
        {
          text: 'toy-cli',
          items: [
            { text: 'Lightweight Terminal Assistant', link: '/05-amd-yes/toy-cli' }
          ]
        },
        {
          text: 'minesweeper-agent',
          items: [
            { text: 'Minesweeper Agent: Play Minesweeper with Local LLM', link: '/05-amd-yes/minesweeper-agent' }
          ]
        },
        {
          text: 'wechat-jump',
          items: [
            { text: 'YOLOv10 WeChat Jump', link: '/05-amd-yes/wechat-jump' }
          ]
        },
        {
          text: 'huanhuan-chat',
          items: [
            { text: 'Chat-Huanhuan', link: '/05-amd-yes/huanhuan-chat' }
          ]
        },
        {
          text: 'happy-llm',
          collapsed: false,
          items: [
            { text: 'Happy-LLM ROCm Training', link: '/05-amd-yes/happy-llm/' },
            {
              text: 'chapter5',
              collapsed: false,
              items: [
                { text: 'Execution Process and Scripts', link: '/05-amd-yes/happy-llm/chapter5/' },
                { text: 'Chapter 5 Hands-on LLM Building', link: '/05-amd-yes/happy-llm/chapter5/chapter5-hands-on-llm-building' }
              ]
            },
            {
              text: 'chapter6',
              collapsed: false,
              items: [
                { text: 'Execution Process and Scripts', link: '/05-amd-yes/happy-llm/chapter6/' },
                { text: 'Chapter 6 LLM Training Workflow Practice', link: '/05-amd-yes/happy-llm/chapter6/chapter6-llm-training-workflow-practice' },
                { text: 'Chapter 6.4 Preference Alignment', link: '/05-amd-yes/happy-llm/chapter6/chapter6-4-wip-preference-alignment' }
              ]
            }
          ]
        },
        {
          text: 'hello-agents',
          collapsed: false,
          items: [
            { text: 'Hello Agents Practice', link: '/05-amd-yes/hello-agents/' },
            {
              text: 'smart-travel-planner',
              items: [
                { text: 'Smart Travel Planner', link: '/05-amd-yes/hello-agents/smart-travel-planner/amd395-helloagents-smart-travel-planner' }
              ]
            }
          ]
        },
        {
          text: 'torch-rechub',
          collapsed: false,
          items: [
            { text: 'Torch-RecHub Recommender Practice', link: '/05-amd-yes/torch-rechub/' },
            { text: 'CTR Prediction: DeepFM', link: '/05-amd-yes/torch-rechub/00_QuickStart_CTR_DeepFM' },
            { text: 'Sequential Interest Modeling: DIN', link: '/05-amd-yes/torch-rechub/01_Ranking_DIN' },
            { text: 'Matching / Retrieval: DSSM', link: '/05-amd-yes/torch-rechub/02_Matching_DSSM' },
            { text: 'Multi-task Learning: MMOE', link: '/05-amd-yes/torch-rechub/03_MultiTask_MMOE' },
            { text: 'Experiment Tracking: model_logger', link: '/05-amd-yes/torch-rechub/04_Experiment_Tracking_Light' },
            { text: 'Model Export and Inference Validation: ONNX', link: '/05-amd-yes/torch-rechub/05_Model_Export_and_Serving' }
          ]
        },
        {
          text: 'openclaw',
          items: [
            { text: 'OpenClaw Private Local AI Agent Platform', link: '/05-amd-yes/openclaw' }
          ]
        },
        {
          text: 'every-embodied',
          items: [
            { text: 'Every Embodied Policy Replication', link: '/05-amd-yes/every-embodied' }
          ]
        },
        {
          text: 'ds4',
          items: [
            { text: 'DeepSeek-V4-Flash Native HIP', link: '/05-amd-yes/ds4' }
          ]
        },
        {
          text: 'qwen3.8',
          items: [
            { text: 'Qwen3.8-Flash-CIRU', link: '/05-amd-yes/qwen3.8' }
          ]
        }
      ]
    }
  ],

  '/04-references': [
    {
      text: 'References',
      items: [
        { text: 'ROCm References', link: '/04-references/' }
      ]
    }
  ]
}

const zhThemeConfig: DefaultTheme.Config = {
  nav: zhNav,
  sidebar: zhSidebar,
  socialLinks: [
    { icon: 'github', link: 'https://github.com/datawhalechina/hello-rocm' }
  ],
  search: {
    provider: 'local'
  },
  outline: {
    label: '本页目录',
    level: [2, 3]
  },
  docFooter: {
    prev: '上一页',
    next: '下一页'
  },
  lastUpdated: {
    text: '最后更新'
  }
}

const enThemeConfig: DefaultTheme.Config = {
  nav: enNav,
  sidebar: enSidebar,
  socialLinks: [
    { icon: 'github', link: 'https://github.com/datawhalechina/hello-rocm' }
  ],
  search: {
    provider: 'local'
  },
  outline: {
    label: 'On this page',
    level: [2, 3]
  },
  docFooter: {
    prev: 'Previous',
    next: 'Next'
  },
  lastUpdated: {
    text: 'Last updated'
  }
}

export default defineConfig({
  title: 'hello-rocm',
  description: 'AMD ROCm tutorials and examples',
  base: '/hello-rocm/',
  cleanUrls: true,
  lastUpdated: true,
  rewrites: {
    'en/:rest*': ':rest*'
  },
  markdown: {
    lineNumbers: true,
    theme: {
      light: 'light-plus',
      dark: 'dark-plus'
    },
    config: (md) => {
      md.use(mathjax3)

      const defaultFence = md.renderer.rules.fence
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const language = token.info.trim().split(/\s+/)[0]

        if (language === 'mermaid') {
          return `<Mermaid code="${encodeURIComponent(token.content)}" encoded />`
        }

        return defaultFence
          ? defaultFence(tokens, idx, options, env, self)
          : self.renderToken(tokens, idx, options)
      }
    }
  },
  locales: {
    root: {
      label: 'English',
      lang: 'en-US',
      themeConfig: enThemeConfig
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      themeConfig: zhThemeConfig
    }
  }
})
