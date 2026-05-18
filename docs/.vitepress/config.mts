import { defineConfig, type DefaultTheme } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

const zhNav: DefaultTheme.NavItem[] = [
  { text: '首页', link: '/zh/' },
  { text: '基础环境', link: '/zh/00-environment/' },
  { text: '大模型部署', link: '/zh/01-deploy/' },
  { text: '大模型微调', link: '/zh/02-fine-tune/' },
  { text: '算子优化', link: '/zh/03-infra/' },
  { text: '参考资料', link: '/zh/04-references/' },
  { text: '实践案例', link: '/zh/05-amd-yes/' }
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

const zhSidebar: DefaultTheme.Sidebar = {
  '/zh/00-environment/': [
    {
      text: 'Environment',
      items: [
        { text: 'ROCm 基础环境安装与配置', link: '/zh/00-environment/' },
        { text: 'GPU 架构与 pip 索引对照表', link: '/zh/00-environment/rocm-gpu-architecture-table' }
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
                { text: 'Ubuntu 24.04 + ROCm 7 环境准备', link: '/zh/01-deploy/qwen3/env-prepare-ubuntu24-rocm7' },
                { text: 'LM Studio 零基础部署', link: '/zh/01-deploy/qwen3/lm-studio-rocm7-deploy' },
                { text: 'vLLM 零基础部署', link: '/zh/01-deploy/qwen3/vllm-rocm7-deploy' },
                { text: 'Ollama 零基础部署', link: '/zh/01-deploy/qwen3/ollama-rocm7-deploy' },
                { text: 'llama.cpp 零基础部署', link: '/zh/01-deploy/qwen3/llamacpp-rocm7-deploy' }
              ]
            },
            {
              text: 'Gemma4',
              collapsed: false,
              items: [
                { text: 'Ubuntu 24.04 + ROCm 7 环境准备', link: '/zh/01-deploy/gemma4/env-prepare-ubuntu24-rocm7' },
                { text: 'Gemma 4 模型介绍', link: '/zh/01-deploy/gemma4/gemma4_model' },
                { text: 'LM Studio 零基础部署', link: '/zh/01-deploy/gemma4/lm-studio-rocm7-deploy' },
                { text: 'vLLM 零基础部署', link: '/zh/01-deploy/gemma4/vllm-rocm7-deploy' },
                { text: 'Ollama 零基础部署', link: '/zh/01-deploy/gemma4/ollama-rocm7-deploy' },
                { text: 'llama.cpp 零基础部署', link: '/zh/01-deploy/gemma4/llamacpp-rocm7-deploy' }
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
        }
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
  ]
}

const enSidebar: DefaultTheme.Sidebar = {
  '/00-environment/': [
    {
      text: 'Environment',
      items: [
        { text: 'ROCm Environment Setup', link: '/00-environment/' },
        { text: 'GPU Architecture and pip Index Table', link: '/00-environment/rocm-gpu-architecture-table' }
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
                { text: 'Ubuntu 24.04 + ROCm 7 Environment Preparation', link: '/01-deploy/qwen3/env-prepare-ubuntu24-rocm7' },
                { text: 'LM Studio Deployment', link: '/01-deploy/qwen3/lm-studio-rocm7-deploy' },
                { text: 'vLLM Deployment', link: '/01-deploy/qwen3/vllm-rocm7-deploy' },
                { text: 'Ollama Deployment', link: '/01-deploy/qwen3/ollama-rocm7-deploy' },
                { text: 'llama.cpp Deployment', link: '/01-deploy/qwen3/llamacpp-rocm7-deploy' }
              ]
            },
            {
              text: 'Gemma4',
              collapsed: false,
              items: [
                { text: 'Ubuntu 24.04 + ROCm 7 Environment Preparation', link: '/01-deploy/gemma4/env-prepare-ubuntu24-rocm7' },
                { text: 'Gemma 4 Model Introduction', link: '/01-deploy/gemma4/gemma4_model' },
                { text: 'LM Studio Deployment', link: '/01-deploy/gemma4/lm-studio-rocm7-deploy' },
                { text: 'vLLM Deployment', link: '/01-deploy/gemma4/vllm-rocm7-deploy' },
                { text: 'Ollama Deployment', link: '/01-deploy/gemma4/ollama-rocm7-deploy' },
                { text: 'llama.cpp Deployment', link: '/01-deploy/gemma4/llamacpp-rocm7-deploy' }
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
        }
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
