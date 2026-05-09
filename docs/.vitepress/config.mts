import { defineConfig, type DefaultTheme } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

const nav: DefaultTheme.NavItem[] = [
  { text: '首页', link: '/' },
  { text: '基础环境', link: '/environment/' },
  { text: '大模型部署', link: '/deploy/' },
  { text: '大模型微调', link: '/fine-tune/' },
  { text: '算子优化', link: '/infra/' },
  { text: '参考资料', link: '/references' },
  { text: '实践案例', link: '/amd-yes/' }
]

const sidebar: DefaultTheme.Sidebar = {
  '/environment/': [
    {
      text: '00-Environment',
      items: [
        { text: 'ROCm 基础环境安装与配置', link: '/environment/' },
        { text: 'ROCm Environment Setup', link: '/environment/index_en' },
        { text: 'GPU 架构与 pip 索引对照表', link: '/environment/rocm-gpu-architecture-table' }
      ]
    }
  ],

  '/deploy/': [
    {
      text: '01-Deploy',
      items: [
        { text: 'ROCm 大模型部署实践', link: '/deploy/' },
        { text: 'LLM Deployment on ROCm', link: '/deploy/index_en' },
        {
          text: 'models',
          collapsed: false,
          items: [
            {
              text: 'Qwen3',
              collapsed: false,
              items: [
                { text: 'Ubuntu 24.04 + ROCm 7 环境准备', link: '/deploy/qwen3/env-prepare-ubuntu24-rocm7' },
                { text: 'LM Studio 零基础部署', link: '/deploy/qwen3/lm-studio-rocm7-deploy' },
                { text: 'vLLM 零基础部署', link: '/deploy/qwen3/vllm-rocm7-deploy' },
                { text: 'Ollama 零基础部署', link: '/deploy/qwen3/ollama-rocm7-deploy' },
                { text: 'llama.cpp 零基础部署', link: '/deploy/qwen3/llamacpp-rocm7-deploy' }
              ]
            },
            {
              text: 'Gemma4',
              collapsed: false,
              items: [
                { text: 'Ubuntu 24.04 + ROCm 7 环境准备', link: '/deploy/gemma4/env-prepare-ubuntu24-rocm7' },
                { text: 'Gemma 4 模型介绍', link: '/deploy/gemma4/gemma4_model' },
                { text: 'LM Studio 零基础部署', link: '/deploy/gemma4/lm-studio-rocm7-deploy' },
                { text: 'vLLM 零基础部署', link: '/deploy/gemma4/vllm-rocm7-deploy' },
                { text: 'Ollama 零基础部署', link: '/deploy/gemma4/ollama-rocm7-deploy' },
                { text: 'llama.cpp 零基础部署', link: '/deploy/gemma4/llamacpp-rocm7-deploy' }
              ]
            }
          ]
        }
      ]
    }
  ],

  '/fine-tune/': [
    {
      text: '02-Fine-tune',
      items: [
        { text: 'ROCm 大模型微调实践', link: '/fine-tune/' },
        { text: 'LLM Fine-tuning on ROCm', link: '/fine-tune/index_en' },
        {
          text: 'models',
          collapsed: false,
          items: [
            {
              text: 'Qwen3',
              items: [
                { text: 'Qwen3-0.6B LoRA 及 SwanLab 可视化记录', link: '/fine-tune/qwen3/qwen3-0.6b-lora-swanlab' }
              ]
            },
            {
              text: 'Gemma4',
              items: [
                { text: 'Gemma4-E4B LoRA 及 SwanLab 可视化记录', link: '/fine-tune/gemma4/gemma4-e4b-lora-swanlab' }
              ]
            }
          ]
        }
      ]
    }
  ],

  '/infra/': [
    {
      text: '03-Infra',
      items: [
        { text: 'ROCm 算子优化实践', link: '/infra/' },
        {
          text: '01-embrace-amd-ai',
          items: [
            { text: '拥抱 AMD AI 算力新时代', link: '/infra/embrace-amd-ai' }
          ]
        },
        {
          text: '02-decode-ai-accelerator',
          items: [
            { text: '解密 AI 加速器：从软件栈到硬件架构', link: '/infra/decode-ai-accelerator' }
          ]
        },
        {
          text: '03-handwrite-rocm-operator',
          items: [
            { text: '迈入 ROCm 编程世界：手写 ROCm 算子', link: '/infra/handwrite-rocm-operator' }
          ]
        },
        {
          text: '04-custom-pytorch-operator',
          items: [
            { text: '为 PyTorch 编写自定义 ROCm 算子', link: '/infra/custom-pytorch-operator' }
          ]
        }
      ]
    }
  ],

  '/amd-yes/': [
    {
      text: '05-AMD-YES',
      items: [
        { text: 'AMD 实践案例集合', link: '/amd-yes/' },
        { text: 'AMD Practice Showcases', link: '/amd-yes/index_en' },
        {
          text: '01-toy-cli',
          items: [
            { text: 'LLM 轻量化终端助手', link: '/amd-yes/toy-cli' },
            { text: 'Lightweight Terminal Assistant', link: '/amd-yes/toy-cli_en' }
          ]
        },
        {
          text: '02-wechat-jump',
          items: [
            { text: 'YOLOv10 微信跳一跳', link: '/amd-yes/wechat-jump' },
            { text: 'YOLOv10 WeChat Jump', link: '/amd-yes/wechat-jump_en' }
          ]
        },
        {
          text: '03-huanhuan-chat',
          items: [
            { text: 'Chat-甄嬛：后宫语言模型', link: '/amd-yes/huanhuan-chat' },
            { text: 'Chat-Huanhuan', link: '/amd-yes/huanhuan-chat_en' }
          ]
        },
        {
          text: '04-happy-llm',
          collapsed: false,
          items: [
            { text: 'Happy-LLM：从零训练大模型', link: '/amd-yes/happy-llm/' },
            { text: 'Happy-LLM ROCm Training', link: '/amd-yes/happy-llm/index_en' },
            {
              text: 'chapter5',
              collapsed: false,
              items: [
                { text: '执行流程与脚本说明', link: '/amd-yes/happy-llm/chapter5/' },
                { text: 'Execution Process and Scripts', link: '/amd-yes/happy-llm/chapter5/index_en' },
                { text: 'Chapter 5 Hands-on LLM Building', link: '/amd-yes/happy-llm/chapter5/chapter5-hands-on-llm-building' },
                { text: '第五章 动手搭建大模型', link: '/amd-yes/happy-llm/chapter5/第五章 动手搭建大模型' }
              ]
            },
            {
              text: 'chapter6',
              collapsed: false,
              items: [
                { text: '执行流程与脚本说明', link: '/amd-yes/happy-llm/chapter6/' },
                { text: 'Execution Process and Scripts', link: '/amd-yes/happy-llm/chapter6/index_en' },
                { text: 'Chapter 6 LLM Training Workflow Practice', link: '/amd-yes/happy-llm/chapter6/chapter6-llm-training-workflow-practice' },
                { text: '第六章 大模型训练流程实践', link: '/amd-yes/happy-llm/chapter6/第六章 大模型训练流程实践' },
                { text: 'Chapter 6.4 Preference Alignment', link: '/amd-yes/happy-llm/chapter6/chapter6-4-wip-preference-alignment' },
                { text: '6.4 偏好对齐', link: '/amd-yes/happy-llm/chapter6/6.4-wip-preference-alignment-zh' }
              ]
            }
          ]
        },
        {
          text: '05-hello-agents',
          collapsed: false,
          items: [
            { text: 'Hello Agents 智能体实践', link: '/amd-yes/hello-agents/' },
            { text: 'Hello Agents Practice', link: '/amd-yes/hello-agents/index_en' },
            {
              text: 'smart-travel-planner',
              items: [
                { text: '智能旅行规划助手', link: '/amd-yes/hello-agents/smart-travel-planner/amd395-helloagents-smart-travel-planner-zh' },
                { text: 'Smart Travel Planner', link: '/amd-yes/hello-agents/smart-travel-planner/amd395-helloagents-smart-travel-planner-en' }
              ]
            }
          ]
        },
        {
          text: '06-openclaw',
          items: [
            { text: 'OpenClaw 全隐私本地 AI 智能体平台', link: '/amd-yes/openclaw' },
            { text: 'OpenClaw Private Local AI Agent Platform', link: '/amd-yes/openclaw_en' }
          ]
        }
      ]
    }
  ],

  '/references': [
    {
      text: '04-References',
      items: [
        { text: 'ROCm 优质参考资料', link: '/references' }
      ]
    }
  ]
}

export default defineConfig({
  lang: 'zh-CN',
  title: 'hello-rocm',
  description: 'AMD ROCm tutorials and examples',
  base: '/hello-rocm/',
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    lineNumbers: true,
    config: (md) => {
      md.use(mathjax3)
    }
  },
  themeConfig: {
    nav,
    sidebar,
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
})
