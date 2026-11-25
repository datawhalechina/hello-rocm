import { defineConfig, type DefaultTheme } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

const zhNav: DefaultTheme.NavItem[] = [
  { text: '首页', link: '/zh/' },
  {
    text: '入门主线 (第0~6章)',
    items: [
      { text: '第0章｜导读与体验旅程', link: '/zh/00-guide/' },
      { text: '第1章｜ROCm 10 环境与三大引擎部署', link: '/zh/01-first-model/' },
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
    text: '云算力与组队营',
    items: [
      { text: '☁️ AMD Radeon Cloud 免费算力指南', link: '/zh/cloud/amd-radeon-cloud' },
      { text: '🎓 AUP Learning Cloud 高校算力指南', link: '/zh/cloud/aup-learning-cloud' },
      { text: '🤝 组队学习与 20 天打卡计划', link: '/zh/learning/' },
      { text: '🗺️ ROCm 软件栈交互式图谱', link: '/atlas/rocm/index.html', target: '_blank' }
    ]
  }
]

const enNav: DefaultTheme.NavItem[] = [
  { text: 'Overview', link: '/' },
  { text: '中文完整教材 (Ch 0–9)', link: '/zh/' },
  { text: 'ROCm Atlas', link: '/atlas/rocm/index.html', target: '_blank' }
]

const zhCourseSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '第 0 章｜导读与体验旅程',
    collapsed: false,
    items: [
      { text: '0.1~0.2 为什么装本地 AI 与六站旅程', link: '/zh/00-guide/' },
      { text: '0.3 3分钟硬件体检、模型选型与云算力', link: '/zh/00-guide/03-hardware-and-cloud' },
      { text: '0.4 课程全景地图、导流表与伴学 Skill', link: '/zh/00-guide/04-roadmap-and-skill' }
    ]
  },
  {
    text: '第 1 章｜环境配置与三大引擎部署',
    collapsed: false,
    items: [
      { text: '导览：四大主体模型全景档案', link: '/zh/01-first-model/' },
      { text: '1.1 参数量、量化精度与“显存三本账”', link: '/zh/01-first-model/01-model-intuition' },
      { text: '1.2 ROCm 10.0.0 与基础环境完整配置', link: '/zh/01-first-model/02-rocm-env-setup' },
      { text: '1.3 引擎一：llama-server (:8080)', link: '/zh/01-first-model/03-llama-server' },
      { text: '1.4 引擎二：lemonade-server (:13305)', link: '/zh/01-first-model/04-lemonade-server' },
      { text: '1.5 引擎三：vLLM (:8000) 与横评导流', link: '/zh/01-first-model/05-vllm-server' }
    ]
  },
  {
    text: '第 2 章｜办公先行：WorkBuddy 与 Qoder',
    collapsed: false,
    items: [
      { text: '2.1 万能插座原理与三条安全铁律', link: '/zh/02-connect-tools/' },
      { text: '2.2.1 接入 WorkBuddy 与五大避坑指南', link: '/zh/02-connect-tools/02-workbuddy-setup' },
      { text: '2.2.2 🌟 实战一：dashi-ppt-skill 大师 PPT', link: '/zh/02-connect-tools/03-workbuddy-dashi-ppt' },
      { text: '2.2.3 🌟 实战二与三：AI 自动填表与定时任务', link: '/zh/02-connect-tools/04-workbuddy-excel-and-cron' },
      { text: '2.3 接入 Qoder 全离线 AI 编程 IDE', link: '/zh/02-connect-tools/05-qoder-ide' },
      { text: '2.4 接入 Codex CLI 与解剖百行 toy-cli', link: '/zh/02-connect-tools/06-codex-and-toy-cli' }
    ]
  },
  {
    text: '第 3 章｜提示词闯关与个人知识库',
    collapsed: false,
    items: [
      { text: '导览：提示词 vs RAG 决策边界', link: '/zh/03-prompt-and-rag/' },
      { text: '3.1 提示词闯关：6 关驯服本地模型', link: '/zh/03-prompt-and-rag/01-prompt-6-levels' },
      { text: '3.2 黑暗料理菜谱切分对比实验', link: '/zh/03-prompt-and-rag/02-chunking-intuition' },
      { text: '3.3~3.4 WorkBuddy + Obsidian 双链知识库', link: '/zh/03-prompt-and-rag/03-workbuddy-obsidian-rag' }
    ]
  },
  {
    text: '第 4 章｜知识管理 MCP 与 Skills',
    collapsed: false,
    items: [
      { text: '4.1 ReAct 循环与 MCP 万能扩展坞', link: '/zh/04-mcp-and-skills/' },
      { text: '4.2 🌟 实战：可共享「知识管理 MCP」', link: '/zh/04-mcp-and-skills/02-knowledge-mcp-server' },
      { text: '4.3 SKILL.md 省显存原理与办公技能', link: '/zh/04-mcp-and-skills/03-custom-office-skill' },
      { text: '4.4 四道安全护栏与助手 v2.0 验收', link: '/zh/04-mcp-and-skills/04-safety-guardrails' }
    ]
  },
  {
    text: '第 5 章｜第一次中文模型微调',
    collapsed: false,
    items: [
      { text: '5.1 微调决策树与纯中文技术栈', link: '/zh/05-fine-tune/' },
      { text: '5.2 LoRA 显存账与弱智吧 5 分钟破冰', link: '/zh/05-fine-tune/02-lora-and-ruozhiba' },
      { text: '5.3 主线：中文群通知 JSON 抽取微调', link: '/zh/05-fine-tune/03-office-json-sft' },
      { text: '5.4 自建中文测试集前后客观评测', link: '/zh/05-fine-tune/04-eval-before-after' },
      { text: '5.5 🌟 WeClone 微信分身与小红书种草体', link: '/zh/05-fine-tune/05-weclone-and-xhs' }
    ]
  },
  {
    text: '第 6 章｜AMD 量化对比与自动加速',
    collapsed: false,
    items: [
      { text: '6.1 两阶段瓶颈与 TraceLens 体检', link: '/zh/06-optimize-speed/' },
      { text: '6.2 🌟 AMD Quark 单模型量化前后对比', link: '/zh/06-optimize-speed/02-quark-before-after' },
      { text: '6.3 零 C++ 唤醒 @torch.compile 与 AITER', link: '/zh/06-optimize-speed/03-zero-cpp-operators' },
      { text: '6.4 Prefix Caching 秒回与 NPU+GPU 异构', link: '/zh/06-optimize-speed/04-agent-prefix-caching' },
      { text: '6.5 🌟 Hyperloom 与 AMD Skills 自动寻优', link: '/zh/06-optimize-speed/05-hyperloom-auto-opt' }
    ]
  },
  {
    text: '第 7 章｜作品集与简历成果锻造',
    collapsed: false,
    items: [
      { text: '7.1 写下你的本地 AI 改造成功故事', link: '/zh/07-portfolio-resume/' },
      { text: '7.2 个人 AI 作品集「五件套」打包指南', link: '/zh/07-portfolio-resume/02-portfolio-5-pieces' },
      { text: '7.3 STAR 简历模板、面试真题与深造驿站', link: '/zh/07-portfolio-resume/03-star-resume-interview' }
    ]
  },
  {
    text: '第 8 章｜15 篇趣味实战案例超市',
    collapsed: false,
    items: [
      { text: '全景选菜矩阵与新手试学机制', link: '/zh/08-fun-cases/' },
      { text: 'Case 01 🌟 大师 PPT 一键出可编辑课件', link: '/zh/08-fun-cases/case-01-dashi-ppt' },
      { text: 'Case 02 🌟 AI 零幻觉填表与文件夹归档', link: '/zh/08-fun-cases/case-02-auto-excel-archive' },
      { text: 'Case 03 🌟 AI 定时任务日常办公值守', link: '/zh/08-fun-cases/case-03-scheduled-office' },
      { text: 'Case 04 全本地万字深度调研报告专家', link: '/zh/08-fun-cases/case-04-local-deep-research' },
      { text: 'Case 05 多篇英文论文横评直出组会 PPT', link: '/zh/08-fun-cases/case-05-paper-review-ppt' },
      { text: 'Case 06 双语公式图表零损精翻神器', link: '/zh/08-fun-cases/case-06-pdf-math-translate' },
      { text: 'Case 07 Obsidian 双链与可共享知识库 MCP', link: '/zh/08-fun-cases/case-07-obsidian-knowledge-mcp' },
      { text: 'Case 08 复习讲义秒变双人播客与分身联动', link: '/zh/08-fun-cases/case-08-podcast-and-weclone' },
      { text: 'Case 09 全离线保密会议与网课纪要神器', link: '/zh/08-fun-cases/case-09-offline-meeting-minutes' },
      { text: 'Case 10 桌面 Live2D 实时英语口语陪练', link: '/zh/08-fun-cases/case-10-live2d-voice-tutor' },
      { text: 'Case 11 Qoder 全离线内网重构与单测', link: '/zh/08-fun-cases/case-11-qoder-offline-ide' },
      { text: 'Case 12 Codex 终端修 Bug 与魔改 toy-cli', link: '/zh/08-fun-cases/case-12-codex-and-toy-cli' },
      { text: 'Case 13 高德 MCP 智能旅行规划与网页操控', link: '/zh/08-fun-cases/case-13-travel-planner-browser' },
      { text: 'Case 14 OpenClaw 24h 全隐私飞书管家中枢', link: '/zh/08-fun-cases/case-14-openclaw-hub' },
      { text: 'Case 15 扫雷挑战赛、Minecraft 与跑团剧场', link: '/zh/08-fun-cases/case-15-minesweeper-and-game-ai' },
      { text: '附：五要素标准案例贡献模板', link: '/zh/08-fun-cases/_CASE_TEMPLATE' }
    ]
  },
  {
    text: '第 9 章｜AMD 优秀项目与资源百宝箱',
    collapsed: false,
    items: [
      { text: '9.1 AMD 官方明星开源项目全景巡礼', link: '/zh/09-amd-ecosystem/' },
      { text: '9.2 前沿科研与工业案例精讲 (具身/搜广推)', link: '/zh/09-amd-ecosystem/02-research-and-industry-cases' },
      { text: '9.3 全系 AMD gfx 硬件大字典与算力福利', link: '/zh/09-amd-ecosystem/03-hardware-dictionary-and-cloud' }
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

const enSidebar: DefaultTheme.Sidebar = {}

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
  description: '每个 AI 小白，从零开始，动手实践的第一站 —— 基于 AMD 平台的 AI 与 Agentic AI 入门实践手册',
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
