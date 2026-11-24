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
