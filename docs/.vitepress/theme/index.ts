import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import { useData, useRoute } from 'vitepress'
import 'viewerjs/dist/viewer.min.css'
import imageViewer from 'vitepress-plugin-image-viewer'
import vImageViewer from 'vitepress-plugin-image-viewer/lib/vImageViewer.vue'
import Mermaid from './components/Mermaid.vue'
import ScrollTopProgress from './components/ScrollTopProgress.vue'
import './style.css'
import './custom.css'

const skillCopy = {
  zh: {
    eyebrow: 'hello-rocm Skill',
    title: '把本项目一键交给你的 AI 助手',
    description: '复制下面这句话，粘贴给支持 Skill / Rules / Agent 配置的工具，让它自动判断如何加载 src/hello-rocm-skill。',
    button: '一键复制 Skill 使用提示',
    copied: '已复制，粘贴给你的 AI 助手即可',
    note: '适合询问 GPU 架构、ROCm 快速安装、vLLM / Ollama / llama.cpp 部署、常见报错排查与学习路径。',
    prompt: '请使用当前仓库的 src/hello-rocm-skill 作为 hello-rocm Skill；如果你的工具支持 Skills、Rules 或 Agent 配置，请把它安装或加载到合适位置（例如 .claude/skills、.cursor/skills 或 .agents/skills），然后根据该 Skill 帮我学习、部署和排查 AMD ROCm。',
  },
  en: {
    eyebrow: 'hello-rocm Skill',
    title: 'Give this project to your AI assistant in one copy',
    description: 'Copy the sentence below and paste it into any tool that supports Skills, Rules, or Agent configuration. Let the tool decide how to load src/hello-rocm-skill.',
    button: 'Copy Skill prompt',
    copied: 'Copied — paste it into your AI assistant',
    note: 'Use it for GPU architecture lookup, ROCm quick installs, vLLM / Ollama / llama.cpp deployment, troubleshooting, and learning paths.',
    prompt: 'Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place, such as .claude/skills, .cursor/skills, or .agents/skills, then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.',
  },
}

function SkillCopyCard() {
  const { lang } = useData()
  const copy = lang.value?.startsWith('zh') ? skillCopy.zh : skillCopy.en

  const copyPrompt = async (event: MouseEvent) => {
    const button = event.currentTarget as HTMLButtonElement
    const original = copy.button
    await navigator.clipboard.writeText(copy.prompt)
    button.textContent = copy.copied
    window.setTimeout(() => {
      button.textContent = original
    }, 1800)
  }

  return h('div', { class: 'home-skill-copy-wrap' }, [
    h('div', { class: 'home-skill-copy' }, [
      h('p', { class: 'home-skill-eyebrow' }, copy.eyebrow),
      h('h2', copy.title),
      h('p', [copy.description.split('src/hello-rocm-skill')[0], h('code', 'src/hello-rocm-skill'), copy.description.split('src/hello-rocm-skill')[1] ?? '']),
      h('button', { class: 'home-skill-copy-button', onClick: copyPrompt }, copy.button),
      h('p', { class: 'home-skill-note' }, copy.note),
    ]),
  ])
}

export default {
  extends: DefaultTheme,
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      'home-features-before': () => h(SkillCopyCard),
      'layout-bottom': () => h(ScrollTopProgress),
    }),
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp?.(ctx)
    ctx.app.component('vImageViewer', vImageViewer)
    ctx.app.component('Mermaid', Mermaid)
  },
  setup() {
    const route = useRoute()
    imageViewer(route)
  }
} satisfies Theme
