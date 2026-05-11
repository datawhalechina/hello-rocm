<template>
  <div ref="mermaidContainer" class="mermaid-container">
    <div v-if="error" class="mermaid-error">
      <p>❌ Mermaid 渲染错误：</p>
      <pre>{{ error }}</pre>
    </div>
    <div v-else ref="mermaidOutput" class="mermaid-output"></div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import mermaid from 'mermaid'

const props = defineProps<{
  code: string
  id?: string
  encoded?: boolean
}>()

const mermaidContainer = ref<HTMLElement>()
const mermaidOutput = ref<HTMLElement>()
const error = ref<string>('')
const mermaidCode = computed(() => props.encoded ? decodeURIComponent(props.code) : props.code)
let themeObserver: MutationObserver | undefined

// Dark-mode overrides for known classDef names used across /docs.
// Mermaid compiles classDef styles into the SVG's internal <style> with an
// ID-selector prefix, so external CSS can't beat them on specificity. We
// rewrite the classDef source instead before parsing.
const darkClassDefs: Record<string, string> = {
  'layer-framework': 'fill:#0a2e16,stroke:#4ade80,stroke-width:1px,color:#d1fae5',
  'layer-interface': 'fill:#3d2600,stroke:#fbbf24,stroke-width:1px,color:#fef3c7',
  'layer-library': 'fill:#2a1050,stroke:#a78bfa,stroke-width:1px,color:#ede9fe',
  'layer-driver': 'fill:#1e3a5f,stroke:#60a5fa,stroke-width:1px,color:#dbeafe',
  'layer-hardware': 'fill:#3d0720,stroke:#f472b6,stroke-width:1px,color:#fce7f3',
  'layer-compute': 'fill:#3d2600,stroke:#fb923c,stroke-width:1px,color:#ffedd5'
}

const applyDarkClassDefs = (code: string): string => {
  let out = code
  for (const [name, style] of Object.entries(darkClassDefs)) {
    const re = new RegExp(`(^|\\n)([ \\t]*)classDef[ \\t]+${name}\\b[^\\n]*`, 'g')
    out = out.replace(re, `$1$2classDef ${name} ${style}`)
  }
  return out
}

// 初始化 Mermaid
const initMermaid = () => {
  // 检测当前主题（暗色或亮色）
  const isDark = document.documentElement.classList.contains('dark')

  mermaid.initialize({
    startOnLoad: false,
    theme: isDark ? 'dark' : 'default',
    securityLevel: 'loose',
    fontFamily: 'inherit',
    logLevel: 'error'
  })
}

// 渲染 Mermaid 图表
const renderMermaid = async () => {
  if (!mermaidOutput.value || !mermaidCode.value) return

  try {
    error.value = ''

    // 生成唯一 ID
    const id = props.id || `mermaid-${Math.random().toString(36).substr(2, 9)}`

    const isDark = document.documentElement.classList.contains('dark')
    const source = isDark ? applyDarkClassDefs(mermaidCode.value) : mermaidCode.value

    // 渲染图表
    const { svg } = await mermaid.render(id, source)

    // 插入 SVG
    mermaidOutput.value.innerHTML = svg
  } catch (err: any) {
    console.error('Mermaid rendering error:', err)
    error.value = err.message || String(err)
  }
}

// 监听主题变化
const observeThemeChange = () => {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.attributeName === 'class') {
        initMermaid()
        renderMermaid()
      }
    })
  })

  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class']
  })

  return observer
}

onMounted(() => {
  initMermaid()
  renderMermaid()
  
  // 监听主题变化
  themeObserver = observeThemeChange()
})

// 监听 code 变化
watch(mermaidCode, () => {
  renderMermaid()
})

onBeforeUnmount(() => {
  themeObserver?.disconnect()
})
</script>

<style scoped>
.mermaid-container {
  margin: 1.5rem 0;
  padding: 1rem;
  background-color: var(--vp-c-bg-soft);
  border-radius: 8px;
  overflow-x: auto;
}

.mermaid-output {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 60px;
}

.mermaid-output :deep(svg) {
  max-width: none;
  height: auto;
}

/*
 * VitePress's .vp-doc styles (large line-height, paragraph margins, custom
 * font-size) leak into mermaid's htmlLabels foreignObjects and make the text
 * render larger than what mermaid measured when sizing the node rects, so
 * labels overflow. Reset label typography back to mermaid's expected values.
 */
.mermaid-output :deep(foreignObject) {
  overflow: visible;
}

.mermaid-output :deep(foreignObject div),
.mermaid-output :deep(foreignObject p),
.mermaid-output :deep(foreignObject span),
.mermaid-output :deep(.nodeLabel),
.mermaid-output :deep(.edgeLabel),
.mermaid-output :deep(.cluster-label),
.mermaid-output :deep(.label) {
  margin: 0 !important;
  padding: 0 !important;
  font-size: 14px !important;
  line-height: 1.3 !important;
  font-weight: 400 !important;
  letter-spacing: normal !important;
  white-space: normal;
}

.mermaid-output :deep(foreignObject p + p) {
  margin-top: 0 !important;
}

.mermaid-output :deep(.edgeLabel) {
  padding: 2px 4px !important;
}

.mermaid-error {
  color: var(--vp-c-danger-1);
  padding: 1rem;
  background-color: var(--vp-c-danger-soft);
  border-radius: 4px;
}

.mermaid-error pre {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background-color: var(--vp-c-bg);
  border-radius: 4px;
  overflow-x: auto;
  font-size: 0.875rem;
}
</style>
