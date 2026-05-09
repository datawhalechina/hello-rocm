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
    
    // 渲染图表
    const { svg } = await mermaid.render(id, mermaidCode.value)
    
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
}

.mermaid-output :deep(svg) {
  max-width: 100%;
  height: auto;
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
