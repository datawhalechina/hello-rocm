import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import { useRoute } from 'vitepress'
import 'viewerjs/dist/viewer.min.css'
import imageViewer from 'vitepress-plugin-image-viewer'
import vImageViewer from 'vitepress-plugin-image-viewer/lib/vImageViewer.vue'
import Mermaid from './components/Mermaid.vue'
import ScrollTopProgress from './components/ScrollTopProgress.vue'
import './style.css'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout: () =>
    h(DefaultTheme.Layout, null, {
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
