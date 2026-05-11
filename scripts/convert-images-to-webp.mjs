#!/usr/bin/env node
/*
 * Generate WebP copies of docs/public/images/**.{png,jpg,jpeg} alongside the
 * originals, then rewrite references inside docs/** so VitePress serves .webp
 * at build time. The source files are kept untouched — CI only needs to ship
 * the WebP variants, and local dev is unaffected.
 *
 * Requires the `cwebp` binary (apt: webp). Fails loudly if missing so we don't
 * silently ship unoptimized PNGs.
 */

import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname, extname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '..')
const docsRoot = join(repoRoot, 'docs')
const imagesRoot = join(docsRoot, 'public', 'images')

const IMAGE_EXT = new Set(['.png', '.jpg', '.jpeg'])
const TEXT_EXT = new Set(['.md', '.vue', '.ts', '.mts', '.js', '.mjs', '.json', '.html', '.css'])
const SKIP_DIRS = new Set(['node_modules', '.git', 'cache', 'dist', '.vitepress/cache', '.vitepress/dist'])

const CWEBP_QUALITY = process.env.CWEBP_QUALITY || '82'

function assertCwebp() {
  const probe = spawnSync('cwebp', ['-version'], { stdio: 'ignore' })
  if (probe.status !== 0) {
    console.error('cwebp not found in PATH. Install libwebp (apt install webp).')
    process.exit(1)
  }
}

function walk(dir, onFile) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.') && entry.name !== '.vitepress') continue
    const full = join(dir, entry.name)
    if (entry.isDirectory()) {
      if (SKIP_DIRS.has(entry.name)) continue
      walk(full, onFile)
    } else if (entry.isFile()) {
      onFile(full)
    }
  }
}

function convertImages() {
  let converted = 0
  let reused = 0
  let failed = 0
  let bytesSource = 0
  let bytesWebp = 0

  walk(imagesRoot, (file) => {
    const ext = extname(file).toLowerCase()
    if (!IMAGE_EXT.has(ext)) return

    const webpPath = file.slice(0, -ext.length) + '.webp'
    const sourceSize = statSync(file).size
    bytesSource += sourceSize

    if (existsSync(webpPath) && statSync(webpPath).mtimeMs >= statSync(file).mtimeMs) {
      bytesWebp += statSync(webpPath).size
      reused += 1
      return
    }

    try {
      execFileSync('cwebp', ['-quiet', '-q', CWEBP_QUALITY, file, '-o', webpPath], {
        stdio: ['ignore', 'ignore', 'inherit']
      })
    } catch (err) {
      console.error(`cwebp failed for ${relative(repoRoot, file)}:`, err.message)
      failed += 1
      return
    }

    bytesWebp += statSync(webpPath).size
    converted += 1
  })

  const saved = bytesSource - bytesWebp
  const pct = bytesSource ? ((saved / bytesSource) * 100).toFixed(1) : '0'
  console.log(
    `WebP: converted ${converted}, reused ${reused}, failed ${failed}. ` +
    `${(bytesSource / 1024 / 1024).toFixed(2)} MB source -> ${(bytesWebp / 1024 / 1024).toFixed(2)} MB webp (${pct}% smaller).`
  )
}

function rewriteReferences() {
  // Match /images/... paths ending in .png/.jpg/.jpeg with optional query/fragment.
  // Only swap the extension; the rest of the path and any suffix is preserved.
  const pattern = /(\/images\/[^\s"'`)<>]+?)\.(png|jpg|jpeg)(\b)/gi
  let filesChanged = 0
  let replacements = 0

  walk(docsRoot, (file) => {
    if (file.startsWith(join(docsRoot, 'public'))) return
    if (file.startsWith(join(docsRoot, '.vitepress', 'cache'))) return
    if (file.startsWith(join(docsRoot, '.vitepress', 'dist'))) return
    const ext = extname(file).toLowerCase()
    if (!TEXT_EXT.has(ext)) return

    const src = readFileSync(file, 'utf8')
    let localCount = 0
    const next = src.replace(pattern, (_m, prefix, _extMatch, trailing) => {
      localCount += 1
      return `${prefix}.webp${trailing}`
    })
    if (localCount > 0) {
      writeFileSync(file, next)
      filesChanged += 1
      replacements += localCount
    }
  })

  console.log(`Rewrote ${replacements} references across ${filesChanged} files.`)
}

assertCwebp()
convertImages()
rewriteReferences()
