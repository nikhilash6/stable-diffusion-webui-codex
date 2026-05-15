/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vite configuration for the Codex WebUI frontend.
Configures dev server host/ports + API proxy/HMR and a small plugin that watches root env files to trigger a restart on change.

Symbols (top-level; keep in sync; no ghosts):
- `watchRootConfigs` (function): Vite plugin that watches root config files and restarts the dev server on change.
- `default` (function): Vite config factory that loads env from repo root and returns dev/build settings.
*/

// tags: vite-config, frontend-build
import { defineConfig, loadEnv } from 'vite'
import path from 'node:path'
import vue from '@vitejs/plugin-vue'
import tailwind from '@tailwindcss/vite'

// Restart vite dev server if root env files change.
const watchRootConfigs = () => ({
  name: 'watch-root-configs',
  configureServer(server) {
    const root = path.resolve(__dirname, '../../')
    server.watcher.add(path.join(root, '.env*'))
    server.watcher.on('change', (file) => {
      const base = path.basename(file)
      if (base.startsWith('.env')) {
        console.log(`[vite] Detected config change (${base}). Restarting dev server...`)
        server.restart()
      }
    })
  }
})

export default defineConfig(({ mode }) => {
  const repoRoot = path.resolve(__dirname, '../../')
  const env = loadEnv(mode, repoRoot, '')

  const WEB_PORT = Number(env.WEB_PORT || process.env.WEB_PORT || 7860)
  const SERVER_HOST = String(env.SERVER_HOST || process.env.SERVER_HOST || 'localhost')
  const API_HOST = String(env.API_HOST || process.env.API_HOST || 'localhost')
  // If API_PORT not explicitly provided, derive from UI port by -10 (7860→7850, 17860→17850, 27860→27850)
  let API_PORT = Number(env.API_PORT || process.env.API_PORT || (WEB_PORT - 10))
  const HMR_HOST = String(env.HMR_HOST || process.env.HMR_HOST || SERVER_HOST)
  const HMR_PROTOCOL = String(env.HMR_PROTOCOL || process.env.HMR_PROTOCOL || 'ws')

  const allowedHosts = new Set<string>(['localhost', '127.0.0.1', '::1'])
  ;(env.ALLOWED_HOSTS || process.env.ALLOWED_HOSTS || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .forEach((h) => allowedHosts.add(h))

  return {
    envDir: repoRoot,
    plugins: [vue(), tailwind(), watchRootConfigs()],
    server: {
      port: WEB_PORT,
      strictPort: true,
      host: SERVER_HOST,
      allowedHosts: Array.from(allowedHosts),
      proxy: {
        '/api': {
          target: `http://${API_HOST}:${API_PORT}`,
          changeOrigin: true
        }
      },
      // Avoid full-page reloads triggered by backend runtime state writes under apps/interface/*.json.
      // These are not frontend source modules (they are persisted by the backend).
      watch: {
        ignored: ['**/tabs.json', '**/workflows.json'],
      },
      hmr: {
        host: HMR_HOST,
        port: WEB_PORT,
        protocol: HMR_PROTOCOL as 'ws' | 'wss'
      }
    }
  }
})
