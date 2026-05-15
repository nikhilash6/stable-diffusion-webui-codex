#!/usr/bin/env node
import net from 'node:net'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const COLOR = {
  red: (s) => `\u001b[31m${s}\u001b[0m`,
  yellow: (s) => `\u001b[33m${s}\u001b[0m`,
  cyan: (s) => `\u001b[36m${s}\u001b[0m`,
}

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))

const PORT_GUARD_HOSTS = [
  // IPv4 wildcard + loopback
  '0.0.0.0',
  '127.0.0.1',
  // IPv6 wildcard + loopback (covers the common “localhost → ::1” split-brain case)
  '::',
  '::1',
]

function isWsl() {
  const env = process.env.WSL_DISTRO_NAME || ''
  if (env) return true
  const rel = os.release().toLowerCase()
  return rel.includes('microsoft') || rel.includes('wsl')
}

function repoRoot() {
  const env = String(process.env.CODEX_ROOT || '').trim()
  if (env && fs.existsSync(path.join(env, 'apps')) && fs.existsSync(path.join(env, '.gitignore'))) {
    return env
  }
  const candidate = path.resolve(SCRIPT_DIR, '..', '..', '..')
  if (fs.existsSync(path.join(candidate, 'apps')) && fs.existsSync(path.join(candidate, '.gitignore'))) {
    return candidate
  }
  // Last resort: preserve old behaviour for ad-hoc invocations.
  return process.cwd()
}

function interfaceRoot() {
  const root = repoRoot()
  const iface = path.join(root, 'apps', 'interface')
  if (fs.existsSync(path.join(iface, 'package.json'))) return iface
  return process.cwd()
}

function pidFilePath(port) {
  const root = repoRoot()
  const safePort = Number.isFinite(port) ? String(port) : 'unknown'
  return path.join(root, `.webui-ui-${safePort}.pid`)
}

function writePidFile(port) {
  const file = pidFilePath(port)
  const launcherUiToken = String(process.env.CODEX_LAUNCHER_UI_INSTANCE_TOKEN || '').trim()
  const payload = {
    service: 'ui',
    pid: process.pid,
    port,
    started_at: new Date().toISOString(),
    platform: process.platform,
    hostname: os.hostname(),
    wsl: isWsl(),
    cwd: process.cwd(),
  }
  if (launcherUiToken) payload.launcher_ui_token = launcherUiToken
  try {
    fs.writeFileSync(file, JSON.stringify(payload, null, 2), 'utf-8')
  } catch (_) {
    // best-effort only; never block dev server
  }
  return file
}

function installPidCleanup(file) {
  const cleanup = () => {
    try { fs.unlinkSync(file) } catch (_) {}
  }
  process.on('exit', cleanup)
  process.on('SIGINT', () => { cleanup(); process.exit(130) })
  process.on('SIGTERM', () => { cleanup(); process.exit(143) })
}

function checkPortOnHost(port, host) {
  return new Promise((resolve) => {
    const srv = net.createServer()
    srv.once('error', (err) => resolve({ ok: false, host, code: err?.code || 'UNKNOWN' }))
    srv.once('listening', () => srv.close(() => resolve({ ok: true, host })))
    srv.listen(port, host)
  })
}

async function checkPortEverywhere(port) {
  for (const host of PORT_GUARD_HOSTS) {
    // eslint-disable-next-line no-await-in-loop
    const res = await checkPortOnHost(port, host)
    if (res.ok) continue
    // Ignore unsupported address families; still catch EADDRINUSE.
    if (res.code === 'EAFNOSUPPORT' || res.code === 'EADDRNOTAVAIL') continue
    return { ok: false, host: res.host, code: res.code }
  }
  return { ok: true }
}

async function probeCodex(port) {
  const urls = [
    `http://127.0.0.1:${port}/api/version`,
    `http://[::1]:${port}/api/version`,
  ]
  for (const url of urls) {
    try {
      const ctrl = new AbortController()
      const t = setTimeout(() => ctrl.abort(), 800)
      // Node 18+ has global fetch
      // eslint-disable-next-line no-await-in-loop
      const res = await fetch(url, { signal: ctrl.signal })
      clearTimeout(t)
      if (!res.ok) continue
      // eslint-disable-next-line no-await-in-loop
      const data = await res.json().catch(() => null)
      if (!data || typeof data !== 'object') continue
      if (typeof data.app_version !== 'string') continue
      return { url, data }
    } catch (_) {
      // ignore probes
    }
  }
  return null
}

const cliArgs = process.argv.slice(2)

async function main() {
  const base = Number(process.env.WEB_PORT) || 7860
  const c1 = await checkPortEverywhere(base)
  if (c1.ok) return runVite(base, false, cliArgs)

  const existing = await probeCodex(base)
  if (existing) {
    const commit = existing.data?.git_commit || 'unknown'
    console.log(COLOR.yellow(`[port-guard] Port ${base} already responds to /api/version (${existing.url}, git=${commit}). You may have another Codex UI/API running (WSL/Windows).`))
  } else {
    console.log(COLOR.yellow(`[port-guard] Port ${base} is busy. You may have another service (or another Codex instance) running.`))
  }
  const f1 = base + 10000
  const c2 = await checkPortEverywhere(f1)
  if (c2.ok) {
    banner(f1, base)
    return runVite(f1, true, cliArgs)
  }
  const f2 = base + 20000
  const c3 = await checkPortEverywhere(f2)
  if (c3.ok) {
    banner(f2, base)
    return runVite(f2, true, cliArgs)
  }
  console.error(COLOR.red(`[port-guard] No free port for UI. Tried ${base}, ${f1}, ${f2}.`))
  if (!c1.ok) console.error(COLOR.red(`[port-guard] ${base} blocked at host=${c1.host} code=${c1.code}`))
  if (!c2.ok) console.error(COLOR.red(`[port-guard] ${f1} blocked at host=${c2.host} code=${c2.code}`))
  if (!c3.ok) console.error(COLOR.red(`[port-guard] ${f2} blocked at host=${c3.host} code=${c3.code}`))
  process.exit(1)
}

function runVite(port, show, extraArgs = []) {
  process.env.WEB_PORT = String(port)
  const pidFile = writePidFile(port)
  installPidCleanup(pidFile)
  const cwd = interfaceRoot()
  const viteEntrypoint = path.join(cwd, 'node_modules', 'vite', 'bin', 'vite.js')
  if (!fs.existsSync(viteEntrypoint)) {
    console.error(COLOR.red(`[port-guard] Missing Vite entrypoint at ${viteEntrypoint}. Run 'npm install' in apps/interface.`))
    process.exit(1)
  }
  const child = spawn(process.execPath, [viteEntrypoint, ...extraArgs], {
    stdio: 'inherit',
    env: process.env,
    cwd,
    shell: false,
  })
  child.on('error', (err) => {
    console.error(COLOR.red(`[port-guard] Failed to launch Vite: ${err?.stack || err}`))
    process.exit(1)
  })
  child.on('exit', (code) => process.exit(code ?? 0))
}

function banner(chosen, base) {
  const text = [
    '',
    '==============================================',
    '  PORT GUARD — UI Fallback                   ',
    '==============================================',
    ` Using UI port ${chosen} (base ${base} busy).`,
    '==============================================',
    ''
  ].join('\n')
  console.log(COLOR.cyan(text))
}

main().catch((err) => {
  console.error(COLOR.red(`[port-guard] ${err?.stack || err}`))
  process.exit(1)
})
