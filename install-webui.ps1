[CmdletBinding(PositionalBinding = $false)]
param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$CliArgs
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Install {
  param([Parameter(Mandatory = $true)][string]$Message)
  Write-Host "[install] $Message"
}

function Write-InstallWarning {
  param([Parameter(Mandatory = $true)][string]$Message)
  Write-Host "[install] Warning: $Message" -ForegroundColor Yellow
}

function Fail {
  param([Parameter(Mandatory = $true)][string]$Message)
  Write-Error $Message
  exit 1
}

function Show-Usage {
  Write-Host "Usage: install-webui.cmd [--menu|--no-menu|--simple|--check|--reinstall-deps|--help]"
  Write-Host "  --check           Verify installer-managed dependencies only."
  Write-Host "  --reinstall-deps  Reinstall dependencies in-place without deleting environments."
}

function Ensure-Dir {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
  }
}

function Get-Root {
  $root = Split-Path -Parent $PSCommandPath
  if ([string]::IsNullOrWhiteSpace($root)) {
    Fail "Error: unable to resolve script root."
  }
  if (-not $root.EndsWith("\")) {
    $root += "\"
  }
  return $root
}

function Set-Or-PrependPythonPath {
  param([Parameter(Mandatory = $true)][string]$Root)
  $existing = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
  if ([string]::IsNullOrWhiteSpace($existing)) {
    $env:PYTHONPATH = $Root
    return
  }
  $env:PYTHONPATH = "$Root;$existing"
}

function Get-EnvValue {
  param([Parameter(Mandatory = $true)][string]$Name)
  $value = [Environment]::GetEnvironmentVariable($Name, "Process")
  if ($null -ne $value) {
    return [string]$value
  }
  return ""
}

function Is-EnvDefined {
  param([Parameter(Mandatory = $true)][string]$Name)
  $value = [Environment]::GetEnvironmentVariable($Name, "Process")
  return -not [string]::IsNullOrEmpty($value)
}

function Get-EnvOrDefault {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Default
  )
  $value = [Environment]::GetEnvironmentVariable($Name, "Process")
  if ([string]::IsNullOrWhiteSpace($value)) {
    return $Default
  }
  return $value
}

function Parse-BoolFromEnv {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][int]$Default
  )

  $raw = Get-EnvValue -Name $Name
  if ([string]::IsNullOrWhiteSpace($raw)) {
    return $Default
  }

  switch ($raw.Trim().ToLowerInvariant()) {
    "1" { return 1 }
    "true" { return 1 }
    "yes" { return 1 }
    "on" { return 1 }
    "0" { return 0 }
    "false" { return 0 }
    "no" { return 0 }
    "off" { return 0 }
    default {
      Fail "Error: invalid $Name='$raw' (expected 0|1|true|false|yes|no|on|off)."
    }
  }
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter()][string[]]$Arguments,
    [Parameter()][string]$WorkingDirectory,
    [Parameter()][string]$FailureMessage
  )

  if ([string]::IsNullOrWhiteSpace($FailureMessage)) {
    $FailureMessage = "Error: command failed: $Executable"
  }

  if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
    & $Executable @Arguments
  }
  else {
    Push-Location $WorkingDirectory
    try {
      & $Executable @Arguments
    }
    finally {
      Pop-Location
    }
  }

  if ($LASTEXITCODE -ne 0) {
    Fail $FailureMessage
  }
}

function Get-UvAssetName {
  $arch = $env:PROCESSOR_ARCHITECTURE
  if (-not [string]::IsNullOrWhiteSpace($env:PROCESSOR_ARCHITEW6432)) {
    $arch = $env:PROCESSOR_ARCHITEW6432
  }
  switch ($arch.ToUpperInvariant()) {
    "AMD64" { return "uv-x86_64-pc-windows-msvc.zip" }
    "ARM64" { return "uv-aarch64-pc-windows-msvc.zip" }
    "X86" { return "uv-i686-pc-windows-msvc.zip" }
    default { Fail "Error: unsupported Windows architecture '$arch'." }
  }
}

function Download-File {
  param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Destination
  )

  if (Test-Path -LiteralPath $Destination) {
    Remove-Item -Force -LiteralPath $Destination
  }

  $curl = Get-Command "curl.exe" -ErrorAction SilentlyContinue
  if ($null -ne $curl) {
    Write-Install "Downloading: $Url"
    & $curl.Source "-L" "--fail" "--retry" "3" "--retry-delay" "2" "-o" $Destination $Url
    if ($LASTEXITCODE -eq 0) {
      return
    }
  }

  $certutil = Get-Command "certutil.exe" -ErrorAction SilentlyContinue
  if ($null -ne $certutil) {
    Write-Install "Downloading via certutil: $Url"
    & $certutil.Source "-urlcache" "-split" "-f" $Url $Destination | Out-Null
    if ($LASTEXITCODE -eq 0) {
      return
    }
  }

  Write-Install "Downloading via Invoke-WebRequest: $Url"
  try {
    Invoke-WebRequest -Uri $Url -OutFile $Destination
    return
  }
  catch {
    Fail "Error: failed to download '$Url'. This can be AV/Defender interference; try allow-listing the download and rerun."
  }
}

function Extract-Zip {
  param(
    [Parameter(Mandatory = $true)][string]$ZipPath,
    [Parameter(Mandatory = $true)][string]$DestinationDir
  )

  Ensure-Dir -Path $DestinationDir

  $tar = Get-Command "tar.exe" -ErrorAction SilentlyContinue
  if ($null -ne $tar) {
    Write-Install "Extracting uv via tar ..."
    & $tar.Source "-xf" $ZipPath "-C" $DestinationDir
    if ($LASTEXITCODE -eq 0) {
      return
    }
  }

  Write-Install "Extracting uv via PowerShell ZipFile ..."
  try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($ZipPath, $DestinationDir, $true)
  }
  catch {
    Fail "Error: failed to extract uv zip."
  }
}

function Ensure-Uv {
  param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$UvVersion
  )

  $uvDir = Join-Path $Root ".uv\bin"
  $uvBin = Join-Path $uvDir "uv.exe"
  if (Test-Path -LiteralPath $uvBin) {
    return @{ UvDir = $uvDir; UvBin = $uvBin }
  }

  Ensure-Dir -Path $uvDir
  $asset = Get-UvAssetName
  $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$asset"
  $zipPath = Join-Path $uvDir $asset

  Write-Install "Installing uv $UvVersion into $uvDir ..."
  Download-File -Url $url -Destination $zipPath
  Extract-Zip -ZipPath $zipPath -DestinationDir $uvDir
  Remove-Item -Force -LiteralPath $zipPath -ErrorAction SilentlyContinue

  if (-not (Test-Path -LiteralPath $uvBin)) {
    Fail "Error: uv extracted but '$uvBin' is missing."
  }

  return @{ UvDir = $uvDir; UvBin = $uvBin }
}

function Resolve-CudaExtra {
  param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$CudaVariant)

  switch ($CudaVariant.Trim().ToLowerInvariant()) {
    "12.6" { return "cu126" }
    "cu126" { return "cu126" }
    "12.8" { return "cu128" }
    "cu128" { return "cu128" }
    "13" { return "cu130" }
    "cu130" { return "cu130" }
    default { return "cu128" }
  }
}

function Try-ParseInt {
  param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Raw)
  if ([string]::IsNullOrWhiteSpace($Raw)) {
    return $null
  }
  $value = 0
  if ([int]::TryParse($Raw, [ref]$value)) {
    return $value
  }
  return $null
}

function Try-GetNvidiaInfo {
  $nvidia = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
  if ($null -eq $nvidia) {
    $nvidia = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
  }
  if ($null -eq $nvidia) {
    return $null
  }

  $cudaRaw = ((& $nvidia.Source "--query-gpu=cuda_version" "--format=csv,noheader" 2>$null) | Select-Object -First 1)
  $driverRaw = ((& $nvidia.Source "--query-gpu=driver_version" "--format=csv,noheader" 2>$null) | Select-Object -First 1)
  $gpuNameRaw = ((& $nvidia.Source "--query-gpu=name" "--format=csv,noheader" 2>$null) | Select-Object -First 1)

  $driverMajor = $null
  if (-not [string]::IsNullOrWhiteSpace($driverRaw)) {
    $driverMajorToken = (($driverRaw -as [string]).Trim().Split(".")[0])
    $driverMajor = Try-ParseInt -Raw $driverMajorToken
  }

  $cudaMajor = $null
  $cudaMinor = 0
  if (-not [string]::IsNullOrWhiteSpace($cudaRaw)) {
    $parts = ($cudaRaw -as [string]).Trim().Split(".")
    if ($parts.Length -ge 1) {
      $cudaMajor = Try-ParseInt -Raw $parts[0]
    }
    if ($parts.Length -ge 2) {
      $parsedMinor = Try-ParseInt -Raw $parts[1]
      if ($null -ne $parsedMinor) {
        $cudaMinor = $parsedMinor
      }
    }
  }

  $gpuName = ""
  if (-not [string]::IsNullOrWhiteSpace($gpuNameRaw)) {
    $gpuName = ($gpuNameRaw -as [string]).Trim()
  }

  return @{
    DriverMajor = $driverMajor
    CudaMajor = $cudaMajor
    CudaMinor = $cudaMinor
    GpuName = $gpuName
  }
}

function Resolve-TorchExtra {
  param(
    [Parameter(Mandatory = $true)][string]$TorchMode,
    [Parameter(Mandatory = $true)][AllowEmptyString()][string]$TorchBackend,
    [Parameter(Mandatory = $true)][AllowEmptyString()][string]$CudaVariant
  )

  if (-not [string]::IsNullOrWhiteSpace($TorchBackend)) {
    return $TorchBackend.ToLowerInvariant()
  }

  $mode = $TorchMode.Trim().ToLowerInvariant()
  switch ($mode) {
    "skip" { return "" }
    "cpu" { return "cpu" }
    "rocm" {
      Write-InstallWarning "ROCm is Linux-only; falling back to CPU."
      return "cpu"
    }
    "cuda" { return Resolve-CudaExtra -CudaVariant $CudaVariant }
    "custom" { return "" }
    "auto" {
      $info = Try-GetNvidiaInfo
      if ($null -eq $info) {
        return "cpu"
      }

      if (-not [string]::IsNullOrWhiteSpace($CudaVariant)) {
        return Resolve-CudaExtra -CudaVariant $CudaVariant
      }

      $torchExtra = "cu128"

      if ($null -ne $info.DriverMajor -and $info.DriverMajor -lt 525) {
        return "cpu"
      }

      if ($null -ne $info.CudaMajor -and $info.CudaMajor -eq 13) {
        if ($null -ne $info.DriverMajor -and $info.DriverMajor -ge 580) {
          return "cu130"
        }
        return "cu128"
      }

      if (-not [string]::IsNullOrWhiteSpace($info.GpuName) -and $info.GpuName -match "(?i)RTX\s*50|RTX\s*5[0-9]{3}") {
        return "cu128"
      }

      if ($null -ne $info.CudaMajor -and $info.CudaMajor -eq 12) {
        if ($info.CudaMinor -ge 8) {
          return "cu128"
        }
        if ($info.CudaMinor -ge 6) {
          return "cu126"
        }
        return "cu126"
      }

      return $torchExtra
    }
    default {
      Fail "Error: invalid CODEX_TORCH_MODE='$TorchMode' (expected auto|cpu|cuda|rocm|skip|custom)."
    }
  }
}

function Get-NodeenvTools {
  param([Parameter(Mandatory = $true)][string]$NodeenvDir)

  $node = Join-Path $NodeenvDir "Scripts\node.exe"
  $npm = Join-Path $NodeenvDir "Scripts\npm.cmd"
  if (-not (Test-Path -LiteralPath $node)) {
    $node = Join-Path $NodeenvDir "bin\node.exe"
  }
  if (-not (Test-Path -LiteralPath $npm)) {
    $npm = Join-Path $NodeenvDir "bin\npm.cmd"
  }

  if (-not (Test-Path -LiteralPath $node)) {
    return $null
  }
  if (-not (Test-Path -LiteralPath $npm)) {
    return $null
  }

  return @{ Node = $node; Npm = $npm }
}

function Prepend-NodeenvPath {
  param([Parameter(Mandatory = $true)][string]$NodeenvDir)

  if (-not [string]::IsNullOrWhiteSpace($env:CODEX_NODEENV_PATH_APPLIED)) {
    return
  }

  $scripts = Join-Path $NodeenvDir "Scripts"
  $bin = Join-Path $NodeenvDir "bin"

  if (Test-Path -LiteralPath (Join-Path $scripts "node.exe")) {
    $env:PATH = "$scripts;$env:PATH"
    if (Test-Path -LiteralPath (Join-Path $bin "node.exe")) {
      $env:PATH = "$bin;$env:PATH"
    }
    $env:CODEX_NODEENV_PATH_APPLIED = "1"
    return
  }

  if (Test-Path -LiteralPath (Join-Path $bin "node.exe")) {
    $env:PATH = "$bin;$env:PATH"
    $env:CODEX_NODEENV_PATH_APPLIED = "1"
  }
}

function Check-NodeenvOnly {
  param(
    [Parameter(Mandatory = $true)][string]$NodeenvDir,
    [Parameter(Mandatory = $true)][string]$NodeVersion
  )

  $tools = Get-NodeenvTools -NodeenvDir $NodeenvDir
  if ($null -eq $tools) {
    Fail "Error: node/npm executable missing under '$NodeenvDir'."
  }

  $nodeVersionRaw = (& $tools.Node "-v" 2>$null)
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($nodeVersionRaw)) {
    Fail "Error: failed to read Node.js version from '$($tools.Node)'."
  }

  $normalized = ($nodeVersionRaw.Trim()).TrimStart("v")
  if ($normalized -ne $NodeVersion) {
    Fail "Error: '$NodeenvDir' contains Node.js $normalized, expected $NodeVersion."
  }

  return $tools
}

function Install-NodeenvWithWindowsLinkFallback {
  param(
    [Parameter(Mandatory = $true)][string]$UvBin,
    [Parameter(Mandatory = $true)][string]$NodeVersion,
    [Parameter(Mandatory = $true)][string]$NodeenvDir
  )

  $args = @("tool", "run", "--from", "nodeenv", "nodeenv", "-n", $NodeVersion, $NodeenvDir)
  $rawOutput = @(& $UvBin @args 2>&1)
  $exitCode = $LASTEXITCODE

  $hasNodejsLinkWarning = $false
  foreach ($entry in $rawOutput) {
    $line = [string]$entry
    if ($line -match "(?i)Failed to create nodejs\.exe link") {
      $hasNodejsLinkWarning = $true
      break
    }
  }

  $suppressedLinkNoisePatterns = @(
    "(?i)^Error:\s*Failed to create nodejs\.exe link$",
    "(?i)^You do not have sufficient privileges to perform this operation\.?$",
    "(?i)^Voc.{0,3}\s*n.{0,3}o\s*tem\s*privil.{0,64}opera.{0,64}$"
  )

  foreach ($entry in $rawOutput) {
    $line = [string]$entry
    if ([string]::IsNullOrWhiteSpace($line)) {
      continue
    }

    if ($hasNodejsLinkWarning -and (Test-MatchAny -Value $line -Patterns $suppressedLinkNoisePatterns)) {
      continue
    }

    Write-Host $line
  }

  if ($exitCode -ne 0) {
    Fail "Error: nodeenv install failed."
  }

  if ($hasNodejsLinkWarning) {
    Write-InstallWarning "nodeenv could not create nodejs.exe link (insufficient Windows privileges). Continuing; node/npm availability will be validated next."
  }
}

function Ensure-Nodeenv {
  param(
    [Parameter(Mandatory = $true)][string]$UvBin,
    [Parameter(Mandatory = $true)][string]$NodeVersion,
    [Parameter(Mandatory = $true)][string]$NodeenvDir
  )

  $tools = Get-NodeenvTools -NodeenvDir $NodeenvDir
  if ($null -ne $tools) {
    Prepend-NodeenvPath -NodeenvDir $NodeenvDir
    $nodeVersionRaw = (& $tools.Node "-v" 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($nodeVersionRaw)) {
      Fail "Error: found '$NodeenvDir', but it does not contain a working node binary."
    }
    $normalized = ($nodeVersionRaw.Trim()).TrimStart("v")
    if ($normalized -ne $NodeVersion) {
      Fail "Error: '$NodeenvDir' already contains Node.js $normalized, but CODEX_NODE_VERSION=$NodeVersion."
    }
    return $tools
  }

  Write-Install "Installing Node.js $NodeVersion into $NodeenvDir ..."
  Install-NodeenvWithWindowsLinkFallback -UvBin $UvBin -NodeVersion $NodeVersion -NodeenvDir $NodeenvDir

  Prepend-NodeenvPath -NodeenvDir $NodeenvDir
  $installed = Get-NodeenvTools -NodeenvDir $NodeenvDir
  if ($null -eq $installed) {
    Fail "Error: nodeenv completed, but node/npm is missing under '$NodeenvDir'."
  }

  return $installed
}

function Select-CustomTorchFromManifest {
  param([Parameter(Mandatory = $true)][string]$ManifestUrl)

  if ([string]::IsNullOrWhiteSpace($ManifestUrl)) {
    throw "Error: CODEX_PYTORCH_MANIFEST_URL is empty."
  }

  $manifest = Invoke-RestMethod -Uri $ManifestUrl
  $entries = @()
  foreach ($candidate in @($manifest.windows_custom_torch)) {
    $url = [string]$candidate.url
    if (-not [string]::IsNullOrWhiteSpace($url)) {
      $entries += $candidate
    }
  }

  if ($entries.Count -eq 0) {
    throw "Error: manifest has no windows_custom_torch entries: $ManifestUrl"
  }

  Write-Host ""
  Write-Host "  Custom PyTorch options (from $ManifestUrl):"
  for ($index = 0; $index -lt $entries.Count; $index++) {
    $label = [string]$entries[$index].label
    if ([string]::IsNullOrWhiteSpace($label)) {
      $label = [string]$entries[$index].id
    }
    Write-Host ("   [{0}] {1}" -f ($index + 1), $label)
  }
  Write-Host "   [0] Back"

  $rawChoice = Read-Host ("Select custom wheel (0-{0})" -f $entries.Count)
  if ([string]::IsNullOrWhiteSpace($rawChoice) -or $rawChoice -eq "0") {
    return $null
  }

  if ($rawChoice -notmatch "^[0-9]+$") {
    throw "Error: invalid selection '$rawChoice'."
  }

  $choice = [int]$rawChoice
  if ($choice -lt 1 -or $choice -gt $entries.Count) {
    throw "Error: invalid selection '$rawChoice'."
  }

  $selected = $entries[$choice - 1]
  $url = [string]$selected.url
  if ([string]::IsNullOrWhiteSpace($url)) {
    throw "Error: selected entry has empty url."
  }

  $sha = [string]$selected.sha256
  if (-not [string]::IsNullOrWhiteSpace($sha)) {
    $sha = $sha.Trim().ToLowerInvariant()
  }

  return @{ Url = $url; Sha256 = $sha }
}

function Resolve-CustomSourceForInstall {
  param(
    [Parameter(Mandatory = $true)][string]$CustomSource,
    [Parameter()][string]$CustomSha256
  )

  $source = $CustomSource.Trim()
  if ([string]::IsNullOrWhiteSpace($source)) {
    Fail "Error: CODEX_TORCH_MODE=custom requires CODEX_CUSTOM_TORCH_SRC to point to a real wheel/source."
  }

  if ([string]::IsNullOrWhiteSpace($CustomSha256)) {
    return $source
  }

  $normalizedSha = $CustomSha256.Trim().ToLowerInvariant()
  if ($normalizedSha -notmatch "^[0-9a-f]{64}$") {
    Fail "Error: invalid CODEX_CUSTOM_TORCH_SHA256 '$CustomSha256'."
  }

  if ($source -match "^(?i)https?://") {
    $tmpDir = Join-Path $env:TEMP "codex-custom-torch"
    Ensure-Dir -Path $tmpDir
    $tmpFile = Join-Path $tmpDir ("custom-torch-{0}.whl" -f ([Guid]::NewGuid().ToString("N")))
    Download-File -Url $source -Destination $tmpFile
    $actual = (Get-FileHash -LiteralPath $tmpFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $normalizedSha) {
      Remove-Item -Force -LiteralPath $tmpFile -ErrorAction SilentlyContinue
      Fail "Error: custom torch SHA256 mismatch. expected=$normalizedSha actual=$actual"
    }
    return $tmpFile
  }

  if (-not (Test-Path -LiteralPath $source)) {
    Fail "Error: custom torch source path does not exist: $source"
  }

  $actualLocal = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actualLocal -ne $normalizedSha) {
    Fail "Error: custom torch SHA256 mismatch. expected=$normalizedSha actual=$actualLocal"
  }

  return $source
}

function Run-DependencyChecks {
  param(
    [Parameter(Mandatory = $true)][string]$UvBin,
    [Parameter(Mandatory = $true)][string]$VenvPython,
    [Parameter(Mandatory = $true)][string]$TorchMode,
    [Parameter(Mandatory = $true)][string]$NodeenvDir,
    [Parameter(Mandatory = $true)][string]$NodeVersion,
    [Parameter(Mandatory = $true)][string]$Root
  )

  if (-not (Test-Path -LiteralPath $UvBin)) {
    Fail "Error: uv binary not found at '$UvBin'. Run install mode first."
  }
  if (-not (Test-Path -LiteralPath $VenvPython)) {
    Fail "Error: venv python not found at '$VenvPython'. Run install mode first."
  }

  if ($TorchMode.Trim().ToLowerInvariant() -eq "skip") {
    Invoke-Checked -Executable $VenvPython -Arguments @("-c", "from apps.backend.video.runtime_dependencies import resolve_ffmpeg_binary, resolve_rife_model_path; import cv2, ccvfi; print('[install] torch check skipped (CODEX_TORCH_MODE=skip)'); print('[install] ffmpeg: ' + str(resolve_ffmpeg_binary('ffmpeg'))); print('[install] ffprobe: ' + str(resolve_ffmpeg_binary('ffprobe'))); print('[install] RIFE model: ' + str(resolve_rife_model_path(None))); print('[install] opencv-python: ' + cv2.__version__); print('[install] ccvfi: ' + str(getattr(ccvfi, '__version__', 'unknown')))") -FailureMessage "Error: Python dependency/runtime check failed."
  }
  else {
    Invoke-Checked -Executable $VenvPython -Arguments @("-c", "from apps.backend.video.runtime_dependencies import resolve_ffmpeg_binary, resolve_rife_model_path; import torch, cv2, ccvfi; print('[install] torch: ' + str(torch.__version__)); print('[install] ffmpeg: ' + str(resolve_ffmpeg_binary('ffmpeg'))); print('[install] ffprobe: ' + str(resolve_ffmpeg_binary('ffprobe'))); print('[install] RIFE model: ' + str(resolve_rife_model_path(None))); print('[install] opencv-python: ' + cv2.__version__); print('[install] ccvfi: ' + str(getattr(ccvfi, '__version__', 'unknown')))") -FailureMessage "Error: Python dependency/runtime check failed."
  }

  $tools = Check-NodeenvOnly -NodeenvDir $NodeenvDir -NodeVersion $NodeVersion
  $lockPath = Join-Path $Root "apps\interface\package-lock.json"
  if (-not (Test-Path -LiteralPath $lockPath)) {
    Fail "Error: missing frontend lockfile '$lockPath'."
  }
  $vitePath = Join-Path $Root "apps\interface\node_modules\vite\package.json"
  if (-not (Test-Path -LiteralPath $vitePath)) {
    Fail "Error: frontend dependency sentinel missing: apps\interface\node_modules\vite\package.json."
  }

  Write-Install "node:"
  & $tools.Node "-v"
  if ($LASTEXITCODE -ne 0) {
    Fail "Error: failed to read Node.js version from '$($tools.Node)'."
  }

  Write-Install "npm:"
  & $tools.Npm "-v"
  if ($LASTEXITCODE -ne 0) {
    Fail "Error: failed to read npm version from '$($tools.Npm)'."
  }
}

function Provision-VideoRuntime {
  param([Parameter(Mandatory = $true)][string]$VenvPython)

  if (-not (Test-Path -LiteralPath $VenvPython)) {
    Fail "Error: venv python not found at '$VenvPython' after uv sync."
  }

  Invoke-Checked -Executable $VenvPython -Arguments @("-c", "import os; from apps.backend.video.runtime_dependencies import ensure_ffmpeg_binaries; p=ensure_ffmpeg_binaries(version=os.environ.get('CODEX_FFMPEG_VERSION')); print('[install] ffmpeg: ' + str(p['ffmpeg'])); print('[install] ffprobe: ' + str(p['ffprobe']))") -FailureMessage "Error: failed to provision ffmpeg runtime dependencies."
  Invoke-Checked -Executable $VenvPython -Arguments @("-c", "from apps.backend.video.runtime_dependencies import ensure_rife_model_file; path=ensure_rife_model_file(); print('[install] RIFE model: ' + str(path))") -FailureMessage "Error: failed to provision default RIFE model checkpoint."
  Invoke-Checked -Executable $VenvPython -Arguments @("-c", "import cv2, ccvfi; print('[install] opencv-python: ' + cv2.__version__); print('[install] ccvfi: ' + str(getattr(ccvfi, '__version__', 'unknown')))") -FailureMessage "Error: failed to import video runtime dependencies (opencv-python/ccvfi)."
}

function Install-Frontend {
  param(
    [Parameter(Mandatory = $true)][string]$NodeenvNpm,
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$NpmCache
  )

  $interfaceDir = Join-Path $Root "apps\interface"
  if (-not (Test-Path -LiteralPath (Join-Path $interfaceDir "package-lock.json"))) {
    Fail "Error: lock-preserving frontend install requires '$interfaceDir\\package-lock.json'."
  }

  Write-Install "Installing frontend dependencies with npm ci [lock-preserving] ..."
  Push-Location $interfaceDir
  try {
    & $NodeenvNpm "ci" "--cache" $NpmCache "--no-audit" "--no-fund"
    if ($LASTEXITCODE -ne 0) {
      Fail "Error: npm ci failed."
    }
    if (-not (Test-Path -LiteralPath "node_modules\vite\package.json")) {
      Fail "Error: npm ci completed, but frontend deps are missing."
    }
  }
  finally {
    Pop-Location
  }
}

function Show-MenuPost {
  while ($true) {
    Write-Host ""
    Write-Host "=================================================="
    Write-Host " Installer finished."
    Write-Host "=================================================="
    Write-Host ""
    Write-Host " [1] Back to menu"
    Write-Host " [2] Exit"
    Write-Host ""
    $choice = Read-Host "Select an option (1-2)"
    if ($choice -eq "1") {
      return $true
    }
    if ($choice -eq "2") {
      return $false
    }
  }
}

function Show-AdvancedMenu {
  param([Parameter(Mandatory = $true)][int]$ReinstallLocked)

  while ($true) {
    Clear-Host
    Write-Host "=================================================="
    Write-Host " Advanced Install Options"
    Write-Host "=================================================="
    Write-Host ""
    Write-Host "  Choose a backend:"
    Write-Host "   [1] AUTO (default)"
    Write-Host "   [2] CPU"
    Write-Host "   [3] CUDA (pick 12.6/12.8/13)"
    Write-Host "   [4] SKIP torch install (not recommended)"
    Write-Host "   [5] CUSTOM PyTorch source (Windows only)"
    Write-Host "   [6] Back"
    Write-Host ""

    $choice = Read-Host "Select backend (1-6)"
    switch ($choice) {
      "6" { return $null }
      "1" {
        return @{ TorchMode = "auto"; TorchBackend = ""; CudaVariant = ""; CustomTorchSrc = ""; CustomTorchSha256 = ""; InstallCheck = 0; ReinstallDeps = $ReinstallLocked }
      }
      "2" {
        return @{ TorchMode = "cpu"; TorchBackend = ""; CudaVariant = ""; CustomTorchSrc = ""; CustomTorchSha256 = ""; InstallCheck = 0; ReinstallDeps = $ReinstallLocked }
      }
      "3" {
        while ($true) {
          Clear-Host
          Write-Host "=================================================="
          Write-Host " CUDA Variant"
          Write-Host "=================================================="
          Write-Host ""
          Write-Host "  Pick a CUDA wheel family (PyTorch):"
          Write-Host "   [1] CUDA 12.6  (cu126)"
          Write-Host "   [2] CUDA 12.8  (cu128)  [recommended]"
          Write-Host "   [3] CUDA 13    (cu130)"
          Write-Host "   [4] Back"
          Write-Host ""
          $cudaChoice = Read-Host "Select CUDA (1-4)"
          if ($cudaChoice -eq "4") {
            break
          }
          if ($cudaChoice -eq "1") {
            return @{ TorchMode = "cuda"; TorchBackend = ""; CudaVariant = "12.6"; CustomTorchSrc = ""; CustomTorchSha256 = ""; InstallCheck = 0; ReinstallDeps = $ReinstallLocked }
          }
          if ($cudaChoice -eq "2") {
            return @{ TorchMode = "cuda"; TorchBackend = ""; CudaVariant = "12.8"; CustomTorchSrc = ""; CustomTorchSha256 = ""; InstallCheck = 0; ReinstallDeps = $ReinstallLocked }
          }
          if ($cudaChoice -eq "3") {
            return @{ TorchMode = "cuda"; TorchBackend = ""; CudaVariant = "13"; CustomTorchSrc = ""; CustomTorchSha256 = ""; InstallCheck = 0; ReinstallDeps = $ReinstallLocked }
          }
        }
      }
      "4" {
        return @{ TorchMode = "skip"; TorchBackend = ""; CudaVariant = ""; CustomTorchSrc = ""; CustomTorchSha256 = ""; InstallCheck = 0; ReinstallDeps = $ReinstallLocked }
      }
      "5" {
        try {
          $picked = Select-CustomTorchFromManifest -ManifestUrl $script:Config.PyTorchManifestUrl
          if ($null -eq $picked) {
            continue
          }
          return @{ TorchMode = "custom"; TorchBackend = ""; CudaVariant = ""; CustomTorchSrc = $picked.Url; CustomTorchSha256 = $picked.Sha256; InstallCheck = 0; ReinstallDeps = $ReinstallLocked }
        }
        catch {
          Write-Host ""
          Write-Host ("Error: custom PyTorch selection failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
          [void](Read-Host "Press Enter to return to Advanced menu")
        }
      }
      default { }
    }
  }
}

function Show-MainMenu {
  while ($true) {
    Clear-Host
    Write-Host "=================================================="
    Write-Host " Codex WebUI Installer (Windows)"
    Write-Host "=================================================="
    Write-Host ""
    Write-Host " [1] Simple install (AUTO)"
    Write-Host " [2] Advanced (choose backend / CUDA / custom torch)"
    Write-Host " [3] Check installed dependencies"
    Write-Host " [4] Reinstall dependencies (in-place)"
    Write-Host " [5] Exit"
    Write-Host ""

    $choice = Read-Host "Select an option (1-5)"
    switch ($choice) {
      "1" {
        return @{ TorchMode = "auto"; TorchBackend = ""; CudaVariant = ""; CustomTorchSrc = ""; CustomTorchSha256 = ""; InstallCheck = 0; ReinstallDeps = 0; Cancel = $false }
      }
      "2" {
        $advanced = Show-AdvancedMenu -ReinstallLocked 0
        if ($null -ne $advanced) {
          $advanced.Cancel = $false
          return $advanced
        }
      }
      "3" {
        return @{ TorchMode = "auto"; TorchBackend = ""; CudaVariant = ""; CustomTorchSrc = ""; CustomTorchSha256 = ""; InstallCheck = 1; ReinstallDeps = 0; Cancel = $false }
      }
      "4" {
        $advancedWithLock = Show-AdvancedMenu -ReinstallLocked 1
        if ($null -ne $advancedWithLock) {
          $advancedWithLock.Cancel = $false
          return $advancedWithLock
        }
      }
      "5" {
        return @{ Cancel = $true }
      }
      default { }
    }
  }
}

$root = Get-Root
$env:CODEX_ROOT = $root
Set-Or-PrependPythonPath -Root $root

while ($true) {
  $script:Config = [ordered]@{
    UvVersion = Get-EnvOrDefault -Name "CODEX_UV_VERSION" -Default "0.9.17"
    PythonVersion = Get-EnvOrDefault -Name "CODEX_PYTHON_VERSION" -Default "3.12.10"
    NodeVersion = Get-EnvOrDefault -Name "CODEX_NODE_VERSION" -Default "24.15.0"
    FFmpegVersion = Get-EnvOrDefault -Name "CODEX_FFMPEG_VERSION" -Default "7.0.2"
    TorchMode = Get-EnvOrDefault -Name "CODEX_TORCH_MODE" -Default "auto"
    TorchBackend = Get-EnvValue -Name "CODEX_TORCH_BACKEND"
    CudaVariant = Get-EnvValue -Name "CODEX_CUDA_VARIANT"
    CustomTorchSrc = Get-EnvValue -Name "CODEX_CUSTOM_TORCH_SRC"
    CustomTorchSha256 = Get-EnvValue -Name "CODEX_CUSTOM_TORCH_SHA256"
    PyTorchManifestUrl = Get-EnvOrDefault -Name "CODEX_PYTORCH_MANIFEST_URL" -Default "https://raw.githubusercontent.com/sangoi-exe/stable-diffusion-webui-codex/main/pytorch_manifest.json"
    InstallCheck = Parse-BoolFromEnv -Name "CODEX_INSTALL_CHECK" -Default 0
    ReinstallDeps = Parse-BoolFromEnv -Name "CODEX_REINSTALL_DEPS" -Default 0
    HelpOnly = $false
    ForceMenu = $false
    ArgNoMenu = $false
    MenuUsed = $false
  }

  foreach ($arg in $CliArgs) {
    $lower = $arg.ToLowerInvariant()
    switch ($lower) {
      "--no-menu" {
        $script:Config.ArgNoMenu = $true
        continue
      }
      "--menu" {
        $script:Config.ForceMenu = $true
        continue
      }
      "--simple" {
        $script:Config.ArgNoMenu = $true
        $script:Config.TorchMode = "auto"
        $script:Config.TorchBackend = ""
        $script:Config.CudaVariant = ""
        $script:Config.CustomTorchSrc = ""
        $script:Config.CustomTorchSha256 = ""
        $script:Config.InstallCheck = 0
        $script:Config.ReinstallDeps = 0
        continue
      }
      "--check" {
        $script:Config.ArgNoMenu = $true
        $script:Config.TorchMode = "auto"
        $script:Config.TorchBackend = ""
        $script:Config.CudaVariant = ""
        $script:Config.CustomTorchSrc = ""
        $script:Config.CustomTorchSha256 = ""
        $script:Config.InstallCheck = 1
        $script:Config.ReinstallDeps = 0
        continue
      }
      "--reinstall-deps" {
        $script:Config.ArgNoMenu = $true
        $script:Config.InstallCheck = 0
        $script:Config.ReinstallDeps = 1
        continue
      }
      "--help" {
        $script:Config.HelpOnly = $true
        continue
      }
      "-h" {
        $script:Config.HelpOnly = $true
        continue
      }
      "/?" {
        $script:Config.HelpOnly = $true
        continue
      }
      default {
        [Console]::Error.WriteLine("Error: unknown argument '$arg'.")
        Show-Usage
        exit 1
      }
    }
  }

  if ($script:Config.HelpOnly) {
    Show-Usage
    exit 0
  }

  $allowedModes = @("auto", "cpu", "cuda", "rocm", "skip", "custom")
  if ($allowedModes -notcontains $script:Config.TorchMode.Trim().ToLowerInvariant()) {
    Fail "Error: invalid CODEX_TORCH_MODE='$($script:Config.TorchMode)' (expected auto|cpu|cuda|rocm|skip|custom)."
  }

  $allowedBackends = @("cpu", "cu126", "cu128", "cu130", "rocm64")
  if (-not [string]::IsNullOrWhiteSpace($script:Config.TorchBackend)) {
    if ($allowedBackends -notcontains $script:Config.TorchBackend.Trim().ToLowerInvariant()) {
      Fail "Error: invalid CODEX_TORCH_BACKEND='$($script:Config.TorchBackend)' (expected cpu|cu126|cu128|cu130|rocm64)."
    }
  }

  $allowedCuda = @("12.6", "12.8", "13", "cu126", "cu128", "cu130")
  if (-not [string]::IsNullOrWhiteSpace($script:Config.CudaVariant)) {
    if ($allowedCuda -notcontains $script:Config.CudaVariant.Trim().ToLowerInvariant()) {
      Fail "Error: invalid CODEX_CUDA_VARIANT='$($script:Config.CudaVariant)' (expected 12.6|12.8|13|cu126|cu128|cu130)."
    }
  }

  if ($script:Config.TorchMode.Trim().ToLowerInvariant() -eq "custom") {
    if (-not [string]::IsNullOrWhiteSpace($script:Config.TorchBackend)) {
      Fail "Error: CODEX_TORCH_MODE=custom cannot be combined with CODEX_TORCH_BACKEND."
    }
    if (-not [string]::IsNullOrWhiteSpace($script:Config.CudaVariant)) {
      Fail "Error: CODEX_TORCH_MODE=custom cannot be combined with CODEX_CUDA_VARIANT."
    }
  }

  if ($script:Config.InstallCheck -eq 1 -and $script:Config.ReinstallDeps -eq 1) {
    Fail "Error: --check and --reinstall-deps are mutually exclusive."
  }

  $hasMenuSuppressor = $script:Config.ArgNoMenu -or (Is-EnvDefined -Name "CODEX_NO_MENU") -or (Is-EnvDefined -Name "CI") -or (Is-EnvDefined -Name "GITHUB_ACTIONS")

  $showMenu = $true
  if ($hasMenuSuppressor) {
    $showMenu = $false
  }

  if ($script:Config.ForceMenu) {
    $showMenu = $true
    if ($hasMenuSuppressor) {
      $showMenu = $false
    }
  }

  if (-not $script:Config.ForceMenu) {
    $explicitEnvVars = @(
      "CODEX_TORCH_MODE",
      "CODEX_TORCH_BACKEND",
      "CODEX_CUDA_VARIANT",
      "CODEX_CUSTOM_TORCH_SRC",
      "CODEX_INSTALL_CHECK",
      "CODEX_REINSTALL_DEPS",
      "CODEX_PYTHON_VERSION",
      "CODEX_UV_VERSION"
    )
    foreach ($name in $explicitEnvVars) {
      if (Is-EnvDefined -Name $name) {
        $showMenu = $false
        break
      }
    }
  }

  if ($showMenu) {
    $script:Config.MenuUsed = $true
    $menuResult = Show-MainMenu
    if ($menuResult.Cancel) {
      exit 0
    }
    $script:Config.TorchMode = $menuResult.TorchMode
    $script:Config.TorchBackend = $menuResult.TorchBackend
    $script:Config.CudaVariant = $menuResult.CudaVariant
    $script:Config.CustomTorchSrc = $menuResult.CustomTorchSrc
    $script:Config.CustomTorchSha256 = $menuResult.CustomTorchSha256
    $script:Config.InstallCheck = [int]$menuResult.InstallCheck
    $script:Config.ReinstallDeps = [int]$menuResult.ReinstallDeps
  }

  $uvDir = Join-Path $root ".uv\bin"
  $uvBin = Join-Path $uvDir "uv.exe"
  $venv = Join-Path $root ".venv"
  $venvPython = Join-Path $venv "Scripts\python.exe"
  $nodeenv = Join-Path $root ".nodeenv"
  $uvCacheDir = Join-Path $root ".uv\cache"
  $npmCache = Join-Path $root ".npm-cache"
  $xdgDataHome = Join-Path $root ".uv\xdg-data"
  $xdgCacheHome = Join-Path $root ".uv\xdg-cache"

  if ($script:Config.InstallCheck -ne 1) {
    Ensure-Dir -Path $uvCacheDir
    Ensure-Dir -Path $npmCache
    Ensure-Dir -Path $xdgDataHome
    Ensure-Dir -Path $xdgCacheHome
  }

  $env:UV_CACHE_DIR = $uvCacheDir
  $env:NPM_CONFIG_CACHE = $npmCache
  $env:XDG_DATA_HOME = $xdgDataHome
  $env:XDG_CACHE_HOME = $xdgCacheHome
  $env:CODEX_FFMPEG_VERSION = $script:Config.FFmpegVersion

  Write-Install "Repo: $root"
  Write-Install "uv: $uvBin  version pin: $($script:Config.UvVersion)"
  Write-Install "uv cache: $uvCacheDir"
  Write-Install "Python: $($script:Config.PythonVersion)  managed by uv"
  Write-Install "Venv: $venv  created by uv; uses the managed Python"
  Write-Install "Node.js: $($script:Config.NodeVersion)  managed by nodeenv  (installs into $nodeenv)"
  Write-Install "FFmpeg runtime version: $($script:Config.FFmpegVersion)  managed by ffmpeg-downloader"
  Write-Install "Torch mode: $($script:Config.TorchMode)  CODEX_TORCH_MODE=auto|cpu|cuda|rocm|skip|custom"
  if (-not [string]::IsNullOrWhiteSpace($script:Config.TorchBackend)) {
    Write-Install "Torch backend override: $($script:Config.TorchBackend)  CODEX_TORCH_BACKEND"
  }
  if (-not [string]::IsNullOrWhiteSpace($script:Config.CudaVariant)) {
    Write-Install "CUDA variant override: $($script:Config.CudaVariant)  CODEX_CUDA_VARIANT"
  }
  if ($script:Config.TorchMode.Trim().ToLowerInvariant() -eq "custom") {
    Write-Install "Custom torch source: $($script:Config.CustomTorchSrc)"
  }
  Write-Install "PyTorch manifest URL: $($script:Config.PyTorchManifestUrl)  CODEX_PYTORCH_MANIFEST_URL"
  Write-Install "Dependency check mode: $($script:Config.InstallCheck)  CODEX_INSTALL_CHECK"
  Write-Install "Reinstall dependencies: $($script:Config.ReinstallDeps)  CODEX_REINSTALL_DEPS"
  Write-Install "npm cache: $npmCache"

  if ($script:Config.InstallCheck -eq 1) {
    Run-DependencyChecks -UvBin $uvBin -VenvPython $venvPython -TorchMode $script:Config.TorchMode -NodeenvDir $nodeenv -NodeVersion $script:Config.NodeVersion -Root $root
    Write-Host ""
    Write-Install "Done (check-only)."

    if ($script:Config.MenuUsed) {
      $rerun = Show-MenuPost
      if ($rerun) {
        continue
      }
    }
    exit 0
  }

  $uvInfo = Ensure-Uv -Root $root -UvVersion $script:Config.UvVersion
  $uvBin = $uvInfo.UvBin

  $env:UV_PYTHON_INSTALL_DIR = Join-Path $root ".uv\python"
  $env:UV_PYTHON_INSTALL_BIN = "0"
  $env:UV_PYTHON_INSTALL_REGISTRY = "0"
  $env:UV_PYTHON_PREFERENCE = "only-managed"
  $env:UV_PYTHON_DOWNLOADS = "manual"
  $env:UV_PROJECT_ENVIRONMENT = $venv

  Write-Install "Installing managed Python $($script:Config.PythonVersion) ..."
  Invoke-Checked -Executable $uvBin -Arguments @("python", "install", $script:Config.PythonVersion) -FailureMessage "Error: failed to install Python $($script:Config.PythonVersion) via uv."

  $torchExtra = Resolve-TorchExtra -TorchMode $script:Config.TorchMode -TorchBackend $script:Config.TorchBackend -CudaVariant $script:Config.CudaVariant
  $syncArgs = @("sync", "--locked")
  if ($script:Config.ReinstallDeps -eq 1) {
    $syncArgs += "--reinstall"
  }

  $customSourceForInstall = ""
  if ($script:Config.TorchMode.Trim().ToLowerInvariant() -eq "custom") {
    $placeholder = "https://placeholder.invalid/path/to/custom-torch.whl"
    if ([string]::IsNullOrWhiteSpace($script:Config.CustomTorchSrc)) {
      $script:Config.CustomTorchSrc = $placeholder
    }
    if ($script:Config.CustomTorchSrc.Trim().ToLowerInvariant() -eq $placeholder) {
      Fail "Error: CODEX_TORCH_MODE=custom requires CODEX_CUSTOM_TORCH_SRC to point to a real wheel/source."
    }

    Write-Install "Syncing Python dependencies [locked] without bundled torch extras ..."
    Invoke-Checked -Executable $uvBin -Arguments $syncArgs -FailureMessage "Error: uv sync failed."

    $customSourceForInstall = Resolve-CustomSourceForInstall -CustomSource $script:Config.CustomTorchSrc -CustomSha256 $script:Config.CustomTorchSha256
    Write-Install "Installing custom PyTorch from source: $($script:Config.CustomTorchSrc)"
    Invoke-Checked -Executable $uvBin -Arguments @("pip", "install", "--python", $venvPython, "--force-reinstall", "--no-deps", $customSourceForInstall) -FailureMessage "Error: custom PyTorch install failed."
  }
  elseif ([string]::IsNullOrWhiteSpace($torchExtra)) {
    Write-InstallWarning "skipping torch/torchvision install. CODEX_TORCH_MODE=skip. WebUI requires PyTorch."
    Write-Install "Syncing Python dependencies [locked] ..."
    Invoke-Checked -Executable $uvBin -Arguments $syncArgs -FailureMessage "Error: uv sync failed."
  }
  else {
    Write-Install "Syncing Python dependencies [locked] with torch extra: $torchExtra ..."
    $withExtra = @()
    $withExtra += $syncArgs
    $withExtra += @("--extra", $torchExtra)
    Invoke-Checked -Executable $uvBin -Arguments $withExtra -FailureMessage "Error: uv sync failed."
  }

  Provision-VideoRuntime -VenvPython $venvPython
  $nodeTools = Ensure-Nodeenv -UvBin $uvBin -NodeVersion $script:Config.NodeVersion -NodeenvDir $nodeenv

  Write-Install "node:"
  & $nodeTools.Node "-v"
  if ($LASTEXITCODE -ne 0) {
    Fail "Error: failed to read Node.js version from '$($nodeTools.Node)'."
  }

  Write-Install "npm:"
  & $nodeTools.Npm "-v"
  if ($LASTEXITCODE -ne 0) {
    Fail "Error: failed to read npm version from '$($nodeTools.Npm)'."
  }

  Install-Frontend -NodeenvNpm $nodeTools.Npm -Root $root -NpmCache $npmCache

  Write-Host ""
  Write-Install "Done."
  Write-Install "Next: run-webui.bat"

  if ($script:Config.MenuUsed) {
    $rerunInstall = Show-MenuPost
    if ($rerunInstall) {
      continue
    }
  }

  exit 0
}
