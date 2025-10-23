param([string]$EnvPath = ".env")
if (-not (Test-Path $EnvPath)) { throw ".env not found at $EnvPath" }
Get-Content $EnvPath | ForEach-Object {
  if ($_ -match '^\s*([^#=]+)\s*=\s*(.+)\s*$') {
    $name  = $matches[1].Trim()
    $value = $matches[2].Trim()
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
  }
}
if (-not $env:DATABASE_URL) { throw "DATABASE_URL not loaded from .env" }
