<# 
 Start the local stack and open the Streamlit UI.
 - Clears any stale docker-compose lock for this project.
 - Brings the services up in detached mode.
 - Opens the UI at http://localhost:8501 in your default browser.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to repo root (one level up from this script)
Push-Location (Join-Path $PSScriptRoot "..")

try {
    Write-Host "Stopping any running stack (clears stale locks)..." -ForegroundColor Cyan
    docker compose down | Out-Null
} catch {
    Write-Warning "docker compose down failed (maybe nothing was running): $_"
}

Write-Host "Starting stack in detached mode..." -ForegroundColor Cyan
docker compose up -d

Write-Host "Opening Streamlit UI at http://localhost:8501" -ForegroundColor Cyan
Start-Process "http://localhost:8501"

Write-Host "Done. Use 'docker compose logs -f' to tail logs, or 'docker compose down' to stop." -ForegroundColor Green

Pop-Location


