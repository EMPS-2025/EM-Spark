param()
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "Created .env. Edit it with your DATABASE_URL"; } else { Write-Host ".env already exists." }
