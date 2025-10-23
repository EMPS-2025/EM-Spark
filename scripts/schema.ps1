param()
. "$PSScriptRoot\_load-env.ps1"
psql "$env:DATABASE_URL" -f "sql\001_init.sql"
