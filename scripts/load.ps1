param([Parameter(Mandatory=$true)][string]$CsvPath)
. "$PSScriptRoot\_load-env.ps1"
psql "$env:DATABASE_URL" -f "sql\002_stage.sql"
$abs = (Resolve-Path $CsvPath).Path
$absUnix = $abs -replace '\\','/'
$copy = "\\copy stage_prices_text(market,delivery_date,block_index,duration_min,area,price_rs_per_mwh,source_file) FROM '$absUnix' CSV HEADER"
psql "$env:DATABASE_URL" -c $copy
psql "$env:DATABASE_URL" -f "sql\003_convert_upsert.sql"
