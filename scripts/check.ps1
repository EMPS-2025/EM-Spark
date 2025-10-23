param()
. "$PSScriptRoot\_load-env.ps1"
psql "$env:DATABASE_URL" -c "SELECT MIN(delivery_date) AS min_date, MAX(delivery_date) AS max_date FROM price_points;"
psql "$env:DATABASE_URL" -c "SELECT m.code, pp.delivery_date, COUNT(*) AS rows FROM price_points pp JOIN markets m ON m.id=pp.market_id GROUP BY 1,2 ORDER BY 2,1 LIMIT 20;"
