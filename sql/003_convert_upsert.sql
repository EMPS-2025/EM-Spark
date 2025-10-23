TRUNCATE stage_prices;
INSERT INTO stage_prices
(market, delivery_date, block_index, duration_min, area, price_rs_per_mwh, source_file)
SELECT
  market,
  TO_DATE(delivery_date, 'YYYY-MM-DD'),
  block_index,
  duration_min,
  area,
  price_rs_per_mwh,
  source_file
FROM stage_prices_text
WHERE delivery_date ~ '^\d{4}-\d{2}-\d{2}$';
WITH s AS (
  SELECT
    market, area, delivery_date, block_index,
    MAX(duration_min)     AS duration_min,
    MAX(price_rs_per_mwh) AS price_rs_per_mwh,
    MAX(source_file)      AS source_file
  FROM stage_prices
  WHERE delivery_date >= DATE '2010-01-01'
  GROUP BY market, area, delivery_date, block_index
),
lk AS (
  SELECT
    m.id  AS market_id,
    a.id  AS area_id,
    tb.id AS block_id,
    s.delivery_date,
    COALESCE(s.duration_min, tb.duration_min) AS duration_min,
    s.price_rs_per_mwh,
    s.source_file
  FROM s
  JOIN markets m      ON m.code = s.market
  JOIN areas   a      ON a.code = s.area
  JOIN time_blocks tb ON tb.block_index = s.block_index
),
upsert AS (
  INSERT INTO price_points (
    market_id, area_id, delivery_date, block_id, duration_min, price_rs_per_mwh, source_file
  )
  SELECT market_id, area_id, delivery_date, block_id, duration_min, price_rs_per_mwh, source_file
  FROM lk
  ON CONFLICT (market_id, area_id, delivery_date, block_id)
  DO UPDATE SET
    price_rs_per_mwh = EXCLUDED.price_rs_per_mwh,
    duration_min      = EXCLUDED.duration_min,
    source_file       = EXCLUDED.source_file
  RETURNING xmax = 0 AS inserted
)
SELECT
  COUNT(*) AS affected_rows,
  COUNT(*) FILTER (WHERE inserted) AS inserted_rows,
  COUNT(*) FILTER (WHERE NOT inserted) AS updated_rows;
