CREATE TABLE IF NOT EXISTS stage_prices_text (
  market TEXT,
  delivery_date TEXT,
  block_index INT,
  duration_min INT,
  area TEXT,
  price_rs_per_mwh NUMERIC(10,5),
  source_file TEXT
);
TRUNCATE stage_prices_text;
CREATE TABLE IF NOT EXISTS stage_prices (
  market TEXT,
  delivery_date DATE,
  block_index INT,
  duration_min INT,
  area TEXT,
  price_rs_per_mwh NUMERIC(10,5),
  source_file TEXT
);
TRUNCATE stage_prices;
