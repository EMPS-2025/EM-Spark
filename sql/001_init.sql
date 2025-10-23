CREATE TABLE IF NOT EXISTS markets (id SERIAL PRIMARY KEY, code TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS areas   (id SERIAL PRIMARY KEY, code TEXT UNIQUE NOT NULL, name TEXT);
CREATE TABLE IF NOT EXISTS time_blocks (
  id SERIAL PRIMARY KEY,
  block_index INT UNIQUE NOT NULL,
  start_time TIME,
  duration_min INT NOT NULL
);
CREATE TABLE IF NOT EXISTS price_points (
  id BIGSERIAL PRIMARY KEY,
  market_id INT NOT NULL REFERENCES markets(id),
  area_id   INT NOT NULL REFERENCES areas(id),
  delivery_date DATE NOT NULL,
  block_id  INT NOT NULL REFERENCES time_blocks(id),
  duration_min INT NOT NULL DEFAULT 60,
  price_rs_per_mwh NUMERIC(10,2) NOT NULL,
  source_file TEXT,
  ingested_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (market_id, area_id, delivery_date, block_id)
);
INSERT INTO markets(code) VALUES ('DAM'), ('GDAM') ON CONFLICT (code) DO NOTHING;
INSERT INTO areas(code, name) VALUES ('ALL','India MCP') ON CONFLICT (code) DO NOTHING;
INSERT INTO time_blocks(block_index, start_time, duration_min)
SELECT g, (make_time(0,0,0) + ((g-1)*interval '60 minutes'))::time, 60
FROM generate_series(1,24) g
ON CONFLICT (block_index) DO NOTHING;
CREATE INDEX IF NOT EXISTS price_points_date_idx ON price_points(delivery_date);
ALTER TABLE price_points
  ADD CONSTRAINT IF NOT EXISTS chk_delivery_date_reasonable
  CHECK (delivery_date >= DATE '2010-01-01');
