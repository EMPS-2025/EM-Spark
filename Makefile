SHELL := bash
.ONESHELL:
ENV_FILE := .env

ifndef VERBOSE
.SILENT:
endif

help:
	@echo "Targets:"
	@echo "  setup             - create venv & install deps"
	@echo "  schema            - apply DB schema (001_init.sql)"
	@echo "  load CSV=path     - load CSV into staging (TEXT), convert & upsert to final"
	@echo "  check             - basic verification queries"
	@echo "  run               - start Chainlit app"
	@echo "  clean_bad         - delete any rows before 2010 (safety)"
	@echo "  truncate_stage    - clear staging tables"

setup:
	python -m venv .venv
	source .venv/bin/activate && pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "Edit .env with your Supabase DATABASE_URL (sslmode=require)."

schema:
	set -o allexport; source $(ENV_FILE); set +o allexport; \
	psql "$$DATABASE_URL" -f sql/001_init.sql

truncate_stage:
	set -o allexport; source $(ENV_FILE); set +o allexport; \
	psql "$$DATABASE_URL" -c "TRUNCATE stage_prices_text; TRUNCATE stage_prices;"

load:
	@if [ -z "$(CSV)" ]; then echo "Usage: make load CSV=path/to.csv"; exit 1; fi
	set -o allexport; source $(ENV_FILE); set +o allexport; \
	psql "$$DATABASE_URL" -f sql/002_stage.sql
	psql "$$DATABASE_URL" -c "\\copy stage_prices_text(market,delivery_date,block_index,duration_min,area,price_rs_per_mwh,source_file) FROM '$(CSV)' CSV HEADER"
	psql "$$DATABASE_URL" -f sql/003_convert_upsert.sql

check:
	set -o allexport; source $(ENV_FILE); set +o allexport; \
	psql "$$DATABASE_URL" -c "SELECT MIN(delivery_date) AS min_date, MAX(delivery_date) AS max_date FROM price_points;"
	set -o allexport; source $(ENV_FILE); set +o allexport; \
	psql "$$DATABASE_URL" -c "SELECT m.code, pp.delivery_date, COUNT(*) AS rows FROM price_points pp JOIN markets m ON m.id=pp.market_id GROUP BY 1,2 ORDER BY 2,1 LIMIT 20;"

clean_bad:
	set -o allexport; source $(ENV_FILE); set +o allexport; \
	psql "$$DATABASE_URL" -c "DELETE FROM price_points WHERE delivery_date < DATE '2010-01-01';"

run:
	source .venv/bin/activate && chainlit run app/app.py -w
