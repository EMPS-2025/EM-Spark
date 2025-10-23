import os, re, calendar
from typing import List, Tuple, Optional, Dict, Iterable
import chainlit as cl
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import dateparser
from dateparser.search import search_dates
from datetime import date, datetime, timedelta
import time
import asyncio
import traceback

# ─────────────────────────────────────────────────────────────
# Environment & DB
# ─────────────────────────────────────────────────────────────
load_dotenv(override=True)

# Preferred: set DATABASE_URL in Railway Variables (and locally in .env)
DB_URL = os.getenv("DATABASE_URL", "").strip()
# Optional discrete vars if you ever need them (ignored when DB_URL is set)
DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "").strip()
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
# Most managed Postgres want SSL. Use "prefer" only if your DB doesn’t support SSL.
DB_SSLMODE = os.getenv("DB_SSLMODE", "require").strip()

def _connect():
    """
    Create a short-lived connection for each query.
    Keepalives help across NAT/idle; avoids stale global handles on Railway.
    """
    keepalive = dict(
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    if DB_URL:
        return psycopg2.connect(DB_URL, sslmode=DB_SSLMODE, **keepalive)

    # Fallback if you don’t have a URL
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE,
        **keepalive,
    )

Y_MIN = 2010  # guard against accidental 1970-era parsing etc.

# ─────────────────────────────────────────────────────────────
# Progress helpers (single-line, then hidden)
# ─────────────────────────────────────────────────────────────
async def progress_start(text: str = "💭 Thinking …") -> cl.Message:
    m = cl.Message(content=text)
    await m.send()
    return m

async def progress_update(m: cl.Message, text: str) -> None:
    try:
        await m.update(content=text)
    except Exception:
        pass

async def progress_hide(m: cl.Message) -> None:
    try:
        await m.remove()
    except Exception:
        try:
            await m.update(content="")
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────
# Disclaimer at chat start
# ─────────────────────────────────────────────────────────────
@cl.on_chat_start
async def chat_start():
    disclaimer = (
        "### Disclaimer\n"
        "- **Start date for market data:** 03/08/2022 to 08/08/2025\n"
        "- **Presently data available for:** Hourly slots\n"
        "- **Data sourced from:** IEX\n"
        "- **Status:** Platform still in development phase\n"
    )
    await cl.Message(content=disclaimer).send()

# ─────────────────────────────────────────────────────────────
# Help text used when parsing fails
# ─────────────────────────────────────────────────────────────
BLOCK_HELP = """Blocks (hourly): 1=00–01, 2=01–02, …, 24=23–24.
Examples:
• '3-8' → blocks 3..8
• '2-6 and 12-20' → blocks 2..6 plus 12..20
• '0-8 hours' is accepted and mapped to blocks 1..8
• 'full day' or 'all 24' → blocks 1..24
"""

EXAMPLE_QUERIES = """Examples:
• "DAM today"
• "GDAM yesterday 0–8 hours"
• "DAM 26 Aug 2024 3–8"
• "DAM Aug 2024" (full month)
• "DAM 2023" (full year)
• "GDAM 12–20 and 2–6 on 12 Sep 2023"
"""

# ─────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────
def parse_market(text: str) -> Optional[str]:
    if re.search(r"\bGDAM\b", text, re.IGNORECASE):
        return "GDAM"
    if re.search(r"\bDAM\b", text, re.IGNORECASE):
        return "DAM"
    return None

def _safe_date(d: datetime) -> Optional[date]:
    if d and d.year >= Y_MIN:
        return d.date()
    return None

def _has_month_word(s: str) -> bool:
    return bool(re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
        r"anuary|ebruary|arch|ril|une|uly|ugust|eptember|"
        r"october|november|december)", s, re.IGNORECASE))

def parse_date_or_range(text: str) -> Tuple[Optional[date], Optional[date]]:
    """Return (start_date, end_date). If a single date, start==end."""
    cleaned = re.sub(r"\b\d{1,2}\s*[-–]\s*\d{1,2}\b", " ", text)  # remove hour ranges before date parsing
    cleaned = cleaned.strip()
    lower = cleaned.lower()
    today = date.today()

    if "yesterday" in lower:
        d = today - timedelta(days=1)
        return (d, d)
    if "today" in lower:
        return (today, today)
    if "this month" in lower:
        start = date(today.year, today.month, 1)
        end = date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
        return (start, end)
    if "last month" in lower:
        y, m = today.year, today.month - 1
        if m == 0:
            y -= 1
            m = 12
        start = date(y, m, 1)
        end = date(y, m, calendar.monthrange(y, m)[1])
        return (start, end)
    if "this year" in lower:
        return (date(today.year, 1, 1), date(today.year, 12, 31))

    matches = search_dates(cleaned, settings={"DATE_ORDER": "DMY"})
    if matches:
        for s, dt in matches:
            if dt.year >= Y_MIN:
                # Month-only phrase → whole month
                if _has_month_word(s) and not re.search(r"\b([12]\d|3[01])\b", s):
                    start = date(dt.year, dt.month, 1)
                    end = date(dt.year, dt.month, calendar.monthrange(dt.year, dt.month)[1])
                    return (start, end)
                # Year-only → whole year
                if re.fullmatch(r"\s*20\d{2}\s*", s) or "year" in lower or "full year" in lower:
                    y = dt.year
                    return (date(y, 1, 1), date(y, 12, 31))
                # Single date
                d = _safe_date(dt)
                if d:
                    return (d, d)

    dt = dateparser.parse(cleaned, settings={"DATE_ORDER": "DMY"})
    if dt and dt.year >= Y_MIN:
        return (dt.date(), dt.date())
    return (None, None)

def parse_ranges(text: str) -> List[Tuple[int, int]]:
    """
    Accepts:
      - '3-8'
      - '2-6 and 12-20'
      - '0-8 hours' (maps 0->1)
      - 'full day' / 'all 24'
    """
    lower = text.lower()
    if "full day" in lower or "all 24" in lower or "entire day" in lower:
        return [(1, 24)]
    pieces = re.split(r"\s*(?:,|and)\s*", text, flags=re.IGNORECASE)
    out: List[Tuple[int, int]] = []
    for tok in pieces:
        m = re.search(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            a = max(1, min(24, 1 if a == 0 else a))
            b = max(1, min(24, 1 if b == 0 else b))
            lo, hi = min(a, b), max(a, b)
            out.append((lo, hi))
    return out

def expand_ranges(ranges: List[Tuple[int, int]]) -> List[int]:
    seq = []
    for lo, hi in ranges:
        seq.extend(range(lo, hi + 1))
    seen, out = set(), []
    for b in seq:
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out

# ─────────────────────────────────────────────────────────────
# DB Fetch & Stats
# ─────────────────────────────────────────────────────────────
def fetch_prices_day(market: str, d: date, blocks: List[int]) -> Dict[int, Dict[str, float]]:
    if not blocks:
        return {}
    q = """
      SELECT tb.block_index, pp.price_rs_per_mwh, pp.duration_min
      FROM price_points pp
      JOIN markets m ON m.id = pp.market_id
      JOIN time_blocks tb ON tb.id = pp.block_id
      WHERE m.code = %s
        AND pp.delivery_date = %s
        AND tb.block_index = ANY(%s)
      ORDER BY tb.block_index
    """
    # Fresh connection per query → no stale handles after Railway idle/wake
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(q, (market, d, blocks))
            rows = cur.fetchall()
    return {
        int(r["block_index"]): {
            "price": float(r["price_rs_per_mwh"]),   # DB value; UI labels as kWh (cosmetic)
            "minutes": int(r["duration_min"]),
        }
        for r in rows
    }

def fetch_prices_range(market: str, start: date, end: date, blocks: List[int]):
    """Return dict: { date -> { block_index -> {price, minutes} } }"""
    if not blocks:
        return {}
    q = """
      SELECT pp.delivery_date, tb.block_index, pp.price_rs_per_mwh, pp.duration_min
      FROM price_points pp
      JOIN markets m ON m.id = pp.market_id
      JOIN time_blocks tb ON tb.id = pp.block_id
      WHERE m.code = %s
        AND pp.delivery_date BETWEEN %s AND %s
        AND tb.block_index = ANY(%s)
      ORDER BY pp.delivery_date, tb.block_index
    """
    out: Dict[date, Dict[int, Dict[str, float]]] = {}
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(q, (market, start, end, blocks))
            for r in cur.fetchall():
                d = r["delivery_date"]
                out.setdefault(d, {})
                out[d][int(r["block_index"])] = {
                    "price": float(r["price_rs_per_mwh"]),
                    "minutes": int(r["duration_min"]),
                }
    return out

def _stats_over_items(items: Iterable[Dict[str, float]]):
    items = list(items)
    if not items:
        return {"count": 0, "twap": None, "sum_1mw": 0.0, "min": None, "max": None}
    # Weighted by minutes
    num = sum(it["price"] * it["minutes"] for it in items)
    den = sum(it["minutes"] for it in items)
    twap = (num / den) if den else None
    # Sum (1 MW): price (₹/MWh) * hours  => ₹
    s1 = sum(it["price"] * (it["minutes"] / 60.0) for it in items)
    prices = [it["price"] for it in items]
    return {"count": len(items), "twap": twap, "sum_1mw": s1,
            "min": min(prices), "max": max(prices)}

def compute_stats(blocks: List[int], data: Dict[int, Dict[str, float]]):
    found = [b for b in blocks if b in data]
    missing = sorted(set(blocks) - set(found))
    base = _stats_over_items([data[b] for b in found])
    base.update({"found": found, "missing": missing})
    return base

def compute_stats_over_range(per_day: Dict[date, Dict[int, Dict[str, float]]], blocks: List[int]):
    # Combined across all days/blocks
    all_items = []
    for d in sorted(per_day.keys()):
        day_map = per_day[d]
        for b in blocks:
            if b in day_map:
                all_items.append(day_map[b])
    combined = _stats_over_items(all_items)

    # Daily TWAPs (kept for completeness; not rendered while table UI is off)
    daily_rows = []
    for d in sorted(per_day.keys()):
        day_map = per_day[d]
        items = [day_map[b] for b in blocks if b in day_map]
        st = _stats_over_items(items)
        daily_rows.append((d, st["twap"], st["min"], st["max"], len(items),  len(blocks) - len(items)))

    return combined, daily_rows

# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────
def fmt_money(v: float) -> str:
    return f"₹{v:.2f}"

def dmy(d: date) -> str:
    return d.strftime("%d/%m/%Y")  # dd/mm/yyyy

def is_full_day(blocks: List[int]) -> bool:
    return sorted(blocks) == list(range(1, 25))

def _compress_blocks(blocks: List[int]) -> List[Tuple[int, int]]:
    """Group consecutive blocks: [1,2,3,5,6] -> [(1,3), (5,6)]."""
    if not blocks:
        return []
    b = sorted(set(blocks))
    ranges = []
    start = prev = b[0]
    for x in b[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    return ranges

def blocks_compact_label(blocks: List[int]) -> str:
    parts = []
    for s, e in _compress_blocks(blocks):
        parts.append(f"{s}-{e}" if s != e else f"{s}")
    return ", ".join(parts)

def months_between_inclusive(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month) + 1

def duration_line(start: date, end: date, blocks: List[int]) -> str:
    # If not full day, show explicit blocks compactly
    if not is_full_day(blocks):
        return f"Blocks: {blocks_compact_label(blocks)} ({len(blocks)} blocks)"
    # Full day
    if start == end:
        return "Duration: 0–24 (24 blocks)"
    # Multi-day/month/year → show months count
    months = months_between_inclusive(start, end)
    return f'Duration: {months} month{"s" if months != 1 else ""}'

def stats_md(label: str, st, blocks: List[int], start: date, end: date) -> str:
    if st["count"] == 0 or st["twap"] is None:
        return f"### {label}\n_No data for the requested selection._"
    lines = [
        f"### {label}",
        f"- **Rate:** {fmt_money(st['twap'])}/kWh",
        # Removed Sum line as requested
        f"- **Min–Max:** {fmt_money(st['min'])} – {fmt_money(st['max'])}",
        f"- **{duration_line(start, end, blocks)}**",
    ]
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# Main message handler (tables disabled)
# ─────────────────────────────────────────────────────────────
@cl.on_message
async def on_message(msg: cl.Message):
    text = msg.content.strip()

    # Progress line
    progress = await progress_start("💭 Thinking …")
    t0 = time.perf_counter()

    # Small beat, then Calculating
    await asyncio.sleep(0.6)
    await progress_update(progress, "🧮 Calculating …")

    final_content = None
    try:
        market = parse_market(text) or "DAM"
        ranges = parse_ranges(text)
        blocks = expand_ranges(ranges) if ranges else list(range(1, 25))
        start, end = parse_date_or_range(text)

        if not start or not end:
            final_content = (
                "I couldn't find a valid date or period (day/month/year).\n\n"
                "Try: `26 Aug 2024`, `Aug 2024`, `this month`, `2023`, or `this year`.\n\n"
                + BLOCK_HELP + "\n\n" + EXAMPLE_QUERIES
            )

        elif start == end:
            # Single day
            data = fetch_prices_day(market, start, blocks)
            combined = compute_stats(blocks, data)

            header = f"## **{market} — {dmy(start)}**"
            summary_md = stats_md("Summary", combined, blocks, start, end)

            per_range_md = []
            if ranges:
                for lo, hi in ranges:
                    r_blocks = list(range(lo, hi + 1))
                    r_stats = compute_stats(r_blocks, data)
                    per_range_md.append(stats_md(f"Range {lo}-{hi}", r_stats, r_blocks, start, end))

            final_content = "\n\n".join([header, summary_md, *per_range_md])

        else:
            # Multi-day / month / year
            data_by_day = fetch_prices_range(market, start, end, blocks)
            combined, _daily_rows = compute_stats_over_range(data_by_day, blocks)

            header = f"## **{market} — {dmy(start)} → {dmy(end)}**"
            parts = [header, stats_md("Summary", combined, blocks, start, end)]
            final_content = "\n\n".join(parts)

    except Exception:
        # Log full trace for debugging; show a helpful message to the user
        traceback.print_exc()
        try:
            await progress_hide(progress)
        except Exception:
            pass
        await cl.Message(content="⚠️ Temporary data connection issue. Please try again.").send()
        return
    finally:
        # Ensure ≥ 2s total (for the staged Thinking/Calculating UX)
        elapsed = time.perf_counter() - t0
        if elapsed < 2.0:
            await asyncio.sleep(2.0 - elapsed)
        # Flash 'Calculated' briefly, then hide the progress line
        try:
            await progress_update(progress, "✅ Calculated.")
            await asyncio.sleep(0.35)
            await progress_hide(progress)
        except Exception:
            pass

    # Final answer
    await cl.Message(content=final_content or "_No content to display._").send()
