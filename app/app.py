import os, re
from typing import List, Tuple, Optional, Dict
import chainlit as cl
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import dateparser
from dateparser.search import search_dates
from datetime import date

# --- env & DB -------------------------------------------------
load_dotenv(override=True)
DB_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DB_URL, sslmode="require")
conn.autocommit = True

# --- help text ------------------------------------------------
BLOCK_HELP = """Blocks (hourly): 1=00–01, 2=01–02, …, 24=23–24.
Examples:
• '3-8' → blocks 3..8
• '2-6 and 12-20' → blocks 2..6 plus 12..20
"""

# --- parsing --------------------------------------------------
def parse_market(text: str) -> Optional[str]:
    m = re.search(r"\b(G?DAM)\b", text, re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).upper()
    return "GDAM" if "G" in val else "DAM"

def parse_date(text: str) -> Optional[date]:
    """
    Robust date extractor:
    1) remove hour/block ranges like '2-6' and '12-20'
    2) try dateparser.parse on the cleaned text
    3) fallback to search_dates on the cleaned text
    """
    # Remove ranges 'd-d' so they don't look like dates
    cleaned = re.sub(r"\b\d{1,2}\s*[-–]\s*\d{1,2}\b", " ", text)

    dt = dateparser.parse(cleaned, settings={"DATE_ORDER": "DMY"})
    if dt and dt.year >= 2000:
        return dt.date()

    matches = search_dates(cleaned, settings={"DATE_ORDER": "DMY"})
    if matches:
        # Prefer matches that contain a 4-digit year or a month name
        for s, dtt in matches:
            if (re.search(r"\d{4}", s) or re.search(r"[A-Za-z]", s)) and dtt.year >= 2000:
                return dtt.date()
        # else just take the last one if all else fails
        return matches[-1][1].date()
    return None

def parse_ranges(text: str) -> List[Tuple[int, int]]:
    pieces = re.split(r"\s*(?:,|and)\s*", text, flags=re.IGNORECASE)
    out: List[Tuple[int, int]] = []
    for tok in pieces:
        m = re.search(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = min(a, b), max(a, b)
            if 1 <= lo <= 24 and 1 <= hi <= 24:
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

# --- db fetch & stats ----------------------------------------
def fetch_prices(market: str, d: date, blocks: List[int]) -> Dict[int, Dict[str, float]]:
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
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(q, (market, d, blocks))
        rows = cur.fetchall()
    return {
        int(r["block_index"]): {
            "price": float(r["price_rs_per_mwh"]),
            "minutes": int(r["duration_min"]),
        }
        for r in rows
    }

def compute_stats(blocks: List[int], data: Dict[int, Dict[str, float]]):
    found = [b for b in blocks if b in data]
    missing = sorted(set(blocks) - set(found))

    if not found:
        return {
            "found": [],
            "missing": missing,
            "count": 0,
            "twap": None,
            "sum_1mw": 0.0,
            "min": None,
            "max": None,
        }

    num = sum(data[b]["price"] * data[b]["minutes"] for b in found)
    den = sum(data[b]["minutes"] for b in found)
    twap = num / den if den > 0 else None

    sum_1mw = sum(data[b]["price"] * (data[b]["minutes"] / 60.0) for b in found)

    prices = [data[b]["price"] for b in found]
    return {
        "found": found,
        "missing": missing,
        "count": len(found),
        "twap": twap,
        "sum_1mw": sum_1mw,
        "min": min(prices),
        "max": max(prices),
    }

# --- pretty formatting ---------------------------------------
def fmt_money(v: float) -> str:
    return f"₹{v:.2f}"

def stats_md(label: str, st) -> str:
    if st["count"] == 0:
        return f"### {label}\n_No data for the requested blocks._"
    lines = [
        f"### {label}",
        f"- **Blocks:** {st['count']}  _(missing: {len(st['missing'])})_",
        f"- **TWAP:** {fmt_money(st['twap'])}/MWh" if st["twap"] is not None else "- **TWAP:** n/a",
        f"- **Sum (1 MW):** {fmt_money(st['sum_1mw'])}",
        f"- **Min–Max:** {fmt_money(st['min'])} – {fmt_money(st['max'])}",
    ]
    if st["missing"]:
        lines.append(f"- **Missing blocks:** {st['missing']}")
    return "\n".join(lines)

def per_block_table(found_blocks, data_dict) -> str:
    hdr = "| Block | Price (₹/MWh) | Duration (min) |\n|:----:|:-------------:|:--------------:|"
    rows = [
        f"| {b:02d} | {fmt_money(data_dict[b]['price'])} | {int(data_dict[b]['minutes'])} |"
        for b in found_blocks
    ]
    return "\n".join([hdr, *rows]) if rows else "_No blocks found._"

# --- Chainlit handler ----------------------------------------
@cl.on_message
async def on_message(msg: cl.Message):
    text = msg.content.strip()

    market = parse_market(text)
    d = parse_date(text)
    ranges = parse_ranges(text)
    blocks = expand_ranges(ranges)

    # Missing bits notice (use real newlines)
    missing_bits = []
    if not market:
        missing_bits.append("market (DAM or GDAM)")
    if not d:
        missing_bits.append("date (e.g., 26 Aug 2024)")
    if not blocks:
        missing_bits.append("block ranges (e.g., 3-8 or 2-6 and 12-20)")
    if missing_bits:
        await cl.Message(content="I need " + ", ".join(missing_bits) + ".\n\n" + BLOCK_HELP).send()
        return

    data = fetch_prices(market, d, blocks)
    combined = compute_stats(blocks, data)

    # Combined + per-range + table
    combined_md = stats_md("Combined", combined)

    per_range_md = []
    for lo, hi in ranges:
        r_blocks = list(range(lo, hi + 1))
        r_stats = compute_stats(r_blocks, data)
        per_range_md.append(stats_md(f"Range {lo}-{hi}", r_stats))

    details_table = per_block_table(combined["found"], data)

    header = f"## **{market} — {d.strftime('%Y-%m-%d')}**"
    content = "\n\n".join([header, combined_md, *per_range_md, "#### Details", details_table])

    await cl.Message(content=content).send()
