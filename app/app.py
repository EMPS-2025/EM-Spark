import os, re, calendar, asyncio, traceback
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Tuple, Optional, Dict

import chainlit as cl
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Branding: custom avatar (logo)
# Place your logo at: public/avatars/emps.png
# ─────────────────────────────────────────────────────────────
ASSISTANT_AUTHOR = "EMPS_v2"

# ─────────────────────────────────────────────────────────────
# Environment & DB
# ─────────────────────────────────────────────────────────────
load_dotenv(override=True)
DB_URL = os.getenv("DATABASE_URL", "").strip()
DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = int(os.getenv("DB_PORT", os.getenv("PGPORT", "5432")))
DB_NAME = os.getenv("DB_NAME", "").strip()
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
DB_SSLMODE = os.getenv("DB_SSLMODE", "require").strip()

DEFAULT_STAT = os.getenv("DEFAULT_STAT", "twap").strip().lower()
if DEFAULT_STAT not in ("twap", "vwap", "list", "daily_avg"):
    DEFAULT_STAT = "twap"

DATE_MIN_GUARD = date(2010, 1, 1)

# print("DB PATH:", "DATABASE_URL" if DB_URL else "split fields")
# print("DATABASE_URL =", (DB_URL or "<none>"))
# print("DB_USER =", DB_USER)



def _connect():
    keepalive = dict(connect_timeout=10, keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    if DB_URL:
        return psycopg2.connect(DB_URL, sslmode=DB_SSLMODE, **keepalive)
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD, sslmode=DB_SSLMODE, **keepalive
    )
# Disclaimer footer for derivative market focus
DISCLAIMER_FOOTER = """

---

**Primary Service:** MCX/NSE Derivative Market Analysis | **Complementary:** DAM/GDAM/RTM Spot Data  
*For research purposes only • Not financial advice • Consult licensed professionals*

© 2025 Energy Minds Power Solutions Pvt. Ltd.
"""

# ─────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────
async def progress_start(text: str = "💭 Interpreting …") -> cl.Message:
    m = cl.Message(author=ASSISTANT_AUTHOR, content=text)
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

# @cl.on_chat_start
# async def chat_start():
    
#     await cl.Message(author=ASSISTANT_AUTHOR, content="Welcome to SPARK" + "\n").send()

# ─────────────────────────────────────────────────────────────
# Parsing – deterministic pipeline
# ─────────────────────────────────────────────────────────────
MONTHS = {
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
    "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
    "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
}
MONTH_WORDS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)"
EXCH_RE = re.compile(r"\b(MCX|NSE)\b", re.IGNORECASE)


def normalize(text: str) -> str:
    s = text.strip()
    s = s.replace("–", "-").replace("—", "-").replace("-", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\bbetween\s+(\S.*?)\s+and\s+(\S.*?)\b", r"\1 to \2", s, flags=re.I)
    s = re.sub(r"\b(upto|through|till|until)\b", "to", s, flags=re.I)
    s = re.sub(rf"\b({MONTH_WORDS})\s*[-']\s*(\d{{2}})\b", lambda m: f"{m.group(1)} 20{m.group(2)}", s, flags=re.I)
    return s

def parse_market(text: str) -> str:
    if re.search(r"\b(gdam|green day\.?\s*ahead)\b", text, re.I): return "GDAM"
    if re.search(r"\b(dam)\b", text, re.I): return "DAM"
    return "DAM"

def parse_stat(text: str) -> str:
    s = text.lower()
    if re.search(r"\b(vwap|weighted)\b", s): return "vwap"
    if re.search(r"\bdaily\s+(avg|average)\b", s): return "daily_avg"
    if re.search(r"\b(list|table|rows|detailed)\b", s): return "list"
    if re.search(r"\b(avg|average|mean|twap)\b", s): return "twap"
    return DEFAULT_STAT


def parse_multi_year_months(text: str) -> List[Tuple[date, date]]:
    """
    Robust parser for multi-year month queries.
    Handles:
    - "November 2022, November 2023, November 2024"
    - "November 2022, 2023, 2024"
    - "Nov 2022, 2023, and 2024"
    - "dam november 2022, 2023, 2024"
    - Any combination with commas and 'and'
    
    Returns list of (start_date, end_date) tuples for each month
    """
    s = text.lower().strip()
    results = []
    
    # Pattern 1: "November 2022, 2023, 2024" (month once, multiple years)
    # More precise: month followed by year, then more years separated by commas
    month_match = re.search(
        rf"\b({MONTH_WORDS})\s+(\d{{4}})\b(?:\s*,\s*(?:and\s+)?(\d{{4}}))+", 
        s, re.I
    )
    if month_match:
        month_name = month_match.group(1).lower()
        if month_name in MONTHS:
            month_num = MONTHS[month_name]
            # Get ALL years from the entire matched portion
            full_match = month_match.group(0)
            years = re.findall(r'\b\d{4}\b', full_match)
            
            if len(years) > 1:  # Found multiple years
                for year_str in years:
                    try:
                        year = int(year_str)
                        if 2000 <= year <= 2100:  # Sanity check
                            start = date(year, month_num, 1)
                            end = date(year, month_num, calendar.monthrange(year, month_num)[1])
                            results.append((start, end))
                    except (ValueError, calendar.IllegalMonthError):
                        continue  # Skip invalid dates
                
                if len(results) > 1:  # Only return if we got multiple valid periods
                    return results
    
    # Pattern 2: "November 2022, November 2023, November 2024" (month repeated)
    # More precise: only match complete "Month YYYY" patterns
    pattern = rf"\b({MONTH_WORDS})\s+(\d{{4}})\b"
    matches = re.findall(pattern, s, re.I)
    
    if len(matches) > 1:  # Multiple month-year pairs found
        seen_periods = set()  # Avoid duplicates
        
        for match in matches:
            month_name = match[0].lower()
            year_str = match[1]
            
            if month_name in MONTHS:
                try:
                    month_num = MONTHS[month_name]
                    year = int(year_str)
                    
                    if 2000 <= year <= 2100:  # Sanity check
                        period_key = (year, month_num)
                        if period_key not in seen_periods:
                            seen_periods.add(period_key)
                            start = date(year, month_num, 1)
                            end = date(year, month_num, calendar.monthrange(year, month_num)[1])
                            results.append((start, end))
                except (ValueError, calendar.IllegalMonthError):
                    continue  # Skip invalid dates
        
        if len(results) > 1:  # Only return if we got multiple valid periods
            return results
    
    return []  # Not a multi-year query or couldn't parse

def parse_date_or_range(text: str) -> Tuple[Optional[date], Optional[date]]:
    s = " " + text.lower().strip() + " "
    today = date.today()

    if " yesterday " in s:
        d = today - timedelta(days=1); return (d, d)
    if " today " in s:
        return (today, today)
    if " this month " in s:
        start = date(today.year, today.month, 1)
        end = date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
        return (start, end)
    if " last month " in s:
        y, m = today.year, today.month - 1
        if m == 0: y, m = y - 1, 12
        return (date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1]))
    # Month-day to month-day range in SAME year: "24 September to 24 October 2025"
    m = re.search(
        rf"(?:from\s+)?(\d{{1,2}})\s+({MONTH_WORDS})\s+(?:to|until|till|-)\s+(\d{{1,2}})\s+({MONTH_WORDS})\s+(\d{{2,4}})",
        s, re.I
    )
    if m:
        d1 = int(m.group(1))
        mon1 = MONTHS[m.group(2).lower()]
        d2 = int(m.group(3))
        mon2 = MONTHS[m.group(4).lower()]
        yr = int(m.group(5))
        if yr < 100:
            yr += 2000
        start, end = date(yr, mon1, d1), date(yr, mon2, d2)
        if start > end:
            start, end = end, start
        return (start, end)

    # word-date RANGE (original pattern continues below)
    m = re.search(rf"\b(?:from\s*)?(\d{{1,2}})\s+{MONTH_WORDS}\s+(\d{{2,4}})\s*(?:to|-)\s*(\d{{1,2}})\s+{MONTH_WORDS}\s+(\d{{2,4}})\b", s, re.I)

    # Month-day to month-day range in SAME year: "24 September to 24 October 2025"
    m = re.search(
        rf"(?:from\s+)?(\d{{1,2}})\s+({MONTH_WORDS})\s+(?:to|until|till|-)\s+(\d{{1,2}})\s+({MONTH_WORDS})\s+(\d{{2,4}})",
        s, re.I
    )
    if m:
        d1 = int(m.group(1))
        mon1 = MONTHS[m.group(2).lower()]
        d2 = int(m.group(3))
        mon2 = MONTHS[m.group(4).lower()]
        yr = int(m.group(5))
        if yr < 100:
            yr += 2000
        start, end = date(yr, mon1, d1), date(yr, mon2, d2)
        if start > end:
            start, end = end, start
        return (start, end)

    # word-date RANGE (original pattern continues below)
    m = re.search(rf"\b(?:from\s*)?(\d{{1,2}})\s+{MONTH_WORDS}\s+(\d{{2,4}})\s*(?:to|-)\s*(\d{{1,2}})\s+{MONTH_WORDS}\s+(\d{{2,4}})\b", s, re.I)

    # word-date RANGE
    m = re.search(rf"\b(?:from\s*)?(\d{{1,2}})\s+{MONTH_WORDS}\s+(\d{{2,4}})\s*(?:to|-)\s*(\d{{1,2}})\s+{MONTH_WORDS}\s+(\d{{2,4}})\b", s, re.I)
    if m:
        d1 = int(m.group(1)); mon1 = MONTHS[m.group(2).lower()]; y1 = int(m.group(3))
        d2 = int(m.group(4)); mon2 = MONTHS[m.group(5).lower()]; y2 = int(m.group(6))
        if y1 < 100: y1 += 2000
        if y2 < 100: y2 += 2000
        start, end = date(y1, mon1, d1), date(y2, mon2, d2)
        if start > end: start, end = end, start
        return (start, end)

    # day range within SAME month (e.g., 10-15 Aug 2025)
    m = re.search(rf"\b(\d{{1,2}})\s*(?:to|-)\s*(\d{{1,2}})\s+{MONTH_WORDS}(?:\s+(\d{{2,4}}))?\b", s, re.I)
    if m:
        d1, d2 = int(m.group(1)), int(m.group(2))
        mon = MONTHS[m.group(3).lower()]
        yr  = int(m.group(4)) if m.group(4) else today.year
        if yr < 100: yr += 2000
        return (date(yr, mon, min(d1, d2)), date(yr, mon, max(d1, d2)))

    # numeric date RANGE
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s*(?:to|-)\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", s, re.I)
    if m:
        d1, m1, y1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d2, m2, y2 = int(m.group(4)), int(m.group(5)), int(m.group(6))
        y1 = y1 + 2000 if y1 < 100 else y1
        y2 = y2 + 2000 if y2 < 100 else y2
        start, end = date(y1, m1, d1), date(y2, m2, d2)
        if start > end: start, end = end, start
        return (start, end)

    # single numeric date
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", s)
    if m:
        d0, m0, y0 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y0 < 100: y0 += 2000
        d = date(y0, m0, d0)
        if d >= DATE_MIN_GUARD:
            return (d, d)

    # single day with month word
    m = re.search(rf"\b(\d{{1,2}})\s+{MONTH_WORDS}(?:\s+(\d{{2,4}}))?\b", s, re.I)
    if m:
        d0 = int(m.group(1)); mon0 = MONTHS[m.group(2).lower()]
        y0 = int(m.group(3)) if m.group(3) else today.year
        if y0 < 100: y0 += 2000
        d = date(y0, mon0, d0)
        if d >= DATE_MIN_GUARD:
            return (d, d)

    # month + year
    m = re.search(rf"\b{MONTH_WORDS}\s+(\d{{2,4}})\b", s, re.I)
    if m:
        mon = MONTHS[m.group(1).lower()]
        yr  = int(m.group(2));  yr = yr + 2000 if yr < 100 else yr
        last = calendar.monthrange(yr, mon)[1]
        s1, s2 = date(yr, mon, 1), date(yr, mon, last)
        if s1 >= DATE_MIN_GUARD:
            return (s1, s2)

    # year only (if phrased)
    m = re.search(r"\b(20\d{2})\b", s)
    if m and re.search(r"\b(in|for|year|full\s+year)\b", s):
        y = int(m.group(1)); return (date(y, 1, 1), date(y, 12, 31))

    return (None, None)

def is_month_intent(text: str, start: Optional[date], end: Optional[date]) -> bool:
    if not start or not end:
        return False
    first = date(start.year, start.month, 1)
    last  = date(start.year, start.month, calendar.monthrange(start.year, start.month)[1])
    whole_month = (start == first and end == last)
    yyyymm = bool(re.search(r"\b20\d{2}-(0[1-9]|1[0-2])\b", text or ""))
    has_month_word = bool(re.search(MONTH_WORDS, text or "", re.I))
    return whole_month and has_month_word or yyyymm

# Ranges for hours/slots (unchanged from your current app)
def _fmt_hhmm(total_min: int) -> str:
    h = (total_min // 60) % 24
    m = total_min % 60
    return f"{h:02d}:{m:02d}"

def hour_block_window(b: int) -> str:
    return f"{_fmt_hhmm((b-1)*60)}–{_fmt_hhmm(b*60)}"

def slot_window(s: int) -> str:
    return f"{_fmt_hhmm((s-1)*15)}–{_fmt_hhmm(s*15)}"


def _compress_ranges(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    b = sorted(set(indices))
    out = []
    s = p = b[0]
    for x in b[1:]:
        if x == p + 1:
            p = x
        else:
            out.append((s, p))
            s = p = x
    out.append((s, p))
    return out

@dataclass
class QuerySpec:
    market: str
    start_date: date
    end_date: date
    granularity: str            # 'hour' | 'quarter'
    hours: Optional[List[int]]
    slots: Optional[List[int]]
    stat: str                   # 'list' | 'twap' | 'vwap' | 'daily_avg'
    area: str = "ALL"


def parse_ranges(text: str) -> dict:
    s_orig = normalize(text)
    s = s_orig.lower()

    hours: List[int] = []
    quarters: List[int] = []

    prefer_quarter = bool(re.search(r"\b(blocks?|slots?|quarters?)\b", s))
    prefer_hour    = bool(re.search(r"\b(hours?|hrs?)\b", s))

    any_minute_nonzero = False

    if re.search(r"\b(full day|all 24|entire day|whole day)\b", s):
        hours = list(range(1, 25))

    time_pat = re.compile(
        r"\b(?:from\s*)?"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*"
        r"(?:to|till|until|-)\s*"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        re.I,
    )
    found_time_range = False

    def _to24_inner(h: int, ampm: Optional[str]) -> int:
        if ampm:
            h = h % 12
            if ampm.lower() == "pm":
                h += 12
        return max(0, min(23, h))

    for m in time_pat.finditer(s):
        found_time_range = True
        h1, m1, a1 = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        h2, m2, a2 = int(m.group(4)), int(m.group(5) or 0), m.group(6)

        if m1 > 0 or m2 > 0:
            any_minute_nonzero = True

        H1, H2 = _to24_inner(h1, a1), _to24_inner(h2, a2)

        start_block = min(24, H1 + 1 + (1 if m1 > 0 else 0))
        end_block   = min(24, H2 + (0 if m2 == 0 else 1))
        if m2 == 0:
            end_block = max(1, H2)
        if end_block >= start_block:
            hours.extend(range(start_block, end_block + 1))

        def ceil_slot(h: int, mm: int) -> int:
            return (h * 60 + mm + 14) // 15 + 1

        def end_slot(h: int, mm: int) -> int:
            return (h * 60 + mm) // 15

        sslot = max(1, min(96, ceil_slot(H1, m1)))
        eslot = max(1, min(96, end_slot(H2, m2)))
        if eslot >= sslot:
            quarters.extend(range(sslot, eslot + 1))

    clean = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", s)

    for m in re.finditer(r"\b(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*(?:hours?|hrs?)\b", clean, re.I):
        h1 = max(0, min(23, int(m.group(1))))
        h2 = max(0, min(23, int(m.group(2))))
        start_block = min(24, h1 + 1)
        end_block   = max(1,  min(24, h2))
        if end_block >= start_block:
            hours.extend(range(start_block, end_block + 1))

    for m in re.finditer(r"\b(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*(?:blocks?|slots?|quarters?)\b", clean, re.I):
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = sorted((a, b))
        lo = max(1, lo); hi = min(96, hi)
        quarters.extend(range(lo, hi + 1))
        prefer_quarter = True

    if not found_time_range:
        for m in re.finditer(r"\b(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\b", clean, re.I):
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = sorted((a, b))
            if prefer_quarter or hi > 24:
                lo = max(1, lo); hi = min(96, hi)
                quarters.extend(range(lo, hi + 1))
            else:
                lo = max(1, lo); hi = min(24, hi)
                hours.extend(range(lo, hi + 1))

    hours    = sorted({h for h in hours    if 1 <= h <= 24})
    quarters = sorted({q for q in quarters if 1 <= q <= 96})

    if prefer_quarter or any_minute_nonzero:
        gran = "quarter"
    elif prefer_hour:
        gran = "hour"
    else:
        gran = "hour" if hours else "quarter"

    return {"hours": hours, "quarters": quarters, "granularity": gran}


def canonicalize(market: str, start: Optional[date], end: Optional[date], gran: str, hours: List[int], slots: List[int], stat: str) -> Optional[QuerySpec]:
    if not start or not end:
        return None
    if start > end:
        start, end = end, start
    market = "GDAM" if market.upper()=="GDAM" else "DAM"
    gran = "hour" if gran=="hour" else "quarter"
    if gran == "hour":
        hs = sorted(set([h for h in hours if 1<=h<=24])) or list(range(1,25))
        return QuerySpec(market, start, end, "hour", hs, None, stat if stat in ("list","twap","vwap","daily_avg") else "list")
    else:
        qs = sorted(set([q for q in slots if 1<=q<=96])) or list(range(1,97))
        return QuerySpec(market, start, end, "quarter", None, qs, stat if stat in ("list","twap","vwap","daily_avg") else "list")


def dmy(d: date) -> str: return d.strftime("%d %b %Y")

def _block_to_time(b: int) -> str:
    # block 1 => 00:00, block 24 ends at 24:00
    return f"{b-1:02d}:00"

def format_duration(block_start: Optional[int], block_end: Optional[int]) -> str:
    # Full day if blocks not provided or explicitly 1..24
    if not block_start or not block_end or (block_start == 1 and block_end == 24):
        return "00:00–24:00 (24 hrs)"
    hours = max(1, block_end - block_start + 1)
    start_txt = _block_to_time(block_start)   # start of the first block
    end_txt   = f"{block_end:02d}:00"         # end of the last block
    return f"{start_txt}–{end_txt} ({hours} hrs)"


# ─────────────────────────────────────────────────────────────
# DB calls (DAM/GDAM)
# ─────────────────────────────────────────────────────────────

def fetch_hourly(market: str, ds: date, de: date, b1: Optional[int], b2: Optional[int]) -> List[Dict]:
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        if b1 and b2:
            cur.execute("SELECT * FROM public.rpc_get_hourly_prices_range(%s,%s,%s,%s,%s);", (market, ds, de, b1, b2))
        else:
            cur.execute("SELECT * FROM public.rpc_get_hourly_prices_range(%s,%s,%s,NULL,NULL);", (market, ds, de))
        return [dict(r) for r in cur.fetchall()]


def fetch_quarter(market: str, ds: date, de: date, s1: Optional[int], s2: Optional[int]) -> List[Dict]:
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        if s1 and s2:
            cur.execute("SELECT * FROM public.rpc_get_quarter_prices_range(%s,%s,%s,%s,%s);", (market, ds, de, s1, s2))
        else:
            cur.execute("SELECT * FROM public.rpc_get_quarter_prices_range(%s,%s,%s,NULL,NULL);", (market, ds, de))
        return [dict(r) for r in cur.fetchall()]

# ─────────────────────────────────────────────────────────────
# DB calls (Derivatives)
# ─────────────────────────────────────────────────────────────

def fetch_deriv_daily_fallback(target_day: date, exchange: Optional[str]) -> List[Dict]:
    """Returns daily close for nearest prior trading day (<= target_day) per exchange.
       If no rows (i.e., before Jul 2025), the caller renders N/A."""
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("SELECT * FROM public.rpc_deriv_daily_with_fallback(%s,%s);", (exchange, target_day))
        return [dict(r) for r in cur.fetchall()]


def fetch_deriv_month_expiry(cm_first: date, exchange: Optional[str]) -> List[Dict]:
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("SELECT * FROM public.rpc_deriv_expiry_for_month(%s,%s);", (exchange, cm_first))
        return [dict(r) for r in cur.fetchall()]

# ─────────────────────────────────────────────────────────────
# Math (₹/kWh)
# ─────────────────────────────────────────────────────────────

def twap_kwh(rows: List[Dict], price_key: str, minute_key: str) -> Optional[float]:
    if not rows: return None
    num = sum(float(r[price_key]) * float(r[minute_key]) for r in rows)
    den = sum(float(r[minute_key]) for r in rows)
    return None if den==0 else (num/den)/1000.0


def vwap_kwh(rows: List[Dict], price_key: str, sched_key: str, minute_key: str) -> Optional[float]:
    if not rows: return None
    weights = [float(r.get(sched_key) or 0) * float(r[minute_key]) for r in rows]
    num = sum(float(r[price_key]) * w for r, w in zip(rows, weights))
    den = sum(weights)
    if den>0: return (num/den)/1000.0
    return twap_kwh(rows, price_key, minute_key)


def money(v: Optional[float]) -> str:
    return "—" if v is None else f"₹{v:.4f}"

# ─────────────────────────────────────────────────────────────
# Presentation helpers (selection card + tables)
# ─────────────────────────────────────────────────────────────

def _label_hour_ranges(hours: List[int]) -> Tuple[str, str, int]:
    rngs = _compress_ranges(hours)
    time_parts  = [f"{_fmt_hhmm((s-1)*60)}–{_fmt_hhmm(e*60)}" for s, e in rngs]
    idx_parts   = [f"{s}–{e}" if s != e else f"{s}" for s, e in rngs]
    total_count = sum(e - s + 1 for s, e in rngs)
    return " + ".join(time_parts), ", ".join(idx_parts), total_count


def _label_slot_ranges(slots: List[int]) -> Tuple[str, str, int]:
    rngs = _compress_ranges(slots)
    time_parts  = [f"{_fmt_hhmm((s-1)*15)}–{_fmt_hhmm(e*15)}" for s, e in rngs]
    idx_parts   = [f"{s}–{e}" if s != e else f"{s}" for s, e in rngs]
    total_count = sum(e - s + 1 for s, e in rngs)
    return " + ".join(time_parts), ", ".join(idx_parts), total_count


def _primary_metric_label(stat: str) -> str:
    if stat == "vwap": return "Average price (VWAP)"
    if stat == "twap": return "Average price"
    if stat == "daily_avg": return "Daily average"
    return "Average price"


def _render_selection_card(spec, time_label: str, idx_label: str, count: int) -> str:
    if spec.granularity == "hour":
        count_label  = f"{count} hrs"; index_label  = f"blocks {idx_label}"; mode_label   = "Hourly"
    else:
        count_label  = f"{count}×15min"; index_label  = f"slots {idx_label}"; mode_label   = "15-min"
    metric_label = _primary_metric_label(spec.stat)
    return (
        "### Summary\n\n"
        "| **Parameter** | **Value** |\n"
        "|----------------|------------|\n"
        f"| **Market** | {spec.market} |\n"
        f"| **Period** | {dmy(spec.start_date)} to {dmy(spec.end_date)} |\n"
        f"| **Duration** | {time_label} ({count_label}) |\n"
    )

# Tables for DAM/GDAM

def rows_to_md_hour(rows: List[Dict], limit=120) -> str:
    if not rows: return "_No rows._"
    hdr="| Date | Hour (HH:MM–HH:MM) | Block | Price (₹/kWh) | Sched MW |\n|---|---|---:|---:|---:|"
    show = rows if len(rows)<=limit else rows[:60]+rows[-60:]
    lines=[hdr]
    for r in show:
        dd = r["delivery_date"] if isinstance(r["delivery_date"], str) else dmy(r["delivery_date"])
        b  = int(r["block_index"])
        price_kwh = float(r["price_avg_rs_per_mwh"])/1000.0
        lines.append(f"| {dd} | {hour_block_window(b)} | {b:>2} | {price_kwh:.4f} | {float(r['scheduled_mw_sum'] or 0):.2f} |")
    if len(rows)>limit: lines.insert(1,f"_Showing first 60 and last 60 of {len(rows)} rows (total {len(rows)})._")
    return "\n".join(lines)


def rows_to_md_quarter(rows: List[Dict], limit=120) -> str:
    if not rows: return "_No rows._"
    hdr="| Date | Slot (HH:MM–HH:MM) | Slot # | Price (₹/kWh) | Sched MW |\n|---|---|---:|---:|---:|"
    show = rows if len(rows)<=limit else rows[:60]+rows[-60:]
    lines=[hdr]
    for r in show:
        dd = r["delivery_date"] if isinstance(r["delivery_date"], str) else dmy(r["delivery_date"])
        s  = int(r["slot_index"])
        price_kwh = float(r["price_rs_per_mwh"])/1000.0
        lines.append(f"| {dd} | {slot_window(s)} | {s:>2} | {price_kwh:.4f} | {float(r['scheduled_mw'] or 0):.2f} |")
    if len(rows)>limit: lines.insert(1,f"_Showing first 60 and last 60 of {len(rows)} rows (total {len(rows)})._")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# Derivatives rendering
# ─────────────────────────────────────────────────────────────

# def render_deriv_companion_for_day(requested_day: date, rows: List[Dict]) -> str:
#     """Side panel for derivatives when user asked a single day for DAM/GDAM."""
#     if not rows:
#         return f"### ⭐ **Derivative Market (MCX/NSE)** — {requested_day.strftime('%d %b %Y')}\n*Primary Service*\n\nN/A (no derivative data before Jul 2025)."

#     # Different exchanges might have different last-open days
#     used_dates = sorted({(r['used_trading_date'].date() if isinstance(r['used_trading_date'], datetime) else r['used_trading_date']) for r in rows})
#     if len(used_dates) == 1 and used_dates[0] != requested_day:
#         note = f" _(market closed; showing last close on {used_dates[0].strftime('%d %b %Y')})_"
#     elif len(used_dates) > 1:
#         note = " _(market closed; showing last close per exchange)_"
#     else:
#         note = ""
#         lines = [f"### ⭐ **Derivative Market (MCX/NSE)** — {requested_day.strftime('%d %b %Y')}\n*Primary Service*{note}\n"]
#     for r in rows:
#         td = r['trading_date']; cm = r['contract_month']
#         if isinstance(td, datetime): td = td.date()
#         if isinstance(cm, datetime): cm = cm.date()
#         tag = "" if td == requested_day else f" _(from {td.strftime('%d %b %Y')})_"
#         lines.append(f"- **{r['exchange']} • {r['commodity']} • {cm.strftime('%b %Y')}** → ₹{float(r['close_price_rs_per_mwh']):.2f}/MWh{tag}")
#     return "\n".join(lines)

def render_deriv_companion_for_day(requested_day: date, rows: List[Dict]) -> str:
    """Side panel for derivatives when user asked a single day for DAM/GDAM."""
    if not rows:
        return f"### **Derivative Market (MCX/NSE)** — {requested_day.strftime('%d %b %Y')}\n**\n\nN/A (no derivative data before Jul 2025)."
    
    # Get all unique used trading dates from the results
    used_dates = sorted([
        r['used_trading_date'].date() if isinstance(r['used_trading_date'], datetime) else r['used_trading_date']
        for r in rows
    ])
    
    # Determine if market was closed and add note
    if len(used_dates) == 1 and used_dates[0] != requested_day:
        note = f"\n\n*Market closed on {requested_day.strftime('%d %b %Y')} — showing last close on {used_dates[0].strftime('%d %b %Y')}*"
    elif len(used_dates) > 1:
        note = "\n\n*Market closed — showing last close per exchange*"
    else:
        note = ""
    
    # Initialize lines BEFORE using it
    lines = [f"### **Derivative Market (MCX/NSE)** — {requested_day.strftime('%d %b %Y')}\n{note}\n"]

    # Add each derivative contract
    for r in rows:
        td = r['trading_date']
        cm = r['contract_month']
        

        # Convert to date if datetime
        if isinstance(td, datetime):
            td = td.date()
        if isinstance(cm, datetime):
            cm = cm.date()
        
        # Add tag if showing previous day's data
        tag = "" if td == requested_day else f" ({td.strftime('%d %b %Y')})"
        
        lines.append(f"- **{r['exchange']} • {r['commodity']} • {cm.strftime('%b %Y')}** → ₹{float(r['close_price_rs_per_mwh']/1000):.2f}/KWh{tag}")
    
    return "\n".join(lines)
def highlight_gdam(text: str) -> str:
    """Highlight GDAM with bold green dot emoji (cross-platform, works in tables)"""
    # This will work inside markdown tables and all markdown blocks
    return re.sub(
        r'\b(GDAM)\b',
        r'🟢 **\1**',
        text
    )

def render_deriv_expiry(cm_first: date, rows: List[Dict]) -> str:
    year, month = cm_first.year, cm_first.month
    from datetime import date as _date
    import calendar as _cal
    cm_last = _date(year, month, _cal.monthrange(year, month)[1])
    label = f"{dmy(cm_first)} to {dmy(cm_last)}"
    if not rows:
        return f"### ⭐ **Derivative Market — {label}**\n*Primary Service*\n\n_Expiry not available yet._"

    lines = [f"### ⭐ **Derivative Market — {label} (Expiry Close)**\n*Primary Service*\n"]
    for r in rows:
        ed = r["expiry_date"]
        if isinstance(ed, datetime):
            ed = ed.date()
        lines.append(
            f"- **{r['exchange']} • {r['commodity']}** → ₹{float(r['expiry_close']):.2f}/MWh (on {dmy(ed)})"
        )
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# Main handler
# ─────────────────────────────────────────────────────────────
@cl.on_message
async def on_message(msg: cl.Message):
    text_raw = msg.content.strip()
    s_norm = normalize(text_raw)

    progress = await progress_start("💭 Interpreting …")
    await asyncio.sleep(0.15)
    await progress_update(progress, "🧮 Querying …")

    try:
            # Parse query  
        # Parse query
        market = parse_market(s_norm)
        stat = parse_stat(s_norm)
        
        # Check for multi-year query FIRST
        multi_periods = parse_multi_year_months(s_norm)
        
        if multi_periods:
            # Handle multi-period query - render each period separately
            await progress_hide(progress)
            
            all_responses = []
            
            for period_start, period_end in multi_periods:
                if period_start < DATE_MIN_GUARD:
                    continue
                
                try:
                    # Parse ranges for this period
                    rng = parse_ranges(s_norm)
                    spec = canonicalize(market, period_start, period_end,
                                    rng["granularity"], rng["hours"], rng["quarters"], stat)
                    
                    if spec is None:
                        continue
                    
                    # Build content parts for this period
                    content_parts = []
                    
                    # Build header
                    if spec.granularity == "hour":
                        tlabel, blabel, n = label_hour_ranges(spec.hours)
                        selection_card = render_selection_card(spec, tlabel, blabel, n)
                    else:
                        tlabel, slabel, n = label_slot_ranges(spec.slots)
                        selection_card = render_selection_card(spec, tlabel, slabel, n)
                    
                    title = f"## Spot Market ({spec.market}) — {dmy(spec.start_date)} to {dmy(spec.end_date)}"
                    complementary_note = "*Complementary Reference Data*\n\n"
                    header = f"{title}\n{complementary_note}{selection_card}"
                    content_parts.append(header)
                    
                    # Fetch data
                    if spec.granularity == "hour":
                        rows = []
                        for b1, b2 in compress_ranges(spec.hours):
                            rows += fetch_hourly(spec.market, spec.start_date, spec.end_date, b1, b2)
                        
                        twap_val = twap_kwh(rows, "price_avg_rs_per_mwh", "duration_min")
                        vwap_val = vwap_kwh(rows, "price_avg_rs_per_mwh", "scheduled_mw_sum", "duration_min")
                        
                        primary_label = "Average price (VWAP)" if spec.stat == "vwap" else "Average price"
                        primary_value = vwap_val if spec.stat == "vwap" else twap_val
                        content_parts.append(f"**{primary_label}: {money(primary_value)} /kWh**")
                        
                    else:  # quarters
                        rows = []
                        for s1, s2 in compress_ranges(spec.slots):
                            rows += fetch_quarter(spec.market, spec.start_date, spec.end_date, s1, s2)
                        
                        twap_val = twap_kwh(rows, "price_rs_per_mwh", "duration_min")
                        vwap_val = vwap_kwh(rows, "price_rs_per_mwh", "scheduled_mw", "duration_min")
                        
                        primary_label = "Average price (VWAP)" if spec.stat == "vwap" else "Average price"
                        primary_value = vwap_val if spec.stat == "vwap" else twap_val
                        content_parts.append(f"**{primary_label}: {money(primary_value)} /kWh**")
                    
                    # Add derivatives for this period
                    if is_month_intent(s_norm, spec.start_date, spec.end_date):
                        cm_first = date(spec.start_date.year, spec.start_date.month, 1)
                        mrows = fetch_deriv_month_expiry(cm_first, None)
                        content_parts.append("\n" + render_deriv_expiry(cm_first, mrows))
                    
                    all_responses.append("\n\n".join(content_parts))
                    
                except Exception as e:
                    print(f"Error processing period {dmy(period_start)}: {e}")
                    traceback.print_exc()
                    continue
            
            # Send all responses combined
            if all_responses:
                final_content = "\n\n---\n\n".join(all_responses) + DISCLAIMER_FOOTER
                await cl.Message(author=ASSISTANT_AUTHOR, content=final_content).send()
            else:
                await cl.Message(author=ASSISTANT_AUTHOR, 
                            content="Could not process your multi-period query. Please try one period at a time." + DISCLAIMER_FOOTER).send()
            return  # IMPORTANT: Exit here so we don't continue to single-period logic
        
        # If not multi-period, continue with existing single date range logic
        start, end = parse_date_or_range(s_norm)
            # ... rest of existing code

        if start and start < DATE_MIN_GUARD:
            await progress_hide(progress)
            await cl.Message(author=ASSISTANT_AUTHOR, content="Date appears invalid (before 2010). Please try again.").send()
            return

        rng = parse_ranges(s_norm)
        spec = canonicalize(market, start, end, rng["granularity"], rng["hours"], rng["quarters"], stat)

        if spec is None:
            await progress_hide(progress)
            await cl.Message(author = ASSISTANT_AUTHOR ,content=(
                "I couldn't find a valid date or period. Try one of:\n"
                "- `15/08/2025`\n- `10–15 Aug 2025`\n- `Aug 2025`\n- `this month`, `yesterday`\n\n"
                "Also include hours or blocks:\n- `08:00–18:00`, `10 to 12 hrs`, or `blocks 5–12`"
            )).send()
            return

        # Selection card (pretty header)
        if spec.granularity == "hour":
            tlabel, blabel, n = _label_hour_ranges(spec.hours)
            selection_card = _render_selection_card(spec, tlabel, blabel, n)
        else:
            tlabel, slabel, n = _label_slot_ranges(spec.slots)
            selection_card = _render_selection_card(spec, tlabel, slabel, n)

        title = f"## Spot Market ({spec.market}) — {dmy(spec.start_date)} to {dmy(spec.end_date)}"
        complementary_note = "\n\n" #Complimentary data
        header = f"{title}\n{complementary_note}{selection_card}"


        # Query & render DAM/GDAM
        if spec.granularity=="hour":
            rows: List[Dict] = []
            for b1,b2 in _compress_ranges(spec.hours):
                rows += fetch_hourly(spec.market, spec.start_date, spec.end_date, b1, b2)

            twap = twap_kwh(rows, "price_avg_rs_per_mwh", "duration_min")
            vwap = vwap_kwh(rows, "price_avg_rs_per_mwh", "scheduled_mw_sum", "duration_min")
            energy_mwh = sum((float(r.get("scheduled_mw_sum") or 0)) * (float(r["duration_min"])/60.0) for r in rows)
            primary_label = "Average price (VWAP)" if spec.stat=="vwap" else ("Daily average" if spec.stat=="daily_avg" else "Average price")
            primary_value = vwap if spec.stat == "vwap" else twap
            kpi = f"**{primary_label}: {money(primary_value)} /kWh**\n\n"

            if spec.stat=="daily_avg":
                byday: Dict[date, List[Dict]] = {}
                for r in rows:
                    dd = r["delivery_date"] if isinstance(r["delivery_date"], date) else datetime.fromisoformat(r["delivery_date"]).date()
                    byday.setdefault(dd, []).append(r)
                lines=["| Date | Daily Avg (₹/kWh) |","|---|---:|"]
                for dd in sorted(byday):
                    dv = twap_kwh(byday[dd], "price_avg_rs_per_mwh", "duration_min")
                    lines.append(f"| {dmy(dd)} | {money(dv)} |")
                body = kpi + "\n\n" + "\n".join(lines)
            elif spec.stat=="list":
                body = kpi + "\n\n" + rows_to_md_hour(rows)
            else:
                body = kpi

            content_parts = [header, body]

        else:  # quarter path
            rows: List[Dict] = []
            for s1,s2 in _compress_ranges(spec.slots):
                rows += fetch_quarter(spec.market, spec.start_date, spec.end_date, s1, s2)

            twap = twap_kwh(rows, "price_rs_per_mwh", "duration_min")
            vwap = vwap_kwh(rows, "price_rs_per_mwh", "scheduled_mw", "duration_min")
            energy_mwh = sum((float(r.get("scheduled_mw") or 0)) * (float(r["duration_min"])/60.0) for r in rows)
            primary_label = "Average price (VWAP)" if spec.stat=="vwap" else ("Daily average" if spec.stat=="daily_avg" else "Average price")
            primary_value = vwap if spec.stat == "vwap" else twap
            kpi = f"**{primary_label}: {money(primary_value)} /kWh**\n\n"

            if spec.stat=="daily_avg":
                byday: Dict[date, List[Dict]] = {}
                for r in rows:
                    dd = r["delivery_date"] if isinstance(r["delivery_date"], date) else datetime.fromisoformat(r["delivery_date"]).date()
                    byday.setdefault(dd, []).append(r)
                lines=["| Date | Daily Avg (₹/kWh) |","|---|---:|"]
                for dd in sorted(byday):
                    dv = twap_kwh(byday[dd], "price_rs_per_mwh", "duration_min")
                    lines.append(f"| {dmy(dd)} | {money(dv)} |")
                body = kpi + "\n\n" + "\n".join(lines)
            elif spec.stat=="list":
                body = kpi + "\n\n" + rows_to_md_quarter(rows)
            else:
                body = kpi

            content_parts = [header, body]

        # ── Derivative companion beside DAM/GDAM ─────────────────────
        if spec.start_date == spec.end_date:
            # Single date: daily close with fallback (NA if before Jul 2025)...
            drows = fetch_deriv_daily_fallback(spec.start_date, None)
            content_parts.append("\n" + render_deriv_companion_for_day(spec.start_date, drows))
        elif is_month_intent(s_norm, spec.start_date, spec.end_date):
            # Month intent: expiry close
            cm_first = date(spec.start_date.year, spec.start_date.month, 1)
            mrows = fetch_deriv_month_expiry(cm_first, None)
            content_parts.append("\n" + render_deriv_expiry(cm_first, mrows))
        else:
            # Date range: Show derivatives for the END date of the range
            drows = fetch_deriv_daily_fallback(spec.end_date, None)
            if drows:
                content_parts.append("\n" + render_deriv_companion_for_day(spec.end_date, drows))


        await progress_hide(progress)
        final_content = "\n\n".join(content_parts)
        final_content = highlight_gdam(final_content)
        await cl.Message(author=ASSISTANT_AUTHOR, content=final_content).send()


    except Exception:
        traceback.print_exc()
        try: await progress_hide(progress)
        except Exception: pass
        await cl.Message(author=ASSISTANT_AUTHOR, content="⚠️ Temporary data connection issue. Please try again.").send()
