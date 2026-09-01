"""
SFG Weekly Acquisition Target Report

Runs headless on a schedule (Railway cron), builds three ranked top-10 lists of
RIA acquisition targets from SEC Form ADV data, and emails them via Resend.

Deliberately self-contained: agent_sfg_acquisition.py is Streamlit-coupled
(st.caption / st.expander calls inside its data path) and cannot run headless.
The data logic here is a copy of the logic verified working in that app.

Verified facts this relies on (do not "correct" without re-checking):
  * SEC ROSTER FILENAMES ARE NOT STABLE. This has now changed twice:
      Aug 2026: ia08032026-exempt.zip  held the REGISTERED advisers (inverted),
                ia08032026.zip         held the ERAs.
      Sep 2026: ia09012026-registered.zip holds the REGISTERED advisers,
                ia09012026-exempt.zip     holds the ERAs.  (naming now sane)
    The August files were also later re-published with "_1" appended.
    Because of this the roster is ALWAYS chosen by CONTENT — count the rows
    whose Firm Type is not exempt/ERA — and never by filename. Do not add
    logic that trusts a suffix.
  * The registered roster is 448 columns with total regulatory AUM in
    column 5F(2)(c). The ERA roster is ~173 columns with no AUM at all.
    Both column layouts re-verified against the live September file.
  * Item 5.D letters: (a) individuals, (b) high-net-worth individuals,
    (f) pooled investment vehicles, (g) pension/profit-sharing plans.
    Sub-column (1) is a client count, (3) is AUM from that client type.
  * NO COLUMN IN THE ROSTER NAMES A PERSON. Founder/principal names come from
    the IAPD individual API via iapd.py, keyed on Organization CRD#.

Env vars:
  RESEND_API_KEY   required
  REPORT_TO        required, recipient address
  REPORT_FROM      default "SFG Acquisition Agent <onboarding@resend.dev>"
  SEC_USER_AGENT   required by sec.gov (org + contact email)
  SEC_ADV_FILE_URL optional pin; skips discovery
  REPORT_REGIONS   default "Southwest"
  REPORT_AUM_MIN / REPORT_AUM_MAX      default 50 / 250 (millions)
  REPORT_MIN_RETAIL / REPORT_MIN_CLIENTS  default 50 (%) / 25
  REPORT_TOP_N     default 10
  REPORT_DRY_RUN   "1" prints the HTML instead of sending
  REPORT_NO_PEOPLE "1" skips IAPD principal lookup
"""

import io
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

import iapd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEC_FILE_BASE = (
    "https://www.sec.gov/files/investment/data/other/"
    "information-about-registered-investment-advisers-exempt-reporting-advisers/"
)
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "").strip()
SEC_ADV_FILE_URL = os.getenv("SEC_ADV_FILE_URL", "").strip()
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT or "SFG Acquisition Report (set SEC_USER_AGENT)",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
}
PROBE_DELAY = 0.15  # sec.gov limits automated clients to ~10 req/s

REGION_STATES = {
    "Southwest": ["AZ", "NM", "NV", "UT", "TX", "OK"],
    "Mountain West": ["CO", "MT", "ID", "WY", "UT", "NV"],
    "Southeast": ["FL", "GA", "NC", "SC", "TN", "AL", "MS", "KY", "VA", "WV", "AR", "LA"],
    "Midwest": ["OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"],
    "Northeast": ["NY", "NJ", "PA", "MA", "CT", "RI", "NH", "VT", "ME", "MD", "DE"],
    "West Coast": ["CA", "OR", "WA"],
    "National": [],
}

COLUMN_PATTERNS = {
    "firm_name":      [r"primary business name", r"legal name"],
    "crd":            [r"organization\s*crd", r"\bcrd\s*#"],
    "city":           [r"main office.*city"],
    "state":          [r"main office.*state"],
    "website":        [r"website address"],
    "firm_type":      [r"^firm type$"],
    "aum":            [r"^5F\(2\)\(c\)$"],
    "employees":      [r"^5A$"],
    "indiv_aum":      [r"^5D\(a\)\(3\)$"],
    "hnw_aum":        [r"^5D\(b\)\(3\)$"],
    "pension_aum":    [r"^5D\(g\)\(3\)$"],
    "indiv_clients":  [r"^5D\(a\)\(1\)$"],
    "hnw_clients":    [r"^5D\(b\)\(1\)$"],
}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def detect_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    for field, patterns in COLUMN_PATTERNS.items():
        found = None
        for pattern in patterns:
            for col in columns:
                if re.search(pattern, str(col), re.IGNORECASE):
                    found = col
                    break
            if found:
                break
        mapping[field] = found
    return mapping


def wanted_column(col: Any) -> bool:
    text = str(col)
    return any(
        re.search(p, text, re.IGNORECASE)
        for patterns in COLUMN_PATTERNS.values()
        for p in patterns
    )


# Every filename shape the SEC has used. "-registered.zip" is September's and
# must stay first; "-exempt.zip" was August's registered roster despite the
# name. The "_1" variants are the SEC's own re-publications. Which one holds
# the registered advisers is decided by content, never by position here.
SEC_SUFFIXES = (
    "-registered.zip", "-exempt.zip", ".zip",
    "-registered_1.zip", "-exempt_1.zip", "_1.zip",
)


def candidate_stamps(months_back: int = 3, max_day: int = 12) -> List[str]:
    """Publication date stamps (MMDDYYYY), newest first."""
    stamps: List[str] = []
    today = datetime.now()
    year, month = today.year, today.month
    for _ in range(months_back):
        for day in range(max_day, 0, -1):
            stamps.append(f"{month:02d}{day:02d}{year}")
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    return stamps


def stamp_urls(stamp: str, primary_only: bool = False) -> List[str]:
    """
    URLs for one publication date.

    `primary_only` limits probing to the three shapes the SEC actually
    publishes on the day; the "_1" re-publications are only worth asking for
    once a date is known to exist. Halves the probe count, which matters —
    un-throttled probing past ~10 req/s is what made sec.gov answer 403 to
    everything once before.
    """
    suffixes = SEC_SUFFIXES[:3] if primary_only else SEC_SUFFIXES
    return [f"{SEC_FILE_BASE}ia{stamp}{s}" for s in suffixes]


def registered_rows(df: "pd.DataFrame", mapping: Dict[str, Optional[str]]) -> int:
    col = mapping.get("firm_type")
    if not col or col not in df.columns:
        return 0
    types = df[col].astype(str).str.strip()
    return int((~types.str.contains(r"exempt|\bERA\b", case=False, regex=True, na=False)).sum())


def try_urls(client: "httpx.Client", urls: List[str]) -> Optional[Dict[str, Any]]:
    """Download the given URLs and return the best registered roster among them."""
    best = None
    for url in urls:
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                members = [n for n in zf.namelist() if n.lower().endswith((".csv", ".xlsx"))]
                if not members:
                    continue
                raw = zf.read(members[0])
            df = pd.read_csv(
                io.BytesIO(raw), low_memory=False, encoding_errors="replace",
                usecols=wanted_column, dtype=str,
            )
            mapping = detect_columns(list(df.columns))
            n_reg = registered_rows(df, mapping)
            log(f"{url.rsplit('/', 1)[-1]}: {len(df):,} rows, {n_reg:,} registered")
            if n_reg > 0 and mapping.get("aum") and (best is None or n_reg > best["registered"]):
                best = {"df": df, "mapping": mapping, "source": url, "registered": n_reg}
        except Exception as e:
            log(f"skip {url.rsplit('/', 1)[-1]}: {e}")
    return best


def load_roster() -> Dict[str, Any]:
    """
    Download the SEC roster containing REGISTERED advisers, chosen by content.

    Walks publication dates newest-first and only stops when a date actually
    yields a registered roster with AUM. The previous version latched onto the
    newest date that had ANY file and gave up there, which is how a month where
    the ERA file publishes first silently produced an empty report.
    """
    with httpx.Client(timeout=180, follow_redirects=True, headers=SEC_HEADERS) as client:
        # A pin is a hint, not a contract. Try it and its siblings first, but
        # fall through to discovery if it has gone stale — SEC has renamed
        # these files twice, and a dead pin must not kill the whole report.
        if SEC_ADV_FILE_URL:
            stamp = (re.search(r"ia(\d{8})", SEC_ADV_FILE_URL) or [None, ""])[1]
            pinned = [SEC_ADV_FILE_URL] + ([u for u in stamp_urls(stamp)
                                            if u != SEC_ADV_FILE_URL] if stamp else [])
            best = try_urls(client, pinned)
            if best:
                return best
            log(f"pinned URL yielded no registered roster ({SEC_ADV_FILE_URL}); discovering")

        checked = 0
        for stamp in candidate_stamps():
            live: List[str] = []
            for url in stamp_urls(stamp, primary_only=True):
                time.sleep(PROBE_DELAY)
                checked += 1
                try:
                    if client.head(url, timeout=15).status_code == 200:
                        live.append(url)
                except Exception:
                    continue
            if not live:
                continue
            # This date exists. Now it is worth asking for the "_1" variants
            # too, cheaply, in case the SEC re-published a corrected file.
            for url in stamp_urls(stamp):
                if url not in live:
                    time.sleep(PROBE_DELAY)
                    checked += 1
                    try:
                        if client.head(url, timeout=15).status_code == 200:
                            live.append(url)
                    except Exception:
                        continue
            log(f"stamp {stamp}: {len(live)} file(s) present")
            best = try_urls(client, live)
            if best:
                return best
            log(f"stamp {stamp} had files but no registered roster — trying older date")

    raise RuntimeError(
        f"No registered adviser roster found after probing {checked} URLs. "
        "If every probe returned 403, sec.gov is rate-limiting this host."
    )


# ---------------------------------------------------------------------------
# Scoring — pure, unit-testable
# ---------------------------------------------------------------------------

def to_millions(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0.0
        return float(str(value).replace(",", "").replace("$", "").strip()) / 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


def to_count(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0.0
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def build_candidates(df, mapping: Dict[str, Optional[str]], f: Dict[str, Any],
                     stats: Optional[Dict[str, int]] = None) -> "pd.DataFrame":
    """
    Filter the roster to qualifying wealth-management practices.

    `stats` is filled in with the intermediate counts so the email can report
    them accurately — the in-state pool is NOT the national total, and saying
    otherwise puts a wrong number in front of the reader every week.
    """
    cols = mapping
    frame = df
    stats = stats if stats is not None else {}

    if cols.get("firm_type"):
        types = frame[cols["firm_type"]].astype(str).str.strip()
        frame = frame[~types.str.contains(r"exempt|\bERA\b", case=False, regex=True, na=False)]

    states = sorted({s for r in f["regions"] for s in REGION_STATES.get(r, [])})
    if states and cols.get("state"):
        frame = frame[frame[cols["state"]].astype(str).str.strip().str.upper().isin(states)]
    stats["state_pool"] = len(frame)

    def money(key):
        col = cols.get(key)
        if not col or col not in frame.columns:
            return frame[cols["aum"]].map(lambda _: 0.0)
        return frame[col].map(to_millions)

    def count(key):
        col = cols.get(key)
        if not col or col not in frame.columns:
            return frame[cols["aum"]].map(lambda _: 0.0)
        return frame[col].map(to_count)

    aum = frame[cols["aum"]].map(to_millions)
    safe = aum.where(aum > 0, other=1.0)
    frame = frame.assign(
        _aum=aum,
        _clients=count("indiv_clients") + count("hnw_clients"),
        _retail=((money("indiv_aum") + money("hnw_aum")) / safe * 100).clip(0, 100),
        _pension=(money("pension_aum") / safe * 100).clip(0, 100),
    )

    frame = frame[(frame["_aum"] >= f["aum_min"]) & (frame["_aum"] <= f["aum_max"])]
    stats["in_band"] = len(frame)

    frame = frame[(frame["_retail"] >= f["min_retail"]) & (frame["_clients"] >= f["min_clients"])]
    stats["dropped_funds"] = stats["in_band"] - len(frame)
    stats["qualifying"] = len(frame)
    return frame


def rank(frame, mapping, by: str, top_n: int) -> List[Dict[str, Any]]:
    col = {"aum": "_aum", "clients": "_clients", "pension": "_pension"}[by]
    ordered = frame.sort_values(col, ascending=False).head(top_n)
    out = []
    for _, row in ordered.iterrows():
        site = str(row.get(mapping.get("website") or "", "") or "").strip()
        if site.lower() in ("nan", "none"):
            site = ""
        if site and not site.lower().startswith(("http://", "https://")):
            site = "https://" + site
        out.append({
            "firm": str(row.get(mapping["firm_name"], "") or "").strip(),
            "city": str(row.get(mapping.get("city") or "", "") or "").strip().title(),
            "state": str(row.get(mapping.get("state") or "", "") or "").strip().upper(),
            "aum": row["_aum"],
            "clients": int(row["_clients"]),
            "retail": row["_retail"],
            "pension": row["_pension"],
            "crd": str(row.get(mapping.get("crd") or "", "") or "").strip(),
            "website": site,
        })
    return out


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def principal_html(row: Dict[str, Any]) -> str:
    """
    One line naming the likely owner, always carrying its confidence.

    Deliberately never renders a bare name. These people get cold-called, and a
    guess presented as a fact is worse than an empty cell.
    """
    p = row.get("principal") or {}
    name = str(p.get("name") or "").strip()
    if not name:
        return ("<div style='font:11px system-ui;color:#9ca3af'>"
                "no registered rep found in IAPD</div>")

    conf = p.get("confidence", "none")
    label = iapd.CONFIDENCE_LABEL.get(conf, "")
    color = iapd.CONFIDENCE_COLOR.get(conf, "#6b7280")
    since = p.get("industry_since")
    tenure = f" &middot; in industry since {since}" if since else ""
    url = p.get("profile_url") or ""
    shown = (f"<a href='{url}' style='color:{color};text-decoration:none'>{name}</a>"
             if url else name)
    return (f"<div style='font:12px system-ui;color:{color};margin-top:2px'>"
            f"<strong>{shown}</strong>{tenure}"
            f"<span style='color:#9ca3af'> &middot; {label}</span></div>")


def table_html(title: str, note: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return f"<h2 style='font:600 16px system-ui;margin:28px 0 4px'>{title}</h2><p style='font:14px system-ui;color:#6b7280'>No firms matched.</p>"
    head = (
        "<tr>" + "".join(
            f"<th style='text-align:{a};padding:8px 10px;border-bottom:2px solid #e5e7eb;"
            f"font:600 12px system-ui;color:#374151;text-transform:uppercase;letter-spacing:.04em'>{h}</th>"
            for h, a in [("Firm", "left"), ("Location", "left"), ("AUM", "right"),
                         ("Clients", "right"), ("Retail", "right"), ("Pension", "right")]
        ) + "</tr>"
    )
    body = ""
    for i, r in enumerate(rows):
        bg = "#ffffff" if i % 2 == 0 else "#f9fafb"
        name = f"<a href='{r['website']}' style='color:#111827;text-decoration:none'>{r['firm']}</a>" if r["website"] else r["firm"]
        body += (
            f"<tr style='background:{bg}'>"
            f"<td style='padding:8px 10px;font:14px system-ui;color:#111827'>{name}"
            f"{principal_html(r)}"
            f"<div style='font:11px system-ui;color:#9ca3af'>CRD {r['crd']}</div></td>"
            f"<td style='padding:8px 10px;font:13px system-ui;color:#4b5563'>{r['city']}, {r['state']}</td>"
            f"<td style='padding:8px 10px;font:14px system-ui;text-align:right;color:#111827'>${r['aum']:,.1f}M</td>"
            f"<td style='padding:8px 10px;font:14px system-ui;text-align:right;color:#111827'>{r['clients']:,}</td>"
            f"<td style='padding:8px 10px;font:14px system-ui;text-align:right;color:#4b5563'>{r['retail']:.0f}%</td>"
            f"<td style='padding:8px 10px;font:14px system-ui;text-align:right;color:#4b5563'>{r['pension']:.0f}%</td>"
            f"</tr>"
        )
    return (
        f"<h2 style='font:600 16px system-ui;margin:28px 0 2px;color:#111827'>{title}</h2>"
        f"<p style='font:13px system-ui;color:#6b7280;margin:0 0 10px'>{note}</p>"
        f"<table style='width:100%;border-collapse:collapse;border:1px solid #e5e7eb'>{head}{body}</table>"
    )


def build_email(lists: Dict[str, List[Dict[str, Any]]], meta: Dict[str, Any]) -> str:
    overlap = set(r["crd"] for r in lists["aum"]) & set(r["crd"] for r in lists["clients"]) & set(r["crd"] for r in lists["pension"])
    return f"""<div style="max-width:760px;margin:0 auto;padding:24px;background:#fff">
<h1 style="font:700 20px system-ui;color:#111827;margin:0 0 2px">SFG Acquisition Targets</h1>
<p style="font:13px system-ui;color:#6b7280;margin:0 0 4px">
Week of {meta['date']} &middot; {meta['regions']} &middot; ${meta['aum_min']:,.0f}M–${meta['aum_max']:,.0f}M
</p>
<p style="font:13px system-ui;color:#6b7280;margin:0 0 18px">
{meta['state_pool']:,} registered advisers in these states ({meta['universe']:,} nationally).
{meta['in_band']:,} fall inside the AUM band; {meta['dropped_funds']:,} of those were dropped as fund managers
(under {meta['min_retail']:.0f}% of AUM from individual clients, or fewer than {meta['min_clients']} clients),
leaving <strong>{meta['qualifying']:,} qualifying practices</strong>.
AUM, client counts and client mix are as filed on Form ADV.
</p>
{table_html("Ranked by AUM", "Largest practices inside the band. Tends to cluster at the ceiling.", lists["aum"])}
{table_html("Ranked by client count", "Most individual client relationships — the deepest books.", lists["clients"])}
{table_html("Ranked by retirement focus", "Highest share of AUM from pension and profit-sharing plans — closest to SFG's model.", lists["pension"])}
<p style="font:13px system-ui;color:#374151;margin:24px 0 0;padding-top:14px;border-top:1px solid #e5e7eb">
<strong>{len(overlap)}</strong> firm(s) appear in all three rankings.
</p>
<p style="font:11px system-ui;color:#9ca3af;margin:8px 0 0">
Source: SEC Form ADV, {meta['source']}. Figures are self-reported by each adviser.
Names come from the firm's registered investment adviser representatives on IAPD.
Form ADV does not disclose ownership, so the principal is inferred &mdash;
<span style="color:{iapd.CONFIDENCE_COLOR['high']}">green</span> means the surname matches
the firm name or the firm has one or two reps;
<span style="color:{iapd.CONFIDENCE_COLOR['medium']}">amber</span> is the longest-tenured of a
small team; grey is unresolved. Verify before you call.
"In industry since" is the year that person entered the securities industry anywhere,
not the year the firm was founded.
</p>
</div>"""


def send_email(html: str, subject: str) -> None:
    api_key = os.getenv("RESEND_API_KEY", "").strip()
    to = os.getenv("REPORT_TO", "").strip()
    sender = os.getenv("REPORT_FROM", "SFG Acquisition Agent <onboarding@resend.dev>").strip()
    if not api_key or not to:
        raise RuntimeError("RESEND_API_KEY and REPORT_TO must be set.")

    resp = httpx.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"from": sender, "to": [to], "subject": subject, "html": html},
        timeout=30,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Resend returned {resp.status_code}: {resp.text[:400]}")
    log(f"sent: {resp.json().get('id')}")


# ---------------------------------------------------------------------------

def main() -> int:
    filters = {
        "regions": [r.strip() for r in os.getenv("REPORT_REGIONS", "Southwest").split(",") if r.strip()],
        "aum_min": float(os.getenv("REPORT_AUM_MIN", "50")),
        "aum_max": float(os.getenv("REPORT_AUM_MAX", "250")),
        "min_retail": float(os.getenv("REPORT_MIN_RETAIL", "50")),
        "min_clients": int(os.getenv("REPORT_MIN_CLIENTS", "25")),
    }
    top_n = int(os.getenv("REPORT_TOP_N", "10"))

    log(f"filters: {filters}")
    roster = load_roster()
    log(f"roster: {roster['source'].rsplit('/', 1)[-1]} ({roster['registered']:,} registered)")

    stats: Dict[str, int] = {}
    qualifying = build_candidates(roster["df"], roster["mapping"], filters, stats)
    log(f"in-state: {stats['state_pool']:,} | in-band: {stats['in_band']:,} | "
        f"dropped as funds: {stats['dropped_funds']:,} | qualifying: {stats['qualifying']:,}")

    if qualifying.empty:
        log("no qualifying firms — sending nothing")
        return 0

    lists = {k: rank(qualifying, roster["mapping"], k, top_n) for k in ("aum", "clients", "pension")}

    # Principal enrichment. Only the firms actually appearing in the email get
    # looked up — the three lists overlap heavily, so this is typically ~20
    # requests, not 361. Failure here degrades the email, it never kills it.
    if os.getenv("REPORT_NO_PEOPLE", "").strip() != "1":
        crds = [r["crd"] for lst in lists.values() for r in lst if r.get("crd")]
        try:
            cache = iapd.enrich_crds(crds, log=log)
            for lst in lists.values():
                for r in lst:
                    r["principal"] = iapd.principal_for(r["firm"], r["crd"], cache)
            named = {r["crd"] for lst in lists.values() for r in lst
                     if (r.get("principal") or {}).get("name")}
            high = {r["crd"] for lst in lists.values() for r in lst
                    if (r.get("principal") or {}).get("confidence") == "high"}
            log(f"principals: {len(named)} named, {len(high)} high-confidence, "
                f"of {len(set(crds))} firms")
        except Exception as e:
            log(f"IAPD enrichment skipped: {e}")

    meta = {
        "date": datetime.now(timezone.utc).strftime("%d %b %Y"),
        "regions": ", ".join(filters["regions"]),
        "aum_min": filters["aum_min"], "aum_max": filters["aum_max"],
        "min_retail": filters["min_retail"], "min_clients": filters["min_clients"],
        "universe": roster["registered"],
        "state_pool": stats["state_pool"], "in_band": stats["in_band"],
        "dropped_funds": stats["dropped_funds"], "qualifying": stats["qualifying"],
        "source": roster["source"].rsplit("/", 1)[-1],
    }
    html = build_email(lists, meta)

    if os.getenv("REPORT_DRY_RUN", "").strip() == "1":
        print(html)
        return 0

    send_email(html, f"SFG acquisition targets — week of {meta['date']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FAILED: {exc}")
        sys.exit(1)
