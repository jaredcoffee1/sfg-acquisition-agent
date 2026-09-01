"""
SFG Acquisition Research Agent

A two-stage AI agent system for researching RIA acquisition targets:
1. Research Agent: Gathers data via ScrapeGraph MCP (scrape, extract, search)
2. Acquisition Analyst Agent: Synthesizes into deal assessment & scorecard

Usage:
    streamlit run agent_sfg_acquisition.py
"""

import asyncio
import os
import re
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import streamlit as st
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import httpx
import pandas as pd
import csv

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ScrapeGraph configuration
SGAI_API_KEY = os.getenv("SGAI_API_KEY")
SGAI_API_URL = os.getenv("SGAI_API_URL", "https://v2-api.scrapegraphai.com/api")
SGAI_TIMEOUT = int(os.getenv("SGAI_TIMEOUT", "120"))

# Configure Streamlit page
st.set_page_config(
    page_title="SFG Acquisition Research Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SCRAPEGRAPH MCP CLIENT WRAPPER
# ============================================================================

class ScrapeGraphMCPClient:
    """Wrapper for ScrapeGraph API calls (MCP-compatible)"""

    def __init__(self, api_key: str, base_url: str = SGAI_API_URL, timeout: int = SGAI_TIMEOUT):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.headers = {
            "SGAI-APIKEY": api_key,
            "Content-Type": "application/json"
        }

    async def scrape(self, website_url: str, output_format: str = "markdown") -> Dict[str, Any]:
        """Scrape a website and convert to markdown/html/screenshot"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "url": website_url,
                "formats": [output_format]
            }
            try:
                response = await client.post(
                    f"{self.base_url}/scrape",
                    json=payload,
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "success": False}

    async def extract(self, website_url: str, user_prompt: str, output_schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract structured data from a website using AI"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "url": website_url,
                "prompt": user_prompt,
                "mode": "normal"
            }
            if output_schema:
                payload["output_schema"] = output_schema
            try:
                response = await client.post(
                    f"{self.base_url}/extract",
                    json=payload,
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "success": False}

    async def search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Perform web search via ScrapeGraph"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "search_query": query,
                "num_results": min(num_results, 20)  # Clamp to 20
            }
            try:
                response = await client.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "success": False}


# ============================================================================
# SCRAPEGRAPH v2 RESPONSE HANDLING
# ----------------------------------------------------------------------------
# The v2 API wraps every response as {status, data, error, elapsedMs}. The
# "error" key is ALWAYS present and is an empty string on success, so success
# must be tested on the VALUE, never on key presence (`"error" in resp` is
# always True and silently discards every result).
# ============================================================================

NO_RESEARCH_SENTINEL = "__NO_RESEARCH_DATA__"


def sg_failed(resp: Any) -> Optional[str]:
    """Return an error message if the response failed, else None."""
    if not isinstance(resp, dict):
        return f"unexpected response type: {type(resp).__name__}"
    err = resp.get("error")
    if err:
        return str(err)
    status = str(resp.get("status", "")).lower()
    if status in ("failed", "error"):
        return f"status={status}"
    return None


def sg_body(resp: Dict[str, Any]) -> Any:
    """Unwrap the payload envelope: prefer resp['data'], fall back to resp."""
    if isinstance(resp, dict) and isinstance(resp.get("data"), (dict, list)):
        return resp["data"]
    return resp


def coerce_json(payload: Any) -> Dict[str, Any]:
    """Turn a payload into a dict. Handles dicts, JSON strings, fenced JSON."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1]
            text = re.sub(r"^\s*json\s*", "", text, flags=re.IGNORECASE).strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def sg_extracted(resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull structured data out of an /api/extract response.
    Docs describe the payload as data.json_data (Python) / data.json (JS);
    older shapes used 'result' or 'content'. Try each, then fall back to the
    body itself, so a doc change degrades instead of silently emptying.
    """
    body = sg_body(resp)
    if isinstance(body, dict):
        for key in ("json_data", "json", "result", "content", "output"):
            if key in body:
                parsed = coerce_json(body[key])
                if parsed:
                    return parsed
        # The body may already BE the structured data.
        noise = {"status", "error", "elapsedMs", "request_id", "website_url"}
        if any(k not in noise for k in body):
            return {k: v for k, v in body.items() if k not in noise}
    return {}


def sg_markdown(resp: Dict[str, Any], url: str = "") -> str:
    """Pull markdown out of an /api/scrape response (results keyed by format)."""
    body = sg_body(resp)
    if not isinstance(body, dict):
        return ""
    results = body.get("results") if isinstance(body.get("results"), dict) else body
    md = results.get("markdown") if isinstance(results, dict) else None
    if isinstance(md, dict):
        return str(md.get("data") or "")
    if isinstance(md, str):
        return md
    legacy = body.get(f"{url}_markdown")
    return str(legacy or "")


def sg_sources(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull the ranked result list out of an /api/search response."""
    body = sg_body(resp)
    if isinstance(body, list):
        return [s for s in body if isinstance(s, dict)]
    if isinstance(body, dict):
        for key in ("results", "sources", "items"):
            val = body.get(key)
            if isinstance(val, list):
                return [s for s in val if isinstance(s, dict)]
    return []


def parse_aum(aum_str: str) -> tuple:
    """
    Parse an AUM string into (millions_as_float, unit).

    Always normalises to MILLIONS so sorting is correct. The previous version
    stored the raw number with its unit, so "$1.2B" sorted as 1.2 and ranked
    BELOW "$125M" — silently burying the largest firms.
    """
    if not aum_str:
        return 0.0, "M"
    text = str(aum_str).replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(billion|bn|b|million|mm|m|k)?", text, re.IGNORECASE)
    if not match:
        return 0.0, "M"
    value = float(match.group(1))
    unit = (match.group(2) or "M").lower()
    if unit in ("billion", "bn", "b"):
        return value * 1000.0, "B"
    if unit == "k":
        return value / 1000.0, "M"
    return value, "M"


def sg_debug(label: str, resp: Any) -> None:
    """Surface the raw response so a shape mismatch is visible, not silent."""
    with st.expander(f"🔧 Raw response — {label}", expanded=False):
        st.json(resp if isinstance(resp, (dict, list)) else {"repr": repr(resp)})


# ============================================================================
# SEC FORM ADV DATA SOURCE
# ----------------------------------------------------------------------------
# The SEC publishes a monthly report of every registered investment adviser,
# with columns keyed to Form ADV items: firm name (Item 1A), principal office
# city/state (Item 1F), website (Item 1I) and regulatory AUM (Item 5F(2)(c)).
#
# This is the authoritative RIA universe. A marketing site saying "over $100
# million" is copy; the ADV figure is a filed number with a date on it.
#
# Column headings are NOT hardcoded here. They are detected at runtime from the
# real file and shown in the UI, because guessing a schema is what produced the
# silent empty rows this module replaces.
# ============================================================================

SEC_IA_LANDING = (
    "https://www.sec.gov/data-research/sec-markets-data/"
    "information-about-registered-investment-advisers-exempt-reporting-advisers"
)

# The SEC requires a descriptive User-Agent with contact info for automated
# access, else it returns 403. Set SEC_USER_AGENT in Railway; do not commit a
# personal email to a public repo.
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "SFG Acquisition Research Agent (set SEC_USER_AGENT env var with contact email)"
)

# Patterns are ordered: the first match wins. Form ADV item numbers are the
# most reliable signal; the plain-language fallbacks cover header renames.
ADV_COLUMN_PATTERNS = {
    "firm_name":  [r"primary business name", r"^1A\b", r"legal name", r"\bfirm name\b"],
    "crd":        [r"organization\s*crd", r"\bcrd\s*#", r"\bcrd\b"],
    "city":       [r"main office.*city", r"1F\(1\).*city", r"\bcity\b"],
    "state":      [r"main office.*state", r"1F\(1\).*state", r"\bstate\b"],
    # "Website Address" holds the URL; a bare "1I" column is the yes/no flag,
    # so it must never win — matching it fed junk URLs to the enrichment step.
    "website":    [r"website address", r"web\s*site\s*address", r"^website$"],
    "aum":        [
        r"5F\(2\)\(c\)",
        r"5F\(2\)c",
        r"total.*regulatory assets under management",
        r"regulatory assets under management",
        r"total.*assets under management",
        r"assets under management",
        r"\bRAUM\b",
    ],
    "employees":  [r"^5A$", r"5A\b", r"number of employees"],
    # Discriminates registered advisers from Exempt Reporting Advisers. ERAs
    # file an abbreviated ADV with NO Item 5.F, so they can never yield AUM —
    # and they are VC/PE funds, not wealth managers, so they are not targets.
    "firm_type":  [r"^firm type$", r"firm type"],
}


# Region definitions are a BUSINESS choice, not a technical one — they decide
# your entire candidate universe. Edit these deliberately.
REGION_STATES = {
    "Southwest": ["AZ", "NM", "NV", "UT", "TX", "OK"],
    "Mountain West": ["CO", "MT", "ID", "WY", "UT", "NV"],
    "Southeast": ["FL", "GA", "NC", "SC", "TN", "AL", "MS", "KY", "VA", "WV", "AR", "LA"],
    "Midwest": ["OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"],
    "Northeast": ["NY", "NJ", "PA", "MA", "CT", "RI", "NH", "VT", "ME", "MD", "DE"],
    "West Coast": ["CA", "OR", "WA"],
    "National": [],  # empty = no state filter
}


def detect_adv_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    """Map our field names onto whatever headings this month's file actually uses."""
    mapping: Dict[str, Optional[str]] = {}
    for field, patterns in ADV_COLUMN_PATTERNS.items():
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


SEC_FILE_BASE = (
    "https://www.sec.gov/files/investment/data/other/"
    "information-about-registered-investment-advisers-exempt-reporting-advisers/"
)

# sec.gov serves the DATA FILES to automated clients, but Akamai returns 403 on
# the HTML landing page from datacenter IPs. So the landing page is only a
# best-effort hint; the reliable path is addressing the file directly.
# NOTE: no explicit Host header. httpx derives it from the URL, and hardcoding
# one here broke every reports.adviserinfo.sec.gov request with a 403 because
# the header said www.sec.gov. These headers are reused across both hosts.
SEC_REQUEST_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
}


class AdvDataError(RuntimeError):
    """Raised when the SEC dataset cannot be loaded.

    Raised rather than returned so st.cache_data does NOT memoise the failure —
    caching an error for a week would leave the app stuck on a transient outage.
    """

    def __init__(self, message: str, columns: Optional[List[str]] = None,
                 mapping: Optional[Dict[str, Optional[str]]] = None):
        super().__init__(message)
        self.columns = columns
        self.mapping = mapping


# Pin the exact file and skip discovery entirely. Set this in Railway once the
# app reports which URL it resolved — brute-force probing repeatedly tripped
# the SEC's rate limiter, and a pinned URL is deterministic.
SEC_ADV_FILE_URL = os.getenv("SEC_ADV_FILE_URL", "").strip()

# The SEC asks automated clients to stay under ~10 requests/second.
SEC_PROBE_DELAY = 0.15


def candidate_adv_urls(months_back: int = 3, max_day: int = 10) -> List[str]:
    """
    Build direct file URLs newest-first.

    The SEC embeds the publication date in the filename (ia08032026_0.zip) and
    publishes early each month, so probing the first couple of weeks of recent
    months finds the current file without touching the blocked landing page.
    """
    urls: List[str] = []
    today = datetime.now()
    year, month = today.year, today.month

    # Only .zip variants. The .xlsx was verified NOT to exist (404 on both
    # naming forms), and probing for it doubled request volume for nothing —
    # which is how the SEC's ~10 req/s limiter got tripped.
    suffixes = ["_0.zip", ".zip"]

    for _ in range(months_back):
        for day in range(max_day, 0, -1):
            stamp = f"{month:02d}{day:02d}{year}"
            for suffix in suffixes:
                urls.append(f"{SEC_FILE_BASE}ia{stamp}{suffix}")
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    return urls


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def load_adv_dataset() -> Dict[str, Any]:
    """
    Download and parse the latest SEC investment adviser report.

    Cached for a week (the SEC publishes monthly and the file is large).
    Failures raise AdvDataError so they are never cached.
    """
    import io
    import zipfile

    found_url: Optional[str] = None
    content: Optional[bytes] = None
    probe_errors: List[str] = []

    with httpx.Client(timeout=180, follow_redirects=True, headers=SEC_REQUEST_HEADERS) as client:
        # Strategy 0: a pinned URL. Deterministic, one request, no probing.
        if SEC_ADV_FILE_URL:
            found_url = SEC_ADV_FILE_URL

        # Strategy 1: the landing page, if this host is allowed to read it.
        try:
            landing = client.get(SEC_IA_LANDING) if not found_url else None
            if landing is not None and landing.status_code == 200:
                links = re.findall(
                    r'href="([^"]*ia\d{8}[^"]*\.(?:zip|xlsx))"', landing.text, re.IGNORECASE
                )
                links = [l for l in links if "exempt" not in l.lower()]
                if links:
                    def file_date(url: str):
                        m = re.search(r"ia(\d{2})(\d{2})(\d{4})", url)
                        return (m.group(3), m.group(1), m.group(2)) if m else ("0", "0", "0")
                    newest = sorted(links, key=file_date)[-1]
                    found_url = newest if newest.startswith("http") else "https://www.sec.gov" + newest
            elif landing is not None:
                probe_errors.append(f"landing page {landing.status_code}")
        except Exception as e:
            probe_errors.append(f"landing page error: {e}")

        # Strategy 2: address files directly. Collect EVERY variant that exists
        # for the newest available date — the .zip is an identity-only roster
        # and the .xlsx is the fuller report, so we need to inspect both rather
        # than stopping at the first file that happens to exist.
        found_urls: List[str] = [found_url] if found_url else []
        if not found_urls:
            head_blocked = False
            newest_stamp: Optional[str] = None

            statuses: Dict[str, int] = {}
            for url in candidate_adv_urls():
                stamp_match = re.search(r"ia(\d{8})", url)
                stamp = stamp_match.group(1) if stamp_match else ""
                # Once a date has yielded files, don't walk back to older dates.
                if newest_stamp and stamp != newest_stamp:
                    break

                # Throttle: the SEC limits automated clients to roughly 10
                # requests/second and answers 403 past that. Un-throttled
                # probing is what made this step fail intermittently.
                time.sleep(SEC_PROBE_DELAY)
                try:
                    exists = False
                    if not head_blocked:
                        head = client.head(url, timeout=15)
                        statuses[url.rsplit("/", 1)[-1]] = head.status_code
                        if head.status_code == 200:
                            exists = True
                        elif head.status_code in (403, 405, 501):
                            head_blocked = True
                        else:
                            continue

                    if head_blocked:
                        probe = client.get(url, headers={"Range": "bytes=0-0"}, timeout=15)
                        statuses[url.rsplit("/", 1)[-1]] = probe.status_code
                        exists = probe.status_code in (200, 206)

                    if exists:
                        found_urls.append(url)
                        newest_stamp = stamp
                except Exception as e:
                    statuses[url.rsplit("/", 1)[-1]] = -1
                    probe_errors.append(str(e)[:80])
                    continue

            if head_blocked:
                probe_errors.append("HEAD refused by sec.gov; used ranged GET")

            if not found_urls and statuses:
                # Show what the SEC actually answered, so a rate-limit (403 on
                # everything) is distinguishable from "file genuinely absent"
                # (404s) without another guessing round.
                seen = {}
                for name, code in statuses.items():
                    seen.setdefault(code, []).append(name)
                probe_errors.append(
                    "responses: " + ", ".join(
                        f"HTTP {code} x{len(names)}" for code, names in sorted(seen.items())
                    )
                )

        if not found_urls:
            raise AdvDataError(
                "Could not locate a current SEC adviser data file. "
                "If the responses below are all 403, the SEC is rate-limiting this host — "
                "set SEC_ADV_FILE_URL in Railway to pin the file and skip discovery. "
                + ("Probe notes: " + "; ".join(probe_errors[-3:]) if probe_errors else "")
            )

        # Both "_0.zip" and ".zip" exist for a given date and hold DIFFERENT
        # rosters (registered vs exempt-only). A HEAD probe that gets throttled
        # can hide one of them, so once a date is known, ask for every variant
        # outright — a 404 costs one cheap request and is handled below.
        expanded: List[str] = list(found_urls)
        for url in list(found_urls):
            stamp_match = re.search(r"ia(\d{8})", url)
            if not stamp_match:
                continue
            for suffix in ("_0.zip", ".zip"):
                sibling = f"{SEC_FILE_BASE}ia{stamp_match.group(1)}{suffix}"
                if sibling not in expanded:
                    expanded.append(sibling)
        found_urls = expanded

        downloads: List[tuple] = []  # (url, bytes)
        for url in found_urls:
            try:
                resp = client.get(url)
                if resp.status_code == 200:
                    downloads.append((url, resp.content))
                else:
                    probe_errors.append(f"{url.rsplit('/', 1)[-1]} -> HTTP {resp.status_code}")
            except Exception as e:
                probe_errors.append(f"{url.rsplit('/', 1)[-1]} -> {e}")

        if not downloads:
            raise AdvDataError(
                "Found candidate SEC files but none downloaded. "
                + "; ".join(probe_errors[:4])
            )

        found_url = downloads[0][0]
        content = downloads[0][1]

    def read_member(raw: bytes, name: str) -> "pd.DataFrame":
        if name.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw), low_memory=False, encoding_errors="replace")
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")

    # The SEC ships the report split across several files — Item 1-3 in one,
    # Item 5 (AUM) in another. Taking the first member blindly is why AUM came
    # back NULL. Pick the member that actually carries the fields we need, and
    # join the rest on CRD.
    frames: List[tuple] = []  # (name, df, mapping)
    for url, blob in downloads:
        label = url.rsplit("/", 1)[-1]
        if url.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    members = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".csv"))]
                    for name in members[:8]:  # bound the work
                        try:
                            with zf.open(name) as fh:
                                raw = fh.read()
                            d = read_member(raw, name)
                        except Exception as e:
                            frames.append((f"{label}:{name}", None, {"_read_error": str(e)}))
                            continue
                        frames.append((f"{label}:{name}", d, detect_adv_columns(list(d.columns))))
            except Exception as e:
                frames.append((label, None, {"_read_error": f"zip error: {e}"}))
        else:
            try:
                d = pd.read_excel(io.BytesIO(blob), engine="openpyxl")
                frames.append((label, d, detect_adv_columns(list(d.columns))))
            except Exception as e:
                frames.append((label, None, {"_read_error": str(e)}))

    if not any(d is not None for (_, d, _) in frames):
        raise AdvDataError(
            "Downloaded SEC file(s) but could not parse any of them.",
            columns=[f"{n}: {m.get('_read_error', 'unknown error')}" for (n, d, m) in frames if d is None],
        )

    usable = [(n, d, m) for (n, d, m) in frames if d is not None]

    # The roster is identity-only by design — AUM comes per-firm from each ADV
    # filing (see fetch_firm_raum). So CRD is what we require here: it is the
    # key that lets us look the filing up.
    qualified = [(n, d, m) for (n, d, m) in usable
                 if m.get("firm_name") and m.get("state") and m.get("crd")]

    # Confirmed the hard way: the SEC publishes two rosters under near-identical
    # names, and one contains ONLY Exempt Reporting Advisers (VC/PE funds that
    # file no Item 5.F and are not acquisition targets). Both have identical
    # columns, so choose on CONTENT: the file with actual registered advisers.
    def registered_count(d, m) -> int:
        col = m.get("firm_type")
        if not col or col not in d.columns:
            return len(d)  # no firm-type column: cannot tell, assume usable
        types = d[col].astype(str).str.strip()
        return int((~types.str.contains(r"exempt|\bERA\b", case=False, regex=True, na=False)).sum())

    scored = sorted(
        ((registered_count(d, m), n, d, m) for (n, d, m) in qualified),
        key=lambda t: -t[0],
    )

    if scored and scored[0][0] > 0:
        n_reg, name, df, mapping = scored[0]
        return {"df": df, "mapping": mapping, "source_url": found_url,
                "rows": len(df), "member": name, "registered": n_reg,
                "candidates": {n: c for (c, n, _, _) in scored}}

    # Every downloaded roster was exempt-only. Report the per-file breakdown so
    # the correct URL is identified from data rather than guessed at again.
    breakdown: List[str] = []
    for (n, d, m) in usable:
        col = m.get("firm_type")
        if col and col in d.columns:
            counts = d[col].astype(str).str.strip().value_counts().to_dict()
            breakdown.append(f"{n}: rows={len(d)} firm_types={counts}")
        else:
            breakdown.append(f"{n}: rows={len(d)} (no firm_type column)")

    raise AdvDataError(
        "Every SEC roster downloaded contained only exempt-reporting advisers. "
        "The registered-adviser roster is a different file — see the per-file "
        "breakdown below and pin the right one via SEC_ADV_FILE_URL.",
        columns=[f"downloaded: {u}" for (u, _) in downloads] + breakdown,
        mapping={n: (m if isinstance(m, dict) else {}) for (n, _, m) in frames},
    )


# ----------------------------------------------------------------------------
# Per-firm regulatory AUM, from the ADV filing itself.
#
# Confirmed by inspection: AUM is absent from the SEC bulk roster (172 columns,
# Items 1/2/3/6/7 only), the .xlsx variant does not exist, and the IAPD firm
# API returns no monetary fields. The filing PDF is the only free source.
# ----------------------------------------------------------------------------

ADV_PDF_URL = "https://reports.adviserinfo.sec.gov/reports/ADV/{crd}/PDF/{crd}.pdf"

# Item 5.F(2) reports three figures: (a) discretionary, (b) non-discretionary,
# (c) total. We want (c). Layout varies between filings, so several shapes are
# tried and the raw snippet is kept for display — this heuristic has NOT been
# validated against a live filing, so it must show its work rather than assert.
RAUM_PATTERNS = [
    r"\(c\)\s*Total[^$\n]{0,40}\$?\s*([\d,]{4,})",
    r"Total\s*(?:Amount)?[^$\n]{0,30}\$\s*([\d,]{4,})",
    r"regulatory assets under management[^$]{0,200}?\$\s*([\d,]{4,})",
]


def parse_raum_from_text(text: str) -> tuple:
    """
    Extract total regulatory AUM (dollars) from ADV Part 1A text.

    Returns (dollars, snippet). dollars is 0.0 when nothing convincing is found,
    and the snippet is always returned so a wrong parse is visible on screen
    instead of silently becoming a number you might act on.
    """
    if not text:
        return 0.0, ""

    flat = re.sub(r"[ \t]+", " ", text)
    anchor = re.search(r"regulatory assets under management", flat, re.IGNORECASE)
    window = flat[max(0, anchor.start() - 400): anchor.start() + 1500] if anchor else flat[:4000]

    for pattern in RAUM_PATTERNS:
        m = re.search(pattern, window, re.IGNORECASE)
        if m:
            try:
                value = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            # An RIA's RAUM is at least six figures; anything smaller is a
            # mis-capture (a count of accounts, a page number, a year).
            if value >= 100_000:
                return value, window[:600].strip()

    return 0.0, window[:600].strip()


def fetch_firm_raum(crd: str, client: "httpx.Client") -> tuple:
    """Download one firm's ADV filing and pull its total regulatory AUM."""
    import io
    try:
        import pdfplumber
    except ImportError:
        return 0.0, "pdfplumber not installed — add it to requirements.txt"

    if not crd or not str(crd).strip().isdigit():
        return 0.0, f"no usable CRD ({crd!r})"

    url = ADV_PDF_URL.format(crd=str(crd).strip())
    # Served from reports.adviserinfo.sec.gov, not www.sec.gov. Ask for a PDF
    # and present a Referer from the IAPD site it is normally linked from.
    pdf_headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept": "application/pdf,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://adviserinfo.sec.gov/firm/summary/{str(crd).strip()}",
    }
    try:
        resp = client.get(url, timeout=60, headers=pdf_headers)
        if resp.status_code != 200:
            return 0.0, f"ADV fetch HTTP {resp.status_code} for {url}"

        text_parts: List[str] = []
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            # Item 5 sits early in Part 1A; reading the whole filing (which can
            # run to hundreds of pages for large firms) is unnecessary.
            for page in pdf.pages[:14]:
                text_parts.append(page.extract_text() or "")
                if any("regulatory assets under management" in t.lower() for t in text_parts):
                    # Grab one more page in case the figures wrap.
                    continue
        full_text = "\n".join(text_parts)

        # Belt and braces alongside the roster's Firm Type filter: an ERA
        # filing has no Item 5.F at all, so say so plainly rather than
        # reporting it as an unexplained parse failure.
        if re.search(r"EXEMPT REPORTING ADVISER|ERA Report", full_text, re.IGNORECASE):
            return 0.0, "exempt-reporting adviser — files no Item 5.F"

        return parse_raum_from_text(full_text)
    except Exception as e:
        return 0.0, f"ADV parse error: {e}"


def adv_to_millions(value: Any) -> float:
    """ADV reports regulatory AUM in whole dollars; convert to millions."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0.0
        num = float(str(value).replace(",", "").replace("$", "").strip())
        return num / 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


# ============================================================================
# DATA MODELS FOR ACQUISITION CURATOR
# ============================================================================

@dataclass
class RIATarget:
    """Represents a curated RIA acquisition target with full research data"""
    firm_name: str           # Company name
    city: str                # City location
    state: str               # State (2-letter code)
    aum: str                 # AUM string (e.g., "$125M")
    aum_numeric: float       # Numeric AUM for sorting (125.0)
    aum_unit: str            # Unit (M, B)

    founder_name: str = ""           # Founder's name
    founder_title: str = ""          # Founder's title/role
    founder_info: str = ""           # Additional founder background
    ceo_name: str = ""               # Current CEO
    ceo_info: str = ""               # CEO background

    website: str = ""                # Firm website URL
    research_status: str = "Not Started"  # Not Started, In Progress, Complete, Contacted
    confidence_score: float = 0.0    # 0.0-1.0 confidence in data
    notes: str = ""                  # User research notes
    last_updated: str = ""           # ISO 8601 timestamp
    created_at: str = ""             # ISO 8601 timestamp
    source: str = ""                 # How we found them (SEC EDGAR, Web Search, Manual)


# ============================================================================
# ACQUISITION CURATOR AGENT
# ============================================================================

class CuratorAgent:
    """
    Actively searches for and curates RIA acquisition targets.
    Orchestrates multi-stage pipeline: search → extract → enrich → compile.
    """

    def __init__(self, scrapegraph_client: ScrapeGraphMCPClient, openai_client: OpenAI, company_context: Dict[str, str]):
        self.scraper = scrapegraph_client
        self.llm = openai_client
        self.context = company_context
        self.curator_file = "riaTargets_curated.csv"

    # ========================================================================
    # Core Research Methods
    # ========================================================================

    async def search_ria_universe(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for RIA candidates matching criteria.

        Args:
            filters: {
                "regions": List[str],      # ["Southwest", "Southeast", etc.]
                "aum_min": float,          # Minimum AUM in millions
                "aum_max": float,          # Maximum AUM in millions
                "firm_types": List[str]    # ["Independent RIA", etc.]
            }

        Returns:
            List of dicts with keys: {firm_name, website, state, aum_estimate, source}
        """
        try:
            dataset = load_adv_dataset()
        except AdvDataError as e:
            st.error(f"❌ {e}")
            if e.columns:
                with st.expander("🔧 Column headings found in the SEC file", expanded=True):
                    # Plain text, not st.write — a Python list gets collapsed
                    # into "[0-100]" ranges and hides the names we need to read.
                    st.text("\n".join(str(c) for c in e.columns))
            if e.mapping:
                with st.expander("🔧 Detected mapping per file", expanded=True):
                    st.json(e.mapping)
            return []
        except Exception as e:
            st.error(f"❌ Unexpected error loading SEC data: {e}")
            return []

        df = dataset["df"]
        cols = dataset["mapping"]

        # Resolve selected regions to a concrete state list.
        wanted_states: List[str] = []
        for region in filters.get("regions", []):
            wanted_states.extend(REGION_STATES.get(region, []))
        wanted_states = sorted(set(wanted_states))

        frame = df
        if wanted_states and cols.get("state"):
            states_norm = frame[cols["state"]].astype(str).str.strip().str.upper()
            frame = frame[states_norm.isin(wanted_states)]

        state_pool = len(frame)
        if state_pool == 0:
            st.warning("No registered advisers in the selected states.")
            return []

        # Drop Exempt Reporting Advisers. Confirmed from a live filing: an ERA
        # files an "SEC ERA Report" with no Item 5.F, so its AUM is unreadable
        # by construction — and a VC fund is not an acquisition target anyway.
        era_dropped = 0
        if cols.get("firm_type"):
            types = frame[cols["firm_type"]].astype(str).str.strip()
            with st.expander("🔧 Firm Type values in the roster", expanded=False):
                st.write(types.value_counts().to_dict())
            keep = ~types.str.contains(r"exempt|\bERA\b", case=False, regex=True, na=False)
            era_dropped = int((~keep).sum())
            frame = frame[keep]
            if len(frame) == 0:
                st.error(
                    "Every adviser in these states was classified as exempt-reporting. "
                    "Check the Firm Type values above — the filter may be too broad."
                )
                return []

        # Cheap pre-filter before spending a PDF fetch per firm: Item 1.O flags
        # advisers at $1B or more, which are above the target range. SEC
        # registration generally implies $100M+, so this brackets the pool.
        big_col = next((c for c in frame.columns
                        if re.search(r"1O.*amount of assets", str(c), re.IGNORECASE)), None)
        aum_max = float(filters.get("aum_max", 10 ** 9))
        excluded_big = 0
        if big_col is not None and aum_max < 1000:
            flagged = frame[big_col].astype(str).str.strip()
            keep = ~flagged.str.contains(r"\d", regex=True, na=False)
            excluded_big = int((~keep).sum())
            frame = frame[keep]

        limit = int(filters.get("limit", 10))
        # Bounded pool: each firm costs one ADV PDF download and parse, so this
        # is deliberately capped rather than run across the whole state pool.
        pool_size = min(len(frame), max(limit * 3, limit + 10), 45)
        pool = frame.head(pool_size)

        st.caption(
            f"{state_pool:,} advisers in selected states ({dataset['rows']:,} in the roster)"
            + (f"; {era_dropped:,} exempt-reporting advisers dropped" if era_dropped else "")
            + (f"; {excluded_big:,} excluded as $1B+" if excluded_big else "")
            + f". Reading Item 5.F from {pool_size} ADV filings."
        )
        # Printed so this URL can be pinned via SEC_ADV_FILE_URL, removing the
        # discovery step that keeps hitting the SEC's rate limiter.
        st.caption(
            f"Roster source: {dataset.get('member', '?')} "
            f"({dataset.get('registered', '?')} registered advisers). "
            f"Files considered: {dataset.get('candidates', {})}"
        )

        aum_min = float(filters.get("aum_min", 0))
        rows: List[Dict[str, Any]] = []
        progress = st.progress(0.0, text="Reading ADV filings...")
        parse_failures = 0
        first_snippet = ""
        failure_reasons: List[str] = []

        with httpx.Client(follow_redirects=True, headers=SEC_REQUEST_HEADERS) as client:
            for i, (_, row) in enumerate(pool.iterrows()):
                crd = str(row.get(cols["crd"], "") or "").strip()
                dollars, snippet = fetch_firm_raum(crd, client)
                if not first_snippet:
                    first_snippet = f"CRD {crd}: {snippet}"
                if dollars <= 0:
                    parse_failures += 1
                    failure_reasons.append(snippet or "no text extracted")
                    progress.progress((i + 1) / pool_size, text=f"Reading ADV filings... {i+1}/{pool_size}")
                    continue

                aum_m = dollars / 1_000_000.0
                if not (aum_min <= aum_m <= aum_max):
                    progress.progress((i + 1) / pool_size, text=f"Reading ADV filings... {i+1}/{pool_size}")
                    continue

                website = str(row.get(cols["website"], "") or "").strip() if cols.get("website") else ""
                if website and not website.lower().startswith(("http://", "https://")):
                    website = "https://" + website

                rows.append({
                    "firm_name": str(row.get(cols["firm_name"], "") or "").strip(),
                    "website": website,
                    "city": str(row.get(cols["city"], "") or "").strip() if cols.get("city") else "",
                    "state": str(row.get(cols["state"], "") or "").strip().upper() if cols.get("state") else "",
                    "aum_millions": aum_m,
                    "crd": crd,
                    "source": "SEC Form ADV",
                })
                progress.progress((i + 1) / pool_size, text=f"Reading ADV filings... {i+1}/{pool_size}")

        progress.empty()

        if parse_failures:
            st.warning(
                f"Could not read AUM from {parse_failures} of {pool_size} filings. "
                "Those firms were dropped rather than recorded with a guessed figure."
            )
            if failure_reasons:
                counts: Dict[str, int] = {}
                for reason in failure_reasons:
                    key = reason.split("—")[0].strip()[:70]
                    counts[key] = counts.get(key, 0) + 1
                with st.expander("🔧 Why filings were skipped", expanded=True):
                    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                        st.text(f"{n:>3} x  {key}")
        if first_snippet:
            # The AUM regex is a heuristic against a variable PDF layout, so the
            # raw text it matched on is shown for verification.
            with st.expander("🔧 Item 5.F text extracted from the first filing", expanded=False):
                st.text(first_snippet[:1200])

        rows.sort(key=lambda r: r["aum_millions"], reverse=True)
        candidates = rows[:limit]
        st.caption(
            f"{len(rows)} firms fell inside ${aum_min:,.0f}M–${aum_max:,.0f}M; "
            f"taking top {len(candidates)} by filed AUM."
        )
        return candidates

    @staticmethod
    def target_from_adv(candidate: Dict[str, Any]) -> RIATarget:
        """
        Build a target straight from filed ADV data.

        Name, city, state and AUM come from the SEC filing, so no scraping and
        no LLM sit between the source and the record — these fields cannot be
        hallucinated or silently blanked.
        """
        aum_m = float(candidate.get("aum_millions") or 0.0)
        if aum_m >= 1000:
            aum_display = f"${aum_m / 1000:.2f}B"
        else:
            aum_display = f"${aum_m:,.1f}M"

        now = datetime.now().isoformat()
        return RIATarget(
            firm_name=candidate.get("firm_name", ""),
            city=candidate.get("city", ""),
            state=candidate.get("state", ""),
            aum=aum_display,
            aum_numeric=aum_m,
            aum_unit="B" if aum_m >= 1000 else "M",
            website=candidate.get("website", ""),
            notes=f"CRD {candidate['crd']}" if candidate.get("crd") else "",
            source=candidate.get("source", "SEC Form ADV"),
            created_at=now,
            last_updated=now,
        )

    async def extract_ria_data(self, firm_list: List[Dict[str, Any]]) -> List[RIATarget]:
        """
        Extract structured data from firm websites and filings.

        Args:
            firm_list: List of dicts from search_ria_universe

        Returns:
            List of RIATarget objects with basic fields populated
        """
        targets = []

        for firm in firm_list:
            if not firm.get("website"):
                continue

            try:
                # Extract structured data straight from the URL. The previous
                # version also called /api/scrape and used the result purely as
                # a gate, which cost an extra API call per firm for nothing.
                extract_prompt = """
                Extract from this RIA (registered investment advisor) website.
                Return ONLY a JSON object, no commentary, with exactly these keys:
                  "firm_name": exact legal firm name
                  "city": headquarters city
                  "state": headquarters state as a 2-letter US code
                  "aum": assets under management as written, e.g. "$125M" or "$1.2B"
                  "fee_structure": how they charge
                  "team_size": number of advisors/staff
                  "services": list of services offered
                  "geographic_focus": regions served
                Use an empty string for any field you cannot find. Do not guess.
                """

                extract_result = await self.scraper.extract(firm["website"], extract_prompt)
                failure = sg_failed(extract_result)
                if failure:
                    st.warning(f"Extract failed for {firm['firm_name']}: {failure}")
                    continue

                parsed = sg_extracted(extract_result)
                if not parsed:
                    st.warning(
                        f"No structured data returned for {firm['firm_name']} — "
                        "response shape did not match any known format."
                    )
                    sg_debug(f"extract: {firm['firm_name']}", extract_result)

                firm_name = parsed.get("firm_name") or firm["firm_name"]
                city = str(parsed.get("city") or "")
                state = str(parsed.get("state") or "")
                aum_str = str(
                    parsed.get("aum")
                    or parsed.get("assets_under_management")
                    or ""
                )

                aum_numeric, aum_unit = parse_aum(aum_str)

                target = RIATarget(
                    firm_name=firm_name,
                    city=city,
                    state=state,
                    aum=aum_str,
                    aum_numeric=aum_numeric,
                    aum_unit=aum_unit,
                    website=firm["website"],
                    source=firm.get("source", "Web Search"),
                    created_at=datetime.now().isoformat(),
                    last_updated=datetime.now().isoformat()
                )
                targets.append(target)
            except Exception as e:
                st.warning(f"Extraction failed for {firm['firm_name']}: {e}")

        return targets

    async def enrich_ria_profile(self, target: RIATarget) -> RIATarget:
        """
        Add deep founder/CEO/owner information via AI synthesis.

        Args:
            target: Partially populated RIATarget

        Returns:
            Enriched RIATarget with founder/CEO information
        """
        if not target.website:
            return target

        try:
            enrich_prompt = f"""
            From this wealth management firm's website, extract leadership info
            for {target.firm_name}.

            Return ONLY a JSON object, no commentary, with exactly these keys:
              "founder_name": full name of the founder
              "founder_title": their title/role
              "founder_background": education, prior firms, years in industry
              "ceo_name": current CEO, if different from the founder
              "ceo_background": their background

            Use an empty string for any field you cannot find. Do not guess.
            """

            enrich_result = await self.scraper.extract(target.website, enrich_prompt)
            failure = sg_failed(enrich_result)
            if failure:
                st.warning(f"Enrichment failed for {target.firm_name}: {failure}")
            else:
                parsed = sg_extracted(enrich_result)
                if parsed:
                    target.founder_name = str(parsed.get("founder_name") or "")
                    target.founder_title = str(parsed.get("founder_title") or "")
                    target.founder_info = str(parsed.get("founder_background") or "")
                    target.ceo_name = str(parsed.get("ceo_name") or "")
                    target.ceo_info = str(parsed.get("ceo_background") or "")
                else:
                    st.warning(
                        f"No leadership data returned for {target.firm_name} — "
                        "response shape did not match any known format."
                    )
                    sg_debug(f"enrich: {target.firm_name}", enrich_result)
        except Exception as e:
            st.warning(f"Enrichment failed for {target.firm_name}: {e}")

        # Score completeness on whatever actually landed, success or not, so
        # the number reflects reality instead of staying 0 on a silent failure.
        fields_populated = sum([
            bool(target.firm_name),
            bool(target.city),
            bool(target.state),
            bool(target.aum),
            bool(target.founder_name),
            bool(target.founder_info)
        ])
        target.confidence_score = min(1.0, fields_populated / 6.0)

        target.last_updated = datetime.now().isoformat()
        return target

    async def compile_curated_list(self, filters: Dict[str, Any]) -> List[RIATarget]:
        """
        Execute full pipeline: search → extract → enrich.

        Args:
            filters: Search filter dict (see search_ria_universe)

        Returns:
            List of fully enriched RIATarget objects
        """
        # Stage 1: Authoritative universe from the SEC Form ADV registry.
        st.info("🔍 Loading SEC Form ADV registry...")
        candidates = await self.search_ria_universe(filters)
        if not candidates:
            st.warning("No registered advisers matched those filters.")
            return []
        st.success(f"Found {len(candidates)} matching advisers")

        # Stage 2: Name, city, state and AUM come straight from the filing.
        extracted_targets = [self.target_from_adv(c) for c in candidates]

        # Stage 3: Enrich founder/CEO from each firm's own site — the one thing
        # the ADV does not carry. Concurrency capped to avoid API throttling.
        with_site = [t for t in extracted_targets if t.website]
        without_site = [t for t in extracted_targets if not t.website]
        if without_site:
            st.caption(f"{len(without_site)} adviser(s) list no website in their ADV — leadership lookup skipped.")

        st.info("👥 Enriching founder/CEO information...")
        enriched_targets = list(without_site)
        max_concurrent = 5
        progress = st.progress(0.0)

        for i in range(0, len(with_site), max_concurrent):
            batch = with_site[i:i + max_concurrent]
            tasks = [self.enrich_ria_profile(t) for t in batch]
            batch_results = await asyncio.gather(*tasks)
            enriched_targets.extend(batch_results)
            progress.progress(min(1.0, (i + len(batch)) / max(1, len(with_site))))

        progress.empty()
        found_founder = sum(1 for t in enriched_targets if t.founder_name)
        st.success(
            f"Enriched {len(enriched_targets)} targets — "
            f"founder identified for {found_founder} of {len(with_site)} with a website."
        )

        # Stage 4: Save and return
        self.save_targets(enriched_targets)

        # Sort by AUM (descending)
        enriched_targets.sort(key=lambda t: t.aum_numeric, reverse=True)

        return enriched_targets

    async def update_single_target(self, firm_name: str, website: str) -> RIATarget:
        """
        Research one firm in depth.

        Args:
            firm_name: Target firm name
            website: Target firm website URL

        Returns:
            Fully researched RIATarget
        """
        # Create base target
        target = RIATarget(
            firm_name=firm_name,
            website=website,
            city="",
            state="",
            aum="",
            aum_numeric=0.0,
            aum_unit="M",
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            source="Manual"
        )

        # Extract data
        extracted_list = await self.extract_ria_data([{
            "firm_name": firm_name,
            "website": website,
            "state": "",
            "aum_estimate": "",
            "source": "Manual"
        }])

        if extracted_list:
            target = extracted_list[0]

        # Enrich
        target = await self.enrich_ria_profile(target)

        # Update or append to CSV
        existing = self.load_targets()
        existing_names = {t.firm_name for t in existing}

        if firm_name in existing_names:
            # Update existing
            for i, t in enumerate(existing):
                if t.firm_name == firm_name:
                    existing[i] = target
                    break
        else:
            # Append new
            existing.append(target)

        self.save_targets(existing)
        return target

    # ========================================================================
    # Persistence Methods
    # ========================================================================

    def save_targets(self, targets: List[RIATarget]) -> None:
        """Write all targets to CSV, overwriting existing file"""
        try:
            df = pd.DataFrame([asdict(t) for t in targets])
            df.to_csv(self.curator_file, index=False)
        except Exception as e:
            st.error(f"Failed to save targets: {e}")

    def load_targets(self) -> List[RIATarget]:
        """Load targets from CSV into RIATarget objects"""
        if not os.path.exists(self.curator_file):
            return []

        try:
            df = pd.read_csv(self.curator_file)
            targets = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                # Handle NaN values
                for key in row_dict:
                    if pd.isna(row_dict[key]):
                        row_dict[key] = 0.0 if key == "aum_numeric" or key == "confidence_score" else ""
                targets.append(RIATarget(**row_dict))
            return targets
        except Exception as e:
            st.error(f"Failed to load targets: {e}")
            return []

    def append_targets(self, new_targets: List[RIATarget]) -> None:
        """Add new targets to existing list (avoid duplicates)"""
        existing = self.load_targets()
        existing_names = {t.firm_name for t in existing}
        to_add = [t for t in new_targets if t.firm_name not in existing_names]
        all_targets = existing + to_add
        self.save_targets(all_targets)

    def update_target(self, firm_name: str, updated_fields: Dict[str, Any]) -> None:
        """Update specific fields in a target"""
        targets = self.load_targets()
        for target in targets:
            if target.firm_name == firm_name:
                for field, value in updated_fields.items():
                    if hasattr(target, field):
                        setattr(target, field, value)
                target.last_updated = datetime.now().isoformat()
                break
        self.save_targets(targets)

    def get_target(self, firm_name: str) -> Optional[RIATarget]:
        """Retrieve single target by name"""
        targets = self.load_targets()
        return next((t for t in targets if t.firm_name == firm_name), None)


# ============================================================================
# RESEARCH AGENT TOOLS (via MCP)
# ============================================================================

async def research_ria_target(target_name: str, target_website: Optional[str] = None) -> str:
    """
    Research an RIA acquisition target using ScrapeGraph.

    This tool:
    1. Scrapes the target's website (if provided)
    2. Extracts key financial metrics
    3. Searches for regulatory filings and news
    4. Compiles into research summary
    """

    if not SGAI_API_KEY:
        return "ERROR: SGAI_API_KEY not configured. Set it in .env file."

    sg_client = ScrapeGraphMCPClient(SGAI_API_KEY)
    research_findings = []

    with st.spinner(f"🔍 Researching {target_name}..."):

        # Step 1: Scrape target website (if URL provided)
        if target_website:
            st.info(f"📄 Scraping {target_website}...")
            scrape_result = await sg_client.scrape(target_website, output_format="markdown")
            failure = sg_failed(scrape_result)
            if failure:
                st.warning(f"Scrape failed: {failure}")
            else:
                markdown = sg_markdown(scrape_result, target_website)
                if markdown:
                    research_findings.append({
                        "source": "Website Scrape",
                        "data": markdown[:2000]
                    })
                else:
                    st.warning("Scrape returned no markdown content.")
                    sg_debug("scrape", scrape_result)

        # Step 2: Extract structured data
        if target_website:
            st.info(f"📊 Extracting key metrics from {target_name}...")
            extract_prompt = f"""
            Extract the following information from {target_name}'s website:
            - Assets Under Management (AUM)
            - Fee structure/rates
            - Number of advisors/team size
            - Services offered
            - Geographic focus
            - Client retention claims
            - Ownership/leadership

            Return ONLY the extracted data, no commentary.
            """

            extract_result = await sg_client.extract(target_website, extract_prompt)
            failure = sg_failed(extract_result)
            if failure:
                st.warning(f"Extraction failed: {failure}")
            else:
                extracted = sg_extracted(extract_result)
                if extracted:
                    research_findings.append({
                        "source": "Website Data Extraction",
                        "data": json.dumps(extracted, indent=2)[:2000]
                    })
                else:
                    st.warning("Extraction returned no structured data.")
                    sg_debug("extract", extract_result)

        # Step 3: Search for SEC filings and news
        st.info(f"🔎 Searching for SEC filings and news about {target_name}...")
        search_query = f"{target_name} RIA SEC EDGAR Form ADV"
        search_result = await sg_client.search(search_query, num_results=5)
        failure = sg_failed(search_result)
        if failure:
            st.warning(f"SEC filings search failed: {failure}")
        else:
            sources = sg_sources(search_result)
            if sources:
                research_findings.append({
                    "source": "SEC Filings Search",
                    "data": json.dumps(sources[:3], indent=2)  # Top 3 results
                })
            else:
                st.warning("SEC filings search returned no results.")
                sg_debug("search: SEC filings", search_result)

        # Step 4: News search
        st.info(f"📰 Searching for news about {target_name}...")
        news_query = f"{target_name} wealth management news acquisition"
        news_result = await sg_client.search(news_query, num_results=5)
        failure = sg_failed(news_result)
        if failure:
            st.warning(f"News search failed: {failure}")
        else:
            sources = sg_sources(news_result)
            if sources:
                research_findings.append({
                    "source": "News & Articles",
                    "data": json.dumps(sources[:2], indent=2)  # Top 2 articles
                })
            else:
                st.warning("News search returned no results.")
                sg_debug("search: news", news_result)

    # Refuse to hand an empty brief to the analyst. With no findings, the LLM
    # will still produce a confident-looking deal scorecard — entirely invented.
    # For an M&A decision that is worse than returning nothing.
    if not research_findings:
        return NO_RESEARCH_SENTINEL

    # Compile research summary
    research_summary = f"# Research Summary: {target_name}\n\n"
    research_summary += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    research_summary += f"**Target:** {target_name}\n\n"

    for finding in research_findings:
        research_summary += f"## {finding['source']}\n"
        research_summary += f"{finding['data']}\n\n"

    return research_summary


# ============================================================================
# ACQUISITION ANALYST AGENT
# ============================================================================

def analyze_acquisition_opportunity(research_data: str, target_name: str) -> str:
    """
    Analyze research data and produce acquisition deal scorecard.

    This uses Claude to synthesize research into:
    - Deal score (1-10)
    - Risk assessment
    - Opportunity analysis
    - Recommended offer strategy
    """

    analyst_prompt = f"""
    You are an expert acquisition analyst for Strategy Financial Group (SFG),
    a registered investment advisor specializing in retirement wealth management.

    Analyze the following research data on acquisition target: {target_name}

    RESEARCH DATA:
    {research_data}

    Produce a structured acquisition assessment including:

    1. DEAL SCORECARD (1-10 scale)
       - Financial Fit (AUM, revenue, profitability)
       - Strategic Fit (services, client base, geography)
       - Cultural Fit (team retention risk, philosophy alignment)
       - Regulatory Risk (compliance issues, regulatory standing)
       - Overall Opportunity Score

    2. KEY METRICS EXTRACTED
       - Estimated AUM
       - Fee structure
       - Approximate team size
       - Service offerings
       - Geographic coverage
       - Estimated annual revenue (if inferable)

    3. OPPORTUNITY ANALYSIS
       - Top 3 reasons to acquire this firm
       - Synergies with SFG (services, geography, client overlap)
       - Revenue/AUM accretion potential

    4. RISK ASSESSMENT
       - Top 3 acquisition risks
       - Regulatory or compliance concerns
       - Team retention concerns
       - Integration complexity

    5. RECOMMENDED OFFER STRATEGY
       - Suggested AUM/revenue multiple
       - Deal structure considerations (earn-out, equity retention, etc.)
       - Integration priorities
       - Timeline considerations

    6. NEXT STEPS
       - Recommended due diligence focus areas
       - Information gaps to fill
       - Contact/outreach strategy

    Format as markdown with clear sections. Be concise and actionable.
    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert M&A analyst for wealth management firms."},
            {"role": "user", "content": analyst_prompt}
        ],
        temperature=0.3,  # Lower temp for consistency
        max_tokens=2000
    )

    return response.choices[0].message.content


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🎯 SFG Acquisition Research Agent")
st.markdown("**AI-Powered RIA Acquisition Target Research**")
st.markdown("Research and score acquisition targets using AI-driven research and deal analysis.")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key configuration
    with st.expander("API Configuration", expanded=False):
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        sgai_key = st.text_input("ScrapeGraph API Key", type="password", value=os.getenv("SGAI_API_KEY", ""))

        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if sgai_key:
            os.environ["SGAI_API_KEY"] = sgai_key

        if not openai_key or not sgai_key:
            st.warning("⚠️ Missing API keys. Set in sidebar or .env file.")

    st.divider()
    st.markdown("**About This Agent**")
    st.markdown("""
    Two-stage research pipeline:
    1. **Research Agent** - Gathers data via ScrapeGraph MCP
    2. **Analyst Agent** - Creates deal assessment
    """)

# ============================================================================
# INITIALIZE CURATOR AGENT (Session State)
# ============================================================================

if "curator_agent" not in st.session_state:
    curator_agent = CuratorAgent(
        scrapegraph_client=ScrapeGraphMCPClient(os.getenv("SGAI_API_KEY", "")),
        openai_client=openai_client,
        company_context={
            "name": os.getenv("SFG_COMPANY_NAME", "Strategy Financial Group"),
            "description": os.getenv("SFG_DESCRIPTION", ""),
            "focus": os.getenv("SFG_ACQUISITION_FOCUS", "")
        }
    )
    st.session_state.curator_agent = curator_agent
else:
    curator_agent = st.session_state.curator_agent

if "curator_targets" not in st.session_state:
    st.session_state.curator_targets = curator_agent.load_targets()

if "curator_filters" not in st.session_state:
    st.session_state.curator_filters = {
        "regions": ["Southwest"],
        "aum_min": 50.0,
        "aum_max": 500.0,
        "firm_types": ["Independent RIA"]
    }

# ============================================================================
# MAIN CONTENT - TABBED INTERFACE
# ============================================================================

tab1, tab2 = st.tabs(["🔍 Acquisition Research", "🎯 Acquisition Curator"])

# ==============================================================================
# TAB 1: ACQUISITION RESEARCH (Original functionality)
# ==============================================================================

with tab1:
    st.header("Individual Target Research")
    st.markdown("Deep dive research into a specific RIA target with AI-powered analysis.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Target Research")
        target_name = st.text_input("RIA Target Name", placeholder="e.g., Evergreen Wealth Management")
        target_website = st.text_input("Target Website (optional)", placeholder="e.g., https://www.example-ria.com")

    with col2:
        st.subheader("Research Options")
        include_news = st.checkbox("Include News Search", value=True)
        include_sec = st.checkbox("Include SEC Filing Search", value=True)

    # Main research button
    if st.button("🚀 Start Acquisition Research", type="primary", disabled=not (target_name and os.getenv("SGAI_API_KEY"))):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OpenAI API key not configured.")
        elif not os.getenv("SGAI_API_KEY"):
            st.error("❌ ScrapeGraph API key not configured.")
        elif not target_name:
            st.error("❌ Please enter a target name.")
        else:
            # Stage 1: Research
            st.divider()
            st.header("📊 Research Phase")

            research_data = asyncio.run(research_ria_target(target_name, target_website))

            if research_data == NO_RESEARCH_SENTINEL:
                st.error(
                    "❌ No research data could be collected for this target. "
                    "Every ScrapeGraph call failed or returned nothing — see the "
                    "warnings above. Deal analysis was **not** run, because "
                    "scoring an acquisition on zero data would produce a "
                    "confident but entirely fabricated assessment."
                )
                st.stop()

            with st.expander("📄 View Raw Research Data", expanded=False):
                st.markdown(research_data)

            # Stage 2: Analysis
            st.divider()
            st.header("🔍 Deal Analysis")

            with st.spinner("Analyzing acquisition opportunity..."):
                deal_assessment = analyze_acquisition_opportunity(research_data, target_name)

            st.markdown(deal_assessment)

            # Export options
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "📥 Download Research Report",
                    data=research_data,
                    file_name=f"{target_name.replace(' ', '_')}_research.md",
                    mime="text/markdown"
                )

            with col2:
                st.download_button(
                    "📥 Download Deal Assessment",
                    data=deal_assessment,
                    file_name=f"{target_name.replace(' ', '_')}_deal_assessment.md",
                    mime="text/markdown"
                )

# ==============================================================================
# TAB 2: ACQUISITION CURATOR
# ==============================================================================

with tab2:
    st.header("🎯 Acquisition Curator")
    st.markdown("Actively search and curate RIA acquisition targets matching your criteria.")

    # Sidebar filters for curator
    with st.sidebar:
        st.subheader("Curator Search Filters")
        regions = st.multiselect(
            "Regions",
            list(REGION_STATES.keys()),
            default=st.session_state.curator_filters.get("regions", ["Southwest"])
        )
        resolved_states = sorted({s for r in regions for s in REGION_STATES.get(r, [])})
        if resolved_states:
            st.caption("States searched: " + ", ".join(resolved_states))
        elif regions:
            st.caption("No state filter — searching all registered advisers.")
        aum_min, aum_max = st.slider(
            "AUM Range ($M)",
            0, 1000,
            (
                int(st.session_state.curator_filters.get("aum_min", 50)),
                int(st.session_state.curator_filters.get("aum_max", 500))
            )
        )
        firm_types = st.multiselect(
            "Firm Types",
            ["Independent RIA", "Registered Investment Advisor", "Wealth Management"],
            default=st.session_state.curator_filters.get("firm_types", ["Independent RIA"])
        )

        result_limit = st.slider(
            "How many candidates to research",
            5, 50,
            int(st.session_state.curator_filters.get("limit", 10)),
            help="Top N by regulatory AUM. Each one costs an enrichment API call."
        )

        # Update session state filters
        st.session_state.curator_filters = {
            "regions": regions,
            "aum_min": float(aum_min),
            "aum_max": float(aum_max),
            "firm_types": firm_types,
            "limit": int(result_limit)
        }

        st.divider()

        # Run curator search
        if st.button("🔍 Run Acquisition Search", type="primary"):
            with st.spinner("Running curation pipeline... this may take 3-5 minutes"):
                try:
                    targets = asyncio.run(curator_agent.compile_curated_list(st.session_state.curator_filters))
                    if targets:
                        st.session_state.curator_targets = targets
                        st.success(f"✅ Found and curated {len(targets)} targets!")
                    else:
                        # Never dress a failed run as a success — see the errors above.
                        st.error(
                            "❌ Search produced no targets. The existing list was left "
                            "untouched. See the errors above for the cause."
                        )
                except Exception as e:
                    st.error(f"❌ Curation failed: {e}")

        st.divider()

        # Manual add form
        st.subheader("Add Manual Entry")
        with st.form("manual_add_form"):
            manual_firm_name = st.text_input("Firm Name", placeholder="e.g., Jackson Hole Wealth")
            manual_website = st.text_input("Website", placeholder="https://...")
            manual_city = st.text_input("City (optional)")
            manual_state = st.text_input("State (optional)", max_chars=2)

            if st.form_submit_button("➕ Add Target"):
                if manual_firm_name and manual_website:
                    with st.spinner("Researching target..."):
                        try:
                            new_target = asyncio.run(curator_agent.update_single_target(manual_firm_name, manual_website))
                            st.session_state.curator_targets.append(new_target)
                            st.success(f"✅ Added {manual_firm_name} to curator list!")
                        except Exception as e:
                            st.error(f"❌ Failed to add target: {e}")
                else:
                    st.error("Firm name and website required")

    # Main results area
    st.subheader("Curator Targets")

    if st.session_state.curator_targets:
        targets = st.session_state.curator_targets

        # Search and filter
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            search_term = st.text_input("Search by firm name", placeholder="Type to filter...")
        with col2:
            sort_by = st.selectbox("Sort by", ["AUM (Highest)", "Confidence Score", "Recently Updated", "Created Date"])
        with col3:
            status_filter = st.multiselect("Status", ["Not Started", "In Progress", "Complete", "Contacted"], default=None)

        # Apply filters
        filtered_targets = targets
        if search_term:
            filtered_targets = [t for t in filtered_targets if search_term.lower() in t.firm_name.lower() or search_term.lower() in t.founder_name.lower()]

        if status_filter:
            filtered_targets = [t for t in filtered_targets if t.research_status in status_filter]

        # Apply sorting
        if sort_by == "AUM (Highest)":
            filtered_targets.sort(key=lambda t: t.aum_numeric, reverse=True)
        elif sort_by == "Confidence Score":
            filtered_targets.sort(key=lambda t: t.confidence_score, reverse=True)
        elif sort_by == "Recently Updated":
            filtered_targets.sort(key=lambda t: t.last_updated, reverse=True)
        elif sort_by == "Created Date":
            filtered_targets.sort(key=lambda t: t.created_at, reverse=True)

        # Display results table
        st.write(f"**Showing {len(filtered_targets)} of {len(targets)} targets**")

        # Build display dataframe
        display_data = []
        for i, target in enumerate(filtered_targets):
            display_data.append({
                "Firm Name": target.firm_name,
                "City": target.city,
                "State": target.state,
                "AUM": target.aum,
                "Founder": target.founder_name,
                "Status": target.research_status,
                "Confidence": f"{target.confidence_score:.0%}",
                "Index": i
            })

        df_display = pd.DataFrame(display_data)

        # Display table
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Export buttons
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = pd.DataFrame([asdict(t) for t in filtered_targets]).to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                data=csv_data,
                file_name=f"sfg_curator_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col2:
            json_data = json.dumps([asdict(t) for t in filtered_targets], indent=2, default=str)
            st.download_button(
                "📥 Download as JSON",
                data=json_data,
                file_name=f"sfg_curator_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

        with col3:
            st.info(f"📊 Total targets in curator list: {len(targets)}")

    else:
        st.info("📭 No targets yet. Click 'Run Acquisition Search' or add manual entries to get started.")

# Footer
st.divider()
st.markdown("**SFG Acquisition Research Agent** | Powered by OpenAI + ScrapeGraph MCP")
