"""
IAPD people lookup — founder / principal enrichment for RIA acquisition targets.

Shared by agent_sfg_acquisition.py (Streamlit) and weekly_report.py (cron).
Pure Python: no streamlit, no pandas. Import it from anywhere.

WHY THIS EXISTS
---------------
The SEC roster we already download every week has 448 columns and NOT ONE of
them names a human being. Verified against the live September file: the only
person-adjacent columns are "Control/Controlled by Related Person" (a Y/N flag)
and "Count of Control person Public Reporting Company" (an integer). No
Schedule A, no chief compliance officer, no signatory. So the owner has to come
from somewhere else.

WHAT THIS USES
--------------
    https://api.adviserinfo.sec.gov/search/individual?firm=<CRD>&...

An undocumented JSON endpoint behind the public IAPD site. It returns every
investment adviser representative registered at that firm. Keyed on the
Organization CRD# we already carry from the roster, so no name matching, no
scraping, no LLM, no API credits.

VERIFIED 01 Sep 2026 against live responses:
  * 40 consecutive qualifying firms -> 40x HTTP 200, zero empty results.
  * ~7 requests/second sustained with no throttling or 429.
  * hits.total is a plain integer; hits.hits is a list.
  * Each _source carries ind_source_id, ind_firstname, ind_lastname,
    ind_industry_cal_date_iapd (YYYY-MM-DD) and ind_ia_current_employments.
  * Headcount at firms in our $50-250M band is typically 1-6, which is the
    regime where the principal heuristic below actually resolves.
  * Example: firm 298237 "DEAN WEALTH PARTNERS, LLC" -> MAX DEAN (industry
    start 1997) and Bradley Reddin (2024).

THE HONESTY RULE
----------------
ind_industry_cal_date_iapd is the date the person entered the securities
industry ANYWHERE. It is NOT the firm's founding date and NOT their tenure at
this firm. Never render it as "founded in". Every consumer of this module must
carry the confidence level through to the reader; these names get cold-called,
and a confident wrong name is worse than a blank.

DEFENSIVE PARSING
-----------------
This endpoint is undocumented and unversioned, so it can change shape without
notice. Every level is shape-checked and every failure returns empty rather
than raising. That is the same lesson that cost us nine occurrences of the
ScrapeGraph key-presence bug: never assume a key, always test the value.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import httpx

IAPD_INDIVIDUAL_URL = "https://api.adviserinfo.sec.gov/search/individual"
IAPD_INDIVIDUAL_PROFILE = "https://adviserinfo.sec.gov/individual/summary/{id}"
IAPD_FIRM_PROFILE = "https://adviserinfo.sec.gov/firm/summary/{crd}"

# Measured at ~7 req/s with no throttling; this leaves a wide margin and still
# enriches a 20-firm email in about four seconds.
IAPD_DELAY = 0.15

# Tokens that are business-entity noise rather than a person's name. Used to
# decide whether a firm is named after somebody.
_ENTITY_WORDS = {
    "ADVISOR", "ADVISORS", "ADVISER", "ADVISERS", "ADVISORY", "ASSET", "ASSETS",
    "ASSOCIATES", "CAPITAL", "COMPANY", "CO", "CORP", "CORPORATION", "COUNSEL",
    "FINANCIAL", "FIRM", "GROUP", "HOLDINGS", "INC", "INVESTMENT", "INVESTMENTS",
    "LLC", "LLP", "LP", "LTD", "MANAGEMENT", "PARTNERS", "PARTNERSHIP", "PC",
    "PLANNING", "PLLC", "PORTFOLIO", "PORTFOLIOS", "PRIVATE", "RETIREMENT",
    "SERVICES", "SOLUTIONS", "STRATEGIES", "THE", "AND", "OF", "WEALTH",
    "BROTHERS", "SONS", "FAMILY", "OFFICE", "TRUST", "SECURITIES", "COUNSEL",
    "EQUITY", "PLANNERS", "REGISTERED", "NETWORK", "CONSULTING", "CONSULTANTS",
}


def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z]+", str(text).upper()) if len(t) > 2]


def firm_name_tokens(firm_name: str) -> List[str]:
    """Name-ish tokens from a firm name, entity boilerplate removed."""
    return [t for t in _tokens(firm_name) if t not in _ENTITY_WORDS]


def _title(name: str) -> str:
    """Normalise ALVIN LY / Bradley Reddin to a consistent display form."""
    parts = [p for p in re.split(r"\s+", str(name).strip()) if p]
    out = []
    for p in parts:
        if len(p) <= 3 and p.isupper() and "." not in p:
            out.append(p.title())
        else:
            out.append(p[:1].upper() + p[1:].lower() if p.isupper() else p)
    return " ".join(out)


def _year(date_str: Any) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(date_str or ""))
    return int(m.group(0)) if m else None


def fetch_people(crd: str, client: Optional[httpx.Client] = None,
                 max_rows: int = 50, timeout: float = 20.0) -> List[Dict[str, Any]]:
    """
    Every investment adviser rep registered at this firm.

    Returns [] on any failure or unexpected shape — never raises. An empty list
    means "we could not establish this", never "this firm has no people".
    """
    crd = str(crd or "").strip()
    if not crd.isdigit():
        return []

    params = {
        "firm": crd,
        "start": "0",
        "nrows": str(max_rows),
        "investmentAdvisorReps": "true",
    }

    owned = client is None
    if owned:
        client = httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        resp = client.get(IAPD_INDIVIDUAL_URL, params=params, timeout=timeout)
        if resp.status_code != 200:
            return []
        payload = resp.json()
    except Exception:
        return []
    finally:
        if owned:
            try:
                client.close()
            except Exception:
                pass

    if not isinstance(payload, dict):
        return []
    hits = payload.get("hits")
    if not isinstance(hits, dict):
        return []
    rows = hits.get("hits")
    if not isinstance(rows, list):
        return []

    people: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        src = row.get("_source")
        if not isinstance(src, dict):
            continue
        first = str(src.get("ind_firstname") or "").strip()
        last = str(src.get("ind_lastname") or "").strip()
        if not (first or last):
            continue
        people.append({
            "id": str(src.get("ind_source_id") or "").strip(),
            "first": first,
            "last": last,
            "name": _title(f"{first} {last}".strip()),
            "last_upper": last.upper(),
            "industry_since": _year(src.get("ind_industry_cal_date_iapd")),
            "ia_active": str(src.get("ind_ia_scope") or "").strip().lower() == "active",
            "has_disclosure": str(src.get("ind_ia_disclosure_fl") or "").strip().upper() == "Y",
        })
    return people


def pick_principal(firm_name: str, people: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Best guess at who owns the practice, with an honest confidence level.

    Signals, cheapest and strongest first:
      1. Eponymous  — a rep's surname appears in the firm name. Practically
                      decisive: people do not name a firm after an employee.
      2. Tiny firm  — one or two registered reps means the owner is in the list
                      by definition; the longest-tenured of the two is the bet.
      3. Small firm — up to six reps, take the longest-tenured. A real guess.
      4. Anything larger — name the longest-tenured but flag it as unresolved.

    Returns a dict that ALWAYS has the same keys, so callers never branch on
    presence. confidence is one of: high, medium, low, none.
    """
    blank = {
        "name": "", "id": "", "industry_since": None, "confidence": "none",
        "basis": "no registered reps returned by IAPD",
        "headcount": 0, "profile_url": "", "eponymous": False,
    }
    if not people:
        return blank

    active = [p for p in people if p.get("ia_active")] or list(people)
    headcount = len(people)

    def newest_first(p: Dict[str, Any]) -> int:
        y = p.get("industry_since")
        return y if isinstance(y, int) else 9999

    by_tenure = sorted(active, key=newest_first)
    senior = by_tenure[0]

    tokens = set(firm_name_tokens(firm_name))
    eponymous = [p for p in active if p["last_upper"] in tokens and len(p["last_upper"]) > 2]

    if eponymous:
        pick = sorted(eponymous, key=newest_first)[0]
        confidence = "high"
        basis = f"surname appears in the firm name; {headcount} registered rep(s)"
    elif headcount <= 2:
        pick, confidence = senior, "high"
        basis = ("sole registered rep" if headcount == 1
                 else "only 2 registered reps — longest in the industry")
    elif headcount <= 6:
        pick, confidence = senior, "medium"
        basis = f"longest-tenured of {headcount} registered reps"
    else:
        pick, confidence = senior, "low"
        basis = f"longest-tenured of {headcount} registered reps — not resolved"

    return {
        "name": pick["name"],
        "id": pick["id"],
        "industry_since": pick.get("industry_since"),
        "confidence": confidence,
        "basis": basis,
        "headcount": headcount,
        "eponymous": bool(eponymous),
        "profile_url": IAPD_INDIVIDUAL_PROFILE.format(id=pick["id"]) if pick.get("id") else "",
    }


def enrich_crds(crds: List[str], delay: float = IAPD_DELAY,
                log: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    """
    Look up several firms in one client, throttled. Keyed by CRD.

    `crds` should already be de-duplicated by the caller and kept to the firms
    actually being shown — this is a per-firm HTTP call, not a bulk feed.
    """
    out: Dict[str, Dict[str, Any]] = {}
    seen = [c for c in dict.fromkeys(str(c or "").strip() for c in crds) if c]
    if not seen:
        return out
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        for i, crd in enumerate(seen):
            if i:
                time.sleep(delay)
            out[crd] = {"people": fetch_people(crd, client=client)}
    if log:
        found = sum(1 for v in out.values() if v["people"])
        log(f"IAPD: people found for {found} of {len(seen)} firms")
    return out


def principal_for(firm_name: str, crd: str,
                  cache: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Convenience: resolve one firm, using a pre-fetched cache when present."""
    crd = str(crd or "").strip()
    if cache is not None and crd in cache:
        people = cache[crd].get("people") or []
    else:
        people = fetch_people(crd)
    return pick_principal(firm_name, people)


CONFIDENCE_LABEL = {
    "high": "likely principal",
    "medium": "probable principal",
    "low": "senior rep — owner not resolved",
    "none": "",
}
CONFIDENCE_COLOR = {
    "high": "#047857",
    "medium": "#b45309",
    "low": "#6b7280",
    "none": "#9ca3af",
}
