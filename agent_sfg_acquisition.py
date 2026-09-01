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
        candidates = []

        # Build search query from filters
        regions = ", ".join(filters.get("regions", ["Southwest"]))
        aum_range = f"${filters.get('aum_min', 50)}M to ${filters.get('aum_max', 500)}M AUM"

        # Search for RIA candidates via SEC EDGAR and web
        search_queries = [
            f"RIA {regions} {aum_range} independent wealth management Form ADV",
            f"{regions} registered investment advisor $50M-$500M assets",
            f"independent wealth management firm {aum_range} {regions}"
        ]

        for query in search_queries:
            try:
                result = await self.scraper.search(query, num_results=10)
                if "error" not in result and "data" in result:
                    sources = result.get("data", {}).get("sources", [])
                    for source in sources[:5]:  # Take top 5 per query
                        candidates.append({
                            "firm_name": source.get("title", "Unknown"),
                            "website": source.get("url", ""),
                            "state": "",  # Will be extracted later
                            "aum_estimate": "",
                            "source": "SEC EDGAR" if "sec" in source.get("url", "").lower() else "Web Search"
                        })
            except Exception as e:
                st.warning(f"Search query failed: {e}")

        # Deduplicate by firm_name
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c["firm_name"] not in seen:
                seen.add(c["firm_name"])
                unique_candidates.append(c)

        return unique_candidates[:30]  # Return top 30

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
                # Scrape firm website
                scrape_result = await self.scraper.scrape(firm["website"], output_format="markdown")
                if "error" in scrape_result:
                    continue

                # Extract structured data
                extract_prompt = """
                Extract from this RIA website:
                - Firm name (exact legal name)
                - City and state
                - Assets Under Management (AUM) - numeric and unit (M/B)
                - Fee structure
                - Team size
                - Services offered
                - Geographic focus

                Return as JSON only, no commentary.
                """

                extract_result = await self.scraper.extract(firm["website"], extract_prompt)
                if "error" not in extract_result:
                    data = extract_result.get("data", {}).get("content", "")

                    # Parse extracted data (attempt JSON parse)
                    try:
                        parsed = json.loads(data) if isinstance(data, str) else data
                    except:
                        parsed = {}

                    # Create RIATarget with extracted data
                    firm_name = parsed.get("firm_name", firm["firm_name"])
                    city = parsed.get("city", "")
                    state = parsed.get("state", "")
                    aum_str = parsed.get("aum", "")

                    # Parse AUM numeric and unit
                    aum_numeric = 0.0
                    aum_unit = "M"
                    if aum_str:
                        # Extract numeric value (e.g., "$125M" → 125.0, "M")
                        import re
                        match = re.search(r'(\d+\.?\d*)\s*([MB])?', str(aum_str))
                        if match:
                            aum_numeric = float(match.group(1))
                            aum_unit = match.group(2) or "M"

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
            # Scrape for leadership information
            scrape_result = await self.scraper.scrape(target.website, output_format="markdown")
            if "error" in scrape_result:
                return target

            # Extract founder/CEO info
            enrich_prompt = f"""
            From this wealth management firm website, extract leadership information:
            - Founder name
            - Founder title/role
            - Founder background (education, experience, years in industry)
            - Current CEO (if different from founder)
            - CEO background

            For {target.firm_name}, return as JSON with keys:
            founder_name, founder_title, founder_background, ceo_name, ceo_background

            Return JSON only, no commentary.
            """

            enrich_result = await self.scraper.extract(target.website, enrich_prompt)
            if "error" not in enrich_result:
                data = enrich_result.get("data", {}).get("content", "")
                try:
                    parsed = json.loads(data) if isinstance(data, str) else data
                    target.founder_name = parsed.get("founder_name", "")
                    target.founder_title = parsed.get("founder_title", "")
                    target.founder_info = parsed.get("founder_background", "")
                    target.ceo_name = parsed.get("ceo_name", "")
                    target.ceo_info = parsed.get("ceo_background", "")

                    # Assign confidence score based on data completeness
                    fields_populated = sum([
                        bool(target.firm_name),
                        bool(target.city),
                        bool(target.state),
                        bool(target.aum),
                        bool(target.founder_name),
                        bool(target.founder_info)
                    ])
                    target.confidence_score = min(1.0, fields_populated / 6.0)
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"Enrichment failed for {target.firm_name}: {e}")

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
        # Stage 1: Search
        st.info("🔍 Searching RIA universe...")
        candidates = await self.search_ria_universe(filters)
        st.success(f"Found {len(candidates)} candidates")

        # Stage 2: Extract
        st.info(f"📊 Extracting data from {len(candidates)} firms...")
        extracted_targets = await self.extract_ria_data(candidates)
        st.success(f"Extracted {len(extracted_targets)} targets")

        # Stage 3: Enrich (with concurrency limit to avoid API throttling)
        st.info(f"👥 Enriching founder/CEO information...")
        enriched_targets = []
        max_concurrent = 5

        for i in range(0, len(extracted_targets), max_concurrent):
            batch = extracted_targets[i:i+max_concurrent]
            tasks = [self.enrich_ria_profile(t) for t in batch]
            batch_results = await asyncio.gather(*tasks)
            enriched_targets.extend(batch_results)

        st.success(f"Enriched {len(enriched_targets)} targets")

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
            if "error" not in scrape_result:
                research_findings.append({
                    "source": "Website Scrape",
                    "data": scrape_result.get("data", {}).get(f"{target_website}_markdown", "")[:1000]  # First 1000 chars
                })

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
            if "error" not in extract_result:
                research_findings.append({
                    "source": "Website Data Extraction",
                    "data": extract_result.get("data", {}).get("content", "")[:500]
                })

        # Step 3: Search for SEC filings and news
        st.info(f"🔎 Searching for SEC filings and news about {target_name}...")
        search_query = f"{target_name} RIA SEC EDGAR Form ADV"
        search_result = await sg_client.search(search_query, num_results=5)
        if "error" not in search_result:
            sources = search_result.get("data", {}).get("sources", [])
            research_findings.append({
                "source": "SEC Filings Search",
                "data": json.dumps(sources[:3], indent=2)  # Top 3 results
            })

        # Step 4: News search
        st.info(f"📰 Searching for news about {target_name}...")
        news_query = f"{target_name} wealth management news acquisition"
        news_result = await sg_client.search(news_query, num_results=5)
        if "error" not in news_result:
            sources = news_result.get("data", {}).get("sources", [])
            research_findings.append({
                "source": "News & Articles",
                "data": json.dumps(sources[:2], indent=2)  # Top 2 articles
            })

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
            ["Southwest", "Southeast", "Midwest", "Northeast", "National"],
            default=st.session_state.curator_filters.get("regions", ["Southwest"])
        )
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

        # Update session state filters
        st.session_state.curator_filters = {
            "regions": regions,
            "aum_min": float(aum_min),
            "aum_max": float(aum_max),
            "firm_types": firm_types
        }

        st.divider()

        # Run curator search
        if st.button("🔍 Run Acquisition Search", type="primary"):
            with st.spinner("Running curation pipeline... this may take 3-5 minutes"):
                try:
                    targets = asyncio.run(curator_agent.compile_curated_list(st.session_state.curator_filters))
                    st.session_state.curator_targets = targets
                    st.success(f"✅ Found and curated {len(targets)} targets!")
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
