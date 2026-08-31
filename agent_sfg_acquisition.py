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
import streamlit as st
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import httpx

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

# Main content
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
                value=research_data,
                file_name=f"{target_name.replace(' ', '_')}_research.md",
                mime="text/markdown"
            )

        with col2:
            st.download_button(
                "📥 Download Deal Assessment",
                value=deal_assessment,
                file_name=f"{target_name.replace(' ', '_')}_deal_assessment.md",
                mime="text/markdown"
            )

# Footer
st.divider()
st.markdown("**SFG Acquisition Research Agent** | Powered by OpenAI + ScrapeGraph MCP")
