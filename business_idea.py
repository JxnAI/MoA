from textwrap import dedent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.assistant import Assistant
from phi.tools.yfinance import YFinanceTools
from phi.model.openai import OpenAIChat

# Market Research Assistant
market_researcher = Assistant(
    name="MarketResearcher",
    role="Conducts comprehensive market research and analysis",
    description=dedent(
        """\
    You are an expert market research analyst. Given a business idea, you analyze market size,
    trends, competitors, and customer segments. Your goal is to provide data-backed insights
    about market viability.
    """
    ),
    model=Gemini(id="gemini-1.5-flash"),
    instructions=[
        "Generate 3 specific search terms to research: market size, competitors, and customer demographics.",
        "For each search term, use `search_google` to find relevant market data and statistics.",
        "Analyze market trends, growth potential, and competitive landscape.",
        "Identify target customer segments and their characteristics.",
        "Focus on finding concrete numbers and statistics when available.",
        "Look for recent industry reports and market analyses.",
        "Identify any regulatory or legal considerations.",
    ],
    tools=[DuckDuckGo()],
    add_datetime_to_instructions=True,
)

# Financial Analyst
financial_analyst = Assistant(
    name="FinancialAnalyst",
    role="Analyzes financial viability and projections",
    description=dedent(
        """\
    You are a seasoned financial analyst specializing in startup validation. Your role is to
    assess the financial viability of business ideas and create initial projections.
    """
    ),
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "Analyze potential revenue streams and pricing models.",
        "Estimate initial startup costs and operational expenses.",
        "Project cash flow for the first 2 years.",
        "Calculate key metrics: break-even point, potential ROI, burn rate.",
        "Identify major cost drivers and financial risks.",
        "Consider different monetization strategies.",
        "Suggest funding requirements and potential sources.",
        "Be conservative in estimations and clearly state assumptions.",
    ],
    tools=[YFinanceTools(analyst_recommendations=True)],
    add_datetime_to_instructions=True,
)

# Risk Analyst
risk_analyst = Assistant(
    name="RiskAnalyst",
    role="Identifies and assesses potential risks",
    description=dedent(
        """\
    You are a risk assessment specialist. Your role is to identify, analyze, and evaluate
    all potential risks associated with the business idea.
    """
    ),
    model=Gemini(id="gemini-1.5-flash"),
    instructions=[
        "Identify market risks (competition, market changes, trends).",
        "Analyze operational risks (supply chain, technology, scalability).",
        "Assess financial risks (funding needs, cash flow, pricing).",
        "Consider regulatory and legal risks.",
        "Evaluate technical and implementation risks.",
        "Rate risks by likelihood and potential impact.",
        "Suggest potential mitigation strategies for each major risk.",
        "Consider both short-term and long-term risks.",
    ],
    tools=[DuckDuckGo()],
    add_datetime_to_instructions=True,
)

# Business Validator
validator = Assistant(
    name="BusinessValidator",
    team=[market_researcher, financial_analyst, risk_analyst],
    description="You are a seasoned business consultant specializing in validating business ideas and providing strategic recommendations.",
    model=Gemini(id="gemini-1.5-flash"),
    instructions=[
        "First, ask the market researcher to analyze market potential and competitive landscape.",
        "Then, have the financial analyst assess financial viability and projections.",
        "Next, get the risk analyst to identify and evaluate potential risks.",
        "Synthesize all findings into a comprehensive validation report.",
        "Provide a clear GO/NO-GO recommendation with supporting rationale.",
        "Include specific action items and next steps if recommended to proceed.",
        "Be objective and data-driven in your analysis.",
        "Highlight both opportunities and challenges.",
        "Structure the report in clear sections: Market Analysis, Financial Assessment, Risk Analysis, Recommendations.",
    ],
    add_datetime_to_instructions=True,
    markdown=True,
)

#Example usage:
validator.print_response(
     """
     Validate this business idea:
     An AI-powered personal shopping assistant app that learns user preferences,
     tracks prices across multiple stores, and provides personalized recommendations
     for the best deals on fashion items.
     """,
     stream=True
)
