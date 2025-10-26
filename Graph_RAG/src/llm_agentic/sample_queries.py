"""
Sample Complex Queries for LLM-driven Agentic System
"""

COMPLEX_QUERIES = [
    """
    Perform a comprehensive analysis of all banks' financial health:
    1. Calculate the average NPL ratio for each bank over the last 4 quarters
    2. Identify banks with deteriorating asset quality (NPL increasing quarter-over-quarter)
    3. Compare deposit growth rates between banks
    4. Rank banks by their risk-adjusted returns (ROTE vs NPL ratio)
    5. Find correlations between CASA ratios and profitability
    Finally, identify the top 3 banks for investment based on all these factors.
    """,
    
    """
    Analyze market position and competitive landscape:
    First, identify the top 5 banks by market capitalization.
    Then for each of these banks:
    - Calculate their quarterly profit growth rate
    - Analyze their cost efficiency trends (cost-to-income ratio)
    - Compare their capital adequacy (CET1) against regulatory minimums
    - Evaluate share price performance over the last year
    Create a competitive ranking and predict which bank is best positioned for growth.
    """,
    
    """
    Conduct a stress test analysis:
    1. For each bank, calculate the NPL coverage ratio (provisions/NPL)
    2. Identify banks with capital buffers less than 2% above minimum requirements
    3. Find banks with high dependency on volatile funding (low CASA ratios)
    4. Calculate the correlation between NPL ratios and economic indicators
    5. Rank banks by their resilience score based on all factors
    Provide recommendations for the 3 most vulnerable banks.
    """,
    
    """
    Create a year-over-year performance comparison:
    Compare Q4 2023 with Q4 2022 for all banks across:
    - Revenue growth
    - Profit margins
    - Asset quality changes (NPL)
    - Capital adequacy improvements (CET1)
    - Market valuation changes (P/B ratio)
    Identify the best and worst performing banks with detailed reasoning.
    """,
    
    """
    Analyze operational efficiency across the banking sector:
    1. Calculate the trend in cost-to-income ratios over 8 quarters
    2. Identify banks with consistent efficiency improvements
    3. Find the correlation between efficiency and bank size (total assets)
    4. Compare efficiency metrics against profitability (ROTE)
    5. Predict which banks will improve efficiency next quarter
    Create an efficiency scorecard for all banks.
    """
]

SIMPLE_QUERIES = [
    "Show all banks with their current market capitalization",
    "What is the NPL ratio for Mashreq bank in Q4 2023?",
    "List banks with CET1 ratio above 14%",
    "Calculate the average deposits across all banks",
    "Show the quarterly profit trend for ADCB"
]


def get_test_queries():
    """Get a mix of simple and complex queries for testing"""
    return {
        'simple': SIMPLE_QUERIES,
        'complex': COMPLEX_QUERIES
    }