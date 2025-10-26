"""
Comprehensive Test Queries for Banking Domain
Multiple variations of the same semantic query
"""

def get_comprehensive_test_queries():
    """
    Returns test queries grouped by complexity
    Each group contains variations of the same semantic query
    """
    
    return {
        'simple': [
            # Query Set 1: High NPL banks
            {
                'variations': [
                    "Show all banks with NPL ratio above 4%",
                    "List banks where non-performing loans exceed 4 percent",
                    "Which banks have NPL greater than 4?",
                    "Get banks with bad loans over 4%",
                    "Find banks where NPL ratio is more than 4"
                ],
                'expected_sql_pattern': 'SELECT.*FROM.*WHERE.*NPL.*>.*4'
            },
            # Query Set 2: Bank market caps
            {
                'variations': [
                    "Show market capitalization for all banks",
                    "List all banks with their market cap",
                    "What is the market value of each bank?",
                    "Get market capitalizations of banks",
                    "Display bank market caps"
                ],
                'expected_sql_pattern': 'SELECT.*market_cap.*FROM'
            },
            # Query Set 3: Specific bank data
            {
                'variations': [
                    "Show Mashreq bank's latest financial data",
                    "Get the most recent financials for Mashreq",
                    "What are Mashreq's latest numbers?",
                    "Display Mashreq bank recent performance",
                    "Find Mashreq's latest financial metrics"
                ],
                'expected_sql_pattern': 'SELECT.*FROM.*WHERE.*Mashreq'
            }
        ],
        
        'medium_complex': [
            # Query Set 1: Comparative analysis
            {
                'variations': [
                    "Compare deposit growth between Mashreq and ADCB for 2023",
                    "Show deposit changes for Mashreq vs ADCB in 2023",
                    "How did Mashreq and ADCB deposits grow in 2023?",
                    "Analyze deposit trends for Mashreq compared to ADCB for 2023",
                    "Get 2023 deposit growth comparison between Mashreq and ADCB"
                ],
                'expected_sql_pattern': 'SELECT.*deposit.*Mashreq.*ADCB.*2023'
            },
            # Query Set 2: Ranking with filters
            {
                'variations': [
                    "Rank banks by profitability for Q4 2023",
                    "Show top banks by profit in Q4 2023",
                    "Which banks were most profitable in Q4 2023?",
                    "List banks ordered by Q4 2023 profitability",
                    "Get bank profitability rankings for Q4 2023"
                ],
                'expected_sql_pattern': 'SELECT.*profit.*Q4.*2023.*ORDER BY'
            },
            # Query Set 3: Trend analysis
            {
                'variations': [
                    "Show quarterly NPL trend for top 5 banks",
                    "Get NPL changes over quarters for the 5 largest banks",
                    "How did NPL ratios change quarterly for top 5 banks?",
                    "Display quarterly NPL trends for the biggest 5 banks",
                    "Track NPL movement by quarter for top 5 banks"
                ],
                'expected_sql_pattern': 'SELECT.*NPL.*quarter.*TOP.*5|LIMIT 5'
            }
        ],
        
        'complex': [
            # Query Set 1: Risk analysis
            {
                'variations': [
                    "Analyze bank risk by combining NPL ratios with capital adequacy and rank them",
                    "Create risk assessment using non-performing loans and CET1 to rank banks",
                    "Rank banks by risk considering both asset quality (NPL) and capital strength (CET1)",
                    "Generate risk scores from NPL and capital ratios then rank all banks",
                    "Assess bank risk profiles using NPL and CET1 metrics with ranking"
                ],
                'expected_components': ['NPL calculation', 'CET1 retrieval', 'Risk scoring', 'Ranking']
            },
            # Query Set 2: Performance correlation
            {
                'variations': [
                    "Find correlation between CASA ratios and profitability across all banks with market cap analysis",
                    "Analyze how CASA impacts profits and include market capitalization insights",
                    "Study relationship between CASA, profitability, and market value for all banks",
                    "Correlate CASA ratios with bank profits and market caps",
                    "Examine CASA-profitability connection including market capitalization factors"
                ],
                'expected_components': ['CASA ratios', 'Profitability metrics', 'Market cap', 'Correlation']
            },
            # Query Set 3: Comprehensive health check
            {
                'variations': [
                    "Comprehensive bank health check: NPL trends, capital adequacy, deposit growth, and profitability ranking",
                    "Full analysis of bank health including asset quality, capital, deposits, and profits with rankings",
                    "Complete assessment: NPL patterns, CET1 levels, deposit changes, profit performance, then rank",
                    "Analyze all banks: bad loans trend, capital strength, deposit momentum, profitability, provide rankings",
                    "Evaluate bank health across NPL, capital, deposits, and profits with final ranking"
                ],
                'expected_components': ['NPL trends', 'Capital adequacy', 'Deposit analysis', 'Profitability', 'Final ranking']
            }
        ]
    }


def get_edge_case_queries():
    """
    Returns queries that test edge cases and validation
    """
    
    return {
        'ambiguous': [
            "Show performance",  # Which performance metric?
            "Get top banks",  # Top by what? How many?
            "Compare banks",  # Which banks? What metrics?
            "Show trends",  # What trends? What period?
        ],
        
        'time_sensitive': [
            "Show yesterday's data",  # Needs date calculation
            "Last quarter performance",  # Needs quarter identification
            "Year to date analysis",  # Needs YTD calculation
            "Compare this year vs last year",  # Needs year comparison
        ],
        
        'complex_aggregations': [
            "Moving average of NPL ratios",
            "Weighted average by asset size",
            "Percentile rankings across metrics",
            "Standard deviation of profits",
        ]
    }


def get_validation_test_queries():
    """
    Returns queries specifically designed to test validation
    """
    
    return {
        'will_need_correction': [
            # These might initially generate incorrect SQL
            "Average NPL for each bank",  # Might miss GROUP BY
            "Banks where profit > average",  # Subquery needed
            "Latest data for all metrics",  # Ambiguous 'latest'
        ],
        
        'will_need_suggestions': [
            # These will benefit from intelligent suggestions
            "Performance analysis",  # Needs metric suggestion
            "Risk assessment",  # Needs criteria suggestion
            "Competitive analysis",  # Needs comparison points
        ],
        
        'will_validate_well': [
            # These should generate correct SQL first time
            "SELECT bank_name, npl_ratio FROM financial_performance WHERE npl_ratio > 4",
            "List all banks ordered by market cap descending",
            "Show Q4 2023 profits for Mashreq bank"
        ]
    }