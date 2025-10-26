"""
LLM Interface for Agentic System
Provides proper methods for different types of LLM interactions
"""

import json
import logging
from typing import Dict, Any, Optional, List
from openai import OpenAI
import re

logger = logging.getLogger(__name__)


class AgenticLLMInterface:
    """
    Proper interface for LLM interactions in agentic system
    Separates SQL generation from reasoning/decision making
    """
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, 
                 model: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        
        if self.provider == "openai":
            self.model = model or "gpt-4o-mini"
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
            else:
                logger.warning("No OpenAI API key provided")
                self.client = None
        else:
            self.model = None
            self.client = None
    
    async def check_complexity(self, query: str) -> bool:
        """
        Check if a query needs decomposition
        Returns True for complex queries, False for simple ones
        """
        
        prompt = f"""
Analyze this database query and determine if it needs to be broken down into multiple SQL tasks.

QUERY: {query}

A query needs decomposition if it:
- Requires multiple distinct data retrievals from different tables
- Needs intermediate calculations before the final result
- Involves complex comparisons across different metrics
- Requires analyzing trends over multiple time periods
- Needs conditional logic (if X then check Y)
- Asks for rankings based on calculated metrics
- Requires correlation analysis between different factors

Simple queries that DON'T need decomposition:
- Basic SELECT statements
- Single table aggregations
- Direct lookups or filters
- Simple calculations on one table

Respond with only "COMPLEX" or "SIMPLE" based on your analysis.
"""
        
        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a SQL query analyzer. Respond with only 'COMPLEX' or 'SIMPLE'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                
                result = response.choices[0].message.content.strip().upper()
                logger.info(f"Complexity check result: {result}")
                return "COMPLEX" in result
                
            except Exception as e:
                logger.error(f"OpenAI complexity check failed: {e}")
                return self._heuristic_complexity_check(query)
        else:
            return self._heuristic_complexity_check(query)
    
    def _heuristic_complexity_check(self, query: str) -> bool:
        """Fallback heuristic for complexity checking"""
        query_lower = query.lower()
        
        # Complex indicators
        complex_indicators = [
            'analyze', 'compare', 'trend', 'correlation',
            'rank', 'forecast', 'predict', 'multiple',
            'across all', 'over time', 'quarter', 'year',
            'then', 'finally', 'based on', 'identify'
        ]
        
        # Count indicators
        indicator_count = sum(1 for ind in complex_indicators if ind in query_lower)
        
        # Check for numbered steps
        has_steps = any(f"{i}." in query or f"{i})" in query for i in range(1, 10))
        
        # Check query length
        is_long = len(query.split()) > 30
        
        # Decision
        is_complex = indicator_count >= 2 or has_steps or is_long
        
        logger.info(f"Heuristic complexity: indicators={indicator_count}, steps={has_steps}, complex={is_complex}")
        return is_complex
    
    async def decompose_query(self, query: str, schema_context: str) -> Dict[str, Any]:
        """
        Decompose a complex query into tasks
        Returns structured plan with SQL tasks
        """
        
        prompt = f"""
You are an expert SQL query planner. Break down this complex query into smaller, executable SQL tasks.

ORIGINAL QUERY: {query}

DATABASE SCHEMA:
{schema_context}

Create a detailed execution plan with actual SQL queries that can be executed.
Each task should be a complete, valid SQL statement.

Return a JSON object with this EXACT structure (no additional text):
{{
    "tasks": [
        {{
            "task_id": "T1",
            "description": "What this task accomplishes",
            "sql_query": "SELECT ... FROM ... WHERE ...",
            "dependencies": [],
            "execution_type": "parallel"
        }},
        {{
            "task_id": "T2", 
            "description": "What this task accomplishes",
            "sql_query": "SELECT ... FROM ... WHERE ...",
            "dependencies": ["T1"],
            "execution_type": "sequential"
        }}
    ],
    "execution_order": [
        ["T1", "T3"],
        ["T2"],
        ["T4"]
    ],
    "synthesis_instructions": "How to combine all results to answer the original query"
}}

IMPORTANT RULES:
1. Create real, executable SQL queries using the actual table and column names from the schema
2. Use proper JOIN syntax when combining tables
3. Include GROUP BY when using aggregate functions
4. Tasks that don't depend on each other can run in parallel
5. Tasks that need results from others must be marked sequential with dependencies
"""

        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a SQL expert. Return only valid JSON with SQL tasks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                return self._parse_json_response(content)
                
            except Exception as e:
                logger.error(f"OpenAI decomposition failed: {e}")
                return self._create_fallback_decomposition(query)
        else:
            return self._create_fallback_decomposition(query)
    
    async def modify_sql_with_context(self, task_description: str, 
                                      original_sql: str, 
                                      dependency_results: str) -> str:
        """
        Modify SQL based on results from dependencies
        """
        
        prompt = f"""
Given a SQL task and results from its dependencies, modify the SQL if needed.

TASK: {task_description}
ORIGINAL SQL: {original_sql}

DEPENDENCY RESULTS:
{dependency_results}

If the SQL needs to use specific values from the dependency results (like IDs, thresholds, or calculated values),
modify it accordingly. Otherwise, return the original SQL.

Return ONLY the SQL query, no explanations.
"""

        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return only SQL queries, no explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                sql = response.choices[0].message.content.strip()
                # Clean up SQL
                sql = sql.replace("```sql", "").replace("```", "").strip()
                return sql
                
            except Exception as e:
                logger.error(f"SQL modification failed: {e}")
                return original_sql
        else:
            return original_sql
    
    async def synthesize_results(self, original_query: str, 
                                 synthesis_instructions: str,
                                 task_results: str) -> Dict[str, Any]:
        """
        Synthesize all task results into final answer
        """
        
        prompt = f"""
Synthesize the results from multiple SQL tasks to answer the original query.

ORIGINAL QUERY: {original_query}

SYNTHESIS INSTRUCTIONS: {synthesis_instructions}

TASK RESULTS:
{task_results}

Provide a comprehensive answer with insights and recommendations.

Return a JSON object with this structure:
{{
    "answer": "Complete answer to the original query with specific details from the results",
    "key_insights": [
        "First key insight with specific numbers/names",
        "Second key insight",
        "Third key insight"
    ],
    "recommendations": [
        "Actionable recommendation based on the analysis"
    ],
    "summary_metrics": {{
        "metric_name": "value",
        "another_metric": "value"
    }},
    "final_sql": "Optional: A single SQL query that would get the complete result"
}}
"""

        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a data analyst. Provide insights based on the SQL results."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )
                
                content = response.choices[0].message.content.strip()
                return self._parse_json_response(content)
                
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                return self._create_fallback_synthesis(task_results)
        else:
            return self._create_fallback_synthesis(task_results)
    
    async def generate_sql(self, query: str, schema_context: str) -> str:
        """
        Generate a simple SQL query (for non-complex queries)
        """
        
        prompt = f"""
Generate a SQL query for this request:

REQUEST: {query}

DATABASE SCHEMA:
{schema_context}

Return ONLY the SQL query, no explanations or markdown.
"""

        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a SQL expert. Return only SQL queries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                sql = response.choices[0].message.content.strip()
                sql = sql.replace("```sql", "").replace("```", "").strip()
                return sql
                
            except Exception as e:
                logger.error(f"SQL generation failed: {e}")
                return "SELECT * FROM banks LIMIT 10"
        else:
            return self._generate_heuristic_sql(query)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Clean response
            cleaned = response.strip()
            
            # Remove markdown code blocks
            if "```json" in cleaned:
                match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            elif "```" in cleaned:
                cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
            
            # Parse JSON
            return json.loads(cleaned)
            
        except json.JSONDecodeError:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            logger.error(f"Failed to parse JSON from response: {response[:200]}...")
            return {}
    
    def _create_fallback_decomposition(self, query: str) -> Dict[str, Any]:
        """Create fallback decomposition when LLM fails"""
        return {
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "Retrieve all relevant data",
                    "sql_query": "SELECT * FROM banks JOIN financial_performance ON banks.bank_id = financial_performance.bank_id",
                    "dependencies": [],
                    "execution_type": "parallel"
                }
            ],
            "execution_order": [["T1"]],
            "synthesis_instructions": "Return the query results"
        }
    
    def _create_fallback_synthesis(self, results: str) -> Dict[str, Any]:
        """Create fallback synthesis when LLM fails"""
        return {
            "answer": "Query executed successfully. Results processed.",
            "key_insights": ["Data retrieved successfully"],
            "recommendations": [],
            "summary_metrics": {},
            "final_sql": None
        }
    
    def _generate_heuristic_sql(self, query: str) -> str:
        """Generate SQL using heuristics"""
        query_lower = query.lower()
        
        if "npl" in query_lower:
            return "SELECT bank_name, npl_ratio FROM financial_performance ORDER BY npl_ratio DESC"
        elif "market" in query_lower:
            return "SELECT bank_name, market_cap FROM market_data ORDER BY market_cap DESC"
        elif "deposit" in query_lower:
            return "SELECT bank_name, total_deposits FROM financial_performance"
        else:
            return "SELECT * FROM banks LIMIT 10"    
    async def synthesize_to_sql(self, original_query: str,
                                tasks_description: str,
                                synthesis_instructions: str,
                                execution_order: List[List[str]]) -> str:
        """
        Synthesize multiple SQL tasks into a final SQL query
        """
        
        prompt = f"""
You are an expert SQL developer. You have decomposed a complex query into multiple SQL tasks.
Now synthesize these tasks into a FINAL SQL query that answers the original request.

{tasks_description}

IMPORTANT RULES:
1. Create a single, executable SQL query that combines all the logic from the tasks
2. Use CTEs (Common Table Expressions) to organize the query logically
3. Ensure the final query returns the data needed to answer the original question
4. The query should be optimized and efficient
5. Include appropriate JOINs, aggregations, and filters as shown in the tasks
6. Return ONLY the final SQL query, no explanations

Generate the final SQL query:
"""

        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a SQL expert. Return only SQL queries, no explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )
                
                sql = response.choices[0].message.content.strip()
                sql = sql.replace("```sql", "").replace("```", "").strip()
                return sql
                
            except Exception as e:
                logger.error(f"SQL synthesis failed: {e}")
                return self._create_fallback_combined_sql(tasks_description)
        else:
            return self._create_fallback_combined_sql(tasks_description)
    
    def _create_fallback_combined_sql(self, tasks_description: str) -> str:
        """Create a fallback combined SQL when LLM fails"""
        # Extract SQL queries from tasks description
        import re
        sql_pattern = r'SQL Query:\s*\n(.*?)(?:\n-{40}|\Z)'
        matches = re.findall(sql_pattern, tasks_description, re.DOTALL)
        
        if not matches:
            return "-- Unable to synthesize SQL queries"
        
        if len(matches) == 1:
            return matches[0].strip()
        
        # Create CTEs from the SQL queries
        cte_parts = []
        for i, sql in enumerate(matches, 1):
            if i == 1:
                cte_parts.append(f"WITH task_{i} AS (")
            else:
                cte_parts.append(f"),\ntask_{i} AS (")
            cte_parts.append(f"  {sql.strip()}")
        
        cte_parts.append(")")
        cte_parts.append(f"SELECT * FROM task_{len(matches)}")
        
        return "\n".join(cte_parts)