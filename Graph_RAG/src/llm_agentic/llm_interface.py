"""
LLM Interface for Agentic SQL Generation
Unified interface for different LLM providers
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


class AgenticLLMInterface:
    """Unified interface for LLM providers used in agentic SQL generation"""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
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
        Check if a query is complex and needs decomposition
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Analyze if this SQL query request is complex and needs decomposition into multiple steps.
            
            Query: {query}
            
            A query is COMPLEX if it needs:
            - Multiple aggregations at different levels
            - Data from 3+ tables with complex joins
            - Multiple CTEs or subqueries
            - Comparison of multiple time periods
            - Ranking combined with other operations
            - Complex calculations with multiple steps
            
            Return ONLY: true or false
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a SQL complexity analyzer. Return only 'true' or 'false'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                
                result = response.choices[0].message.content.strip().lower()
                return result == "true"
                
            except Exception as e:
                logger.error(f"Complexity check failed: {e}")
                return False
        else:
            # Simple heuristic
            complex_indicators = ['rank', 'compare', 'trend', 'correlation', 'multiple', 'analyze', 'comprehensive']
            return any(indicator in query.lower() for indicator in complex_indicators)
    
    async def analyze_query_similarity(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze if a historical SQL can be reused for current query
        """
        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a SQL similarity analyzer. Return only valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to parse JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Extract JSON from response if wrapped in markdown
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        return {
                            "can_reuse": False,
                            "reason": "Failed to parse response",
                            "confidence": 0
                        }
                        
            except Exception as e:
                logger.error(f"Query similarity analysis failed: {e}")
                return {
                    "can_reuse": False,
                    "reason": str(e),
                    "confidence": 0
                }
        else:
            # Fallback: simple keyword matching
            return {
                "can_reuse": False,
                "reason": "LLM not available for similarity analysis",
                "confidence": 0
            }
    
    async def generate_sql(self, query: str, schema_context: str) -> str:
        """
        Generate SQL for a simple query
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Generate SQL for this request:
            {query}
            
            Available Schema:
            {schema_context}
            
            Return only the SQL query, no explanations.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert SQL developer. Generate clean, efficient SQL queries."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                sql = response.choices[0].message.content.strip()
                # Remove markdown code blocks if present
                sql = re.sub(r'```sql?\n?', '', sql)
                sql = sql.replace('```', '')
                
                return sql
                
            except Exception as e:
                logger.error(f"SQL generation failed: {e}")
                return f"-- Error: {e}\nSELECT * FROM banks LIMIT 10"
        else:
            # Fallback to simple SQL
            return "SELECT * FROM banks LIMIT 10"
    
    async def decompose_query(self, query: str, schema_context: str) -> Dict[str, Any]:
        """
        Decompose complex query into tasks
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Decompose this complex SQL request into smaller tasks.
            
            Request: {query}
            
            Schema:
            {schema_context}
            
            Return a JSON object with:
            {{
                "tasks": [
                    {{
                        "task_id": "T1",
                        "description": "task description",
                        "sql_query": "SELECT ...",
                        "dependencies": [],
                        "execution_type": "sequential"
                    }}
                ],
                "execution_order": [["T1"], ["T2", "T3"]],
                "synthesis_instructions": "how to combine results"
            }}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a SQL query decomposer. Return only valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Extract JSON from markdown
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        raise ValueError("Failed to parse decomposition response")
                        
            except Exception as e:
                logger.error(f"Query decomposition failed: {e}")
                # Return fallback single task
                return {
                    "tasks": [
                        {
                            "task_id": "T1",
                            "description": "Execute full query",
                            "sql_query": f"-- {query}\nSELECT * FROM banks LIMIT 10",
                            "dependencies": [],
                            "execution_type": "sequential"
                        }
                    ],
                    "execution_order": [["T1"]],
                    "synthesis_instructions": "Return results as is"
                }
        else:
            # Fallback
            return {
                "tasks": [
                    {
                        "task_id": "T1",
                        "description": "Execute query",
                        "sql_query": "SELECT * FROM banks LIMIT 10",
                        "dependencies": [],
                        "execution_type": "sequential"
                    }
                ],
                "execution_order": [["T1"]],
                "synthesis_instructions": "Return as is"
            }
    
    async def synthesize_to_sql(self, original_query: str, tasks_description: str, 
                                synthesis_instructions: str, execution_order: List[List[str]]) -> str:
        """
        Synthesize multiple tasks into final SQL
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Combine these SQL tasks into a single, efficient SQL query.
            
            Original Request: {original_query}
            
            Tasks and SQL:
            {tasks_description}
            
            Synthesis Instructions: {synthesis_instructions}
            
            Create a single SQL query that accomplishes all tasks efficiently.
            Use CTEs or subqueries as needed.
            
            Return only the final SQL query.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a SQL synthesis expert. Combine multiple queries efficiently."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                sql = response.choices[0].message.content.strip()
                sql = re.sub(r'```sql?\n?', '', sql)
                sql = sql.replace('```', '')
                
                return sql
                
            except Exception as e:
                logger.error(f"SQL synthesis failed: {e}")
                return f"-- Synthesis failed: {e}\nSELECT * FROM banks LIMIT 10"
        else:
            return "-- No LLM available for synthesis\nSELECT * FROM banks LIMIT 10"
    
    async def modify_sql_with_context(self, task_description: str, 
                                      original_sql: str, dependency_context: str) -> str:
        """
        Modify SQL based on dependency results
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Modify this SQL based on dependency results.
            
            Task: {task_description}
            Original SQL: {original_sql}
            
            Dependency Results:
            {dependency_context}
            
            Return only the modified SQL.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Modify SQL based on context. Return only SQL."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                sql = response.choices[0].message.content.strip()
                sql = re.sub(r'```sql?\n?', '', sql)
                sql = sql.replace('```', '')
                
                return sql
                
            except Exception as e:
                logger.error(f"SQL modification failed: {e}")
                return original_sql
        else:
            return original_sql
    
    async def synthesize_results(self, original_query: str, 
                                 synthesis_instructions: str, results_summary: str) -> Dict[str, Any]:
        """
        Synthesize execution results into final answer
        """
        if self.provider == "openai" and self.client:
            prompt = f"""
            Synthesize these query results into a final answer.
            
            Original Question: {original_query}
            Instructions: {synthesis_instructions}
            
            Results:
            {results_summary}
            
            Return a JSON with:
            {{
                "answer": "main answer",
                "key_insights": ["insight1", "insight2"],
                "recommendations": ["rec1", "rec2"],
                "summary_metrics": {{"metric": value}},
                "final_sql": "the final SQL if applicable"
            }}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Synthesize query results. Return valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                
                content = response.choices[0].message.content.strip()
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        return {
                            "answer": "Results processed",
                            "key_insights": [],
                            "recommendations": [],
                            "summary_metrics": {},
                            "final_sql": None
                        }
                        
            except Exception as e:
                logger.error(f"Results synthesis failed: {e}")
                return {
                    "answer": "Query executed",
                    "key_insights": [],
                    "recommendations": [],
                    "summary_metrics": {},
                    "final_sql": None
                }
        else:
            return {
                "answer": "Query completed",
                "key_insights": ["Data retrieved"],
                "recommendations": [],
                "summary_metrics": {},
                "final_sql": None
            }