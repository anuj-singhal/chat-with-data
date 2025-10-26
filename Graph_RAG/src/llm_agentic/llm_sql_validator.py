"""
SQL Validator Module - Enhanced Version with Detailed Logging
Uses LLM for validation with intelligent suggestions and self-correction loop
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from groq import Groq
import os
import json
import re
from colorama import Fore, Style

logger = logging.getLogger(__name__)


class LLMSQLValidator:
    """
    Enhanced LLM-based SQL validator with intelligent suggestions and detailed logging
    """
    
    def __init__(self, schema_context: Dict[str, Any], groq_api_key: Optional[str] = None):
        """
        Initialize validator with schema context and Groq client
        """
        self.schema_context = schema_context
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        if self.groq_api_key:
            self.groq_client = Groq(api_key=self.groq_api_key)
            self.validation_model = "llama-3.3-70b-versatile"
        else:
            logger.warning("No Groq API key provided, validation will be limited")
            self.groq_client = None
    
    def _extract_json(self, response: str) -> dict:
        """Extract JSON from LLM response, handling markdown formatting"""
        try:
            # Direct parse attempt
            return json.loads(response)
        except:
            # Clean markdown
            cleaned = response.strip()
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
            
            # Find JSON object
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    json_str = match.group(0)
                    # Fix common issues
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    return json.loads(json_str)
                except:
                    pass
            
            logger.warning(f"Could not parse JSON from: {response[:200]}...")
            return {}
    
    async def intelligent_pre_validate(self, user_query: str, extracted_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligent pre-validation that provides suggestions instead of blocking
        """
        print(f"\n{Fore.CYAN}📝 Pre-validating query...{Style.RESET_ALL}")
        
        if not self.groq_client:
            print(f"  {Fore.YELLOW}⚠ Validation skipped (no Groq API key){Style.RESET_ALL}")
            return {
                'status': 'passed',
                'suggestions': [],
                'confidence': 100
            }
        
        prompt = f"""
Analyze this query and provide intelligent suggestions based on the schema.
DO NOT ask for clarifications unless absolutely necessary (entity doesn't exist).
Instead, make intelligent assumptions and provide suggestions.

USER QUERY: {user_query}

AVAILABLE SCHEMA:
{self._format_schema_for_validation(extracted_context)}

Rules:
1. If the query mentions "performance", assume they mean financial performance metrics
2. If time period is not specified, suggest using the latest available period
3. If "top" is mentioned without number, suggest top 5
4. If metric is ambiguous, suggest the most likely one based on context
5. Only flag as needs_clarification if entities don't exist in schema

Return ONLY a JSON object (no markdown):
{{
    "validation_passed": true/false,
    "suggestions": [
        {{
            "aspect": "time_period|metric|entity|aggregation",
            "assumption": "what I'm assuming",
            "suggestion": "Latest quarter (Q4 2023)"
        }}
    ],
    "confidence_score": 0-100
}}"""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.validation_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful SQL assistant. Make intelligent assumptions. Return only JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = self._extract_json(response.choices[0].message.content)
            
            status = 'passed' if result.get('validation_passed', True) else 'needs_attention'
            suggestions = result.get('suggestions', [])
            confidence = result.get('confidence_score', 80)
            
            if suggestions:
                print(f"  {Fore.YELLOW}💡 {len(suggestions)} suggestion(s) generated{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}💡 {suggestions}" )
            else:
                print(f"  {Fore.GREEN}✓ No suggestions needed{Style.RESET_ALL}")
            
            return {
                'status': status,
                'suggestions': suggestions,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Pre-validation failed: {e}")
            print(f"  {Fore.RED}✗ Pre-validation failed: {e}{Style.RESET_ALL}")
            return {'status': 'passed', 'suggestions': [], 'confidence': 100}
    
    async def validate_sql(self, generated_sql: str, original_query: str) -> Dict[str, Any]:
        """
        Comprehensive SQL validation with detailed logging
        """
        
        if not self.groq_client:
            return {
                'valid': True,
                'confidence': 100,
                'validations': {
                    'schema': 100,
                    'semantic': 100,
                    'syntax': 100,
                    'completeness': 100
                }
            }
        
        prompt = f"""
Validate this SQL against the request and schema. Check multiple aspects.

ORIGINAL REQUEST: {original_query}

GENERATED SQL:
{generated_sql}

SCHEMA:
{self._format_schema_for_validation(self.schema_context)}

Validate these aspects:
1. Schema Validation: Do all tables/columns exist? Are joins correct?
2. Semantic Validation: Does the SQL answer the user's question?
3. Syntax Validation: Is the SQL syntactically correct?
4. Completeness: Does it retrieve all requested information?

Return ONLY a JSON object (no markdown):
{{
    "is_valid": true/false,
    "validations": {{
        "schema": 0-100,
        "semantic": 0-100,
        "syntax": 0-100,
        "completeness": 0-100
    }},
    "overall_confidence": 0-100,
    "issues": [
        {{
            "type": "error|warning",
            "category": "schema|semantic|syntax|completeness",
            "description": "issue description",
            "fix": "how to fix"
        }}
    ]
}}"""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.validation_model,
                messages=[
                    {
                        "role": "system",
                        "content": "SQL validation expert. Return only JSON, no markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result = self._extract_json(response.choices[0].message.content)
            
            return {
                'valid': result.get('is_valid', False),
                'confidence': result.get('overall_confidence', 0),
                'validations': result.get('validations', {}),
                'issues': result.get('issues', [])
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'valid': False,
                'confidence': 0,
                'error': str(e)
            }
    
    async def self_correct_with_loop(self, 
                                    sql: str, 
                                    original_query: str,
                                    openai_interface,
                                    max_attempts: int = 3) -> Dict[str, Any]:
        """
        Self-correction loop between OpenAI regeneration and Groq validation
        """
        print(f"\n{Fore.CYAN}🔄 Starting self-correction loop (max {max_attempts} attempts)...{Style.RESET_ALL}")
        
        best_sql = sql
        best_validation = None
        attempt_history = []
        
        for attempt in range(max_attempts):
            print(f"\n{Fore.BLUE}  Attempt {attempt + 1}/{max_attempts}:{Style.RESET_ALL}")
            logger.info(f"Self-correction attempt {attempt + 1}/{max_attempts}")
            
            # Validate current SQL
            print(f"    Validating SQL...")
            validation = await self.validate_sql(best_sql, original_query)
            
            # Display validation scores
            if validation.get('validations'):
                for metric, score in validation['validations'].items():
                    if score >= 80:
                        color = Fore.GREEN
                        symbol = "✓"
                    elif score >= 60:
                        color = Fore.YELLOW
                        symbol = "⚠"
                    else:
                        color = Fore.RED
                        symbol = "✗"
                    print(f"      {color}{symbol} {metric}: {score}%{Style.RESET_ALL}")
            
            confidence = validation.get('confidence', 0)
            print(f"    Overall confidence: {confidence}%")
            
            attempt_history.append({
                'attempt': attempt + 1,
                'sql': best_sql,
                'validation': validation
            })
            
            # If valid with high confidence, we're done
            if validation['valid'] and confidence >= 80:
                print(f"{Fore.GREEN}  ✓ SQL validated successfully!{Style.RESET_ALL}")
                logger.info(f"SQL validated successfully on attempt {attempt + 1}")
                return {
                    'success': True,
                    'final_sql': best_sql,
                    'validation': validation,
                    'attempts': attempt + 1,
                    'history': attempt_history
                }
            
            # Store best validation so far
            if not best_validation or validation.get('confidence', 0) > best_validation.get('confidence', 0):
                best_validation = validation
            
            # If we're not on the last attempt and SQL is invalid, regenerate with OpenAI
            if attempt < max_attempts - 1 and not validation['valid']:
                print(f"{Fore.YELLOW}    Issues found, requesting regeneration...{Style.RESET_ALL}")
                logger.info("Asking OpenAI to regenerate based on issues")
                
                # Format issues for regeneration
                if validation.get('issues'):
                    print(f"    Issues to fix:")
                    for issue in validation['issues'][:3]:  # Show first 3 issues
                        print(f"      - {issue['category']}: {issue['description']}")
                
                issues_text = "\n".join([
                    f"- {issue['category']}: {issue['description']} (Fix: {issue.get('fix', 'N/A')})"
                    for issue in validation.get('issues', [])
                ])
                
                regeneration_prompt = f"""
    The SQL has validation issues. Please regenerate it.

    Original Query: {original_query}

    Current SQL: {best_sql}

    Validation Issues:
    {issues_text}

    Generate a corrected SQL that fixes these issues. Return only the SQL, no explanations.
    """
                
                # Regenerate using OpenAI
                new_sql = await openai_interface.generate_sql(
                    regeneration_prompt,
                    self._format_schema_for_validation(self.schema_context)
                )
                
                best_sql = new_sql
                print(f"    Regenerated SQL based on validation feedback")
                logger.info("OpenAI regenerated SQL based on validation feedback")
        
        # Max attempts reached
        print(f"\n{Fore.YELLOW}  Max attempts reached. Using best available SQL.{Style.RESET_ALL}")
        return {
            'success': best_validation and best_validation['valid'],
            'final_sql': best_sql,
            'validation': best_validation,
            'attempts': max_attempts,
            'history': attempt_history,
            'message': f'Completed {max_attempts} validation attempts'
        }

    def _format_schema_for_validation(self, context: Dict[str, Any]) -> str:
        """Format schema context for validation prompts"""
        if not context:
            context = self.schema_context
            
        schema_text = []
        
        for table_name, table_info in context.get('tables', {}).items():
            schema_text.append(f"Table: {table_name}")
            
            columns = table_info.get('columns', [])
            if columns:
                for col in columns[:5]:  # Limit columns to reduce token usage
                    schema_text.append(f"  - {col['name']} ({col['type']})")
        
        return "\n".join(schema_text)