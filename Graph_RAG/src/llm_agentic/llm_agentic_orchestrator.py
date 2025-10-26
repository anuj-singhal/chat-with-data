"""
LLM Agentic Orchestrator - Enhanced Version
Includes query history similarity checking
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from colorama import Fore, Style
from .llm_interface import AgenticLLMInterface
from .llm_task_decomposer import LLMTaskDecomposer
from .llm_sql_synthesizer import LLMSQLSynthesizer
from .llm_sql_validator import LLMSQLValidator

logger = logging.getLogger(__name__)


class LLMAgenticOrchestratorImproved:
    """
    Enhanced orchestrator with query history checking and intelligent validation
    """
    
    def __init__(self, rag_system=None, 
                 llm_provider: str = "openai", llm_api_key: Optional[str] = None,
                 llm_model: Optional[str] = None, groq_api_key: Optional[str] = None):
        """
        Initialize the orchestrator with validation
        """
        
        self.llm_interface = AgenticLLMInterface(
            provider=llm_provider,
            api_key=llm_api_key or os.getenv("OPENAI_API_KEY"),
            model=llm_model
        )
        
        self.rag_system = rag_system
        self.decomposer = LLMTaskDecomposer(self.llm_interface, rag_system)
        self.synthesizer = LLMSQLSynthesizer(self.llm_interface)
        
        # Initialize improved validator
        self.validator = None
        if groq_api_key or os.getenv("GROQ_API_KEY"):
            self.validator = LLMSQLValidator(
                schema_context={},
                groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY")
            )
            logger.info("Validator initialized with Groq")
        
        self.query_history = []
    
    async def process_query(self, query: str, auto_approve_suggestions: bool = True) -> Dict[str, Any]:
        """
        Process query with query history checking and intelligent validation
        
        Args:
            query: User's query
            auto_approve_suggestions: Auto-approve validator suggestions
            
        Returns:
            Result with SQL and validation details
        """
        
        logger.info(f"Processing: {query[:100]}...")
        start_time = datetime.now()
        
        try:
            # Step 1: Check query history for similar queries
            similar_query = await self._check_query_history(query)
            
            if similar_query:
                logger.info(f"Found similar query with {similar_query['similarity_score']:.2%} match")
                
                # Ask LLM if the historical SQL can be reused
                can_reuse = await self._verify_sql_reuse(query, similar_query)
                
                if can_reuse['can_reuse']:
                    logger.info("Reusing SQL from query history")
                    return {
                        'success': True,
                        'query': query,
                        'query_type': 'cached',
                        'sql': similar_query['sql_query'],
                        'validation': similar_query.get('validation_result', {}),
                        'validation_confidence': 100,
                        'from_cache': True,
                        'cache_query_id': similar_query['query_id'],
                        'similarity_score': similar_query['similarity_score'],
                        'message': f'Used cached SQL from similar query (ID: {similar_query["query_id"]})'
                    }
                else:
                    logger.info("Similar query found but SQL needs regeneration")
            
            # Step 2: Continue with regular processing
            # Get RAG context
            context = self._get_rag_context(query)
            
            # Update validator context
            if self.validator:
                self.validator.schema_context = context
            
            # Pre-validation with intelligent suggestions
            suggestions = []
            if self.validator:
                pre_result = await self.validator.intelligent_pre_validate(query, context)
                suggestions = pre_result.get('suggestions', [])
                
                if suggestions and auto_approve_suggestions:
                    # Apply suggestions to query
                    enhanced_query = self._apply_suggestions(query, suggestions)
                    logger.info(f"Applied {len(suggestions)} suggestions")
                else:
                    enhanced_query = query
            else:
                enhanced_query = query
            
            # Check complexity
            needs_decomposition = await self._check_needs_decomposition(enhanced_query)
            
            if not needs_decomposition:
                # Simple query
                result = await self._process_simple_with_validation(
                    enhanced_query, query, context, suggestions
                )
            else:
                # Complex query
                result = await self._process_complex_with_validation(
                    enhanced_query, query, context, suggestions
                )
            
            # Step 3: Add successful query to history
            if result['success'] and self.rag_system:
                overall_confidence = result.get('validation_confidence', 100)
                
                # Only add if confidence > 90%
                if overall_confidence > 90:
                    print(f"\n{Fore.CYAN}💾 Saving to query history...{Style.RESET_ALL}")
                    
                    validation_result = {
                        'schema': result.get('validation', {}).get('schema', 100),
                        'syntax': result.get('validation', {}).get('syntax', 100),
                        'semantic': result.get('validation', {}).get('semantic', 100),
                        'completeness': result.get('validation', {}).get('completeness', 100)
                    }
                    
                    self.rag_system.add_validated_query_to_history(
                        nl_query=query,
                        sql_query=result['sql'],
                        validation_result=validation_result,
                        overall_confidence=overall_confidence,
                        variations=[]
                    )
            
            # Add to local history
            self.query_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def _check_query_history(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check if a similar query exists in history
        """
        if not self.rag_system:
            return None
        
        try:
            # Look for similar queries with 90% threshold
            similar = self.rag_system.find_similar_query_enhanced(query, threshold=0.90)
            return similar
        except Exception as e:
            logger.error(f"Query history check failed: {e}")
            return None
    
    async def _verify_sql_reuse(self, current_query: str, similar_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask LLM if the historical SQL can be reused for current query
        """
        print(f"\n{Fore.CYAN}🤔 Verifying if cached SQL can be reused...{Style.RESET_ALL}")
        print(f"  Original: '{similar_query['natural_language'][:60]}...'")
        print(f"  Current:  '{current_query[:60]}...'")
        
        prompt = f"""
        Determine if an existing SQL query can be reused for a new request.
        
        ORIGINAL QUERY: {similar_query['natural_language']}
        ORIGINAL SQL: {similar_query['sql_query']}
        
        NEW QUERY: {current_query}
        
        Analyze if the SQL can be reused as-is or needs modification.
        Consider:
        1. Are the entities (tables, columns) the same?
        2. Are the filters/conditions equivalent?
        3. Is the aggregation/grouping the same?
        4. Is the time period the same or compatible?
        
        Return JSON:
        {{
            "can_reuse": true/false,
            "reason": "explanation",
            "confidence": 0-100
        }}
        """
        
        try:
            response = await self.llm_interface.analyze_query_similarity(prompt)
            
            # Parse response
            if isinstance(response, dict):
                can_reuse = response.get('can_reuse', False)
                reason = response.get('reason', '')
                confidence = response.get('confidence', 0)
                
                if can_reuse:
                    print(f"{Fore.GREEN}  ✓ SQL can be reused (confidence: {confidence}%){Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}  ✗ SQL needs regeneration: {reason}{Style.RESET_ALL}")
                
                return response
            else:
                # Default to not reusing if parsing fails
                print(f"{Fore.YELLOW}  ✗ Unable to verify similarity{Style.RESET_ALL}")
                return {
                    "can_reuse": False,
                    "reason": "Unable to determine similarity",
                    "confidence": 0
                }
        except Exception as e:
            logger.error(f"SQL reuse verification failed: {e}")
            print(f"{Fore.RED}  ✗ Verification failed: {e}{Style.RESET_ALL}")
            return {
                "can_reuse": False,
                "reason": str(e),
                "confidence": 0
            }
    
    async def _process_simple_with_validation(self, enhanced_query: str, original_query: str,
                                             context: Dict[str, Any], suggestions: List) -> Dict[str, Any]:
        """Process simple query with validation loop"""
        print("Process Simple Query...")
        schema_context = self._format_schema_context(context)
        
        # Generate SQL
        sql = await self.llm_interface.generate_sql(enhanced_query, schema_context)
        
        # Validation loop if validator available
        if self.validator:
            correction_result = await self.validator.self_correct_with_loop(
                sql, original_query, self.llm_interface, max_attempts=3
            )
            
            final_sql = correction_result['final_sql']
            validation = correction_result.get('validation', {})
            attempts = correction_result['attempts']
        else:
            final_sql = sql
            validation = {'validations': {}}
            attempts = 1
        
        return {
            'success': True,
            'query': original_query,
            'enhanced_query': enhanced_query if suggestions else None,
            'query_type': 'simple',
            'sql': final_sql,
            'suggestions_applied': suggestions,
            'validation': validation.get('validations', {}),
            'validation_confidence': validation.get('confidence', 100),
            'correction_attempts': attempts,
            'from_cache': False,
            'message': f'Generated SQL with {attempts} validation attempt(s)'
        }
    
    async def _process_complex_with_validation(self, enhanced_query: str, original_query: str,
                                              context: Dict[str, Any], suggestions: List) -> Dict[str, Any]:
        """Process complex query with validation"""
        print("Process Complex Query...")
        # Decompose
        plan = await self.decomposer.decompose_query(enhanced_query)
        
        # Synthesize
        final_sql = await self.synthesizer.synthesize_sql_plan(plan)
        
        # Validation loop
        if self.validator:
            correction_result = await self.validator.self_correct_with_loop(
                final_sql, original_query, self.llm_interface, max_attempts=3
            )
            
            final_sql = correction_result['final_sql']
            validation = correction_result.get('validation', {})
            attempts = correction_result['attempts']
        else:
            validation = {'validations': {}}
            attempts = 1
        
        # Format task details
        sql_queries = [
            {
                'task_id': task.task_id,
                'description': task.description,
                'sql': task.sql_query,
                'dependencies': task.dependencies,
                'execution_type': task.execution_type
            }
            for task in plan.tasks
        ]
        
        return {
            'success': True,
            'query': original_query,
            'enhanced_query': enhanced_query if suggestions else None,
            'query_type': 'complex',
            'sql': final_sql,
            'sql_queries': sql_queries,
            'execution_order': plan.execution_order,
            'suggestions_applied': suggestions,
            'validation': validation.get('validations', {}),
            'validation_confidence': validation.get('confidence', 100),
            'correction_attempts': attempts,
            'from_cache': False,
            'message': f'Generated complex SQL with {len(sql_queries)} tasks and {attempts} validation attempt(s)'
        }
    
    async def _check_needs_decomposition(self, query: str) -> bool:
        """Check if query needs decomposition"""
        try:
            return await self.llm_interface.check_complexity(query)
        except:
            return False
    
    def _get_rag_context(self, query: str) -> Dict[str, Any]:
        """Get RAG context"""
        if not self.rag_system:
            return {}
        
        search_results = self.rag_system.search_relevant_tables(query)
        return self.rag_system.build_context(search_results['tables'], query)
    
    def _format_schema_context(self, context: Dict[str, Any]) -> str:
        """Format schema for LLM"""
        schema_text = []
        
        for table_name, table_info in context.get('tables', {}).items():
            schema_text.append(f"\nTable: {table_name}")
            schema_text.append(f"Description: {table_info['description']}")
            schema_text.append("Columns:")
            
            for col in table_info['columns']:
                col_text = f"  - {col['name']} ({col['type']}): {col['description']}"
                if col.get('samples'):
                    col_text += f" (Examples: {col['samples'][:2]})"
                schema_text.append(col_text)
        
        return "\n".join(schema_text)
    
    def _apply_suggestions(self, query: str, suggestions: List[Dict]) -> str:
        """Apply validator suggestions to enhance query"""
        enhanced = query
        
        for suggestion in suggestions:
            if suggestion.get('suggestion'):
                enhanced += f"\n- {suggestion['aspect']}: {suggestion['suggestion']}"
        
        return enhanced