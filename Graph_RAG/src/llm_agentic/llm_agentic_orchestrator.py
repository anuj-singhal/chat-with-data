"""
LLM Agentic Orchestrator - Improved Version
Intelligent validation with suggestions instead of blocking
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from .llm_interface import AgenticLLMInterface
from .llm_task_decomposer import LLMTaskDecomposer
from .llm_sql_synthesizer import LLMSQLSynthesizer
from .llm_sql_validator import LLMSQLValidator

logger = logging.getLogger(__name__)


class LLMAgenticOrchestratorImproved:
    """
    Improved orchestrator with intelligent validation
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
        Process query with intelligent validation
        
        Args:
            query: User's query
            auto_approve_suggestions: Auto-approve validator suggestions
            
        Returns:
            Result with SQL and validation details
        """
        
        logger.info(f"Processing: {query[:100]}...")
        start_time = datetime.now()
        
        try:
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
            
            # Add to history
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
    
    async def _process_simple_with_validation(self, enhanced_query: str, original_query: str,
                                             context: Dict[str, Any], suggestions: List) -> Dict[str, Any]:
        """Process simple query with validation loop"""
        
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
            'message': f'Generated SQL with {attempts} validation attempt(s)'
        }
    
    async def _process_complex_with_validation(self, enhanced_query: str, original_query: str,
                                              context: Dict[str, Any], suggestions: List) -> Dict[str, Any]:
        """Process complex query with validation"""
        
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