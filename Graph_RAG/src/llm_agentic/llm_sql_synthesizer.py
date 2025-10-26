"""
LLM SQL Synthesizer
Synthesizes multiple SQL queries from decomposed tasks into final SQL
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .llm_interface import AgenticLLMInterface
from .llm_task_decomposer import AgenticPlan, AgenticTask

logger = logging.getLogger(__name__)


class LLMSQLSynthesizer:
    """Synthesizes SQL queries from decomposed tasks"""
    
    def __init__(self, llm_interface: AgenticLLMInterface):
        self.llm = llm_interface
        
    async def synthesize_sql_plan(self, plan: AgenticPlan) -> str:
        """
        Synthesize all SQL tasks into a final SQL query
        
        Args:
            plan: The decomposed plan with SQL tasks
            
        Returns:
            Final synthesized SQL query
        """
        logger.info(f"Synthesizing {len(plan.tasks)} SQL tasks into final query")
        
        try:
            # Format task SQLs for synthesis
            tasks_description = self._format_tasks_for_synthesis(plan)
            
            # Ask LLM to synthesize into final SQL
            final_sql = await self.llm.synthesize_to_sql(
                plan.original_query,
                tasks_description,
                plan.synthesis_instructions,
                plan.execution_order
            )
            
            if not final_sql:
                logger.warning("LLM synthesis returned empty result")
                return self._create_fallback_synthesis(plan)
            
            return final_sql
            
        except Exception as e:
            logger.error(f"SQL synthesis failed: {e}")
            return self._create_fallback_synthesis(plan)
    
    def _format_tasks_for_synthesis(self, plan: AgenticPlan) -> str:
        """
        Format all task SQLs for LLM synthesis
        
        Args:
            plan: The execution plan with tasks
            
        Returns:
            Formatted string describing all tasks and their SQL
        """
        formatted = []
        
        # Add original query
        formatted.append(f"Original Request: {plan.original_query}")
        formatted.append("\nDecomposed SQL Tasks:")
        formatted.append("=" * 40)
        
        # Format each task
        for task in plan.tasks:
            formatted.append(f"\nTask {task.task_id}: {task.description}")
            formatted.append(f"Execution Type: {task.execution_type}")
            
            if task.dependencies:
                formatted.append(f"Dependencies: {', '.join(task.dependencies)}")
            else:
                formatted.append("Dependencies: None (can run in parallel)")
            
            formatted.append(f"SQL Query:")
            formatted.append(task.sql_query)
            formatted.append("-" * 40)
        
        # Add execution order
        formatted.append("\nExecution Order:")
        for batch_idx, batch in enumerate(plan.execution_order, 1):
            if len(batch) > 1:
                formatted.append(f"  Batch {batch_idx} (parallel): {', '.join(batch)}")
            else:
                formatted.append(f"  Step {batch_idx}: {batch[0]}")
        
        # Add synthesis instructions
        if plan.synthesis_instructions:
            formatted.append(f"\nSynthesis Instructions: {plan.synthesis_instructions}")
        
        return "\n".join(formatted)
    
    def _create_fallback_synthesis(self, plan: AgenticPlan) -> str:
        """
        Create fallback SQL when LLM synthesis fails
        
        Args:
            plan: The execution plan
            
        Returns:
            Fallback SQL query
        """
        logger.warning("Using fallback SQL synthesis")
        
        # If there's only one task, return its SQL
        if len(plan.tasks) == 1:
            return plan.tasks[0].sql_query
        
        # For multiple tasks, try to combine with CTEs
        cte_parts = []
        for idx, task in enumerate(plan.tasks):
            if idx == 0:
                cte_parts.append(f"WITH {task.task_id} AS (")
            else:
                cte_parts.append(f"),\n{task.task_id} AS (")
            cte_parts.append(f"  {task.sql_query}")
        
        cte_parts.append(")")
        
        # Create a simple SELECT from the last CTE
        last_task = plan.tasks[-1].task_id
        cte_parts.append(f"SELECT * FROM {last_task}")
        
        return "\n".join(cte_parts)