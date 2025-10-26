"""
LLM-driven Task Executor
Executes tasks and lets LLM decide how to handle results
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .llm_interface import AgenticLLMInterface
from .llm_task_decomposer import AgenticPlan, AgenticTask

logger = logging.getLogger(__name__)


class LLMTaskExecutor:
    """Executes tasks and uses LLM for decision making"""
    
    def __init__(self, llm_interface: AgenticLLMInterface, query_executor=None):
        self.llm = llm_interface
        self.query_executor = query_executor
        self.execution_history = []
        
    async def execute_plan(self, plan: AgenticPlan) -> Dict[str, Any]:
        """
        Execute the plan created by LLM
        """
        logger.info(f"Executing plan {plan.plan_id} with {len(plan.tasks)} tasks")
        
        results = {}
        
        try:
            # Execute tasks according to execution order
            for batch_idx, batch in enumerate(plan.execution_order):
                logger.info(f"Executing batch {batch_idx + 1}: {batch}")
                
                # Check if tasks can run in parallel
                parallel_tasks = [
                    task for task in plan.tasks 
                    if task.task_id in batch and task.execution_type == "parallel"
                ]
                sequential_tasks = [
                    task for task in plan.tasks 
                    if task.task_id in batch and task.execution_type == "sequential"
                ]
                
                # Execute parallel tasks
                if parallel_tasks:
                    batch_results = await self._execute_parallel_tasks(parallel_tasks, results)
                    results.update(batch_results)
                
                # Execute sequential tasks
                for task in sequential_tasks:
                    result = await self._execute_single_task(task, results)
                    results[task.task_id] = result
                    task.result = result
                    task.status = "completed"
            
            # All tasks completed - synthesize results
            final_result = await self._synthesize_results(plan, results)
            
            return {
                'success': True,
                'plan_id': plan.plan_id,
                'tasks_executed': len(plan.tasks),
                'results': results,
                'final_result': final_result,
                'execution_time': (datetime.now() - plan.created_at).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            return {
                'success': False,
                'plan_id': plan.plan_id,
                'error': str(e),
                'partial_results': results
            }
    
    async def _execute_parallel_tasks(self, tasks: List[AgenticTask], 
                                     previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple tasks in parallel"""
        
        coroutines = [
            self._execute_single_task(task, previous_results)
            for task in tasks
        ]
        
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        task_results = {}
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task_results[task.task_id] = {'error': str(result)}
                logger.error(f"Task {task.task_id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                task_results[task.task_id] = result
        
        return task_results
    
    async def _execute_single_task(self, task: AgenticTask, 
                                   previous_results: Dict[str, Any]) -> Any:
        """Execute a single task"""
        
        logger.info(f"Executing task {task.task_id}: {task.description}")
        task.status = "running"
        
        try:
            # Check if SQL needs modification based on previous results
            sql_to_execute = await self._prepare_sql_with_context(
                task, previous_results
            )
            
            # Execute the SQL
            if self.query_executor:
                result = await self.query_executor.execute_query(sql_to_execute)
                
                # Store execution details
                return {
                    'task_id': task.task_id,
                    'description': task.description,
                    'sql_executed': sql_to_execute,
                    'row_count': result.get('row_count', 0),
                    'columns': result.get('columns', []),
                    'data': result.get('rows', []),
                    'execution_time': datetime.now().isoformat()
                }
            else:
                # Mock execution for testing
                return {
                    'task_id': task.task_id,
                    'description': task.description,
                    'sql_executed': sql_to_execute,
                    'data': [{'mock': 'data'}]
                }
                
        except Exception as e:
            logger.error(f"Task {task.task_id} execution failed: {e}")
            raise
    
    async def _prepare_sql_with_context(self, task: AgenticTask, 
                                        previous_results: Dict[str, Any]) -> str:
        """
        Use LLM to modify SQL based on previous results if needed
        """
        
        # If task has no dependencies, use SQL as is
        if not task.dependencies:
            return task.sql_query
        
        # Build context from dependent results
        dependency_context = self._build_dependency_context(task.dependencies, previous_results)
        
        # Ask LLM to modify SQL if needed
        try:
            modified_sql = await self.llm.modify_sql_with_context(
                task.description,
                task.sql_query,
                dependency_context
            )
            return modified_sql
        except Exception as e:
            logger.error(f"Failed to modify SQL with context: {e}")
            return task.sql_query
    
    def _build_dependency_context(self, dependencies: List[str], 
                                  results: Dict[str, Any]) -> str:
        """Build context from dependency results"""
        
        context_parts = []
        
        for dep_id in dependencies:
            if dep_id in results:
                result = results[dep_id]
                context_parts.append(f"\nTask {dep_id} Results:")
                
                if isinstance(result, dict):
                    if 'data' in result and result['data']:
                        # Show sample of data
                        context_parts.append(f"  Row count: {len(result['data'])}")
                        context_parts.append(f"  Sample data: {result['data'][:2]}")
                    if 'columns' in result:
                        context_parts.append(f"  Columns: {result['columns']}")
                else:
                    context_parts.append(f"  Result: {result}")
        
        return "\n".join(context_parts)
    
    async def _synthesize_results(self, plan: AgenticPlan, 
                                  results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to synthesize all results into final answer
        """
        
        logger.info("Synthesizing results using LLM")
        
        # Format results for LLM
        results_summary = self._format_results_for_synthesis(plan, results)
        
        try:
            synthesis_result = await self.llm.synthesize_results(
                plan.original_query,
                plan.synthesis_instructions,
                results_summary
            )
            
            if not synthesis_result:
                logger.warning("LLM synthesis returned empty result")
                return self._create_fallback_synthesis(results_summary)
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return self._create_fallback_synthesis(results_summary)
    
    def _create_fallback_synthesis(self, results: str) -> Dict[str, Any]:
        """Create fallback synthesis when LLM fails"""
        return {
            'answer': 'Query executed successfully. Results compiled.',
            'key_insights': ['Data retrieved and processed'],
            'recommendations': [],
            'summary_metrics': {},
            'final_sql': None
        }
    
    def _format_results_for_synthesis(self, plan: AgenticPlan, 
                                      results: Dict[str, Any]) -> str:
        """Format all results for LLM synthesis"""
        
        formatted = []
        
        for task in plan.tasks:
            task_result = results.get(task.task_id, {})
            
            formatted.append(f"\nTask {task.task_id}: {task.description}")
            formatted.append(f"SQL: {task.sql_query[:100]}...")
            
            if isinstance(task_result, dict):
                if 'data' in task_result:
                    data = task_result['data']
                    formatted.append(f"Results: {len(data)} rows")
                    
                    # Show sample data
                    if data:
                        if len(data) <= 5:
                            formatted.append(f"Data: {data}")
                        else:
                            formatted.append(f"Sample (first 3): {data[:3]}")
                            formatted.append(f"Sample (last 2): {data[-2:]}")
                
                if 'row_count' in task_result:
                    formatted.append(f"Row count: {task_result['row_count']}")
            else:
                formatted.append(f"Result: {task_result}")
            
            formatted.append("-" * 40)
        
        return "\n".join(formatted)