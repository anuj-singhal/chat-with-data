"""
Main script to run LLM-driven Text-to-SQL System with Validation
Generates SQL queries without execution
Includes pre and post validation using Groq LLM
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Import RAG components
from rag_agent.rag_system import RAGSystem

# Import LLM Agentic components
from llm_agentic.llm_agentic_orchestrator import LLMAgenticOrchestrator
from llm_agentic.sample_queries import get_test_queries


async def handle_clarification_request(orchestrator: LLMAgenticOrchestrator, 
                                       result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle clarification requests from pre-validation
    """
    
    print("\n⚠️ Additional information needed:")
    print("-" * 40)
    
    for i, question in enumerate(result['clarification_questions'], 1):
        print(f"{i}. {question}")
    
    print("\n📝 Issues identified:")
    for issue in result.get('issues', []):
        print(f"  - {issue['type']}: {issue['description']}")
    
    # Simulate user providing clarifications
    print("\n💡 Simulating user responses...")
    clarifications = {
        "Time period": "Q4 2023",
        "Metric": "NPL ratio for asset quality",
        "Bank selection": "Top 5 by market cap"
    }
    
    print("User clarifications provided:")
    for key, value in clarifications.items():
        print(f"  {key}: {value}")
    
    # Process with clarifications
    query_id = result['query_id']
    return await orchestrator.process_with_clarifications(query_id, clarifications)


async def run_query_with_validation(orchestrator: LLMAgenticOrchestrator, 
                                    query: str, query_type: str = "SIMPLE"):
    """
    Run a query through the validation system
    """
    
    print(f"\n{'='*80}")
    print(f"{query_type} QUERY: {query}")
    print('='*80)
    
    result = await orchestrator.process_query(query)
    
    # Handle clarification requests
    if result.get('needs_clarification'):
        print("🔄 Pre-validation identified ambiguities")
        result = await handle_clarification_request(orchestrator, result)
    
    # Handle results
    if result.get('success'):
        print(f"✅ Success!")
        print(f"Query Type: {result.get('query_type')}")
        
        # Show validation confidence if available
        if result.get('validation_confidence') is not None:
            confidence = result['validation_confidence']
            if confidence >= 80:
                print(f"🟢 Validation Confidence: {confidence}%")
            elif confidence >= 60:
                print(f"🟡 Validation Confidence: {confidence}%")
            else:
                print(f"🔴 Validation Confidence: {confidence}%")
        
        # Show SQL
        if result.get('sql'):
            print(f"\n📝 Generated SQL:")
            print(f"{result['sql'][:500]}...")
            
        # For complex queries, show task breakdown
        if result.get('sql_queries') and len(result['sql_queries']) > 1:
            print(f"\n📋 Task Breakdown ({len(result['sql_queries'])} tasks):")
            for task in result['sql_queries']:
                print(f"  [{task['task_id']}] {task['description']}")
    
    elif result.get('needs_user_help'):
        print(f"⚠️ Validation failed - User assistance needed")
        print(f"Issues found:")
        for issue in result.get('validation_issues', []):
            print(f"  - {issue['severity']}: {issue['description']}")
        print(f"\nAttempted SQL:")
        print(f"{result.get('attempted_sql', 'N/A')[:300]}...")
    
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    return result


async def demonstrate_validation_scenarios():
    """
    Demonstrate different validation scenarios
    """
    
    print("\n" + "="*80)
    print("VALIDATION SCENARIOS DEMONSTRATION")
    print("="*80)
    
    # Scenario 1: Ambiguous query requiring clarification
    print("\n" + "="*80)
    print("SCENARIO 1: Ambiguous Query (Pre-validation)")
    print("="*80)
    
    ambiguous_query = "Show me the performance of top banks"
    # This should trigger pre-validation clarification request
    
    # Scenario 2: Query with incorrect SQL generation
    print("\n" + "="*80)
    print("SCENARIO 2: Post-validation Correction")
    print("="*80)
    
    problematic_query = "Calculate average NPL ratio for each bank"
    # This might generate SQL without GROUP BY initially
    
    # Scenario 3: Complex query validation
    print("\n" + "="*80)
    print("SCENARIO 3: Complex Query Synthesis Validation")
    print("="*80)
    
    complex_query = """
    Analyze bank risk profiles:
    1. Get NPL ratios above 4%
    2. Check their capital adequacy
    3. Compare with industry average
    4. Rank by risk level
    """
    # This should validate the synthesized SQL against individual tasks
    
    return [ambiguous_query, problematic_query, complex_query]


async def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("LLM-DRIVEN TEXT-TO-SQL SYSTEM WITH VALIDATION")
    print("SQL Generation with Pre/Post Validation using Groq")
    print("="*80)
    
    # Initialize components
    print("\n🔧 Initializing components...")
    
    # Initialize RAG system
    rag_system = RAGSystem(
        data_dict_path="./Graph_RAG/data/data_dictionary.json",
        relationships_path="./Graph_RAG/data/relationships.json", 
        query_history_path="./Graph_RAG/data/query_history.json",
        use_sample_values=True
    )
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if openai_key:
        print("✅ Using OpenAI for SQL generation")
        llm_provider = "openai"
    else:
        print("⚠️ Using heuristic fallback for SQL generation")
        llm_provider = "heuristic"
    
    if groq_key:
        print("✅ Using Groq (Llama 3.3 70B) for validation")
    else:
        print("⚠️ Validation disabled (set GROQ_API_KEY to enable)")
    
    # Initialize orchestrator with validation
    orchestrator = LLMAgenticOrchestrator(
        rag_system=rag_system,
        llm_provider=llm_provider,
        llm_api_key=openai_key,
        groq_api_key=groq_key
    )
    
    print("✅ System initialized!")
    
    # Get test queries
    test_queries = get_test_queries()
    
    # Test validation scenarios
    print("\n" + "="*80)
    print("TESTING VALIDATION SCENARIOS")
    print("="*80)
    
    # validation_queries = await demonstrate_validation_scenarios()
    
    # for i, query in enumerate(validation_queries, 1):
    #     print(f"\n--- Validation Scenario {i} ---")
    #     await run_query_with_validation(orchestrator, query, f"VALIDATION_{i}")
    #     await asyncio.sleep(1)
    
    # # Test regular queries
    # print("\n" + "="*80)
    # print("TESTING REGULAR QUERIES")
    # print("="*80)
    
    # Simple query
    simple_query = test_queries['simple'][0]
    await run_query_with_validation(orchestrator, simple_query, "SIMPLE")
    
    # Complex query
    complex_query = test_queries['complex'][0]
    await run_query_with_validation(orchestrator, complex_query, "COMPLEX")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE!")
    print("="*80)
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"• SQL Generation: {llm_provider.title()}")
    print(f"• Validation: {'Groq LLM' if groq_key else 'Disabled'}")
    print(f"• RAG System: Enabled")
    
    print("\n💡 Validation Features:")
    print("  1. Pre-validation: Identifies ambiguities and missing info")
    print("  2. Post-validation: Validates generated SQL structure")
    print("  3. Self-correction: Automatically fixes validation issues")
    print("  4. Complex validation: Validates synthesized SQL against tasks")
    print("  5. User fallback: Requests help when unable to generate valid SQL")
    
    print("\n📝 Validation Flow:")
    print("  Query → Pre-validate → Generate SQL → Post-validate →")
    print("  → Self-correct if needed → Final validation → Return SQL or request help")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())