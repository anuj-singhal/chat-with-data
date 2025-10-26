"""
Chat-based SQL Agent with Query History Matching
Interactive agent for text-to-SQL generation with comprehensive validation
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List
from colorama import Fore, Style, init

# Initialize colorama for colored output
init()

# Import components
from rag_agent.rag_system import RAGSystem
from llm_agentic.llm_agentic_orchestrator import LLMAgenticOrchestratorImproved


class SQLChatAgent:
    """Interactive chat agent for SQL generation with query history"""
    
    def __init__(self):
        """Initialize the chat agent"""
        print(f"{Fore.CYAN}Initializing SQL Chat Agent with Query History...{Style.RESET_ALL}")
        
        # Initialize RAG
        self.rag_system = RAGSystem(
            data_dict_path="./Graph_RAG/data/data_dictionary.json",
            relationships_path="./Graph_RAG/data/relationships.json", 
            query_history_path="./Graph_RAG/data/query_history.json",
            use_sample_values=True
        )
        
        # Check API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        # Initialize orchestrator
        self.orchestrator = LLMAgenticOrchestratorImproved(
            rag_system=self.rag_system,
            llm_provider="openai" if openai_key else "heuristic",
            llm_api_key=openai_key,
            groq_api_key=groq_key
        )
        
        self.session_history = []
        
        print(f"{Fore.GREEN}✓ Agent initialized{Style.RESET_ALL}")
        print(f"  • SQL Generation: {'OpenAI' if openai_key else 'Heuristic'}")
        print(f"  • Validation: {'Groq LLM' if groq_key else 'Disabled'}")
        print(f"  • Query History: {len(self.rag_system.query_history)} cached queries")
    
    def display_validation_scores(self, validation: Dict[str, Any]):
        """Display validation scores with colors"""
        
        print(f"\n{Fore.CYAN}📊 Validation Results:{Style.RESET_ALL}")
        
        for metric, score in validation.items():
            if isinstance(score, (int, float)):
                if score >= 80:
                    color = Fore.GREEN
                    symbol = "✓"
                elif score >= 60:
                    color = Fore.YELLOW
                    symbol = "⚠"
                else:
                    color = Fore.RED
                    symbol = "✗"
                
                print(f"  {color}{symbol} {metric.title()}: {score}%{Style.RESET_ALL}")
    
    def display_cache_info(self, result: Dict[str, Any]):
        """Display cache information if query was from history"""
        
        if result.get('from_cache'):
            print(f"\n{Fore.MAGENTA}🔄 Using Cached Query:{Style.RESET_ALL}")
            print(f"  • Query ID: {result['cache_query_id']}")
            print(f"  • Similarity: {result['similarity_score']:.2%}")
    
    def display_suggestions(self, suggestions: List[Dict[str, Any]]):
        """Display applied suggestions"""
        
        if suggestions:
            print(f"\n{Fore.YELLOW}💡 Applied Suggestions:{Style.RESET_ALL}")
            for sug in suggestions:
                print(f"  • {sug['aspect']}: {sug['suggestion']}")
    
    async def process_user_query(self, query: str) -> Dict[str, Any]:
        """Process user query and display results"""
        
        print(f"\n{Fore.BLUE}Processing your query...{Style.RESET_ALL}")
        
        # Process query
        result = await self.orchestrator.process_query(query, auto_approve_suggestions=True)
        
        if result['success']:
            # Display cache info if applicable
            self.display_cache_info(result)
            
            # Display suggestions if applied
            if result.get('suggestions_applied'):
                self.display_suggestions(result['suggestions_applied'])
            
            # Display SQL
            print(f"\n{Fore.GREEN}✓ Generated SQL:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{result['sql']}{Style.RESET_ALL}")
            
            # Display validation scores
            if result.get('validation'):
                self.display_validation_scores(result['validation'])
            
            # Display overall confidence
            if result.get('validation_confidence') is not None:
                conf = result['validation_confidence']
                if conf >= 80:
                    color = Fore.GREEN
                elif conf >= 60:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
                print(f"\n{color}Overall Confidence: {conf}%{Style.RESET_ALL}")
            
            # Display correction attempts
            if result.get('correction_attempts', 1) > 1:
                print(f"{Fore.YELLOW}ℹ Self-correction attempts: {result['correction_attempts']}{Style.RESET_ALL}")
            
        else:
            print(f"{Fore.RED}✗ Failed: {result.get('error', 'Unknown error')}{Style.RESET_ALL}")
        
        return result
    
    async def run_test_examples(self):
        """Run test examples to demonstrate query history matching"""
        
        test_queries = [
            # Test 1: Exact match with existing query
            {
                'type': 'Simple - Cache Hit',
                'queries': [
                    "Show banks with NPL above 4%",  # Should match cached query
                    "List banks where NPL exceeds 4 percent"  # Should also match
                ]
            },
            # Test 2: Similar but different query
            {
                'type': 'Simple - Cache Miss',
                'queries': [
                    "Show banks with NPL below 3%",  # Different threshold, should regenerate
                    "Display banks with CET1 above 15%"  # Different metric
                ]
            },
            # Test 3: Complex cached query
            {
                'type': 'Complex - Cache Hit',
                'queries': [
                    "Compare deposits for Mashreq and ADCB in 2023",  # Should match cached
                    "Show deposit growth between Mashreq and ADCB for 2023"  # Variation
                ]
            },
            # Test 4: New complex query
            {
                'type': 'Complex - New Query',
                'queries': [
                    "Analyze risk profile combining NPL and capital adequacy ratios with ranking"
                ]
            }
        ]
        
        print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}RUNNING TEST EXAMPLES WITH QUERY HISTORY{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        
        for test_set in test_queries:
            print(f"\n{Fore.CYAN}--- {test_set['type']} ---{Style.RESET_ALL}")
            
            for query in test_set['queries']:
                print(f"\n{Fore.YELLOW}Query:{Style.RESET_ALL} {query}")
                result = await self.process_user_query(query)
                
                # Show if it was from cache
                if result.get('from_cache'):
                    print(f"{Fore.GREEN}✓ Used cached SQL (saved processing time){Style.RESET_ALL}")
                else:
                    print(f"{Fore.BLUE}✓ Generated new SQL{Style.RESET_ALL}")
                
                await asyncio.sleep(1)
    
    async def interactive_chat(self):
        """Run interactive chat mode"""
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}SQL CHAT AGENT - Interactive Mode{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Query History: {len(self.rag_system.query_history)} cached queries{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"Type your queries in natural language.")
        print(f"Commands: 'test' - run test examples, 'history' - show cached queries, 'exit' - quit\n")
        
        while True:
            try:
                # Get user input
                user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'exit':
                    print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                    break
                
                elif user_input.lower() == 'test':
                    await self.run_test_examples()
                
                elif user_input.lower() == 'history':
                    self.show_cached_queries()
                
                else:
                    # Process as SQL query
                    result = await self.process_user_query(user_input)
                    self.session_history.append({
                        'query': user_input,
                        'result': result,
                        'timestamp': datetime.now()
                    })
            
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Interrupted. Type 'exit' to quit.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    def show_cached_queries(self):
        """Show cached queries from history"""
        
        if not self.rag_system.query_history:
            print(f"{Fore.YELLOW}No cached queries{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}Cached Queries:{Style.RESET_ALL}")
        for i, query in enumerate(self.rag_system.query_history[:5], 1):  # Show first 5
            print(f"\n{i}. {Fore.YELLOW}{query.natural_language}{Style.RESET_ALL}")
            print(f"   ID: {query.id}")
            
            # Show overall confidence
            overall_conf = getattr(query, 'overall_confidence', None)
            if overall_conf:
                if overall_conf >= 90:
                    color = Fore.GREEN
                elif overall_conf >= 80:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
                print(f"   {color}Overall Confidence: {overall_conf}%{Style.RESET_ALL}")
            
            # Show validation scores
            if query.validation_result:
                scores = query.validation_result
                avg_score = sum(scores.values()) / len(scores) if scores else 0
                print(f"   Validation: {avg_score:.0f}% (avg)")
            
            print(f"   Variations: {len(query.variations)}")
            print(f"   Last used: {query.last_used}")
    
    def show_history(self):
        """Show session history"""
        
        if not self.session_history:
            print(f"{Fore.YELLOW}No queries in session history{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}Session History:{Style.RESET_ALL}")
        for i, entry in enumerate(self.session_history[-5:], 1):  # Last 5 queries
            print(f"\n{i}. Query: {entry['query'][:50]}...")
            if entry['result']['success']:
                print(f"   SQL: {entry['result']['sql'][:100]}...")
                if entry['result'].get('from_cache'):
                    print(f"   Source: Cached (ID: {entry['result']['cache_query_id']})")
                else:
                    print(f"   Source: Generated")
                if entry['result'].get('validation_confidence'):
                    print(f"   Confidence: {entry['result']['validation_confidence']}%")


async def main():
    """Main entry point"""
    
    # Create agent
    agent = SQLChatAgent()
    
    # Display menu
    print(f"\n{Fore.MAGENTA}Select Mode:{Style.RESET_ALL}")
    print("1. Interactive Chat")
    print("2. Run Test Examples")
    print("3. Both (Test then Chat)")
    
    choice = input(f"{Fore.GREEN}Choice (1-3): {Style.RESET_ALL}").strip()
    
    if choice == '1':
        await agent.interactive_chat()
    elif choice == '2':
        await agent.run_test_examples()
    elif choice == '3':
        await agent.run_test_examples()
        await agent.interactive_chat()
    else:
        print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")


if __name__ == "__main__":
    asyncio.run(main())