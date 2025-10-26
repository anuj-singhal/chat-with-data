"""
Chat-based SQL Agent with Validation
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
    """Interactive chat agent for SQL generation"""
    
    def __init__(self):
        """Initialize the chat agent"""
        print(f"{Fore.CYAN}Initializing SQL Chat Agent...{Style.RESET_ALL}")
        
        # Initialize RAG
        self.rag_system = RAGSystem(
            data_dict_path="./data/data_dictionary.json",
            relationships_path="./data/relationships.json",
            query_history_path="./data/query_history.json",
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
    
    async def run_examples(self):
        """Run comprehensive examples"""
        
        examples = {
            'simple': [
                # Same query, different phrasing
                "Show all banks with NPL ratio above 4%",
                "List banks where NPL ratio exceeds 4 percent",
                "Which banks have non-performing loans ratio greater than 4?"
            ],
            'medium_complex': [
                # Same query, different phrasing
                "Compare deposit growth between Mashreq and ADCB for last year",
                "Show me how deposits changed for Mashreq versus ADCB in the previous year",
                "I want to see deposit trends comparing Mashreq bank with ADCB bank over the past year"
            ],
            'complex': [
                # Same query, different phrasing
                "Analyze bank risk by combining NPL ratios with capital adequacy and rank them",
                "Create a risk assessment using non-performing loans and CET1 ratios to rank banks",
                "I need a comprehensive risk ranking based on asset quality (NPL) and capital strength (CET1)"
            ]
        }
        
        print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}RUNNING COMPREHENSIVE EXAMPLES{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        
        for complexity, queries in examples.items():
            print(f"\n{Fore.CYAN}--- {complexity.upper().replace('_', ' ')} QUERIES ---{Style.RESET_ALL}")
            print(f"Testing {len(queries)} variations of the same query:\n")
            
            sql_results = []
            for i, query in enumerate(queries, 1):
                print(f"{Fore.YELLOW}Variation {i}:{Style.RESET_ALL} {query}")
                result = await self.process_user_query(query)
                
                if result['success']:
                    sql_results.append(result['sql'])
                
                await asyncio.sleep(1)
            
            # Check if all SQLs are similar
            if len(sql_results) == len(queries):
                # Simple similarity check (you could make this more sophisticated)
                base_sql = sql_results[0].lower().strip()
                all_similar = all(
                    self._sql_similarity(base_sql, sql.lower().strip()) > 0.7
                    for sql in sql_results[1:]
                )
                
                if all_similar:
                    print(f"{Fore.GREEN}✓ All variations generated similar SQL{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}⚠ Some variations generated different SQL{Style.RESET_ALL}")
    
    def _sql_similarity(self, sql1: str, sql2: str) -> float:
        """Simple SQL similarity check (0-1)"""
        # This is a basic implementation - could use more sophisticated comparison
        tokens1 = set(sql1.split())
        tokens2 = set(sql2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    async def interactive_chat(self):
        """Run interactive chat mode"""
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}SQL CHAT AGENT - Interactive Mode{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"Type your queries in natural language.")
        print(f"Commands: 'examples' - run examples, 'history' - show history, 'exit' - quit\n")
        
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
                
                elif user_input.lower() == 'examples':
                    await self.run_examples()
                
                elif user_input.lower() == 'history':
                    self.show_history()
                
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
    
    def show_history(self):
        """Show session history"""
        
        if not self.session_history:
            print(f"{Fore.YELLOW}No queries in history{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}Session History:{Style.RESET_ALL}")
        for i, entry in enumerate(self.session_history[-5:], 1):  # Last 5 queries
            print(f"\n{i}. Query: {entry['query'][:50]}...")
            if entry['result']['success']:
                print(f"   SQL: {entry['result']['sql'][:100]}...")
                if entry['result'].get('validation_confidence'):
                    print(f"   Confidence: {entry['result']['validation_confidence']}%")


async def main():
    """Main entry point"""
    
    # Create agent
    agent = SQLChatAgent()
    
    # Display menu
    print(f"\n{Fore.MAGENTA}Select Mode:{Style.RESET_ALL}")
    print("1. Interactive Chat")
    print("2. Run Examples")
    print("3. Both")
    
    choice = input(f"{Fore.GREEN}Choice (1-3): {Style.RESET_ALL}").strip()
    
    if choice == '1':
        await agent.interactive_chat()
    elif choice == '2':
        await agent.run_examples()
    elif choice == '3':
        await agent.run_examples()
        await agent.interactive_chat()
    else:
        print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")


if __name__ == "__main__":
    asyncio.run(main())