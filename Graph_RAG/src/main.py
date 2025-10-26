"""
Main integration point for RAG System and LLM Query Generator
"""

import os
from typing import Dict, Any, Optional
from rag_agent.rag_system import RAGSystem
from llm_agent.llm_query_generator import LLMQueryGenerator


class GraphRAGPipeline:
    """Main pipeline integrating RAG and LLM for SQL generation"""
    
    def __init__(self,
                 llm_provider: str = "openai",
                 llm_api_key: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 use_sample_values: bool = True):
        
        print("Initializing Graph RAG Pipeline...")
        
        # Initialize RAG System
        self.rag_system = RAGSystem()
        
        # Initialize LLM Query Generator
        self.llm_generator = LLMQueryGenerator(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model
        )
        
        print("Pipeline initialized successfully!")
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main method to process user query
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Dictionary containing SQL, tables, source, and metadata
        """
        print(f"\nProcessing: {user_query}")
        
        # Step 1: Check query history for similar queries
        cached_result = self.rag_system.find_similar_query(user_query)
        
        if cached_result:
            query, similarity = cached_result
            print(f"Found cached query with similarity: {similarity:.2f}")
            return {
                'sql': query.sql_query,
                'tables': query.tables_used,
                'source': 'cache',
                'similarity_score': similarity,
                'query_id': query.id
            }
        
        # Step 2: Find relevant tables using RAG
        search_results = self.rag_system.search_relevant_tables(user_query)
        tables = search_results['tables']
        
        print(f"Identified tables: {tables}")
        
        # Step 3: Build context for LLM
        context = self.rag_system.build_context(tables, user_query)
        
        # Step 4: Generate SQL using LLM
        sql = self.llm_generator.generate_sql(context)
        
        # Step 5: Save to history for future use
        self.rag_system.add_query_to_history(user_query, sql, tables)
        
        return {
            'sql': sql,
            'tables': tables,
            'source': 'generated',
            'llm_provider': self.llm_generator.provider,
            'sample_values': search_results.get('sample_values', {})
        }
    
    def get_schema_info(self, table_name: Optional[str] = None) -> str:
        """Get schema information for tables"""
        if table_name:
            if table_name in self.rag_system.tables_dict:
                table = self.rag_system.tables_dict[table_name]
                return self._format_table_info(table)
            else:
                return f"Table {table_name} not found"
        else:
            info = "Database Schema:\n\n"
            for table in self.rag_system.tables:
                info += self._format_table_info(table) + "\n" + "="*50 + "\n"
            return info
    
    def _format_table_info(self, table) -> str:
        """Format table information"""
        info = f"Table: {table.table_name}\n"
        info += f"Description: {table.table_description}\n\n"
        info += "Columns:\n"
        
        for col in table.columns:
            info += f"  - {col.column_name} ({col.data_type})"
            if col.is_primary_key:
                info += " [PK]"
            info += f"\n    {col.description}"
            if col.sample_values:
                info += f"\n    Examples: {col.sample_values[:3]}"
            info += "\n"
        
        return info


# ============== Standalone Usage ==============

def main():
    """Example usage of the Graph RAG Pipeline"""
    
    # Initialize pipeline
    pipeline = GraphRAGPipeline(
        llm_provider="openai",  # or "heuristic" for no LLM
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        use_sample_values=True
    )
    
    # Test queries
    test_queries = [
        "Show all banks",
        "Which banks have NPL ratio above 4%?",
        "Show financial performance of Mashreq bank",
        "Get market capitalization for all banks",
        "Find banks with CET1 ratio above 14%"
    ]
    
    print("\n" + "="*60)
    print("TESTING QUERIES")
    print("="*60)
    
    for query in test_queries:
        result = pipeline.process_query(query)
        
        print(f"\nQuery: {query}")
        print(f"Source: {result['source']}")
        print(f"Tables: {', '.join(result['tables'])}")
        print(f"SQL: {result['sql'][:100]}...")
        
        if result['source'] == 'cache':
            print(f"Similarity: {result['similarity_score']:.2f}")
        print("-"*40)


if __name__ == "__main__":
    main()