"""
SQL Formatter Utility
Formats SQL queries for clean display and Oracle execution
"""

import re
from typing import Optional
from colorama import Fore, Style


class SQLFormatter:
    """Formats SQL queries for display and execution"""
    
    @staticmethod
    def format_for_display(sql: str, colorize: bool = True) -> str:
        """
        Format SQL for terminal display with proper indentation and colors
        
        Args:
            sql: Raw SQL string with escape characters
            colorize: Whether to add color formatting
            
        Returns:
            Formatted SQL string for display
        """
        # Clean up the SQL
        formatted = SQLFormatter.clean_sql(sql)
        
        # Add indentation for readability
        formatted = SQLFormatter._add_indentation(formatted)
        
        # Add colors if requested
        if colorize:
            formatted = SQLFormatter._colorize_sql(formatted)
        
        return formatted
    
    @staticmethod
    def format_for_execution(sql: str) -> str:
        """
        Format SQL for Oracle execution
        Removes formatting artifacts and ensures Oracle compatibility
        
        Args:
            sql: Raw SQL string
            
        Returns:
            Clean SQL ready for Oracle execution
        """
        # Clean basic formatting
        formatted = SQLFormatter.clean_sql(sql)
        
        # Remove any markdown artifacts
        formatted = re.sub(r'```sql\s*', '', formatted)
        formatted = re.sub(r'```\s*', '', formatted)
        
        # Ensure single spaces between words
        formatted = re.sub(r'\s+', ' ', formatted)
        
        # Ensure semicolon at the end if not present
        formatted = formatted.strip()
        
        return formatted
    
    @staticmethod
    def clean_sql(sql: str) -> str:
        """
        Clean SQL string by handling escape characters
        
        Args:
            sql: SQL with potential escape characters
            
        Returns:
            Clean SQL string
        """
        # Replace literal \n with actual newlines
        sql = sql.replace('\\n', '\n')
        
        # Replace literal \t with actual tabs
        sql = sql.replace('\\t', '\t')
        
        # Remove excessive whitespace
        lines = sql.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip trailing whitespace
            line = line.rstrip()
            # Skip empty lines
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _add_indentation(sql: str) -> str:
        """
        Add proper indentation to SQL for readability
        
        Args:
            sql: Clean SQL string
            
        Returns:
            Indented SQL string
        """
        # Keywords that should start a new line
        keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 
            'INNER JOIN', 'OUTER JOIN', 'GROUP BY', 'ORDER BY', 'HAVING',
            'UNION', 'WITH', 'AS', 'INSERT', 'UPDATE', 'DELETE', 'VALUES',
            'SET', 'AND', 'OR', 'LIMIT', 'OFFSET'
        ]
        
        # Split by keywords
        formatted = sql
        for keyword in keywords:
            # Add newline before keyword (case-insensitive)
            pattern = rf'(?<!^)\s+(?={keyword}\s)'
            formatted = re.sub(pattern, '\n', formatted, flags=re.IGNORECASE)
        
        # Add indentation for subqueries and CTEs
        lines = formatted.split('\n')
        indented_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Decrease indent for closing parenthesis
            if stripped.startswith(')'):
                indent_level = max(0, indent_level - 1)
            
            # Add current line with indentation
            if indent_level > 0:
                indented_lines.append('    ' * indent_level + stripped)
            else:
                indented_lines.append(stripped)
            
            # Increase indent after opening parenthesis
            if '(' in stripped and not ')' in stripped:
                indent_level += 1
            elif ')' in stripped and not '(' in stripped:
                pass  # Already handled above
            
        return '\n'.join(indented_lines)
    
    @staticmethod
    def _colorize_sql(sql: str) -> str:
        """
        Add color formatting to SQL keywords
        
        Args:
            sql: Formatted SQL string
            
        Returns:
            Colorized SQL string
        """
        # SQL keywords to highlight
        keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
            'ON', 'AS', 'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET',
            'WITH', 'UNION', 'ALL', 'DISTINCT', 'CASE', 'WHEN', 'THEN', 'ELSE',
            'END', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE',
            'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'CREATE',
            'TABLE', 'DROP', 'ALTER', 'ADD', 'COLUMN', 'PRIMARY', 'KEY',
            'FOREIGN', 'REFERENCES', 'INDEX', 'UNIQUE', 'DEFAULT', 'NULL',
            'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'ROUND', 'LAG', 'LEAD',
            'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'OVER', 'PARTITION'
        ]
        
        # Functions to highlight
        functions = [
            'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'ROUND', 'UPPER', 'LOWER',
            'SUBSTR', 'LENGTH', 'TRIM', 'COALESCE', 'NVL', 'DECODE',
            'TO_DATE', 'TO_CHAR', 'TO_NUMBER', 'SYSDATE', 'CURRENT_DATE'
        ]
        
        colored = sql
        
        # Color keywords in cyan
        for keyword in keywords:
            pattern = rf'\b{keyword}\b'
            colored = re.sub(
                pattern, 
                f'{Fore.CYAN}{keyword}{Style.RESET_ALL}',
                colored,
                flags=re.IGNORECASE
            )
        
        # Color functions in yellow
        for func in functions:
            pattern = rf'\b{func}\b(?=\s*\()'
            colored = re.sub(
                pattern,
                f'{Fore.YELLOW}{func}{Style.RESET_ALL}',
                colored,
                flags=re.IGNORECASE
            )
        
        # Color strings in green
        colored = re.sub(
            r"'([^']*)'",
            f"{Fore.GREEN}'\\1'{Style.RESET_ALL}",
            colored
        )
        
        # Color numbers in magenta
        colored = re.sub(
            r'\b\d+\.?\d*\b',
            f'{Fore.MAGENTA}\\g<0>{Style.RESET_ALL}',
            colored
        )
        
        return colored
    
    @staticmethod
    def validate_oracle_sql(sql: str) -> tuple[bool, Optional[str]]:
        """
        Basic validation for Oracle SQL compatibility
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Remove comments and string literals for validation
        clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.DOTALL)
        clean = re.sub(r"'[^']*'", "''", clean)
        
        # Check for common issues
        issues = []
        
        # Check for LIMIT (not Oracle syntax)
        if re.search(r'\bLIMIT\b', clean, re.IGNORECASE):
            issues.append("LIMIT is not Oracle syntax. Use ROWNUM or FETCH FIRST")
        
        # Check for backticks (MySQL syntax)
        if '`' in clean:
            issues.append("Backticks (`) are not Oracle syntax. Use double quotes for identifiers if needed")
        
        # Check for AUTO_INCREMENT (MySQL syntax)
        if re.search(r'\bAUTO_INCREMENT\b', clean, re.IGNORECASE):
            issues.append("AUTO_INCREMENT is not Oracle syntax. Use SEQUENCE or IDENTITY columns")
        
        # Check balanced parentheses
        open_count = clean.count('(')
        close_count = clean.count(')')
        if open_count != close_count:
            issues.append(f"Unbalanced parentheses: {open_count} open, {close_count} close")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, None
    
    @staticmethod
    def extract_table_names(sql: str) -> list[str]:
        """
        Extract table names from SQL query
        
        Args:
            sql: SQL query
            
        Returns:
            List of table names
        """
        # Clean the SQL first
        clean = SQLFormatter.clean_sql(sql)
        
        # Remove string literals
        clean = re.sub(r"'[^']*'", "", clean)
        
        tables = set()
        
        # Find tables after FROM
        from_pattern = r'FROM\s+([A-Z_][A-Z0-9_]*)'
        tables.update(re.findall(from_pattern, clean, re.IGNORECASE))
        
        # Find tables after JOIN
        join_pattern = r'JOIN\s+([A-Z_][A-Z0-9_]*)'
        tables.update(re.findall(join_pattern, clean, re.IGNORECASE))
        
        # Find tables after INTO (for INSERT)
        into_pattern = r'INTO\s+([A-Z_][A-Z0-9_]*)'
        tables.update(re.findall(into_pattern, clean, re.IGNORECASE))
        
        # Find tables after UPDATE
        update_pattern = r'UPDATE\s+([A-Z_][A-Z0-9_]*)'
        tables.update(re.findall(update_pattern, clean, re.IGNORECASE))
        
        return list(tables)


def display_sql(sql: str, title: str = "SQL Query"):
    """
    Display formatted SQL in terminal
    
    Args:
        sql: Raw SQL string
        title: Title to display above SQL
    """
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    formatted = SQLFormatter.format_for_display(sql, colorize=True)
    print(formatted)
    
    # Validate Oracle compatibility
    is_valid, error = SQLFormatter.validate_oracle_sql(sql)
    if not is_valid:
        print(f"\n{Fore.YELLOW}⚠ Oracle Compatibility Warning: {error}{Style.RESET_ALL}")
    
    # Show tables used
    tables = SQLFormatter.extract_table_names(sql)
    if tables:
        print(f"\n{Fore.BLUE}Tables: {', '.join(tables)}{Style.RESET_ALL}")

# # Example usage
# if __name__ == "__main__":
#     # Test SQL with escape characters
#     test_sql = "SELECT b.BANK_NAME, fp.NPL_RATIO\\nFROM FINANCIAL_PERFORMANCE fp\\nJOIN BANKS b ON fp.BANK_ID = b.BANK_ID\\nWHERE fp.NPL_RATIO > 4\\nORDER BY fp.NPL_RATIO DESC"
    
#     print("Original SQL:")
#     print(test_sql)
    
#     print("\n" + "="*60)
#     print("Formatted for Display:")
#     display_sql(test_sql, "Test Query")
    
#     print("\n" + "="*60)
#     print("Formatted for Execution:")
#     exec_sql = SQLFormatter.format_for_execution(test_sql)
#     print(exec_sql)