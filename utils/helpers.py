# utils/helpers.py
"""
Helper utilities for AStarBot
Common functions used across the application
"""

import re
import json
from datetime import datetime
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Clean and normalize text input
    Args: text (str): Raw text input
    Returns: str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with embedding
    text = re.sub(r'[^\w\s\-\.\,\!\?\'\"]', '', text)
    
    return text

def format_context_for_display(contexts: List[str], max_length: int = 100) -> List[str]:
    """
    Format context chunks for debugging/display
    Args: 
        contexts (List[str]): List of context strings
        max_length (int): Maximum length for display
    Returns: List[str]: Formatted context strings
    """
    formatted = []
    for i, ctx in enumerate(contexts, 1):
        if len(ctx) > max_length:
            formatted.append(f"{i}. {ctx[:max_length]}...")
        else:
            formatted.append(f"{i}. {ctx}")
    return formatted

def validate_env_vars(required_vars: List[str]) -> Dict[str, bool]:
    """
    Validate that required environment variables are set
    Args: required_vars (List[str]): List of required variable names
    Returns: Dict[str, bool]: Validation results
    """
    import os
    results = {}
    for var in required_vars:
        results[var] = bool(os.getenv(var))
    return results

def log_chat_interaction(question: str, answer: str, contexts: List[str] = None):
    """
    Log chat interactions for debugging (optional)
    Args:
        question (str): User's question
        answer (str): Bot's response
        contexts (List[str]): Retrieved contexts
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "context_count": len(contexts) if contexts else 0
    }
    
    # You can extend this to write to file if needed
    print(f"💭 Chat logged: {timestamp}")

def truncate_response(response: str, max_words: int = 50) -> str:
    """
    Ensure responses stay within word limit
    Args:
        response (str): Generated response
        max_words (int): Maximum word count
    Returns: str: Truncated response if needed
    """
    words = response.split()
    if len(words) <= max_words:
        return response
    
    truncated = " ".join(words[:max_words])
    return truncated + "..."

def format_tags(tags: List[str]) -> str:
    """
    Format tags for display
    Args: tags (List[str]): List of tags
    Returns: str: Formatted tag string
    """
    if not tags:
        return "No tags"
    return " • ".join([f"#{tag}" for tag in tags])
