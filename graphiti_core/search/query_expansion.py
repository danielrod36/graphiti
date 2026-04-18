"""
Query expansion for complex multi-keyword searches.

Splits complex queries into sub-queries when the vector search
might miss relevant content. Each sub-query is searched independently,
candidates are merged, and the original query is used for final reranking.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Threshold: queries longer than this trigger expansion
EXPANSION_WORD_THRESHOLD = 3


# Common conjunctions/prepositions to split on
SPLIT_PATTERNS = [
    r'\s+and\s+',
    r'\s+or\s+',
    r'\s+with\s+',
    r'\s+for\s+',
    r'\s+in\s+',
    r'\s+using\s+',
    r'\s+on\s+',
    r'\s+from\s+',
]


def should_expand_query(query: str) -> bool:
    """Check if a query is complex enough to benefit from expansion."""
    words = query.strip().split()
    if len(words) <= EXPANSION_WORD_THRESHOLD:
        return False
    # Check if the query contains split patterns
    for pattern in SPLIT_PATTERNS:
        if re.search(pattern, query.lower()):
            return True
    # Also expand 4+ word queries via bigram decomposition
    return len(words) >= 4


def expand_query(query: str) -> list[str]:
    """
    Expand a complex query into sub-queries.

    Example:
        "Python error exception debugging" →
        ["Python error", "exception debugging", "Python debugging"]

        "server deployment VPS Docker" →
        ["server deployment", "VPS Docker", "server VPS"]

    Returns the original query plus sub-queries (original first).
    """
    query = query.strip()
    if not should_expand_query(query):
        return [query]

    sub_queries = [query]  # Always include original

    # Split on conjunctions/prepositions
    parts = [query]
    for pattern in SPLIT_PATTERNS:
        new_parts = []
        for part in parts:
            new_parts.extend(re.split(pattern, part, flags=re.IGNORECASE))
        parts = new_parts

    # Clean and filter
    parts = [p.strip() for p in parts if p.strip() and len(p.strip().split()) >= 2]
    sub_queries.extend(parts)

    # Also create bigram sub-queries (skip function words)
    words = query.split()
    stopwords = {'and', 'or', 'with', 'for', 'in', 'using', 'on', 'from', 'the', 'a', 'an', 'to', 'of'}
    if len(words) >= 4:
        for i in range(len(words) - 1):
            # Skip bigrams that are entirely stopwords
            if words[i].lower() in stopwords or words[i+1].lower() in stopwords:
                continue
            bigram = f"{words[i]} {words[i+1]}"
            if bigram.lower() not in [sq.lower() for sq in sub_queries]:
                sub_queries.append(bigram)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for sq in sub_queries:
        key = sq.lower()
        if key not in seen:
            seen.add(key)
            unique.append(sq)

    logger.info(f"Query expansion: '{query}' → {len(unique)} sub-queries: {unique}")
    return unique


def merge_search_results(
    result_sets: list[list[dict]],
    key_field: str = 'uuid',
    score_field: str = 'reranker_score',
) -> list[dict]:
    """
    Merge multiple search result lists, deduplicating by key_field.

    Keeps the highest score for each unique item.
    """
    seen: dict[str, dict] = {}
    for results in result_sets:
        for item in results:
            item_key = item.get(key_field, '')
            if not item_key:
                continue
            if item_key not in seen:
                seen[item_key] = item
            else:
                existing_score = seen[item_key].get(score_field, 0)
                new_score = item.get(score_field, 0)
                if new_score > existing_score:
                    seen[item_key] = item

    merged = list(seen.values())
    merged.sort(key=lambda x: x.get(score_field, 0), reverse=True)
    return merged
