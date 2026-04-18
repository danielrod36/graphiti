---
title: "MemVault query expansion for complex multi-keyword searches"
category: "best-practices"
track: "knowledge"
problem_type: "search-quality"
module: "graphiti_core/search"
tags: [search, query-expansion, bigrams, vector-search, zero-results]
date: 2026-04-18
difficulty: "moderate"
---

# Query Expansion for Complex Searches

## Context
Multi-keyword queries like "Python error exception debugging" returned 0 results because:
1. Vector search embeds the full query as one vector — doesn't match any single entity
2. BM25 fulltext search fails on non-exact multi-term phrases
3. No candidates → no reranking → 0 results

## Guidance

### Solution: Bigram Decomposition
Split complex queries (4+ words) into overlapping bigram sub-queries:
```
"Python error exception debugging" →
  ["Python error exception debugging", "Python error", "error exception", "exception debugging"]
```

Each sub-query runs independently, results are deduplicated by UUID, then reranked against the original query using the cross-encoder.

### Implementation
- `graphiti_core/search/query_expansion.py` — utility module
- `should_expand_query()` — only expands 4+ word queries
- `expand_query()` — creates bigrams, skips stopwords
- `Graphiti.search_with_expansion()` — drop-in replacement for `Graphiti.search()`
- Server endpoint: `POST /search-expand`

### When to Apply
- Queries with 4+ content words
- Technical/domain terms that might not co-occur in any single entity
- Don't expand short queries (1-3 words) — they work fine with vector search

### Stopword Filtering
Bigrams containing stopwords (and, or, with, for, etc.) are skipped:
```
"Django and React frontend" → ["Django and React frontend", "React frontend"]
(not "Django and" or "and React")
```

## Pitfalls
- Don't expand too aggressively — too many sub-queries increases latency
- Always include the original query as the first sub-query
- Rerank merged results against the original query, not sub-queries
- Deduplicate by UUID, not fact text (same fact may appear with different UUIDs)

## Related
- `docs/solutions/reranker-fix-bge-local-2026-04-18.md` — Reranker threshold fix
