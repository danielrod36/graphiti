---
title: "Graphiti fork: BGE reranker hardening — configurable model, error handling, threshold defaults"
category: "best-practices"
track: "knowledge"
problem_type: "configuration"
module: "graphiti_core/cross_encoder"
tags: [reranker, bge, cross-encoder, config, threshold, error-handling]
date: 2026-04-18
difficulty: "moderate"
---

# Reranker Hardening in Graphiti Fork

## Context
The Graphiti fork's BGE reranker had three issues: hardcoded model name, no error handling on model load, and cross-encoder search recipes missing threshold filtering by default.

## Guidance

### 1. Make model name configurable
```python
# Before
class BGERerankerClient(CrossEncoderClient):
    def __init__(self):
        self.model = CrossEncoder('BAAI/bge-reranker-v2-m3')

# After
class BGERerankerClient(CrossEncoderClient):
    def __init__(self, model: str | None = None):
        model_name = model or DEFAULT_BGE_MODEL
        self.model = CrossEncoder(model_name)
```
This allows swapping models (e.g., MiniLM-L6 for Pi, BGE-v2-m3 for servers) without code changes.

### 2. Wrap model loading in error handling
Model downloads can fail (network, disk full, HuggingFace outage). Without try/except, the entire server crashes at startup with no useful error message.

### 3. Set reranker_min_score in cross-encoder recipes
The default `reranker_min_score=0` means no filtering. Cross-encoder recipes should always filter by relevance. Default: 0.3 for sigmoid-normalized scores.

### 4. Export all reranker clients in __init__.py
If a reranker exists in the codebase, it should be importable via the package API. Missing exports force users to use fragile deep imports.

## When to Apply
- Adding any new reranker client to the cross_encoder package
- Creating search config recipes that use rerankers
- Any model-loading code that runs at server startup

## Examples
All 4 cross-encoder recipes now include `reranker_min_score=0.3`:
- `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`
- `EDGE_HYBRID_SEARCH_CROSS_ENCODER`
- `NODE_HYBRID_SEARCH_CROSS_ENCODER`
- `COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER`

## Related
- `docs/solutions/reranker-fix-bge-local-2026-04-18.md` — Original reranker investigation
