# Reranker Fix Requirements

> **Created:** 2026-04-18
> **Scope:** Standard

## Problem (FINAL — after reading Pi server code)

**Root cause identified:** The ZAIReranker is properly wired but **never used**. The `/search` endpoint calls `g.search_()` with `COMBINED_HYBRID_SEARCH_RRF` — Reciprocal Rank Fusion — which does NOT use the cross_encoder at all. The "reranker_score" in the API response is actually an RRF fusion rank, not a relevance score.

**Scoring pipeline:**
1. `search_()` with RRF config → BM25 + cosine similarity → RRF rank fusion
2. If `search_()` fails → `g.search()` fallback → position-based scoring (1.0, 0.9, 0.8...)
3. Combined score = `reranker_score × decay_weight`
4. When nothing is relevant, RRF still returns baseline scores → everything looks "relevant"

**Three problems:**
1. **Cross-encoder never runs:** ZAIReranker exists but RRF config bypasses it entirely
2. **No relevance threshold:** Irrelevant results get 0.5-1.0 baseline scores and are returned
3. **Position fallback masks quality:** Even when RRF finds nothing relevant, position-based scores make results look meaningful

## Requirements

### Phase 1: Understand + Quick Wins
- R1. Audit the custom server's actual search pipeline — which reranker is used, how scores are computed
- R2. Wire `ZAIReranker` properly if the stock `search_()` path is ever used
- R3. Test if adding a cross-encoder (BGE local) improves irrelevant-query results (penalize bad matches below 1.0)
- R4. Add `ZAI_API_KEY` to server config if needed for the Z.AI reranker

### Phase 2: Search Quality
- R5. For irrelevant queries, return empty results or low-confidence results (not 1.0 baseline)
- R6. Tune BM25/cosine retrieval to reduce false positives
- R7. Add a relevance threshold — don't return results below score X

### Phase 3: Better Reranker (if needed)
- R8. If Z.AI reranker quality is insufficient, swap to BGE local or OpenAI reranker
- R9. Add integration test that validates score spread and relevance

## Success Criteria

- [ ] "quantum physics black holes" returns 0 results (or very low scores < 0.3)
- [ ] Relevant queries show clear score differentiation (top result > 0.8, bottom < 0.3)
- [ ] Dedup via MemVault search works reliably (no false positive duplicates)
- [ ] No regression in ingest latency or search latency
- [ ] We understand exactly how scores are computed (not guessing)

## Scope Boundaries

### In Scope
- Audit custom server search pipeline
- Wire reranker if needed
- Add relevance threshold
- Evaluate BGE local reranker

### Out of Scope
- Reranking training or fine-tuning
- New search endpoints
- Community build improvements (separate issue)
- Guard script changes (Jaccard workaround stays as belt-and-suspenders)
- Replacing the custom server with stock Graphiti

## Approach

### Phase 1: Enable Cross-Encoder (one line change)
In `server.py`, switch from RRF to cross-encoder search config:
```python
# Before:
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
_search_config = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={"limit": req.num_results})

# After:
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER
_search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(update={"limit": req.num_results})
```
This makes the ZAIReranker actually run — BM25 + cosine + BFS results get reranked by the cross-encoder.

### Phase 2: Add Relevance Threshold
After scoring, filter out results below a minimum combined_score:
```python
MIN_SCORE = 0.3  # tunable
results = [r for r in results if r.combined_score >= MIN_SCORE]
```
This stops irrelevant results from returning with 1.0 baseline.

### Phase 3: Evaluate + Swap Reranker (if Z.AI cross-encoder still returns 1.0)
- Test Z.AI cross-encoder with the new config — if it actually differentiates, done
- If still flat scores, swap to BGE local reranker or OpenAI reranker

| Option | Pros | Cons |
|--------|------|------|
| **BGE reranker (local, Pi)** | No API cost, `bge_reranker_client.py` exists, penalizes irrelevant | RAM on Pi, cold start |
| **OpenAI reranker** | High quality, `openai_reranker_client.py` exists | API cost, latency |
| **Keep Z.AI** | Already working for relevant queries | Doesn't penalize irrelevant |

## Open Questions

- [ ] What does the Pi's actual server.py look like? (custom code, not in this repo)
- [ ] Is `reranker_score` a cross-encoder score or RRF rank score?
- [ ] Does the custom server use `self.cross_encoder` or its own pipeline?
- [ ] Does BGE reranker fit in Pi RAM alongside Neo4j + Graphiti?

## Dependencies

- Phase 1: Access to Pi server code (need SSH or read via memory)
- Phase 2: None (config changes)
- Phase 3: `sentence-transformers` if going BGE route
