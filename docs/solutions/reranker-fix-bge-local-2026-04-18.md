# Reranker Fix — BGE Local Cross-Encoder (April 18, 2026)

## Problem
MemVault search returned irrelevant results with high scores. Root cause: the Z.AI reranker
returned ~0.94-1.0 for everything (no differentiation), and the search endpoint used RRF
(which didn't even call the cross-encoder).

## What We Did
1. **Audited the Pi server code** (custom `server.py`, not stock Graphiti)
2. **Switched search config** from `COMBINED_HYBRID_SEARCH_RRF` to `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`
3. **Replaced ZAIReranker with BGERerankerClient** (`BAAI/bge-reranker-v2-m3`)
4. **Added relevance threshold** (MIN_SEARCH_SCORE=0.3)
5. **Installed sentence-transformers** on Pi (CPU-only torch from pytorch.org/whl/cpu)

## Results
- Relevant queries: scores 0.7-0.99 (clear differentiation)
- Irrelevant queries: scores ~0.0 (filtered by threshold)
- "BambuLab 3D printer" → 2 relevant results (was 5 mixed)
- "quantum physics" → 0 results (was 5 random)

## Trade-off
- BGE on Pi 4B CPU: ~7-30s per search query
- Acceptable for background/async searches, too slow for interactive
- Plan: migrate to GPU hardware when available

## Key Files
- Pi server: `graphiti@192.168.3.250:/home/graphiti/memvault-v2/server.py`
- BGE client: `~/graphiti/graphiti_core/cross_encoder/bge_reranker_client.py`
- Search config recipes: `~/graphiti/graphiti_core/search/search_config_recipes.py`

## Lesson
The Z.AI rerank API returns flat scores (~0.94-1.0 for everything). For meaningful relevance
scoring, a local cross-encoder (BGE) is essential. The RRF config bypasses the cross-encoder
entirely — must use CROSS_ENCODER config to activate it.
