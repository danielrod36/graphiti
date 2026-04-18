---
title: "Audit reviewed wrong codebase — Pi runs full Graphiti, not simple server.py"
category: "workflow-issues"
track: "bug"
problem_type: "wrong_context"
module: "memvault/ce-review"
tags: [audit, code-review, graphiti, memvault, ce-review, false-positive]
date: 2026-04-18
difficulty: "moderate"
---

# CE Review False Positives from Wrong Codebase

## Problem
Multi-persona CE review spawned subagents that reviewed the local Graphiti fork at `/home/danielclaw/graphiti/`, but the Pi deployment (192.168.3.250:8003) runs the full Graphiti MemVault v2 server with Neo4j — a completely different architecture. 4 P0 findings were all false positives.

## Symptoms
- Reviewers found `reranker_min_score=0` default → "no threshold filtering"
- Reviewers found BGE reranker with raw logits → "threshold 0.3 wrong"
- Reviewers found BGERerankerClient not exported → "can't import"
- All findings contradicted by live Pi behavior (threshold filtering works, scores normalized)

## What Didn't Work
- Spawning review subagents with file paths to local fork, not actual deployment
- Assuming the Pi ran a simple FastAPI server.py — it runs Graphiti with ZepGraphiti class
- Not verifying findings against live API before presenting as P0

## Solution
**Before reviewing code, verify what's actually deployed:**
1. Check the API schema (OpenAPI `/openapi.json`) to understand actual server architecture
2. Run live behavior tests (search irrelevant/relevant queries) to verify claimed issues
3. Cross-reference review findings against live behavior before severity classification

### Verification Pattern
```python
# 1. Check actual server architecture
spec = GET /openapi.json
# → Shows real endpoints, schemas, config

# 2. Live behavior test
search("irrelevant query") → 0 results = threshold works
search("relevant query") → results with scores = reranker works

# 3. Score distribution test
search("very specific") vs search("broad") vs search("irrelevant")
# → Shows if scores are differentiated or flat
```

## Why This Works
The Pi deployment has custom server code that wraps Graphiti with its own threshold/filtering logic. The local fork is the upstream source but the Pi has local modifications. Reviewing the fork tells you about the library, not the deployment.

## Prevention
- Always run live verification tests before presenting audit findings
- Check `GET /openapi.json` or equivalent to understand actual server API
- Distinguish between "issues in the library" vs "issues in the deployment"
- Mark findings as "local fork" vs "Pi deployment" when codebases diverge
- Add "verification" step to ce-review skill before presenting findings

## Related
- `docs/solutions/reranker-fix-bge-local-2026-04-18.md` — Original reranker fix
- CE review skill: `compound-engineering/ce-review/SKILL.md`
