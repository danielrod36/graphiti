# Reranker Fix Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make MemVault search return meaningful relevance scores by enabling the cross-encoder reranker and adding a relevance threshold.

**Origin:** docs/brainstorms/2026-04-18-reranker-fix-requirements.md

**Requirements Trace:**
- R1 (Audit pipeline) → done during brainstorm — server.py confirmed
- R2 (Wire reranker) → already wired, not the issue
- R3 (Enable cross-encoder) → Task 1
- R4 (Add relevance threshold) → Task 2
- R5 (Evaluate results) → Task 3
- R8 (Swap reranker if needed) → Task 4 (conditional)

**Architecture:** Switch the `/search` endpoint from RRF (rank fusion, ignores cross-encoder) to CROSS_ENCODER config (which actually calls ZAIReranker.rank()). Add a minimum score threshold to filter noise. If Z.AI cross-encoder still returns flat scores, swap to BGE local reranker.

**Tech Stack:** Python, FastAPI, Graphiti, Z.AI rerank API, sentence-transformers (BGE fallback)

**Server path:** `/home/graphiti/memvault-v2/server.py` on Pi (192.168.3.250)

---

## Task 1: Switch Search Config from RRF to CROSS_ENCODER

**Objective:** Make the ZAIReranker actually fire during search by using the cross-encoder search config.

**Files:**
- Modify: Pi `/home/graphiti/memvault-v2/server.py` (lines 942-943)

**Step 1: Make the change via SSH**

```python
# Line 942-943, change:
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
_search_config = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={"limit": req.num_results})

# To:
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER
_search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(update={"limit": req.num_results})
```

Exact command:
```bash
ssh graphiti@192.168.3.250 "sed -i 's/COMBINED_HYBRID_SEARCH_RRF/COMBINED_HYBRID_SEARCH_CROSS_ENCODER/g' /home/graphiti/memvault-v2/server.py"
```

**Step 2: Restart the server**

```bash
ssh graphiti@192.168.3.250 "sudo systemctl restart memvault" 
# or: ssh graphiti@192.168.3.250 "pkill -f 'uvicorn.*server:app' && cd /home/graphiti/memvault-v2 && nohup python3 server.py &"
```

**Step 3: Verify the server is up**

```bash
curl -s http://192.168.3.250:8003/health
```
Expected: `{"ok":true,"neo4j":true,"queue_depth":0}`

**Step 4: Test relevant query — should show cross-encoder scores**

```bash
curl -s -X POST http://192.168.3.250:8003/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Daniel BambuLab 3D printer","num_results":5}' | python3 -m json.tool
```
Expected: `reranker_score` values should differ from RRF values (cross-encoder scores, not rank fusion).

**Step 5: Test irrelevant query — should show low scores**

```bash
curl -s -X POST http://192.168.3.250:8003/search \
  -H "Content-Type: application/json" \
  -d '{"query":"quantum physics black holes string theory","num_results":5}' | python3 -m json.tool
```
Expected: `reranker_score` values should be LOW (< 0.5) if cross-encoder works. If all 1.0, Z.AI cross-encoder is broken → proceed to Task 4.

**Step 6: Evaluate — decide path forward**

- If scores are differentiated → Task 2 (add threshold)
- If all 1.0 → Task 4 (swap to BGE reranker)

---

## Task 2: Add Relevance Threshold

**Objective:** Filter out irrelevant results by adding a minimum combined_score cutoff.

**Files:**
- Modify: Pi `/home/graphiti/memvault-v2/server.py` (after line 1056, before the return)

**Step 1: Add MIN_SCORE constant near the top of the search endpoint (after line 935)**

```python
MIN_SEARCH_SCORE = 0.3  # Minimum combined_score to return in results
```

**Step 2: Add filter before sorting (around line 1055-1056)**

After:
```python
results.sort(key=lambda r: r.combined_score, reverse=True)
```

Add:
```python
# Filter out low-relevance noise
results = [r for r in results if r.combined_score >= MIN_SEARCH_SCORE]
```

**Step 3: Restart server and test**

```bash
# Restart
ssh graphiti@192.168.3.250 "sudo systemctl restart memvault"

# Test irrelevant query — should return empty or very few results
curl -s -X POST http://192.168.3.250:8003/search \
  -H "Content-Type: application/json" \
  -d '{"query":"quantum physics black holes","num_results":5}' | python3 -m json.tool
```
Expected: `results: []` or only results with combined_score >= 0.3

```bash
# Test relevant query — should still return good results
curl -s -X POST http://192.168.3.250:8003/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Daniel teacher Colombia","num_results":5}' | python3 -m json.tool
```
Expected: 3-5 results with combined_score >= 0.3

**Step 4: Commit**

```bash
ssh graphiti@192.168.3.250 "cd /home/graphiti/memvault-v2 && git add server.py && git commit -m 'fix: enable cross-encoder reranker + add relevance threshold'"
```

---

## Task 3: Verify End-to-End (conditional — only if Task 1+2 succeed)

**Objective:** Confirm search quality improved across diverse queries.

**Step 1: Run the test battery**

```bash
for query in \
  "BambuLab 3D printer" \
  "DaniEdu Django backend" \
  "MemVault Graphiti knowledge graph" \
  "weather forecast" \
  "quantum physics black holes" \
  "completely irrelevant nonsense xyzzy"; do
  echo "=== $query ==="
  curl -s -X POST http://192.168.3.250:8003/search \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$query\",\"num_results\":5}" | \
    python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'  {r[\"combined_score\"]:.3f} | {r[\"reranker_score\"]:.3f} | {r[\"fact\"][:70]}') for r in d.get('results',[])]" 2>/dev/null
  echo ""
done
```

**Success criteria:**
- Relevant queries return 3+ results with good scores
- Irrelevant queries return 0-1 results (filtered by threshold)
- Top result for relevant queries has combined_score > 0.7
- Score range across results is meaningful (> 0.3 spread)

**Step 2: Update MemVault skill file**

Update `~/.hermes/skills/openclaw-imports/memvault/SKILL.md`:
- Remove the "Z.AI Reranker Returns 1.0" known issue
- Document the new search behavior

**Step 3: Ingest the fix into MemVault**

```bash
python3 ~/.hermes/scripts/memvault_guard.py --check "Fixed MemVault search: switched from RRF to CROSS_ENCODER config, added 0.3 relevance threshold. Search now uses ZAIReranker for cross-encoder scoring. Irrelevant queries return empty results instead of noise." --source text --desc "Reranker fix session" --ingest
```

---

## Task 4: Swap to BGE Local Reranker (conditional — only if Z.AI cross-encoder returns 1.0)

**Objective:** Replace Z.AI reranker with a local BGE cross-encoder that returns meaningful relevance scores.

**Prerequisite:** Task 1 showed Z.AI cross-encoder returns flat 1.0 scores.

**Files:**
- Modify: Pi `/home/graphiti/memvault-v2/server.py` — change `make_graphiti()`
- Check: Pi has sentence-transformers installed

**Step 1: Check if sentence-transformers is available on Pi**

```bash
ssh graphiti@192.168.3.250 "source ~/memvault-v2/venv/bin/activate && python3 -c 'from sentence_transformers import CrossEncoder; print(\"OK\")'"
```

If not installed:
```bash
ssh graphiti@192.168.3.250 "source ~/memvault-v2/venv/bin/activate && pip install sentence-transformers"
```

**Step 2: Check Pi RAM usage**

```bash
ssh graphiti@192.168.3.250 "free -h"
```
BGE reranker needs ~500MB-1GB. Pi has 8GB total. Check if there's room.

**Step 3: Change the reranker in make_graphiti()**

In server.py `make_graphiti()`, change:
```python
from graphiti_core.cross_encoder.zai_reranker import ZAIReranker
# ...
cross_encoder=ZAIReranker(api_key=ZAI_KEY),
```

To:
```python
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
# ...
cross_encoder=BGERerankerClient(),
```

**Step 4: Restart and test**

```bash
ssh graphiti@192.168.3.250 "sudo systemctl restart memvault"
# Wait for model load (first time may take 30s)
sleep 10
curl -s http://192.168.3.250:8003/health

# Test irrelevant query
curl -s -X POST http://192.168.3.250:8003/search \
  -H "Content-Type: application/json" \
  -d '{"query":"quantum physics black holes","num_results":5}' | python3 -m json.tool
```
Expected: Low scores (< 0.3) for irrelevant results.

**Step 5: Commit**

```bash
ssh graphiti@192.168.3.250 "cd /home/graphiti/memvault-v2 && git add server.py && git commit -m 'fix: swap Z.AI reranker for BGE local cross-encoder'"
```

---

## Verification Checklist

- [ ] Server starts cleanly after changes
- [ ] Relevant queries return meaningful results with good score differentiation
- [ ] Irrelevant queries return empty or very low-score results
- [ ] Dedup via MemVault search works reliably (no false positives)
- [ ] No regression in ingest latency
- [ ] No regression in search latency (< 2s for typical queries)
- [ ] Guard script dedup still works (Jaccard workaround as belt-and-suspenders)
- [ ] MemVault skill file updated to remove known issue
- [ ] Fix ingested into MemVault knowledge graph
