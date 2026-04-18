---
title: "MemVault Pi deployment — SSH, data quality, and community detection pitfalls"
category: "workflow-issues"
track: "knowledge"
problem_type: "deployment"
module: "memvault-v2"
tags: [deployment, pi, ssh, neo4j, community-detection, data-quality, graphiti]
date: 2026-04-18
difficulty: "moderate"
---

# MemVault Pi Deployment Pitfalls

## Context
Deploying Graphiti fork changes to the Pi (192.168.3.250) and fixing data quality issues that blocked community detection.

## SSH Access
- **User:** `graphiti@192.168.3.250` (NOT `ubuntu@` or `pi@`)
- **No SSH config file** — must use `graphiti@` explicitly
- **No passwordless sudo** — `sudo -n` fails; need password for systemctl restart
- **Neo4j credentials:** user=neo4j, pass=`change_me_password_in_prod` (in `/home/graphiti/memvault-v2/.env`)

## Git Deployment Pattern
The Pi's repo has local commits not on GitHub. Standard `git pull` fails with divergent branches.

```bash
# On Pi:
cd /home/graphiti/memvault-v2/graphiti
git fetch origin
git reset --hard origin/main   # force sync, discards local changes
```

The Pi uses HTTPS remote (set via `git remote set-url origin https://github.com/danielrod36/graphiti.git`) because SSH keys aren't configured for GitHub.

## Server Architecture
- **NOT** the stock `server/graph_service/` from the fork
- Custom 65KB `server.py` at `/home/graphiti/memvault-v2/server.py`
- Systemd service: `memvault-v2.service`
- New endpoints must be added directly to this file (not the fork's server)
- `PYTHONPATH=/home/graphiti/memvault-v2/graphiti` — uses fork's graphiti_core
- Custom `optimized_clusters.py` patches community clustering at startup

## Data Quality Issues
1. **Null entity names** — 1 entity had `name=NULL`, caused EntityNode Pydantic validation to crash community build and ingest
   - Fix: `MATCH (n:Entity) WHERE n.name IS NULL SET n.name = "Unknown Entity"`
2. **Entity type NULL** — 968/968 entities have `entity_type=NULL` in Neo4j (but have label-based types like Person, Service, etc.)
3. **optimized_clusters.py** — `entity_node_from_record()` crashes on invalid records; must wrap in try/except

## Community Detection
- `build_communities()` exists in Graphiti but wasn't exposed on Pi
- Added `POST /communities/build` endpoint to Pi's server.py
- Uses `community_build_lock = asyncio.Lock()` — if build crashes, lock stays held until server restart
- The Pi has a custom `optimized_clusters.py` that patches clustering (2 Neo4j queries instead of O(N), max iteration guard)
- Build can take 2-5 min on 968 entities

## Response Format
Pi's `/search` returns `{"results": [...]}` not `{"facts": [...]}` — check response key when parsing.

## Pitfalls
- **Don't modify fork's `server/graph_service/`** — Pi uses custom server.py
- **Don't assume sudo works** — graphiti user needs password
- **Don't use `git pull` on Pi** — divergent branches; use `git reset --hard origin/main`
- **Do wrap entity_node_from_record** in try/except — null names crash Pydantic validation
- **Do clear community_build_lock** by restarting server if build fails
- **Do check response key** — `results` vs `facts` varies by endpoint version

## Related
- `docs/solutions/reranker-fix-bge-local-2026-04-18.md` — Reranker investigation
- `docs/solutions/workflow-issues/audit-wrong-codebase-2026-04-18.md` — Audit methodology
