# Architecture

## Stack

| Component | Location | Notes |
|-----------|----------|-------|
| Backend | `app/` (this directory) | FastAPI, served via `main_live.py` |
| Frontend | `app/static/index.html` | Standalone HTML file (~12k lines), served by FastAPI |
| K8s | `../k8s/` | Deployment manifests |

## Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| backend | Already present | FastAPI + asyncpg |
| web | Already present | `static/index.html` (not a scaffolded SPA) |
| ios | Not requested | — |
| android | Not requested | — |

## Key Files

- `main_live.py` — FastAPI entrypoint for the Meeting Analyzer
- `static/index.html` — Deployed frontend UI
- `Dockerfile` — Builds `whisper-analyzer` image
- `db_schema.py` — Database schema definitions

## Next Steps

1. For frontend changes: Edit `static/index.html` directly
2. For backend changes: Edit `main_live.py` or related Python files
3. To deploy: Use `/pm:deploy analyzer` or the K8s manifests in `../k8s/`

## Notes

- The `web-portal/` directory at the repository root is an undeployed Angular prototype — it is not connected to the live application
- All production frontend changes go in `static/index.html`
