# Smoke Test Checklist

Run after every refactor phase before proceeding to the next.
A ✅ means the path was verified working. A ❌ means it failed and must be fixed.

Production URL: https://ubuntu.desmana-truck.ts.net/meeting-analyzer

---

## Automated (run test_smoke.sh)

| # | Test | Expected |
|---|------|----------|
| A1 | `GET /health` | 200 `{"status":"ok"}` |
| A2 | `GET /health/live` | 200 `{"status":"alive"}` |
| A3 | `GET /health/ready` | 200 `{"status":"ready"}` (db connected) |
| A4 | `GET /healthz` | 200 |
| A5 | `GET /` (no auth) | 302 redirect to `/auth/login` |
| A6 | `GET /favicon.ico` | 200 |

---

## Manual (requires browser + logged-in session)

### Auth & Navigation
| # | Steps | Expected |
|---|-------|----------|
| M1 | Open https://ubuntu.desmana-truck.ts.net/meeting-analyzer in fresh incognito tab | Redirect to Keycloak login |
| M2 | Log in with valid credentials | Redirect back, landing page renders with workspace list |
| M3 | Reload the page | Workspace list still visible (session persists) |

### Workspace & Folder Management
| # | Steps | Expected |
|---|-------|----------|
| M4 | Create a new workspace from the landing page | Workspace appears in the grid |
| M5 | Create a new folder, move the workspace into it | Workspace shows under folder in breadcrumb |
| M6 | Rename the workspace | New name reflected immediately |
| M7 | Open workspace → verify tabs render (Meetings, Chat, Research, Generate, Todos, Calendar, Documents) | All tabs visible, no JS errors in console |

### Meetings
| # | Steps | Expected |
|---|-------|----------|
| M8 | Upload a short MP3/MP4 recording to a workspace | Upload progress shown; job completes; meeting card appears in Meetings tab |
| M9 | Click a meeting card → detail modal opens | Transcript, summary, and action items visible |

### Chat
| # | Steps | Expected |
|---|-------|----------|
| M10 | Open Chat tab → send a text message | SSE streaming response appears; session saved in sidebar |
| M11 | Ask "create a pdf summary" in chat | SSE streams, PDF download card appears in response |
| M12 | Create a new chat session | Appears in sidebar; old session preserved |

### Research
| # | Steps | Expected |
|---|-------|----------|
| M13 | Open Research tab → type a query → click Run | SSE streams status messages; result card appears |

### Generate
| # | Steps | Expected |
|---|-------|----------|
| M14 | Open Generate tab → create a new task → complete setup → run Questions step | Questions generated; status strip updates |

### Todos & Calendar
| # | Steps | Expected |
|---|-------|----------|
| M15 | Open Todos tab → create a manual todo → change its status → delete it | All operations reflect immediately |
| M16 | Open Calendar tab → create an event → switch between month/week views → delete the event | No rendering errors |

### Documents
| # | Steps | Expected |
|---|-------|----------|
| M17 | Upload a PDF to the Documents tab | Upload progress shown; document appears in list |
| M18 | Click document → preview opens | Document preview renders |
| M19 | Download the document | File downloads without error |

### Settings
| # | Steps | Expected |
|---|-------|----------|
| M20 | Open Global Settings → change a per-workspace LLM model → reload page | Setting persists after reload |

---

## Notes
- Check browser DevTools console for JS errors after each manual test.
- After each phase, record the date and pass/fail result next to each test ID.
- If any automated test fails, do not proceed — fix the regression first.
