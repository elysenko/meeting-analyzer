"""
Application configuration — environment variables and business constants.

Imported by main_live.py (and later by individual router modules).
This module has no imports from the rest of the app, only stdlib.
"""

import os
import re

# Load .env from the app directory (values are only set if not already in the environment,
# so Kubernetes env vars always take precedence)
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ---------------------------------------------------------------------------
# Service URLs
# ---------------------------------------------------------------------------
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper.whisper.svc.cluster.local:9000")
WHISPER_LIVE_URL = os.getenv("WHISPER_LIVE_URL", "ws://whisper-live.whisper.svc.cluster.local:9090")
WHISPER_CHUNK_SECONDS = int(os.getenv("WHISPER_CHUNK_SECONDS", "600"))
WHISPER_CHUNK_SIZE_MB = int(os.getenv("WHISPER_CHUNK_SIZE_MB", "32"))
WHISPER_RETRY_COUNT = int(os.getenv("WHISPER_RETRY_COUNT", "2"))
PIPER_TTS_URL = os.getenv("PIPER_TTS_URL", "http://piper-tts.whisper.svc.cluster.local:5000")
LLM_RUNNER_URL = os.getenv("LLM_RUNNER_URL", "http://llm-runner-dev.aion.svc.cluster.local:8000")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "/models/en_US-lessac-medium.onnx")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://analyzer:analyzer-pass-2026@analyzer-postgres:5432/analyzer",
)

# ---------------------------------------------------------------------------
# MinIO / Object Storage
# ---------------------------------------------------------------------------
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio.whisper.svc.cluster.local:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "mp4-reader")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio-mp4reader")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio-mp4reader-secret-2026")

# ---------------------------------------------------------------------------
# Keycloak OAuth2/OIDC
# ---------------------------------------------------------------------------
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://keycloak.keycloak.svc.cluster.local:8080")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "mycluster")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "meeting-analyzer")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "")
KEYCLOAK_CALLBACK_URL = os.getenv("KEYCLOAK_CALLBACK_URL", "http://localhost:30903/auth/callback")
KEYCLOAK_EXTERNAL_URL = os.getenv("KEYCLOAK_EXTERNAL_URL", "")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "")
APP_PATH_PREFIX = os.getenv("APP_PATH_PREFIX", "")

# ---------------------------------------------------------------------------
# LiveKit
# ---------------------------------------------------------------------------
LIVEKIT_WS_URL = os.getenv("LIVEKIT_WS_URL", "wss://jetson-orin.desmana-truck.ts.net")
LIVEKIT_INTERNAL_URL = os.getenv("LIVEKIT_INTERNAL_URL", "http://livekit-service.robert.svc.cluster.local:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret123456789")

# ---------------------------------------------------------------------------
# Team Chat Widget (Matrix-based)
# ---------------------------------------------------------------------------
TEAM_CHAT_HOMESERVER = os.getenv("TEAM_CHAT_HOMESERVER", "")
TEAM_CHAT_TOKEN = os.getenv("TEAM_CHAT_TOKEN", "")
TEAM_CHAT_USER_ID = os.getenv("TEAM_CHAT_USER_ID", "")
TEAM_CHAT_ROOM_ID = os.getenv("TEAM_CHAT_ROOM_ID", "")

# ---------------------------------------------------------------------------
# Research / LLM Runner paths
# ---------------------------------------------------------------------------
QUICK_RESEARCH_ROOT = os.getenv("QUICK_RESEARCH_ROOT", "/home/ubuntu/qr")
DEEP_RESEARCH_ROOT = os.getenv("DEEP_RESEARCH_ROOT", "/home/ubuntu/dr")
LLM_RUNNER_CONFIG_PATH = os.getenv("LLM_RUNNER_CONFIG_PATH", "/home/ubuntu/llm-runner/llm-config.yaml")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# ---------------------------------------------------------------------------
# LLM task / provider constants
# ---------------------------------------------------------------------------
LLM_TASK_KEYS = ("analysis", "chat", "research", "generate")
SEARCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_LLM_PREFERENCES = {key: {"provider": None, "model": None} for key in LLM_TASK_KEYS}
RESEARCH_TYPE_OVERLAYS = {
    "financial": os.path.join(DEEP_RESEARCH_ROOT, "skills", "deep-research", "FINANCIAL.md"),
    "finance": os.path.join(DEEP_RESEARCH_ROOT, "skills", "deep-research", "FINANCIAL.md"),
    "legal": os.path.join(DEEP_RESEARCH_ROOT, "skills", "deep-research", "LEGAL.md"),
    "healthcare": os.path.join(DEEP_RESEARCH_ROOT, "skills", "deep-research", "HEALTHCARE.md"),
    "market": os.path.join(DEEP_RESEARCH_ROOT, "skills", "deep-research", "MARKET.md"),
}
DR_REFINE_GUIDANCE_PATH = os.path.join(DEEP_RESEARCH_ROOT, "commands", "dr-refine.md")
LLM_PROVIDER_LABELS = {
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI",
    "google": "Google Gemini",
    "groq": "Groq",
    "kimi": "Moonshot Kimi",
    "kimi-cli": "Kimi CLI (Agentic)",
    "claude-code": "Claude Code (Agentic)",
    "codex": "Codex CLI (Agentic)",
}
AGENTIC_LLM_PROVIDERS = {"claude-code", "codex", "kimi-cli"}

# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------
DOCUMENT_CHUNK_TARGET_CHARS = 1400
DOCUMENT_CHUNK_MIN_CHARS = 700
DOCUMENT_CHUNK_OVERLAP_CHARS = 220
DOCUMENT_QUERY_TERM_LIMIT = 18
DOCUMENT_RETRIEVAL_LIMIT = 6

# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------
DEFAULT_CHAT_SESSION_TITLE = "New Chat"
LEGACY_CHAT_SESSION_TITLE = "Previous Chat"
CHAT_SESSION_TITLE_LIMIT = 96

# ---------------------------------------------------------------------------
# Generate template catalog
# ---------------------------------------------------------------------------
GENERATE_TEMPLATE_CATALOG = {
    "requirements": {
        "label": "Requirements Document",
        "description": "Define scope, requirements, constraints, and approvals.",
        "default_output": "pdf",
        "base_questions": [
            {"key": "audience", "label": "Who is the intended audience?", "group": "Audience", "input_type": "textarea", "required": True, "placeholder": "Who will read and approve this document?"},
            {"key": "problem_statement", "label": "What problem is this deliverable solving?", "group": "Problem", "input_type": "textarea", "required": True, "placeholder": "Summarize the business or user problem."},
            {"key": "scope", "label": "What is in scope?", "group": "Scope", "input_type": "textarea", "required": True, "placeholder": "List the features, workflows, or outcomes that must be covered."},
            {"key": "out_of_scope", "label": "What is explicitly out of scope?", "group": "Scope", "input_type": "textarea", "required": False, "placeholder": "What should not be included?"},
            {"key": "functional_requirements", "label": "What functional requirements are known so far?", "group": "Requirements", "input_type": "textarea", "required": True, "placeholder": "List concrete functional needs, behaviors, or capabilities."},
            {"key": "non_functional_requirements", "label": "What non-functional requirements matter?", "group": "Requirements", "input_type": "textarea", "required": False, "placeholder": "Performance, security, compliance, accessibility, reliability."},
            {"key": "dependencies", "label": "What dependencies or integrations are involved?", "group": "Dependencies", "input_type": "textarea", "required": False, "placeholder": "Teams, systems, vendors, approvals, and integrations."},
            {"key": "risks", "label": "What major risks or open questions remain?", "group": "Risks", "input_type": "textarea", "required": False, "placeholder": "Unknowns, blockers, assumptions, and tradeoffs."},
            {"key": "timeline", "label": "What timeline or milestones should be reflected?", "group": "Timeline", "input_type": "textarea", "required": False, "placeholder": "Deadlines, phases, delivery dates, checkpoints."},
            {"key": "approvals", "label": "Who needs to approve or sign off?", "group": "Approvals", "input_type": "textarea", "required": False, "placeholder": "Stakeholders, approvers, or accountable owners."},
        ],
    },
    "proposal": {
        "label": "Proposal",
        "description": "Persuade stakeholders around a recommended plan.",
        "default_output": "pdf",
        "base_questions": [
            {"key": "audience", "label": "Who is this proposal for?", "group": "Audience", "input_type": "textarea", "required": True, "placeholder": "Who will read this proposal?"},
            {"key": "recommended_solution", "label": "What is the recommended solution or plan?", "group": "Proposal", "input_type": "textarea", "required": True, "placeholder": "State the recommendation clearly."},
            {"key": "business_case", "label": "What is the business case?", "group": "Proposal", "input_type": "textarea", "required": True, "placeholder": "Why should this happen now?"},
            {"key": "costs", "label": "What costs, resources, or budget should be covered?", "group": "Proposal", "input_type": "textarea", "required": False, "placeholder": "Budget, staffing, tooling, or investment."},
            {"key": "timeline", "label": "What timeline should be proposed?", "group": "Timeline", "input_type": "textarea", "required": False, "placeholder": "Phases, deadlines, milestones."},
            {"key": "risks", "label": "What risks or tradeoffs need explicit discussion?", "group": "Risks", "input_type": "textarea", "required": False, "placeholder": "Risks, objections, and mitigations."},
        ],
    },
    "executive_deck": {
        "label": "Executive Deck",
        "description": "Build an executive-facing slide deck for decisions or updates.",
        "default_output": "pptx",
        "base_questions": [
            {"key": "audience", "label": "Who is the deck for?", "group": "Audience", "input_type": "textarea", "required": True, "placeholder": "Executives, customers, leadership, board, etc."},
            {"key": "core_message", "label": "What is the single most important message?", "group": "Narrative", "input_type": "textarea", "required": True, "placeholder": "What should the audience remember?"},
            {"key": "supporting_points", "label": "What supporting points must be included?", "group": "Narrative", "input_type": "textarea", "required": True, "placeholder": "Data, rationale, milestones, evidence, asks."},
            {"key": "decision_needed", "label": "What decision or action is needed from the audience?", "group": "Narrative", "input_type": "textarea", "required": False, "placeholder": "Approval, funding, prioritization, staffing, etc."},
            {"key": "risks", "label": "What risks or blockers should be called out?", "group": "Risks", "input_type": "textarea", "required": False, "placeholder": "What could derail the plan?"},
        ],
    },
    "custom": {
        "label": "Custom",
        "description": "Use freeform guidance within the task flow.",
        "default_output": "pdf",
        "base_questions": [
            {"key": "deliverable_goal", "label": "What should the deliverable accomplish?", "group": "Goal", "input_type": "textarea", "required": True, "placeholder": "Describe the desired outcome."},
            {"key": "custom_prompt", "label": "What should the artifact contain?", "group": "Prompt", "input_type": "textarea", "required": True, "placeholder": "Describe the structure, emphasis, and content needed."},
            {"key": "audience", "label": "Who is the audience?", "group": "Audience", "input_type": "textarea", "required": False, "placeholder": "Who will consume this artifact?"},
            {"key": "tone", "label": "What tone or style should it use?", "group": "Style", "input_type": "textarea", "required": False, "placeholder": "Formal, persuasive, concise, technical, etc."},
        ],
    },
}

# ---------------------------------------------------------------------------
# Todo constants
# ---------------------------------------------------------------------------
TODO_STATUS_OPTIONS = ("incomplete", "in_progress", "complete")
TODO_STATUS_ALIASES = {
    "open": "incomplete",
    "todo": "incomplete",
    "blocked": "in_progress",
    "doing": "in_progress",
    "done": "complete",
    "closed": "complete",
}
TODO_STATUS_LABELS = {
    "incomplete": "Incomplete",
    "in_progress": "In Progress",
    "complete": "Complete",
}

# ---------------------------------------------------------------------------
# Misc / Markdown
# ---------------------------------------------------------------------------
RESEARCH_MATCH_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "into", "your", "have", "will",
    "about", "what", "when", "where", "which", "should", "could", "would", "task", "create",
    "needs", "need", "meeting", "document", "research", "deliverable",
}
HEX_COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")
MARKDOWN_ALLOWED_TAGS = [
    "a", "blockquote", "br", "code", "em", "h1", "h2", "h3", "h4", "h5", "h6",
    "hr", "li", "ol", "p", "pre", "strong", "table", "tbody", "td", "th", "thead", "tr", "ul",
]
MARKDOWN_ALLOWED_ATTRIBUTES = {
    "a": ["href", "title", "target", "rel"],
    "th": ["colspan", "rowspan"],
    "td": ["colspan", "rowspan"],
}
