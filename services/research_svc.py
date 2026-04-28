"""
Research service helpers.

Refinement questions, research session DB helpers, quick/deep research orchestration.
All DB-touching functions accept a `pool` parameter (asyncpg pool).
"""

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from config import (
    DEEP_RESEARCH_ROOT,
    DOCUMENT_RETRIEVAL_LIMIT,
    DR_REFINE_GUIDANCE_PATH,
    QUICK_RESEARCH_ROOT,
    RESEARCH_TYPE_OVERLAYS,
)
from services.text_svc import _excerpt_text, _render_markdown_html

logger = logging.getLogger("meeting-analyzer")


# ---------------------------------------------------------------------------
# Guidance file loaders
# ---------------------------------------------------------------------------


def _read_optional_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def load_dr_refine_guidance() -> str:
    return _read_optional_text(DR_REFINE_GUIDANCE_PATH)


def load_quick_research_guidance() -> str:
    parts = [
        _read_optional_text(os.path.join(QUICK_RESEARCH_ROOT, "commands", "qr.md")),
        _read_optional_text(os.path.join(QUICK_RESEARCH_ROOT, "agents", "quick-research.md")),
        _read_optional_text(os.path.join(QUICK_RESEARCH_ROOT, "skills", "quick-research", "SYNTHESIS.md")),
    ]
    return "\n\n".join(part for part in parts if part)


def load_deep_research_guidance(research_type: str) -> str:
    parts = [
        _read_optional_text(os.path.join(DEEP_RESEARCH_ROOT, "commands", "dr.md")),
        _read_optional_text(os.path.join(DEEP_RESEARCH_ROOT, "agents", "deep-research.md")),
    ]
    overlay = RESEARCH_TYPE_OVERLAYS.get((research_type or "").strip().lower())
    if overlay:
        parts.append(_read_optional_text(overlay))
    return "\n\n".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# Refinement helpers
# ---------------------------------------------------------------------------


def _refinement_question(
    question_id: str,
    phase: str,
    label: str,
    prompt: str,
    placeholder: str,
) -> dict[str, str]:
    return {
        "id": question_id,
        "phase": phase,
        "label": label,
        "prompt": prompt,
        "placeholder": placeholder,
    }


def _normalize_research_refinement(refinement: Any) -> dict[str, Any] | None:
    if not isinstance(refinement, dict):
        return None
    query_type = str(refinement.get("query_type") or "").strip().upper().replace(" ", "_")
    answers = []
    for item in refinement.get("answers") or []:
        if not isinstance(item, dict):
            continue
        answer_text = str(item.get("answer") or "").strip()
        if not answer_text:
            continue
        answers.append({
            "id": str(item.get("id") or "").strip()[:80],
            "phase": str(item.get("phase") or "").strip()[:80],
            "label": str(item.get("label") or "").strip()[:160],
            "prompt": str(item.get("prompt") or "").strip()[:1000],
            "answer": answer_text[:4000],
        })
    if not query_type and not answers:
        return None
    suggested_subquestions = [
        str(item).strip()[:500]
        for item in (refinement.get("suggested_subquestions") or [])
        if str(item).strip()
    ][:8]
    return {
        "query_type": query_type or None,
        "suggested_reframe": str(refinement.get("suggested_reframe") or "").strip()[:1000] or None,
        "success_criteria_hint": str(refinement.get("success_criteria_hint") or "").strip()[:1000] or None,
        "answers": answers,
        "suggested_subquestions": suggested_subquestions,
    }


def _infer_research_query_type(topic: str) -> str:
    lowered = (topic or "").lower()
    if any(token in lowered for token in ("risk", "due diligence", "failure", "go wrong", "exposure")):
        return "DUE_DILIGENCE"
    if any(token in lowered for token in ("validate", "verify", "is it true", "check whether", "fact check")):
        return "VALIDATION"
    if any(token in lowered for token in ("should", "choose", "select", "vs", "versus", "compare", "better", "best")):
        return "DECISION"
    if any(token in lowered for token in ("understand", "learn", "explain", "how does", "what is")):
        return "LEARNING"
    if any(token in lowered for token in ("explore", "options", "direction", "where do we start")):
        return "EXPLORATION"
    return "EXPLORATION"


def _split_freeform_list(text: str, limit: int = 6) -> list[str]:
    values = []
    for chunk in re.split(r"[\n;]+", text or ""):
        item = chunk.strip(" -\t")
        if item:
            values.append(item[:400])
        if len(values) >= limit:
            break
    return values


def _dedupe_preserve_order(items: list[str], limit: int = 8) -> list[str]:
    results = []
    seen = set()
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(cleaned[:500])
        if len(results) >= limit:
            break
    return results


def _get_refinement_answer(refinement: dict[str, Any], *ids: str) -> str:
    answers = refinement.get("answers") or []
    for answer in answers:
        if answer.get("id") in ids:
            return str(answer.get("answer") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Source prompt formatting
# ---------------------------------------------------------------------------


def sources_prompt_block(sources: list[dict[str, Any]]) -> str:
    blocks = []
    for source in sources:
        blocks.append(
            f'{source["id"]}: {source["title"]}\n'
            f'URL: {source["url"]}\n'
            f'Domain: {source.get("domain", "")}\n'
            f'Query: {source.get("query", "")}\n'
            f'Content:\n{source.get("content", "")}\n'
        )
    return "\n\n".join(blocks)


def source_refs(source_ids: list[str] | list[Any], source_lookup: dict[str, dict[str, Any]]) -> str:
    refs = []
    for source_id in source_ids or []:
        sid = str(source_id)
        source = source_lookup.get(sid)
        if not source:
            continue
        refs.append(f'([{source.get("domain") or source.get("title")}]({source["url"]}))')
    return " ".join(refs).strip()


def render_quick_research_markdown(
    payload: dict[str, Any],
    sources: list[dict[str, Any]],
    prior_research: list[dict[str, Any]] | None = None,
) -> str:
    source_lookup = {source["id"]: source for source in sources}
    detail_lines = []
    for item in payload.get("details") or []:
        if isinstance(item, dict):
            claim = item.get("claim", "").strip()
            refs = source_refs(item.get("source_ids") or [], source_lookup)
            if claim:
                detail_lines.append(f"- {claim}{(' ' + refs) if refs else ''}")
        elif isinstance(item, str):
            detail_lines.append(f"- {item}")
    source_lines = [f'- [{source["title"]}]({source["url"]})' for source in sources]
    prior_lines = [
        f'- {item.get("title") or item.get("topic")} ({item.get("mode") or "research"})'
        for item in (prior_research or [])
        if item.get("title") or item.get("topic")
    ]
    return "\n".join([
        f'## {payload.get("title") or "Research Brief"}',
        "",
        f'**Key Finding**: {payload.get("key_finding") or ""}',
        "",
        "**Details**:",
        *(detail_lines or ["- No specific claims extracted."]),
        "",
        f'**Context**: {payload.get("context") or ""}',
        "",
        f'**Caveats**: {payload.get("caveats") or ""}',
        "",
        f'**Confidence**: {payload.get("confidence") or "MEDIUM"} — {payload.get("confidence_rationale") or ""}',
        "",
        "**Referenced Research**:",
        *(prior_lines or ["- No prior research was referenced."]),
        "",
        "**Sources**:",
        *(source_lines or ["- No sources collected."]),
    ]).strip()


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------


def _fallback_prior_coverage(prior_research: list[dict[str, Any]] | None) -> str:
    items = prior_research or []
    if not items:
        return "No prior research was linked."
    titles = [item.get("title") or item.get("topic") or f"Research {item.get('id')}" for item in items[:4]]
    return "Existing linked research covers: " + "; ".join(titles) + "."


def _fallback_research_queries(
    topic: str,
    research_type: str,
    refinement_contract: dict[str, Any] | None,
    *,
    max_queries: int,
) -> list[str]:
    seed = (refinement_contract or {}).get("refined_question") or topic
    queries = [seed]
    for item in (refinement_contract or {}).get("subquestions") or []:
        query = str(item or "").strip()
        if not query:
            continue
        if seed.lower() not in query.lower():
            query = f"{seed} {query}"
        queries.append(query)
    queries.append(f"{seed} best practices")
    queries.append(f"{seed} examples")
    if research_type and research_type != "general":
        queries.append(f"{seed} {research_type}")
    return _dedupe_preserve_order(queries, max_queries)


def _fallback_quick_synthesis_payload(
    topic: str,
    research_type: str,
    sources: list[dict[str, Any]],
    prior_research: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    detail_items = []
    for source in sources[:5]:
        basis = _excerpt_text(source.get("content") or source.get("title") or "", 240)
        if not basis:
            continue
        detail_items.append({
            "claim": basis,
            "source_ids": [source.get("id")],
        })
    first_source = sources[0] if sources else {}
    key_finding_basis = _excerpt_text(first_source.get("content") or first_source.get("title") or topic, 220)
    return {
        "title": f"Quick Research: {topic}",
        "key_finding": key_finding_basis or f"Collected source material for {topic}.",
        "details": detail_items,
        "context": (
            _fallback_prior_coverage(prior_research)
            + f" Research type: {research_type or 'general'}."
        ).strip(),
        "caveats": "Fallback synthesis was used because the selected model did not return valid JSON.",
        "confidence": "MEDIUM" if len(sources) >= 4 else "LOW",
        "confidence_rationale": f"Based on {len(sources)} collected source(s) summarized deterministically.",
    }


def _build_local_context_sources(
    document_evidence: list[dict[str, Any]] | None,
    prior_research: list[dict[str, Any]] | None = None,
    *,
    max_sources: int = 8,
) -> list[dict[str, Any]]:
    from services.documents_svc import _dedupe_document_refs  # noqa: PLC0415

    sources: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for index, item in enumerate(_dedupe_document_refs(document_evidence or [], limit=max_sources), start=1):
        title = str(item.get("filename") or f"Document {item.get('document_id') or index}").strip()
        if not title:
            continue
        normalized = title.lower()
        if normalized in seen_titles:
            continue
        seen_titles.add(normalized)
        sources.append({
            "id": f"L{index}",
            "title": title,
            "url": "#",
            "domain": "local-document",
            "query": "attached document evidence",
            "content": _excerpt_text(item.get("snippet"), 420),
        })
        if len(sources) >= max_sources:
            return sources
    for item in (prior_research or [])[: max(0, max_sources - len(sources))]:
        title = str(item.get("title") or item.get("topic") or "").strip()
        if not title:
            continue
        normalized = title.lower()
        if normalized in seen_titles:
            continue
        seen_titles.add(normalized)
        sources.append({
            "id": f"R{item.get('id') or len(sources) + 1}",
            "title": title,
            "url": "#",
            "domain": "workspace-research",
            "query": "previous workspace research",
            "content": _excerpt_text(item.get("summary") or item.get("content"), 420),
        })
        if len(sources) >= max_sources:
            break
    return sources


def fallback_local_quick_research_result(
    topic: str,
    research_type: str,
    *,
    task_brief: str | None = None,
    document_evidence: list[dict[str, Any]] | None = None,
    prior_research: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from services.documents_svc import _dedupe_document_refs  # noqa: PLC0415

    local_sources = _build_local_context_sources(document_evidence, prior_research, max_sources=8)
    synthesis = _fallback_quick_synthesis_payload(topic, research_type, local_sources, prior_research)
    synthesis["title"] = f"Grounded Task Research: {topic}"
    synthesis["key_finding"] = (
        _excerpt_text(task_brief, 220)
        or synthesis.get("key_finding")
        or f"Used the attached workspace context to ground {topic}."
    )
    synthesis["context"] = (
        "Used the current Setup context, attached documents, and prior workspace research because web search did not return usable sources."
    )
    synthesis["caveats"] = (
        "This fallback is grounded in local workspace context only. Review the attached document evidence before treating it as externally validated research."
    )
    synthesis["confidence"] = "MEDIUM" if local_sources else "LOW"
    synthesis["confidence_rationale"] = (
        f"Based on {len(local_sources)} local context source(s) from attached documents and saved research."
        if local_sources
        else "Only minimal local task context was available."
    )
    result = {
        "title": synthesis.get("title") or topic,
        "summary": synthesis.get("key_finding") or "",
        "content": render_quick_research_markdown(synthesis, local_sources, prior_research),
        "sources": local_sources,
        "mode": "quick",
        "research_type": research_type,
        "refinement": None,
        "source_research_ids": [item["id"] for item in (prior_research or []) if item.get("id")],
        "source_document_refs": _dedupe_document_refs(document_evidence or []),
    }
    meta = {
        "provider": None,
        "model": None,
        "warning": "No usable web sources were found; continued with grounded local context only.",
    }
    return result, meta


def _fallback_deep_research_markdown(
    topic: str,
    plan: dict[str, Any],
    sources: list[dict[str, Any]],
    prior_research: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    source_lines = []
    for source in sources[:8]:
        excerpt = _excerpt_text(source.get("content") or "", 320)
        cite = source.get("id") or source.get("title") or "S?"
        line = f"- [{cite}] {source.get('title') or source.get('url')}"
        if excerpt:
            line += f": {excerpt}"
        source_lines.append(line)
    prior_lines = []
    for item in (prior_research or [])[:4]:
        title = item.get("title") or item.get("topic") or f"Research {item.get('id')}"
        summary = _excerpt_text(item.get("summary") or "", 220)
        prior_lines.append(f"- {title}: {summary or 'Previously completed workspace research.'}")
    subquestions = [str(item).strip() for item in (plan.get("subquestions") or []) if str(item).strip()]
    summary = _excerpt_text((sources[0].get("content") if sources else "") or topic, 260)
    report = "\n".join([
        f"## {topic}",
        "",
        "### Overview",
        summary or f"Collected source material for {topic}.",
        "",
        "### Existing Research Coverage",
        *(prior_lines or ["- No prior workspace research was linked."]),
        "",
        "### Key Questions",
        *([f"- {item}" for item in subquestions] or [f"- {topic}"]),
        "",
        "### Source Highlights",
        *(source_lines or ["- No source highlights were available."]),
        "",
        "### Caveats",
        "- Fallback synthesis was used because the selected model did not return valid JSON.",
        "- Review the cited source highlights before treating this as decision-grade output.",
    ]).strip()
    return report, (summary or f"Collected source material for {topic}.")


def fallback_local_deep_research_result(
    topic: str,
    *,
    plan: dict[str, Any],
    task_brief: str | None = None,
    document_evidence: list[dict[str, Any]] | None = None,
    prior_research: list[dict[str, Any]] | None = None,
    research_type: str = "general",
    refinement_contract: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from services.documents_svc import _dedupe_document_refs  # noqa: PLC0415

    local_sources = _build_local_context_sources(document_evidence, prior_research, max_sources=10)
    report_markdown, summary = _fallback_deep_research_markdown(topic, plan, local_sources, prior_research)
    if task_brief:
        report_markdown = "\n".join([
            report_markdown,
            "",
            "### Grounded Task Context",
            _excerpt_text(task_brief, 1600) or "No additional task brief was available.",
        ]).strip()
        summary = _excerpt_text(task_brief, 260) or summary
    result = {
        "title": f"Grounded Task Research: {topic}",
        "summary": summary,
        "content": report_markdown,
        "sources": local_sources,
        "mode": "deep",
        "research_type": research_type,
        "plan": plan,
        "refinement": refinement_contract,
        "source_research_ids": [item["id"] for item in (prior_research or []) if item.get("id")],
        "source_document_refs": _dedupe_document_refs(document_evidence or [], limit=DOCUMENT_RETRIEVAL_LIMIT),
    }
    meta = {
        "provider": None,
        "model": None,
        "warning": "No usable web sources were found; continued with grounded local context only.",
    }
    return result, meta


# ---------------------------------------------------------------------------
# Research session DB helpers
# ---------------------------------------------------------------------------


async def create_research_session(
    pool,
    workspace_id: int,
    topic: str,
    mode: str,
    research_type: str,
    *,
    title: str | None = None,
    status: str = "running",
    linked_todo_id: str | None = None,
    source_research_ids: list[int] | None = None,
    source_document_refs: list[dict[str, Any]] | None = None,
) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO research_sessions (
                    workspace_id, title, topic, mode, research_type, status, linked_todo_id, source_research_ids, source_document_refs
               )
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb)
               RETURNING id""",
            workspace_id,
            title or topic[:120],
            topic,
            mode,
            research_type or "general",
            status,
            linked_todo_id,
            json.dumps(source_research_ids or []),
            json.dumps(source_document_refs or []),
        )
    return row["id"]


async def update_research_session(
    pool,
    research_id: int,
    *,
    title: str | None = None,
    summary: str | None = None,
    content: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    status: str | None = None,
    error: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    refinement: dict[str, Any] | None = None,
    source_research_ids: list[int] | None = None,
    source_document_refs: list[dict[str, Any]] | None = None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE research_sessions
            SET title = COALESCE($1, title),
                summary = COALESCE($2, summary),
                content = COALESCE($3, content),
                sources = COALESCE($4::jsonb, sources),
                status = COALESCE($5, status),
                error = COALESCE($6, error),
                llm_provider = COALESCE($7, llm_provider),
                llm_model = COALESCE($8, llm_model),
                refinement = COALESCE($9::jsonb, refinement),
                source_research_ids = COALESCE($10::jsonb, source_research_ids),
                source_document_refs = COALESCE($11::jsonb, source_document_refs),
                updated_at = NOW()
            WHERE id = $12
            """,
            title,
            summary,
            content,
            json.dumps(sources) if sources is not None else None,
            status,
            error,
            llm_provider,
            llm_model,
            json.dumps(refinement) if refinement is not None else None,
            json.dumps(source_research_ids) if source_research_ids is not None else None,
            json.dumps(source_document_refs) if source_document_refs is not None else None,
            research_id,
        )

    if status == "completed" and content and content.strip():
        async def _vectorize_research_safe():
            try:
                await replace_research_chunks(pool, research_id)
            except Exception as exc:
                logger.warning("Research vectorization failed for %s: %s", research_id, exc)
        asyncio.create_task(_vectorize_research_safe())


async def replace_research_chunks(pool, research_id: int) -> None:
    """Chunk and embed a completed research session."""
    from services.documents_svc import _split_document_into_chunks  # noqa: PLC0415
    from services.embeddings import embed_texts  # noqa: PLC0415

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT workspace_id, content FROM research_sessions WHERE id = $1",
            research_id,
        )
    if not row or not row["content"] or not row["content"].strip():
        return
    workspace_id = row["workspace_id"]
    chunks = _split_document_into_chunks(row["content"])
    if not chunks:
        return

    texts = [str(chunk["content"]) for chunk in chunks]
    try:
        embeddings = embed_texts(texts)
    except Exception as exc:
        logger.warning("Embedding generation failed for research %s: %s", research_id, exc)
        embeddings = [None] * len(chunks)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM research_chunks WHERE research_id = $1", research_id)
            for chunk, emb in zip(chunks, embeddings):
                await conn.execute(
                    """
                    INSERT INTO research_chunks (
                        workspace_id, research_id, chunk_index, char_start, char_end, content, search_vector, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, to_tsvector('english', $6), $7)
                    """,
                    workspace_id,
                    research_id,
                    int(chunk["chunk_index"]),
                    int(chunk["char_start"]),
                    int(chunk["char_end"]),
                    str(chunk["content"]),
                    str(emb) if emb else None,
                )
    logger.info("Vectorized research %s: %d chunks", research_id, len(chunks))


def serialize_research_row(row: Any) -> dict[str, Any]:
    from services.text_svc import _json_list, _json_dict, _coerce_int_list  # noqa: PLC0415

    data = dict(row)
    for key in ("created_at", "updated_at"):
        if isinstance(data.get(key), datetime):
            data[key] = data[key].isoformat()
    data["sources"] = _json_list(data.get("sources"))
    data["source_research_ids"] = _coerce_int_list(data.get("source_research_ids"))
    data["source_document_refs"] = _json_list(data.get("source_document_refs"))
    if data.get("refinement") and isinstance(data["refinement"], str):
        data["refinement"] = _json_dict(data["refinement"])
    data["content_html"] = _render_markdown_html(data.get("content") or data.get("summary") or "")
    return data


# ---------------------------------------------------------------------------
# Refinement question generator
# ---------------------------------------------------------------------------


async def generate_research_refinement_questions(
    workspace_id: int,
    topic: str,
    mode: str,
    research_type: str,
    document_manifest: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inferred_query_type = _infer_research_query_type(topic)
    guidance_source = "local dr-refine guide" if load_dr_refine_guidance().strip() else "built-in dr-refine flow"
    topic_subject = (topic or "").strip().rstrip("?.! ") or topic
    query_prompts = {
        "LEARNING": {
            "confirmation": "This sounds like a learning question. Are you primarily trying to understand something new or explain it to others?",
            "reframe": f"What do you most need to understand about: {topic_subject}?",
            "subquestions": [
                "What are the core concepts or moving parts involved?",
                "What mechanisms or causal factors actually drive the outcome?",
                "What misconceptions or oversimplifications should be corrected?",
                "What concrete examples would make the topic easier to apply?",
            ],
            "success": "A useful learning brief makes the needed depth explicit, identifies the audience, and names what the explanation must clarify.",
        },
        "DECISION": {
            "confirmation": "This sounds like a decision question. Are you trying to choose between options or commit to a path?",
            "reframe": f"What decision should be made about: {topic_subject}?",
            "subquestions": [
                "What options or approaches should be compared directly?",
                "What are the tradeoffs, risks, and implementation costs for each option?",
                "What happens if no change is made right now?",
                "What evidence would make one option clearly better than the others?",
            ],
            "success": "A useful decision brief compares real alternatives, surfaces constraints, and makes it obvious what evidence would favor one path.",
        },
        "VALIDATION": {
            "confirmation": "This sounds like a validation question. Are you trying to test whether a belief or claim holds up under scrutiny?",
            "reframe": f"What evidence would confirm or disconfirm the current view about: {topic_subject}?",
            "subquestions": [
                "What is the strongest evidence supporting the current belief?",
                "What is the strongest evidence against it?",
                "Which assumptions are carrying the most weight?",
                "What would count as a decisive disconfirming signal?",
            ],
            "success": "A useful validation brief states the current belief, what would disconfirm it, and the cost of being wrong.",
        },
        "DUE_DILIGENCE": {
            "confirmation": "This sounds like a due diligence question. Are you trying to uncover hidden risks, blind spots, or failure modes before acting?",
            "reframe": f"What could go wrong or be overlooked in: {topic_subject}?",
            "subquestions": [
                "What failure modes or downside scenarios deserve the most attention?",
                "Which dependencies or assumptions could break the plan?",
                "Who is most exposed if the decision turns out poorly?",
                "What monitoring or mitigation would reduce downside risk?",
            ],
            "success": "A useful diligence brief names the worst cases, exposed stakeholders, and what risks are unacceptable.",
        },
        "EXPLORATION": {
            "confirmation": "This sounds exploratory. Are you trying to map the space, identify good directions, or discover what the real question should be?",
            "reframe": f"What are the most promising directions or frames for: {topic_subject}?",
            "subquestions": [
                "What are the main directions or approaches worth considering?",
                "What constraints or prerequisites shape the space?",
                "Which directions look promising but risky?",
                "What information would narrow the space fastest?",
            ],
            "success": "A useful exploration brief explains what a successful scan should produce and what directions or gaps matter most.",
        },
    }
    prompt_pack = query_prompts.get(inferred_query_type, query_prompts["EXPLORATION"])
    common = {
        "action_test": _refinement_question(
            "action_test", "real_question", "What you will do",
            "If you had a perfect answer right now, what would you do with it?",
            "Describe the decision, action, or outcome this research should enable.",
        ),
        "trigger_now": _refinement_question(
            "trigger_now", "real_question", "Why now",
            "What changed that made this important now?",
            "What deadline, event, or pressure made this question urgent?",
        ),
        "prior_gap": _refinement_question(
            "prior_gap", "real_question", "What almost answered it",
            "If you already looked into this, what did you find that almost answered the question but did not quite get there?",
            "Summarize what you already found and what was still missing.",
        ),
        "current_belief": _refinement_question(
            "current_belief", "beliefs", "Current belief",
            "What is your current belief or gut feeling about the answer?",
            "What do you think is probably true right now, and how confident are you?",
        ),
        "belief_source": _refinement_question(
            "belief_source", "beliefs", "Source of belief",
            "Where did that current belief come from?",
            "Past experience, a source you trust, a recent event, or something else.",
        ),
        "surprise_test": _refinement_question(
            "surprise_test", "beliefs", "What would surprise you",
            "What finding would genuinely surprise you if the research revealed it?",
            "Describe the result that would make you say you had the situation wrong.",
        ),
        "contrarian_probe": _refinement_question(
            "contrarian_probe", "beliefs", "Best argument against",
            "If someone smart disagreed with your current view, what would they say?",
            "State the strongest counterargument or opposing interpretation.",
        ),
        "premortem": _refinement_question(
            "premortem", "premortem", "Miss the mark",
            "Imagine the research is finished and feels useless. What would it have missed or gotten wrong?",
            "Describe the blind spots, wrong framing, or assumptions that would make the report fail.",
        ),
        "scope_boundaries": _refinement_question(
            "scope_boundaries", "scope", "Scope boundaries",
            "What should the research definitely include, and what should it avoid or only cover lightly?",
            "Focus areas, exclusions, or context that would keep the research on target.",
        ),
    }
    question_matrix = {
        "LEARNING": {
            "quick": [
                common["action_test"],
                _refinement_question("learning_depth", "stakes", "Depth needed",
                    "How deep do you need this to go: survey level, practitioner level, or expert level?",
                    "Describe the needed depth and any areas that must be included or avoided."),
                _refinement_question("audience", "stakes", "Who this is for",
                    "Is this mainly for your own understanding or to explain to someone else?",
                    "Say who will use the result and what they need to be able to understand afterward."),
                common["scope_boundaries"],
            ],
            "deep": [
                common["action_test"], common["surprise_test"],
                _refinement_question("learning_depth", "stakes", "Depth needed",
                    "How deep do you need this to go: survey level, practitioner level, or expert level?",
                    "Describe the needed depth and any areas that must be included or avoided."),
                _refinement_question("audience", "stakes", "Who this is for",
                    "Is this mainly for your own understanding or to explain to someone else?",
                    "Say who will use the result and what they need to be able to understand afterward."),
                common["prior_gap"], common["scope_boundaries"],
            ],
        },
        "DECISION": {
            "quick": [
                common["action_test"], common["trigger_now"],
                _refinement_question("alternatives", "stakes", "Alternatives on the table",
                    "What options are you actually comparing?",
                    "List the real alternatives or approaches that should be compared."),
                _refinement_question("constraints", "stakes", "Constraints",
                    "What constraints are non-negotiable?",
                    "Budget, timing, staffing, compliance, technical limits, or other hard constraints."),
                common["premortem"],
            ],
            "deep": [
                common["action_test"], common["trigger_now"], common["current_belief"],
                common["contrarian_probe"], common["premortem"],
                _refinement_question("alternatives", "stakes", "Alternatives on the table",
                    "What options are you actually comparing?",
                    "List the real alternatives or approaches that should be compared."),
                _refinement_question("default_path", "stakes", "If nothing changes",
                    "What is the default if you do nothing?",
                    "Describe the status quo or default path if no decision is made."),
                _refinement_question("stakeholders", "stakes", "Who has to buy in",
                    "Who else needs to be convinced, and what would change their mind?",
                    "List stakeholders and the evidence or framing they will care about."),
                _refinement_question("constraints", "stakes", "Constraints",
                    "What constraints are non-negotiable?",
                    "Budget, timing, staffing, compliance, technical limits, or other hard constraints."),
            ],
        },
        "VALIDATION": {
            "quick": [
                common["action_test"],
                _refinement_question("doubt_source", "real_question", "What introduced doubt",
                    "What is making you doubt the current view now?",
                    "Describe the event, signal, or contradiction that triggered this check."),
                common["current_belief"],
                _refinement_question("what_changes_mind", "stakes", "What would change your mind",
                    "What evidence would actually change your mind either way?",
                    "Describe the kind of proof, data, or source that would move you."),
                _refinement_question("cost_of_wrong", "stakes", "Cost of being wrong",
                    "What is the cost of being wrong in either direction?",
                    "Describe the downside if the belief is false, and the downside if you reject something true."),
            ],
            "deep": [
                common["action_test"],
                _refinement_question("doubt_source", "real_question", "What introduced doubt",
                    "What is making you doubt the current view now?",
                    "Describe the event, signal, or contradiction that triggered this check."),
                common["current_belief"], common["belief_source"], common["contrarian_probe"], common["premortem"],
                _refinement_question("what_changes_mind", "stakes", "What would change your mind",
                    "What evidence would actually change your mind either way?",
                    "Describe the kind of proof, data, or source that would move you."),
                _refinement_question("cost_of_wrong", "stakes", "Cost of being wrong",
                    "What is the cost of being wrong in either direction?",
                    "Describe the downside if the belief is false, and the downside if you reject something true."),
            ],
        },
        "EXPLORATION": {
            "quick": [
                common["action_test"],
                _refinement_question("exploration_success", "stakes", "What success looks like",
                    "What should a successful exploration produce for you?",
                    "A map of options, key questions, promising directions, decision criteria, or something else."),
                common["scope_boundaries"], common["premortem"],
            ],
            "deep": [
                common["action_test"], common["prior_gap"], common["surprise_test"],
                _refinement_question("exploration_success", "stakes", "What success looks like",
                    "What should a successful exploration produce for you?",
                    "A map of options, key questions, promising directions, decision criteria, or something else."),
                common["scope_boundaries"], common["premortem"],
            ],
        },
        "DUE_DILIGENCE": {
            "quick": [
                common["action_test"], common["trigger_now"],
                _refinement_question("worst_case", "stakes", "Worst-case scenario",
                    "What worst-case scenario do you need this research to surface clearly?",
                    "Describe the failure mode or downside you are most worried about."),
                _refinement_question("acceptable_risk", "stakes", "Risk threshold",
                    "What risk level is acceptable versus unacceptable?",
                    "Explain what downside is tolerable and what would be a deal-breaker."),
                common["premortem"],
            ],
            "deep": [
                common["action_test"], common["trigger_now"], common["current_belief"],
                common["contrarian_probe"], common["premortem"],
                _refinement_question("worst_case", "stakes", "Worst-case scenario",
                    "What worst-case scenario do you need this research to surface clearly?",
                    "Describe the failure mode or downside you are most worried about."),
                _refinement_question("acceptable_risk", "stakes", "Risk threshold",
                    "What risk level is acceptable versus unacceptable?",
                    "Explain what downside is tolerable and what would be a deal-breaker."),
                _refinement_question("stakeholders", "stakes", "Who gets hurt",
                    "Who are the stakeholders most affected if this goes badly?",
                    "List the people, teams, or customers with the most downside exposure."),
                _refinement_question("cost_of_not_knowing", "stakes", "Cost of missing something",
                    "What is the cost of not knowing something important before acting?",
                    "Describe what happens if a critical risk is missed."),
            ],
        },
    }
    query_mode = "quick" if (mode or "").lower() == "quick" else "deep"
    questions = list(question_matrix.get(inferred_query_type, question_matrix["EXPLORATION"])[query_mode])

    if document_manifest:
        doc_names = ", ".join(
            item.get("title") or item.get("filename") or f"item {item.get('id', '?')}"
            for item in document_manifest[:8]
        )
        suffix = f" (and {len(document_manifest) - 8} more)" if len(document_manifest) > 8 else ""
        questions.insert(0, _refinement_question(
            "document_focus", "scope", "Document focus",
            f"You selected: {doc_names}{suffix}. What specifically should the research look for in these sources?",
            "Key topics, claims to verify, comparisons across documents, specific sections or data points to extract.",
        ))

    result = {
        "inferred_query_type": inferred_query_type,
        "query_type_confirmation_prompt": prompt_pack["confirmation"],
        "suggested_reframe": prompt_pack["reframe"],
        "questions": questions,
        "suggested_subquestions": prompt_pack["subquestions"],
        "success_criteria_hint": prompt_pack["success"],
        "guidance_source": guidance_source,
        "has_document_sources": bool(document_manifest),
    }
    return result, {"provider": None, "model": None}


async def build_research_refinement_contract(
    workspace_id: int,
    topic: str,
    research_type: str,
    refinement: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = _normalize_research_refinement(refinement)
    if not normalized:
        raise RuntimeError("Deep research refinement answers are required.")
    query_type = normalized.get("query_type") or _infer_research_query_type(topic)
    action = _get_refinement_answer(normalized, "action_test")
    trigger = _get_refinement_answer(normalized, "trigger_now", "doubt_source", "prior_gap", "exploration_success")
    belief = _get_refinement_answer(normalized, "current_belief")
    belief_source = _get_refinement_answer(normalized, "belief_source") or trigger
    contrarian = _get_refinement_answer(normalized, "contrarian_probe")
    premortem = _get_refinement_answer(normalized, "premortem")
    scope = _get_refinement_answer(normalized, "scope_boundaries")
    document_focus = _get_refinement_answer(normalized, "document_focus")
    alternatives = _get_refinement_answer(normalized, "alternatives", "default_path")
    stakeholders = _get_refinement_answer(normalized, "stakeholders")
    constraints_text = _get_refinement_answer(
        normalized, "constraints", "learning_depth", "acceptable_risk", "cost_of_wrong", "cost_of_not_knowing",
    )
    what_changes_mind = _get_refinement_answer(normalized, "what_changes_mind", "surprise_test", "contrarian_probe")
    audience = _get_refinement_answer(normalized, "audience")
    worst_case = _get_refinement_answer(normalized, "worst_case")
    suggested_reframe = normalized.get("suggested_reframe") or ""
    refined_question = suggested_reframe or topic
    focus_areas = _dedupe_preserve_order(
        _split_freeform_list(document_focus, 3) + (normalized.get("suggested_subquestions") or []) + _split_freeform_list(audience, 2),
        6,
    ) or [topic]
    exclusions = _split_freeform_list(scope, 4)
    constraints = _dedupe_preserve_order(
        _split_freeform_list(constraints_text, 4) + _split_freeform_list(stakeholders, 3),
        6,
    )
    decision_context = _dedupe_preserve_order(_split_freeform_list(alternatives, 4), 4)
    key_uncertainties = _dedupe_preserve_order(
        _split_freeform_list(trigger, 2)
        + _split_freeform_list(belief, 2)
        + _split_freeform_list(contrarian, 2)
        + _split_freeform_list(what_changes_mind, 2)
        + _split_freeform_list(worst_case, 2),
        6,
    )
    premortem_insights = _dedupe_preserve_order(
        _split_freeform_list(premortem, 4) + _split_freeform_list(worst_case, 2),
        5,
    )
    success_criteria = _dedupe_preserve_order(_split_freeform_list(normalized.get("success_criteria_hint") or "", 3), 5)
    if action:
        success_criteria.insert(0, f"Enable the user to act on: {action}")
    if what_changes_mind:
        success_criteria.append(f"Address what would change their mind: {what_changes_mind[:220]}")
    if audience:
        success_criteria.append(f"Be useful for this audience: {audience[:220]}")
    success_criteria = _dedupe_preserve_order(success_criteria, 5)
    research_contract = (
        f"Investigate '{refined_question}' as a {query_type.replace('_', ' ').title()} question. "
        f"The research should help the user {action or 'make a well-grounded decision'}. "
        f"Focus on {', '.join(focus_areas[:3]) or topic}. "
        f"Respect these constraints: {constraints_text or 'No additional constraints provided'}. "
        f"Avoid missing the mark by addressing: {premortem or 'the main decision risks and framing assumptions'}."
    )
    brief_markdown = "\n".join([
        "### Research Question",
        refined_question,
        "",
        "### Query Type",
        query_type.replace("_", " "),
        "",
        "### What They'll Do With It",
        action or "Not specified",
        "",
        "### Current Belief",
        belief or "Not specified",
        "",
        "### Source Of Belief",
        belief_source or "Not specified",
        "",
        "### Key Uncertainties",
        *([f"- {item}" for item in key_uncertainties] or ["- Not specified"]),
        "",
        "### What Would Change Their Mind",
        what_changes_mind or "Not specified",
        "",
        "### Pre-Mortem Insights",
        *([f"- {item}" for item in premortem_insights] or ["- Not specified"]),
        "",
        "### Stakeholders / Decision Context",
        *([f"- {item}" for item in _dedupe_preserve_order(decision_context + _split_freeform_list(stakeholders, 4), 6)] or ["- Not specified"]),
        "",
        "### Document Focus",
        document_focus or "Not specified",
        "",
        "### Scope Boundaries",
        *([f"- {item}" for item in exclusions] or ["- Not specified"]),
        "",
        "### Suggested Sub-Questions",
        *([f"- {item}" for item in focus_areas] or ["- Not specified"]),
        "",
        "### Success Criteria",
        *([f"- {item}" for item in success_criteria] or ["- Not specified"]),
    ]).strip()
    contract = {
        "refined_question": refined_question,
        "query_type": query_type,
        "what_they_will_do": action or "",
        "current_belief": belief or "",
        "confidence_level": "",
        "belief_source": belief_source or "",
        "key_uncertainties": key_uncertainties,
        "what_would_change_their_mind": what_changes_mind or "",
        "document_focus": document_focus or "",
        "premortem_insights": premortem_insights,
        "subquestions": focus_areas,
        "scope_boundaries": {
            "focus_areas": focus_areas[:5],
            "exclusions": exclusions,
            "constraints": constraints,
            "stakeholders": _split_freeform_list(stakeholders, 4),
            "decision_context": decision_context,
        },
        "success_criteria": success_criteria,
        "research_contract": research_contract,
        "brief_markdown": brief_markdown,
    }
    return contract, {"provider": None, "model": None}


# ---------------------------------------------------------------------------
# Quick / deep research orchestration
# ---------------------------------------------------------------------------


async def run_quick_research(
    pool,
    workspace_id: int,
    topic: str,
    research_type: str,
    refinement_contract: dict[str, Any] | None = None,
    *,
    prior_research: list[dict[str, Any]] | None = None,
    task_brief: str | None = None,
    document_evidence: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from services.llm_prefs import get_workspace_llm_preferences, _resolve_task_llm  # noqa: PLC0415
    from services.documents_svc import _dedupe_document_refs, document_evidence_prompt_block  # noqa: PLC0415
    from services.web_svc import collect_research_sources, fetch_web_source, extract_urls_from_text  # noqa: PLC0415
    from services.text_svc import _json_dict  # noqa: PLC0415
    from llm import call_llm_runner_json  # noqa: PLC0415

    preferences = await get_workspace_llm_preferences(pool, workspace_id)
    provider, model = _resolve_task_llm(preferences, "research")
    guidance = load_quick_research_guidance()
    effective_topic = (refinement_contract or {}).get("refined_question") or topic
    refinement_block = ""
    if refinement_contract:
        refinement_block = f"\nRefinement brief:\n{refinement_contract.get('brief_markdown') or json.dumps(refinement_contract, indent=2)}\n"
    prior_research = prior_research or []
    prior_block = ""
    if prior_research:
        prior_lines = []
        for item in prior_research[:6]:
            prior_lines.append(
                f'- [{item.get("id")}] {item.get("title") or item.get("topic")}: {item.get("summary") or ""}'
            )
        prior_block = "Existing relevant research already completed:\n" + "\n".join(prior_lines) + "\n"
    task_block = f"\nDeliverable task:\n{task_brief}\n" if task_brief else ""
    document_block = ""
    if document_evidence:
        document_block = "\n" + document_evidence_prompt_block(document_evidence) + "\n"

    plan_prompt = f"""\
You are executing the quick research skill.

Methodology:
{guidance}

Topic: {effective_topic}
Research type: {research_type or "general"}
{refinement_block}
{task_block}{prior_block}{document_block}
Before proposing new search queries, summarize what existing research already covers and focus new search effort on gaps, verification, or changed facts.

Return ONLY valid JSON with:
- "complexity": "SIMPLE" | "MODERATE" | "COMPLEX"
- "prior_coverage": string
- "queries": array of 2-5 orthogonal search queries
- "focus_points": array of 3-6 points to validate
"""
    try:
        if progress_callback:
            await progress_callback("Building search plan...")
        plan, plan_meta = await call_llm_runner_json(
            [{"role": "user", "content": plan_prompt}],
            provider=provider,
            model=model,
            use_case="chat",
            max_tokens=1200,
        )
        plan = _json_dict(plan)
    except Exception:
        plan = {
            "complexity": "MODERATE" if refinement_contract else "SIMPLE",
            "prior_coverage": _fallback_prior_coverage(prior_research),
            "queries": _fallback_research_queries(effective_topic, research_type, refinement_contract, max_queries=5),
            "focus_points": _dedupe_preserve_order(
                ((refinement_contract or {}).get("subquestions") or []) + [effective_topic, f"{effective_topic} best practices"],
                5,
            ),
        }
        plan_meta = {
            "provider": provider,
            "model": model,
            "warning": "Search planning fell back to a deterministic plan because the selected model did not return valid JSON.",
        }
    queries = [str(q).strip() for q in (plan.get("queries") or []) if str(q).strip()][:5]
    if not queries:
        queries = [topic]

    explicit_urls = extract_urls_from_text(f"{task_brief or ''} {effective_topic}")
    explicit_sources: list[dict[str, Any]] = []
    if explicit_urls:
        if progress_callback:
            await progress_callback(f"Fetching {len(explicit_urls)} explicitly provided source(s)...")
        url_items = [
            {"url": u, "title": u, "domain": urlparse(u).netloc.lower(), "query": effective_topic}
            for u in explicit_urls[:6]
        ]
        fetched_explicit = await asyncio.gather(*[fetch_web_source(item) for item in url_items], return_exceptions=True)
        for item in fetched_explicit:
            if isinstance(item, dict):
                explicit_sources.append(item)
        logger.info("QR explicit URL fetch: %d/%d succeeded", len(explicit_sources), len(url_items))

    remaining_slots = max(0, 6 - len(explicit_sources))
    if remaining_slots > 0:
        if progress_callback:
            await progress_callback("Searching the web for sources...")
        search_sources = await collect_research_sources(queries, per_query_results=4, max_sources=remaining_slots)
    else:
        search_sources = []

    sources = explicit_sources + search_sources
    for idx, src in enumerate(sources, start=1):
        src["id"] = f"S{idx}"

    if not sources:
        raise RuntimeError(
            "Web search returned no usable sources. "
            "Try including a specific URL in your query, or paste the source text directly into the chat."
        )

    synth_prompt = f"""\
You are finishing the quick research skill.

Topic: {effective_topic}
Research type: {research_type or "general"}
Planned focus points: {json.dumps(plan.get("focus_points") or [])}
{refinement_block}
Prior research coverage: {plan.get("prior_coverage") or ""}
{task_block}{prior_block}{document_block}

Allowed sources:
{sources_prompt_block(sources)}

Return ONLY valid JSON with:
- "title": string
- "key_finding": string
- "details": array of objects with keys "claim" and "source_ids" (source ids must come from the allowed list above)
- "context": string
- "caveats": string
- "confidence": "HIGH" | "MEDIUM" | "LOW"
- "confidence_rationale": string
"""
    if progress_callback:
        await progress_callback("Synthesizing findings from collected sources...")
    try:
        synthesis, synth_meta = await call_llm_runner_json(
            [{"role": "user", "content": synth_prompt}],
            provider=provider,
            model=model,
            use_case="chat",
            max_tokens=2200,
        )
        synthesis = _json_dict(synthesis)
    except Exception:
        synthesis = _fallback_quick_synthesis_payload(effective_topic, research_type, sources, prior_research)
        synth_meta = {
            "provider": provider,
            "model": model,
            "warning": "Final synthesis fell back to a deterministic summary because the selected model did not return valid JSON.",
            "json_repair_used": False,
        }
    markdown = render_quick_research_markdown(synthesis, sources, prior_research)
    result = {
        "title": synthesis.get("title") or effective_topic,
        "summary": synthesis.get("key_finding") or "",
        "content": markdown,
        "sources": sources,
        "mode": "quick",
        "research_type": research_type,
        "refinement": refinement_contract,
        "source_research_ids": [item["id"] for item in prior_research if item.get("id")],
        "source_document_refs": _dedupe_document_refs(document_evidence or []),
    }
    meta = {
        "provider": synth_meta.get("provider") or plan_meta.get("provider"),
        "model": synth_meta.get("model") or plan_meta.get("model"),
    }
    warnings = []
    if plan_meta.get("warning"):
        warnings.append(plan_meta["warning"])
    if plan_meta.get("json_repair_used"):
        warnings.append("Search planning needed a JSON repair pass.")
    if synth_meta.get("json_repair_used"):
        warnings.append("Final synthesis needed a JSON repair pass.")
    if warnings:
        meta["warning"] = " ".join(warnings)
    return result, meta


async def run_deep_research(
    pool,
    workspace_id: int,
    topic: str,
    research_type: str,
    refinement_contract: dict[str, Any] | None = None,
    *,
    prior_research: list[dict[str, Any]] | None = None,
    task_brief: str | None = None,
    document_evidence: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from services.llm_prefs import get_workspace_llm_preferences, _resolve_task_llm  # noqa: PLC0415
    from services.documents_svc import _dedupe_document_refs, document_evidence_prompt_block  # noqa: PLC0415
    from services.web_svc import collect_research_sources  # noqa: PLC0415
    from services.text_svc import _json_dict  # noqa: PLC0415
    from llm import call_llm_runner_json  # noqa: PLC0415

    preferences = await get_workspace_llm_preferences(pool, workspace_id)
    provider, model = _resolve_task_llm(preferences, "research")
    guidance = load_deep_research_guidance(research_type)
    effective_topic = (refinement_contract or {}).get("refined_question") or topic
    refinement_block = ""
    if refinement_contract:
        refinement_block = f"\nRefinement brief:\n{refinement_contract.get('brief_markdown') or json.dumps(refinement_contract, indent=2)}\n"
    prior_research = prior_research or []
    prior_block = ""
    if prior_research:
        prior_lines = []
        for item in prior_research[:8]:
            prior_lines.append(
                f'- [{item.get("id")}] {item.get("title") or item.get("topic")}: {item.get("summary") or ""}'
            )
        prior_block = "Existing relevant research already completed:\n" + "\n".join(prior_lines) + "\n"
    task_block = f"\nDeliverable task:\n{task_brief}\n" if task_brief else ""
    document_block = ""
    if document_evidence:
        document_block = "\n" + document_evidence_prompt_block(document_evidence) + "\n"

    plan_prompt = f"""\
You are executing the deep research skill.

Methodology:
{guidance}

Topic: {effective_topic}
Research type: {research_type or "general"}
{refinement_block}
{task_block}{prior_block}{document_block}
Before planning new queries, account for the existing research, summarize what it already covers, and focus the new plan on gaps, contradictions, unresolved questions, or changed facts.

Return ONLY valid JSON with:
- "classification": string
- "research_contract": string
- "prior_coverage": string
- "perspectives": array of strings
- "subquestions": array of 4-7 strings
- "queries": array of 6-12 search queries
- "report_outline": array of section titles
"""
    try:
        if progress_callback:
            await progress_callback("Building deep research plan...")
        plan, plan_meta = await call_llm_runner_json(
            [{"role": "user", "content": plan_prompt}],
            provider=provider,
            model=model,
            use_case="chat",
            max_tokens=2400,
        )
        plan = _json_dict(plan)
    except Exception:
        plan = {
            "classification": "Exploratory research",
            "research_contract": f"Investigate how to approach {effective_topic} and what evidence, examples, and risks should shape the deliverable.",
            "prior_coverage": _fallback_prior_coverage(prior_research),
            "perspectives": _dedupe_preserve_order(["Implementation", "Stakeholder expectations", "Risks"], 4),
            "subquestions": _dedupe_preserve_order(((refinement_contract or {}).get("subquestions") or []) + [effective_topic], 6),
            "queries": _fallback_research_queries(effective_topic, research_type, refinement_contract, max_queries=10),
            "report_outline": ["Overview", "What existing research already covers", "What changed or still matters", "Recommendations"],
        }
        plan_meta = {
            "provider": provider,
            "model": model,
            "warning": "Deep-research planning fell back to a deterministic plan because the selected model did not return valid JSON.",
        }
    queries = [str(q).strip() for q in (plan.get("queries") or []) if str(q).strip()][:12]
    if not queries:
        queries = [topic]

    if progress_callback:
        await progress_callback("Searching the web for supporting sources...")
    sources = await collect_research_sources(queries, per_query_results=4, max_sources=10)
    if not sources:
        local_result, local_meta = fallback_local_deep_research_result(
            effective_topic,
            plan=plan,
            task_brief=task_brief,
            document_evidence=document_evidence,
            prior_research=prior_research,
            research_type=research_type,
            refinement_contract=refinement_contract,
        )
        if progress_callback:
            await progress_callback("No usable web sources were found; grounding the deep report from local context only...")
        return local_result, local_meta

    synth_prompt = f"""\
You are finishing the deep research skill.

Topic: {effective_topic}
Research type: {research_type or "general"}
Classification: {plan.get("classification") or ""}
Research contract:
{plan.get("research_contract") or ""}
{refinement_block}
Prior research coverage:
{plan.get("prior_coverage") or ""}
{task_block}{prior_block}{document_block}

Perspectives: {json.dumps(plan.get("perspectives") or [])}
Subquestions: {json.dumps(plan.get("subquestions") or [])}
Outline: {json.dumps(plan.get("report_outline") or [])}

Allowed sources:
{sources_prompt_block(sources)}

Write a decision-grade markdown report grounded ONLY in the allowed sources.
Inline citations must use the source ids exactly as [S1], [S2], etc.

Return ONLY valid JSON with:
- "title": string
- "summary": string
- "report_markdown": string
"""
    if progress_callback:
        await progress_callback("Drafting the research report...")
    try:
        synthesis, synth_meta = await call_llm_runner_json(
            [{"role": "user", "content": synth_prompt}],
            provider=provider,
            model=model,
            use_case="chat",
            max_tokens=5200,
        )
        synthesis = _json_dict(synthesis)
    except Exception:
        report_markdown, summary = _fallback_deep_research_markdown(effective_topic, plan, sources, prior_research)
        synthesis = {
            "title": f"Deep Research: {effective_topic}",
            "summary": summary,
            "report_markdown": report_markdown,
        }
        synth_meta = {
            "provider": provider,
            "model": model,
            "warning": "Final report fell back to a deterministic source digest because the selected model did not return valid JSON.",
            "json_repair_used": False,
        }
    result = {
        "title": synthesis.get("title") or effective_topic,
        "summary": synthesis.get("summary") or "",
        "content": synthesis.get("report_markdown") or "",
        "sources": sources,
        "mode": "deep",
        "research_type": research_type,
        "plan": plan,
        "refinement": refinement_contract,
        "source_research_ids": [item["id"] for item in prior_research if item.get("id")],
        "source_document_refs": _dedupe_document_refs(document_evidence or [], limit=DOCUMENT_RETRIEVAL_LIMIT),
    }
    meta = {
        "provider": synth_meta.get("provider") or plan_meta.get("provider"),
        "model": synth_meta.get("model") or plan_meta.get("model"),
    }
    warnings = []
    if plan_meta.get("warning"):
        warnings.append(plan_meta["warning"])
    if plan_meta.get("json_repair_used"):
        warnings.append("Search planning needed a JSON repair pass.")
    if synth_meta.get("json_repair_used"):
        warnings.append("Final synthesis needed a JSON repair pass.")
    if warnings:
        meta["warning"] = " ".join(warnings)
    return result, meta
