"""Pydantic request/response models for the Meeting Analyzer API."""
from typing import Any

from pydantic import BaseModel, Field

class WorkspaceCreate(BaseModel):
    name: str
    folder_id: int | None = None


class FolderCreate(BaseModel):
    name: str
    parent_folder_id: int | None = None


class FolderUpdate(BaseModel):
    name: str | None = None
    parent_folder_id: int | None = Field(default=0)


class WorkspaceUpdate(BaseModel):
    name: str | None = None


class WorkspaceFolderUpdate(BaseModel):
    folder_id: int | None = None


class LLMTaskPreference(BaseModel):
    provider: str | None = None
    model: str | None = None


class WorkspaceLLMPreferences(BaseModel):
    analysis: LLMTaskPreference = Field(default_factory=LLMTaskPreference)
    chat: LLMTaskPreference = Field(default_factory=LLMTaskPreference)
    research: LLMTaskPreference = Field(default_factory=LLMTaskPreference)
    generate: LLMTaskPreference = Field(default_factory=LLMTaskPreference)


class ChatRequest(BaseModel):
    workspace_id: int | None = None
    chat_session_id: int | None = None
    meeting_ids: list[int] = []
    include_transcripts: list[int] = []
    include_document_ids: list[int] = []
    include_research_ids: list[int] = []
    question: str


class ChatTurnProxyRequest(BaseModel):
    message: str | None = None
    messages: list[dict[str, Any]] | None = None
    provider: str | None = None
    model: str | None = None
    tools: list[Any] | None = None
    system: str | None = None
    max_tokens: int = 4096
    workspace_id: int | None = None
    chat_session_id: int | None = None
    meeting_ids: list[int] = []
    include_transcripts: list[int] = []
    include_document_ids: list[int] = []
    include_research_ids: list[int] = []
    attachment_ids: list[dict[str, Any]] = []


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None


class ChatSessionUpdateRequest(BaseModel):
    title: str | None = None
    context_meeting_ids: list[int] | None = None
    context_document_ids: list[int] | None = None
    context_research_ids: list[int] | None = None


class ChatRenderRequest(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str


class AnalyzeTextRequest(BaseModel):
    text: str
    workspace_id: int | None = None


class ResearchRequest(BaseModel):
    topic: str
    mode: str = "quick"
    research_type: str = "general"
    refinement: dict[str, Any] | None = None
    document_ids: list[int] = []
    meeting_ids: list[int] = []
    research_ids: list[int] = []


class ResearchRefineRequest(BaseModel):
    topic: str
    mode: str = "deep"
    research_type: str = "general"
    document_ids: list[int] = []
    meeting_ids: list[int] = []


class ApplyLLMDefaultsRequest(BaseModel):
    scope: str = "current"
    workspace_id: int | None = None


class MeetingWorkspaceUpdate(BaseModel):
    workspace_id: int | None = None


class MeetingMergeRequest(BaseModel):
    meeting_ids: list[int]
    workspace_id: int | None = None
    delete_originals: bool = True


class WorkspaceShareRequest(BaseModel):
    email: str | None = None
    user_id: str | None = None
    name: str | None = None


class TodoUpdateRequest(BaseModel):
    due_date: str | None = None
    assignee: str | None = None
    status: str | None = None


class WorkspaceTodoCreateRequest(BaseModel):
    task: str
    assignee: str | None = None
    due_date: str | None = None
    notes: str | None = None
    status: str = "incomplete"


class WorkspaceTodoUpdateRequest(BaseModel):
    task: str | None = None
    assignee: str | None = None
    due_date: str | None = None
    notes: str | None = None
    status: str | None = None


class GenerateTaskCreateRequest(BaseModel):
    todo_id: str
    artifact_template: str = "requirements"
    output_type: str | None = None
    reset_setup: bool = False


class GenerateTaskUpdateRequest(BaseModel):
    title: str | None = None
    artifact_template: str | None = None
    output_type: str | None = None
    website_url: str | None = None
    branding: dict[str, Any] | None = None
    selected_meeting_ids: list[int] | None = None
    selected_document_ids: list[int] | None = None
    selected_research_ids: list[int] | None = None
    selected_todo_ids: list[str] | None = None
    selected_todo_people: list[str] | None = None
    related_research_ids: list[int] | None = None
    question_plan: list[dict[str, Any]] | None = None
    grounding_pack: dict[str, Any] | None = None
    answers: dict[str, Any] | None = None
    answer_evidence: dict[str, Any] | None = None
    answer_meta: dict[str, Any] | None = None
    question_research: dict[str, Any] | None = None
    template_draft: dict[str, Any] | None = None
    prompt: str | None = None
    current_step: str | None = None
    linked_research_id: int | None = None
    active_run_id: int | None = None
    latest_document_id: int | None = None
    status: str | None = None
    is_stale: bool | None = None
    stale_flags: list[str] | None = None


class GenerateTaskResearchRequest(BaseModel):
    topic: str | None = None
    mode: str = "quick"
    research_type: str = "general"
    refinement: dict[str, Any] | None = None


class GenerateTaskAutofillRequest(BaseModel):
    overwrite: bool = False
    overwrite_manual: bool = False
    question_keys: list[str] | None = None
    guidance: str | None = None


class AddCustomSectionRequest(BaseModel):
    heading: str
    content: str = ""


class GenerateTaskQuestionResearchRequest(BaseModel):
    question_key: str
    guidance: str | None = None


class QuestionChatRequest(BaseModel):
    question_key: str
    message: str


class BrandRefreshRequest(BaseModel):
    website_url: str


class DeliverableRequest(BaseModel):
    output_type: str | None = None


class GenerateRequest(BaseModel):
    title: str | None = None
    prompt: str
    output_type: str = "document"
    meeting_ids: list[int] = []
    include_transcripts: list[int] = []
    include_document_ids: list[int] = []
    include_research_ids: list[int] = []
    include_todo_ids: list[str] = []
    include_todo_people: list[str] = []


class LiveQARequest(BaseModel):
    question: str
    transcript_context: str | None = None
    meeting_ids: list[int] = []
    document_ids: list[int] = []
    research_ids: list[int] = []
    session_id: int | None = None
    auto_lookup: bool = False
    confidence_threshold: float = 0.7


class CalendarEventCreateRequest(BaseModel):
    title: str
    all_day: bool = False
    is_due_only: bool = False
    start_time: str
    end_time: str
    notes: str | None = None
    color: str | None = None


class CalendarEventUpdateRequest(BaseModel):
    title: str | None = None
    all_day: bool | None = None
    is_due_only: bool | None = None
    start_time: str | None = None
    end_time: str | None = None
    notes: str | None = None
    color: str | None = None
