"""Database initialization and schema management for the Meeting Analyzer."""
import logging

import asyncpg

logger = logging.getLogger("meeting-analyzer")

async def init_db(database_url: str) -> asyncpg.Pool:
    """Create connection pool and run schema migrations. Returns the pool."""
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                filename TEXT,
                date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                transcript TEXT,
                summary TEXT,
                action_items JSONB DEFAULT '[]'::jsonb,
                email_body TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            ALTER TABLE workspaces
            ADD COLUMN IF NOT EXISTS llm_preferences JSONB NOT NULL DEFAULT '{}'::jsonb
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workspace_folders (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                parent_folder_id INTEGER REFERENCES workspace_folders(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            ALTER TABLE workspaces
            ADD COLUMN IF NOT EXISTS folder_id INTEGER
        """)
        # Fix stale FK if it points to wrong table
        existing_fk = await conn.fetchval("""
            SELECT pg_get_constraintdef(oid) FROM pg_constraint
            WHERE conrelid = 'workspaces'::regclass AND conname = 'workspaces_folder_id_fkey'
        """)
        if existing_fk and 'workspace_folders' not in existing_fk:
            await conn.execute("ALTER TABLE workspaces DROP CONSTRAINT workspaces_folder_id_fkey")
            await conn.execute("""
                ALTER TABLE workspaces
                ADD CONSTRAINT workspaces_folder_id_fkey FOREIGN KEY (folder_id)
                REFERENCES workspace_folders(id) ON DELETE SET NULL
            """)
        elif not existing_fk:
            await conn.execute("""
                ALTER TABLE workspaces
                ADD CONSTRAINT workspaces_folder_id_fkey FOREIGN KEY (folder_id)
                REFERENCES workspace_folders(id) ON DELETE SET NULL
            """)
        # Recreate users table with TEXT id if it has wrong schema
        users_id_type = await conn.fetchval(
            "SELECT data_type FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'id'"
        )
        if users_id_type and users_id_type != 'text':
            await conn.execute("DROP TABLE IF EXISTS users CASCADE")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT,
                name TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                last_login_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ DEFAULT NOW()")
        await conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS user_id TEXT")
        await conn.execute("ALTER TABLE workspace_folders ADD COLUMN IF NOT EXISTS user_id TEXT")
        await conn.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS user_id TEXT")
        for _tbl in ('workspaces', 'meetings'):
            _col_type = await conn.fetchval(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = $1 AND column_name = 'user_id'", _tbl
            )
            if _col_type and _col_type != 'text':
                await conn.execute(f"ALTER TABLE {_tbl} ALTER COLUMN user_id TYPE TEXT USING user_id::TEXT")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_workspaces_user_id ON workspaces(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_folders_user_id ON workspace_folders(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_meetings_user_id ON meetings(user_id)")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workspace_shares (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(workspace_id, user_id)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            ALTER TABLE meetings
            ADD COLUMN IF NOT EXISTS workspace_id INTEGER REFERENCES workspaces(id) ON DELETE SET NULL
        """)
        await conn.execute("""
            ALTER TABLE meetings
            ADD COLUMN IF NOT EXISTS todos JSONB NOT NULL DEFAULT '[]'::jsonb
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workspace_todos (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                task TEXT NOT NULL,
                assignee TEXT,
                due_date DATE,
                notes TEXT,
                status TEXT NOT NULL DEFAULT 'incomplete',
                source_type TEXT NOT NULL DEFAULT 'manual',
                source_meeting_id INTEGER REFERENCES meetings(id) ON DELETE SET NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            ALTER TABLE workspace_todos
            ADD COLUMN IF NOT EXISTS notes TEXT
        """)
        await conn.execute("""
            ALTER TABLE workspace_todos
            ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'incomplete'
        """)
        await conn.execute("""
            ALTER TABLE workspace_todos
            ALTER COLUMN status SET DEFAULT 'incomplete'
        """)
        await conn.execute("""
            ALTER TABLE workspace_todos
            ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'manual'
        """)
        await conn.execute("""
            ALTER TABLE workspace_todos
            ADD COLUMN IF NOT EXISTS source_meeting_id INTEGER REFERENCES meetings(id) ON DELETE SET NULL
        """)
        await conn.execute("""
            ALTER TABLE workspace_todos
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        """)
        await conn.execute("""
            ALTER TABLE workspace_todos
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workspace_todos_workspace
            ON workspace_todos(workspace_id, due_date, updated_at DESC)
        """)
        # ---- Calendar events ----
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                all_day BOOLEAN NOT NULL DEFAULT false,
                start_time TIMESTAMPTZ NOT NULL,
                end_time TIMESTAMPTZ NOT NULL,
                notes TEXT,
                color TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calendar_events_workspace_time
            ON calendar_events(workspace_id, start_time, end_time)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                object_key TEXT NOT NULL UNIQUE,
                file_size BIGINT NOT NULL,
                mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                extracted_text TEXT,
                executive_summary TEXT,
                key_takeaways JSONB NOT NULL DEFAULT '[]'::jsonb,
                analyzed_at TIMESTAMPTZ,
                analysis_provider TEXT,
                analysis_model TEXT,
                uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS executive_summary TEXT
        """)
        await conn.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS key_takeaways JSONB NOT NULL DEFAULT '[]'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS analyzed_at TIMESTAMPTZ
        """)
        await conn.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS analysis_provider TEXT
        """)
        await conn.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS analysis_model TEXT
        """)
        await conn.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS tables_json JSONB NOT NULL DEFAULT '[]'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS preview_pdf_key TEXT
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace_id)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                char_start INTEGER NOT NULL DEFAULT 0,
                char_end INTEGER NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                search_vector TSVECTOR,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(document_id, chunk_index)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_document
            ON document_chunks(workspace_id, document_id, chunk_index)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_search
            ON document_chunks USING GIN(search_vector)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                topic TEXT NOT NULL,
                mode TEXT NOT NULL,
                research_type TEXT NOT NULL DEFAULT 'general',
                status TEXT NOT NULL DEFAULT 'running',
                summary TEXT,
                content TEXT,
                sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                llm_provider TEXT,
                llm_model TEXT,
                error TEXT,
                refinement JSONB,
                source_document_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            ALTER TABLE research_sessions
            ADD COLUMN IF NOT EXISTS refinement JSONB
        """)
        await conn.execute("""
            ALTER TABLE research_sessions
            ADD COLUMN IF NOT EXISTS linked_todo_id TEXT
        """)
        await conn.execute("""
            ALTER TABLE research_sessions
            ADD COLUMN IF NOT EXISTS source_research_ids JSONB NOT NULL DEFAULT '[]'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE research_sessions
            ADD COLUMN IF NOT EXISTS source_document_refs JSONB NOT NULL DEFAULT '[]'::jsonb
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_research_sessions_workspace
            ON research_sessions(workspace_id, created_at DESC)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS generate_task_sessions (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                todo_id TEXT NOT NULL UNIQUE,
                title TEXT,
                artifact_template TEXT NOT NULL DEFAULT 'requirements',
                output_type TEXT NOT NULL DEFAULT 'pdf',
                website_url TEXT,
                branding JSONB NOT NULL DEFAULT '{}'::jsonb,
                selected_meeting_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                selected_document_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                selected_research_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                selected_todo_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                selected_todo_people JSONB NOT NULL DEFAULT '[]'::jsonb,
                related_research_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                linked_research_id INTEGER REFERENCES research_sessions(id) ON DELETE SET NULL,
                question_plan JSONB NOT NULL DEFAULT '[]'::jsonb,
                grounding_pack JSONB NOT NULL DEFAULT '{}'::jsonb,
                template_draft JSONB NOT NULL DEFAULT '{}'::jsonb,
                answers JSONB NOT NULL DEFAULT '{}'::jsonb,
                answer_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
                question_research JSONB NOT NULL DEFAULT '{}'::jsonb,
                prompt TEXT,
                current_step TEXT NOT NULL DEFAULT 'setup',
                latest_document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_generate_task_sessions_workspace
            ON generate_task_sessions(workspace_id, updated_at DESC)
        """)
        await conn.execute("""
            ALTER TABLE generate_task_sessions
            ADD COLUMN IF NOT EXISTS selected_todo_ids JSONB NOT NULL DEFAULT '[]'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE generate_task_sessions
            ADD COLUMN IF NOT EXISTS selected_todo_people JSONB NOT NULL DEFAULT '[]'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE generate_task_sessions
            ADD COLUMN IF NOT EXISTS question_research JSONB NOT NULL DEFAULT '{}'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE generate_task_sessions
            ADD COLUMN IF NOT EXISTS answer_evidence JSONB NOT NULL DEFAULT '{}'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE generate_task_sessions
            ADD COLUMN IF NOT EXISTS template_draft JSONB NOT NULL DEFAULT '{}'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE generate_task_sessions
            ADD COLUMN IF NOT EXISTS grounding_pack JSONB NOT NULL DEFAULT '{}'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE generate_task_sessions
            ADD COLUMN IF NOT EXISTS active_run_id INTEGER
        """)
        await conn.execute("""
            ALTER TABLE generate_task_sessions
            ADD COLUMN IF NOT EXISTS template_chat_history JSONB NOT NULL DEFAULT '[]'::jsonb
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS generate_task_runs (
                id SERIAL PRIMARY KEY,
                task_id INTEGER NOT NULL REFERENCES generate_task_sessions(id) ON DELETE CASCADE,
                run_number INTEGER NOT NULL,
                title TEXT,
                artifact_template TEXT NOT NULL DEFAULT 'requirements',
                output_type TEXT NOT NULL DEFAULT 'pdf',
                website_url TEXT,
                branding JSONB NOT NULL DEFAULT '{}'::jsonb,
                selected_meeting_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                selected_document_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                selected_research_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                selected_todo_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                selected_todo_people JSONB NOT NULL DEFAULT '[]'::jsonb,
                related_research_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                grounding_research_id INTEGER REFERENCES research_sessions(id) ON DELETE SET NULL,
                question_plan JSONB NOT NULL DEFAULT '[]'::jsonb,
                grounding_pack JSONB NOT NULL DEFAULT '{}'::jsonb,
                template_draft JSONB NOT NULL DEFAULT '{}'::jsonb,
                answers JSONB NOT NULL DEFAULT '{}'::jsonb,
                answer_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
                question_research JSONB NOT NULL DEFAULT '{}'::jsonb,
                answer_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                prompt TEXT,
                current_step TEXT NOT NULL DEFAULT 'setup',
                latest_document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                is_stale BOOLEAN NOT NULL DEFAULT FALSE,
                stale_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(task_id, run_number)
            )
        """)
        await conn.execute("""
            ALTER TABLE generate_task_runs
            ADD COLUMN IF NOT EXISTS answer_meta JSONB NOT NULL DEFAULT '{}'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE generate_task_runs
            ADD COLUMN IF NOT EXISTS template_draft JSONB NOT NULL DEFAULT '{}'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE generate_task_runs
            ADD COLUMN IF NOT EXISTS grounding_pack JSONB NOT NULL DEFAULT '{}'::jsonb
        """)
        await conn.execute("""
            ALTER TABLE generate_task_runs
            ADD COLUMN IF NOT EXISTS is_stale BOOLEAN NOT NULL DEFAULT FALSE
        """)
        await conn.execute("""
            ALTER TABLE generate_task_runs
            ADD COLUMN IF NOT EXISTS stale_flags JSONB NOT NULL DEFAULT '[]'::jsonb
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_generate_task_runs_task
            ON generate_task_runs(task_id, run_number DESC, updated_at DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_generate_task_runs_research
            ON generate_task_runs(grounding_research_id)
        """)
        task_rows = await conn.fetch(
            """
            SELECT *
            FROM generate_task_sessions
            ORDER BY id
            """
        )
        for task_row in task_rows:
            task_data = dict(task_row)
            active_run_id = task_data["active_run_id"]
            existing_run_id = await conn.fetchval(
                "SELECT id FROM generate_task_runs WHERE task_id = $1 ORDER BY run_number DESC, id DESC LIMIT 1",
                task_data["id"],
            )
            if not existing_run_id:
                answers = _json_dict(task_data.get("answers"))
                updated_marker = task_data["updated_at"] or task_data["created_at"] or datetime.now(timezone.utc)
                answer_meta: dict[str, Any] = {}
                for key, value in answers.items():
                    text = str(value or "").strip()
                    if not text:
                        continue
                    answer_meta[str(key)] = {
                        "source": "manual",
                        "updated_at": updated_marker.isoformat(),
                    }
                existing_run_id = await conn.fetchval(
                    """
                    INSERT INTO generate_task_runs (
                        task_id, run_number, title, artifact_template, output_type,
                        website_url, branding, selected_meeting_ids, selected_document_ids,
                        selected_research_ids, selected_todo_ids, selected_todo_people,
                        related_research_ids, grounding_research_id, question_plan, grounding_pack, template_draft,
                        answers, answer_evidence, question_research, answer_meta,
                        prompt, current_step, latest_document_id, status,
                        created_at, updated_at
                    )
                    VALUES (
                        $1, 1, $2, $3, $4,
                        $5, $6::jsonb, $7::jsonb, $8::jsonb,
                        $9::jsonb, $10::jsonb, $11::jsonb,
                        $12::jsonb, $13, $14::jsonb, $15::jsonb, $16::jsonb,
                        $17::jsonb, $18::jsonb, $19::jsonb, $20::jsonb,
                        $21, $22, $23, $24,
                        $25, $26
                    )
                    RETURNING id
                    """,
                    task_data["id"],
                    task_data["title"],
                    task_data["artifact_template"],
                    task_data["output_type"],
                    task_data["website_url"],
                    json.dumps(_json_dict(task_data.get("branding")), default=_json_default),
                    json.dumps(_json_list(task_data.get("selected_meeting_ids")), default=_json_default),
                    json.dumps(_json_list(task_data.get("selected_document_ids")), default=_json_default),
                    json.dumps(_json_list(task_data.get("selected_research_ids")), default=_json_default),
                    json.dumps(_json_list(task_data.get("selected_todo_ids")), default=_json_default),
                    json.dumps(_json_list(task_data.get("selected_todo_people")), default=_json_default),
                    json.dumps(_json_list(task_data.get("related_research_ids")), default=_json_default),
                    task_data["linked_research_id"],
                    json.dumps(_json_list(task_data.get("question_plan")), default=_json_default),
                    json.dumps(_json_dict(task_data.get("grounding_pack")), default=_json_default),
                    json.dumps(_json_dict(task_data.get("template_draft")), default=_json_default),
                    json.dumps(answers, default=_json_default),
                    json.dumps(_json_dict(task_data.get("answer_evidence")), default=_json_default),
                    json.dumps(_json_dict(task_data.get("question_research")), default=_json_default),
                    json.dumps(answer_meta, default=_json_default),
                    task_data["prompt"],
                    task_data["current_step"],
                    task_data["latest_document_id"],
                    task_data["status"],
                    task_data["created_at"] or datetime.now(timezone.utc),
                    updated_marker,
                )
            if not active_run_id and existing_run_id:
                await conn.execute(
                    """
                    UPDATE generate_task_sessions
                    SET active_run_id = $1
                    WHERE id = $2
                    """,
                    existing_run_id,
                    task_data["id"],
                )
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workspace_chat_sessions (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                archived BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workspace_chat_sessions_workspace
            ON workspace_chat_sessions(workspace_id, updated_at DESC, id DESC)
        """)
        await conn.execute("ALTER TABLE workspace_chat_sessions ADD COLUMN IF NOT EXISTS context_meeting_ids JSONB NOT NULL DEFAULT '[]'")
        await conn.execute("ALTER TABLE workspace_chat_sessions ADD COLUMN IF NOT EXISTS context_document_ids JSONB NOT NULL DEFAULT '[]'")
        await conn.execute("ALTER TABLE workspace_chat_sessions ADD COLUMN IF NOT EXISTS context_research_ids JSONB NOT NULL DEFAULT '[]'")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workspace_chat_messages (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                chat_session_id INTEGER REFERENCES workspace_chat_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            ALTER TABLE workspace_chat_messages
            ADD COLUMN IF NOT EXISTS chat_session_id INTEGER REFERENCES workspace_chat_sessions(id) ON DELETE CASCADE
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workspace_chat_messages_workspace
            ON workspace_chat_messages(workspace_id, id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workspace_chat_messages_session
            ON workspace_chat_messages(chat_session_id, id)
        """)
        await conn.execute("ALTER TABLE workspace_chat_messages ADD COLUMN IF NOT EXISTS attachment_ids JSONB NOT NULL DEFAULT '[]'")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_session_attachments (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                chat_session_id INTEGER NOT NULL REFERENCES workspace_chat_sessions(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                mime_type TEXT,
                file_size INTEGER,
                extracted_text TEXT,
                status TEXT NOT NULL DEFAULT 'processing',
                error_message TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_session_attachments_session
            ON chat_session_attachments(chat_session_id, id)
        """)
        await conn.execute("ALTER TABLE chat_session_attachments ADD COLUMN IF NOT EXISTS activated BOOLEAN NOT NULL DEFAULT FALSE")
        # ---- Meeting chunks (FTS) ----
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS meeting_chunks (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                char_start INTEGER NOT NULL DEFAULT 0,
                char_end INTEGER NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                search_vector TSVECTOR,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(meeting_id, chunk_index)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_meeting_chunks_meeting
            ON meeting_chunks(workspace_id, meeting_id, chunk_index)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_meeting_chunks_search
            ON meeting_chunks USING GIN(search_vector)
        """)
        # Enable pgvector and add embedding columns for RAG
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector(384)")
            await conn.execute("ALTER TABLE meeting_chunks ADD COLUMN IF NOT EXISTS embedding vector(384)")
        except Exception as exc:
            logger.warning("pgvector setup skipped (extension not available or tables not ready): %s", exc)
        # ---- Research chunks (RAG) ----
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS research_chunks (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                research_id INTEGER NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                char_start INTEGER NOT NULL DEFAULT 0,
                char_end INTEGER NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                search_vector TSVECTOR,
                embedding vector(384),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(research_id, chunk_index)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_research_chunks_research
            ON research_chunks(workspace_id, research_id, chunk_index)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_research_chunks_search
            ON research_chunks USING GIN(search_vector)
        """)
        # ---- Library & Sharing ----
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id TEXT")
        await conn.execute("ALTER TABLE documents ALTER COLUMN workspace_id DROP NOT NULL")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS content_shares (
                id SERIAL PRIMARY KEY,
                content_type TEXT NOT NULL,
                content_id INTEGER NOT NULL,
                owner_user_id TEXT NOT NULL,
                shared_with_user_id TEXT NOT NULL,
                permission TEXT NOT NULL DEFAULT 'read',
                shared_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(content_type, content_id, shared_with_user_id)
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_content_shares_shared_with ON content_shares (shared_with_user_id, content_type)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_content_shares_owner ON content_shares (owner_user_id)")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workspace_content_links (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                content_type TEXT NOT NULL,
                content_id INTEGER NOT NULL,
                linked_by TEXT,
                linked_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(workspace_id, content_type, content_id)
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_content_links_workspace ON workspace_content_links (workspace_id)")
        # Drive sync columns
        await conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS drive_folder_id TEXT")
        await conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS drive_folder_name TEXT")
        await conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS drive_last_synced_at TIMESTAMPTZ")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS drive_file_id TEXT")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS drive_modified_time TIMESTAMPTZ")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS folder_path TEXT DEFAULT ''")
        await conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS drive_changes_token TEXT")
        try:
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_drive_file_unique ON documents (workspace_id, drive_file_id) WHERE drive_file_id IS NOT NULL")
        except Exception:
            pass
        # Canvas LMS sync columns
        await conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS canvas_course_id TEXT")
        await conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS canvas_instance_url TEXT")
        await conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS canvas_course_name TEXT")
        await conn.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS canvas_last_synced_at TIMESTAMPTZ")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS canvas_file_id TEXT")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS canvas_modified_at TIMESTAMPTZ")
        # Calendar events source tracking for Canvas assignments
        await conn.execute("ALTER TABLE calendar_events ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'manual'")
        await conn.execute("ALTER TABLE calendar_events ADD COLUMN IF NOT EXISTS source_id TEXT")
        await conn.execute("ALTER TABLE calendar_events ADD COLUMN IF NOT EXISTS is_due_only BOOLEAN NOT NULL DEFAULT FALSE")
        try:
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_canvas_file_unique ON documents (workspace_id, canvas_file_id) WHERE canvas_file_id IS NOT NULL")
        except Exception:
            pass
        try:
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_calendar_events_source_unique ON calendar_events (workspace_id, source_type, source_id) WHERE source_type IS NOT NULL AND source_id IS NOT NULL")
        except Exception:
            pass
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_tokens (
                user_id TEXT PRIMARY KEY,
                access_token TEXT,
                refresh_token TEXT,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        # Add Canvas token column to user_tokens
        await conn.execute("ALTER TABLE user_tokens ADD COLUMN IF NOT EXISTS canvas_access_token TEXT")
        await conn.execute("ALTER TABLE user_tokens ADD COLUMN IF NOT EXISTS canvas_instance_url TEXT")
        # ---- Live Q&A tables ----
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS live_qa_sessions (
                id SERIAL PRIMARY KEY,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                meeting_id INTEGER REFERENCES meetings(id) ON DELETE SET NULL,
                context_meeting_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                context_document_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                context_research_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_live_qa_sessions_workspace
            ON live_qa_sessions(workspace_id)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS live_qa_entries (
                id SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES live_qa_sessions(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                answer TEXT NOT NULL DEFAULT '',
                sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                detected BOOLEAN NOT NULL DEFAULT FALSE,
                transcript_context TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_live_qa_entries_session
            ON live_qa_entries(session_id)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS upload_jobs (
                id          SERIAL PRIMARY KEY,
                workspace_id INT REFERENCES workspaces(id) ON DELETE SET NULL,
                user_id     TEXT,
                filename    TEXT NOT NULL,
                minio_key   TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'queued',
                error       TEXT,
                meeting_id  INT REFERENCES meetings(id) ON DELETE SET NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_upload_jobs_status
            ON upload_jobs(status, created_at)
        """)
        # Migrate upload_jobs.user_id from INT to TEXT (Keycloak user IDs are UUIDs)
        _upload_jobs_uid_type = await conn.fetchval(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'upload_jobs' AND column_name = 'user_id'"
        )
        if _upload_jobs_uid_type and _upload_jobs_uid_type != 'text':
            await conn.execute(
                "ALTER TABLE upload_jobs ALTER COLUMN user_id TYPE TEXT USING user_id::TEXT"
            )
        legacy_chat_workspaces = await conn.fetch(
            """
            SELECT workspace_id, MIN(created_at) AS first_message_at, MAX(created_at) AS last_message_at
            FROM workspace_chat_messages
            WHERE chat_session_id IS NULL
            GROUP BY workspace_id
            """
        )
        for legacy_row in legacy_chat_workspaces:
            workspace_id = legacy_row["workspace_id"]
            first_message_at = legacy_row["first_message_at"] or datetime.now(timezone.utc)
            last_message_at = legacy_row["last_message_at"] or first_message_at
            session_row = await conn.fetchrow(
                """
                INSERT INTO workspace_chat_sessions (workspace_id, title, created_at, updated_at)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                workspace_id,
                LEGACY_CHAT_SESSION_TITLE,
                first_message_at,
                last_message_at,
            )
            await conn.execute(
                """
                UPDATE workspace_chat_messages
                SET chat_session_id = $1
                WHERE workspace_id = $2 AND chat_session_id IS NULL
                """,
                session_row["id"],
                workspace_id,
            )
    return pool

