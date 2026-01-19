"""add_genai_tables

Revision ID: 002_genai_pipeline_schema
Revises: 001_initial_auth_schema
Create Date: 2024-01-19

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = "002_genai_pipeline_schema"
down_revision = "001_initial_auth_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create GenAI pipeline tables."""

    # GenAI Pipelines
    op.create_table(
        "genai_pipelines",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "pipelineType",
            sa.Enum("CLASSIC_ML", "GENAI", "HYBRID", name="pipelinetype"),
            nullable=False,
        ),
        sa.Column(
            "status", sa.Enum("DRAFT", "ACTIVE", "ARCHIVED", name="pipelinestatus"), nullable=False
        ),
        sa.Column("studentId", sa.Integer(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("isTemplate", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("templateId", sa.Integer(), nullable=True),
        sa.Column("createdAt", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column(
            "updatedAt",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.Column("lastRunAt", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["studentId"], ["students.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["templateId"], ["genai_pipelines.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_genai_pipelines_studentId", "genai_pipelines", ["studentId"])
    op.create_index("ix_genai_pipelines_status", "genai_pipelines", ["status"])

    # GenAI Nodes
    op.create_table(
        "genai_nodes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pipelineId", sa.Integer(), nullable=False),
        sa.Column(
            "nodeType",
            sa.Enum(
                "llm",
                "system_prompt",
                "few_shot",
                "prompt_template",
                "text_generation",
                "chatbot",
                "rag",
                "image_generation",
                "output_parser",
                "memory",
                "conditional",
                "aggregator",
                name="nodetype",
            ),
            nullable=False,
        ),
        sa.Column("nodeId", sa.String(255), nullable=False),
        sa.Column("label", sa.String(255), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("positionX", sa.Float(), nullable=True),
        sa.Column("positionY", sa.Float(), nullable=True),
        sa.Column("isEnabled", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("createdAt", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column(
            "updatedAt",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["pipelineId"], ["genai_pipelines.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pipelineId", "nodeId", name="uq_pipeline_nodeid"),
    )
    op.create_index("ix_genai_nodes_pipelineId", "genai_nodes", ["pipelineId"])

    # GenAI Edges
    op.create_table(
        "genai_edges",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pipelineId", sa.Integer(), nullable=False),
        sa.Column("sourceNodeId", sa.Integer(), nullable=False),
        sa.Column("targetNodeId", sa.Integer(), nullable=False),
        sa.Column("sourceHandle", sa.String(255), nullable=True),
        sa.Column("targetHandle", sa.String(255), nullable=True),
        sa.Column("label", sa.String(255), nullable=True),
        sa.Column("condition", sa.JSON(), nullable=True),
        sa.Column("createdAt", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["pipelineId"], ["genai_pipelines.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["sourceNodeId"], ["genai_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["targetNodeId"], ["genai_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_genai_edges_pipelineId", "genai_edges", ["pipelineId"])

    # GenAI Pipeline Runs
    op.create_table(
        "genai_pipeline_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("runId", sa.String(255), nullable=False, unique=True),
        sa.Column("pipelineId", sa.Integer(), nullable=False),
        sa.Column("studentId", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "PAUSED", name="runstatus"
            ),
            nullable=False,
        ),
        sa.Column("inputData", sa.JSON(), nullable=True),
        sa.Column("finalOutput", sa.JSON(), nullable=True),
        sa.Column("totalTokensUsed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("totalCostUSD", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("executionTimeMs", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("createdAt", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("startedAt", sa.DateTime(), nullable=True),
        sa.Column("completedAt", sa.DateTime(), nullable=True),
        sa.Column(
            "updatedAt",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["pipelineId"], ["genai_pipelines.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["studentId"], ["students.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_genai_runs_runId", "genai_pipeline_runs", ["runId"], unique=True)
    op.create_index("ix_genai_runs_pipelineId", "genai_pipeline_runs", ["pipelineId"])
    op.create_index("ix_genai_runs_studentId", "genai_pipeline_runs", ["studentId"])

    # GenAI Node Executions
    op.create_table(
        "genai_node_executions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("runId", sa.Integer(), nullable=False),
        sa.Column("nodeId", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "PAUSED", name="runstatus"
            ),
            nullable=False,
        ),
        sa.Column("inputData", sa.JSON(), nullable=True),
        sa.Column("outputData", sa.JSON(), nullable=True),
        sa.Column("tokensUsed", sa.Integer(), nullable=True),
        sa.Column("costUSD", sa.Float(), nullable=True),
        sa.Column("executionTimeMs", sa.Integer(), nullable=True),
        sa.Column("provider", sa.String(100), nullable=True),
        sa.Column("model", sa.String(255), nullable=True),
        sa.Column("retryCount", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("startedAt", sa.DateTime(), nullable=True),
        sa.Column("completedAt", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["runId"], ["genai_pipeline_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["nodeId"], ["genai_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_genai_node_exec_runId", "genai_node_executions", ["runId"])

    # Knowledge Bases
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("studentId", sa.Integer(), nullable=False),
        sa.Column("embeddingModel", sa.String(255), nullable=False),
        sa.Column("chunkSize", sa.Integer(), nullable=False),
        sa.Column("chunkOverlap", sa.Integer(), nullable=False),
        sa.Column("vectorStore", sa.String(100), nullable=False),
        sa.Column("totalDocuments", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("totalChunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("indexedAt", sa.DateTime(), nullable=True),
        sa.Column("createdAt", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column(
            "updatedAt",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["studentId"], ["students.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_kb_studentId", "knowledge_bases", ["studentId"])

    # Knowledge Base Documents
    op.create_table(
        "knowledge_base_documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("knowledgeBaseId", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("fileType", sa.String(100), nullable=True),
        sa.Column("fileSize", sa.Integer(), nullable=True),
        sa.Column("totalChunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("isIndexed", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("uploadedAt", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("indexedAt", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["knowledgeBaseId"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_kbdocs_kbId", "knowledge_base_documents", ["knowledgeBaseId"])

    # Document Chunks
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("documentId", sa.Integer(), nullable=False),
        sa.Column("chunkIndex", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("createdAt", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["documentId"], ["knowledge_base_documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("documentId", "chunkIndex", name="uq_doc_chunk_index"),
    )
    op.create_index("ix_chunks_documentId", "document_chunks", ["documentId"])

    # API Secrets
    op.create_table(
        "api_secrets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "provider",
            sa.Enum("openai", "anthropic", "huggingface", "local", name="llmprovider"),
            nullable=False,
        ),
        sa.Column("studentId", sa.Integer(), nullable=False),
        sa.Column("encryptedKey", sa.Text(), nullable=False),
        sa.Column("keyHash", sa.String(64), nullable=False),
        sa.Column("isActive", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("lastUsedAt", sa.DateTime(), nullable=True),
        sa.Column("usageCount", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("createdAt", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("expiresAt", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["studentId"], ["students.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_secrets_studentId", "api_secrets", ["studentId"])
    op.create_index("ix_secrets_keyHash", "api_secrets", ["keyHash"])

    # Conversation Memory
    op.create_table(
        "conversation_memory",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("studentId", sa.Integer(), nullable=False),
        sa.Column("sessionId", sa.String(255), nullable=False),
        sa.Column("messages", sa.JSON(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("createdAt", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("lastMessageAt", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["studentId"], ["students.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("studentId", "sessionId", name="uq_student_session"),
    )
    op.create_index("ix_memory_sessionId", "conversation_memory", ["sessionId"])


def downgrade() -> None:
    """Drop GenAI pipeline tables."""

    # Drop in reverse order due to foreign keys
    op.drop_table("conversation_memory")
    op.drop_table("api_secrets")
    op.drop_table("document_chunks")
    op.drop_table("knowledge_base_documents")
    op.drop_table("knowledge_bases")
    op.drop_table("genai_node_executions")
    op.drop_table("genai_pipeline_runs")
    op.drop_table("genai_edges")
    op.drop_table("genai_nodes")
    op.drop_table("genai_pipelines")

    # Drop enums (SQLite doesn't use enums, but keeping for PostgreSQL compatibility)
    # For SQLite, these are just strings, so no need to drop
