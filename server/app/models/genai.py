"""
GenAI Pipeline Database Models.
Supports DAG-based node pipelines with branching and flexible execution.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    Float,
    DateTime,
    ForeignKey,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.db.session import Base


class PipelineType(str, enum.Enum):
    """Pipeline type enumeration."""

    CLASSIC_ML = "CLASSIC_ML"
    GENAI = "GENAI"
    HYBRID = "HYBRID"


class PipelineStatus(str, enum.Enum):
    """Pipeline status enumeration."""

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class RunStatus(str, enum.Enum):
    """Run status enumeration."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


class GenAIPipeline(Base):
    """
    GenAI Pipeline model.
    Represents a complete node-based pipeline with DAG structure.
    """

    __tablename__ = "genai_pipelines"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    pipelineType = Column(SQLEnum(PipelineType), nullable=False, default=PipelineType.GENAI)
    status = Column(SQLEnum(PipelineStatus), nullable=False, default=PipelineStatus.DRAFT)

    # Owner (student)
    studentId = Column(
        Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Pipeline configuration
    config = Column(JSON, nullable=True)  # Global pipeline settings
    tags = Column(JSON, nullable=True)  # Array of tags for search

    # Metadata
    version = Column(Integer, nullable=False, default=1)
    isTemplate = Column(Boolean, nullable=False, default=False)
    templateId = Column(
        Integer, ForeignKey("genai_pipelines.id", ondelete="SET NULL"), nullable=True
    )

    # Sharing and collaboration
    is_public = Column(Boolean, nullable=False, default=False)
    share_token = Column(String(255), nullable=True, unique=True, index=True)
    allow_cloning = Column(Boolean, nullable=False, default=True)
    view_count = Column(Integer, nullable=False, default=0)
    clone_count = Column(Integer, nullable=False, default=0)

    # Timestamps
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    lastRunAt = Column(DateTime, nullable=True)

    # Relationships
    nodes = relationship("GenAINode", back_populates="pipeline", cascade="all, delete-orphan")
    edges = relationship("GenAIEdge", back_populates="pipeline", cascade="all, delete-orphan")
    runs = relationship("GenAIPipelineRun", back_populates="pipeline", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<GenAIPipeline(id={self.id}, name='{self.name}', type={self.pipelineType})>"


class GenAINode(Base):
    """
    GenAI Node model.
    Represents a single node in the pipeline (LLM, RAG, Prompt, etc.).
    """

    __tablename__ = "genai_nodes"

    id = Column(Integer, primary_key=True, index=True)
    pipelineId = Column(
        Integer, ForeignKey("genai_pipelines.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Node identity
    nodeType = Column(String(100), nullable=False, index=True)  # llm, rag, prompt_template, etc.
    nodeId = Column(String(255), nullable=False, unique=True, index=True)  # Unique node identifier
    label = Column(String(255), nullable=True)  # User-friendly label

    # Node configuration (type-specific)
    config = Column(JSON, nullable=False)  # Node-specific settings

    # UI positioning (for canvas)
    positionX = Column(Float, nullable=True)
    positionY = Column(Float, nullable=True)

    # Execution
    isEnabled = Column(Boolean, nullable=False, default=True)
    executionOrder = Column(Integer, nullable=True)  # For sequential pipelines

    # Timestamps
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    pipeline = relationship("GenAIPipeline", back_populates="nodes")
    outgoing_edges = relationship(
        "GenAIEdge",
        foreign_keys="GenAIEdge.sourceNodeId",
        back_populates="source_node",
        cascade="all, delete-orphan",
    )
    incoming_edges = relationship(
        "GenAIEdge",
        foreign_keys="GenAIEdge.targetNodeId",
        back_populates="target_node",
        cascade="all, delete-orphan",
    )
    executions = relationship(
        "GenAINodeExecution", back_populates="node", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<GenAINode(id={self.id}, type='{self.nodeType}', nodeId='{self.nodeId}')>"


class GenAIEdge(Base):
    """
    GenAI Edge model.
    Represents connections between nodes in a DAG.
    """

    __tablename__ = "genai_edges"

    id = Column(Integer, primary_key=True, index=True)
    pipelineId = Column(
        Integer, ForeignKey("genai_pipelines.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Connection
    sourceNodeId = Column(
        Integer, ForeignKey("genai_nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    targetNodeId = Column(
        Integer, ForeignKey("genai_nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Edge metadata
    sourceHandle = Column(String(50), nullable=True)  # For multi-output nodes
    targetHandle = Column(String(50), nullable=True)  # For multi-input nodes
    label = Column(String(255), nullable=True)

    # Conditional execution (optional)
    condition = Column(
        JSON, nullable=True
    )  # e.g., {"type": "success", "field": "score", "operator": ">", "value": 0.8}

    # Timestamps
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    pipeline = relationship("GenAIPipeline", back_populates="edges")
    source_node = relationship(
        "GenAINode", foreign_keys=[sourceNodeId], back_populates="outgoing_edges"
    )
    target_node = relationship(
        "GenAINode", foreign_keys=[targetNodeId], back_populates="incoming_edges"
    )

    def __repr__(self):
        return f"<GenAIEdge(id={self.id}, {self.sourceNodeId} -> {self.targetNodeId})>"


class GenAIPipelineRun(Base):
    """
    GenAI Pipeline Run model.
    Tracks execution of a pipeline.
    """

    __tablename__ = "genai_pipeline_runs"

    id = Column(Integer, primary_key=True, index=True)
    runId = Column(String(255), nullable=False, unique=True, index=True)

    pipelineId = Column(
        Integer, ForeignKey("genai_pipelines.id", ondelete="CASCADE"), nullable=False, index=True
    )
    studentId = Column(
        Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Run configuration
    status = Column(SQLEnum(RunStatus), nullable=False, default=RunStatus.PENDING, index=True)
    startNodeId = Column(
        Integer, ForeignKey("genai_nodes.id", ondelete="SET NULL"), nullable=True
    )  # For partial runs

    # Execution metadata
    inputData = Column(JSON, nullable=True)  # Initial input
    finalOutput = Column(JSON, nullable=True)  # Final result

    # Performance tracking
    executionTimeMs = Column(Integer, nullable=True)

    # Error handling
    error = Column(Text, nullable=True)
    errorStack = Column(Text, nullable=True)

    # Timestamps
    startedAt = Column(DateTime, nullable=True, index=True)
    completedAt = Column(DateTime, nullable=True)
    cancelledAt = Column(DateTime, nullable=True)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    pipeline = relationship("GenAIPipeline", back_populates="runs")
    node_executions = relationship(
        "GenAINodeExecution", back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<GenAIPipelineRun(id={self.id}, runId='{self.runId}', status={self.status})>"


class GenAINodeExecution(Base):
    """
    GenAI Node Execution model.
    Tracks individual node execution within a run.
    """

    __tablename__ = "genai_node_executions"

    id = Column(Integer, primary_key=True, index=True)

    runId = Column(
        Integer,
        ForeignKey("genai_pipeline_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    nodeId = Column(
        Integer, ForeignKey("genai_nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Execution data
    status = Column(SQLEnum(RunStatus), nullable=False, default=RunStatus.PENDING)
    inputData = Column(JSON, nullable=True)
    outputData = Column(JSON, nullable=True)

    # Performance metrics
    executionTimeMs = Column(Integer, nullable=True)

    # Provider metadata (for LLM nodes)
    provider = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)

    # Error tracking
    error = Column(Text, nullable=True)
    retryCount = Column(Integer, nullable=False, default=0)

    # Timestamps
    startedAt = Column(DateTime, nullable=True)
    completedAt = Column(DateTime, nullable=True)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    run = relationship("GenAIPipelineRun", back_populates="node_executions")
    node = relationship("GenAINode", back_populates="executions")

    def __repr__(self):
        return f"<GenAINodeExecution(id={self.id}, node={self.nodeId}, status={self.status})>"


class APISecret(Base):
    """
    API Secret model.
    Encrypted storage for API keys (OpenAI, Anthropic, etc.).
    """

    __tablename__ = "api_secrets"

    id = Column(Integer, primary_key=True, index=True)

    # Owner
    studentId = Column(
        Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Secret metadata
    name = Column(String(255), nullable=False)  # User-friendly name
    provider = Column(String(50), nullable=False, index=True)  # openai, anthropic, huggingface

    # Encrypted key (never log this!)
    encryptedKey = Column(Text, nullable=False)
    keyHash = Column(
        String(255), nullable=False, unique=True, index=True
    )  # SHA256 for deduplication

    # Metadata
    isActive = Column(Boolean, nullable=False, default=True)
    lastUsedAt = Column(DateTime, nullable=True)
    usageCount = Column(Integer, nullable=False, default=0)

    # Timestamps
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    expiresAt = Column(DateTime, nullable=True)  # Optional expiry

    def __repr__(self):
        return f"<APISecret(id={self.id}, provider='{self.provider}', name='{self.name}')>"


