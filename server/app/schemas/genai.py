"""
Pydantic schemas for GenAI Pipeline API.
Type-safe request/response models for all GenAI operations.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


# ========== Enums ==========


class PipelineTypeEnum(str, Enum):
    """Pipeline type enumeration."""

    CLASSIC_ML = "CLASSIC_ML"
    GENAI = "GENAI"
    HYBRID = "HYBRID"


class PipelineStatusEnum(str, Enum):
    """Pipeline status enumeration."""

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class RunStatusEnum(str, Enum):
    """Run status enumeration."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


class NodeTypeEnum(str, Enum):
    """GenAI node type enumeration."""

    LLM = "llm"
    SYSTEM_PROMPT = "system_prompt"
    FEW_SHOT = "few_shot"
    PROMPT_TEMPLATE = "prompt_template"
    TEXT_GENERATION = "text_generation"
    CHATBOT = "chatbot"
    RAG = "rag"
    IMAGE_GENERATION = "image_generation"
    OUTPUT_PARSER = "output_parser"
    MEMORY = "memory"
    CONDITIONAL = "conditional"
    AGGREGATOR = "aggregator"


class LLMProviderEnum(str, Enum):
    """LLM provider enumeration."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


# ========== Pipeline Schemas ==========


class PipelineCreate(BaseModel):
    """Create pipeline request."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    pipelineType: PipelineTypeEnum = PipelineTypeEnum.GENAI
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class PipelineUpdate(BaseModel):
    """Update pipeline request."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[PipelineStatusEnum] = None
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class PipelineResponse(BaseModel):
    """Pipeline response."""

    id: int
    name: str
    description: Optional[str]
    pipelineType: PipelineTypeEnum
    status: PipelineStatusEnum
    studentId: int
    version: int
    isTemplate: bool
    createdAt: datetime
    updatedAt: datetime
    lastRunAt: Optional[datetime]

    class Config:
        from_attributes = True


class PipelineListItem(BaseModel):
    """Pipeline list item (lightweight)."""

    id: int
    name: str
    pipelineType: PipelineTypeEnum
    status: PipelineStatusEnum
    createdAt: datetime
    lastRunAt: Optional[datetime]

    class Config:
        from_attributes = True


# ========== Node Schemas ==========


class NodePosition(BaseModel):
    """Node UI position."""

    x: float
    y: float


class LLMNodeConfig(BaseModel):
    """LLM node configuration."""

    provider: LLMProviderEnum
    model: str = Field(..., description="Model name (e.g., gpt-4, claude-3-opus)")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    maxTokens: int = Field(1000, ge=1, le=100000)
    topP: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    frequencyPenalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    presencePenalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    stopSequences: Optional[List[str]] = None
    useOwnApiKey: bool = False
    apiKeyRef: Optional[int] = None  # Reference to APISecret.id
    stream: bool = False


class SystemPromptNodeConfig(BaseModel):
    """System prompt node configuration."""

    systemRole: str = Field(..., description="System role (e.g., 'helpful assistant')")
    systemPrompt: str = Field(..., description="System instruction")


class FewShotExample(BaseModel):
    """Few-shot example pair."""

    input: str = Field(..., description="Example input")
    output: str = Field(..., description="Expected output")


class FewShotNodeConfig(BaseModel):
    """Few-shot examples node configuration."""

    examples: List[FewShotExample] = Field(..., min_length=1)
    prefix: Optional[str] = Field(
        "Here are some examples:", description="Prefix text before examples"
    )


class PromptTemplateNodeConfig(BaseModel):
    """Prompt template node configuration."""

    template: str = Field(..., description="Template with {{variables}}")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variable definitions")
    allowMissingVars: bool = Field(False, description="Allow missing variables")
    defaultValues: Optional[Dict[str, str]] = None


class RAGNodeConfig(BaseModel):
    """RAG node configuration."""

    knowledgeBaseId: int = Field(..., description="Knowledge base ID")
    topK: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")
    scoreThreshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    contextTemplate: str = Field(
        "Context:\n{context}\n\nQuestion: {query}", description="Template for injecting context"
    )


class ImageGenerationNodeConfig(BaseModel):
    """Image generation node configuration."""

    provider: str = Field("openai", description="openai, stable-diffusion")
    model: str = Field("dall-e-3", description="Model name")
    size: str = Field("1024x1024", description="Image size")
    quality: str = Field("standard", description="standard or hd")
    useOwnApiKey: bool = False
    apiKeyRef: Optional[int] = None


class OutputParserNodeConfig(BaseModel):
    """Output parser node configuration."""

    parserType: str = Field("json", description="json, regex, schema")
    jsonSchema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for validation")
    regexPattern: Optional[str] = None
    retryOnError: bool = Field(True, description="Retry with fix prompt on parse error")
    maxRetries: int = Field(2, ge=0, le=5)


class MemoryNodeConfig(BaseModel):
    """Memory node configuration."""

    sessionId: str = Field(..., description="Session identifier")
    maxTurns: int = Field(10, ge=1, le=100, description="Max messages to keep")
    summarizeMemory: bool = Field(False, description="Summarize old messages")
    retrieveRelevant: bool = Field(False, description="Retrieve relevant past context")


class NodeCreate(BaseModel):
    """Create node request."""

    pipelineId: int
    nodeType: NodeTypeEnum
    nodeId: str = Field(..., description="Unique node identifier")
    label: Optional[str] = None
    config: Dict[str, Any] = Field(..., description="Node-specific configuration")
    position: Optional[NodePosition] = None
    isEnabled: bool = True


class NodeUpdate(BaseModel):
    """Update node request."""

    label: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    position: Optional[NodePosition] = None
    isEnabled: Optional[bool] = None


class NodeResponse(BaseModel):
    """Node response."""

    id: int
    pipelineId: int
    nodeType: NodeTypeEnum
    nodeId: str
    label: Optional[str]
    config: Dict[str, Any]
    positionX: Optional[float]
    positionY: Optional[float]
    isEnabled: bool
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True


# ========== Edge Schemas ==========


class EdgeCreate(BaseModel):
    """Create edge request."""

    pipelineId: int
    sourceNodeId: int
    targetNodeId: int
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    label: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None


class EdgeUpdate(BaseModel):
    """Update edge request."""

    label: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None


class EdgeResponse(BaseModel):
    """Edge response."""

    id: int
    pipelineId: int
    sourceNodeId: int
    targetNodeId: int
    sourceHandle: Optional[str]
    targetHandle: Optional[str]
    label: Optional[str]
    condition: Optional[Dict[str, Any]]
    createdAt: datetime

    class Config:
        from_attributes = True


# ========== Pipeline Run Schemas ==========


class RunPipelineRequest(BaseModel):
    """Run pipeline request."""

    inputData: Optional[Dict[str, Any]] = None
    startFromNodeId: Optional[int] = None  # For partial runs
    config: Optional[Dict[str, Any]] = None


class RunStepRequest(BaseModel):
    """Run single step request."""

    nodeId: int
    inputData: Dict[str, Any]


class NodeExecutionResult(BaseModel):
    """Node execution result."""

    nodeId: int
    nodeType: str
    status: RunStatusEnum
    inputData: Optional[Dict[str, Any]]
    outputData: Optional[Dict[str, Any]]
    tokensUsed: Optional[int]
    costUSD: Optional[float]
    executionTimeMs: Optional[int]
    provider: Optional[str]
    model: Optional[str]
    error: Optional[str]
    startedAt: Optional[datetime]
    completedAt: Optional[datetime]


class PipelineRunResponse(BaseModel):
    """Pipeline run response."""

    id: int
    runId: str
    pipelineId: int
    status: RunStatusEnum
    finalOutput: Optional[Dict[str, Any]]
    totalTokensUsed: int
    totalCostUSD: float
    executionTimeMs: Optional[int]
    error: Optional[str]
    startedAt: Optional[datetime]
    completedAt: Optional[datetime]
    nodeExecutions: Optional[List[NodeExecutionResult]] = None

    class Config:
        from_attributes = True


class PipelineRunListItem(BaseModel):
    """Pipeline run list item."""

    id: int
    runId: str
    status: RunStatusEnum
    totalTokensUsed: int
    totalCostUSD: float
    startedAt: Optional[datetime]
    completedAt: Optional[datetime]

    class Config:
        from_attributes = True


# ========== Knowledge Base Schemas ==========


class KnowledgeBaseCreate(BaseModel):
    """Create knowledge base request."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    embeddingModel: str = Field("text-embedding-ada-002")
    chunkSize: int = Field(500, ge=100, le=2000)
    chunkOverlap: int = Field(50, ge=0, le=500)
    vectorStore: str = Field("chroma", description="chroma, pinecone, qdrant")


class KnowledgeBaseUpdate(BaseModel):
    """Update knowledge base request."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class KnowledgeBaseResponse(BaseModel):
    """Knowledge base response."""

    id: int
    name: str
    description: Optional[str]
    studentId: int
    embeddingModel: str
    chunkSize: int
    chunkOverlap: int
    vectorStore: str
    totalDocuments: int
    totalChunks: int
    indexedAt: Optional[datetime]
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseModel):
    """Document upload response."""

    id: int
    filename: str
    fileType: Optional[str]
    fileSize: Optional[int]
    totalChunks: int
    isIndexed: bool
    uploadedAt: datetime


class DocumentListItem(BaseModel):
    """Document list item."""

    id: int
    filename: str
    fileType: Optional[str]
    fileSize: Optional[int]
    totalChunks: int
    isIndexed: bool
    uploadedAt: datetime
    indexedAt: Optional[datetime]


class IndexKnowledgeBaseRequest(BaseModel):
    """Index knowledge base request."""

    forceReindex: bool = Field(False, description="Force re-indexing")


class RetrievalResult(BaseModel):
    """RAG retrieval result."""

    documentId: int
    filename: str
    chunkIndex: int
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class RAGQueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    sources: List[RetrievalResult]
    totalTokens: int
    costUSD: float


# ========== API Secret Schemas ==========


class APISecretCreate(BaseModel):
    """Create API secret request."""

    name: str = Field(..., min_length=1, max_length=255)
    provider: LLMProviderEnum
    apiKey: str = Field(..., min_length=10, description="API key (will be encrypted)")
    expiresAt: Optional[datetime] = None

    @field_validator("apiKey")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format."""
        if len(v) < 10:
            raise ValueError("API key too short")
        # Basic validation (don't log the key!)
        return v


class APISecretUpdate(BaseModel):
    """Update API secret request."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    isActive: Optional[bool] = None


class APISecretResponse(BaseModel):
    """API secret response (masked key)."""

    id: int
    name: str
    provider: LLMProviderEnum
    keyPreview: str  # Only first/last 4 chars
    isActive: bool
    lastUsedAt: Optional[datetime]
    usageCount: int
    createdAt: datetime
    expiresAt: Optional[datetime]

    class Config:
        from_attributes = True


# ========== Streaming Schemas ==========


class StreamChunk(BaseModel):
    """Streaming response chunk."""

    type: str  # "token", "metadata", "error", "done"
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ========== Complete Pipeline Schema ==========


class CompletePipelineResponse(BaseModel):
    """Complete pipeline with nodes and edges."""

    pipeline: PipelineResponse
    nodes: List[NodeResponse]
    edges: List[EdgeResponse]

    class Config:
        from_attributes = True


# ========== Error Schemas ==========


class ValidationError(BaseModel):
    """Validation error detail."""

    field: str
    message: str
    type: str


class ErrorResponse(BaseModel):
    """Generic error response."""

    error: str
    message: str
    details: Optional[Union[Dict[str, Any], List[ValidationError]]] = None
    suggestion: Optional[str] = None
