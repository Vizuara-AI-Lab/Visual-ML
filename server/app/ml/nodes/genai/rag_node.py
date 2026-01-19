"""
RAG (Retrieval-Augmented Generation) Node.
Semantic search in knowledge base + context injection.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.ml.nodes.genai.base import GenAIBaseNode, GenAINodeInput, GenAINodeOutput, ProviderConfig


class RetrievalResult(BaseModel):
    """Single retrieval result."""

    documentId: int
    filename: str
    chunkIndex: int
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class RAGNode(GenAIBaseNode):
    """
    RAG Node - retrieve relevant context and inject into prompt.

    Config:
        knowledgeBaseId: Knowledge base ID
        topK: Number of chunks to retrieve (default: 5)
        scoreThreshold: Minimum similarity score (default: 0.7)
        contextTemplate: Template for injecting context
            Default: "Context:\n{context}\n\nQuestion: {query}"

    Input:
        query: Search query
        messages: Existing messages (optional)

    Output:
        messages: Messages with context injected
        retrievedChunks: List of retrieved chunks
        contextInjected: Final context string
    """

    node_type = "rag"

    def _parse_provider_config(self, config: Dict[str, Any]) -> ProviderConfig:
        """RAG doesn't use LLM provider (only embeddings)."""
        return ProviderConfig(
            provider="none",
            model="text-embedding-ada-002",  # Default embedding model
        )

    async def _execute_genai(self, input_data: GenAINodeInput) -> GenAINodeOutput:
        """Retrieve and inject context."""
        data = input_data.data

        query = data.get("query", "")
        if not query:
            return GenAINodeOutput(success=False, error="query is required", data={})

        kb_id = self.config.get("knowledgeBaseId")
        if not kb_id:
            return GenAINodeOutput(
                success=False, error="knowledgeBaseId is required in config", data={}
            )

        top_k = self.config.get("topK", 5)
        score_threshold = self.config.get("scoreThreshold", 0.7)
        context_template = self.config.get(
            "contextTemplate", "Context:\n{context}\n\nQuestion: {query}"
        )

        # Retrieve relevant chunks
        # Note: In production, this will query vector store (Chroma/Pinecone/Qdrant)
        # For now, we'll simulate retrieval
        try:
            chunks = await self._retrieve_chunks(kb_id, query, top_k, score_threshold)
        except Exception as e:
            return GenAINodeOutput(success=False, error=f"Retrieval error: {str(e)}", data={})

        if not chunks:
            return GenAINodeOutput(
                success=False, error="No relevant context found", data={"retrievedChunks": []}
            )

        # Build context string
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] {chunk.content} (Source: {chunk.filename}, Score: {chunk.score:.2f})"
            )
        context_str = "\n\n".join(context_parts)

        # Inject into template
        injected_prompt = context_template.format(context=context_str, query=query)

        # Build messages
        existing_messages = data.get("messages", [])
        messages = existing_messages.copy()
        messages.append({"role": "user", "content": injected_prompt})

        return GenAINodeOutput(
            success=True,
            data={
                "messages": messages,
                "retrievedChunks": [chunk.dict() for chunk in chunks],
                "contextInjected": context_str,
                "query": query,
            },
        )

    async def _retrieve_chunks(
        self, kb_id: int, query: str, top_k: int, score_threshold: float
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks from vector store.

        Production implementation will:
        1. Get knowledge base config from DB
        2. Generate query embedding
        3. Search vector store (Chroma/Pinecone/Qdrant)
        4. Filter by score threshold
        5. Return top_k results

        For now, returns mock data.
        """
        # TODO: Implement actual vector store retrieval
        # This is a placeholder for the structure

        # Example implementation outline:
        # from app.models.genai import KnowledgeBase, DocumentChunk
        # from app.ml.embeddings import get_embedding
        #
        # kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        # query_embedding = await get_embedding(query, kb.embeddingModel)
        #
        # if kb.vectorStore == "chroma":
        #     results = await search_chroma(kb.id, query_embedding, top_k)
        # elif kb.vectorStore == "pinecone":
        #     results = await search_pinecone(kb.id, query_embedding, top_k)
        #
        # return [RetrievalResult(...) for r in results if r.score >= score_threshold]

        # Mock data for now
        return [
            RetrievalResult(
                documentId=1,
                filename="sample.pdf",
                chunkIndex=0,
                content="This is a sample retrieved chunk with relevant information.",
                score=0.85,
                metadata={"page": 1},
            ),
            RetrievalResult(
                documentId=1,
                filename="sample.pdf",
                chunkIndex=1,
                content="This is another relevant chunk from the knowledge base.",
                score=0.78,
                metadata={"page": 2},
            ),
        ]

    def get_input_schema(self) -> Dict[str, Any]:
        """RAG input schema."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "messages": {"type": "array", "description": "Existing messages (optional)"},
            },
            "required": ["query"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        """RAG output schema."""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {
                    "type": "object",
                    "properties": {
                        "messages": {"type": "array"},
                        "retrievedChunks": {"type": "array"},
                        "contextInjected": {"type": "string"},
                        "query": {"type": "string"},
                    },
                },
            },
        }
