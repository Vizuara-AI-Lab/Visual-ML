"""
Knowledge Base API routes.
RAG document management and retrieval.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.genai import KnowledgeBase, KnowledgeBaseDocument
from app.models.user import Student
from app.schemas.genai import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
    DocumentUploadResponse,
    DocumentListItem,
    IndexKnowledgeBaseRequest,
)
from app.core.security import get_current_student
from app.core.logging import logger

router = APIRouter()


@router.post("/kb", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    kb_data: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    Create knowledge base for RAG.

    - **name**: Knowledge base name
    - **description**: Optional description
    - **embeddingModel**: Embedding model (default: text-embedding-ada-002)
    - **chunkSize**: Chunk size for splitting (default: 500)
    - **chunkOverlap**: Overlap between chunks (default: 50)
    - **vectorStore**: Vector store backend (chroma, pinecone, qdrant)
    """
    try:
        kb = KnowledgeBase(
            name=kb_data.name,
            description=kb_data.description,
            studentId=current_student.id,
            embeddingModel=kb_data.embeddingModel,
            chunkSize=kb_data.chunkSize,
            chunkOverlap=kb_data.chunkOverlap,
            vectorStore=kb_data.vectorStore,
            totalDocuments=0,
            totalChunks=0,
        )

        db.add(kb)
        db.commit()
        db.refresh(kb)

        logger.info(f"Created knowledge base {kb.id} by student {current_student.id}")
        return kb

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create knowledge base: {str(e)}",
        )


@router.get("/kb", response_model=List[KnowledgeBaseResponse])
async def list_knowledge_bases(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """List student's knowledge bases."""
    kbs = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.studentId == current_student.id)
        .order_by(KnowledgeBase.createdAt.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return kbs


@router.get("/kb/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Get knowledge base details."""
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.studentId == current_student.id)
        .first()
    )

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found"
        )

    return kb


@router.patch("/kb/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: int,
    kb_data: KnowledgeBaseUpdate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Update knowledge base metadata."""
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.studentId == current_student.id)
        .first()
    )

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found"
        )

    update_data = kb_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(kb, field, value)

    db.commit()
    db.refresh(kb)

    return kb


@router.delete("/kb/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    kb_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Delete knowledge base and all documents."""
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.studentId == current_student.id)
        .first()
    )

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found"
        )

    # Cascade delete will handle documents and chunks
    db.delete(kb)
    db.commit()

    logger.info(f"Deleted knowledge base {kb_id}")


@router.post("/kb/{kb_id}/upload", response_model=DocumentUploadResponse)
async def upload_document(
    kb_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    Upload document to knowledge base.

    Supported formats: PDF, TXT, MD, DOCX
    """
    # Verify KB ownership
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.studentId == current_student.id)
        .first()
    )

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found"
        )

    # Validate file type
    allowed_types = [
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}",
        )

    try:
        # Read file content
        content = await file.read()
        file_size = len(content)

        # Create document record
        doc = KnowledgeBaseDocument(
            knowledgeBaseId=kb_id,
            filename=file.filename,
            fileType=file.content_type,
            fileSize=file_size,
            totalChunks=0,  # Will be updated during indexing
            isIndexed=False,
        )

        db.add(doc)

        # TODO: Process document in background
        # 1. Extract text
        # 2. Chunk with configured size/overlap
        # 3. Generate embeddings
        # 4. Store in vector store
        # 5. Update doc.isIndexed = True

        # For now, just save document record
        db.commit()
        db.refresh(doc)

        # Update KB stats
        kb.totalDocuments += 1
        db.commit()

        logger.info(f"Uploaded document {doc.id} to KB {kb_id}")

        return DocumentUploadResponse(
            id=doc.id,
            filename=doc.filename,
            fileType=doc.fileType,
            fileSize=doc.fileSize,
            totalChunks=doc.totalChunks,
            isIndexed=doc.isIndexed,
            uploadedAt=doc.uploadedAt,
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to upload document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}",
        )


@router.get("/kb/{kb_id}/documents", response_model=List[DocumentListItem])
async def list_documents(
    kb_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """List documents in knowledge base."""
    # Verify ownership
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.studentId == current_student.id)
        .first()
    )

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found"
        )

    docs = (
        db.query(KnowledgeBaseDocument)
        .filter(KnowledgeBaseDocument.knowledgeBaseId == kb_id)
        .order_by(KnowledgeBaseDocument.uploadedAt.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return docs


@router.post("/kb/{kb_id}/index")
async def index_knowledge_base(
    kb_id: int,
    index_data: IndexKnowledgeBaseRequest,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    Index/re-index knowledge base.

    Triggers background job to:
    1. Extract text from all documents
    2. Chunk documents
    3. Generate embeddings
    4. Store in vector store
    """
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.studentId == current_student.id)
        .first()
    )

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found"
        )

    # TODO: Trigger background indexing job
    # For now, return placeholder

    return {
        "message": "Indexing started",
        "knowledgeBaseId": kb_id,
        "forceReindex": index_data.forceReindex,
    }


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Delete document from knowledge base."""
    doc = (
        db.query(KnowledgeBaseDocument)
        .join(KnowledgeBase)
        .filter(KnowledgeBaseDocument.id == doc_id, KnowledgeBase.studentId == current_student.id)
        .first()
    )

    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    kb = doc.knowledgeBase
    kb.totalDocuments -= 1
    kb.totalChunks -= doc.totalChunks

    db.delete(doc)
    db.commit()

    logger.info(f"Deleted document {doc_id}")
