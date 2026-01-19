"""
API Secrets management routes.
Encrypted storage for OpenAI, Anthropic, etc. API keys.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.session import get_db
from app.models.genai import APISecret
from app.models.user import Student
from app.schemas.genai import APISecretCreate, APISecretUpdate, APISecretResponse, LLMProviderEnum
from app.core.security import (
    get_current_student,
    encrypt_api_key,
    decrypt_api_key,
    hash_api_key,
    mask_api_key,
)
from app.core.logging import logger

router = APIRouter()


@router.post("/secrets", response_model=APISecretResponse, status_code=status.HTTP_201_CREATED)
async def create_secret(
    secret_data: APISecretCreate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    Store encrypted API key.

    - **name**: Friendly name for the key
    - **provider**: openai, anthropic, huggingface
    - **apiKey**: API key (will be encrypted)
    - **expiresAt**: Optional expiration date

    SECURITY:
    - Keys are encrypted using Fernet
    - Only hash is stored for deduplication
    - Keys are never logged or returned in plaintext
    """
    try:
        # Check for duplicate key (using hash)
        key_hash = hash_api_key(secret_data.apiKey)

        existing = (
            db.query(APISecret)
            .filter(APISecret.studentId == current_student.id, APISecret.keyHash == key_hash)
            .first()
        )

        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="This API key already exists"
            )

        # Encrypt key
        encrypted_key = encrypt_api_key(secret_data.apiKey)

        # Create secret
        secret = APISecret(
            name=secret_data.name,
            provider=secret_data.provider,
            studentId=current_student.id,
            encryptedKey=encrypted_key,
            keyHash=key_hash,
            isActive=True,
            usageCount=0,
            expiresAt=secret_data.expiresAt,
        )

        db.add(secret)
        db.commit()
        db.refresh(secret)

        logger.info(f"Created API secret {secret.id} for student {current_student.id}")

        # Return response with masked key
        return APISecretResponse(
            id=secret.id,
            name=secret.name,
            provider=secret.provider,
            keyPreview=mask_api_key(secret_data.apiKey),
            isActive=secret.isActive,
            lastUsedAt=secret.lastUsedAt,
            usageCount=secret.usageCount,
            createdAt=secret.createdAt,
            expiresAt=secret.expiresAt,
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create secret: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create secret"
        )


@router.get("/secrets", response_model=List[APISecretResponse])
async def list_secrets(
    provider: str = Query(None, description="Filter by provider"),
    active_only: bool = Query(True, description="Show only active keys"),
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    List stored API keys.

    - Keys are returned with masked preview only
    - Never returns plaintext keys
    """
    query = db.query(APISecret).filter(APISecret.studentId == current_student.id)

    if provider:
        try:
            provider_enum = LLMProviderEnum(provider)
            query = query.filter(APISecret.provider == provider_enum.value)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid provider: {provider}"
            )

    if active_only:
        query = query.filter(APISecret.isActive == True)

    secrets = query.order_by(APISecret.createdAt.desc()).all()

    # Return with masked keys
    return [
        APISecretResponse(
            id=s.id,
            name=s.name,
            provider=s.provider,
            keyPreview=mask_api_key(decrypt_api_key(s.encryptedKey)),  # Decrypt to mask
            isActive=s.isActive,
            lastUsedAt=s.lastUsedAt,
            usageCount=s.usageCount,
            createdAt=s.createdAt,
            expiresAt=s.expiresAt,
        )
        for s in secrets
    ]


@router.get("/secrets/{secret_id}", response_model=APISecretResponse)
async def get_secret(
    secret_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Get API secret details (masked)."""
    secret = (
        db.query(APISecret)
        .filter(APISecret.id == secret_id, APISecret.studentId == current_student.id)
        .first()
    )

    if not secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Secret not found")

    return APISecretResponse(
        id=secret.id,
        name=secret.name,
        provider=secret.provider,
        keyPreview=mask_api_key(decrypt_api_key(secret.encryptedKey)),
        isActive=secret.isActive,
        lastUsedAt=secret.lastUsedAt,
        usageCount=secret.usageCount,
        createdAt=secret.createdAt,
        expiresAt=secret.expiresAt,
    )


@router.patch("/secrets/{secret_id}", response_model=APISecretResponse)
async def update_secret(
    secret_id: int,
    secret_data: APISecretUpdate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    Update API secret.

    Can update:
    - name: Friendly name
    - isActive: Enable/disable key

    Cannot update:
    - provider: Cannot change provider
    - apiKey: Create new secret instead
    """
    secret = (
        db.query(APISecret)
        .filter(APISecret.id == secret_id, APISecret.studentId == current_student.id)
        .first()
    )

    if not secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Secret not found")

    # Update fields
    update_data = secret_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(secret, field, value)

    db.commit()
    db.refresh(secret)

    return APISecretResponse(
        id=secret.id,
        name=secret.name,
        provider=secret.provider,
        keyPreview=mask_api_key(decrypt_api_key(secret.encryptedKey)),
        isActive=secret.isActive,
        lastUsedAt=secret.lastUsedAt,
        usageCount=secret.usageCount,
        createdAt=secret.createdAt,
        expiresAt=secret.expiresAt,
    )


@router.delete("/secrets/{secret_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_secret(
    secret_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    Delete API secret.

    WARNING: This will break any pipelines using this key!
    """
    secret = (
        db.query(APISecret)
        .filter(APISecret.id == secret_id, APISecret.studentId == current_student.id)
        .first()
    )

    if not secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Secret not found")

    db.delete(secret)
    db.commit()

    logger.info(f"Deleted API secret {secret_id}")


# Internal helper for nodes to retrieve keys
async def get_api_key_for_node(secret_id: int, student_id: int, db: Session) -> str:
    """
    Internal function for nodes to retrieve decrypted API keys.

    NOTE: This is NOT an API endpoint - only for internal use by nodes.
    """
    secret = (
        db.query(APISecret)
        .filter(
            APISecret.id == secret_id, APISecret.studentId == student_id, APISecret.isActive == True
        )
        .first()
    )

    if not secret:
        raise ValueError(f"API secret {secret_id} not found or inactive")

    # Check expiration
    if secret.expiresAt and secret.expiresAt < datetime.utcnow():
        raise ValueError(f"API secret {secret_id} has expired")

    # Update usage stats
    secret.lastUsedAt = datetime.utcnow()
    secret.usageCount += 1
    db.commit()

    # Decrypt and return
    return decrypt_api_key(secret.encryptedKey)
