"""
Project Sharing API endpoints.
Handles project sharing, viewing shared projects, and cloning.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uuid

from app.db.session import get_db
from app.models.user import Student
from app.models.genai import GenAIPipeline, GenAINode, GenAIEdge
from app.core.security import get_current_student, get_optional_current_student
from app.core.logging import logger


router = APIRouter()


# Schemas
class ShareProjectRequest(BaseModel):
    is_public: bool = True
    allow_cloning: bool = True


class ShareProjectResponse(BaseModel):
    share_url: str
    share_token: str
    is_public: bool
    allow_cloning: bool


class SharedProjectResponse(BaseModel):
    project: dict
    permissions: dict
    owner: dict


class CloneProjectResponse(BaseModel):
    project_id: int
    message: str


# Helper functions
def generate_share_token() -> str:
    """Generate a unique share token"""
    return f"share_{uuid.uuid4().hex[:16]}"


def get_project_by_id(db: Session, project_id: int, user: Student) -> GenAIPipeline:
    """Get project by ID and verify ownership"""
    project = db.query(GenAIPipeline).filter(GenAIPipeline.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.studentId != user.id:
        raise HTTPException(status_code=403, detail="You don't own this project")

    return project


def get_project_by_share_token(db: Session, share_token: str) -> GenAIPipeline:
    """Get project by share token"""
    project = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.share_token == share_token, GenAIPipeline.is_public == True)
        .first()
    )

    if not project:
        raise HTTPException(status_code=404, detail="Shared project not found or not public")

    return project


# Endpoints
@router.post("/{project_id}/share", response_model=ShareProjectResponse)
async def share_project(
    project_id: int,
    request: ShareProjectRequest,
    db: Session = Depends(get_db),
    current_user: Student = Depends(get_current_student),
):
    """
    Generate a shareable link for a project.
    Only the project owner can share their project.
    """
    try:
        # Get project and verify ownership
        project = get_project_by_id(db, project_id, current_user)

        # Generate share token if not exists
        if not project.share_token:
            project.share_token = generate_share_token()

        # Update sharing settings
        project.is_public = request.is_public
        project.allow_cloning = request.allow_cloning

        db.commit()
        db.refresh(project)

        # Construct share URL (you may want to use env variable for base URL)
        share_url = f"/shared/{project.share_token}"

        logger.info(f"Project {project_id} shared by user {current_user.id}")

        return ShareProjectResponse(
            share_url=share_url,
            share_token=project.share_token,
            is_public=project.is_public,
            allow_cloning=project.allow_cloning,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sharing project {project_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{project_id}/share")
async def unshare_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: Student = Depends(get_current_student),
):
    """
    Revoke public access to a project.
    """
    try:
        project = get_project_by_id(db, project_id, current_user)

        project.is_public = False
        project.share_token = None

        db.commit()

        logger.info(f"Project {project_id} unshared by user {current_user.id}")

        return {"message": "Project unshared successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unsharing project {project_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shared/{share_token}", response_model=SharedProjectResponse)
async def get_shared_project(
    share_token: str,
    db: Session = Depends(get_db),
    current_user: Optional[Student] = Depends(get_optional_current_student),
):
    """
    Get a shared project by its sharetoken.
    Anonymous access allowed for public projects.
    """
    try:
        project = get_project_by_share_token(db, share_token)

        # Increment view count
        project.view_count += 1
        db.commit()

        # Get nodes
        nodes = db.query(GenAINode).filter(GenAINode.pipelineId == project.id).all()
        nodes_data = [
            {
                "id": node.nodeId,
                "type": node.nodeType,
                "position": {"x": node.positionX, "y": node.positionY},
                "data": {
                    "label": node.label,
                    "type": node.nodeType,
                    "config": node.config or {},
                    "isConfigured": node.isEnabled,
                    "validationErrors": [],
                },
            }
            for node in nodes
        ]

        # Get edges
        edges = db.query(GenAIEdge).filter(GenAIEdge.pipelineId == project.id).all()
        edges_data = [
            {
                "id": f"{edge.sourceNodeId}-{edge.targetNodeId}",
                "source": edge.source_node.nodeId if edge.source_node else None,
                "target": edge.target_node.nodeId if edge.target_node else None,
                "sourceHandle": edge.sourceHandle,
                "targetHandle": edge.targetHandle,
            }
            for edge in edges
        ]

        # Get owner info
        owner = db.query(Student).filter(Student.id == project.studentId).first()
        owner_data = {
            "name": owner.fullName if owner else "Unknown",
            "id": owner.id if owner else None,
        }

        # Determine permissions
        is_owner = current_user and current_user.id == project.studentId
        permissions = {
            "can_edit": is_owner,
            "can_clone": project.allow_cloning or is_owner,
            "can_run": True,
            "can_view": True,
        }

        project_data = {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "nodes": nodes_data,
            "edges": edges_data,
            "config": project.config or {},
            "view_count": project.view_count,
            "clone_count": project.clone_count,
            "created_at": project.createdAt.isoformat() if project.createdAt else None,
            "updated_at": project.updatedAt.isoformat() if project.updatedAt else None,
        }

        return SharedProjectResponse(
            project=project_data, permissions=permissions, owner=owner_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting shared project {share_token}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/shared/{share_token}/clone", response_model=CloneProjectResponse)
async def clone_shared_project(
    share_token: str,
    db: Session = Depends(get_db),
    current_user: Student = Depends(get_current_student),
):
    """
    Clone a shared project to the current user's account.
    Requires authentication.
    """
    try:
        # Get original project
        original_project = get_project_by_share_token(db, share_token)

        # Check if cloning is allowed
        if not original_project.allow_cloning and original_project.studentId != current_user.id:
            raise HTTPException(status_code=403, detail="Cloning not allowed for this project")

        # Create new project
        new_project = GenAIPipeline(
            name=f"{original_project.name} (Copy)",
            description=original_project.description,
            pipelineType=original_project.pipelineType,
            status=original_project.status,
            studentId=current_user.id,
            config=original_project.config,
            tags=original_project.tags,
            templateId=original_project.id,  # Reference to original
            isTemplate=False,
            is_public=False,  # Clones are private by default
            allow_cloning=True,
        )

        db.add(new_project)
        db.flush()  # Get the new project ID

        # Clone nodes
        original_nodes = (
            db.query(GenAINode).filter(GenAINode.pipelineId == original_project.id).all()
        )

        node_id_mapping = {}  # Map old node IDs to new node IDs

        for original_node in original_nodes:
            # Generate new node ID
            new_node_id = f"{original_node.nodeType}-{uuid.uuid4().hex[:12]}"
            node_id_mapping[original_node.nodeId] = new_node_id

            new_node = GenAINode(
                pipelineId=new_project.id,
                nodeType=original_node.nodeType,
                nodeId=new_node_id,
                label=original_node.label,
                config=original_node.config,
                positionX=original_node.positionX,
                positionY=original_node.positionY,
                isEnabled=original_node.isEnabled,
                executionOrder=original_node.executionOrder,
            )
            db.add(new_node)

        db.flush()  # Commit nodes to get their IDs

        # Clone edges
        original_edges = (
            db.query(GenAIEdge).filter(GenAIEdge.pipelineId == original_project.id).all()
        )

        # Get new node DB IDs
        new_nodes = db.query(GenAINode).filter(GenAINode.pipelineId == new_project.id).all()
        new_node_db_ids = {node.nodeId: node.id for node in new_nodes}

        for original_edge in original_edges:
            # Get original source and target node IDs
            source_node_id = original_edge.source_node.nodeId if original_edge.source_node else None
            target_node_id = original_edge.target_node.nodeId if original_edge.target_node else None

            if source_node_id and target_node_id:
                # Map to new node IDs
                new_source_node_id = node_id_mapping.get(source_node_id)
                new_target_node_id = node_id_mapping.get(target_node_id)

                if new_source_node_id and new_target_node_id:
                    new_edge = GenAIEdge(
                        pipelineId=new_project.id,
                        sourceNodeId=new_node_db_ids[new_source_node_id],
                        targetNodeId=new_node_db_ids[new_target_node_id],
                        sourceHandle=original_edge.sourceHandle,
                        targetHandle=original_edge.targetHandle,
                        label=original_edge.label,
                        condition=original_edge.condition,
                    )
                    db.add(new_edge)

        # Increment clone count on original
        original_project.clone_count += 1

        db.commit()

        logger.info(
            f"Project {original_project.id} cloned to {new_project.id} by user {current_user.id}"
        )

        return CloneProjectResponse(
            project_id=new_project.id, message="Project cloned successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cloning project {share_token}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/share-stats")
async def get_share_stats(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: Student = Depends(get_current_student),
):
    """
    Get sharing statistics for a project.
    """
    try:
        project = get_project_by_id(db, project_id, current_user)

        return {
            "is_public": project.is_public,
            "share_token": project.share_token,
            "allow_cloning": project.allow_cloning,
            "view_count": project.view_count,
            "clone_count": project.clone_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting share stats for project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
