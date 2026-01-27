"""
Project API endpoints for managing student ML projects.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListItem,
    ProjectState,
    ProjectStateResponse,
)
from app.models.genai import GenAIPipeline, GenAINode, GenAIEdge, PipelineStatus
from app.core.security import get_current_student
from app.core.logging import logger
from app.db.session import get_db
from app.models.user import Student
from app.core.redis_cache import redis_cache
from datetime import datetime

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    data: ProjectCreate,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Create a new project.

    **Student Authentication Required**
    """
    try:
        project = GenAIPipeline(
            name=data.name,
            description=data.description,
            studentId=student.id,
            status=PipelineStatus.DRAFT,
        )

        db.add(project)
        db.commit()
        db.refresh(project)

        # Invalidate projects cache
        await redis_cache.delete(f"projects:student:{student.id}")

        logger.info(f"Student {student.id} created project {project.id}: {project.name}")

        return ProjectResponse.model_validate(project)

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create project"
        )


@router.get("", response_model=List[ProjectListItem])
async def list_projects(
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    List all projects for the current student.

    **Student Authentication Required**
    """
    # Query database
    projects = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.studentId == student.id)
        .order_by(GenAIPipeline.updatedAt.desc())
        .all()
    )

    # Convert to Pydantic models
    result = [ProjectListItem.model_validate(p) for p in projects]

    return result


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Get project details by ID.

    **Student Authentication Required**
    """
    project = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == project_id, GenAIPipeline.studentId == student.id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    return ProjectResponse.model_validate(project)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    data: ProjectUpdate,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Update project details.

    **Student Authentication Required**
    """
    project = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == project_id, GenAIPipeline.studentId == student.id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description

    project.updatedAt = datetime.utcnow()

    db.commit()
    db.refresh(project)

    # Invalidate caches
    await redis_cache.delete(f"projects:student:{student.id}")
    await redis_cache.delete(f"project:state:{project.id}")

    logger.info(f"Student {student.id} updated project {project.id}")

    return ProjectResponse.model_validate(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: int,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Delete a project.

    **Student Authentication Required**
    """
    project = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == project_id, GenAIPipeline.studentId == student.id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    db.delete(project)
    db.commit()

    # Invalidate caches
    await redis_cache.delete(f"projects:student:{student.id}")
    await redis_cache.delete(f"project:state:{project_id}")

    logger.info(f"Student {student.id} deleted project {project_id}")


@router.post("/{project_id}/save", response_model=ProjectStateResponse)
async def save_project_state(
    project_id: int,
    state: ProjectState,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Save playground state for a project.

    **Student Authentication Required**

    This endpoint saves the complete playground state including:
    - Nodes (with positions and configurations)
    - Edges (connections between nodes)
    - Dataset metadata
    - Execution results
    """
    project = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == project_id, GenAIPipeline.studentId == student.id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    try:
        # Delete existing nodes (edges deleted via CASCADE)
        db.query(GenAINode).filter(GenAINode.pipelineId == project_id).delete()

        # Create node ID mapping (React Flow ID -> DB ID)
        node_id_map = {}

        # Save nodes
        for node_data in state.nodes:
            node = GenAINode(
                pipelineId=project_id,
                nodeType=node_data.get("type", "unknown"),
                nodeId=node_data["id"],
                label=node_data.get("data", {}).get("label"),
                config=node_data.get("data", {}).get("config", {}),
                positionX=node_data.get("position", {}).get("x"),
                positionY=node_data.get("position", {}).get("y"),
                isEnabled=True,
            )
            db.add(node)
            db.flush()  # Get the DB ID
            node_id_map[node_data["id"]] = node.id

        # Save edges
        for edge_data in state.edges:
            source_db_id = node_id_map.get(edge_data["source"])
            target_db_id = node_id_map.get(edge_data["target"])

            if source_db_id and target_db_id:
                edge = GenAIEdge(
                    pipelineId=project_id,
                    sourceNodeId=source_db_id,
                    targetNodeId=target_db_id,
                    sourceHandle=edge_data.get("sourceHandle"),
                    targetHandle=edge_data.get("targetHandle"),
                    label=edge_data.get("label"),
                )
                db.add(edge)

        # Save metadata in config field
        project.config = {
            "datasetMetadata": state.datasetMetadata,
            "executionResult": state.executionResult,
        }
        project.updatedAt = datetime.utcnow()

        db.commit()
        db.refresh(project)

        logger.info(
            f"Saved state for project {project_id}: {len(state.nodes)} nodes, {len(state.edges)} edges"
        )

        return ProjectStateResponse(
            projectId=project.id,
            state=state,
            updatedAt=project.updatedAt,
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save project state: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save project state: {str(e)}",
        )


@router.get("/{project_id}/state", response_model=ProjectStateResponse)
async def get_project_state(
    project_id: int,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Get playground state for a project.

    **Student Authentication Required**
    """
    project = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == project_id, GenAIPipeline.studentId == student.id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    # Load nodes
    nodes = db.query(GenAINode).filter(GenAINode.pipelineId == project_id).all()

    # Load edges
    edges = db.query(GenAIEdge).filter(GenAIEdge.pipelineId == project_id).all()

    # Convert to React Flow format
    nodes_data = [
        {
            "id": node.nodeId,
            "type": node.nodeType,
            "position": {"x": node.positionX or 0, "y": node.positionY or 0},
            "data": {
                "label": node.label or node.nodeType,
                "config": node.config or {},
                "isConfigured": bool(node.config),
            },
        }
        for node in nodes
    ]

    # Create node ID mapping (DB ID -> React Flow ID)
    node_id_reverse_map = {node.id: node.nodeId for node in nodes}

    edges_data = [
        {
            "id": f"edge-{edge.id}",
            "source": node_id_reverse_map.get(edge.sourceNodeId),
            "target": node_id_reverse_map.get(edge.targetNodeId),
            "sourceHandle": edge.sourceHandle,
            "targetHandle": edge.targetHandle,
            "label": edge.label,
        }
        for edge in edges
        if edge.sourceNodeId in node_id_reverse_map and edge.targetNodeId in node_id_reverse_map
    ]

    # Get metadata from config
    config = project.config or {}

    state = ProjectState(
        nodes=nodes_data,
        edges=edges_data,
        datasetMetadata=config.get("datasetMetadata"),
        executionResult=config.get("executionResult"),
    )

    return ProjectStateResponse(
        projectId=project.id,
        state=state,
        updatedAt=project.updatedAt,
    )
