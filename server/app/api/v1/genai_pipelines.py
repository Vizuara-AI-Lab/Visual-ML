"""
GenAI Pipeline API routes.
CRUD operations for pipelines, nodes, edges, and execution.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.genai import (
    GenAIPipeline,
    GenAINode,
    GenAIEdge,
    GenAIPipelineRun,
    GenAINodeExecution,
    PipelineType,
    PipelineStatus,
    RunStatus,
)
from app.models.user import Student
from app.schemas.genai import (
    PipelineCreate,
    PipelineUpdate,
    PipelineResponse,
    PipelineListItem,
    CompletePipelineResponse,
    NodeCreate,
    NodeUpdate,
    NodeResponse,
    EdgeCreate,
    EdgeUpdate,
    EdgeResponse,
    RunPipelineRequest,
    PipelineRunResponse,
    PipelineRunListItem,
    NodeExecutionResult,
)
from app.core.security import get_current_student, get_current_admin
from app.core.logging import logger
from app.ml.genai_engine import GenAIPipelineEngine

router = APIRouter()


# ========== Pipeline Endpoints ==========


@router.post("/pipelines", response_model=PipelineResponse, status_code=status.HTTP_201_CREATED)
async def create_pipeline(
    pipeline_data: PipelineCreate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    Create new GenAI pipeline.

    - **name**: Pipeline name (required)
    - **description**: Optional description
    - **pipelineType**: CLASSIC_ML, GENAI, or HYBRID
    - **config**: Optional configuration JSON
    - **tags**: Optional tags array
    """
    try:
        pipeline = GenAIPipeline(
            name=pipeline_data.name,
            description=pipeline_data.description,
            pipelineType=pipeline_data.pipelineType,
            status=PipelineStatus.DRAFT,
            studentId=current_student.id,
            config=pipeline_data.config,
            tags=pipeline_data.tags or [],
            version=1,
        )

        db.add(pipeline)
        db.commit()
        db.refresh(pipeline)

        logger.info(f"Created pipeline {pipeline.id} by student {current_student.id}")
        return pipeline

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create pipeline: {str(e)}",
        )


@router.get("/pipelines", response_model=List[PipelineListItem])
async def list_pipelines(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status_filter: Optional[PipelineStatus] = None,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    List student's pipelines.

    - **skip**: Pagination offset
    - **limit**: Items per page (max 100)
    - **status_filter**: Filter by status (DRAFT, ACTIVE, ARCHIVED)
    """
    query = db.query(GenAIPipeline).filter(GenAIPipeline.studentId == current_student.id)

    if status_filter:
        query = query.filter(GenAIPipeline.status == status_filter)

    pipelines = query.order_by(GenAIPipeline.createdAt.desc()).offset(skip).limit(limit).all()

    return pipelines


@router.get("/pipelines/{pipeline_id}", response_model=CompletePipelineResponse)
async def get_pipeline(
    pipeline_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    Get complete pipeline with nodes and edges.
    """
    pipeline = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == pipeline_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not pipeline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")

    # Get nodes and edges
    nodes = db.query(GenAINode).filter(GenAINode.pipelineId == pipeline_id).all()
    edges = db.query(GenAIEdge).filter(GenAIEdge.pipelineId == pipeline_id).all()

    return {"pipeline": pipeline, "nodes": nodes, "edges": edges}


@router.patch("/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(
    pipeline_id: int,
    pipeline_data: PipelineUpdate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Update pipeline metadata."""
    pipeline = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == pipeline_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not pipeline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")

    # Update fields
    update_data = pipeline_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(pipeline, field, value)

    db.commit()
    db.refresh(pipeline)

    logger.info(f"Updated pipeline {pipeline_id}")
    return pipeline


@router.delete("/pipelines/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pipeline(
    pipeline_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Delete pipeline (soft delete by archiving)."""
    pipeline = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == pipeline_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not pipeline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")

    # Soft delete by archiving
    pipeline.status = PipelineStatus.ARCHIVED
    db.commit()

    logger.info(f"Archived pipeline {pipeline_id}")


# ========== Node Endpoints ==========


@router.post(
    "/pipelines/{pipeline_id}/nodes",
    response_model=NodeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_node(
    pipeline_id: int,
    node_data: NodeCreate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Create node in pipeline."""
    # Verify pipeline ownership
    pipeline = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == pipeline_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not pipeline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")

    try:
        node = GenAINode(
            pipelineId=pipeline_id,
            nodeType=node_data.nodeType,
            nodeId=node_data.nodeId,
            label=node_data.label,
            config=node_data.config,
            positionX=node_data.position.x if node_data.position else None,
            positionY=node_data.position.y if node_data.position else None,
            isEnabled=node_data.isEnabled,
        )

        db.add(node)
        db.commit()
        db.refresh(node)

        logger.info(f"Created node {node.id} in pipeline {pipeline_id}")
        return node

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create node: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create node: {str(e)}",
        )


@router.patch("/nodes/{node_id}", response_model=NodeResponse)
async def update_node(
    node_id: int,
    node_data: NodeUpdate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Update node configuration."""
    # Verify ownership through pipeline
    node = (
        db.query(GenAINode)
        .join(GenAIPipeline)
        .filter(GenAINode.id == node_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")

    # Update fields
    update_data = node_data.dict(exclude_unset=True)

    if "position" in update_data and update_data["position"]:
        update_data["positionX"] = update_data["position"].x
        update_data["positionY"] = update_data["position"].y
        del update_data["position"]

    for field, value in update_data.items():
        setattr(node, field, value)

    db.commit()
    db.refresh(node)

    return node


@router.delete("/nodes/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_node(
    node_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Delete node from pipeline."""
    node = (
        db.query(GenAINode)
        .join(GenAIPipeline)
        .filter(GenAINode.id == node_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")

    # Delete connected edges
    db.query(GenAIEdge).filter(
        (GenAIEdge.sourceNodeId == node_id) | (GenAIEdge.targetNodeId == node_id)
    ).delete()

    db.delete(node)
    db.commit()

    logger.info(f"Deleted node {node_id}")


# ========== Edge Endpoints ==========


@router.post(
    "/pipelines/{pipeline_id}/edges",
    response_model=EdgeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_edge(
    pipeline_id: int,
    edge_data: EdgeCreate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Create edge between nodes."""
    # Verify pipeline ownership
    pipeline = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == pipeline_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not pipeline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")

    # Verify nodes exist
    source_node = (
        db.query(GenAINode)
        .filter(GenAINode.id == edge_data.sourceNodeId, GenAINode.pipelineId == pipeline_id)
        .first()
    )

    target_node = (
        db.query(GenAINode)
        .filter(GenAINode.id == edge_data.targetNodeId, GenAINode.pipelineId == pipeline_id)
        .first()
    )

    if not source_node or not target_node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Source or target node not found"
        )

    try:
        edge = GenAIEdge(
            pipelineId=pipeline_id,
            sourceNodeId=edge_data.sourceNodeId,
            targetNodeId=edge_data.targetNodeId,
            sourceHandle=edge_data.sourceHandle,
            targetHandle=edge_data.targetHandle,
            label=edge_data.label,
            condition=edge_data.condition,
        )

        db.add(edge)
        db.commit()
        db.refresh(edge)

        logger.info(f"Created edge in pipeline {pipeline_id}")
        return edge

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create edge: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create edge: {str(e)}",
        )


@router.delete("/edges/{edge_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_edge(
    edge_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Delete edge."""
    edge = (
        db.query(GenAIEdge)
        .join(GenAIPipeline)
        .filter(GenAIEdge.id == edge_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not edge:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Edge not found")

    db.delete(edge)
    db.commit()


# ========== Execution Endpoints ==========


@router.post("/pipelines/{pipeline_id}/run", response_model=PipelineRunResponse)
async def run_pipeline(
    pipeline_id: int,
    run_data: RunPipelineRequest,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """
    Execute pipeline.

    - **inputData**: Initial input data
    - **startFromNodeId**: Resume from specific node (optional)
    - **config**: Runtime configuration overrides
    """
    # Verify pipeline ownership
    pipeline = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == pipeline_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not pipeline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")

    # Get nodes and edges
    nodes = db.query(GenAINode).filter(GenAINode.pipelineId == pipeline_id).all()
    edges = db.query(GenAIEdge).filter(GenAIEdge.pipelineId == pipeline_id).all()

    if not nodes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pipeline has no nodes")

    # Create run record
    import uuid

    run = GenAIPipelineRun(
        runId=str(uuid.uuid4()),
        pipelineId=pipeline_id,
        studentId=current_student.id,
        status=RunStatus.PENDING,
        inputData=run_data.inputData,
    )

    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        # Update status to running
        run.status = RunStatus.RUNNING
        run.startedAt = (
            db.query(GenAIPipelineRun).filter(GenAIPipelineRun.id == run.id).first().createdAt
        )  # Use created time as start
        db.commit()

        # Execute pipeline
        engine = GenAIPipelineEngine()

        nodes_data = [
            {"id": n.id, "nodeType": n.nodeType.value, "config": n.config, "isEnabled": n.isEnabled}
            for n in nodes
            if n.isEnabled
        ]

        edges_data = [
            {"sourceNodeId": e.sourceNodeId, "targetNodeId": e.targetNodeId} for e in edges
        ]

        result = await engine.execute_pipeline(
            nodes=nodes_data,
            edges=edges_data,
            input_data=run_data.inputData,
            start_from_node_id=run_data.startFromNodeId,
        )

        # Update run record
        run.status = RunStatus.COMPLETED if result["success"] else RunStatus.FAILED
        run.finalOutput = result.get("finalOutput")
        run.executionTimeMs = result.get("executionTimeMs")
        run.error = result.get("error")
        run.completedAt = (
            db.query(GenAIPipelineRun).filter(GenAIPipelineRun.id == run.id).first().updatedAt
        )

        # Save node executions
        for node_exec in result.get("nodeExecutions", []):
            execution = GenAINodeExecution(
                runId=run.id,
                nodeId=node_exec["nodeId"],
                status=RunStatus[node_exec["status"]],
                executionTimeMs=node_exec.get("executionTimeMs"),
                error=node_exec.get("error"),
            )
            db.add(execution)

        # Update pipeline lastRunAt
        pipeline.lastRunAt = run.completedAt

        db.commit()
        db.refresh(run)

        logger.info(f"Pipeline {pipeline_id} run {run.runId} completed")

        # Build response
        node_executions = [
            NodeExecutionResult(
                nodeId=ne["nodeId"],
                nodeType=ne["nodeType"],
                status=RunStatus[ne["status"]],
                inputData=None,
                outputData=None,
                executionTimeMs=ne.get("executionTimeMs"),
                error=ne.get("error"),
                provider=ne.get("provider"),
                model=ne.get("model"),
                startedAt=None,
                completedAt=None,
            )
            for ne in result.get("nodeExecutions", [])
        ]

        return PipelineRunResponse(
            id=run.id,
            runId=run.runId,
            pipelineId=run.pipelineId,
            status=run.status,
            finalOutput=run.finalOutput,
            executionTimeMs=run.executionTimeMs,
            error=run.error,
            startedAt=run.startedAt,
            completedAt=run.completedAt,
            nodeExecutions=node_executions,
        )

    except Exception as e:
        run.status = RunStatus.FAILED
        run.error = str(e)
        db.commit()

        logger.error(f"Pipeline execution failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(e)}",
        )


@router.get("/pipelines/{pipeline_id}/runs", response_model=List[PipelineRunListItem])
async def list_pipeline_runs(
    pipeline_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """List pipeline execution history."""
    # Verify ownership
    pipeline = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == pipeline_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not pipeline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")

    runs = (
        db.query(GenAIPipelineRun)
        .filter(GenAIPipelineRun.pipelineId == pipeline_id)
        .order_by(GenAIPipelineRun.createdAt.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return runs


@router.get("/runs/{run_id}", response_model=PipelineRunResponse)
async def get_run(
    run_id: str,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    """Get run details with node executions."""
    run = (
        db.query(GenAIPipelineRun)
        .join(GenAIPipeline)
        .filter(GenAIPipelineRun.runId == run_id, GenAIPipeline.studentId == current_student.id)
        .first()
    )

    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    # Get node executions
    executions = db.query(GenAINodeExecution).filter(GenAINodeExecution.runId == run.id).all()

    node_executions = [
        NodeExecutionResult(
            nodeId=e.nodeId,
            nodeType="",  # Would need to join with GenAINode
            status=e.status,
            inputData=None,
            outputData=None,
            executionTimeMs=e.executionTimeMs,
            error=e.error,
            provider=e.provider,
            model=e.model,
            startedAt=e.startedAt,
            completedAt=e.completedAt,
        )
        for e in executions
    ]

    return PipelineRunResponse(
        id=run.id,
        runId=run.runId,
        pipelineId=run.pipelineId,
        status=run.status,
        finalOutput=run.finalOutput,
        executionTimeMs=run.executionTimeMs,
        error=run.error,
        startedAt=run.startedAt,
        completedAt=run.completedAt,
        nodeExecutions=node_executions,
    )
