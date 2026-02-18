"""
Custom Apps API endpoints.
CRUD for student-built UI pages + public view + pipeline execution.
"""

import re
import uuid
import time
import base64
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from app.db.session import get_db
from app.models.user import Student
from app.models.genai import GenAIPipeline, GenAINode, GenAIEdge
from app.models.custom_app import CustomApp
from app.schemas.custom_app import (
    CustomAppCreate,
    CustomAppUpdate,
    CustomAppPublish,
    CustomAppResponse,
    PublicAppResponse,
    PublicAppExecuteRequest,
    PublicAppExecuteResponse,
    SuggestedBlocksResponse,
)
from app.core.security import get_current_student
from app.core.logging import logger

router = APIRouter()


# ─── Helpers ──────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def _generate_unique_slug(db: Session, base_slug: str) -> str:
    """Generate a unique slug, appending a short hash if needed."""
    slug = base_slug 
    if db.query(CustomApp).filter(CustomApp.slug == slug).first():
        slug = f"{base_slug}-{uuid.uuid4().hex[:6]}"
    return slug


def _get_app_by_id(db: Session, app_id: int, user: Student) -> CustomApp:
    """Get app by ID and verify ownership."""
    app = db.query(CustomApp).filter(CustomApp.id == app_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Custom app not found")
    if app.studentId != user.id:
        raise HTTPException(status_code=403, detail="You don't own this app")
    return app


# ─── Node-to-Block Mapping Config ────────────────────────────────

# User-tweakable hyperparameters per ML node type
NODE_USER_FIELDS = {
    "linear_regression": [
        {"name": "fit_intercept", "label": "Fit Intercept", "type": "select",
         "options": ["true", "false"], "configKey": "fit_intercept"},
    ],
    "logistic_regression": [
        {"name": "C", "label": "Regularization (C)", "type": "number",
         "placeholder": "1.0", "configKey": "C"},
        {"name": "max_iter", "label": "Max Iterations", "type": "number",
         "placeholder": "1000", "configKey": "max_iter"},
    ],
    "decision_tree": [
        {"name": "max_depth", "label": "Max Depth", "type": "number",
         "placeholder": "None", "configKey": "max_depth"},
        {"name": "min_samples_split", "label": "Min Samples Split", "type": "number",
         "placeholder": "2", "configKey": "min_samples_split"},
    ],
    "random_forest": [
        {"name": "n_estimators", "label": "Number of Trees", "type": "number",
         "placeholder": "100", "configKey": "n_estimators"},
        {"name": "max_depth", "label": "Max Depth", "type": "number",
         "placeholder": "None", "configKey": "max_depth"},
    ],
}

# Metric node output keys
METRIC_OUTPUT_MAP = {
    "r2_score":         {"key": "r2_score",         "label": "R² Score",         "format": "percentage"},
    "mse_score":        {"key": "mse_score",        "label": "MSE",              "format": "number"},
    "rmse_score":       {"key": "rmse_score",       "label": "RMSE",             "format": "number"},
    "mae_score":        {"key": "mae_score",         "label": "MAE",             "format": "number"},
    "confusion_matrix": {"key": "confusion_matrix", "label": "Confusion Matrix", "format": "text"},
}

# Node types that are internal (not exposed to end users)
INTERNAL_NODE_TYPES = {
    "preprocess", "missing_value_handler", "encoding", "scaling",
    "feature_selection", "split", "sample_dataset", "select_dataset",
}


def _suggest_blocks_from_pipeline(nodes, edges, pipeline_name: str) -> list:
    """Analyze pipeline nodes and generate a recommended block layout."""
    blocks = []
    order = 0

    # GenAIEdge.sourceNodeId/targetNodeId are integer FKs to genai_nodes.id,
    # NOT the string nodeId field. Build a mapping to convert.
    id_to_nodeId = {n.id: n.nodeId for n in nodes}

    # Simple topological sort by in-degree (using string nodeIds)
    node_map = {n.nodeId: n for n in nodes}
    in_degree = {n.nodeId: 0 for n in nodes}
    for edge in edges:
        target_nid = id_to_nodeId.get(edge.targetNodeId)
        if target_nid and target_nid in in_degree:
            in_degree[target_nid] += 1

    from collections import deque
    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    sorted_nodes = []
    adj: dict = {}
    for edge in edges:
        src_nid = id_to_nodeId.get(edge.sourceNodeId)
        tgt_nid = id_to_nodeId.get(edge.targetNodeId)
        if src_nid and tgt_nid:
            adj.setdefault(src_nid, []).append(tgt_nid)
    while queue:
        nid = queue.popleft()
        if nid in node_map:
            sorted_nodes.append(node_map[nid])
        for child in adj.get(nid, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # 1. Hero block
    blocks.append({
        "id": "block_hero",
        "type": "hero",
        "config": {
            "title": pipeline_name,
            "subtitle": "Powered by machine learning",
            "alignment": "center",
            "showGradient": True,
        },
        "order": order,
    })
    order += 1

    # 2. Iterate sorted nodes
    metric_items = []

    for node in sorted_nodes:
        node_type = node.nodeType
        node_id = node.nodeId
        node_label = node.label or node_type.replace("_", " ").title()

        # Data source: upload_file → file_upload block
        if node_type == "upload_file":
            blocks.append({
                "id": f"block_{node_id}",
                "type": "file_upload",
                "config": {
                    "label": "Upload Dataset",
                    "acceptTypes": ".csv",
                    "maxSizeMB": 10,
                    "helpText": f"Upload a CSV file for {node_label}",
                    "nodeId": node_id,
                },
                "order": order,
                "nodeId": node_id,
                "nodeType": node_type,
                "nodeLabel": node_label,
            })
            order += 1

        # ML algorithm nodes → input_fields for hyperparams
        elif node_type in NODE_USER_FIELDS:
            user_fields = NODE_USER_FIELDS[node_type]
            fields = []
            field_mappings = []

            for uf in user_fields:
                fields.append({
                    "name": uf["name"],
                    "label": uf["label"],
                    "type": uf["type"],
                    "placeholder": uf.get("placeholder", ""),
                    "required": False,
                    "options": uf.get("options"),
                })
                field_mappings.append({
                    "fieldName": uf["name"],
                    "nodeId": node_id,
                    "nodeConfigKey": uf["configKey"],
                })

            if fields:
                blocks.append({
                    "id": f"block_{node_id}_inputs",
                    "type": "input_fields",
                    "config": {
                        "fields": fields,
                        "fieldMappings": field_mappings,
                    },
                    "order": order,
                    "nodeId": node_id,
                    "nodeType": node_type,
                    "nodeLabel": node_label,
                })
                order += 1

        # Metric nodes → collect for metrics_card
        elif node_type in METRIC_OUTPUT_MAP:
            metric_info = METRIC_OUTPUT_MAP[node_type]
            metric_items.append({
                "key": metric_info["key"],
                "label": metric_info["label"],
                "format": metric_info["format"],
                "nodeId": node_id,
                "nodeOutputKey": metric_info["key"],
            })

        # Internal nodes are skipped (preprocessing, split, etc.)

    # 3. Submit button
    blocks.append({
        "id": "block_submit",
        "type": "submit_button",
        "config": {
            "label": "Run Pipeline",
            "variant": "gradient",
            "loadingText": "Executing...",
        },
        "order": order,
    })
    order += 1

    # 4. Metrics card (if any metric nodes found)
    if metric_items:
        blocks.append({
            "id": "block_metrics",
            "type": "metrics_card",
            "config": {
                "title": "Model Metrics",
                "metrics": metric_items,
            },
            "order": order,
        })
        order += 1

    # 5. Generic results display
    blocks.append({
        "id": "block_results",
        "type": "results_display",
        "config": {
            "title": "Detailed Results",
            "displayMode": "card",
        },
        "order": order,
    })

    return blocks


# ─── CRUD Endpoints (authenticated) ──────────────────────────────

@router.post("", response_model=CustomAppResponse, status_code=status.HTTP_201_CREATED)
async def create_app(
    body: CustomAppCreate,
    db: Session = Depends(get_db),
    user: Student = Depends(get_current_student),
):
    """Create a new custom app linked to a pipeline."""
    # Verify pipeline ownership
    pipeline = db.query(GenAIPipeline).filter(GenAIPipeline.id == body.pipeline_id).first()
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    if pipeline.studentId != user.id:
        raise HTTPException(status_code=403, detail="You don't own this pipeline")

    slug = _generate_unique_slug(db, _slugify(body.name))

    app = CustomApp(
        studentId=user.id,
        pipelineId=body.pipeline_id,
        name=body.name,
        slug=slug,
        blocks=[],
        is_published=False,
    )
    db.add(app)
    db.commit()
    db.refresh(app)

    logger.info(f"Custom app created: {app.id} (slug={slug}) by student {user.id}")
    return app


@router.get("", response_model=List[CustomAppResponse])
async def list_apps(
    db: Session = Depends(get_db),
    user: Student = Depends(get_current_student),
):
    """List all custom apps for the current student."""
    apps = (
        db.query(CustomApp)
        .filter(CustomApp.studentId == user.id)
        .order_by(CustomApp.updatedAt.desc())
        .all()
    )
    return apps


@router.get("/{app_id}", response_model=CustomAppResponse)
async def get_app(
    app_id: int,
    db: Session = Depends(get_db),
    user: Student = Depends(get_current_student),
):
    """Get a custom app by ID (owner only)."""
    return _get_app_by_id(db, app_id, user)


@router.put("/{app_id}", response_model=CustomAppResponse)
async def update_app(
    app_id: int,
    body: CustomAppUpdate,
    db: Session = Depends(get_db),
    user: Student = Depends(get_current_student),
):
    """Update custom app fields (blocks, theme, name, slug, description)."""
    app = _get_app_by_id(db, app_id, user)

    if body.name is not None:
        app.name = body.name
    if body.description is not None:
        app.description = body.description
    if body.blocks is not None:
        app.blocks = body.blocks
    if body.theme is not None:
        app.theme = body.theme
    if body.slug is not None:
        # Check uniqueness
        existing = db.query(CustomApp).filter(
            CustomApp.slug == body.slug, CustomApp.id != app_id
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail="Slug already in use")
        app.slug = body.slug

    app.updatedAt = datetime.utcnow()
    db.commit()
    db.refresh(app)

    logger.info(f"Custom app updated: {app.id}")
    return app


@router.delete("/{app_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_app(
    app_id: int,
    db: Session = Depends(get_db),
    user: Student = Depends(get_current_student),
):
    """Delete a custom app."""
    app = _get_app_by_id(db, app_id, user)
    db.delete(app)
    db.commit()
    logger.info(f"Custom app deleted: {app_id}")


@router.post("/{app_id}/publish", response_model=CustomAppResponse)
async def publish_app(
    app_id: int,
    body: CustomAppPublish,
    db: Session = Depends(get_db),
    user: Student = Depends(get_current_student),
):
    """Publish or unpublish a custom app."""
    app = _get_app_by_id(db, app_id, user)

    if body.slug is not None:
        existing = db.query(CustomApp).filter(
            CustomApp.slug == body.slug, CustomApp.id != app_id
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail="Slug already in use")
        app.slug = body.slug

    app.is_published = body.is_published
    if body.is_published and not app.published_at:
        app.published_at = datetime.utcnow()
    elif not body.is_published:
        app.published_at = None

    app.updatedAt = datetime.utcnow()
    db.commit()
    db.refresh(app)

    action = "published" if body.is_published else "unpublished"
    logger.info(f"Custom app {action}: {app.id} (slug={app.slug})")
    return app


@router.get("/check-slug/{slug}")
async def check_slug(
    slug: str,
    db: Session = Depends(get_db),
    user: Student = Depends(get_current_student),
):
    """Check if a slug is available."""
    existing = db.query(CustomApp).filter(CustomApp.slug == slug).first()
    return {"available": existing is None, "slug": slug}


@router.get("/pipeline/{pipeline_id}/suggest-blocks", response_model=SuggestedBlocksResponse)
async def suggest_blocks(
    pipeline_id: int,
    db: Session = Depends(get_db),
    user: Student = Depends(get_current_student),
):
    """Analyze a pipeline and suggest UI blocks based on its nodes."""
    pipeline = db.query(GenAIPipeline).filter(GenAIPipeline.id == pipeline_id).first()
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    if pipeline.studentId != user.id:
        raise HTTPException(status_code=403, detail="You don't own this pipeline")

    nodes = db.query(GenAINode).filter(GenAINode.pipelineId == pipeline.id).all()
    edges = db.query(GenAIEdge).filter(GenAIEdge.pipelineId == pipeline.id).all()

    if not nodes:
        raise HTTPException(status_code=400, detail="Pipeline has no nodes")

    blocks = _suggest_blocks_from_pipeline(nodes, edges, pipeline.name or "My App")

    logger.info(f"Suggested {len(blocks)} blocks for pipeline {pipeline_id}")
    return SuggestedBlocksResponse(
        blocks=blocks,
        pipeline_name=pipeline.name or "My App",
        node_count=len(nodes),
    )


# ─── Public Endpoints (no auth) ──────────────────────────────────

@router.get("/public/{slug}", response_model=PublicAppResponse)
async def get_public_app(
    slug: str,
    db: Session = Depends(get_db),
):
    """Get a published app by slug (public access, no auth)."""
    app = db.query(CustomApp).filter(
        CustomApp.slug == slug,
        CustomApp.is_published == True,
    ).first()

    if not app:
        raise HTTPException(status_code=404, detail="App not found or not published")

    # Increment view count
    app.view_count = (app.view_count or 0) + 1
    db.commit()

    # Get owner name
    owner = db.query(Student).filter(Student.id == app.studentId).first()
    owner_name = owner.fullName if owner else "Unknown"

    return PublicAppResponse(
        name=app.name,
        description=app.description,
        blocks=app.blocks or [],
        theme=app.theme,
        owner_name=owner_name,
    )


@router.post("/public/{slug}/execute", response_model=PublicAppExecuteResponse)
async def execute_public_app(
    slug: str,
    body: PublicAppExecuteRequest,
    db: Session = Depends(get_db),
):
    """Execute a pipeline through a published custom app (public, no auth)."""
    app = db.query(CustomApp).filter(
        CustomApp.slug == slug,
        CustomApp.is_published == True,
    ).first()

    if not app:
        raise HTTPException(status_code=404, detail="App not found or not published")

    # Load the pipeline with nodes and edges
    pipeline = db.query(GenAIPipeline).filter(GenAIPipeline.id == app.pipelineId).first()
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    nodes = db.query(GenAINode).filter(GenAINode.pipelineId == pipeline.id).all()
    edges = db.query(GenAIEdge).filter(GenAIEdge.pipelineId == pipeline.id).all()

    if not nodes:
        raise HTTPException(status_code=400, detail="Pipeline has no nodes")

    try:
        from app.ml.engine import MLPipelineEngine

        engine = MLPipelineEngine()
        start_time = time.time()

        # GenAIEdge.sourceNodeId/targetNodeId are integer FKs to genai_nodes.id,
        # NOT the string nodeId field. Build a mapping to convert.
        id_to_nodeId = {n.id: n.nodeId for n in nodes}

        # Build pipeline config from stored nodes/edges
        pipeline_nodes = []
        for node in nodes:
            node_input = dict(node.config or {})

            # Node-routed data: only send mapped fields to each node
            if body.node_inputs and node.nodeId in body.node_inputs:
                node_input.update(body.node_inputs[node.nodeId])
            elif body.node_inputs is None and body.input_data:
                # Legacy fallback: dump all data into all nodes
                node_input.update(body.input_data)

            # Handle file upload — route to mapped upload node
            if body.file_data:
                target_node = body.file_node_id or (
                    node.nodeId if node.nodeType == "upload_file" else None
                )
                if target_node == node.nodeId:
                    node_input["file_content"] = base64.b64decode(body.file_data)
                    node_input["filename"] = body.input_data.get("filename", "upload.csv")

            pipeline_nodes.append({
                "node_id": node.nodeId,
                "node_type": node.nodeType,
                "input": node_input,
                "label": node.label or node.nodeType,
            })

        # Convert integer FK IDs to string nodeIds for the engine
        pipeline_edges = [
            {
                "source": id_to_nodeId.get(edge.sourceNodeId, ""),
                "target": id_to_nodeId.get(edge.targetNodeId, ""),
                "sourceHandle": edge.sourceHandle,
                "targetHandle": edge.targetHandle,
            }
            for edge in edges
            if edge.sourceNodeId in id_to_nodeId and edge.targetNodeId in id_to_nodeId
        ]

        result = await engine.execute_pipeline(
            nodes=pipeline_nodes,
            edges=pipeline_edges,
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Increment execution count
        app.execution_count = (app.execution_count or 0) + 1
        db.commit()

        if result.get("success"):
            # Build node-keyed results for targeted block display
            node_results = {}
            for r in result.get("results", []):
                nid = r.get("node_id")
                if nid:
                    node_results[nid] = r

            return PublicAppExecuteResponse(
                success=True,
                results={**result, "node_results": node_results},
                execution_time_ms=execution_time_ms,
            )
        else:
            return PublicAppExecuteResponse(
                success=False,
                error=result.get("error", "Pipeline execution failed"),
                execution_time_ms=execution_time_ms,
            )

    except Exception as e:
        logger.error(f"Public app execution failed for slug={slug}: {str(e)}", exc_info=True)
        return PublicAppExecuteResponse(
            success=False,
            error=str(e),
        )
