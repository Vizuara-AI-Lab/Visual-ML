"""
Celery tasks for async pipeline and node execution.
All lengthy ML tasks (preprocessing, training, feature engineering) run here.
"""

from app.core.celery_app import celery_app
from app.core.logging import logger
from app.services.pipeline_service import ml_service
from app.ml.pipeline_engine import MLPipelineEngine
import asyncio
from typing import Dict, Any, List
import traceback


@celery_app.task(name="execute_pipeline_async", bind=True)
def execute_pipeline_task(
    self,
    pipeline: List[Dict[str, Any]],
    pipeline_name: str = "Pipeline",
    dry_run: bool = False,
    current_user: Dict[str, Any] = None,
):
    """
    Execute a complete ML pipeline asynchronously.

    This task handles:
    - Data upload and validation
    - Preprocessing (missing values, encoding, transformation, scaling)
    - Feature engineering (feature selection)
    - Train/test splitting
    - Model training (linear regression, logistic regression, decision tree, random forest)
    - Model evaluation

    Args:
        pipeline: List of node configurations
        pipeline_name: Name of the pipeline
        dry_run: If True, validate only without executing
        current_user: User context for authentication

    Returns:
        dict: Execution results with node outputs
    """
    try:
        # Update task state to show progress
        self.update_state(
            state="PROGRESS",
            meta={
                "status": "Starting pipeline execution...",
                "current_node": 0,
                "total_nodes": len(pipeline),
                "percent": 0,
            },
        )

        logger.info(
            f"🚀 Starting async pipeline execution: {pipeline_name} ({len(pipeline)} nodes)"
        )

        # Run async function in sync Celery worker
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        results = []

        for idx, node_config in enumerate(pipeline):
            node_type = node_config.get("node_type")

            # Update progress
            percent = int((idx / len(pipeline)) * 90)  # Reserve 10% for final processing
            self.update_state(
                state="PROGRESS",
                meta={
                    "status": f"Executing node: {node_type}",
                    "current_node": idx + 1,
                    "total_nodes": len(pipeline),
                    "percent": percent,
                    "node_type": node_type,
                },
            )

            logger.info(f"📋 Executing node {idx + 1}/{len(pipeline)}: {node_type}")

            # Execute node
            node_result = loop.run_until_complete(
                ml_service.execute_node(
                    node_type=node_type,
                    input_data=node_config.get("input", {}),
                    dry_run=dry_run,
                    current_user=current_user,
                )
            )

            results.append(node_result)

            # If node failed and not in dry run, stop execution
            if not node_result.get("success", True) and not dry_run:
                logger.error(f"❌ Node {node_type} failed: {node_result.get('error')}")
                break

        loop.close()

        # Final progress update
        self.update_state(
            state="PROGRESS",
            meta={
                "status": "Finalizing results...",
                "current_node": len(pipeline),
                "total_nodes": len(pipeline),
                "percent": 95,
            },
        )

        success = all(r.get("success", True) for r in results)

        logger.info(f"✅ Pipeline execution completed. Success: {success}")

        return {
            "success": success,
            "pipeline_name": pipeline_name,
            "results": results,
            "total_nodes": len(pipeline),
            "completed_nodes": len(results),
        }

    except Exception as e:
        logger.error(f"❌ Pipeline execution task failed: {str(e)}")
        logger.error(traceback.format_exc())

        # Re-raise to mark task as failed
        self.update_state(
            state="FAILURE",
            meta={
                "status": "Pipeline execution failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        raise e


@celery_app.task(name="execute_node_async", bind=True)
def execute_node_task(
    self,
    node_type: str,
    input_data: Dict[str, Any],
    dry_run: bool = False,
    current_user: Dict[str, Any] = None,
):
    """
    Execute a single ML node asynchronously.

    Supports all node types:
    - upload_file: Dataset upload and validation
    - missing_value_handler: Handle missing data
    - encoding: Categorical encoding (one-hot, label)
    - transformation: Feature transformations (log, sqrt, polynomial - removed)
    - scaling: Feature scaling (standard, minmax, robust)
    - feature_selection: Feature selection (Chi-Square, F-Score, RFE)
    - split: Train/test split
    - linear_regression: Linear regression training
    - logistic_regression: Logistic regression training
    - decision_tree: Decision tree training
    - random_forest: Random forest training

    Args:
        node_type: Type of node to execute
        input_data: Node input configuration
        dry_run: If True, validate only without executing
        current_user: User context for authentication

    Returns:
        dict: Node execution result
    """
    try:
        self.update_state(
            state="PROGRESS",
            meta={"status": f"Executing {node_type}...", "node_type": node_type, "percent": 50},
        )

        logger.info(f"🔧 Executing async node: {node_type}")

        # Run async function in sync Celery worker
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            ml_service.execute_node(
                node_type=node_type,
                input_data=input_data,
                dry_run=dry_run,
                current_user=current_user,
            )
        )

        loop.close()

        logger.info(f"✅ Node {node_type} execution completed")

        return result

    except Exception as e:
        logger.error(f"❌ Node execution task failed: {str(e)}")
        logger.error(traceback.format_exc())

        self.update_state(
            state="FAILURE",
            meta={
                "status": f"{node_type} execution failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        raise e


@celery_app.task(name="train_model_async", bind=True)
def train_model_async_task(
    self,
    train_dataset_id: str,
    target_column: str,
    algorithm: str,
    hyperparameters: Dict[str, Any],
    test_dataset_id: str = None,
):
    """
    Train ML model asynchronously.

    This is specifically for ML algorithm nodes:
    - linear_regression
    - logistic_regression
    - decision_tree
    - random_forest

    Args:
        train_dataset_id: Training dataset ID
        target_column: Target column name
        algorithm: Algorithm type
        hyperparameters: Model hyperparameters
        test_dataset_id: Optional test dataset ID for evaluation

    Returns:
        dict: Training results with metrics and model path
    """
    try:
        self.update_state(
            state="PROGRESS",
            meta={"status": f"Loading training data...", "algorithm": algorithm, "percent": 10},
        )

        logger.info(f"🎯 Training {algorithm} model asynchronously")

        # Import the appropriate node
        from app.ml.nodes.linear_regression_node import LinearRegressionNode
        from app.ml.nodes.logistic_regression_node import LogisticRegressionNode
        from app.ml.nodes.decision_tree_node import DecisionTreeNode
        from app.ml.nodes.random_forest_node import RandomForestNode

        node_map = {
            "linear_regression": LinearRegressionNode,
            "logistic_regression": LogisticRegressionNode,
            "decision_tree": DecisionTreeNode,
            "random_forest": RandomForestNode,
        }

        NodeClass = node_map.get(algorithm)
        if not NodeClass:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.update_state(
            state="PROGRESS",
            meta={
                "status": f"Training {algorithm} model...",
                "algorithm": algorithm,
                "percent": 30,
            },
        )

        # Create node instance and execute
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        node_instance = NodeClass()

        # Build input config
        input_config = {
            "train_dataset_id": train_dataset_id,
            "target_column": target_column,
            **hyperparameters,
        }

        if test_dataset_id:
            input_config["test_dataset_id"] = test_dataset_id

        result = loop.run_until_complete(node_instance.execute(input_config))

        loop.close()

        self.update_state(
            state="PROGRESS",
            meta={
                "status": f"Training completed, saving model...",
                "algorithm": algorithm,
                "percent": 95,
            },
        )

        logger.info(f"✅ Model training completed: {algorithm}")

        return result.model_dump() if hasattr(result, "model_dump") else result

    except Exception as e:
        logger.error(f"❌ Model training task failed: {str(e)}")
        logger.error(traceback.format_exc())

        self.update_state(
            state="FAILURE",
            meta={
                "status": f"{algorithm} training failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        raise e
