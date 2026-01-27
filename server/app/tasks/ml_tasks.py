from app.core.celery_app import celery_app
from app.core.logging import logger
from app.services.pipeline_service import ml_service
import asyncio

@celery_app.task(name="train_model_process", bind=True)
def train_model_task(
    self,
    dataset_path: str,
    target_column: str,
    algorithm: str,
    task_type: str,
    hyperparameters: dict,
    test_ratio: float = 0.2,
    val_ratio: float = None,
):
    """
    Celery task to train ML model asynchronously.
    """
    try:
        logger.info(f"Starting background training task for {algorithm}")
        
        # Run async function in sync Celery worker
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            ml_service.train_model(
                dataset_path=dataset_path,
                target_column=target_column,
                algorithm=algorithm,
                task_type=task_type,
                hyperparameters=hyperparameters,
                test_ratio=test_ratio,
                val_ratio=val_ratio
            )
        )
        loop.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Training task failed: {str(e)}")
        # Re-raise to mark task as failed in Celery
        raise e
