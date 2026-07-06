import runpod
import traceback
import logging
import json
import redis

from app.core.config import get_settings
from app.workers.tasks import _get_detector, _get_overview_processor
from app.workers.pipeline import run_pipeline_sync, run_fiber_overview_pipeline
from app.services.fiber_after import run_fiber_after_pipeline
from app.services.coax_before import run_coax_before_pipeline
from app.services.fiber_before import run_fiber_before_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyncRedisDict(dict):
    """A dictionary that automatically syncs its contents back to Upstash Redis when updated."""
    def __init__(self, redis_client, job_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis_client = redis_client
        self.job_id = job_id

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._sync()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._sync()

    def _sync(self):
        try:
            self.redis_client.set(self.job_id, json.dumps(self))
        except Exception as e:
            logger.error(f"Failed to sync job state to Redis: {e}")

def handler(event):
    """
    RunPod Serverless Handler.
    Receives the JSON payload via HTTP POST.
    """
    try:
        settings = get_settings()
        
        # 'input' is the standard RunPod wrapper for the JSON body payload
        job_record = event.get("input", {})
        job_id = job_record.get("job_id", "unknown_job")
        pipeline_type = job_record.get("pipeline_type", "coax")
        
        logger.info(f"Received RunPod job {job_id} for pipeline: {pipeline_type}")
        
        # Create a Redis client and wrap the job_record in our auto-syncing dictionary
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        job_record_dict = SyncRedisDict(redis_client, job_id, job_record)
        job_store = {job_id: job_record_dict}
        
        if pipeline_type == "coax":
            run_pipeline_sync(job_id, job_store, settings, detector=_get_detector(settings))
        elif pipeline_type == "fiber_overview":
            run_fiber_overview_pipeline(job_id, job_store, settings, processor=_get_overview_processor(settings))
        elif pipeline_type == "fiber_after":
            run_fiber_after_pipeline(job_id, job_store, settings)
        elif pipeline_type == "coax_before":
            run_coax_before_pipeline(job_id, job_store, settings)
        elif pipeline_type == "fiber_before":
            run_fiber_before_pipeline(job_id, job_store, settings)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
            
        final_job = job_store[job_id]
        
        # We return the exact status back to the RunPod API, where it is forwarded to the caller
        if final_job.get("status") == "COMPLETED" or final_job.get("status") == "completed":
            return {
                "status": "COMPLETED", 
                "job_id": job_id, 
                "report_path_gcs": final_job.get("report_path_gcs"),
                "callouts": final_job.get("callouts", [])
            }
        else:
            return {
                "status": "FAILED", 
                "job_id": job_id, 
                "message": final_job.get("message")
            }

    except Exception as e:
        logger.error(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
