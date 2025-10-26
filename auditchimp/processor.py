"""Background task processor for audio transcription and diarization."""

import asyncio
import logging
import uuid
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from pathlib import Path
import psutil
import torch
from concurrent.futures import ThreadPoolExecutor

from .models import TranscriptionRequest, ProcessingStatus, ProcessingMetrics
from .config import config
from .diarization import get_diarization_engine
from .transcription_manager import get_transcription_manager

logger = logging.getLogger(__name__)


class JobProcessor:
    """Independent job processor that runs in separate thread."""

    def __init__(self, db_session):
        self.db_session = db_session
        self.diarization_engine = None
        self.transcription_manager = None  # Changed from transcription_engine
        self.is_running = False
        self.processing_requests = set()
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_JOBS, thread_name_prefix="JobProcessor")
        self._lock = threading.Lock()

    async def initialize(self):
        """Initialize processing engines."""
        logger.info("Initializing job processor...")

        # Initialize engines
        self.diarization_engine = await get_diarization_engine(config)
        self.transcription_manager = await get_transcription_manager(config)

        logger.info("Job processor initialized successfully")

    def start(self):
        """Start the job processor in a separate thread."""
        if self.is_running:
            logger.warning("Job processor already running")
            return

        self.is_running = True
        logger.info("Starting job processor thread...")

        # Start processing loop in separate thread
        self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processor_thread.start()

    def stop(self):
        """Stop the job processor."""
        logger.info("Stopping job processor...")
        self.is_running = False

        # Wait for thread to finish
        if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)

        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("Job processor stopped")

    def _processing_loop(self):
        """Main processing loop that runs in separate thread."""
        logger.info("Job processor loop started")

        while self.is_running:
            try:
                # Check for new jobs
                self._process_jobs()

                # Sleep between checks
                time.sleep(2)

            except Exception as e:
                logger.error(f"Error in job processing loop: {e}")
                time.sleep(5)  # Longer delay on error

        logger.info("Job processor loop ended")

    def _process_jobs(self):
        """Process available jobs."""
        try:
            # Get queued requests (limit by concurrent jobs)
            with self._lock:
                queued_requests = self.db_session.query(TranscriptionRequest)\
                    .filter(TranscriptionRequest.status == ProcessingStatus.QUEUED)\
                    .order_by(TranscriptionRequest.created_at)\
                    .limit(config.MAX_CONCURRENT_JOBS - len(self.processing_requests))\
                    .all()

            if not queued_requests:
                return

            logger.info(f"Found {len(queued_requests)} queued requests to process")

            # Process each request
            for request in queued_requests:
                if len(self.processing_requests) >= config.MAX_CONCURRENT_JOBS:
                    break

                self._start_job(request)

        except Exception as e:
            logger.error(f"Error processing jobs: {e}")

    def _start_job(self, request: TranscriptionRequest):
        """Start processing a job in thread pool."""
        try:
            # Mark as processing
            with self._lock:
                request.status = ProcessingStatus.PROCESSING
                request.started_at = datetime.utcnow()
                self.db_session.commit()

            # Add to processing set
            self.processing_requests.add(request.request_id)

            logger.info(f"Starting job {request.request_id}")

            # Submit to thread pool
            future = self.executor.submit(self._process_job_sync, request)

            # Handle completion
            future.add_done_callback(
                lambda f, req=request: self._handle_job_complete(req, f)
            )

        except Exception as e:
            logger.error(f"Error starting job {request.request_id}: {e}")
            with self._lock:
                request.status = ProcessingStatus.FAILED
                request.error_message = str(e)
                request.completed_at = datetime.utcnow()
                self.db_session.commit()

    def _process_job_sync(self, request: TranscriptionRequest):
        """Process a job synchronously in thread pool."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create progress callback
            progress_callback = ProgressCallback(
                lambda p, s, req=request: self._update_progress_sync(req, p, s)
            )

            # Step 1: Diarization
            loop.run_until_complete(progress_callback.update(10, "Starting diarization..."))
            with self._lock:
                request.status = ProcessingStatus.DIARIZING
                self.db_session.commit()

            # Check if diarization engine is properly initialized
            if self.diarization_engine is None:
                raise RuntimeError("Diarization engine not initialized")

            # Run NeMo diarization
            diarization_result = self.diarization_engine._diarize_nemo_sync(request.file_path)
            with self._lock:
                request.diarization_result = json.dumps(diarization_result)
                request.diarization_method = diarization_result.get("method", "unknown")
                self.db_session.commit()

            loop.run_until_complete(progress_callback.update(50, "Diarization completed, starting transcription..."))

            # Step 2: Transcription
            with self._lock:
                request.status = ProcessingStatus.TRANSCRIBING
                self.db_session.commit()

            # Check if transcription manager is properly initialized
            if self.transcription_manager is None:
                raise RuntimeError("Transcription manager not initialized")

            # Transcribe using the manager (routes to appropriate engine)
            transcription_result = loop.run_until_complete(
                self.transcription_manager.transcribe(
                    audio_path=request.file_path,
                    source_language=request.source_language or "auto",
                    target_language=request.target_language or "en",
                    accurate_mode=request.accurate_mode,
                    diarization_segments=diarization_result.get("segments", [])
                )
            )

            # Store results and metadata
            with self._lock:
                request.transcription_result = json.dumps(transcription_result)
                request.whisper_model = config.WHISPER_MODEL
                request.detected_language = transcription_result.get("detected_language")
                request.detected_language_name = transcription_result.get("detected_language_name")
                request.transcription_engine = transcription_result.get("engine")
                request.decoder_type = transcription_result.get("decoder", "unknown")
                request.translation_enabled = transcription_result.get("translation_enabled", False)
                self.db_session.commit()

            # Step 3: Finalize
            loop.run_until_complete(progress_callback.update(100, "Processing completed"))
            with self._lock:
                request.status = ProcessingStatus.COMPLETED
                request.completed_at = datetime.utcnow()

                # Calculate processing time
                if request.started_at:
                    processing_time = (request.completed_at - request.started_at).total_seconds()
                    request.processing_time = processing_time

                self.db_session.commit()

            logger.info(f"Successfully completed job {request.request_id}")

        except Exception as e:
            logger.error(f"Error processing job {request.request_id}: {e}")
            with self._lock:
                request.status = ProcessingStatus.FAILED
                request.error_message = str(e)
                request.completed_at = datetime.utcnow()
                self.db_session.commit()
            raise
        finally:
            loop.close()

    def _handle_job_complete(self, request: TranscriptionRequest, future):
        """Handle job completion."""
        try:
            # Remove from processing set
            self.processing_requests.discard(request.request_id)

            # Get result or exception
            result = future.result()

        except Exception as e:
            logger.error(f"Job failed for request {request.request_id}: {e}")

    def _update_progress_sync(self, request: TranscriptionRequest, progress: float, status: str = None):
        """Update progress from sync context."""
        try:
            with self._lock:
                request.progress = min(progress, 100.0)
                if status:
                    # Update status based on progress ranges
                    if progress < 25:
                        request.status = ProcessingStatus.PROCESSING
                    elif progress < 75:
                        request.status = ProcessingStatus.DIARIZING
                    else:
                        request.status = ProcessingStatus.TRANSCRIBING

                self.db_session.commit()
            logger.debug(f"Progress updated for {request.request_id}: {progress}% - {status}")
        except Exception as e:
            logger.error(f"Error updating progress: {e}")


class ProgressCallback:
    """Callback for progress updates."""

    def __init__(self, update_func: Callable):
        self.update_func = update_func

    async def update(self, progress: float, status: str = None):
        """Update progress."""
        # Call the sync update function directly (no await needed)
        self.update_func(progress, status)


class AudioProcessor:
    """Main audio processing orchestrator - API interface only."""

    def __init__(self, db_session):
        self.db_session = db_session
        self.job_processor = None

    @property
    def is_processing(self) -> bool:
        """Check if the processor is currently running."""
        return self.job_processor.is_running if self.job_processor else False

    @property
    def transcription_engine(self):
        """Backward compatibility property for health check."""
        return self.job_processor.transcription_manager if self.job_processor else None

    async def initialize(self):
        """Initialize processing engines."""
        logger.info("Initializing audio processor...")

        # Create job processor
        self.job_processor = JobProcessor(self.db_session)
        await self.job_processor.initialize()

        # Start job processor in separate thread
        self.job_processor.start()

        logger.info("Audio processor initialized successfully")

    async def stop_processing_loop(self):
        """Stop the processing loop."""
        if self.job_processor:
            self.job_processor.stop()

    async def get_queue_status(self) -> Dict:
        """Get current queue status."""
        try:
            queued = self.db_session.query(TranscriptionRequest)\
                .filter(TranscriptionRequest.status == ProcessingStatus.QUEUED).count()

            processing = self.db_session.query(TranscriptionRequest)\
                .filter(TranscriptionRequest.status == ProcessingStatus.PROCESSING).count()

            completed_today = self.db_session.query(TranscriptionRequest)\
                .filter(
                    TranscriptionRequest.status == ProcessingStatus.COMPLETED,
                    TranscriptionRequest.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                ).count()

            failed_today = self.db_session.query(TranscriptionRequest)\
                .filter(
                    TranscriptionRequest.status == ProcessingStatus.FAILED,
                    TranscriptionRequest.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                ).count()

            return {
                "queued": queued,
                "processing": processing,
                "completed_today": completed_today,
                "failed_today": failed_today
            }
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {"error": str(e)}

    async def get_system_metrics(self) -> Dict:
        """Get system performance metrics."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # GPU usage if available
            gpu_usage = None
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()

            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "gpu_usage": gpu_usage,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}


# Global processor instance
processor: Optional[AudioProcessor] = None


async def get_processor(db_session) -> AudioProcessor:
    """Get or create processor instance."""
    global processor
    if processor is None:
        processor = AudioProcessor(db_session)
        await processor.initialize()
    return processor
