"""Background task processor for audio transcription and diarization."""
import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from pathlib import Path
import psutil
import torch
from concurrent.futures import ThreadPoolExecutor

from .models import TranscriptionRequest, ProcessingStatus, ProcessingMetrics
from .config import config
from .diarization import get_diarization_engine
from .transcription_manager import get_transcription_manager
from .logger import log_event
from .utils import safe_delete_audio_file

logger = logging.getLogger(__name__)


class JobProcessor:
    """Independent job processor that runs in separate thread."""

    def __init__(self, session_factory):
        self.session_factory = session_factory  # Store factory, not session
        self.diarization_engine = None
        self.transcription_manager = None  # Changed from transcription_engine
        self.is_running = False
        self.processing_requests = set()
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_JOBS, thread_name_prefix="JobProcessor")
        self._lock = threading.Lock()
        self._job_available_event = threading.Event()  # Event to wake up processor when jobs complete

    async def initialize(self):
        """Initialize processing engines."""
        log_event(logger, "info", "processor_init_started", "Job processor initialization started")

        self.diarization_engine = await get_diarization_engine(config)
        log_event(logger, "success", "diarization_engine_loaded", "Diarization engine loaded")

        self.transcription_manager = await get_transcription_manager(config)
        log_event(logger, "success", "transcription_manager_loaded", "Transcription manager loaded")

        log_event(logger, "success", "processor_init_completed", "Job processor initialized")

    def start(self):
        """Start the job processor in a separate thread."""
        if self.is_running:
            log_event(logger, "warning", "processor_already_running", "Job processor already running")
            return

        self.is_running = True
        self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processor_thread.start()

        log_event(logger, "success", "processor_started", "Job processor thread started")

    def stop(self):
        """Stop the job processor."""
        log_event(logger, "info", "processor_stopping", "Job processor stopping")
        self.is_running = False

        # Wait for thread to finish
        if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)

        # Shutdown executor
        self.executor.shutdown(wait=True)
        log_event(logger, "success", "processor_stopped", "Job processor stopped")

    def _processing_loop(self):
        """Main processing loop that runs in separate thread."""
        log_event(logger, "info", "processing_loop_started", "Job processing loop started")

        while self.is_running:
            try:
                # Check for timed-out jobs first
                self._check_for_timeouts()

                # Then process new jobs
                self._process_jobs()

                # Wait for either timeout (0.5s) or job completion event
                # This ensures immediate pickup when a job completes
                self._job_available_event.wait(timeout=0.5)
                self._job_available_event.clear()  # Reset event for next iteration

            except Exception as e:
                log_event(logger, "error", "processing_loop_error", "Job processing loop error", error=str(e))
                time.sleep(5)

        log_event(logger, "info", "processing_loop_stopped", "Job processing loop stopped")

    def _check_for_timeouts(self):
        """Check for jobs that have exceeded the 10-minute timeout."""
        db_session = self.session_factory()
        try:
            # Calculate timeout threshold (10 minutes ago)
            timeout_threshold = datetime.utcnow() - timedelta(minutes=10)

            # Query for stuck jobs (processing for more than 10 minutes)
            stuck_jobs = db_session.query(TranscriptionRequest)\
                .filter(
                    TranscriptionRequest.status.in_([
                        ProcessingStatus.DIARIZING,
                        ProcessingStatus.TRANSCRIBING
                    ])
                )\
                .filter(TranscriptionRequest.started_at < timeout_threshold)\
                .all()

            if not stuck_jobs:
                return

            # Process each stuck job
            for request in stuck_jobs:
                log_event(
                    logger, "warning", "job_timeout_detected", "Job exceeded 10-minute timeout",
                    request_id=request.request_id,
                    status=request.status.value,
                    started_at=request.started_at.isoformat() if request.started_at else None,
                    elapsed_minutes=round((datetime.utcnow() - request.started_at).total_seconds() / 60, 2) if request.started_at else None
                )

                # Remove from processing set
                with self._lock:
                    self.processing_requests.discard(request.request_id)

                    # Update status to FAILED with timeout message
                    request.status = ProcessingStatus.FAILED
                    request.error_message = "Job timeout: Processing exceeded 10 minutes"
                    request.completed_at = datetime.utcnow()

                    # Calculate processing time
                    if request.started_at:
                        request.processing_time = (request.completed_at - request.started_at).total_seconds()

                    db_session.commit()

                # Clean up audio file
                safe_delete_audio_file(
                    file_path=request.file_path,
                    request_id=request.request_id,
                    context="job_timeout"
                )

                log_event(
                    logger, "info", "job_timeout_cleaned", "Timed-out job cleaned up",
                    request_id=request.request_id
                )

        except Exception as e:
            log_event(logger, "error", "timeout_check_failed", "Timeout check failed", error=str(e))
        finally:
            db_session.close()

    def _process_jobs(self):
        """Process available jobs."""
        # Create a new session for this thread
        db_session = self.session_factory()
        try:
            # Get queued requests (limit by concurrent jobs)
            with self._lock:
                queued_requests = db_session.query(TranscriptionRequest)\
                    .filter(TranscriptionRequest.status == ProcessingStatus.QUEUED)\
                    .order_by(TranscriptionRequest.created_at)\
                    .limit(config.MAX_CONCURRENT_JOBS - len(self.processing_requests))\
                    .all()

            if not queued_requests:
                return

            # Process each request
            for request in queued_requests:
                if len(self.processing_requests) >= config.MAX_CONCURRENT_JOBS:
                    break

                self._start_job(request, db_session)

        except Exception as e:
            log_event(logger, "error", "job_processing_error", "Job processing error", error=str(e))
        finally:
            db_session.close()

    def _start_job(self, request: TranscriptionRequest, db_session):
        """Start processing a job in thread pool."""
        try:
            # Mark as processing
            with self._lock:
                request.status = ProcessingStatus.DIARIZING
                request.started_at = datetime.utcnow()
                db_session.commit()

            # Add to processing set
            self.processing_requests.add(request.request_id)

            log_event(
                logger, "info", "job_started", "Job processing started",
                request_id=request.request_id,
                filename=request.filename,
                language=request.source_language,
                size_mb=round(request.file_size / 1024 / 1024, 2)
            )

            # Submit to thread pool - pass request_id instead of request object
            future = self.executor.submit(self._process_job_sync, request.request_id)

            # Handle completion
            future.add_done_callback(
                lambda f, req_id=request.request_id: self._handle_job_complete(req_id, f)
            )

        except Exception as e:
            log_event(
                logger, "error", "job_start_failed", "Job start failed",
                request_id=request.request_id,
                error=str(e)
            )
            with self._lock:
                request.status = ProcessingStatus.FAILED
                request.error_message = str(e)
                request.completed_at = datetime.utcnow()
                db_session.commit()


    def _process_job_sync(self, request_id: str):
        """Process a job synchronously in thread pool."""
        # Create a new session for this thread
        db_session = self.session_factory()
        file_path_to_cleanup = None  # Track file for cleanup

        try:
            # Fetch the request in this thread's session
            request = db_session.query(TranscriptionRequest)\
                .filter(TranscriptionRequest.request_id == request_id)\
                .first()

            if not request:
                raise RuntimeError(f"Request {request_id} not found")

            # Store file path for cleanup in finally block
            file_path_to_cleanup = request.file_path

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create progress callback
            progress_callback = ProgressCallback(
                lambda p, s, req=request: self._update_progress_sync(req, p, s)
            )

            # Step 1: Diarization
            log_event(
                logger, "info", "diarization_started", "Diarization phase started",
                request_id=request.request_id
            )

            loop.run_until_complete(progress_callback.update(10, "Starting diarization..."))
            with self._lock:
                request.status = ProcessingStatus.DIARIZING
                db_session.commit()

            # Check if diarization engine is properly initialized
            if self.diarization_engine is None:
                raise RuntimeError("Diarization engine not initialized")

            # Run NeMo diarization with timing
            diarization_start = time.time()
            diarization_result = self.diarization_engine._diarize_nemo_sync(request.file_path)
            diarization_time = time.time() - diarization_start

            # Log diarization results
            log_event(
                logger, "success", "diarization_completed", "Diarization completed",
                request_id=request.request_id,
                duration_sec=round(diarization_time, 2),
                method=diarization_result.get('method'),
                num_speakers=diarization_result.get('num_speakers'),
                segments=len(diarization_result.get('segments', [])),
                vad_segments=len(diarization_result.get('vad_segments', [])),
                overlaps=len(diarization_result.get('overlaps', []))
            )

            with self._lock:
                request.diarization_result = json.dumps(diarization_result)
                request.diarization_method = diarization_result.get("method", "unknown")
                request.diarization_time = diarization_time
                db_session.commit()

            loop.run_until_complete(progress_callback.update(50, "Diarization completed, starting transcription..."))

            # Step 2: Transcription
            log_event(
                logger, "info", "transcription_started", "Transcription phase started",
                request_id=request.request_id,
                language=request.source_language
            )

            with self._lock:
                request.status = ProcessingStatus.TRANSCRIBING
                db_session.commit()

            # Check if transcription manager is properly initialized
            if self.transcription_manager is None:
                raise RuntimeError("Transcription manager not initialized")

            # Transcribe using the manager with timing (routes to appropriate engine)
            transcription_start = time.time()
            transcription_result = loop.run_until_complete(
                self.transcription_manager.transcribe(
                    audio_path=request.file_path,
                    source_language=request.source_language or "auto",
                    diarization_segments=diarization_result.get("segments", []),
                    vad_segments=diarization_result.get("vad_segments", []),
                    overlaps=diarization_result.get("overlaps", [])
                )
            )
            transcription_time = time.time() - transcription_start

            # Log transcription results
            log_event(
                logger, "success", "transcription_completed", "Transcription completed",
                request_id=request.request_id,
                duration_sec=round(transcription_time, 2),
                engine=transcription_result.get('engine'),
                segments=len(transcription_result.get('segments', [])),
                total_words=transcription_result.get('total_words', 0)
            )

            # Store results and metadata
            with self._lock:
                request.transcription_result = json.dumps(transcription_result)
                request.whisper_model = config.MODEL_NAME
                request.output_language = transcription_result.get("output_language")
                request.detected_language = transcription_result.get("detected_language")
                request.detected_language_name = transcription_result.get("detected_language_name")
                request.transcription_engine = transcription_result.get("engine")
                request.decoder_type = transcription_result.get("decoder", "unknown")
                request.translation_enabled = transcription_result.get("translation_enabled", False)
                request.transcription_time = transcription_time
                request.duration = transcription_result.get("duration")
                db_session.commit()

            # Step 3: Finalize
            loop.run_until_complete(progress_callback.update(100, "Processing completed"))
            with self._lock:
                request.status = ProcessingStatus.COMPLETED
                request.completed_at = datetime.utcnow()

                # Calculate processing time
                if request.started_at:
                    processing_time = (request.completed_at - request.started_at).total_seconds()
                    request.processing_time = processing_time

                db_session.commit()

            log_event(
                logger, "success", "job_completed", "Job completed successfully",
                request_id=request.request_id,
                total_time_sec=round(request.processing_time, 2),
                diarization_time_sec=round(diarization_time, 2),
                transcription_time_sec=round(transcription_time, 2)
            )

            # Cleanup audio file after successful completion
            if file_path_to_cleanup:
                safe_delete_audio_file(
                    file_path=file_path_to_cleanup,
                    request_id=request_id,
                    context="job_completion"
                )

        except Exception as e:
            log_event(
                logger, "error", "job_failed", "Job processing failed",
                request_id=request_id,
                error=str(e)
            )
            try:
                # Refetch request in case of error (might be detached)
                request = db_session.query(TranscriptionRequest)\
                    .filter(TranscriptionRequest.request_id == request_id)\
                    .first()
                if request:
                    with self._lock:
                        request.status = ProcessingStatus.FAILED
                        request.error_message = str(e)
                        request.completed_at = datetime.utcnow()
                        db_session.commit()
            except Exception as db_error:
                logger.error(f"Failed to update error status for {request_id}: {db_error}")

            # Cleanup audio file even on failure
            if file_path_to_cleanup:
                safe_delete_audio_file(
                    file_path=file_path_to_cleanup,
                    request_id=request_id,
                    context="job_failure"
                )

            raise
        finally:
            loop.close()
            db_session.close()

    def _handle_job_complete(self, request_id: str, future):
        """Handle job completion."""
        try:
            # Remove from processing set
            self.processing_requests.discard(request_id)

            # Get result or exception
            result = future.result()

            # Signal that a job slot is now available - wake up processor immediately
            self._job_available_event.set()

        except Exception as e:
            logger.error(f"Job failed for request {request_id}: {e}")
            # Still signal job completion even on failure
            self._job_available_event.set()

    def _update_progress_sync(self, request: TranscriptionRequest, progress: float, status: str = None):
        """Update progress from sync context."""
        # Note: This is called from within _process_job_sync which already has a session
        # We just update the request object in memory - commits happen in the main flow
        try:
            request.progress = min(progress, 100.0)
            if status:
                # Update status based on progress ranges
                if progress < 75:
                    request.status = ProcessingStatus.DIARIZING
                else:
                    request.status = ProcessingStatus.TRANSCRIBING

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

    def __init__(self, db_session, session_factory):
        self.db_session = db_session  # Keep for API queries
        self.session_factory = session_factory  # Pass to job processor
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

        # Create job processor with session factory
        self.job_processor = JobProcessor(self.session_factory)
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

            diarizing = self.db_session.query(TranscriptionRequest)\
                .filter(TranscriptionRequest.status == ProcessingStatus.DIARIZING).count()

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
                "diarizing": diarizing,
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


async def get_processor(db_session, session_factory) -> AudioProcessor:
    """Get or create processor instance."""
    global processor
    if processor is None:
        processor = AudioProcessor(db_session, session_factory)
        await processor.initialize()
    return processor
