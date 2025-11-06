"""FastAPI endpoints for the Audio Transcription API."""

import asyncio
import logging
import uuid
import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional
from math import ceil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .models import TranscriptionRequest, ProcessingStatus, init_database
from .config import config
from .processor import get_processor
from .utils import safe_delete_audio_file

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AudioChimp Transcription API",
    description="Fast and accurate audio transcription with speaker diarization",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware - Fixed for FastAPI 0.104.1 compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables
db_session = None
processor = None


async def get_db_session():
    """Get database session."""
    return db_session


async def get_audio_processor():
    """Get audio processor instance."""
    return processor


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global db_session, processor

    logger.info("Starting AudioChimp API...")

    # Set Hugging Face environment variables for Windows compatibility
    os.environ["HF_HOME"] = config.HF_HOME
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = config.HF_HUB_DISABLE_SYMLINKS_WARNING
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = config.HF_HUB_ENABLE_HF_TRANSFER

    # Initialize database
    SessionLocal = init_database(config.DATABASE_URL)
    db_session = SessionLocal()

    # Initialize processor with both session and factory (this starts the job processor automatically)
    processor = await get_processor(db_session, SessionLocal)

    # Create necessary directories
    config.UPLOAD_DIR.mkdir(exist_ok=True)
    config.TEMP_DIR.mkdir(exist_ok=True)
    config.MODEL_CACHE_DIR.mkdir(exist_ok=True)

    # Create logs directory
    # Path(config.LOG_FILE).parent.mkdir(exist_ok=True)

    logger.info("AudioChimp API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down AudioChimp API...")

    if processor:
        await processor.stop_processing_loop()

    if db_session:
        db_session.close()

    logger.info("AudioChimp API shutdown complete")


@app.post("/submit")
async def submit_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    accurate_mode: bool = False,
    source_language: str = "auto"
):
    try:
        # Get source language parameter with header priority
        source_lang = request.headers.get("X-Source-Language", source_language)

        # Determine which model to use based on language
        if source_lang in config.INDIC_LANGUAGES:
            # Use IndicConformer for Indian languages
            use_indic_model = True
        elif source_lang in config.PARAKEET_LANGUAGES:
            # Use Parakeet for European languages
            use_indic_model = False
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {source_lang}. Supported languages: {', '.join(config.SUPPORTED_SOURCE_LANGUAGES)}"
            )

        # Validate source language
        if use_indic_model and source_lang not in config.INDIC_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported Indian language: {source_lang}. Supported: {', '.join(config.INDIC_LANGUAGES)}"
            )
        elif not use_indic_model and source_lang not in config.PARAKEET_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {source_lang}. Supported: {', '.join(config.PARAKEET_LANGUAGES)}"
            )

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in config.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {', '.join(config.SUPPORTED_FORMATS)}"
            )

        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)

        if file_size > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )

        # Calculate file hash
        file_hash = hashlib.sha256(content).hexdigest()

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Save file
        file_path = config.UPLOAD_DIR / f"{request_id}{file_extension}"
        with open(file_path, "wb") as f:
            f.write(content)

        # Create database record
        request_obj = TranscriptionRequest(
            request_id=request_id,
            filename=file.filename,
            file_path=str(file_path),
            file_size=file_size,
            accurate_mode=accurate_mode,
            file_hash=file_hash,
            source_language=source_lang,
            status=ProcessingStatus.QUEUED,
            progress=0.0
        )

        db_session.add(request_obj)
        db_session.commit()

        # Notify processor that a new job is available for immediate pickup
        if processor and hasattr(processor.job_processor, '_job_available_event'):
            processor.job_processor._job_available_event.set()

        model_type = "IndicConformer" if use_indic_model else "Parakeet"
        logger.info(
            f"Audio file submitted: {request_id} - {file.filename} "
            f"(hash: {file_hash[:16]}..., lang: {source_lang}, model: {model_type})"
        )

        return {
            "request_id": request_id,
            "file_hash": file_hash,
            "source_language": source_lang,
            "model": model_type,
            "status": "queued",
            "message": "Audio file submitted successfully",
            "estimated_time": "2-5 minutes depending on file length and processing mode"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting audio: {str(e)}")


@app.get("/status")
async def get_status_list(
    page: int = 1,
    limit: int = 10,
    status: Optional[str] = None
):
    """Get paginated list of transcription requests with optional status filter."""
    try:
        # Validate page and limit
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be >= 1")
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")

        # Validate status filter if provided
        if status:
            valid_statuses = ["queued", "diarizing", "transcribing", "completed", "failed", "canceled"]
            if status.lower() not in valid_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
                )

        # Build query
        query = db_session.query(TranscriptionRequest)

        # Apply status filter if provided
        if status:
            query = query.filter(TranscriptionRequest.status == ProcessingStatus(status.lower()))

        # Get total count
        total_items = query.count()

        # Calculate pagination
        total_pages = ceil(total_items / limit) if total_items > 0 else 1
        has_next = page < total_pages

        # Get paginated results (sorted by created_at DESC - newest first)
        offset = (page - 1) * limit
        requests = query.order_by(TranscriptionRequest.created_at.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()

        # Format items (same structure as single status endpoint)
        items = []
        for request in requests:
            # Parse results if available
            transcription_result = None
            diarization_result = None

            if request.transcription_result:
                transcription_data = json.loads(request.transcription_result)
                transcription_result = {
                    "metadata": {
                        "request_id": request.request_id,
                        "created": request.created_at.isoformat() if request.created_at else None,
                        "duration": request.duration,
                        "models": [request.transcription_engine] if request.transcription_engine else []
                    },
                    "results": {
                        "channels": [
                            {
                                "alternatives": [transcription_data]
                            }
                        ]
                    }
                }

            if request.diarization_result:
                diarization_result = json.loads(request.diarization_result)

            items.append({
                "request_id": request.request_id,
                "file_hash": request.file_hash,
                "status": request.status.value,
                "progress": request.progress,
                "filename": request.filename,
                "file_size": request.file_size,
                "duration": request.duration,
                "source_language": request.source_language,
                "output_language": request.output_language,
                "detected_language": request.detected_language,
                "detected_language_name": request.detected_language_name,
                "transcription_engine": request.transcription_engine,
                "transcription_result": transcription_result,
                "diarization_result": diarization_result,
                "error_message": request.error_message,
                "events": {
                    "queued_at": request.created_at.isoformat(),
                    "process_start": request.started_at.isoformat() if request.started_at else None,
                    "process_end": request.completed_at.isoformat() if request.completed_at else None,
                    "process_status": request.status.value,
                }
            })

        return {
            "items": items,
            "page": page,
            "limit": limit,
            "total_items": total_items,
            "total_pages": total_pages,
            "hasNext": has_next
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status list: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status list: {str(e)}")


@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Get processing status for a single request."""
    try:
        request = db_session.query(TranscriptionRequest)\
            .filter(TranscriptionRequest.request_id == request_id)\
            .first()

        if not request:
            raise HTTPException(status_code=404, detail="Request not found")

        # Parse results if available
        transcription_result = None
        diarization_result = None

        if request.transcription_result:
            transcription_data = json.loads(request.transcription_result)

            # Restructure transcription result in new nested format
            transcription_result = {
                "metadata": {
                    "request_id": request.request_id,
                    "created": request.created_at.isoformat() if request.created_at else None,
                    "duration": request.duration,
                    "models": [request.transcription_engine] if request.transcription_engine else []
                },
                "results": {
                    "channels": [
                        {
                            "alternatives": [transcription_data]
                        }
                    ]
                }
            }

        if request.diarization_result:
            diarization_result = json.loads(request.diarization_result)

        return {
            "request_id": request.request_id,
            "file_hash": request.file_hash,
            "status": request.status.value,
            "progress": request.progress,
            "filename": request.filename,
            "file_size": request.file_size,
            "duration": request.duration,
            "source_language": request.source_language,
            "output_language": request.output_language,
            "detected_language": request.detected_language,
            "detected_language_name": request.detected_language_name,
            "transcription_engine": request.transcription_engine,
            "transcription_result": transcription_result,
            "diarization_result": diarization_result,
            "error_message": request.error_message,
            "events": {
                "queued_at": request.created_at.isoformat(),
                "process_start": request.started_at.isoformat() if request.started_at else None,
                "process_end": request.completed_at.isoformat() if request.completed_at else None,
                "process_status": request.status.value,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@app.get("/queue")
async def get_queue_status():
    """Get current queue status."""
    try:
        if not processor:
            raise HTTPException(status_code=503, detail="Processor not initialized")

        queue_status = await processor.get_queue_status()
        system_metrics = await processor.get_system_metrics()

        return {
            "queue_status": queue_status,
            "system_metrics": system_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting queue status: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get system metrics and performance data."""
    try:
        if not processor:
            raise HTTPException(status_code=503, detail="Processor not initialized")

        # Get queue metrics
        queue_status = await processor.get_queue_status()

        # Get system metrics
        system_metrics = await processor.get_system_metrics()

        # Calculate additional metrics
        total_processed = queue_status.get("completed_today", 0) + queue_status.get("failed_today", 0)
        success_rate = 0.0
        if total_processed > 0:
            success_rate = queue_status.get("completed_today", 0) / total_processed * 100

        return {
            "system": system_metrics,
            "queue": queue_status,
            "performance": {
                "total_processed_today": total_processed,
                "success_rate": success_rate,
                "average_queue_time": "2-3 minutes",  # This would be calculated from actual data
                "estimated_wait_time": f"{queue_status.get('queued', 0) * 3} minutes"
            },
            "configuration": {
                "model_name": config.MODEL_NAME,
                "compute_type": config.MODEL_COMPUTE_TYPE,
                "diarization_gpu": config.DIARIZATION_USE_GPU,
                "max_concurrent_jobs": config.MAX_CONCURRENT_JOBS
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_status = "healthy"
        try:
            db_session.execute("SELECT 1")
        except Exception:
            db_status = "unhealthy"

        # Check processor status
        processor_status = "running" if processor and processor.is_processing else "stopped"

        # Check model status
        model_status = "loaded"
        try:
            if processor and processor.transcription_engine:
                # Try to access the model
                _ = processor.transcription_engine.model
        except Exception:
            model_status = "unloaded"

        # Check GPU availability
        gpu_info = {"available": False, "name": None, "memory": None}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["name"] = torch.cuda.get_device_name(0)
                if torch.cuda.get_device_properties(0).total_memory:
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_info["memory"] = f"{memory_gb:.1f} GB"
        except Exception as e:
            logger.warning(f"Error checking GPU status: {e}")

        # Overall health status
        overall_status = "healthy"
        issues = []

        if db_status != "healthy":
            overall_status = "unhealthy"
            issues.append("Database connection failed")

        if processor_status != "running":
            overall_status = "degraded"
            issues.append("Processor not running")

        if model_status != "loaded":
            overall_status = "degraded"
            issues.append("Models not loaded")

        return {
            "status": overall_status,
            "issues": issues,
            "database": db_status,
            "processor": processor_status,
            "models": model_status,
            "gpu": gpu_info,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.delete("/request/{request_id}")
async def delete_request(request_id: str):
    """Delete a transcription request and associated files."""
    try:
        request = db_session.query(TranscriptionRequest)\
            .filter(TranscriptionRequest.request_id == request_id)\
            .first()

        if not request:
            raise HTTPException(status_code=404, detail="Request not found")

        # Safely delete the associated audio file
        file_deleted = safe_delete_audio_file(
            file_path=request.file_path,
            request_id=request_id,
            context="manual_delete"
        )

        # Delete database record (even if file deletion failed)
        db_session.delete(request)
        db_session.commit()

        logger.info(
            f"Deleted request: {request_id} "
            f"(file_deleted: {file_deleted})"
        )

        return {
            "message": "Request deleted successfully",
            "request_id": request_id,
            "file_deleted": file_deleted
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting request: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting request: {str(e)}")


@app.patch("/request/{request_id}/cancel")
async def cancel_request(request_id: str):
    """Cancel a queued or processing transcription request."""
    try:
        request = db_session.query(TranscriptionRequest)\
            .filter(TranscriptionRequest.request_id == request_id)\
            .first()

        if not request:
            raise HTTPException(status_code=404, detail="Request not found")

        # Check if request can be canceled
        if request.status not in [ProcessingStatus.QUEUED, ProcessingStatus.DIARIZING]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel request with status: {request.status.value}. Only queued or diarizing requests can be canceled."
            )

        # Safely delete the associated audio file
        file_deleted = safe_delete_audio_file(
            file_path=request.file_path,
            request_id=request_id,
            context="cancellation"
        )

        # Update status to canceled
        request.status = ProcessingStatus.CANCELED
        request.completed_at = datetime.utcnow()
        request.error_message = "Request canceled by user"
        db_session.commit()

        logger.info(
            f"Canceled request: {request_id} "
            f"(status: {request.status.value}, file_deleted: {file_deleted})"
        )

        return {
            "message": "Request canceled successfully",
            "request_id": request_id,
            "status": "canceled",
            "file_deleted": file_deleted
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error canceling request: {e}")
        raise HTTPException(status_code=500, detail=f"Error canceling request: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )
