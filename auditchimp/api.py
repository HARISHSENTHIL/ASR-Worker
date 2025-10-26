"""FastAPI endpoints for the Audio Transcription API."""

import asyncio
import logging
import uuid
import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .models import TranscriptionRequest, ProcessingStatus, init_database
from .config import config
from .processor import get_processor

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

    # Initialize processor (this starts the job processor automatically)
    processor = await get_processor(db_session)

    # Create necessary directories
    config.UPLOAD_DIR.mkdir(exist_ok=True)
    config.TEMP_DIR.mkdir(exist_ok=True)
    config.MODEL_CACHE_DIR.mkdir(exist_ok=True)

    # Create logs directory
    Path(config.LOG_FILE).parent.mkdir(exist_ok=True)

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
    source_language: str = "auto",
    target_language: str = "auto"
):
    """
    Submit audio file for transcription and diarization.

    Language Detection Priority:
    1. HTTP headers (X-Source-Language, X-Target-Language)
    2. Form parameters (source_language, target_language)
    3. Defaults (auto, auto)

    Target Language Defaults:
    - If source is Indian language and target is "auto" → same as source (transcription)
    - Otherwise if target is "auto" → English (translation)
    """
    try:
        # Get language parameters with header priority
        source_lang = request.headers.get("X-Source-Language", source_language)
        target_lang = request.headers.get("X-Target-Language", target_language)

        # Resolve auto target language
        if target_lang == "auto":
            # Indian languages default to same language (transcription)
            if source_lang in ["ta", "te", "kn", "ml", "hi", "bn", "mr", "gu", "pa", "or", "as", "ur"]:
                target_lang = source_lang
            else:
                target_lang = "en"

        # Validate source language
        if source_lang not in config.SUPPORTED_SOURCE_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source language: {source_lang}. Supported: {', '.join(config.SUPPORTED_SOURCE_LANGUAGES)}"
            )

        # Validate target language (currently only English supported for translation)
        if target_lang != source_lang and target_lang != "en":
            raise HTTPException(
                status_code=400,
                detail=f"Translation only supported to English. Use target_language='en' or omit for same-language transcription."
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
            quantization=config.QUANTIZATION,  # Always use config value
            file_hash=file_hash,
            source_language=source_lang,
            target_language=target_lang,
            status=ProcessingStatus.QUEUED,
            progress=0.0
        )

        db_session.add(request_obj)
        db_session.commit()

        logger.info(
            f"Audio file submitted: {request_id} - {file.filename} "
            f"(hash: {file_hash[:16]}..., lang: {source_lang}→{target_lang})"
        )

        return {
            "request_id": request_id,
            "file_hash": file_hash,
            "source_language": source_lang,
            "target_language": target_lang,
            "status": "queued",
            "message": "Audio file submitted successfully",
            "estimated_time": "2-5 minutes depending on file length and processing mode"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting audio: {str(e)}")


@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Get processing status for a request."""
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
            transcription_result = json.loads(request.transcription_result)

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
            "created_at": request.created_at.isoformat(),
            "started_at": request.started_at.isoformat() if request.started_at else None,
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "processing_time": request.processing_time,
            "accurate_mode": request.accurate_mode,
            "quantization": request.quantization,
            "source_language": request.source_language,
            "target_language": request.target_language,
            "detected_language": request.detected_language,
            "detected_language_name": request.detected_language_name,
            "transcription_engine": request.transcription_engine,
            "decoder_type": request.decoder_type,
            "translation_enabled": request.translation_enabled,
            "whisper_model": request.whisper_model,
            "diarization_method": request.diarization_method,
            "transcription_result": transcription_result,
            "diarization_result": diarization_result,
            "error_message": request.error_message
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
                "whisper_model": config.WHISPER_MODEL,
                "quantization": config.QUANTIZATION,
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

        # Delete file if it exists
        if Path(request.file_path).exists():
            Path(request.file_path).unlink()

        # Delete database record
        db_session.delete(request)
        db_session.commit()

        logger.info(f"Deleted request: {request_id}")

        return {
            "message": "Request deleted successfully",
            "request_id": request_id
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
        if request.status not in [ProcessingStatus.QUEUED, ProcessingStatus.PROCESSING]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel request with status: {request.status.value}. Only queued or processing requests can be canceled."
            )

        # Delete the associated file
        if Path(request.file_path).exists():
            Path(request.file_path).unlink()

        # Update status to canceled
        request.status = ProcessingStatus.CANCELED
        request.completed_at = datetime.utcnow()
        request.error_message = "Request canceled by user"
        db_session.commit()

        logger.info(f"Canceled request: {request_id} (status: {request.status.value})")

        return {
            "message": "Request canceled successfully",
            "request_id": request_id,
            "status": "canceled"
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
