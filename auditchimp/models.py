"""Database models for the Audio Transcription API."""

import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, Enum, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class ProcessingStatus(str, enum.Enum):
    """Processing status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TranscriptionRequest(Base):
    """Model for transcription requests."""
    __tablename__ = "transcription_requests"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=True)  # Duration in seconds
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.QUEUED, nullable=False)
    progress = Column(Float, default=0.0, nullable=False)  # Progress percentage 0-100

    # Processing options
    accurate_mode = Column(Boolean, default=False, nullable=False)
    file_hash = Column(String(128), nullable=True)  # SHA-256 hash of the uploaded file

    # Language settings
    source_language = Column(String(10), nullable=True, default="auto")  # Source language code
    output_language = Column(String(10), nullable=True)                  # Output language code (always same as source - both engines only transcribe)
    detected_language = Column(String(10), nullable=True)                # Actually detected language
    detected_language_name = Column(String(50), nullable=True)           # Human-readable name

    # Engine information
    transcription_engine = Column(String(50), nullable=True)  # "whisper" or "indic-conformer"
    decoder_type = Column(String(20), nullable=True)          # "rnnt", "ctc", or "whisper"
    translation_enabled = Column(Boolean, default=False, nullable=False)  # Whether translation was used

    # Results
    transcription_result = Column(Text, nullable=True)  # JSON string
    diarization_result = Column(Text, nullable=True)  # JSON string
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Processing metadata
    processing_time = Column(Float, nullable=True)  # Total processing time in seconds
    diarization_time = Column(Float, nullable=True)  # Diarization processing time in seconds
    transcription_time = Column(Float, nullable=True)  # Transcription processing time in seconds
    whisper_model = Column(String(50), nullable=True)
    diarization_method = Column(String(50), nullable=True)  # 'nemo' or 'pyannote'


class ProcessingMetrics(Base):
    """Model for system metrics."""
    __tablename__ = "processing_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # System metrics
    cpu_usage = Column(Float, nullable=False)
    memory_usage = Column(Float, nullable=False)
    gpu_usage = Column(Float, nullable=True)

    # Queue metrics
    queued_requests = Column(Integer, default=0, nullable=False)
    processing_requests = Column(Integer, default=0, nullable=False)
    completed_today = Column(Integer, default=0, nullable=False)
    failed_today = Column(Integer, default=0, nullable=False)

    # Performance metrics
    avg_processing_time = Column(Float, nullable=True)
    total_audio_processed = Column(Float, default=0.0, nullable=False)  # Total hours processed


def init_database(database_url: str = "sqlite:///./auditchimp.db"):
    """Initialize the database and create tables."""
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
