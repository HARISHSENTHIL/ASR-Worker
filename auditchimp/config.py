"""Configuration management for the Audio Transcription API."""

import os
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./auditchimp.db")

    # Audio Processing Configuration
    QUANTIZATION: bool = os.getenv("QUANTIZATION", "true").lower() == "true"
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v2")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "auto")  # auto, cuda, cpu
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16")

    # Diarization Configuration (NeMo only)
    ENABLE_DIARIZATION: bool = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
    DIARIZATION_USE_GPU: bool = os.getenv("DIARIZATION_USE_GPU", "true").lower() == "true"

    # Processing Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100000000"))  # 100MB
    SUPPORTED_FORMATS: List[str] = os.getenv("SUPPORTED_FORMATS", "mp3,wav,flac,m4a,ogg,webm").split(",")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "30"))  # seconds
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))

    # Model Cache Configuration
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "./models"))
    MAX_CACHE_SIZE: str = os.getenv("MAX_CACHE_SIZE", "10GB")

    # Hugging Face Configuration (for Windows compatibility)
    HF_HOME: str = os.getenv("HF_HOME", str(MODEL_CACHE_DIR))
    HF_HUB_DISABLE_SYMLINKS_WARNING: str = os.getenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    HF_HUB_ENABLE_HF_TRANSFER: str = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "0")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)

    # Multi-Engine Configuration
    ENABLE_INDIC_CONFORMER: bool = os.getenv("ENABLE_INDIC_CONFORMER", "true").lower() == "true"
    PRELOAD_WHISPER: bool = os.getenv("PRELOAD_WHISPER", "true").lower() == "true"
    PRELOAD_INDIC_CONFORMER: bool = os.getenv("PRELOAD_INDIC_CONFORMER", "true").lower() == "true"

    # IndicConformer Chunking Configuration
    INDIC_CHUNK_DURATION: int = int(os.getenv("INDIC_CHUNK_DURATION", "30"))  # seconds
    INDIC_CHUNK_OVERLAP: int = int(os.getenv("INDIC_CHUNK_OVERLAP", "3"))    # seconds

    # Language Configuration
    SUPPORTED_SOURCE_LANGUAGES: List[str] = [
        "ta", "te", "kn", "ml",  # Dravidian languages
        "hi", "bn", "mr", "gu",  # Indo-Aryan languages
        "pa", "or", "as", "ur",  # Other Indian languages
        "ar",                     # Arabic (Whisper only)
        "en",                     # English
        "auto"                    # Auto-detect
    ]

    DEFAULT_SOURCE_LANGUAGE: str = os.getenv("DEFAULT_SOURCE_LANGUAGE", "auto")
    DEFAULT_TARGET_LANGUAGE: str = os.getenv("DEFAULT_TARGET_LANGUAGE", "auto")

    # Language code to name mapping
    LANGUAGE_NAMES: Dict[str, str] = {
        "ta": "Tamil",
        "te": "Telugu",
        "kn": "Kannada",
        "ml": "Malayalam",
        "hi": "Hindi",
        "bn": "Bengali",
        "mr": "Marathi",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "or": "Odia",
        "as": "Assamese",
        "ur": "Urdu",
        "ar": "Arabic",
        "en": "English"
    }

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/auditchimp.log")

    # Upload Configuration
    UPLOAD_DIR: Path = Path("./uploads")
    TEMP_DIR: Path = Path("./temp")

    # Derived Configuration
    @property
    def WHISPER_COMPUTE_TYPE_ENUM(self) -> str:
        """Get the compute type for faster-whisper."""
        if self.QUANTIZATION:
            return self.WHISPER_COMPUTE_TYPE
        return "float16" if self.WHISPER_DEVICE == "cuda" else "int8"

    @property
    def SUPPORTED_EXTENSIONS(self) -> List[str]:
        """Get supported file extensions."""
        return [f".{fmt.strip()}" for fmt in self.SUPPORTED_FORMATS]


# Global configuration instance
config = Config()
