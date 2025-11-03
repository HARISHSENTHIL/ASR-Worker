"""Configuration management for the Audio Transcription API."""

import os
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration.
    
    This class manages all configuration settings for the Audio Transcription API,
    including API settings, model configurations, and language support.
    """

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./auditchimp.db")

    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "auto")  # auto, cuda, cpu
    MODEL_COMPUTE_TYPE: str = os.getenv("MODEL_COMPUTE_TYPE", "float16")  # float16 or float32
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))  # Default from model card
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "./models"))

    # IndicConformer Device Configuration
    INDIC_DEVICE: str = os.getenv("INDIC_DEVICE", "auto")  # Device for IndicConformer: auto, cuda, cpu

    # Processing Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100000000"))  # 100MB
    SUPPORTED_FORMATS: List[str] = os.getenv("SUPPORTED_FORMATS", "mp3,wav,flac").split(",")
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))

    # Diarization Configuration
    DIARIZATION_USE_GPU: bool = os.getenv("DIARIZATION_USE_GPU", "true").lower() == "true"  # Use GPU for diarization when available

    # Hugging Face Configuration (for model downloads)
    HF_HOME: str = os.getenv("HF_HOME", "./models")
    HF_HUB_DISABLE_SYMLINKS_WARNING: str = os.getenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    HF_HUB_ENABLE_HF_TRANSFER: str = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "0")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)  # Hugging Face token for gated models

    # Multi-Engine Configuration
    ENABLE_INDIC_CONFORMER: bool = os.getenv("ENABLE_INDIC_CONFORMER", "true").lower() == "true"
    PRELOAD_PARAKEET: bool = os.getenv("PRELOAD_PARAKEET", "true").lower() == "true"
    PRELOAD_INDIC_CONFORMER: bool = os.getenv("PRELOAD_INDIC_CONFORMER", "true").lower() == "true"
    
    # IndicConformer Chunking Configuration
    INDIC_CHUNK_DURATION: int = int(os.getenv("INDIC_CHUNK_DURATION", "30"))  # seconds
    INDIC_CHUNK_OVERLAP: int = int(os.getenv("INDIC_CHUNK_OVERLAP", "3"))    # seconds

    # Language Configuration
    INDIC_LANGUAGES: List[str] = [
        "ta", "te", "kn", "ml",  # Dravidian
        "hi", "bn", "mr", "gu",  # Indo-Aryan
        "pa", "or", "as", "ur"   # Others
    ]

    PARAKEET_LANGUAGES: List[str] = [
        "en", "de", "nl", "sv",  # Germanic
        "fr", "it", "pt", "es", "ro",  # Romance
        "bg", "hr", "cs", "pl", "ru", "sk", "sl", "uk",  # Slavic
        "da", "el", "lt", "lv",  # Other Indo-European
        "et", "fi", "hu",  # Uralic
        "mt"  # Other EU
    ]

    # Language name mappings
    INDIC_LANGUAGES_MAP: Dict[str, str] = {
        "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
        "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
        "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ur": "Urdu"
    }

    PARAKEET_LANGUAGES_MAP: Dict[str, str] = {
        "en": "English", "de": "German", "nl": "Dutch", "sv": "Swedish",
        "fr": "French", "it": "Italian", "pt": "Portuguese", "es": "Spanish",
        "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian", "cs": "Czech",
        "pl": "Polish", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian",
        "uk": "Ukrainian", "da": "Danish", "el": "Greek", "lt": "Lithuanian",
        "lv": "Latvian", "et": "Estonian", "fi": "Finnish", "hu": "Hungarian",
        "mt": "Maltese"
    }

    # Combined list of all supported languages
    SUPPORTED_SOURCE_LANGUAGES: List[str] = sorted(INDIC_LANGUAGES + PARAKEET_LANGUAGES)

    # File path configuration
    UPLOAD_DIR: Path = Path("./uploads")
    TEMP_DIR: Path = Path("./temp")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/auditchimp.log")

    @property
    def SUPPORTED_EXTENSIONS(self) -> List[str]:
        """Get supported file extensions."""
        return [f".{fmt}" for fmt in self.SUPPORTED_FORMATS]


# Global configuration instance
config = Config()
