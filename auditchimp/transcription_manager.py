"""Multi-engine transcription manager for routing requests to appropriate models."""

import asyncio
import logging
from typing import Dict, List, Optional

from .transcription_parakeet import ParakeetTranscriber
from .transcription_indic import IndicConformerEngine
from .logger import log_event

logger = logging.getLogger(__name__)


class TranscriptionEngineManager:
    """
    Manages multiple transcription engines and intelligently routes requests.

    Routing Logic:
    - IndicConformer: For Indian language transcription (same language in/out)
    - Parakeet: For translation to English or unsupported languages
    """

    def __init__(self, config):
        self.config = config
        self.parakeet_engine = None
        self.indic_engine = None

        # Preload configuration (from environment)
        self.preload_parakeet = config.PRELOAD_PARAKEET
        self.preload_indic = config.PRELOAD_INDIC_CONFORMER
        self.enable_indic = config.ENABLE_INDIC_CONFORMER

        # Indian languages supported by IndicConformer
        self.indic_languages = [
            "ta", "te", "kn", "ml",  # Dravidian
            "hi", "bn", "mr", "gu",  # Indo-Aryan
            "pa", "or", "as", "ur"   # Others
        ]

    async def initialize(self):
        """Initialize engines based on preload configuration."""
        log_event(logger, "info", "transcription_manager_init_started", "TranscriptionEngineManager initialization started")

        # Preload Parakeet
        if self.preload_parakeet:
            await self._load_parakeet()

        # Preload IndicConformer if configured and enabled
        if self.enable_indic and self.preload_indic:
            await self._load_indic()

        log_event(logger, "success", "transcription_manager_init_completed", "TranscriptionEngineManager initialized")

    async def _load_parakeet(self):
        """Lazy load Parakeet engine."""
        if not self.parakeet_engine:
            log_event(logger, "info", "parakeet_loading", "Loading Parakeet engine")
            self.parakeet_engine = ParakeetTranscriber(
                model_name=self.config.MODEL_NAME,
                device=self.config.MODEL_DEVICE,
                compute_type=self.config.MODEL_COMPUTE_TYPE,
                batch_size=self.config.BATCH_SIZE
            )
            await self.parakeet_engine.initialize()
            log_event(logger, "success", "parakeet_loaded", "Parakeet engine loaded")
        return self.parakeet_engine

    async def _load_indic(self):
        """Lazy load IndicConformer engine."""
        if not self.indic_engine:
            log_event(logger, "info", "indic_loading", "Loading IndicConformer engine")
            self.indic_engine = IndicConformerEngine(self.config)
            await self.indic_engine.initialize()
            log_event(logger, "success", "indic_loaded", "IndicConformer engine loaded")
        return self.indic_engine

    def _should_use_indic_conformer(self, source_lang: str) -> bool:
        """
        Determine if IndicConformer should be used for this request.

        IndicConformer is used when:
        1. It's enabled in configuration
        2. Source language is an Indian language

        Args:
            source_lang: Source language code

        Returns:
            True if IndicConformer should be used, False otherwise
        """
        return self.enable_indic and source_lang in self.indic_languages

    def _get_output_language(self, source_lang: str) -> str:
        """
        Determine the output language based on the source language.

        IMPORTANT: Both IndicConformer and Parakeet only do transcription (NOT translation).
        Output is always in the same language as input.

        Args:
            source_lang: Source language code

        Returns:
            Output language code (always same as source language)
        """
        # Both engines transcribe in the same language as the source
        return source_lang

    async def transcribe(
        self,
        audio_path: str,
        source_language: str = "auto",
        diarization_segments: Optional[List[Dict]] = None,
        vad_segments: Optional[List[Dict]] = None,
        overlaps: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Transcribe audio using the appropriate engine with VAD-based chunking.

        Language Handling:
        - Indian languages (ta, te, kn, ml, hi, etc.): Uses IndicConformer → transcribes in source language
        - European languages (en, de, fr, es, etc.): Uses Parakeet → transcribes in source language

        IMPORTANT: Both engines only do transcription, NOT translation.
        Output is always in the same language as input.

        Args:
            audio_path: Path to audio file
            source_language: Source language code or "auto"
            diarization_segments: Optional speaker diarization segments
            vad_segments: Optional VAD segments for precise chunking
            overlaps: Optional overlapping speech regions

        Returns:
            Dictionary containing transcription results and metadata
        """
        try:
            # Handle auto language detection
            if source_language == "auto":
                # Default to Parakeet (English) for auto-detection
                # Parakeet has built-in language detection
                source_lang = "en"
                use_indic = False
                logger.info("Auto language detection: defaulting to Parakeet (English)")
            else:
                source_lang = source_language
                use_indic = self._should_use_indic_conformer(source_lang)

            # Output is always same as source (both engines only transcribe)
            output_lang = self._get_output_language(source_lang)

            if use_indic:
                # Use IndicConformer for Indian languages
                log_event(logger, "info", "routing_to_indic", "Routing to IndicConformer", language=source_lang)

                # Lazy load if not preloaded
                await self._load_indic()

                # Use VAD-based chunking if VAD segments are available
                if vad_segments:
                    result = await self.indic_engine.transcribe_with_vad_chunking(
                        audio_path=audio_path,
                        language=source_lang,
                        vad_segments=vad_segments,
                        diarization_segments=diarization_segments,
                        overlaps=overlaps
                    )
                else:
                    result = await self.indic_engine.transcribe_with_chunking(
                        audio_path=audio_path,
                        language=source_lang,
                        chunk_duration=self.config.INDIC_CHUNK_DURATION,
                        overlap=self.config.INDIC_CHUNK_OVERLAP,
                        diarization_segments=diarization_segments
                    )

                # Add engine metadata
                result["engine"] = "indic-conformer"
                result["source_language"] = source_lang
                result["output_language"] = output_lang
                result["translation_enabled"] = False

            else:
                # Use Parakeet for European languages
                log_event(logger, "info", "routing_to_parakeet", "Routing to Parakeet", language=source_lang)

                # Lazy load if not preloaded
                await self._load_parakeet()

                result = await self.parakeet_engine.transcribe(
                    audio_path=audio_path,
                    language=source_lang,
                    task="transcribe"
                )

                # Add engine metadata
                result["engine"] = "parakeet"
                result["source_language"] = source_lang
                result["output_language"] = output_lang
                result["translation_enabled"] = False

            return result

        except Exception as e:
            log_event(logger, "error", "transcription_failed", "Transcription failed", error=str(e))
            raise

    async def get_supported_languages(self) -> Dict[str, List[str]]:
        """
        Get list of supported languages for each engine.

        Returns:
            Dictionary mapping engine names to language lists
        """
        return {
            "parakeet": [
                # Germanic languages
                "en", "de", "nl", "sv",
                # Romance languages
                "fr", "it", "pt", "es", "ro",
                # Slavic languages
                "bg", "hr", "cs", "pl", "ru", "sk", "sl", "uk",
                # Other Indo-European
                "da", "el", "lt", "lv",
                # Uralic languages
                "et", "fi", "hu",
                # Other EU languages
                "mt"
            ],
            "indic_conformer": self.indic_languages if self.enable_indic else []
        }

    async def get_engine_status(self) -> Dict:
        """
        Get status of all engines.

        Returns:
            Dictionary with engine loading status
        """
        return {
            "parakeet": {
                "loaded": self.parakeet_engine is not None,
                "preload": self.preload_parakeet
            },
            "indic_conformer": {
                "enabled": self.enable_indic,
                "loaded": self.indic_engine is not None,
                "preload": self.preload_indic
            }
        }


# Global manager instance
transcription_manager: Optional[TranscriptionEngineManager] = None


async def get_transcription_manager(config) -> TranscriptionEngineManager:
    """Get or create transcription manager instance."""
    global transcription_manager
    if transcription_manager is None:
        transcription_manager = TranscriptionEngineManager(config)
        await transcription_manager.initialize()
    return transcription_manager
