"""Multi-engine transcription manager for routing requests to appropriate models."""

import asyncio
import logging
from typing import Dict, List, Optional

from .transcription import TranscriptionEngine as WhisperEngine
from .transcription_indic import IndicConformerEngine

logger = logging.getLogger(__name__)


class TranscriptionEngineManager:
    """
    Manages multiple transcription engines and intelligently routes requests.

    Routing Logic:
    - IndicConformer: For Indian language transcription (same language in/out)
    - Whisper: For translation to English or unsupported languages
    """

    def __init__(self, config):
        self.config = config
        self.whisper_engine = None
        self.indic_engine = None

        # Preload configuration
        self.preload_whisper = config.PRELOAD_WHISPER
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
        logger.info("Initializing TranscriptionEngineManager...")

        # Preload Whisper if configured
        if self.preload_whisper:
            await self._load_whisper()
        else:
            logger.info("Whisper engine set to lazy load")

        # Preload IndicConformer if configured and enabled
        if self.enable_indic and self.preload_indic:
            await self._load_indic()
        elif self.enable_indic:
            logger.info("IndicConformer engine set to lazy load")
        else:
            logger.info("IndicConformer engine disabled")

        logger.info("TranscriptionEngineManager initialized successfully")

    async def _load_whisper(self):
        """Lazy load Whisper engine."""
        if not self.whisper_engine:
            logger.info("Loading Whisper engine...")
            self.whisper_engine = WhisperEngine(self.config)
            await self.whisper_engine.initialize()
            logger.info("Whisper engine loaded successfully")
        return self.whisper_engine

    async def _load_indic(self):
        """Lazy load IndicConformer engine."""
        if not self.indic_engine:
            logger.info("Loading IndicConformer engine...")
            self.indic_engine = IndicConformerEngine(self.config)
            await self.indic_engine.initialize()
            logger.info("IndicConformer engine loaded successfully")
        return self.indic_engine

    def _should_use_indic_conformer(
        self,
        source_lang: str,
        target_lang: str
    ) -> bool:
        """
        Determine if IndicConformer should be used for this request.

        IndicConformer is used when:
        1. It's enabled in configuration
        2. Source language is an Indian language
        3. Target language equals source (transcription, not translation)

        Args:
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            True if IndicConformer should be used, False otherwise
        """
        return (
            self.enable_indic and
            source_lang in self.indic_languages and
            target_lang == source_lang  # Same language = transcription only
        )

    def _resolve_language(
        self,
        source_language: str,
        target_language: str
    ) -> tuple[str, str]:
        """
        Resolve auto language settings to actual language codes.

        Args:
            source_language: Source language (may be "auto")
            target_language: Target language (may be "auto")

        Returns:
            Tuple of (resolved_source, resolved_target)
        """
        # If target is auto, determine based on source
        if target_language == "auto":
            if source_language in self.indic_languages:
                # Indian language: default to same language (transcription)
                target_language = source_language
            else:
                # Other languages: default to English (translation)
                target_language = "en"

        return source_language, target_language

    async def transcribe(
        self,
        audio_path: str,
        source_language: str = "auto",
        target_language: str = "auto",
        accurate_mode: bool = False,
        diarization_segments: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Transcribe audio using the appropriate engine.

        Routes to IndicConformer for Indian language transcription,
        Whisper for translation or unsupported languages.

        Args:
            audio_path: Path to audio file
            source_language: Source language code or "auto"
            target_language: Target language code or "auto"
            accurate_mode: Whether to use accurate mode (Whisper only)
            diarization_segments: Optional speaker diarization segments

        Returns:
            Dictionary containing transcription results and metadata
        """
        try:
            # Resolve auto language settings
            source_lang, target_lang = self._resolve_language(
                source_language,
                target_language
            )

            # Determine which engine to use
            use_indic = self._should_use_indic_conformer(source_lang, target_lang)

            if use_indic:
                # Use IndicConformer for Indian language transcription
                logger.info(
                    f"Routing to IndicConformer: {source_lang} → {target_lang} (transcription)"
                )

                # Lazy load if not preloaded
                await self._load_indic()

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
                result["target_language"] = target_lang
                result["translation_enabled"] = False

            else:
                # Use Whisper for translation or unsupported languages
                reason = "translation" if source_lang != target_lang else "unsupported language"
                logger.info(
                    f"Routing to Whisper: {source_lang} → {target_lang} ({reason})"
                )

                # Lazy load if not preloaded
                await self._load_whisper()

                result = await self.whisper_engine.transcribe(
                    audio_path=audio_path,
                    accurate_mode=accurate_mode,
                    diarization_segments=diarization_segments,
                    source_language=source_lang,
                    target_language=target_lang
                )

                # Add engine metadata
                result["engine"] = "whisper"

            logger.info(
                f"Transcription completed using {result['engine']}: "
                f"{len(result.get('segments', []))} segments"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    async def get_supported_languages(self) -> Dict[str, List[str]]:
        """
        Get list of supported languages for each engine.

        Returns:
            Dictionary mapping engine names to language lists
        """
        return {
            "whisper": [
                "en", "ar", "ta", "te", "kn", "ml", "hi", "bn",
                "and 90+ other languages"
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
            "whisper": {
                "loaded": self.whisper_engine is not None,
                "preload": self.preload_whisper
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
