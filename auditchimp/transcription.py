"""Audio transcription module using faster-whisper with quantization support."""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """Audio transcription engine using faster-whisper."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = self._get_device()
        self.compute_type = config.WHISPER_COMPUTE_TYPE_ENUM

    def _get_device(self) -> str:
        """Get the appropriate device based on configuration and availability."""
        if self.config.WHISPER_DEVICE == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("CUDA detected, using GPU")
                    return "cuda"
                else:
                    logger.info("CUDA not available, using CPU")
                    return "cpu"
            except Exception as e:
                logger.warning(f"Error detecting CUDA: {e}, falling back to CPU")
                return "cpu"
        else:
            return self.config.WHISPER_DEVICE

    async def initialize(self):
        """Initialize the Whisper model."""
        try:
            logger.info(f"Initializing faster-whisper model: {self.config.WHISPER_MODEL}")
            logger.info(f"Device: {self.device}, Compute type: {self.compute_type}")

            self.model = WhisperModel(
                self.config.WHISPER_MODEL,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.config.MODEL_CACHE_DIR)
            )

            logger.info("faster-whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize faster-whisper: {e}")
            raise

    async def transcribe(
        self,
        audio_path: str,
        accurate_mode: bool = False,
        diarization_segments: Optional[List[Dict]] = None,
        source_language: str = "auto",
        target_language: str = "en"
    ) -> Dict:
        """
        Transcribe audio file with optional accurate mode and language support.

        Args:
            audio_path: Path to audio file
            accurate_mode: Whether to use accurate mode (chunked by diarization)
            diarization_segments: Optional speaker diarization segments
            source_language: Source language code or "auto" for auto-detection
            target_language: Target language code (for translation)

        Returns:
            Dictionary containing transcription results and metadata
        """
        try:
            if accurate_mode and diarization_segments:
                return self._transcribe_accurate_sync(
                    audio_path,
                    diarization_segments,
                    source_language,
                    target_language
                )
            else:
                return await self._transcribe_fast(
                    audio_path,
                    diarization_segments,
                    source_language,
                    target_language
                )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def _transcribe_accurate_sync(
        self,
        audio_path: str,
        diarization_segments: List[Dict],
        source_language: str = "auto",
        target_language: str = "en"
    ) -> Dict:
        """Accurate transcription mode: transcribe each diarization segment separately (sync version)."""
        logger.info(f"Starting accurate transcription mode: {source_language} → {target_language}...")

        from pydub import AudioSegment

        # Load audio file
        audio = AudioSegment.from_file(audio_path)

        all_segments = []
        total_segments = len(diarization_segments)

        for i, dia_segment in enumerate(diarization_segments):
            try:
                # Extract audio chunk
                start_ms = int(dia_segment["start"] * 1000)
                end_ms = int(dia_segment["end"] * 1000)
                chunk = audio[start_ms:end_ms]

                # Save temporary chunk
                temp_path = f"{audio_path}.chunk_{i}.wav"
                chunk.export(temp_path, format="wav")

                # Determine task and language
                task = "translate" if target_language == "en" and source_language != "en" else "transcribe"
                lang_param = None if source_language == "auto" else source_language

                # Transcribe chunk (synchronous call)
                segments, info = self.model.transcribe(
                    temp_path,
                    beam_size=5,
                    language=lang_param,
                    task=task
                )

                # Process segments
                for segment in segments:
                    # Handle different segment attributes based on faster-whisper version
                    confidence = getattr(segment, 'probability', None)
                    if confidence is None:
                        # Try alternative attributes that might contain confidence
                        confidence = getattr(segment, 'avg_logprob', None)
                        if confidence is None:
                            confidence = 0.8  # Default confidence if not available

                    all_segments.append({
                        "start": dia_segment["start"] + segment.start,
                        "end": dia_segment["start"] + segment.end,
                        "text": segment.text.strip(),
                        "speaker": dia_segment["speaker"],
                        "confidence": confidence
                    })

                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

                # Update progress (this would be called via callback in real implementation)
                progress = (i + 1) / total_segments * 100
                logger.info(f"Accurate transcription progress: {progress:.1f}%")

            except Exception as e:
                logger.error(f"Failed to transcribe segment {i}: {e}")
                continue

        # Get detected language from first chunk info (if available)
        detected_lang = source_language if source_language != "auto" else "en"

        return {
            "segments": all_segments,
            "mode": "accurate",
            "total_segments": len(all_segments),
            "language": target_language,
            "source_language": source_language,
            "detected_language": detected_lang,
            "target_language": target_language,
            "translation_enabled": task == "translate"
        }

    async def _transcribe_fast(
        self,
        audio_path: str,
        diarization_segments: Optional[List[Dict]] = None,
        source_language: str = "auto",
        target_language: str = "en"
    ) -> Dict:
        """Fast transcription mode: transcribe full audio, then map speakers."""
        logger.info(f"Starting fast transcription mode: {source_language} → {target_language}...")

        # Determine task and language
        task = "translate" if target_language == "en" and source_language != "en" else "transcribe"
        lang_param = None if source_language == "auto" else source_language

        # Transcribe full audio
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            language=lang_param,
            task=task
        )

        # Convert segments to list
        transcription_segments = []
        for segment in segments:
            # Handle different segment attributes based on faster-whisper version
            confidence = getattr(segment, 'probability', None)
            if confidence is None:
                # Try alternative attributes that might contain confidence
                confidence = getattr(segment, 'avg_logprob', None)
                if confidence is None:
                    confidence = 0.8  # Default confidence if not available

            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "confidence": confidence
            })

        # Map speakers if diarization is available
        if diarization_segments:
            transcription_segments = self._map_speakers(transcription_segments, diarization_segments)

        return {
            "segments": transcription_segments,
            "mode": "fast",
            "total_segments": len(transcription_segments),
            "language": info.language,
            "source_language": source_language,
            "detected_language": info.language,
            "detected_language_name": self._get_language_name(info.language),
            "target_language": target_language,
            "translation_enabled": task == "translate"
        }

    def _map_speakers(self, transcription_segments: List[Dict], diarization_segments: List[Dict]) -> List[Dict]:
        """Map diarization speakers to transcription segments."""
        # Create a mapping of time ranges to speakers
        speaker_map = {}
        for dia_seg in diarization_segments:
            start, end, speaker = dia_seg["start"], dia_seg["end"], dia_seg["speaker"]
            # Map each second in the range to the speaker
            for second in range(int(start), int(end) + 1):
                speaker_map[second] = speaker

        # Assign speakers to transcription segments
        for trans_seg in transcription_segments:
            # Find the dominant speaker for this segment
            seg_start = int(trans_seg["start"])
            seg_end = int(trans_seg["end"])
            seg_duration = seg_end - seg_start

            if seg_duration == 0:
                continue

            # Count speakers in this segment
            speaker_count = {}
            for second in range(seg_start, min(seg_end + 1, seg_start + seg_duration + 1)):
                speaker = speaker_map.get(second)
                if speaker:
                    speaker_count[speaker] = speaker_count.get(speaker, 0) + 1

            # Assign the dominant speaker
            if speaker_count:
                dominant_speaker = max(speaker_count, key=speaker_count.get)
                trans_seg["speaker"] = dominant_speaker
            else:
                trans_seg["speaker"] = "UNKNOWN"

        return transcription_segments

    def _get_language_name(self, lang_code: str) -> str:
        """Convert language code to human-readable name."""
        from .config import config
        return config.LANGUAGE_NAMES.get(lang_code, lang_code.upper())

    async def get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration."""
        try:
            import librosa
            duration = librosa.get_duration(filename=audio_path)
            return duration
        except Exception as e:
            logger.warning(f"Could not get duration with librosa: {e}")
            # Fallback: try with pydub
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                return len(audio) / 1000.0
            except Exception as e2:
                logger.error(f"Could not get duration: {e2}")
                return 0.0


# Global transcription engine instance
transcription_engine: Optional[TranscriptionEngine] = None


async def get_transcription_engine(config) -> TranscriptionEngine:
    """Get or create transcription engine instance."""
    global transcription_engine
    if transcription_engine is None:
        transcription_engine = TranscriptionEngine(config)
        await transcription_engine.initialize()
    return transcription_engine
