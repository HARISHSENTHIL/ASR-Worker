"""IndicConformer transcription engine for Indian languages with chunking support."""

import asyncio
import logging
from typing import Dict, List, Optional
import torch
import torchaudio

logger = logging.getLogger(__name__)


class IndicConformerEngine:
    """
    Transcription engine using AI4Bharat's IndicConformer model.
    Optimized for Indian languages: Tamil, Telugu, Kannada, Malayalam, Hindi, etc.

    Uses chunking with overlap to handle long audio files efficiently.
    """

    def __init__(self, config):
        self.config = config
        self.model = None

        # Languages supported by IndicConformer
        self.supported_languages = [
            "ta",  # Tamil
            "te",  # Telugu
            "kn",  # Kannada
            "ml",  # Malayalam
            "hi",  # Hindi
            "bn",  # Bengali
            "mr",  # Marathi
            "gu",  # Gujarati
            "pa",  # Punjabi
            "or",  # Odia
            "as",  # Assamese
            "ur",  # Urdu
        ]

        # Language code to name mapping
        self.language_names = {
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
        }

    async def initialize(self):
        """Initialize the IndicConformer model."""
        try:
            logger.info("Initializing IndicConformer model...")

            from transformers import AutoModel

            # Prepare kwargs for model loading
            model_kwargs = {
                "trust_remote_code": True,
                "cache_dir": str(self.config.MODEL_CACHE_DIR)
            }

            # Add token if available (required for gated models)
            if self.config.HF_TOKEN:
                model_kwargs["token"] = self.config.HF_TOKEN
                logger.info("Using Hugging Face token for gated model access")

            self.model = AutoModel.from_pretrained(
                "ai4bharat/indic-conformer-600m-multilingual",
                **model_kwargs
            )

            logger.info("IndicConformer model initialized successfully")
            logger.info(f"Supported languages: {', '.join(self.supported_languages)}")

        except Exception as e:
            logger.error(f"Failed to initialize IndicConformer: {e}")
            raise

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported by IndicConformer."""
        return language in self.supported_languages

    async def transcribe_with_chunking(
        self,
        audio_path: str,
        language: str,
        chunk_duration: int = 30,
        overlap: int = 3,
        diarization_segments: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Transcribe audio file with chunking and overlap handling.

        Args:
            audio_path: Path to audio file
            language: Language code (ta, te, kn, ml, etc.)
            chunk_duration: Length of each chunk in seconds (default 30s)
            overlap: Overlap between chunks in seconds (default 3s)
            diarization_segments: Optional speaker diarization segments

        Returns:
            Dictionary containing segments, full text, and metadata
        """
        try:
            logger.info(f"Starting IndicConformer transcription for language: {language}")
            logger.info(f"Chunk size: {chunk_duration}s with {overlap}s overlap")

            # Validate language
            if not self.is_language_supported(language):
                raise ValueError(
                    f"Language '{language}' not supported by IndicConformer. "
                    f"Supported: {', '.join(self.supported_languages)}"
                )

            # Load and preprocess audio
            wav, sr = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if wav.shape[0] > 1:
                logger.info("Converting stereo to mono...")
                wav = torch.mean(wav, dim=0, keepdim=True)

            # Resample to 16kHz if needed (IndicConformer expects 16kHz)
            if sr != 16000:
                logger.info(f"Resampling from {sr}Hz to 16000Hz...")
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
                sr = 16000

            total_duration = wav.shape[1] / sr
            logger.info(f"Total audio duration: {total_duration:.2f}s")

            # Calculate chunk parameters
            chunk_samples = chunk_duration * sr
            overlap_samples = overlap * sr
            step_size = chunk_samples - overlap_samples

            # Process audio in chunks
            transcriptions = []
            segments_with_timing = []
            chunk_num = 0

            for start in range(0, wav.shape[1], step_size):
                end = min(start + chunk_samples, wav.shape[1])
                chunk = wav[:, start:end]
                chunk_num += 1

                # Skip very short chunks at the end (less than 1 second)
                if chunk.shape[1] < sr:
                    logger.debug(f"Skipping very short chunk {chunk_num} (<1s)")
                    continue

                chunk_duration_actual = chunk.shape[1] / sr
                start_time = start / sr
                end_time = end / sr

                logger.info(
                    f"Processing chunk {chunk_num}: {start_time:.1f}s - {end_time:.1f}s "
                    f"({chunk_duration_actual:.1f}s)"
                )

                try:
                    # Transcribe chunk using RNNT decoder (better accuracy)
                    text = self.model(chunk, language, "rnnt")

                    # Remove duplicate text from overlap region
                    if chunk_num > 1 and len(transcriptions) > 0:
                        text = self._remove_overlap(transcriptions[-1], text)

                    # Skip empty transcriptions
                    if not text.strip():
                        logger.warning(f"Chunk {chunk_num} produced empty transcription")
                        continue

                    transcriptions.append(text)

                    # Create segment with timing information
                    segments_with_timing.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text.strip(),
                        "confidence": 0.95,  # IndicConformer doesn't provide confidence scores
                        "chunk_num": chunk_num
                    })

                    logger.debug(f"Chunk {chunk_num} transcribed: {text[:80]}...")

                except Exception as e:
                    logger.error(f"Error transcribing chunk {chunk_num}: {e}")
                    continue

            # Join all chunks into full text
            full_text = ' '.join(transcriptions)

            logger.info(
                f"Transcription completed: {len(segments_with_timing)} chunks, "
                f"{len(full_text)} characters"
            )

            # Map speakers if diarization is available
            if diarization_segments:
                logger.info("Mapping speakers to transcription segments...")
                segments_with_timing = self._map_speakers(
                    segments_with_timing,
                    diarization_segments
                )

            return {
                "segments": segments_with_timing,
                "mode": "chunked_rnnt",
                "total_segments": len(segments_with_timing),
                "language": language,
                "language_name": self.language_names.get(language, language.upper()),
                "full_text": full_text,
                "model": "indic-conformer-600m-multilingual",
                "decoder": "rnnt",
                "chunk_duration": chunk_duration,
                "overlap": overlap,
                "total_chunks": chunk_num
            }

        except Exception as e:
            logger.error(f"IndicConformer transcription failed: {e}")
            raise

    def _remove_overlap(self, previous_text: str, current_text: str) -> str:
        """
        Remove overlapping text between consecutive chunks.

        This is a heuristic approach that looks for matching word sequences
        at the boundary between chunks to avoid duplicate text.

        Args:
            previous_text: Text from the previous chunk
            current_text: Text from the current chunk

        Returns:
            Current text with overlap removed
        """
        # Split into words
        last_words = previous_text.split()[-10:]  # Last 10 words of previous chunk
        current_words = current_text.split()

        # Look for a matching sequence of at least 5 words
        for i in range(len(current_words) - 5):
            if current_words[i:i+5] == last_words[:5]:
                # Found overlap, remove it and return
                overlap_removed = ' '.join(current_words[i+5:])
                logger.debug(f"Overlap detected and removed: {i+5} words")
                return overlap_removed

        # No overlap found, return full current text
        return ' '.join(current_words)

    def _map_speakers(
        self,
        transcription_segments: List[Dict],
        diarization_segments: List[Dict]
    ) -> List[Dict]:
        """
        Map speaker labels from diarization to transcription segments.

        For each transcription segment, finds the dominant speaker based on
        time overlap with diarization segments.

        Args:
            transcription_segments: Segments from transcription with timing
            diarization_segments: Segments from diarization with speaker labels

        Returns:
            Transcription segments with speaker labels added
        """
        # Create a time-to-speaker mapping (second-level granularity)
        speaker_map = {}
        for dia_seg in diarization_segments:
            start = int(dia_seg["start"])
            end = int(dia_seg["end"])
            speaker = dia_seg["speaker"]

            for second in range(start, end + 1):
                speaker_map[second] = speaker

        # Assign speakers to transcription segments
        for trans_seg in transcription_segments:
            seg_start = int(trans_seg["start"])
            seg_end = int(trans_seg["end"])

            # Count occurrences of each speaker in this segment
            speaker_count = {}
            for second in range(seg_start, seg_end + 1):
                if speaker := speaker_map.get(second):
                    speaker_count[speaker] = speaker_count.get(speaker, 0) + 1

            # Assign the dominant speaker
            if speaker_count:
                dominant_speaker = max(speaker_count, key=speaker_count.get)
                trans_seg["speaker"] = dominant_speaker
            else:
                trans_seg["speaker"] = "UNKNOWN"

        return transcription_segments


# Global IndicConformer engine instance
indic_conformer_engine: Optional[IndicConformerEngine] = None


async def get_indic_conformer_engine(config) -> IndicConformerEngine:
    """Get or create IndicConformer engine instance."""
    global indic_conformer_engine
    if indic_conformer_engine is None:
        indic_conformer_engine = IndicConformerEngine(config)
        await indic_conformer_engine.initialize()
    return indic_conformer_engine
