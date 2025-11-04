"""IndicConformer transcription engine for Indian languages with chunking support."""

import asyncio
import logging
from typing import Dict, List, Optional
import torch
import torchaudio

from .logger import log_event

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

        # Handle device selection with proper "auto" detection
        device = config.INDIC_DEVICE
        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cuda", "cpu"]:
            # Validate CUDA availability if explicitly requested
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            else:
                self.device = device
        else:
            logger.warning(f"Invalid device '{device}', defaulting to auto-detection")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"IndicConformer will use device: {self.device}")

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
            log_event(logger, "info", "indic_init_started", "Initializing IndicConformer model", device=self.device)

            from transformers import AutoModel

            # Prepare kwargs for model loading
            model_kwargs = {
                "trust_remote_code": True,
                "cache_dir": str(self.config.MODEL_CACHE_DIR)
            }

            # Add token if available (required for gated models)
            if self.config.HF_TOKEN:
                model_kwargs["token"] = self.config.HF_TOKEN

            self.model = AutoModel.from_pretrained(
                "ai4bharat/indic-conformer-600m-multilingual",
                **model_kwargs
            )

            # Move model to GPU if configured
            if self.device == "cuda":
                self.model = self.model.cuda()
                # Set model to eval mode for inference
                self.model.eval()

                # Optional: Use half precision for faster inference if GPU supports it
                # Uncomment below for FP16 optimization (requires GPU with Tensor Cores)
                # self.model = self.model.half()

            log_event(
                logger, "success", "indic_init_completed", "IndicConformer model initialized",
                device=self.device,
                languages=len(self.supported_languages)
            )

        except Exception as e:
            log_event(logger, "error", "indic_init_failed", "Failed to initialize IndicConformer", error=str(e))
            raise

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported by IndicConformer."""
        return language in self.supported_languages

    async def transcribe_with_vad_chunking(
        self,
        audio_path: str,
        language: str,
        vad_segments: List[Dict],
        diarization_segments: Optional[List[Dict]] = None,
        overlaps: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Transcribe audio using VAD segments for precise chunking.

        Args:
            audio_path: Path to audio file
            language: Language code (ta, te, kn, ml, etc.)
            vad_segments: VAD segments from NeMo (speech boundaries)
            diarization_segments: Speaker diarization segments
            overlaps: Overlapping speech regions

        Returns:
            Dictionary containing word-level segments with speaker labels
        """
        try:
            logger.info("=" * 80)
            logger.info(f"INDIC: Starting VAD-based transcription")
            logger.info(f"INDIC: Language: {self.language_names.get(language, language)}, VAD segments: {len(vad_segments)}")
            logger.info("=" * 80)

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
                logger.info("INDIC: Converting stereo to mono")
                wav = torch.mean(wav, dim=0, keepdim=True)

            # Resample to 16kHz if needed
            if sr != 16000:
                logger.info(f"INDIC: Resampling from {sr}Hz to 16000Hz")
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
                sr = 16000

            total_duration = wav.shape[1] / sr
            logger.info(f"INDIC: Audio prepared - Duration: {total_duration:.2f}s, Sample rate: 16kHz")

            # Process each VAD segment (speech-only chunks)
            all_word_segments = []
            segment_num = 0

            logger.info(f"INDIC: Processing {len(vad_segments)} VAD segments")

            for vad_seg in vad_segments:
                start_time = vad_seg["start"]
                end_time = vad_seg["end"]
                segment_num += 1

                # Extract audio chunk based on VAD boundaries
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                chunk = wav[:, start_sample:end_sample]

                # Skip very short chunks (< 0.5s)
                chunk_duration = (end_sample - start_sample) / sr
                if chunk_duration < 0.5:
                    continue

                logger.info(f"INDIC: Transcribing segment {segment_num}/{len(vad_segments)} ({start_time:.2f}s - {end_time:.2f}s, duration: {chunk_duration:.2f}s)")

                try:
                    # Move chunk to same device as model for GPU acceleration
                    if self.device == "cuda" and chunk.device.type != 'cuda':
                        chunk = chunk.cuda()

                    # Transcribe VAD segment using RNNT decoder
                    text = self.model(chunk, language, "rnnt")

                    if not text.strip():
                        continue

                    # Split into words and create word-level segments
                    words = text.strip().split()
                    if not words:
                        continue

                    # Estimate word timing (proportional distribution)
                    word_duration = chunk_duration / len(words)

                    for word_idx, word in enumerate(words):
                        word_start = start_time + (word_idx * word_duration)
                        word_end = start_time + ((word_idx + 1) * word_duration)

                        all_word_segments.append({
                            "start": round(word_start, 3),
                            "end": round(word_end, 3),
                            "text": word,
                            "confidence": 0.95,  # IndicConformer doesn't provide confidence
                            "vad_segment": segment_num,
                            "speaker": None  # Will be assigned later
                        })

                except Exception as e:
                    log_event(logger, "warning", "vad_segment_error", "Error transcribing VAD segment", segment=segment_num, error=str(e))
                    continue

            logger.info(f"INDIC: Transcription completed - {len(all_word_segments)} words from {segment_num} VAD segments")

            # Map speakers to word-level segments
            if diarization_segments:
                logger.info("INDIC: Mapping speakers to word-level segments")
                all_word_segments = self._map_speakers_word_level(
                    all_word_segments,
                    diarization_segments,
                    overlaps
                )
                logger.info(f"INDIC: Speaker mapping completed")

            # Combine words into utterances for better readability
            logger.info("INDIC: Grouping words into speaker utterances")
            utterances = self._group_words_into_utterances(all_word_segments)

            # Calculate full text
            full_text = ' '.join([word["text"] for word in all_word_segments])

            logger.info("=" * 80)
            logger.info(f"INDIC: Complete - {len(utterances)} utterances, {len(all_word_segments)} words")
            logger.info("=" * 80)

            return {
                "segments": utterances,           # Grouped utterances
                "word_segments": all_word_segments,  # Word-level detail
                "mode": "vad_chunked_word_level",
                "total_words": len(all_word_segments),
                "total_utterances": len(utterances),
                "language": language,
                "language_name": self.language_names.get(language, language.upper()),
                "full_text": full_text,
                "model": "indic-conformer-600m-multilingual",
                "decoder": "rnnt",
                "vad_segments_processed": segment_num,
                "overlaps_detected": len(overlaps) if overlaps else 0,
                "duration": total_duration
            }

        except Exception as e:
            log_event(logger, "error", "indic_vad_transcription_failed", "VAD-based transcription failed", error=str(e))
            raise

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
                    # Move chunk to same device as model for GPU acceleration
                    if self.device == "cuda" and chunk.device.type != 'cuda':
                        chunk = chunk.cuda()

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
                "total_chunks": chunk_num,
                "duration": total_duration
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

    def _map_speakers_word_level(
        self,
        word_segments: List[Dict],
        diarization_segments: List[Dict],
        overlaps: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Map speakers to word-level segments with overlap detection.

        Args:
            word_segments: Word-level transcription segments
            diarization_segments: Speaker diarization segments
            overlaps: Overlapping speech regions

        Returns:
            Word segments with speaker labels and overlap markers
        """
        # Create high-resolution time-to-speaker mapping (100ms granularity)
        speaker_map = {}
        for dia_seg in diarization_segments:
            start_ms = int(dia_seg["start"] * 10)  # Convert to 100ms units
            end_ms = int(dia_seg["end"] * 10)
            speaker = dia_seg["speaker"]

            for time_100ms in range(start_ms, end_ms + 1):
                if time_100ms not in speaker_map:
                    speaker_map[time_100ms] = []
                speaker_map[time_100ms].append(speaker)

        # Create overlap map for quick lookup
        overlap_map = {}
        if overlaps:
            for overlap in overlaps:
                start_ms = int(overlap["start"] * 10)
                end_ms = int(overlap["end"] * 10)
                for time_100ms in range(start_ms, end_ms + 1):
                    overlap_map[time_100ms] = overlap["speakers"]

        # Assign speakers to each word
        for word_seg in word_segments:
            word_start_ms = int(word_seg["start"] * 10)
            word_end_ms = int(word_seg["end"] * 10)

            # Collect all speakers during this word's timespan
            speakers_during_word = []
            overlap_detected = False

            for time_100ms in range(word_start_ms, word_end_ms + 1):
                # Check for speakers at this time
                if time_100ms in speaker_map:
                    speakers_during_word.extend(speaker_map[time_100ms])

                # Check for overlap at this time
                if time_100ms in overlap_map:
                    overlap_detected = True

            if not speakers_during_word:
                word_seg["speaker"] = "UNKNOWN"
                word_seg["overlap"] = False
                continue

            # Count speaker occurrences
            speaker_count = {}
            for spk in speakers_during_word:
                speaker_count[spk] = speaker_count.get(spk, 0) + 1

            # Get dominant speaker
            dominant_speaker = max(speaker_count, key=speaker_count.get)

            # Check if there are multiple speakers (overlap)
            unique_speakers = list(set(speakers_during_word))
            if len(unique_speakers) > 1 or overlap_detected:
                word_seg["speaker"] = dominant_speaker
                word_seg["overlap"] = True
                word_seg["all_speakers"] = sorted(unique_speakers)
            else:
                word_seg["speaker"] = dominant_speaker
                word_seg["overlap"] = False

        return word_segments

    def _group_words_into_utterances(self, word_segments: List[Dict]) -> List[Dict]:
        """
        Group words into speaker utterances for better readability.

        Combines consecutive words from the same speaker into utterances,
        preserving overlap information.

        Args:
            word_segments: Word-level segments with speaker labels

        Returns:
            List of utterances (grouped words by speaker)
        """
        if not word_segments:
            return []

        utterances = []
        current_utterance = {
            "speaker": word_segments[0].get("speaker"),
            "words": [word_segments[0]],
            "start": word_segments[0]["start"],
            "end": word_segments[0]["end"],
            "has_overlap": word_segments[0].get("overlap", False)
        }

        for word_seg in word_segments[1:]:
            current_speaker = word_seg.get("speaker")
            prev_speaker = current_utterance["speaker"]

            # Check if we should continue current utterance or start new one
            # Continue if: same speaker AND time gap < 1.0 second
            time_gap = word_seg["start"] - current_utterance["end"]

            if current_speaker == prev_speaker and time_gap < 1.0:
                # Continue current utterance
                current_utterance["words"].append(word_seg)
                current_utterance["end"] = word_seg["end"]
                if word_seg.get("overlap", False):
                    current_utterance["has_overlap"] = True
            else:
                # Finalize current utterance
                current_utterance["text"] = ' '.join([w["text"] for w in current_utterance["words"]])
                current_utterance["word_count"] = len(current_utterance["words"])

                # Calculate confidence (average)
                avg_confidence = sum(w.get("confidence", 0.95) for w in current_utterance["words"]) / len(current_utterance["words"])
                current_utterance["confidence"] = round(avg_confidence, 3)

                utterances.append(current_utterance)

                # Start new utterance
                current_utterance = {
                    "speaker": current_speaker,
                    "words": [word_seg],
                    "start": word_seg["start"],
                    "end": word_seg["end"],
                    "has_overlap": word_seg.get("overlap", False)
                }

        # Add the last utterance
        current_utterance["text"] = ' '.join([w["text"] for w in current_utterance["words"]])
        current_utterance["word_count"] = len(current_utterance["words"])
        avg_confidence = sum(w.get("confidence", 0.95) for w in current_utterance["words"]) / len(current_utterance["words"])
        current_utterance["confidence"] = round(avg_confidence, 3)
        utterances.append(current_utterance)

        logger.info(f"Grouped {len(word_segments)} words into {len(utterances)} utterances")
        return utterances

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
