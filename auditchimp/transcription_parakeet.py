"""Parakeet-based transcription engine."""

import logging
import torch
import librosa
from pathlib import Path
from typing import Dict, List, Optional

from .logger import log_event

logger = logging.getLogger(__name__)

class ParakeetTranscriber:
    """Transcription engine using NVIDIA Parakeet TDT model."""
    
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = None,
        compute_type: str = "float16",
        batch_size: int = 32,  # Default batch size from model card
        chunk_length_s: int = 30,  # Chunk size for streaming
        max_new_tokens: int = 256  # Maximum new tokens to generate
    ):
        self.model_name = model_name

        # Handle device selection with proper "auto" detection
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

        logger.info(f"Parakeet will use device: {self.device}")

        self.compute_type = compute_type
        self.batch_size = batch_size
        self.model = None

    async def initialize(self):
        """Initialize the model and processor."""
        try:
            log_event(logger, "info", "parakeet_init_started", "Loading Parakeet model", model=self.model_name, device=self.device)

            # Load model with appropriate compute type
            torch_dtype = torch.float16 if self.compute_type == "float16" and self.device == "cuda" else torch.float32

            import nemo.collections.asr as nemo_asr

            # Load the ASR model from NeMo
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )

            # Move model to specified device
            if self.device == "cuda":
                self.model = self.model.cuda()

            # Configure for long-form audio support
            self.model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[256, 256]
            )

            log_event(logger, "success", "parakeet_init_completed", "Parakeet model loaded", device=self.device, batch_size=self.batch_size)
        except Exception as e:
            log_event(logger, "error", "parakeet_init_failed", "Error loading Parakeet model", error=str(e))
            raise

    async def transcribe_with_vad_chunking(
        self,
        audio_path: str,
        language: Optional[str] = None,
        vad_segments: Optional[List[Dict]] = None,
        diarization_segments: Optional[List[Dict]] = None,
        overlaps: Optional[List[Dict]] = None,
        progress_callback=None
    ) -> Dict:
        """
        Transcribe audio using VAD segments for precise chunking (like IndicConformer).

        Args:
            audio_path: Path to audio file
            language: Source language code (optional)
            vad_segments: VAD segments from NeMo (speech boundaries)
            diarization_segments: Optional speaker diarization segments
            overlaps: Optional overlapping speech regions
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary containing word-level transcription with speaker attribution
        """
        try:
            import torchaudio

            log_event(
                logger, "info", "parakeet_vad_transcription_started",
                "Starting VAD-based Parakeet transcription",
                language=language,
                vad_segments=len(vad_segments) if vad_segments else 0
            )

            if not self.model:
                raise RuntimeError("Model not initialized. Call initialize() first.")

            # Load and preprocess audio
            wav, sr = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if wav.shape[0] > 1:
                logger.info("Converting stereo to mono...")
                import torch
                wav = torch.mean(wav, dim=0, keepdim=True)

            # Resample to 16kHz if needed
            if sr != 16000:
                logger.info(f"Resampling from {sr}Hz to 16000Hz...")
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
                sr = 16000

            total_duration = wav.shape[1] / sr

            # Process each VAD segment (speech-only chunks)
            all_word_segments = []
            segment_num = 0

            if not vad_segments:
                # Fallback: process entire audio as one segment
                vad_segments = [{"start": 0.0, "end": total_duration, "type": "speech"}]

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

                try:
                    # Save chunk to temporary file for NeMo processing
                    import tempfile
                    import os

                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                        torchaudio.save(temp_path, chunk, sr)

                    # Transcribe VAD segment using NeMo
                    transcriptions = self.model.transcribe(
                        audio=[temp_path],
                        batch_size=1,
                        return_hypotheses=True
                    )

                    # Clean up temp file
                    os.unlink(temp_path)

                    # Extract hypothesis
                    if isinstance(transcriptions, tuple):
                        hypothesis = transcriptions[0][0]
                    else:
                        hypothesis = transcriptions[0]

                    text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)

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
                            "confidence": 0.95,
                            "vad_segment": segment_num,
                            "speaker": None  # Will be assigned later
                        })

                except Exception as e:
                    log_event(logger, "warning", "vad_segment_error", "Error transcribing VAD segment", segment=segment_num, error=str(e))
                    continue

            log_event(
                logger, "success", "parakeet_vad_transcription_completed",
                "VAD-based transcription completed",
                total_words=len(all_word_segments),
                vad_segments_processed=segment_num
            )

            # Map speakers to word-level segments
            if diarization_segments:
                all_word_segments = self._map_speakers_word_level(
                    all_word_segments,
                    diarization_segments,
                    overlaps
                )

            # Combine words into utterances for better readability
            utterances = self._group_words_into_utterances(all_word_segments)

            # Calculate full text
            full_text = ' '.join([word["text"] for word in all_word_segments])

            return {
                "text": full_text,
                "segments": utterances,           # Grouped utterances
                "word_segments": all_word_segments,  # Word-level detail
                "mode": "vad_chunked_word_level",
                "total_words": len(all_word_segments),
                "total_utterances": len(utterances),
                "language": language or "en",
                "duration": total_duration,
                "vad_segments_processed": segment_num
            }

        except Exception as e:
            log_event(logger, "error", "parakeet_vad_transcription_failed", "VAD-based transcription failed", error=str(e))
            raise

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        enable_long_form: bool = True,
        local_attention_window: int = 256,
        progress_callback=None,
        vad_segments: Optional[List[Dict]] = None,
        diarization_segments: Optional[List[Dict]] = None,
        overlaps: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Transcribe audio using Parakeet model with VAD-based chunking.

        This method now uses the same VAD-based approach as IndicConformer
        for accurate speaker attribution.

        Args:
            audio_path: Path to audio file
            language: Source language code (optional)
            task: Either 'transcribe' or 'translate'
            progress_callback: Optional callback function for progress updates
            vad_segments: VAD segments from diarization (required for speaker attribution)
            diarization_segments: Speaker diarization segments (required for speaker attribution)
            overlaps: Optional overlapping speech regions

        Returns:
            Dictionary containing transcription results with speaker attribution
        """
        # Always use VAD-based chunking (same approach as Indic)
        return await self.transcribe_with_vad_chunking(
            audio_path=audio_path,
            language=language,
            vad_segments=vad_segments,
            diarization_segments=diarization_segments,
            overlaps=overlaps,
            progress_callback=progress_callback
        )

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

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.model:
            try:
                self.model.cpu()
                del self.model
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error cleaning up Parakeet model: {e}")