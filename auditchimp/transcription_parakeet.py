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
        Transcribe audio using Parakeet model.

        Args:
            audio_path: Path to audio file
            language: Source language code (optional)
            task: Either 'transcribe' or 'translate'
            progress_callback: Optional callback function for progress updates
            vad_segments: Optional VAD segments from diarization
            diarization_segments: Optional speaker diarization segments
            overlaps: Optional overlapping speech regions

        Returns:
            Dictionary containing transcription results with speaker attribution
        """
        try:
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            if not self.model:
                raise RuntimeError("Model not initialized. Call initialize() first.")

            # NeMo handles stereo→mono conversion internally via channel_selector
            # No need for manual conversion - NeMo averages channels automatically
            log_event(
                logger, "info", "parakeet_transcription_starting",
                "Starting Parakeet transcription with automatic channel handling",
                audio_path=audio_path
            )

            # Transcribe with timestamps using correct NeMo API
            # channel_selector='average' handles stereo audio by averaging L+R channels
            transcriptions = self.model.transcribe(
                audio=[audio_path],
                batch_size=self.batch_size,
                channel_selector='average',  # Automatically convert stereo to mono
                return_hypotheses=True  # Return hypothesis objects for timestamp access
            )

            # Handle return format - can be tuple or list
            if isinstance(transcriptions, tuple):
                # Format: (best_hypotheses, all_hypotheses)
                hypothesis = transcriptions[0][0]  # First file, best hypothesis
            else:
                # Direct list of hypotheses
                hypothesis = transcriptions[0]

            # Extract text
            transcription_text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)

            # Extract timestamps if available
            word_timestamps = []
            segment_timestamps = []
            duration = 0.0

            if hasattr(hypothesis, 'timestep'):
                # Word-level timestamps from hypothesis
                word_timestamps = hypothesis.timestep.get('word', []) if hasattr(hypothesis.timestep, 'get') else []
                segment_timestamps = hypothesis.timestep.get('segment', []) if hasattr(hypothesis.timestep, 'get') else []

            # Estimate duration from audio file if not in hypothesis
            if not duration:
                duration = librosa.get_duration(filename=audio_path)

            # Format segments with timestamps
            segments = []
            if segment_timestamps:
                for segment in segment_timestamps:
                    segments.append({
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "text": segment.get("segment", segment.get("text", "")).strip()
                    })

                    # Update progress if callback provided
                    if progress_callback and duration > 0:
                        progress = (segment.get("end", 0.0) / duration) * 100
                        progress_callback(progress)
            else:
                # Fallback: create single segment if no timestamps available
                segments = [{
                    "start": 0.0,
                    "end": duration,
                    "text": transcription_text.strip()
                }]

            # Process word-level timestamps if available
            word_segments = []
            if word_timestamps:
                for word_info in word_timestamps:
                    if isinstance(word_info, dict):
                        word_segments.append({
                            "start": word_info.get("start", 0.0),
                            "end": word_info.get("end", 0.0),
                            "text": word_info.get("word", word_info.get("text", "")),
                            "confidence": word_info.get("confidence", 0.95),
                            "speaker": None  # Will be assigned if diarization available
                        })

            # If we have diarization data, map speakers to segments
            if diarization_segments and word_segments:
                log_event(logger, "info", "parakeet_speaker_mapping", "Mapping speakers to Parakeet transcription")
                word_segments = self._map_speakers_word_level(
                    word_segments,
                    diarization_segments,
                    overlaps
                )
                # Group words into speaker utterances
                utterances = self._group_words_into_utterances(word_segments)

                return {
                    "text": transcription_text.strip(),
                    "segments": utterances,  # Speaker-attributed utterances
                    "word_segments": word_segments,  # Word-level detail
                    "language": language or "en",
                    "duration": duration,
                    "word_timestamps": word_timestamps,
                    "mode": "word_level_with_speakers",
                    "total_words": len(word_segments),
                    "total_utterances": len(utterances)
                }
            elif diarization_segments and segments:
                # Fallback: map speakers to segment-level transcription
                log_event(logger, "info", "parakeet_speaker_mapping_segments", "Mapping speakers to segment-level transcription")
                segments = self._map_speakers_to_segments(segments, diarization_segments)

                return {
                    "text": transcription_text.strip(),
                    "segments": segments,
                    "language": language or "en",
                    "duration": duration,
                    "word_timestamps": word_timestamps,
                    "mode": "segment_level_with_speakers"
                }
            else:
                # No diarization: return original format
                return {
                    "text": transcription_text.strip(),
                    "segments": segments,
                    "language": language or "en",
                    "duration": duration,
                    "word_timestamps": word_timestamps
                }

        except Exception as e:
            log_event(logger, "error", "parakeet_transcription_failed", "Parakeet transcription error", error=str(e))
            raise

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

    def _map_speakers_to_segments(
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

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.model:
            try:
                self.model.cpu()
                del self.model
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error cleaning up Parakeet model: {e}")