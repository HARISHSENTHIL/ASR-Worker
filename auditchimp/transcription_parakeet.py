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
        progress_callback=None
    ) -> Dict:
        """
        Transcribe audio using Parakeet model.
        
        Args:
            audio_path: Path to audio file
            language: Source language code (optional)
            task: Either 'transcribe' or 'translate'
            progress_callback: Optional callback function for progress updates
        
        Returns:
            Dictionary containing transcription results
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

            return {
                "text": transcription_text.strip(),
                "segments": segments,
                "language": language or "en",
                "duration": duration,
                "word_timestamps": word_timestamps  # Include word-level timestamps
            }

        except Exception as e:
            log_event(logger, "error", "parakeet_transcription_failed", "Parakeet transcription error", error=str(e))
            raise

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.model:
            try:
                self.model.cpu()
                del self.model
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error cleaning up Parakeet model: {e}")