"""Speaker diarization module using NeMo MSDD for GPU processing."""

import asyncio
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
import torch

from .logger import log_event

logger = logging.getLogger(__name__)


class DiarizationEngine:
    """Speaker diarization engine using NeMo MSDD."""

    def __init__(self, config):
        self.config = config
        self.msdd_model = None
        self.device = "cuda" if torch.cuda.is_available() and config.DIARIZATION_USE_GPU else "cpu"

    async def initialize(self):
        """Initialize NeMo diarization models."""
        try:
            log_event(logger, "info", "diarization_init_started", "Initializing NeMo diarization pipeline", device=self.device)

            # Import NeMo dependencies
            from nemo.collections.asr.models import ClusteringDiarizer
            from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels

            # Initialize NeMo ClusteringDiarizer
            self.msdd_model = ClusteringDiarizer(cfg=self._get_diarizer_config())

            log_event(logger, "success", "diarization_init_completed", "NeMo diarization pipeline initialized", device=self.device)

        except ImportError as e:
            log_event(logger, "error", "diarization_init_failed", "NeMo not installed", error=str(e))
            raise RuntimeError(f"NeMo not available: {e}")
        except Exception as e:
            log_event(logger, "error", "diarization_init_failed", "Failed to initialize NeMo diarization", error=str(e))
            raise

    def _get_diarizer_config(self):
        """Get complete NeMo diarizer configuration with all required parameters."""
        from omegaconf import OmegaConf

        # Complete NeMo diarizer configuration following official structure
        config = {
            'device': self.device,
            'num_workers': 0,  # Force 0 to avoid shared memory issues
            'sample_rate': 16000,
            'batch_size': 1,
            'verbose': False,  # Disable verbose to reduce NeMo logs
            'diarizer': {
                # Core diarizer settings
                'manifest_filepath': None,  # Set per audio file
                'out_dir': str(self.config.TEMP_DIR),
                'oracle_vad': False,
                'collar': 0.25,  # Collar value for scoring tolerance
                'ignore_overlap': True,  # Ignore overlap segments in scoring

                # Worker settings - Force 0 to avoid shared memory issues
                'num_workers': 0,
                'sample_rate': 16000,

                # VAD Configuration
                'vad': {
                    'model_path': 'vad_multilingual_marblenet',
                    'external_vad_manifest': None,
                    'parameters': {
                        'window_length_in_sec': 0.15,
                        'shift_length_in_sec': 0.01,
                        'smoothing': 'median',
                        'overlap': 0.875,
                        'onset': 0.4,  # Standard threshold for multi-speaker
                        'offset': 0.7,
                        'pad_onset': 0.05,
                        'pad_offset': -0.1,
                        'min_duration_on': 0.2,
                        'min_duration_off': 0.2,
                        'filter_speech_first': True,
                    }
                },

                # Speaker Embeddings Configuration - Optimized for quick speaker turns
                'speaker_embeddings': {
                    'model_path': 'titanet_large',
                    'parameters': {
                        'window_length_in_sec': 1.0,  # REDUCED: Shorter window for quick speaker turns
                        'shift_length_in_sec': 0.5,   # REDUCED: More frequent sampling
                        'multiscale_weights': None,
                        'save_embeddings': False,
                    }
                },

                # Clustering Configuration - Optimized for multi-speaker detection
                'clustering': {
                    'parameters': {
                        'oracle_num_speakers': False,
                        'max_num_speakers': 4,  # Limit to 4 speakers for better accuracy
                        'enhanced_count_thres': 40,  # LOWERED: Trigger enhanced counting earlier
                        'max_rp_threshold': 0.15,  # LOWERED: Be more conservative about merging speakers
                        'sparse_search_volume': 30,
                        'maj_vote_spk_count': True,  # ENABLED: Use majority voting for robust speaker counting
                        'cuda': self.config.DIARIZATION_USE_GPU and torch.cuda.is_available(),  # Enable GPU for clustering
                    }
                },
            }
        }

        return OmegaConf.create(config)

    def _extract_vad_segments_from_diarization(self, diarization_segments: List[Dict]) -> List[Dict]:
        """
        Extract VAD segments from diarization results.

        Since ClusteringDiarizer already performs VAD internally,
        we use the diarization segments as VAD segments for simplicity.

        Args:
            diarization_segments: Speaker segments from diarization

        Returns:
            List of VAD segments with start/end timestamps
        """
        try:
            # Convert diarization segments to VAD format (speech regions)
            vad_segments = []

            # Sort segments by start time
            sorted_segments = sorted(diarization_segments, key=lambda x: x["start"])

            # Merge overlapping or adjacent segments to create continuous speech regions
            if not sorted_segments:
                return []

            current_segment = {
                "start": sorted_segments[0]["start"],
                "end": sorted_segments[0]["end"],
                "type": "speech"
            }

            for seg in sorted_segments[1:]:
                # If segments overlap or are very close (within 0.1s), merge them
                if seg["start"] <= current_segment["end"] + 0.1:
                    current_segment["end"] = max(current_segment["end"], seg["end"])
                else:
                    # Save current segment and start new one
                    vad_segments.append(current_segment)
                    current_segment = {
                        "start": seg["start"],
                        "end": seg["end"],
                        "type": "speech"
                    }

            # Don't forget the last segment
            vad_segments.append(current_segment)

            return vad_segments

        except Exception as e:
            log_event(logger, "warning", "vad_extraction_failed", "VAD extraction failed, using fallback", error=str(e))
            # Return diarization segments as-is if extraction fails
            return [{"start": s["start"], "end": s["end"], "type": "speech"}
                    for s in diarization_segments]

    def _detect_overlapping_speech(
        self,
        diarization_segments: List[Dict],
        time_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Detect overlapping speech regions where multiple speakers talk simultaneously.

        Args:
            diarization_segments: Speaker segments from diarization
            time_threshold: Minimum overlap duration to consider (seconds)

        Returns:
            List of overlap segments with participating speakers
        """
        overlaps = []

        # Sort segments by start time
        sorted_segments = sorted(diarization_segments, key=lambda x: x["start"])

        for i, seg1 in enumerate(sorted_segments):
            for seg2 in sorted_segments[i+1:]:
                # Check for overlap
                overlap_start = max(seg1["start"], seg2["start"])
                overlap_end = min(seg1["end"], seg2["end"])
                overlap_duration = overlap_end - overlap_start

                if overlap_duration >= time_threshold:
                    # Found overlapping speech
                    speakers = sorted([seg1["speaker"], seg2["speaker"]])
                    overlaps.append({
                        "start": overlap_start,
                        "end": overlap_end,
                        "duration": overlap_duration,
                        "speakers": speakers,
                        "type": "overlap"
                    })

        return overlaps

    def _diarize_nemo_sync(self, audio_path: str) -> Dict:
        """
        Diarize using NeMo ClusteringDiarizer (synchronous version for thread pool).

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with segments, VAD segments, overlaps, and metadata
        """
        try:
            # Convert to mono 16kHz if needed (NeMo requires mono audio at 16kHz)
            import librosa
            import soundfile as sf
            import shutil
            import uuid

            logger.info("=" * 80)
            logger.info("DIARIZATION: Starting speaker diarization pipeline")
            logger.info("=" * 80)

            # Load audio and convert to mono 16kHz
            logger.info(f"DIARIZATION: Loading audio file: {Path(audio_path).name}")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Create unique output directory for this diarization run to avoid conflicts
            unique_id = str(uuid.uuid4())[:8]
            output_dir = Path(self.config.TEMP_DIR) / f"diarization_{unique_id}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save as temporary mono file
            mono_audio_path = output_dir / f"{Path(audio_path).stem}_mono.wav"
            sf.write(str(mono_audio_path), audio, 16000, format='WAV', subtype='PCM_16')

            # Get audio duration
            duration = librosa.get_duration(y=audio, sr=16000)
            logger.info(f"DIARIZATION: Audio duration: {duration:.2f}s, Sample rate: 16kHz, Channels: Mono")
            logger.info(f"DIARIZATION: Processing device: {self.device.upper()}")

            # Clean up NeMo output directories if they exist (prevents "File exists" errors)
            nemo_dirs = ['speaker_outputs', 'pred_rttms']
            for dir_name in nemo_dirs:
                dir_path = output_dir / dir_name
                if dir_path.exists():
                    shutil.rmtree(dir_path)

            # Create temporary manifest file
            manifest_path = output_dir / f"{Path(audio_path).stem}_manifest.json"

            # Create manifest entry with mono audio path
            # IMPORTANT: NeMo expects specific format
            manifest_data = {
                "audio_filepath": str(mono_audio_path),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,  # Let NeMo detect automatically
                "rttm_filepath": None,
                "uem_filepath": None
            }

            # Write manifest (one JSON object per line)
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)
                f.write('\n')

            # Verify audio file exists
            if not mono_audio_path.exists():
                raise FileNotFoundError(f"Mono audio file not found: {mono_audio_path}")

            # Update config with manifest path and unique output directory
            self.msdd_model._cfg.diarizer.manifest_filepath = str(manifest_path)
            self.msdd_model._cfg.diarizer.out_dir = str(output_dir)

            # Run diarization
            logger.info("DIARIZATION: Step 1/3 - Running Voice Activity Detection (VAD)")
            logger.info("DIARIZATION: Step 2/3 - Extracting speaker embeddings")
            logger.info("DIARIZATION: Step 3/3 - Clustering speakers")

            # Suppress NeMo's verbose logging temporarily
            import logging as std_logging
            nemo_logger = std_logging.getLogger('nemo_logger')
            original_level = nemo_logger.level
            nemo_logger.setLevel(std_logging.WARNING)

            self.msdd_model.diarize()

            # Restore logging level
            nemo_logger.setLevel(original_level)

            logger.info("DIARIZATION: Pipeline completed, processing results")

            # NeMo saves RTTM output to pred_rttms subdirectory
            # The filename uses the base name from the audio file in the manifest
            pred_rttms_dir = output_dir / "pred_rttms"

            # Find RTTM file - NeMo might use different naming conventions
            rttm_path = None
            if pred_rttms_dir.exists():
                # Try exact match first
                expected_rttm = pred_rttms_dir / f"{Path(audio_path).stem}_mono.rttm"
                if expected_rttm.exists():
                    rttm_path = expected_rttm
                else:
                    # Search for any .rttm file in the directory
                    rttm_files = list(pred_rttms_dir.glob("*.rttm"))
                    if rttm_files:
                        rttm_path = rttm_files[0]
                        log_event(logger, "info", "rttm_found_alternative", "RTTM file found with alternative name",
                                 expected=str(expected_rttm), found=str(rttm_path))

            segments = []
            unique_speakers = set()
            if rttm_path and rttm_path.exists():
                with open(rttm_path, 'r') as f:
                    rttm_content = f.read()

                    # Parse RTTM lines
                    for line in rttm_content.strip().split('\n'):
                        if not line.strip():
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            start = float(parts[3])
                            duration = float(parts[4])
                            speaker = parts[7]
                            unique_speakers.add(speaker)

                            segments.append({
                                "start": start,
                                "end": start + duration,
                                "speaker": speaker
                            })

                logger.info(f"DIARIZATION: Detected {len(unique_speakers)} speakers in {len(segments)} segments")
                logger.info(f"DIARIZATION: Speaker labels: {', '.join(sorted(unique_speakers))}")
            else:
                search_path = str(pred_rttms_dir) if pred_rttms_dir.exists() else str(output_dir)
                log_event(logger, "warning", "rttm_not_found", "RTTM file not found", search_path=search_path)

            # Clean up entire output directory with all temp files
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                log_event(logger, "warning", "cleanup_failed", "Failed to clean up temp directory", error=str(e))

            if not segments:
                log_event(logger, "warning", "no_speakers_detected", "No speakers detected, using fallback")
                segments = [{"start": 0.0, "end": duration, "speaker": "SPEAKER_00"}]

            # Extract VAD segments from diarization results
            vad_segments = self._extract_vad_segments_from_diarization(segments)
            logger.info(f"DIARIZATION: Extracted {len(vad_segments)} VAD segments from diarization")

            # Detect overlapping speech
            overlaps = self._detect_overlapping_speech(segments)
            if overlaps:
                logger.info(f"DIARIZATION: Detected {len(overlaps)} overlapping speech regions")
            else:
                logger.info("DIARIZATION: No overlapping speech detected")

            num_speakers = len(set(s['speaker'] for s in segments))

            logger.info("=" * 80)
            logger.info(f"DIARIZATION: Complete - {num_speakers} speakers, {len(segments)} segments, {len(vad_segments)} VAD segments")
            logger.info("=" * 80)

            result = {
                "segments": segments,
                "vad_segments": vad_segments,
                "overlaps": overlaps,
                "method": "nemo",
                "device": self.device,
                "num_speakers": num_speakers,
                "vad_enabled": True
            }

            return result

        except Exception as e:
            log_event(logger, "error", "diarization_failed", "NeMo diarization failed, using fallback", error=str(e))

            # Fallback: return single speaker
            import librosa
            import soundfile as sf
            try:
                # Try soundfile first (more reliable for various formats)
                info = sf.info(audio_path)
                duration = info.duration
            except:
                try:
                    # Fall back to librosa with explicit loading
                    y, sr = librosa.load(audio_path, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                except:
                    duration = 0.0

            return {
                "segments": [{"start": 0.0, "end": duration, "speaker": "SPEAKER_00"}],
                "vad_segments": [{"start": 0.0, "end": duration, "type": "speech"}],
                "overlaps": [],
                "method": "fallback",
                "device": "none",
                "num_speakers": 1,
                "vad_enabled": False
            }

    async def diarize(self, audio_path: str) -> Dict:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with diarization results
        """
        # NeMo diarization runs synchronously
        return self._diarize_nemo_sync(audio_path)


# Global diarization engine instance
diarization_engine: Optional[DiarizationEngine] = None


async def get_diarization_engine(config) -> DiarizationEngine:
    """Get or create diarization engine instance."""
    global diarization_engine
    if diarization_engine is None:
        diarization_engine = DiarizationEngine(config)
        await diarization_engine.initialize()
    return diarization_engine
