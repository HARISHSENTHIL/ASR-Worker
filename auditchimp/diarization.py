"""Speaker diarization module using NeMo MSDD for GPU processing."""

import asyncio
import logging
import tempfile
import json
from typing import Dict, List, Optional
from pathlib import Path
import torch

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
            logger.info("Initializing NeMo diarization pipeline...")

            # Import NeMo dependencies
            from nemo.collections.asr.models import ClusteringDiarizer
            from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels

            logger.info(f"Using device: {self.device}")

            # Initialize NeMo ClusteringDiarizer
            # This handles VAD, speaker embedding, and clustering automatically
            self.msdd_model = ClusteringDiarizer(cfg=self._get_diarizer_config())

            logger.info("NeMo diarization pipeline initialized successfully")

        except ImportError as e:
            logger.error(f"NeMo not installed. Install with: pip install nemo_toolkit[asr]")
            raise RuntimeError(f"NeMo not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize NeMo diarization: {e}")
            raise

    def _get_diarizer_config(self):
        """Get complete NeMo diarizer configuration with all required parameters."""
        from omegaconf import OmegaConf

        # Complete NeMo diarizer configuration following official structure
        config = {
            'device': self.device,
            'num_workers': 0 if self.device == 'cpu' else 2,
            'sample_rate': 16000,
            'batch_size': 1,
            'verbose': True,  # Enable verbose logging to debug
            'diarizer': {
                # Core diarizer settings
                'manifest_filepath': None,  # Set per audio file
                'out_dir': str(self.config.TEMP_DIR),
                'oracle_vad': False,
                'collar': 0.25,  # Collar value for scoring tolerance
                'ignore_overlap': True,  # Ignore overlap segments in scoring

                # Worker settings
                'num_workers': 0 if self.device == 'cpu' else 2,
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

            logger.info(f"Extracted {len(vad_segments)} VAD segments from {len(diarization_segments)} diarization segments")

            return vad_segments

        except Exception as e:
            logger.error(f"VAD extraction failed: {e}")
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

        logger.info(f"Detected {len(overlaps)} overlapping speech regions")
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
            logger.info("Starting NeMo diarization...")

            # Convert to mono 16kHz if needed (NeMo requires mono audio at 16kHz)
            import librosa
            import soundfile as sf
            import shutil
            import uuid

            # Load audio and convert to mono 16kHz
            logger.info(f"Loading audio file: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Create unique output directory for this diarization run to avoid conflicts
            unique_id = str(uuid.uuid4())[:8]
            output_dir = Path(self.config.TEMP_DIR) / f"diarization_{unique_id}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save as temporary mono file
            mono_audio_path = output_dir / f"{Path(audio_path).stem}_mono.wav"
            sf.write(mono_audio_path, audio, 16000)
            logger.info(f"Converted to mono 16kHz: {mono_audio_path}")

            # Get audio duration
            duration = librosa.get_duration(y=audio, sr=16000)
            logger.info(f"Audio duration: {duration:.2f}s")

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

            logger.info(f"Created manifest file: {manifest_path}")
            logger.info(f"Manifest content: {manifest_data}")

            # Verify audio file exists
            if not mono_audio_path.exists():
                raise FileNotFoundError(f"Mono audio file not found: {mono_audio_path}")
            logger.info(f"Mono audio file exists: {mono_audio_path} ({mono_audio_path.stat().st_size} bytes)")

            # Update config with manifest path and unique output directory
            self.msdd_model._cfg.diarizer.manifest_filepath = str(manifest_path)
            self.msdd_model._cfg.diarizer.out_dir = str(output_dir)

            logger.info(f"Updated diarizer config:")
            logger.info(f"  - manifest_filepath: {self.msdd_model._cfg.diarizer.manifest_filepath}")
            logger.info(f"  - out_dir: {self.msdd_model._cfg.diarizer.out_dir}")
            logger.info(f"  - oracle_vad: {self.msdd_model._cfg.diarizer.oracle_vad}")
            logger.info(f"  - clustering params: {self.msdd_model._cfg.diarizer.clustering.parameters}")

            # Run diarization
            logger.info("Running NeMo diarization...")
            self.msdd_model.diarize()
            logger.info("Diarization completed, reading results...")

            # NeMo saves RTTM output to pred_rttms subdirectory
            # The filename uses the base name from the audio file in the manifest
            pred_rttms_dir = output_dir / "pred_rttms"
            rttm_filename = f"{Path(audio_path).stem}_mono.rttm"
            rttm_path = pred_rttms_dir / rttm_filename

            logger.info(f"Looking for RTTM file at: {rttm_path}")
            logger.info(f"pred_rttms directory exists: {pred_rttms_dir.exists()}")
            if pred_rttms_dir.exists():
                logger.info(f"Contents of pred_rttms: {list(pred_rttms_dir.iterdir())}")

            segments = []
            unique_speakers = set()
            if rttm_path.exists():
                logger.info(f"Reading RTTM file: {rttm_path}")
                with open(rttm_path, 'r') as f:
                    rttm_content = f.read()
                    logger.info(f"RTTM content:\n{rttm_content}")

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

                logger.info(f"Detected {len(unique_speakers)} unique speakers: {unique_speakers}")
            else:
                logger.error(f"RTTM file not found at: {rttm_path}")
                logger.error(f"Output directory contents: {list(output_dir.iterdir()) if output_dir.exists() else 'directory does not exist'}")

            # Clean up entire output directory with all temp files
            try:
                shutil.rmtree(output_dir)
                logger.debug(f"Cleaned up temporary directory: {output_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {output_dir}: {e}")

            if not segments:
                logger.warning("No speakers detected, creating single speaker segment")
                # Fallback: create a single speaker for entire audio
                segments = [{"start": 0.0, "end": duration, "speaker": "SPEAKER_00"}]

            # Extract VAD segments from diarization results
            logger.info("Extracting VAD segments from diarization results...")
            vad_segments = self._extract_vad_segments_from_diarization(segments)

            # Detect overlapping speech
            logger.info("Detecting overlapping speech regions...")
            overlaps = self._detect_overlapping_speech(segments)

            num_speakers = len(set(s['speaker'] for s in segments))
            logger.info(f"NeMo diarization completed: {len(segments)} segments, {num_speakers} speakers")
            logger.info(f"Speakers found: {set(s['speaker'] for s in segments)}")
            logger.info(f"VAD extracted: {len(vad_segments)} speech segments")
            logger.info(f"Overlaps detected: {len(overlaps)} regions")

            result = {
                "segments": segments,
                "vad_segments": vad_segments,  # Separate VAD for chunking
                "overlaps": overlaps,          # Overlapping speech regions
                "method": "nemo",
                "device": self.device,
                "num_speakers": num_speakers,
                "vad_enabled": True
            }

            logger.info(f"Returning diarization result with {num_speakers} speakers")
            return result

        except Exception as e:
            logger.error(f"NeMo diarization failed: {e}")
            logger.warning("Falling back to simple single-speaker segmentation")

            # Fallback: return single speaker
            import librosa
            try:
                duration = librosa.get_duration(filename=audio_path)
            except:
                duration = 10.0

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
