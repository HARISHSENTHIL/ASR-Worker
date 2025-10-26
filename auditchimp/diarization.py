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
        self.vad_model = None
        self.speaker_model = None
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
        """Get NeMo diarizer configuration."""
        from omegaconf import OmegaConf

        # NeMo diarizer configuration
        config = {
            'device': self.device,  # Add device configuration
            'diarizer': {
                'manifest_filepath': None,  # Will be set per audio file
                'out_dir': str(self.config.TEMP_DIR),
                'oracle_vad': False,
                'clustering': {
                    'parameters': {
                        'oracle_num_speakers': False,
                        'max_num_speakers': 8,
                        'enhanced_count_thres': 80,
                        'max_rp_threshold': 0.25,
                        'sparse_search_volume': 30,
                    }
                },
                'vad': {
                    'model_path': 'vad_multilingual_marblenet',
                    'parameters': {
                        'window_length_in_sec': 0.15,
                        'shift_length_in_sec': 0.01,
                        'onset': 0.8,
                        'offset': 0.6,
                        'pad_onset': 0.05,
                        'pad_offset': -0.1,
                        'min_duration_on': 0.2,
                        'min_duration_off': 0.2,
                    }
                },
                'speaker_embeddings': {
                    'model_path': 'titanet_large',
                    'parameters': {
                        'window_length_in_sec': 1.5,
                        'shift_length_in_sec': 0.75,
                        'multiscale_weights': [1, 2, 3, 4, 5, 6],
                        'save_embeddings': False,
                    }
                },
            }
        }

        return OmegaConf.create(config)

    def _diarize_nemo_sync(self, audio_path: str) -> Dict:
        """
        Diarize using NeMo ClusteringDiarizer (synchronous version for thread pool).

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with segments and metadata
        """
        try:
            logger.info("Starting NeMo diarization...")

            # Create temporary manifest file
            manifest_path = Path(self.config.TEMP_DIR) / f"{Path(audio_path).stem}_manifest.json"

            # Create manifest entry
            manifest_data = {
                "audio_filepath": str(audio_path),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": None,
                "uem_filepath": None
            }

            # Write manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)
                f.write('\n')

            # Update config with manifest path
            self.msdd_model._cfg.diarizer.manifest_filepath = str(manifest_path)
            self.msdd_model._cfg.diarizer.out_dir = str(self.config.TEMP_DIR)

            # Run diarization
            self.msdd_model.diarize()

            # Read RTTM output
            rttm_path = Path(self.config.TEMP_DIR) / f"{Path(audio_path).stem}.rttm"

            segments = []
            if rttm_path.exists():
                with open(rttm_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            start = float(parts[3])
                            duration = float(parts[4])
                            speaker = parts[7]

                            segments.append({
                                "start": start,
                                "end": start + duration,
                                "speaker": speaker
                            })

                # Clean up temp files
                rttm_path.unlink(missing_ok=True)

            manifest_path.unlink(missing_ok=True)

            if not segments:
                logger.warning("No speakers detected, creating single speaker segment")
                # Fallback: create a single speaker for entire audio
                segments = [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}]

            logger.info(f"NeMo diarization completed: {len(segments)} segments, {len(set(s['speaker'] for s in segments))} speakers")

            return {
                "segments": segments,
                "method": "nemo",
                "device": self.device,
                "num_speakers": len(set(s['speaker'] for s in segments))
            }

        except Exception as e:
            logger.error(f"NeMo diarization failed: {e}")
            logger.warning("Falling back to simple single-speaker segmentation")

            # Fallback: return single speaker
            return {
                "segments": [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}],
                "method": "fallback",
                "device": "none",
                "num_speakers": 1
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
