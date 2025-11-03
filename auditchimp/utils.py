"""Utility functions for AudioChimp API."""

import logging
from pathlib import Path
from .config import config

logger = logging.getLogger(__name__)


def safe_delete_audio_file(file_path: str, request_id: str, context: str = "unknown") -> bool:
    """
    Safely delete audio file with comprehensive verification.

    This function implements a multi-stage verification process:
    1. Check if file exists
    2. Verify file is within upload directory (security)
    3. Get file metadata for logging
    4. Delete the file
    5. Verify deletion was successful

    Args:
        file_path: Path to the file to delete
        request_id: Request ID for logging and tracking
        context: Context of deletion for logging (e.g., "completion", "cancellation", "manual_delete")

    Returns:
        True if file was deleted successfully or already doesn't exist, False on failure
    """
    try:
        file_path_obj = Path(file_path)

        # Stage 1: Check if file exists
        if not file_path_obj.exists():
            logger.info(
                f"[{context}] File cleanup skipped - already deleted: "
                f"request_id={request_id}, path={file_path}"
            )
            return True  # File doesn't exist, goal achieved

        # Stage 2: Security check - ensure file is in upload directory
        upload_dir = Path(config.UPLOAD_DIR).resolve()
        file_absolute = file_path_obj.resolve()

        if not str(file_absolute).startswith(str(upload_dir)):
            logger.error(
                f"[{context}] SECURITY VIOLATION - Attempted to delete file outside upload directory: "
                f"request_id={request_id}, path={file_path}, upload_dir={upload_dir}"
            )
            return False

        # Stage 3: Get file metadata before deletion (for logging)
        try:
            file_size_bytes = file_path_obj.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
        except Exception as e:
            logger.warning(f"[{context}] Could not get file size: {e}")
            file_size_mb = 0.0

        # Stage 4: Delete the file
        file_path_obj.unlink()

        # Stage 5: Verify deletion was successful
        if file_path_obj.exists():
            logger.error(
                f"[{context}] File still exists after deletion attempt: "
                f"request_id={request_id}, path={file_path}"
            )
            return False

        # Success!
        logger.info(
            f"[{context}] Successfully deleted audio file: "
            f"request_id={request_id}, path={file_path}, size={file_size_mb:.2f}MB"
        )
        return True

    except PermissionError as e:
        logger.error(
            f"[{context}] Permission denied when deleting file: "
            f"request_id={request_id}, path={file_path}, error={e}"
        )
        return False
    except Exception as e:
        logger.error(
            f"[{context}] Unexpected error during file deletion: "
            f"request_id={request_id}, path={file_path}, error={e}"
        )
        return False
