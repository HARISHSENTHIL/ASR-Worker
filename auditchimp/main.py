"""Main entry point for the AudioChimp API."""

import sys
import logging
from .config import config
from .logger import setup_logging, log_event
from .api import app

# Configure logging
setup_logging(config.LOG_FILE, config.LOG_LEVEL)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    try:
        log_event(
            logger, "info", "api_startup",
            "AudioChimp API server starting",
            model=config.MODEL_NAME,
            compute_type=config.MODEL_COMPUTE_TYPE,
            host=config.API_HOST,
            port=config.API_PORT
        )

        import uvicorn

        uvicorn.run(
            "auditchimp.api:app",
            host=config.API_HOST,
            port=config.API_PORT,
            workers=config.API_WORKERS,
            reload=False,
            access_log=True,
            timeout_keep_alive=120,
            timeout_graceful_shutdown=30
        )

    except KeyboardInterrupt:
        log_event(logger, "info", "api_shutdown", "AudioChimp API shutting down")
    except Exception as e:
        log_event(logger, "error", "api_startup_failed", "Failed to start AudioChimp API", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
