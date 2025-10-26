"""Main entry point for the AudioChimp API."""

import asyncio
import logging
import sys
from pathlib import Path

from .config import config
from .api import app

# Ensure log directory exists
log_dir = Path(config.LOG_FILE).parent
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    try:
        logger.info("Starting AudioChimp API server...")
        logger.info(f"Configuration: Model={config.WHISPER_MODEL}, Quantization={config.QUANTIZATION}")
        logger.info(f"API will be available at http://{config.API_HOST}:{config.API_PORT}")

        # Import and run the FastAPI app
        import uvicorn

        uvicorn.run(
            "auditchimp.api:app",
            host=config.API_HOST,
            port=config.API_PORT,
            workers=config.API_WORKERS,
            reload=False,
            access_log=True
        )

    except KeyboardInterrupt:
        logger.info("Shutting down AudioChimp API...")
    except Exception as e:
        logger.error(f"Error starting AudioChimp API: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
