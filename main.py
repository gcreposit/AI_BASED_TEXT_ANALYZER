"""
Main entry point for the Multilingual Topic Clustering System
"""

import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Configure logging
def setup_logging():
    """Setup logging configuration"""
    from config import config

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main function to start the application"""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)

        logger.info("Starting Multilingual Topic Clustering System")

        # Import and validate configuration
        from config import config
        config.validate()
        logger.info(f"Configuration validated successfully")
        logger.info(f"Environment: {config.ENVIRONMENT}")
        logger.info(f"Debug mode: {config.DEBUG}")

        # Initialize database
        from database.connection import db_manager

        logger.info("Testing database connection...")
        if not db_manager.test_connection():
            raise Exception("Database connection failed")

        logger.info("Creating database tables...")
        db_manager.create_tables()

        # Import FastAPI app
        from api.main import app

        # Start the server
        import uvicorn

        logger.info(f"Starting server on {config.API_HOST}:{config.API_PORT}")

        uvicorn.run(
            "api.main:app",
            host=config.API_HOST,
            port=config.API_PORT,
            reload=config.DEBUG,
            log_level=config.LOG_LEVEL.lower(),
            access_log=True
        )

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()