"""
Configuration management for the Multilingual Topic Clustering System
"""

import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class containing all application settings"""

    # Database Configuration
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "topic_clustering")

    # Vector Database Configuration
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "multilingual_topics_bge_m3")

    # Model Configuration
    BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
    MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mlx-community/Dolphin-Mistral-24B-Venice-Edition-4bit")
    # MISTRAL_MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH", "/Users/pankajkumar/.cache/huggingface/hub/models--mlx-community--Dolphin-Mistral-24B-Venice-Edition-4bit")

    # Processing Configuration
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    MAX_TOPIC_TITLE_WORDS = int(os.getenv("MAX_TOPIC_TITLE_WORDS", "10"))

    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # Environment Configuration
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "./logs/app.log")

    @property
    def DATABASE_URL(self):
        """Construct database URL for SQLAlchemy with proper URL encoding"""
        encoded_password = quote_plus(self.MYSQL_PASSWORD)
        return (f"mysql+mysqlconnector://{self.MYSQL_USER}:{encoded_password}"
                f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}")

    @property
    def IS_PRODUCTION(self):
        """Check if running in production mode"""
        return self.ENVIRONMENT.lower() == "production"

    def validate(self):
        """Validate configuration settings"""
        required_vars = [
            "MYSQL_PASSWORD", "MYSQL_DATABASE", "BGE_MODEL_NAME", "MISTRAL_MODEL_NAME"
        ]

        missing_vars = []
        for var in required_vars:
            if not getattr(self, var, None):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required configuration variables: {', '.join(missing_vars)}")

        # Validate numeric ranges
        if not 0.5 <= self.SIMILARITY_THRESHOLD <= 1.0:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0.5 and 1.0")

        if not 1 <= self.BATCH_SIZE <= 128:
            raise ValueError("BATCH_SIZE must be between 1 and 128")

        if not 1 <= self.MAX_TOPIC_TITLE_WORDS <= 20:
            raise ValueError("MAX_TOPIC_TITLE_WORDS must be between 1 and 20")


# Create global config instance
config = Config()

# Validate configuration on import
if __name__ != "__main__":
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please check your .env file and ensure all required variables are set.")