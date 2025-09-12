"""
Database connection and session management
"""

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging
import time
from config import config
from database.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database connection manager with connection pooling and session management
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with optimized settings"""
        try:
            self.engine = create_engine(
                self.database_url,
                # Connection pool settings
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,  # Recycle connections every hour
                pool_size=10,  # Base number of connections
                max_overflow=20,  # Additional connections if needed
                poolclass=QueuePool,

                # Connection settings
                connect_args={
                    "charset": "utf8mb4",
                    "use_unicode": True,
                    "autocommit": False
                },

                # Echo SQL queries in debug mode
                echo=config.DEBUG
            )

            # Add connection event listeners
            self._setup_event_listeners()

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            logger.info("Database engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for monitoring"""

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set connection parameters on new connections"""
            if hasattr(dbapi_connection, 'set_charset'):
                dbapi_connection.set_charset('utf8mb4')

        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log slow queries"""
            context._query_start_time = time.time()

        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log query execution time"""
            total = time.time() - context._query_start_time
            if total > 1.0:  # Log queries taking more than 1 second
                logger.warning(f"Slow query detected: {total:.2f}s - {statement[:100]}...")

    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise

    @contextmanager
    def get_session(self):
        """
        Get database session with automatic transaction management

        Usage:
            with db_manager.get_session() as session:
                # Use session here
                pass
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def get_session_raw(self):
        """Get raw session without context management"""
        return self.SessionLocal()

    def create_database_if_not_exists(self):
        """Ensure that the database exists before starting the application"""
        # Extract the database URL components
        engine = create_engine(config.DATABASE_URL)
        db_url = config.DATABASE_URL.split("/")

        # The database name should be the last component in the URL
        db_name = db_url[-1] if len(db_url) > 1 else 'topic_clustering'

        try:
            # Create a connection to MySQL without specifying the database
            base_url = "/".join(db_url[:-1])  # URL without the database name
            temp_engine = create_engine(f"{base_url}")

            with temp_engine.connect() as conn:
                # Wrap SQL query with text() to ensure it's executable
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
            logger.info(f"Database '{db_name}' created or already exists.")
            return True
        except OperationalError as e:
            logger.error(f"Error while creating database: {e}")
            return False

    def test_connection(self):
        """Test database connection"""
        try:
            # Ensure the database exists before testing the connection
            if not self.create_database_if_not_exists():
                raise Exception("Database creation failed")

            # Now test the connection by querying
            with self.get_session() as session:
                session.execute(text("SELECT 1"))

            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_connection_info(self):
        """Get database connection information"""
        return {
            "url": self.database_url.replace(config.MYSQL_PASSWORD, "***"),
            "pool_size": self.engine.pool.size(),
            "checked_in": self.engine.pool.checkedin(),
            "checked_out": self.engine.pool.checkedout(),
            "overflow": self.engine.pool.overflow(),
            "invalid": self.engine.pool.invalid()
        }

    def close(self):
        """Close database engine and all connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Initialize global database manager
try:
    db_manager = DatabaseManager(config.DATABASE_URL)
except Exception as e:
    logger.error(f"Failed to initialize database manager: {e}")
    raise

# Test connection on import (except during testing)
if __name__ != "__main__" and not config.DEBUG:
    if not db_manager.test_connection():
        logger.warning("Database connection test failed during initialization")