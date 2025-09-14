"""
FastAPI main application for the Multilingual Topic Clustering System
"""

import logging
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Import services and dependencies
from config import config
from database.connection import db_manager
from services.ner_extractor import MistralNERExtractor
from services.embedding_service import EmbeddingService
from services.vector_service import VectorService
from services.topic_clustering import TopicClusteringService
from api.schemas import *

logger = logging.getLogger(__name__)

# Global services instances
embedding_service = None
ner_extractor = None
vector_service = None
clustering_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    global embedding_service, ner_extractor, vector_service, clustering_service

    logger.info("🚀 Initializing Multilingual Topic Clustering System...")

    try:
        # Initialize database
        logger.info("📊 Setting up database...")
        db_manager.create_tables()

        # Initialize services
        logger.info("🤖 Loading AI models...")

        # Initialize embedding services (BGE-M3)
        embedding_service = EmbeddingService(config.EMBEDDING_MODEL_NAME)
        logger.info("✅ BGE-M3 embedding services loaded")

        # Assuming config.MISTRAL_MODEL_NAME contains the model path
        ner_extractor = MistralNERExtractor(model_id=config.MISTRAL_MODEL_NAME)
        logger.info("✅ Mistral NER extractor loaded from config path")

        # Initialize vector services (ChromaDB)
        vector_service = VectorService(config.CHROMA_PERSIST_DIR, config.CHROMA_COLLECTION_NAME)
        logger.info("✅ ChromaDB vector services initialized")

        # Initialize clustering services
        clustering_service = TopicClusteringService(
            embedding_service, ner_extractor, vector_service, db_manager
        )
        logger.info("✅ Topic clustering services initialized")

        logger.info("🎉 All services initialized successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("🔄 Shutting down services...")
    if db_manager:
        db_manager.close()
    logger.info("👋 Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Multilingual Topic Clustering API",
    description="Production-ready multilingual topic clustering system combining BGE-M3 embeddings with Mistral 24B NER extraction",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# === API ENDPOINTS ===

@app.post("/api/process-text", response_model=TopicClusteringResponse)
async def process_text(text_input: TextInput, background_tasks: BackgroundTasks):
    """
    Process text and assign to topic clusters using hybrid BGE-M3 + Mistral approach
    """
    try:
        if not clustering_service:
            raise HTTPException(status_code=503, detail="Clustering services not initialized")

        # Process text using the clustering services
        result = clustering_service.process_text(
            text=text_input.text,
            source_type=text_input.source_type,
            user_id=text_input.user_id
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return TopicClusteringResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/topics/{topic_id}", response_model=TopicInfo)
async def get_topic(topic_id: str):
    """Get detailed information about a specific topic"""
    try:
        with db_manager.get_session() as session:
            from database.models import Topic

            topic = session.query(Topic).filter(Topic.id == topic_id).first()
            if not topic:
                raise HTTPException(status_code=404, detail="Topic not found")

            return TopicInfo(
                id=topic.id,
                title=topic.title,
                description=topic.description or "",
                primary_language=topic.primary_language,
                content_count=topic.content_count,
                confidence_score=topic.confidence_score,
                created_at=topic.created_at,
                updated_at=topic.updated_at
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get topic {topic_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/topics")
async def list_topics(
        limit: int = 50,
        offset: int = 0,
        language: Optional[str] = None,
        source_type: Optional[str] = None
):
    """List topics with pagination and filtering"""
    try:
        with db_manager.get_session() as session:
            from database.models import Topic
            from sqlalchemy import desc

            query = session.query(Topic)

            # Apply filters
            if language:
                query = query.filter(Topic.primary_language == language)

            # Get paginated results
            topics = query.order_by(desc(Topic.updated_at)).offset(offset).limit(limit).all()

            return {
                "topics": [
                    {
                        "id": topic.id,
                        "title": topic.title,
                        "primary_language": topic.primary_language,
                        "content_count": topic.content_count,
                        "confidence_score": topic.confidence_score,
                        "updated_at": topic.updated_at.isoformat()
                    }
                    for topic in topics
                ],
                "total": query.count(),
                "limit": limit,
                "offset": offset
            }

    except Exception as e:
        logger.error(f"Failed to list topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        with db_manager.get_session() as session:
            from database.models import Topic, TextEntry, ProcessingLog
            from sqlalchemy import func, distinct

            # Basic counts
            total_topics = session.query(Topic).count()
            total_texts = session.query(TextEntry).count()

            # Language distribution
            lang_dist = session.query(
                TextEntry.detected_language,
                func.count(TextEntry.id)
            ).group_by(TextEntry.detected_language).all()

            language_distribution = {lang or "unknown": count for lang, count in lang_dist}

            # Performance metrics
            avg_processing_time = session.query(
                func.avg(TextEntry.processing_time_ms)
            ).scalar() or 0

            # Recent activity (last 24 hours)
            from datetime import datetime, timedelta
            yesterday = datetime.now() - timedelta(days=1)

            recent_texts = session.query(TextEntry).filter(
                TextEntry.created_at >= yesterday
            ).count()

            # Vector database stats
            vector_stats = vector_service.get_collection_stats() if vector_service else {}

            return SystemStats(
                total_topics=total_topics,
                total_texts=total_texts,
                language_distribution=language_distribution,
                processing_performance={
                    "avg_processing_time_ms": float(avg_processing_time),
                    "recent_texts_24h": recent_texts,
                    "vector_collection_size": vector_stats.get("total_topics", 0)
                },
                uptime_hours=0.0  # Would calculate from startup time in production
            )

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
async def search_topics(query: str, limit: int = 10, threshold: float = 0.6):
    """Search topics using semantic similarity"""
    try:
        if not embedding_service or not vector_service:
            raise HTTPException(status_code=503, detail="Search services not available")

        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Generate embedding for search query
        embeddings = embedding_service.generate_embeddings([query])
        query_embedding = embeddings[0]

        # Search similar topics
        similar_topics = vector_service.search_similar_topics(
            query_embedding=query_embedding,
            n_results=limit,
            threshold=threshold
        )

        return {
            "query": query,
            "threshold": threshold,
            "total_results": len(similar_topics),
            "results": similar_topics
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "services": {}
        }

        # Check database
        try:
            db_healthy = db_manager.test_connection()
            health_status["services"]["database"] = "healthy" if db_healthy else "unhealthy"
        except Exception as e:
            health_status["services"]["database"] = f"error: {str(e)}"

        # Check services
        health_status["services"]["embedding_service"] = "loaded" if embedding_service else "not_loaded"
        health_status["services"]["ner_extractor"] = "loaded" if ner_extractor else "not_loaded"
        health_status["services"]["vector_service"] = "loaded" if vector_service else "not_loaded"
        health_status["services"]["clustering_service"] = "loaded" if clustering_service else "not_loaded"

        # Check vector database
        if vector_service:
            try:
                vector_health = vector_service.health_check()
                health_status["services"]["vector_database"] = vector_health["status"]
            except Exception as e:
                health_status["services"]["vector_database"] = f"error: {str(e)}"

        # Determine overall status
        if any("error" in str(status) or status == "unhealthy" for status in health_status["services"].values()):
            health_status["status"] = "degraded"

        if any(status == "not_loaded" for status in health_status["services"].values()):
            health_status["status"] = "starting"

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


# === WEB INTERFACE ROUTES ===

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with system overview"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/demo", response_class=HTMLResponse)
async def demo_page(request: Request):
    """Interactive demo page for testing the system"""
    return templates.TemplateResponse("demo.html", {"request": request})


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Analytics dashboard with system metrics"""
    return templates.TemplateResponse("analytics.html", {"request": request})


# === ERROR HANDLERS ===

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)


# === MIDDLEWARE ===

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )