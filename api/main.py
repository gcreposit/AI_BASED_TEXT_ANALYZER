"""
FastAPI main application for the Multilingual Topic Clustering System
"""
import json
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
# from services import news_extract
from services.news_extract import NewsExtractor
from services.ner_extractor import MistralNERExtractor
from services.embedding_service import EmbeddingService
from services.vector_service import VectorService
from services.topic_clustering import TopicClusteringService

# ✅ NEW: Import new services
from district_normalizer import DistrictNormalizer  # New file
from sentiment_analyzer import AdvancedSentimentAnalyzer  # New file

from api.schemas import *

logger = logging.getLogger(__name__)

# Global services instances
embedding_service = None
ner_extractor = None
vector_service = None
clustering_service = None
news_extractor = None


# In the lifespan function, add this after clustering_service initialization:


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    global embedding_service, ner_extractor, vector_service, clustering_service, news_extractor

    logger.info("🚀 Initializing Multilingual Topic Clustering System...")

    try:
        # Initialize database
        logger.info("📊 Setting up database...")
        db_manager.create_tables()

        # Initialize services
        logger.info("🤖 Loading AI models...")

        # Initialize embedding services (BGE-M3)
        embedding_service = EmbeddingService(config.EMBEDDING_MODEL_NAME)
        logger.info("✅ Embedding services loaded")

        # Assuming config.MISTRAL_MODEL_NAME contains the model path
        ner_extractor = MistralNERExtractor(model_id=config.MISTRAL_MODEL_NAME)
        logger.info("✅ Mistral NER extractor loaded from config path")

        # Initialize vector services (ChromaDB)
        vector_service = VectorService(config.CHROMA_PERSIST_DIR, config.CHROMA_COLLECTION_NAME)
        logger.info("✅ ChromaDB vector services initialized")

        # Initialize clustering services
        clustering_service = TopicClusteringService(embedding_service, ner_extractor, vector_service, db_manager)
        logger.info("✅ Topic clustering services initialized")

        news_extractor = NewsExtractor(ner_extractor)
        logger.info("✅ News extraction service initialized")

        # ✅ NEW: Create Unassigned Posts topic
        try:
            clustering_service.get_or_create_unassigned_topic()
            logger.info("✅ Unassigned Posts topic ready")
        except Exception as e:
            logger.warning(f"⚠️  Could not create Unassigned Posts topic: {e}")

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
    description="Production-ready multilingual topic clustering system combining BGE-M3 embeddings with Mistral 24B "
                "NER extraction",
    version="2.0.0",
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

# REPLACE the /api/process-text endpoint

@app.post("/api/process-text", response_model=TopicClusteringResponse)
async def process_text(text_input: TextInput, background_tasks: BackgroundTasks):
    """
    ✅ UPGRADED: Process text with intelligent cleaning and simplified output
    - Auto-cleans scraped news and social media clutter
    - Single normalized text output (no processed/enhanced variants)
    - All new features: temporal, sentiment, three-phase matching
    """
    try:
        if not clustering_service:
            raise HTTPException(status_code=503, detail="Clustering services not initialized")

        # Call upgraded process_text_complete
        result = clustering_service.process_text_complete(
            text=text_input.text,
            source_type=text_input.source_type,
            user_id=text_input.user_id
        )

        # Log result
        logger.info(f"Processed result:\n{json.dumps(result, ensure_ascii=False, indent=2)}")

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return TopicClusteringResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# --------------------------------------------------- START NEW MODIDFIED WAALI H ------------------------------
# 🔄 MODIFIED: Update batch processing
@app.post("/api/process-batch", response_model=BatchProcessResponse)
async def process_batch(batch_request: BatchProcessRequest, background_tasks: BackgroundTasks):
    """
    Process multiple texts in batch for improved performance
    NOW WITH: Enhanced matching and validation
    """
    try:
        if not clustering_service:
            raise HTTPException(status_code=503, detail="Clustering services not initialized")

        start_time = time.time()
        results = []
        errors = []
        successful = 0
        failed = 0

        # 🔄 MODIFIED: Use complete batch processing if available
        if hasattr(clustering_service, 'process_text_batch'):
            batch_results = await clustering_service.process_text_batch(
                texts=batch_request.texts,
                source_type=batch_request.source_type,
                user_id=batch_request.user_id
            )
        else:
            # Fallback: process one by one
            batch_results = []
            for text in batch_request.texts:
                result = clustering_service.process_text(
                    text=text,
                    source_type=batch_request.source_type,
                    user_id=batch_request.user_id
                )
                batch_results.append(result)

        # Process results (existing code - no changes)
        for i, result in enumerate(batch_results):
            try:
                if "error" in result:
                    errors.append(f"Text {i + 1}: {result['error']}")
                    failed += 1
                else:
                    results.append(TopicClusteringResponse(**result))
                    successful += 1
            except Exception as e:
                errors.append(f"Text {i + 1}: {str(e)}")
                failed += 1

        processing_time = int((time.time() - start_time) * 1000)

        return BatchProcessResponse(
            total_processed=len(batch_request.texts),
            successful=successful,
            failed=failed,
            processing_time_ms=processing_time,
            results=results,
            errors=errors
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# --------------------------------------------------- END NEW MODIDFIED WAALI H ------------------------------


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
        source_type: Optional[str] = None):
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


# 🔄 MODIFIED: Enhanced health check with new features
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint - NOW WITH NEW FEATURES"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",  # 🔄 Updated version
            "services": {},
            "features": {}  # ✅ NEW: Feature availability
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

        # ✅ NEW: Check feature availability
        if clustering_service:
            health_status["features"] = {
                "district_normalization": hasattr(clustering_service, '_find_similar_topics_three_phase'),
                "three_phase_matching": hasattr(clustering_service, '_find_similar_topics_three_phase'),
                "temporal_extraction": True,  # Part of NER
                "advanced_sentiment": True,  # Part of NER
                "topic_status_management": hasattr(clustering_service, 'mark_topic_inactive'),
                "text_reassignment": hasattr(clustering_service, 'reassign_text_to_topic'),
                "unassigned_topic": hasattr(clustering_service, 'get_or_create_unassigned_topic'),
                "category_validation": hasattr(clustering_service, 'validate_and_correct_categories'),
                "llm_title_generation": hasattr(clustering_service, '_generate_topic_title_with_llm_hindi')
            }

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


@app.get("/topics", response_class=HTMLResponse)
async def topics_page(request: Request):
    """Topics dashboard page"""
    return templates.TemplateResponse("topics.html", {"request": request})


@app.get("/api/topics-dashboard")
async def get_topics_dashboard():
    """Get topics data for dashboard with post counts and contextual understanding"""
    try:
        # Import here to avoid circular imports
        import pymysql
        from sqlalchemy import create_engine, text
        from urllib.parse import quote_plus
        import os

        # Database configuration for DUMP DATABASE
        DUMP_DB_CONFIG = {
            'host': os.getenv('DUMP_DB_HOST', '94.136.189.147'),
            'database': os.getenv('DUMP_DB_NAME', 'twitter_scrapper'),
            'user': os.getenv('DUMP_DB_USER', 'gccloud'),
            'password': os.getenv('DUMP_DB_PASSWORD', 'Gccloud@1489$'),
            'port': int(os.getenv('DUMP_DB_PORT', '3306'))
        }

        # Create database engine for dump database
        encoded_password = quote_plus(DUMP_DB_CONFIG['password'])
        DUMP_DATABASE_URL = f"mysql+pymysql://{DUMP_DB_CONFIG['user']}:{encoded_password}@{DUMP_DB_CONFIG['host']}:{DUMP_DB_CONFIG['port']}/{DUMP_DB_CONFIG['database']}"

        dump_engine = create_engine(DUMP_DATABASE_URL, echo=False, pool_pre_ping=True)

        with dump_engine.connect() as conn:
            # Get topics with post counts
            query = text("""
                SELECT 
                    ad.topic_id,
                    ad.topic_title,
                    ad.detected_language,
                    COUNT(*) as post_count
                FROM analyzed_data ad
                WHERE ad.topic_id IS NOT NULL 
                AND ad.topic_title IS NOT NULL
                GROUP BY ad.topic_id, ad.topic_title, ad.detected_language
                ORDER BY post_count DESC
            """)

            result = conn.execute(query)
            topics_data = []

            for row in result:
                topics_data.append({
                    'topic_id': row.topic_id,
                    'title': row.topic_title,
                    'primary_language': row.detected_language or 'Unknown',
                    'post_count': row.post_count
                })

            # Calculate stats
            total_topics = len(topics_data)
            total_posts = sum(topic['post_count'] for topic in topics_data)
            avg_posts_per_topic = round(total_posts / total_topics, 1) if total_topics > 0 else 0
            languages = set(topic['primary_language'] for topic in topics_data)
            language_count = len(languages)

            stats = {
                'total_topics': total_topics,
                'total_posts': total_posts,
                'avg_posts_per_topic': avg_posts_per_topic,
                'language_count': language_count
            }

            return {
                'success': True,
                'topics': topics_data,
                'stats': stats
            }

    except Exception as e:
        logger.error(f"Failed to get topics dashboard data: {e}")
        return {
            'success': False,
            'error': str(e),
            'topics': [],
            'stats': {}
        }


@app.get("/api/topics/{topic_id}/posts")
async def get_topic_posts(topic_id: str):
    """Get posts for a specific topic with contextual understanding"""
    try:
        # Import here to avoid circular imports
        import pymysql
        from sqlalchemy import create_engine, text
        from urllib.parse import quote_plus
        import os

        # Database configuration for DUMP DATABASE
        DUMP_DB_CONFIG = {
            'host': os.getenv('DUMP_DB_HOST', '94.136.189.147'),
            'database': os.getenv('DUMP_DB_NAME', 'twitter_scrapper'),
            'user': os.getenv('DUMP_DB_USER', 'gccloud'),
            'password': os.getenv('DUMP_DB_PASSWORD', 'Gccloud@1489$'),
            'port': int(os.getenv('DUMP_DB_PORT', '3306'))
        }

        # Create database engine for dump database
        encoded_password = quote_plus(DUMP_DB_CONFIG['password'])
        DUMP_DATABASE_URL = f"mysql+pymysql://{DUMP_DB_CONFIG['user']}:{encoded_password}@{DUMP_DB_CONFIG['host']}:{DUMP_DB_CONFIG['port']}/{DUMP_DB_CONFIG['database']}"

        dump_engine = create_engine(DUMP_DATABASE_URL, echo=False, pool_pre_ping=True)

        with dump_engine.connect() as conn:
            # Get posts for the specific topic
            query = text("""
                SELECT 
                    pb.post_title,
                    pb.post_snippet,
                    pb.source,
                    pb.post_timestamp,
                    ad.detected_language,
                    ad.contextual_understanding
                FROM analyzed_data ad
                JOIN post_bank pb ON ad.dump_table_id = pb.id
                WHERE ad.topic_id = :topic_id
                ORDER BY pb.post_timestamp DESC
                LIMIT 100
            """)

            result = conn.execute(query, {'topic_id': topic_id})
            posts_data = []

            for row in result:
                posts_data.append({
                    'post_title': row.post_title,
                    'post_snippet': row.post_snippet,
                    'source': row.source,
                    'post_timestamp': row.post_timestamp.isoformat() if row.post_timestamp else None,
                    'detected_language': row.detected_language,
                    'contextual_understanding': row.contextual_understanding
                })

            return {
                'success': True,
                'posts': posts_data
            }

    except Exception as e:
        logger.error(f"Failed to get posts for topic {topic_id}: {e}")
        return {
            'success': False,
            'error': str(e),
            'posts': []
        }


@app.post("/api/extract-news")
async def extract_news(request: Request):
    try:
        if not news_extractor:
            raise HTTPException(status_code=503, detail="News extraction service not initialized")

        data = await request.json()
        raw_text = data.get("raw_text", "")
        max_tokens = data.get("max_tokens", 4000)
        temperature = data.get("temperature", 0.1)

        result = news_extractor.extract_news(raw_text, max_tokens, temperature)  # Use instance, not class
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-news-batch")
async def extract_news_batch(request: Request):
    try:
        if not news_extractor:
            raise HTTPException(status_code=503, detail="News extraction service not initialized")

        data = await request.json()
        raw_texts = data.get("raw_texts", [])
        max_tokens = data.get("max_tokens", 4000)
        temperature = data.get("temperature", 0.1)

        results = news_extractor.extract_news_batch(raw_texts, max_tokens, temperature)  # Use instance
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test-enhanced-extraction")
async def test_enhanced_extraction(request: Request):
    """Test endpoint to verify enhanced NER extraction is working"""
    try:
        data = await request.json()
        text = data.get("text", "अनुसूचित जाति की नाबालिग लड़की से दुष्कर्म का मामला")

        if not ner_extractor:
            raise HTTPException(status_code=503, detail="NER service not available")

        # Test direct NER extraction
        ner_result = ner_extractor.extract(text)

        # Test through clustering service
        clustering_result = clustering_service.process_text(text)

        return {
            "success": True,
            "test_text": text,
            "direct_ner_result": {
                "has_category_classifications": "category_classifications" in ner_result,
                "has_primary_classification": "primary_classification" in ner_result,
                "has_incident_location_analysis": "incident_location_analysis" in ner_result,
                "category_count": len(ner_result.get("category_classifications", [])),
                "categories": ner_result.get("category_classifications", []),
                "primary": ner_result.get("primary_classification", {}),
                "location_analysis": ner_result.get("incident_location_analysis", {})
            },
            "clustering_service_result": {
                "has_enhanced_entities": "extracted_entities" in clustering_result,
                "entities_have_classifications": "category_classifications" in clustering_result.get(
                    "extracted_entities", {}),
                "extracted_entities": clustering_result.get("extracted_entities", {})
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ===== ✅ NEW ENDPOINTS - ADD AT THE END =====

@app.post("/api/topics/{topic_id}/status")
async def update_topic_status(
        topic_id: str,
        status: str,  # 'active' or 'inactive'
        reason: str = "",
        user_id: str = "api_user"
):
    """
    ✅ NEW: Mark topic as active or inactive
    Used to prevent old/resolved topics from being reused
    """
    try:
        if not clustering_service:
            raise HTTPException(status_code=503, detail="Clustering service not available")

        if not hasattr(clustering_service, 'mark_topic_inactive'):
            raise HTTPException(status_code=501, detail="Topic status management not available in current version")

        if status == 'inactive':
            success = clustering_service.mark_topic_inactive(
                topic_id=topic_id,
                reason=reason,
                user_id=user_id
            )
        elif status == 'active':
            success = clustering_service.reactivate_topic(
                topic_id=topic_id,
                user_id=user_id
            )
        else:
            raise HTTPException(status_code=400, detail="Status must be 'active' or 'inactive'")

        if not success:
            raise HTTPException(status_code=404, detail="Topic not found")

        return {
            "success": True,
            "topic_id": topic_id,
            "status": status,
            "reason": reason
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update topic status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/texts/{text_entry_id}/reassign")
async def reassign_text(
        text_entry_id: int,
        new_topic_id: str,
        reason: str = "Manual reassignment",
        user_id: str = "api_user"
):
    """
    ✅ NEW: Reassign a text entry to a different topic
    Useful for moving posts from "Unassigned Posts" to proper topics
    """
    try:
        if not clustering_service:
            raise HTTPException(status_code=503, detail="Clustering service not available")

        if not hasattr(clustering_service, 'reassign_text_to_topic'):
            raise HTTPException(status_code=501, detail="Text reassignment not available in current version")

        result = clustering_service.reassign_text_to_topic(
            text_entry_id=text_entry_id,
            new_topic_id=new_topic_id,
            user_id=user_id,
            reason=reason
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Reassignment failed"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reassign text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/texts/{text_entry_id}/suggestions")
async def get_reassignment_suggestions(text_entry_id: int, top_k: int = 5):
    """
    ✅ NEW: Get suggested topics for manual reassignment
    Shows user which topics might be a better fit
    """
    try:
        if not clustering_service:
            raise HTTPException(status_code=503, detail="Clustering service not available")

        if not hasattr(clustering_service, 'get_suggested_topics_for_reassignment'):
            raise HTTPException(status_code=501, detail="Suggestion feature not available in current version")

        suggestions = clustering_service.get_suggested_topics_for_reassignment(
            text_entry_id=text_entry_id,
            top_k=top_k
        )

        return {
            "text_entry_id": text_entry_id,
            "suggestions": suggestions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/districts/validate")
async def validate_district(district_name: str):
    """
    ✅ NEW: Validate and normalize a district name
    Returns canonical English form
    """
    try:
        normalized = DistrictNormalizer.normalize(district_name)

        if normalized:
            return {
                "valid": True,
                "input": district_name,
                "normalized": normalized,
                "hindi_name": DistrictNormalizer.get_hindi_name(normalized)
            }
        else:
            return {
                "valid": False,
                "input": district_name,
                "message": "District not recognized"
            }

    except Exception as e:
        logger.error(f"District validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/districts")
async def list_districts():
    """
    ✅ NEW: Get list of all recognized districts
    Useful for autocomplete and validation
    """
    try:
        districts = DistrictNormalizer.get_all_canonical_districts()

        district_list = [
            {
                "english": district,
                "hindi": DistrictNormalizer.get_hindi_name(district)
            }
            for district in districts
        ]

        return {
            "total": len(district_list),
            "districts": district_list
        }

    except Exception as e:
        logger.error(f"Failed to list districts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories")
async def list_categories():
    """
    ✅ NEW: Get list of all allowed categories
    Shows the complete category structure
    """
    try:
        from services.topic_clustering import ALLOWED_CATEGORIES

        categories = {}
        for broad_cat, sub_cats in ALLOWED_CATEGORIES.items():
            categories[broad_cat] = list(sub_cats)

        return {
            "total_broad_categories": len(categories),
            "total_subcategories": sum(len(subs) for subs in categories.values()),
            "categories": categories
        }

    except Exception as e:
        logger.error(f"Failed to list categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/topics/unassigned")
async def get_unassigned_posts(limit: int = 50, offset: int = 0):
    """
    ✅ NEW: Get posts in the "Unassigned Posts" topic
    These are posts that need manual review/reassignment
    """
    try:
        from services.topic_clustering import UNASSIGNED_TOPIC_ID

        with db_manager.get_session() as session:
            from database.models import TextEntry
            from sqlalchemy import desc

            query = session.query(TextEntry).filter(
                TextEntry.topic_id == UNASSIGNED_TOPIC_ID
            )

            total = query.count()

            entries = query.order_by(desc(TextEntry.created_at)).offset(offset).limit(limit).all()

            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "entries": [
                    {
                        "id": entry.id,
                        "text": entry.original_text[:200] + "..." if len(
                            entry.original_text) > 200 else entry.original_text,
                        "full_text": entry.original_text,
                        "detected_language": entry.detected_language,
                        "created_at": entry.created_at.isoformat() if entry.created_at else None,
                        "extracted_entities": entry.extracted_entities,
                        "notes": entry.notes
                    }
                    for entry in entries
                ]
            }

    except Exception as e:
        logger.error(f"Failed to get unassigned posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== ✅ NEW DEBUG/TESTING ENDPOINTS =====

@app.post("/api/debug/test-district-normalization")
async def test_district_normalization(districts: list[str]):
    """
    ✅ NEW: Test endpoint to verify district normalization
    """
    try:
        results = []
        for district in districts:
            normalized = DistrictNormalizer.normalize(district)
            results.append({
                "input": district,
                "normalized": normalized,
                "hindi": DistrictNormalizer.get_hindi_name(normalized) if normalized else None,
                "valid": normalized is not None
            })

        return {
            "tested": len(districts),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/debug/test-category-validation")
async def test_category_validation(request: Request):
    """
    ✅ NEW: Test category validation with sample classifications
    """
    try:
        data = await request.json()
        classifications = data.get("classifications", [])

        if not clustering_service or not hasattr(clustering_service, 'validate_and_correct_categories'):
            raise HTTPException(status_code=501, detail="Category validation not available")

        validated = clustering_service.validate_and_correct_categories(classifications)

        return {
            "input_count": len(classifications),
            "output_count": len(validated),
            "input": classifications,
            "output": validated
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#=========================FETCH CATEGORIES FROM DB===================================#
@app.post("/api/admin/reload-categories")
async def reload_category_keywords():
    """
    ✅ NEW: Reload category keywords from external source
    Call this endpoint when your DB is updated
    """
    try:
        if not clustering_service:
            raise HTTPException(status_code=503, detail="Service not available")

        # Reload keywords
        clustering_service.keyword_classifier.reload_keywords()

        # Get updated info
        info = clustering_service.keyword_classifier.get_category_info()

        return {
            "success": True,
            "message": "Category keywords reloaded successfully",
            "info": info
        }

    except Exception as e:
        logger.error(f"Failed to reload keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories/info")
async def get_category_info():
    """
    ✅ NEW: Get information about loaded categories and keywords
    """
    try:
        if not clustering_service:
            raise HTTPException(status_code=503, detail="Service not available")

        info = clustering_service.keyword_classifier.get_category_info()
        return info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/sync-categories-from-db")
async def sync_categories_from_external_db(
        db_host: str = None,
        db_name: str = None,
        db_user: str = None,
        db_password: str = None,
        db_port: int = None
):
    """
    ✅ NEW: Fetch categories from external DB and update local JSON
    Admin endpoint - use with caution
    """
    try:
        if not clustering_service:
            raise HTTPException(status_code=503, detail="Service not available")

        # Build config (None values will use env vars)
        db_config = {}
        if db_host:
            db_config['host'] = db_host
        if db_name:
            db_config['database'] = db_name
        if db_user:
            db_config['user'] = db_user
        if db_password:
            db_config['password'] = db_password
        if db_port:
            db_config['port'] = db_port

        # Fetch and update
        success = clustering_service.keyword_classifier.fetch_and_update_from_db(
            db_config if db_config else None
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to sync categories")

        # Get updated info
        info = clustering_service.keyword_classifier.get_category_info()

        return {
            "success": True,
            "message": "Categories synchronized successfully from external DB",
            "timestamp": datetime.now().isoformat(),
            "updated_info": info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
