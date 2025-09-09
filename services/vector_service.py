"""
Vector database services using ChromaDB for similarity search and storage
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import numpy as np
import uuid
import logging
import time
import json

logger = logging.getLogger(__name__)


class VectorService:
    """
    ChromaDB-based vector storage and similarity search services
    Optimized for multilingual topic clustering
    """

    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,
                    is_persistent=True
                )
            )

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",  # Use cosine similarity
                    "hnsw:construction_ef": 128,  # Construction parameter
                    "hnsw:M": 16,  # Number of bi-directional links
                    "hnsw:search_ef": 128,  # Search parameter
                    "description": "Multilingual topic clustering with BGE-M3 embeddings"
                }
            )

            logger.info(f"ChromaDB initialized successfully")
            logger.info(f"Collection: {self.collection_name}")
            logger.info(f"Persist directory: {self.persist_directory}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_topic(self,
                  topic_id: str,
                  embedding: np.ndarray,
                  metadata: Dict[str, Any],
                  document: str) -> bool:
        """
        Add a topic to the vector database

        Args:
            topic_id: Unique topic identifier
            embedding: Topic embedding vector
            metadata: Topic metadata
            document: Representative document text

        Returns:
            Success status
        """
        try:
            # Validate inputs
            if not topic_id or not isinstance(embedding, np.ndarray):
                raise ValueError("Invalid topic_id or embedding")

            # Ensure embedding is the right shape
            if len(embedding.shape) != 1:
                raise ValueError(f"Embedding must be 1D array, got shape: {embedding.shape}")

            # Prepare metadata (ChromaDB doesn't support all types)
            safe_metadata = self._prepare_metadata(metadata)

            # Add to collection
            self.collection.add(
                ids=[topic_id],
                embeddings=[embedding.tolist()],
                metadatas=[safe_metadata],
                documents=[document]
            )

            logger.info(f"Added topic to vector database: {topic_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add topic {topic_id} to vector database: {e}")
            return False

    def search_similar_topics(self,
                              query_embedding: np.ndarray,
                              n_results: int = 10,
                              threshold: float = 0.8,
                              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar topics in the vector database

        Args:
            query_embedding: Query embedding vector
            n_results: Maximum number of results
            threshold: Minimum similarity threshold
            filters: Optional metadata filters

        Returns:
            List of similar topics with metadata
        """
        try:
            # Validate query embedding
            if not isinstance(query_embedding, np.ndarray) or len(query_embedding.shape) != 1:
                raise ValueError("Query embedding must be 1D numpy array")

            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": min(n_results, 100),  # Limit to reasonable number
                "include": ['documents', 'metadatas', 'distances']
            }

            # Add filters if provided
            if filters:
                where_clause = self._build_where_clause(filters)
                if where_clause:
                    query_params["where"] = where_clause

            # Execute search
            results = self.collection.query(**query_params)

            # Process results
            similar_topics = []
            if results['ids'] and results['ids'][0]:
                for i, topic_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Convert distance to similarity

                    # Apply threshold filter
                    if similarity >= threshold:
                        similar_topics.append({
                            'topic_id': topic_id,
                            'similarity': similarity,
                            'distance': distance,
                            'metadata': results['metadatas'][0][i],
                            'document': results['documents'][0][i]
                        })

            # Sort by similarity (highest first)
            similar_topics.sort(key=lambda x: x['similarity'], reverse=True)

            logger.info(f"Found {len(similar_topics)} similar topics above threshold {threshold}")
            return similar_topics

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def update_topic(self,
                     topic_id: str,
                     embedding: np.ndarray,
                     metadata: Dict[str, Any],
                     document: str) -> bool:
        """
        Update an existing topic in the vector database

        Args:
            topic_id: Topic identifier
            embedding: Updated embedding
            metadata: Updated metadata
            document: Updated document

        Returns:
            Success status
        """
        try:
            safe_metadata = self._prepare_metadata(metadata)

            self.collection.upsert(
                ids=[topic_id],
                embeddings=[embedding.tolist()],
                metadatas=[safe_metadata],
                documents=[document]
            )

            logger.info(f"Updated topic in vector database: {topic_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update topic {topic_id}: {e}")
            return False

    def delete_topic(self, topic_id: str) -> bool:
        """
        Delete a topic from the vector database

        Args:
            topic_id: Topic identifier

        Returns:
            Success status
        """
        try:
            self.collection.delete(ids=[topic_id])
            logger.info(f"Deleted topic from vector database: {topic_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete topic {topic_id}: {e}")
            return False

    def get_topic(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific topic by ID

        Args:
            topic_id: Topic identifier

        Returns:
            Topic data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[topic_id],
                include=['documents', 'metadatas', 'embeddings']
            )

            if results['ids'] and len(results['ids']) > 0:
                return {
                    'topic_id': results['ids'][0],
                    'metadata': results['metadatas'][0],
                    'document': results['documents'][0],
                    'embedding': np.array(results['embeddings'][0])
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get topic {topic_id}: {e}")
            return None

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection

        Returns:
            Collection statistics
        """
        try:
            count = self.collection.count()

            # Get collection metadata
            collection_metadata = self.collection.metadata or {}

            return {
                "total_topics": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "metadata": collection_metadata
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "total_topics": 0,
                "collection_name": self.collection_name,
                "error": str(e)
            }

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB storage (convert complex types)

        Args:
            metadata: Original metadata

        Returns:
            ChromaDB-compatible metadata
        """
        safe_metadata = {}

        for key, value in metadata.items():
            if value is None:
                continue

            # Convert complex types to strings
            if isinstance(value, (list, dict)):
                safe_metadata[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (int, float, str, bool)):
                safe_metadata[key] = value
            else:
                safe_metadata[key] = str(value)

        return safe_metadata

    def _build_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build ChromaDB where clause from filters

        Args:
            filters: Filter dictionary

        Returns:
            ChromaDB where clause
        """
        if not filters:
            return None

        where_conditions = []

        for key, value in filters.items():
            if value is not None:
                where_conditions.append({
                    key: {"$eq": value}
                })

        if not where_conditions:
            return None

        if len(where_conditions) == 1:
            return where_conditions[0]

        return {"$and": where_conditions}

    def reset_collection(self):
        """Reset/clear the entire collection (use with caution!)"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 128,
                    "hnsw:M": 16,
                    "description": "Multilingual topic clustering with BGE-M3 embeddings"
                }
            )
            logger.warning(f"Collection {self.collection_name} has been reset")

        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise

    def backup_collection(self, backup_path: str) -> bool:
        """
        Create a backup of the collection

        Args:
            backup_path: Path to save backup

        Returns:
            Success status
        """
        try:
            # Get all data from collection
            all_data = self.collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )

            backup_data = {
                'collection_name': self.collection_name,
                'metadata': self.collection.metadata,
                'data': all_data,
                'timestamp': time.time()
            }

            # Save to file
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Collection backed up to: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup collection: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector services

        Returns:
            Health status information
        """
        try:
            # Test basic operations
            test_embedding = np.random.rand(1024).astype('float32')
            test_id = f"health_check_{int(time.time())}"

            # Test add
            add_success = self.add_topic(
                topic_id=test_id,
                embedding=test_embedding,
                metadata={"test": True},
                document="Health check test"
            )

            # Test search
            search_results = self.search_similar_topics(
                query_embedding=test_embedding,
                n_results=1,
                threshold=0.0
            )

            # Test delete
            delete_success = self.delete_topic(test_id)

            # Get stats
            stats = self.get_collection_stats()

            return {
                "status": "healthy",
                "operations": {
                    "add": add_success,
                    "search": len(search_results) > 0,
                    "delete": delete_success
                },
                "stats": stats,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }