import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

# Core ML libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("sentence-transformers not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("faiss not available")
    FAISS_AVAILABLE = False

# Simple Document class if langchain not available
try:
    from langchain.schema import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedEmbeddingManager:
    """Advanced embedding system using state-of-the-art models"""
    
    def __init__(self, 
                 primary_model: str = "intfloat/e5-large-v2",
                 fallback_model: str = "all-MiniLM-L6-v2",
                 cache_dir: str = "./embedding_cache"):
        """Initialize embedding manager with fallback options"""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info("Loading embedding models...")
        
        # Try primary model, fallback to smaller model if needed
        try:
            self.primary_embedder = SentenceTransformer(primary_model)
            self.primary_dim = self.primary_embedder.get_sentence_embedding_dimension()
            logger.info(f"Loaded primary model {primary_model} (dim: {self.primary_dim})")
            self.model_name = primary_model
        except Exception as e:
            logger.warning(f"Failed to load primary model {primary_model}: {e}")
            logger.info(f"Falling back to {fallback_model}")
            try:
                self.primary_embedder = SentenceTransformer(fallback_model)
                self.primary_dim = self.primary_embedder.get_sentence_embedding_dimension()
                logger.info(f"Loaded fallback model {fallback_model} (dim: {self.primary_dim})")
                self.model_name = fallback_model
            except Exception as e2:
                raise RuntimeError(f"Failed to load any embedding model: {e2}")
        
        # For simplicity, use same model for multilingual (can be enhanced later)
        self.multilingual_embedder = self.primary_embedder
        self.multilingual_dim = self.primary_dim
        
        # Embedding cache for efficiency
        self.embedding_cache = {}
        self.load_cache()

    def load_cache(self):
        """Load embedding cache from disk"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self.embedding_cache = {}

    def save_cache(self):
        """Save embedding cache to disk"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")

    def get_cache_key(self, text: str, model_type: str) -> str:
        """Generate cache key for text and model combination"""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model_type}_{text_hash}"

    def embed_text_primary(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Create embedding using primary model"""
        
        if use_cache:
            cache_key = self.get_cache_key(text, "primary")
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        # Use appropriate prompt format for E5 models
        if "e5" in self.model_name.lower():
            prefixed_text = f"passage: {text}"
        else:
            prefixed_text = text
            
        embedding = self.primary_embedder.encode(prefixed_text, normalize_embeddings=True)
        
        if use_cache:
            self.embedding_cache[cache_key] = embedding
        
        return embedding

    def embed_text_multilingual(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Create embedding using multilingual model"""
        
        if use_cache:
            cache_key = self.get_cache_key(text, "multilingual")
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        embedding = self.multilingual_embedder.encode(text, normalize_embeddings=True)
        
        if use_cache:
            self.embedding_cache[cache_key] = embedding
        
        return embedding

    def embed_query(self, query: str, model_type: str = "primary") -> np.ndarray:
        """Embed a query with appropriate formatting"""
        
        if model_type == "primary":
            if "e5" in self.model_name.lower():
                prefixed_query = f"query: {query}"
            else:
                prefixed_query = query
            return self.primary_embedder.encode(prefixed_query, normalize_embeddings=True)
        else:
            return self.multilingual_embedder.encode(query, normalize_embeddings=True)

    def embed_chunks_batch(self, chunks: List[Document], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Create embeddings for chunks in batches"""
        
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        
        # Prepare texts and metadata
        texts = [chunk.page_content for chunk in chunks]
        chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        
        # Create primary embeddings
        logger.info("Creating primary embeddings...")
        primary_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Add E5 prefix if needed
            if "e5" in self.model_name.lower():
                prefixed_texts = [f"passage: {text}" for text in batch_texts]
            else:
                prefixed_texts = batch_texts
            
            batch_embeddings = self.primary_embedder.encode(
                prefixed_texts, 
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            primary_embeddings.extend(batch_embeddings)
        
        # Create multilingual embeddings (same as primary for now)
        logger.info("Creating multilingual embeddings...")
        multilingual_embeddings = primary_embeddings.copy()
        
        # Create mapping
        embeddings = {
            "primary": {chunk_ids[i]: emb for i, emb in enumerate(primary_embeddings)},
            "multilingual": {chunk_ids[i]: emb for i, emb in enumerate(multilingual_embeddings)}
        }
        
        # Cache embeddings
        for i, (text, chunk_id) in enumerate(zip(texts, chunk_ids)):
            primary_key = self.get_cache_key(text, "primary")
            multilingual_key = self.get_cache_key(text, "multilingual")
            
            self.embedding_cache[primary_key] = primary_embeddings[i]
            self.embedding_cache[multilingual_key] = multilingual_embeddings[i]
        
        self.save_cache()
        
        logger.info(f"Created embeddings: Primary dim={self.primary_dim}, Multilingual dim={self.multilingual_dim}")
        
        return embeddings

class AdvancedFAISSIndex:
    """Advanced FAISS indexing system"""
    
    def __init__(self, 
                 primary_dim: int,
                 multilingual_dim: int,
                 index_dir: str = "./faiss_indices"):
        """Initialize FAISS indexing system"""
        
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        
        self.primary_dim = primary_dim
        self.multilingual_dim = multilingual_dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # Initialize indices
        self.primary_index = None
        self.multilingual_index = None
        
        # Metadata storage for filtering
        self.metadata_store = {}
        self.id_to_chunk_map = {}
        
        logger.info(f"Initialized FAISS indexing system (Primary: {primary_dim}D, Multilingual: {multilingual_dim}D)")

    def create_hnsw_index(self, dimension: int, ef_construction: int = 200, M: int = 32) -> faiss.IndexHNSWFlat:
        """Create HNSW index optimized for speed"""
        
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.ef_construction = ef_construction
        
        logger.info(f"Created HNSW index (dim={dimension}, M={M}, ef_construction={ef_construction})")
        return index

    def create_flat_index(self, dimension: int) -> faiss.IndexFlatL2:
        """Create flat index as fallback"""
        
        index = faiss.IndexFlatL2(dimension)
        logger.info(f"Created Flat index (dim={dimension})")
        return index

    def build_indices(self, 
                     chunks: List[Document], 
                     embeddings: Dict[str, Dict[str, np.ndarray]],
                     index_type: str = "hnsw") -> None:
        """Build FAISS indices from chunks and embeddings"""
        
        logger.info(f"Building {index_type.upper()} indices for {len(chunks)} chunks...")
        
        # Create indices based on type
        if index_type == "hnsw" and len(chunks) > 50:  # HNSW needs sufficient data
            self.primary_index = self.create_hnsw_index(self.primary_dim)
            self.multilingual_index = self.create_hnsw_index(self.multilingual_dim)
        else:
            # Use flat index for small datasets or as fallback
            self.primary_index = self.create_flat_index(self.primary_dim)
            self.multilingual_index = self.create_flat_index(self.multilingual_dim)
        
        # Prepare embedding matrices
        primary_embeddings_matrix = []
        multilingual_embeddings_matrix = []
        chunk_ids = []
        
        for chunk in chunks:
            chunk_id = chunk.metadata["chunk_id"]
            
            if chunk_id in embeddings["primary"] and chunk_id in embeddings["multilingual"]:
                primary_embeddings_matrix.append(embeddings["primary"][chunk_id])
                multilingual_embeddings_matrix.append(embeddings["multilingual"][chunk_id])
                chunk_ids.append(chunk_id)
                
                # Store metadata and chunk mapping
                self.metadata_store[len(chunk_ids) - 1] = chunk.metadata
                self.id_to_chunk_map[len(chunk_ids) - 1] = chunk
        
        # Convert to numpy arrays
        primary_matrix = np.array(primary_embeddings_matrix).astype('float32')
        multilingual_matrix = np.array(multilingual_embeddings_matrix).astype('float32')
        
        logger.info(f"Prepared embedding matrices: Primary {primary_matrix.shape}, Multilingual {multilingual_matrix.shape}")
        
        # Add embeddings to indices
        logger.info("Adding embeddings to indices...")
        self.primary_index.add(primary_matrix)
        self.multilingual_index.add(multilingual_matrix)
        
        # Set search parameters for HNSW
        if index_type == "hnsw":
            self.primary_index.hnsw.ef = 64
            self.multilingual_index.hnsw.ef = 64
        
        logger.info(f"Successfully built indices with {len(chunk_ids)} vectors")

    def search(self, 
               query_embedding: np.ndarray,
               k: int = 10,
               model_type: str = "primary",
               metadata_filters: Dict[str, Any] = None) -> Tuple[List[float], List[int]]:
        """Search the FAISS index with optional metadata filtering"""
        
        # Select appropriate index
        index = self.primary_index if model_type == "primary" else self.multilingual_index
        
        if index is None:
            raise ValueError(f"Index not built for model type: {model_type}")
        
        # Simple search without complex filtering for now
        similarities, indices = index.search(query_embedding.reshape(1, -1), k)
        return similarities[0].tolist(), indices[0].tolist()

    def get_chunks_by_indices(self, indices: List[int]) -> List[Document]:
        """Retrieve chunks by their indices"""
        return [self.id_to_chunk_map[idx] for idx in indices if idx in self.id_to_chunk_map]

    def save_indices(self):
        """Save FAISS indices and metadata to disk"""
        
        if self.primary_index is not None:
            primary_path = self.index_dir / "primary_index.faiss"
            faiss.write_index(self.primary_index, str(primary_path))
            logger.info(f"Saved primary index to {primary_path}")
        
        if self.multilingual_index is not None:
            multilingual_path = self.index_dir / "multilingual_index.faiss"
            faiss.write_index(self.multilingual_index, str(multilingual_path))
            logger.info(f"Saved multilingual index to {multilingual_path}")
        
        # Save metadata
        metadata_path = self.index_dir / "metadata_store.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, indent=2, ensure_ascii=False)
        
        # Save chunk mapping
        chunks_data = {}
        for idx, chunk in self.id_to_chunk_map.items():
            chunks_data[str(idx)] = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
        
        chunks_path = self.index_dir / "chunks_data.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Saved metadata and chunk data")

    def load_indices(self):
        """Load FAISS indices and metadata from disk"""
        
        try:
            primary_path = self.index_dir / "primary_index.faiss"
            if primary_path.exists():
                self.primary_index = faiss.read_index(str(primary_path))
                logger.info(f"Loaded primary index from {primary_path}")
            
            multilingual_path = self.index_dir / "multilingual_index.faiss"
            if multilingual_path.exists():
                self.multilingual_index = faiss.read_index(str(multilingual_path))
                logger.info(f"Loaded multilingual index from {multilingual_path}")
            
            # Load metadata
            metadata_path = self.index_dir / "metadata_store.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    raw_metadata = json.load(f)
                    self.metadata_store = {int(k): v for k, v in raw_metadata.items()}
            
            # Load chunk mapping
            chunks_path = self.index_dir / "chunks_data.json"
            if chunks_path.exists():
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    
                for idx_str, chunk_data in chunks_data.items():
                    idx = int(idx_str)
                    chunk = Document(
                        page_content=chunk_data["page_content"],
                        metadata=chunk_data["metadata"]
                    )
                    self.id_to_chunk_map[idx] = chunk
            
            logger.info("Successfully loaded all indices and metadata")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            return False

class AdvancedRetrievalSystem:
    """Complete retrieval system combining embeddings and FAISS indexing"""
    
    def __init__(self,
                 embedding_manager: AdvancedEmbeddingManager = None,
                 faiss_index: AdvancedFAISSIndex = None):
        """Initialize the retrieval system"""
        
        self.embedding_manager = embedding_manager
        self.faiss_index = faiss_index
        
        logger.info("Initialized Advanced Retrieval System")

    def build_from_chunks(self, chunks: List[Document], index_type: str = "hnsw") -> None:
        """Build complete retrieval system from chunks"""
        
        # Initialize embedding manager if not provided
        if self.embedding_manager is None:
            self.embedding_manager = AdvancedEmbeddingManager()
        
        # Create embeddings
        embeddings = self.embedding_manager.embed_chunks_batch(chunks)
        
        # Initialize FAISS index if not provided
        if self.faiss_index is None:
            self.faiss_index = AdvancedFAISSIndex(
                self.embedding_manager.primary_dim,
                self.embedding_manager.multilingual_dim
            )
        
        # Build indices
        self.faiss_index.build_indices(chunks, embeddings, index_type)
        
        logger.info("Built complete retrieval system")

    def retrieve(self, 
                query: str,
                k: int = 5,
                model_type: str = "primary") -> List[Tuple[Document, float]]:
        """Retrieve relevant chunks for a query"""
        
        if self.embedding_manager is None or self.faiss_index is None:
            raise ValueError("Retrieval system not initialized")
        
        # Create query embedding
        query_embedding = self.embedding_manager.embed_query(query, model_type)
        
        # Search FAISS index
        similarities, indices = self.faiss_index.search(
            query_embedding, k, model_type
        )
        
        # Get chunks
        chunks = self.faiss_index.get_chunks_by_indices(indices)
        
        # Combine with similarities
        results = []
        for chunk, similarity in zip(chunks, similarities):
            results.append((chunk, float(similarity)))
        
        return results

    def save_system(self):
        """Save the complete retrieval system"""
        if self.embedding_manager:
            self.embedding_manager.save_cache()
        if self.faiss_index:
            self.faiss_index.save_indices()
        logger.info("Saved complete retrieval system")

    def load_system(self):
        """Load the complete retrieval system"""
        success = True
        
        if self.embedding_manager is not None:
            self.embedding_manager.load_cache()
        
        if self.faiss_index is not None:
            success = self.faiss_index.load_indices()
        
        if success:
            logger.info("Loaded complete retrieval system")
        
        return success

def main():
    """Main function to demonstrate the retrieval system"""
    
    print("Advanced Retrieval System for Islamic Content")
    print("=" * 50)
    print("Components:")
    print("1. AdvancedEmbeddingManager - E5-large-v2 with fallback")
    print("2. AdvancedFAISSIndex - HNSW/Flat indices")
    print("3. AdvancedRetrievalSystem - Complete pipeline")
    print()
    print("Usage:")
    print("1. Run advanced_chunking_system.py to create chunks")
    print("2. Use this system to build embeddings and indices")
    print("3. Retrieve relevant content for queries")

if __name__ == "__main__":
    main()