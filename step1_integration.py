#!/usr/bin/env python3
"""
FIXED: Step 1 Integration for Advanced Hajj/Umrah RAG System

Key fixes:
1. Correct class import (AdvancedIslamicChunker)
2. Enhanced error handling and debugging
3. Improved metrics tracking
4. Better chunk validation
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np

# Fixed imports
from advanced_chunking_system import AdvancedIslamicChunker  # ✅ FIXED
from advanced_embedding_faiss import AdvancedEmbeddingManager, AdvancedFAISSIndex, AdvancedRetrievalSystem

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

class Step1FoundationSystem:
    """
    FIXED: Complete Step 1 foundation system with enhanced debugging
    """
    
    def __init__(self, 
                 data_dir: str = "./data",
                 output_dir: str = "./foundation_output"):
        """Initialize the foundation system"""
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # System components
        self.chunker = AdvancedIslamicChunker()  # ✅ FIXED: Correct class name
        self.retrieval_system = None
        self.chunks = []
        
        # Enhanced metrics tracking
        self.metrics = {
            "chunking": {
                "total_files": 0,
                "files_processed": 0,
                "total_chunks": 0,
                "chunks_per_file": {},
                "guide_distribution": {},
                "section_detection_stats": {},
                "average_chunk_length": 0,
                "processing_time": 0
            },
            "embedding": {},
            "indexing": {},
            "retrieval": {}
        }
        
        logger.info(f"✅ Initialized Step 1 Foundation System")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def setup_input_files(self) -> List[str]:
        """Setup and validate input files with enhanced detection"""
        
        # Expected file patterns (more flexible)
        file_patterns = [
            "EN-102.txt", "EN-103.txt", "EN-104.txt", "EN-105.txt", "EN-106.txt",
            "EN-108.txt", "EN-109.txt", "EN-110.txt", "EN-111.txt", "EN-112.txt", 
            "EN-113.txt", "EN-114.txt", "EN-115.txt", "EN-116.txt"
        ]
        
        # Alternative patterns
        alt_patterns = ["EN*.txt", "en*.txt", "*102*.txt", "*103*.txt", "*104*.txt"]
        
        existing_files = []
        
        # Try exact matches first
        for pattern in file_patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                existing_files.append(str(file_path))
            else:
                # Try current directory
                if Path(pattern).exists():
                    existing_files.append(pattern)
        
        # If no exact matches, try broader search
        if not existing_files:
            logger.warning("No exact file matches found, searching broadly...")
            
            # Search all .txt files
            for txt_file in self.data_dir.glob("*.txt"):
                if "EN" in txt_file.name or "en" in txt_file.name:
                    existing_files.append(str(txt_file))
            
            # Search current directory
            for txt_file in Path(".").glob("*.txt"):
                if "EN" in txt_file.name or "en" in txt_file.name:
                    existing_files.append(str(txt_file))
        
        if not existing_files:
            logger.error("❌ No input files found!")
            logger.error("Please ensure files are available in:")
            logger.error(f"  - {self.data_dir}")
            logger.error(f"  - Current directory")
            
            # List what files are actually present
            all_files = list(self.data_dir.glob("*")) + list(Path(".").glob("*.txt"))
            logger.error(f"Available files: {[f.name for f in all_files[:10]]}")
            
            raise FileNotFoundError("No input files found")
        
        logger.info(f"✅ Found {len(existing_files)} input files:")
        for f in existing_files:
            file_size = Path(f).stat().st_size if Path(f).exists() else 0
            logger.info(f"  - {Path(f).name} ({file_size:,} bytes)")
        
        return existing_files

    def run_chunking_phase(self, input_files: List[str]) -> List[Document]:
        """Enhanced chunking phase with detailed debugging"""
        
        logger.info("=" * 60)
        logger.info("🔄 PHASE 1: ADVANCED SEMANTIC CHUNKING")
        logger.info("=" * 60)
        
        start_time = time.time()
        all_chunks = []
        
        self.metrics["chunking"]["total_files"] = len(input_files)
        
        for file_idx, file_path in enumerate(input_files):
            logger.info(f"\n📄 Processing file {file_idx + 1}/{len(input_files)}: {Path(file_path).name}")
            
            try:
                # Read file with encoding detection
                content = self._read_file_safely(file_path)
                
                if not content or len(content.strip()) < 100:
                    logger.warning(f"⚠️  File {Path(file_path).name} is too short or empty")
                    continue
                
                logger.info(f"📝 File content: {len(content):,} characters, {len(content.splitlines()):,} lines")
                
                # Extract guide info
                guide_info = self.chunker.extract_guide_info(Path(file_path).name)
                logger.info(f"📋 Guide: {guide_info['guide_type']} ({guide_info.get('guide_number', 'Unknown')})")
                
                # Create chunks with detailed logging
                file_chunks = self.chunker.create_hierarchical_chunks(content, guide_info['guide_type'])
                
                if not file_chunks:
                    logger.warning(f"⚠️  No chunks generated from {Path(file_path).name}")
                    continue
                
                # Validate chunks
                valid_chunks = self._validate_chunks(file_chunks, Path(file_path).name)
                
                logger.info(f"✅ Generated {len(valid_chunks)} valid chunks from {Path(file_path).name}")
                
                # Update metrics
                self.metrics["chunking"]["files_processed"] += 1
                self.metrics["chunking"]["chunks_per_file"][Path(file_path).name] = len(valid_chunks)
                self.metrics["chunking"]["guide_distribution"][guide_info['guide_type']] = \
                    self.metrics["chunking"]["guide_distribution"].get(guide_info['guide_type'], 0) + len(valid_chunks)
                
                all_chunks.extend(valid_chunks)
                
            except Exception as e:
                logger.error(f"❌ Failed to process {Path(file_path).name}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # Update final metrics
        self.metrics["chunking"]["total_chunks"] = len(all_chunks)
        self.metrics["chunking"]["processing_time"] = processing_time
        self.metrics["chunking"]["chunks_per_second"] = len(all_chunks) / processing_time if processing_time > 0 else 0
        
        if all_chunks:
            avg_length = sum(len(chunk.page_content) for chunk in all_chunks) / len(all_chunks)
            self.metrics["chunking"]["average_chunk_length"] = avg_length
        
        logger.info(f"\n📊 CHUNKING SUMMARY:")
        logger.info(f"Files processed: {self.metrics['chunking']['files_processed']}/{self.metrics['chunking']['total_files']}")
        logger.info(f"Total chunks: {len(all_chunks)}")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Average chunk length: {self.metrics['chunking'].get('average_chunk_length', 0):.0f} chars")
        
        if len(all_chunks) < 50:
            logger.warning(f"⚠️  Low chunk count! Expected 100-500, got {len(all_chunks)}")
            logger.warning("This may indicate chunking issues. Check section detection.")
        
        return all_chunks

    def _read_file_safely(self, file_path: str) -> str:
        """Safely read file with encoding detection"""
        
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.debug(f"Successfully read {Path(file_path).name} with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Failed to read {Path(file_path).name} with any encoding")
        return ""

    def _validate_chunks(self, chunks: List[Document], filename: str) -> List[Document]:
        """Validate and filter chunks"""
        
        valid_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Check basic validity
            if not chunk.page_content or len(chunk.page_content.strip()) < 30:
                logger.debug(f"Skipping too short chunk {i} from {filename}")
                continue
            
            # Check metadata
            if not hasattr(chunk, 'metadata') or not chunk.metadata:
                logger.warning(f"Chunk {i} from {filename} missing metadata")
                # Add basic metadata
                chunk.metadata = {
                    'chunk_id': f"{filename}_{i}",
                    'guide_type': 'Unknown',
                    'content_type': 'general',
                    'urgency_level': 2
                }
            
            # Ensure required metadata fields
            required_fields = ['chunk_id', 'guide_type', 'content_type', 'urgency_level']
            for field in required_fields:
                if field not in chunk.metadata:
                    logger.debug(f"Adding missing metadata field {field} to chunk {i}")
                    chunk.metadata[field] = self._get_default_metadata_value(field, filename, i)
            
            valid_chunks.append(chunk)
        
        logger.info(f"Validated {len(valid_chunks)}/{len(chunks)} chunks from {filename}")
        return valid_chunks

    def _get_default_metadata_value(self, field: str, filename: str, chunk_index: int):
        """Get default metadata values"""
        
        defaults = {
            'chunk_id': f"{filename}_{chunk_index}",
            'guide_type': 'General',
            'content_type': 'general',
            'urgency_level': 2,
            'islamic_terms': [],
            'location_contexts': [],
            'voice_keywords': []
        }
        
        return defaults.get(field, '')

    def run_embedding_phase(self, chunks: List[Document]) -> None:
        """Enhanced embedding phase with better error handling"""
        
        logger.info("=" * 60)
        logger.info("🔄 PHASE 2: ADVANCED MULTI-MODEL EMBEDDINGS")
        logger.info("=" * 60)
        
        if not chunks:
            logger.error("❌ No chunks to embed!")
            return
        
        start_time = time.time()
        
        try:
            # Initialize embedding manager
            self.embedding_manager = AdvancedEmbeddingManager()
            
            # Create embeddings
            logger.info(f"Creating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_manager.embed_chunks_batch(chunks, batch_size=8)
            
            embedding_time = time.time() - start_time
            
            # Validate embeddings
            primary_count = len(embeddings.get("primary", {}))
            multilingual_count = len(embeddings.get("multilingual", {}))
            
            logger.info(f"✅ Created {primary_count} primary + {multilingual_count} multilingual embeddings")
            
            # Record metrics
            self.metrics["embedding"] = {
                "total_chunks": len(chunks),
                "embedding_time": embedding_time,
                "chunks_per_second": len(chunks) / embedding_time if embedding_time > 0 else 0,
                "primary_model": getattr(self.embedding_manager, 'model_name', 'unknown'),
                "primary_dim": self.embedding_manager.primary_dim,
                "multilingual_dim": self.embedding_manager.multilingual_dim,
                "cache_hits": len(self.embedding_manager.embedding_cache),
                "primary_embeddings_created": primary_count,
                "multilingual_embeddings_created": multilingual_count
            }
            
            logger.info(f"Embedding completed in {embedding_time:.2f}s")
            logger.info(f"Speed: {self.metrics['embedding']['chunks_per_second']:.1f} chunks/second")
            
            # Store for next phase
            self.embeddings = embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding phase failed: {e}")
            raise

    def run_indexing_phase(self, chunks: List[Document]) -> None:
        """Enhanced indexing phase with validation"""
        
        logger.info("=" * 60)
        logger.info("🔄 PHASE 3: ADVANCED FAISS INDEXING")
        logger.info("=" * 60)
        
        if not hasattr(self, 'embeddings') or not self.embeddings:
            logger.error("❌ No embeddings available for indexing!")
            return
        
        start_time = time.time()
        
        try:
            # Initialize FAISS index
            self.faiss_index = AdvancedFAISSIndex(
                self.embedding_manager.primary_dim,
                self.embedding_manager.multilingual_dim,
                str(self.output_dir / "faiss_indices")
            )
            
            # Build indices using HNSW for speed
            logger.info("Building FAISS indices...")
            self.faiss_index.build_indices(chunks, self.embeddings, index_type="hnsw")

            
            indexing_time = time.time() - start_time
            
            # Validate indices
            primary_count = self.faiss_index.primary_index.ntotal if self.faiss_index.primary_index else 0
            multilingual_count = self.faiss_index.multilingual_index.ntotal if self.faiss_index.multilingual_index else 0
            
            logger.info(f"✅ Built indices: Primary={primary_count}, Multilingual={multilingual_count}")
            
            # Record metrics
            self.metrics["indexing"] = {
                "total_vectors": len(chunks),
                "indexing_time": indexing_time,
                "vectors_per_second": len(chunks) / indexing_time if indexing_time > 0 else 0,
                "index_type": "HNSW",
                "primary_index_size": primary_count,
                "multilingual_index_size": multilingual_count
            }
            
            # Save indices
            self.faiss_index.save_indices()
            
            logger.info(f"Indexing completed in {indexing_time:.2f}s")
            logger.info(f"Speed: {self.metrics['indexing']['vectors_per_second']:.1f} vectors/second")
            
        except Exception as e:
            logger.error(f"❌ Indexing phase failed: {e}")
            raise

    def build_retrieval_system(self, chunks: List[Document]) -> None:
        """Build the complete retrieval system"""
        
        logger.info("=" * 60)
        logger.info("🔄 PHASE 4: ASSEMBLING RETRIEVAL SYSTEM")
        logger.info("=" * 60)
        
        try:
            # Create integrated retrieval system
            self.retrieval_system = AdvancedRetrievalSystem(
                self.embedding_manager,
                self.faiss_index
            )
            
            # Store chunks for reference
            self.chunks = chunks
            
            logger.info("✅ Retrieval system assembled successfully")
            logger.info(f"Ready for queries with {len(chunks)} chunks indexed")
            
        except Exception as e:
            logger.error(f"❌ Failed to build retrieval system: {e}")
            raise

    def run_evaluation_phase(self) -> Dict[str, Any]:
        """Enhanced evaluation with better test cases"""
        
        logger.info("=" * 60)
        logger.info("🔄 PHASE 5: COMPREHENSIVE EVALUATION")
        logger.info("=" * 60)
        
        if not self.retrieval_system:
            logger.error("❌ No retrieval system available for evaluation!")
            return {}
        
        # Enhanced test queries with better coverage
        test_queries = [
            # Financial guidance (EN-102)
            {"query": "currency exchange Saudi Arabia", "expected_guides": ["Financial"], "urgency": 3},
            {"query": "electronic payment methods", "expected_guides": ["Financial"], "urgency": 2},
            
            # Legal guidance (EN-103)
            {"query": "Umrah visa requirements", "expected_guides": ["Legal"], "urgency": 4},
            {"query": "service provider obligations", "expected_guides": ["Legal"], "urgency": 3},
            
            # Grand Mosque (EN-104)
            {"query": "Ka'ba Black Stone location", "expected_guides": ["Grand Mosque"], "urgency": 3},
            {"query": "wheelchair service mosque", "expected_guides": ["Grand Mosque"], "urgency": 2},
            
            # Umrah (EN-105)
            {"query": "how to perform Tawaf", "expected_guides": ["Umrah"], "urgency": 4},
            {"query": "Sa'i between Safa Marwa", "expected_guides": ["Umrah"], "urgency": 4},
            
            # Ihram (EN-110)
            {"query": "Ihram clothing requirements", "expected_guides": ["Ihram"], "urgency": 4},
            {"query": "Miqat boundary locations", "expected_guides": ["Ihram"], "urgency": 3},
            
            # Emergency scenarios
            {"query": "emergency medical help", "expected_guides": ["Any"], "urgency": 5},
            {"query": "lost in crowd what to do", "expected_guides": ["Any"], "urgency": 4}
        ]
        
        logger.info(f"Testing {len(test_queries)} evaluation queries...")
        
        # Enhanced evaluation metrics
        retrieval_times = []
        relevance_scores = []
        guide_matches = []
        content_quality_scores = []
        
        for i, test_case in enumerate(test_queries):
            query = test_case["query"]
            expected_guides = test_case["expected_guides"]
            
            try:
                # Test retrieval
                start_time = time.time()
                results = self.retrieval_system.retrieve(query, k=5, model_type="primary")
                retrieval_time = (time.time() - start_time) * 1000  # ms
                
                retrieval_times.append(retrieval_time)
                
                if results:
                    # Analyze results
                    retrieved_guides = [result[0].metadata.get("guide_type", "Unknown") for result in results]
                    
                    # Guide matching score
                    if "Any" in expected_guides:
                        guide_score = 1.0  # Any guide is acceptable
                    else:
                        matches = sum(1 for guide in retrieved_guides if guide in expected_guides)
                        guide_score = matches / max(len(expected_guides), 1)
                    
                    guide_matches.append(guide_score)
                    
                    # Content relevance (simple keyword matching)
                    query_words = set(query.lower().split())
                    content_scores = []
                    
                    for result in results:
                        content_words = set(result[0].page_content.lower().split())
                        overlap = len(query_words.intersection(content_words))
                        content_score = overlap / len(query_words) if query_words else 0
                        content_scores.append(content_score)
                    
                    avg_content_score = sum(content_scores) / len(content_scores)
                    content_quality_scores.append(avg_content_score)
                    relevance_scores.append((guide_score + avg_content_score) / 2)
                    
                    logger.info(f"✅ Query {i+1}: '{query}' -> {len(results)} results, {guide_score:.2f} guide match")
                
                else:
                    logger.warning(f"⚠️  Query {i+1}: '{query}' -> No results")
                    guide_matches.append(0.0)
                    content_quality_scores.append(0.0)
                    relevance_scores.append(0.0)
                    
            except Exception as e:
                logger.error(f"❌ Query {i+1} failed: {e}")
                retrieval_times.append(999999)  # Mark as failed
                guide_matches.append(0.0)
                content_quality_scores.append(0.0)
                relevance_scores.append(0.0)
        
        # Calculate comprehensive metrics
        if retrieval_times and guide_matches and relevance_scores:
            valid_times = [t for t in retrieval_times if t < 999999]
            
            evaluation_results = {
                "total_queries": len(test_queries),
                "successful_queries": len(valid_times),
                "average_retrieval_time_ms": np.mean(valid_times) if valid_times else 0,
                "max_retrieval_time_ms": np.max(valid_times) if valid_times else 0,
                "min_retrieval_time_ms": np.min(valid_times) if valid_times else 0,
                "average_relevance_score": np.mean(relevance_scores),
                "average_guide_accuracy": np.mean(guide_matches),
                "average_content_quality": np.mean(content_quality_scores),
                "queries_under_100ms": sum(1 for t in valid_times if t < 100),
                "queries_under_500ms": sum(1 for t in valid_times if t < 500),
                "voice_readiness_score": sum(1 for t in valid_times if t < 100) / len(valid_times) if valid_times else 0,
                "system_reliability": len(valid_times) / len(test_queries)
            }
        else:
            evaluation_results = {
                "total_queries": len(test_queries),
                "successful_queries": 0,
                "error": "No successful retrievals"
            }
        
        self.metrics["retrieval"] = evaluation_results
        
        # Enhanced logging
        logger.info("🎯 EVALUATION RESULTS:")
        if evaluation_results.get("successful_queries", 0) > 0:
            logger.info(f"Successful queries: {evaluation_results['successful_queries']}/{evaluation_results['total_queries']}")
            logger.info(f"Average retrieval time: {evaluation_results['average_retrieval_time_ms']:.1f}ms")
            logger.info(f"Guide accuracy: {evaluation_results['average_guide_accuracy']:.2f}")
            logger.info(f"Content quality: {evaluation_results['average_content_quality']:.2f}")
            logger.info(f"Voice-ready queries: {evaluation_results['queries_under_100ms']}/{len(valid_times)}")
        else:
            logger.error("❌ No successful retrievals! System needs debugging.")
        
        return evaluation_results

    def save_system_outputs(self) -> None:
        """Enhanced output saving with detailed debugging info"""
        
        logger.info("💾 Saving system outputs...")
        
        # Save comprehensive metrics
        metrics_file = self.output_dir / "foundation_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False, default=str)
        
        # Save detailed chunk analysis
        if self.chunks:
            chunk_analysis = self._analyze_chunk_distribution()
            
            analysis_file = self.output_dir / "chunk_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_analysis, f, indent=2, ensure_ascii=False)
        
        # Save system configuration
        config = {
            "system_version": "Step 1 Foundation - Fixed",
            "fixes_applied": [
                "Corrected AdvancedIslamicChunker import",
                "Enhanced section detection for guide format",
                "Improved error handling and validation",
                "Added comprehensive debugging",
                "Enhanced metadata generation"
            ],
            "components": {
                "chunking": "AdvancedIslamicChunker with multi-strategy section detection",
                "embeddings": "E5-large-v2 with fallback support",
                "indexing": "FAISS HNSW optimized for <100ms retrieval",
                "evaluation": "Enhanced test suite with guide-specific queries"
            }
        }
        
        config_file = self.output_dir / "system_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Save the retrieval system
        if self.retrieval_system:
            self.retrieval_system.save_system()
        
        logger.info(f"✅ All outputs saved to: {self.output_dir}")

    def _analyze_chunk_distribution(self) -> Dict[str, Any]:
        """Analyze chunk distribution for debugging"""
        
        analysis = {
            "total_chunks": len(self.chunks),
            "chunks_by_guide": {},
            "chunks_by_content_type": {},
            "chunks_by_urgency": {},
            "length_distribution": {},
            "sample_chunks": []
        }
        
        lengths = []
        
        for chunk in self.chunks:
            # Guide type distribution
            guide_type = chunk.metadata.get("guide_type", "Unknown")
            analysis["chunks_by_guide"][guide_type] = analysis["chunks_by_guide"].get(guide_type, 0) + 1
            
            # Content type distribution
            content_type = chunk.metadata.get("content_type", "unknown")
            analysis["chunks_by_content_type"][content_type] = analysis["chunks_by_content_type"].get(content_type, 0) + 1
            
            # Urgency distribution
            urgency = chunk.metadata.get("urgency_level", 0)
            analysis["chunks_by_urgency"][str(urgency)] = analysis["chunks_by_urgency"].get(str(urgency), 0) + 1
            
            # Length tracking
            length = len(chunk.page_content)
            lengths.append(length)
        
        # Length distribution
        if lengths:
            analysis["length_distribution"] = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "median": sorted(lengths)[len(lengths) // 2]
            }
        
        # Sample chunks for inspection
        sample_size = min(5, len(self.chunks))
        for i in range(sample_size):
            chunk = self.chunks[i]
            analysis["sample_chunks"].append({
                "chunk_id": chunk.metadata.get("chunk_id", f"chunk_{i}"),
                "guide_type": chunk.metadata.get("guide_type", "Unknown"),
                "content_type": chunk.metadata.get("content_type", "unknown"),
                "length": len(chunk.page_content),
                "preview": chunk.page_content[:200] + "..."
            })
        
        return analysis

    def print_final_report(self) -> None:
        """Enhanced final report with actionable insights"""
        
        print("\n" + "=" * 80)
        print("🎯 STEP 1 FOUNDATION SYSTEM - ENHANCED FINAL REPORT")
        print("=" * 80)
        
        # System Status
        chunks_generated = self.metrics['chunking']['total_chunks']
        files_processed = self.metrics['chunking']['files_processed']
        total_files = self.metrics['chunking']['total_files']
        
        print(f"\n📊 SYSTEM OVERVIEW:")
        print(f"Files processed: {files_processed}/{total_files}")
        print(f"Total chunks generated: {chunks_generated}")
        print(f"System status: {'✅ HEALTHY' if chunks_generated > 50 else '⚠️  NEEDS ATTENTION'}")
        
        # Performance Metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"Chunking speed: {self.metrics['chunking'].get('chunks_per_second', 0):.1f} chunks/second")
        print(f"Embedding speed: {self.metrics['embedding'].get('chunks_per_second', 0):.1f} chunks/second")
        print(f"Indexing speed: {self.metrics['indexing'].get('vectors_per_second', 0):.1f} vectors/second")
        
        # Retrieval Quality
        retrieval_metrics = self.metrics.get('retrieval', {})
        if retrieval_metrics:
            avg_time = retrieval_metrics.get('average_retrieval_time_ms', 0)
            guide_accuracy = retrieval_metrics.get('average_guide_accuracy', 0)
            content_quality = retrieval_metrics.get('average_content_quality', 0)
            
            print(f"\n🎯 RETRIEVAL QUALITY:")
            print(f"Average retrieval time: {avg_time:.1f}ms")
            print(f"Guide accuracy: {guide_accuracy:.2f} {'✅' if guide_accuracy > 0.5 else '❌'}")
            print(f"Content quality: {content_quality:.2f} {'✅' if content_quality > 0.3 else '❌'}")
            print(f"Voice readiness: {retrieval_metrics.get('voice_readiness_score', 0):.1%}")
        
        # Content Distribution
        print(f"\n📚 CONTENT DISTRIBUTION:")
        guide_dist = self.metrics['chunking'].get('guide_distribution', {})
        for guide_type, count in sorted(guide_dist.items()):
            print(f"  {guide_type}: {count} chunks")
        
        # Diagnostic Information
        print(f"\n🔍 DIAGNOSTIC INFO:")
        
        if chunks_generated < 50:
            print("❌ ISSUE: Low chunk count detected!")
            print("   Possible causes:")
            print("   - Section detection failing")
            print("   - File format not recognized")
            print("   - Content filtering too aggressive")
            print("   Recommendations:")
            print("   - Check file content format")
            print("   - Review section detection patterns")
            print("   - Examine chunker logs for errors")
        
        if retrieval_metrics.get('average_guide_accuracy', 0) < 0.3:
            print("❌ ISSUE: Poor retrieval accuracy!")
            print("   Possible causes:")
            print("   - Metadata not properly assigned")
            print("   - Embedding quality issues")
            print("   - Index building problems")
            print("   Recommendations:")
            print("   - Verify chunk metadata")
            print("   - Test embedding similarity")
            print("   - Check FAISS index integrity")
        
        # Next Steps
        print(f"\n🚀 NEXT STEPS:")
        if chunks_generated > 50 and retrieval_metrics.get('average_guide_accuracy', 0) > 0.3:
            print("✅ System ready for Step 2: Enhanced Retrieval")
            print("   - Multi-vector embeddings combination")
            print("   - Hybrid dense + sparse retrieval")
            print("   - Cross-encoder reranking")
        else:
            print("🔧 System needs debugging before Step 2")
            print("   - Fix chunking issues first")
            print("   - Validate metadata generation")
            print("   - Test retrieval with known queries")
        
        print(f"\n📁 Output Directory: {self.output_dir}")
        print("=" * 80)

    def run_complete_foundation(self) -> None:
        """Run the complete enhanced Step 1 foundation system"""
        
        logger.info("🚀 Starting Enhanced Step 1 Foundation System...")
        logger.info("Building: Advanced chunking + embeddings + FAISS + comprehensive evaluation")
        
        try:
            # Phase 1: Setup and validate inputs
            input_files = self.setup_input_files()
            
            # Phase 2: Enhanced semantic chunking
            chunks = self.run_chunking_phase(input_files)
            
            if not chunks:
                logger.error("❌ No chunks generated! Cannot proceed.")
                return
            
            # Phase 3: Multi-model embeddings
            self.run_embedding_phase(chunks)
            
            # Phase 4: FAISS indexing
            self.run_indexing_phase(chunks)
            
            # Phase 5: Assemble retrieval system
            self.build_retrieval_system(chunks)
            
            # Phase 6: Comprehensive evaluation
            self.run_evaluation_phase()
            
            # Phase 7: Save outputs
            self.save_system_outputs()
            
            # Phase 8: Final report
            self.print_final_report()
            
            logger.info("🎉 Enhanced Step 1 Foundation System completed!")
            
        except Exception as e:
            logger.error(f"❌ Foundation system failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function to run enhanced Step 1 foundation system"""
    
    print("🔧 Enhanced Step 1: Advanced Foundation for Hajj/Umrah RAG System")
    print("Key improvements:")
    print("- Fixed class import issues")
    print("- Enhanced section detection for guide format")
    print("- Comprehensive error handling")
    print("- Detailed debugging and metrics")
    print("- Better chunk validation")
    print("-" * 60)
    
    # Initialize and run foundation system
    foundation = Step1FoundationSystem()
    foundation.run_complete_foundation()
    
    print("\n🎯 Enhancement Summary:")
    print("1. ✅ Fixed AdvancedIslamicChunker import")
    print("2. ✅ Improved section detection patterns")
    print("3. ✅ Added comprehensive validation")
    print("4. ✅ Enhanced error handling")
    print("5. ✅ Better evaluation metrics")

if __name__ == "__main__":
    main()