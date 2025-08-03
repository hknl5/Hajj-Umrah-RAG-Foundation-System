# Hajj/Umrah RAG Foundation System

A foundation system for building Retrieval-Augmented Generation (RAG) applications for Islamic pilgrimage guidance. Processes Hajj and Umrah guides (EN-102 through EN-116) into an optimized knowledge base for question-answering systems.

## Overview

This is Step 1 of a RAG system that creates semantic chunks, embeddings, and FAISS indices from Islamic pilgrimage content. Designed for sub-100ms retrieval times and voice interface compatibility.

**Key Components:**
- Advanced Islamic content chunking with context preservation
- Multi-model embeddings (E5-large-v2 with fallback)
- FAISS HNSW indexing for fast similarity search
- Comprehensive evaluation with Islamic-specific queries

## Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM recommended
- Ollama with Qwen2.5:7b (optional)

### Installation
```bash
pip install -r requirements.txt
```

Optional Ollama setup:
```bash
# Install from https://ollama.ai/
ollama pull qwen2.5:7b
```

### Data Setup
Place EN-*.txt files in `./data/` directory:
```
data/
├── EN-102.txt  # Financial Awareness
├── EN-103.txt  # Legal Awareness  
├── EN-104.txt  # Grand Mosque
├── EN-105.txt  # Umrah Guide
└── ...
```

### Run the System
```bash
python step1_integration.py
```

Alternative with options:
```bash
python step1_runner.py --data-dir ./my_data --output-dir ./results
```

## System Architecture

**AdvancedIslamicChunker**: Semantic chunking with Islamic context preservation
- Multi-strategy section detection
- Adaptive chunk sizing based on content urgency
- Rich metadata with guide types and Islamic terms

**AdvancedEmbeddingManager**: E5-large-v2 embeddings with caching
- 1024-dimensional vectors
- Batch processing and persistent cache
- Automatic fallback to smaller models

**AdvancedFAISSIndex**: HNSW indices for fast retrieval
- Sub-100ms query response times
- Dual indexing for primary/multilingual models
- Metadata integration for filtering

**AdvancedRetrievalSystem**: Complete query pipeline
- Unified query interface
- Quality scoring and ranking
- Voice-optimized performance

## Expected Results

After running the integration:

**Performance Metrics**:
- 50-200 chunks per guide
- 1024-dimensional embeddings
- <100ms average retrieval time
- >70% query accuracy

**Output Files** (in `foundation_output/`):
- `faiss_indices/`: FAISS index files
- `foundation_metrics.json`: Performance statistics
- `chunk_analysis.json`: Content distribution analysis
- `system_config.json`: System configuration

## Usage Example

```python
from step1_integration import Step1FoundationSystem

# Initialize and build system
foundation = Step1FoundationSystem()
foundation.run_complete_foundation()

# Query the system
results = foundation.retrieval_system.retrieve(
    "How to perform Tawaf around Kaaba", 
    k=5
)

for chunk, score in results:
    print(f"Guide: {chunk.metadata['guide_type']}")
    print(f"Content: {chunk.page_content[:200]}...")
```

## Troubleshooting

**No chunks generated**: Check EN-*.txt file format and section headers

**Slow performance**: Install GPU libraries:
```bash
pip install faiss-gpu torch
```

**Poor accuracy**: Verify Ollama is running and qwen2.5:7b is available

**Missing files**: Ensure EN-*.txt files are in `./data/` or current directory
