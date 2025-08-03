#!/usr/bin/env python3
"""
FIXED: Advanced Islamic Chunking System
Addresses all the identified issues with section detection and chunking
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Simple fallback Document class
    class Document:
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedIslamicChunker:  # ✅ FIXED: Renamed to match import
    """Advanced semantic chunking with Islamic context preservation"""
    
    def __init__(self, model_name: str = "qwen2.5:7b"):
        """Initialize the chunker with fallback options"""
        
        # Initialize LLM with fallback
        self.llm = None
        if OLLAMA_AVAILABLE:
            try:
                self.llm = Ollama(model=model_name, temperature=0.1)
                logger.info(f"Initialized Ollama LLM: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {e}")
        
        # Initialize embedder with fallback
        self.embedder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized sentence transformer embedder")
            except Exception as e:
                logger.warning(f"Failed to initialize embedder: {e}")
        
        # Islamic knowledge base
        self.islamic_context = {
            'rituals': {
                'tawaf': ['circumambulation', 'kaaba', '7 circuits', 'black stone', 'counter-clockwise'],
                'sai': ['safa', 'marwa', '7 times', 'walking', 'running'],
                'ihram': ['sacred state', 'miqat', 'restrictions', 'white garments', 'intention'],
                'wuquf': ['standing', 'arafah', 'day 9', 'sunset', 'most important'],
                'ramy': ['stoning', 'jamarat', 'pebbles', '7 stones', 'shaitan'],
                'qurbani': ['sacrifice', 'animal', 'eid', 'sharing meat'],
                'halq': ['shaving', 'cutting hair', 'completion', 'men', 'women']
            },
            'locations': {
                'makkah': ['grand mosque', 'kaaba', 'sacred city', 'birthplace'],
                'madinah': ['prophets mosque', 'grave', 'second holiest'],
                'mina': ['tent city', 'jamarat', 'days of tashreeq'],
                'arafah': ['mount mercy', 'plain', 'day of arafah'],
                'muzdalifah': ['open area', 'pebbles', 'night stay']
            },
            'journey_stages': {
                'preparation': ['visa', 'ihram clothes', 'vaccinations', 'bookings'],
                'arrival': ['airport', 'immigration', 'hotel', 'first sight'],
                'umrah': ['tawaf', 'sai', 'cutting hair', 'completion'],
                'hajj_days': ['8th', '9th', '10th', '11th', '12th', '13th'],
                'departure': ['farewell tawaf', 'airport', 'return']
            }
        }
        
        # Guide type mapping
        self.guide_type_mapping = {
            'EN-102': 'Financial Awareness',
            'EN-103': 'Legal Awareness', 
            'EN-104': 'Grand Mosque',
            'EN-105': 'Umrah',
            'EN-106': 'Prophets Mosque',
            'EN-108': 'Makkah Landmarks',
            'EN-109': 'Madinah Landmarks',
            'EN-110': 'Ihram',
            'EN-111': 'Mina',
            'EN-112': 'Arafah',
            'EN-113': 'Muzdalifah',
            'EN-114': 'Day of Sacrifice',
            'EN-115': 'Jamarat',
            'EN-116': 'Cybersecurity'
        }

    def extract_guide_info(self, filename: str) -> Dict[str, str]:
        """✅ FIXED: Extract guide information from filename"""
        
        # Extract guide number from filename
        match = re.search(r'EN-(\d+)', filename)
        if match:
            guide_num = f"EN-{match.group(1)}"
            guide_type = self.guide_type_mapping.get(guide_num, 'General Guide')
        else:
            guide_type = 'General Guide'
        
        return {
            'filename': filename,
            'guide_type': guide_type,
            'guide_number': guide_num if match else 'Unknown'
        }

    def create_hierarchical_chunks(self, text: str, guide_type: str) -> List[Document]:
        """✅ FIXED: Create hierarchical chunks with improved section detection"""
        
        logger.info(f"Processing {guide_type} guide...")
        
        # Step 1: Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Step 2: Detect sections using multiple strategies
        sections = self._detect_sections_multi_strategy(cleaned_text, guide_type)
        
        logger.info(f"Detected {len(sections)} sections in {guide_type}")
        
        # Step 3: Create chunks from sections
        all_chunks = []
        for i, section in enumerate(sections):
            section_chunks = self._create_chunks_from_section(
                section, guide_type, section_index=i
            )
            all_chunks.extend(section_chunks)
        
        logger.info(f"Generated {len(all_chunks)} chunks from {guide_type}")
        
        return all_chunks

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove page numbers and navigation elements
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Click here.*?download.*?\n', '', text, flags=re.IGNORECASE)
        
        return text.strip()

    def _detect_sections_multi_strategy(self, text: str, guide_type: str) -> List[Dict[str, Any]]:
        """✅ FIXED: Detect sections using multiple strategies tailored to your data format"""
        
        sections = []
        
        # Strategy 1: Contents-based detection (primary for your format)
        contents_sections = self._detect_contents_sections(text)
        if len(contents_sections) > 2:
            logger.info(f"Using contents-based detection: {len(contents_sections)} sections")
            return contents_sections
        
        # Strategy 2: Header pattern detection
        header_sections = self._detect_header_sections(text)
        if len(header_sections) > 2:
            logger.info(f"Using header-based detection: {len(header_sections)} sections")
            return header_sections
        
        # Strategy 3: Topic boundary detection
        topic_sections = self._detect_topic_boundaries(text, guide_type)
        if len(topic_sections) > 2:
            logger.info(f"Using topic-based detection: {len(topic_sections)} sections")
            return topic_sections
        
        # Strategy 4: Fallback - split by length
        logger.warning("Using fallback length-based splitting")
        return self._fallback_length_split(text, guide_type)

    def _detect_contents_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect sections based on table of contents structure"""
        
        sections = []
        
        # Find the contents section
        contents_match = re.search(
            r'Contents.*?Click on a title to get to the desired page(.*?)(?=^[A-Z][^a-z]*$|\Z)',
            text, 
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        
        if not contents_match:
            return sections
        
        contents_text = contents_match.group(1)
        
        # Extract section titles from contents
        # Look for lines that appear to be section titles
        title_patterns = [
            r'^([A-Z][A-Za-z\s]{10,50})$',  # Title case sections
            r'^([A-Z\s]{5,30})$',           # ALL CAPS sections
            r'^([A-Za-z][^.!?]*[A-Za-z])$' # General sections
        ]
        
        potential_titles = []
        for line in contents_text.split('\n'):
            line = line.strip()
            if len(line) > 5 and len(line) < 80:
                for pattern in title_patterns:
                    if re.match(pattern, line):
                        potential_titles.append(line)
                        break
        
        logger.info(f"Found {len(potential_titles)} potential section titles")
        
        # Find these titles in the main text and create sections
        text_lines = text.split('\n')
        
        for title in potential_titles:
            # Look for the title in the main text
            title_escaped = re.escape(title.strip())
            
            for i, line in enumerate(text_lines):
                if re.search(title_escaped, line.strip(), re.IGNORECASE):
                    # Found the section start
                    section_start = i
                    
                    # Find section end (next title or end of text)
                    section_end = len(text_lines)
                    for next_title in potential_titles[potential_titles.index(title) + 1:]:
                        next_title_escaped = re.escape(next_title.strip())
                        for j in range(i + 10, len(text_lines)):  # Skip at least 10 lines
                            if re.search(next_title_escaped, text_lines[j].strip(), re.IGNORECASE):
                                section_end = j
                                break
                        if section_end < len(text_lines):
                            break
                    
                    # Extract section content
                    section_content = '\n'.join(text_lines[section_start:section_end]).strip()
                    
                    if len(section_content) > 100:  # Only include substantial sections
                        sections.append({
                            'title': title.strip(),
                            'content': section_content,
                            'start_line': section_start,
                            'end_line': section_end,
                            'type': 'contents_based'
                        })
                    break
        
        return sections

    def _detect_header_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect sections based on header patterns"""
        
        sections = []
        
        # Updated header patterns for your data format
        header_patterns = [
            r'^([A-Z\s]{3,40})$',                    # ALL CAPS headers
            r'^([A-Z][a-z\s]{10,60})$',             # Title case headers
            r'^([A-Za-z\s]{5,50})\s*$',             # General headers
            r'^\s*([A-Z][^.!?]*[A-Za-z])\s*$',     # Clean headers
        ]
        
        lines = text.split('\n')
        header_positions = []
        
        for i, line in enumerate(lines):
            clean_line = line.strip()
            
            # Skip very short or very long lines
            if len(clean_line) < 5 or len(clean_line) > 80:
                continue
                
            # Skip lines with common non-header patterns
            if any(pattern in clean_line.lower() for pattern in 
                   ['click here', 'download', 'page', 'www.', 'http', '@']):
                continue
            
            # Check if line matches header patterns
            for pattern in header_patterns:
                if re.match(pattern, clean_line):
                    # Additional validation
                    if self._is_likely_header(clean_line, lines, i):
                        header_positions.append((i, clean_line))
                        break
        
        logger.info(f"Found {len(header_positions)} potential headers")
        
        # Create sections from headers
        for i, (line_num, header) in enumerate(header_positions):
            start_line = line_num
            end_line = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(lines)
            
            section_content = '\n'.join(lines[start_line:end_line]).strip()
            
            if len(section_content) > 150:  # Only substantial sections
                sections.append({
                    'title': header,
                    'content': section_content,
                    'start_line': start_line,
                    'end_line': end_line,
                    'type': 'header_based'
                })
        
        return sections

    def _is_likely_header(self, line: str, all_lines: List[str], line_index: int) -> bool:
        """Determine if a line is likely a header based on context"""
        
        # Check if line is standalone (empty lines before/after)
        has_space_before = (line_index == 0 or 
                           all_lines[line_index - 1].strip() == '' or
                           len(all_lines[line_index - 1].strip()) < 10)
        
        has_space_after = (line_index == len(all_lines) - 1 or 
                          all_lines[line_index + 1].strip() == '' or
                          len(all_lines[line_index + 1].strip()) < 10)
        
        # Headers usually don't end with periods
        no_period = not line.endswith('.')
        
        # Headers are usually not too long
        reasonable_length = 5 <= len(line) <= 60
        
        # Contains relevant keywords
        has_keywords = any(keyword in line.lower() for keyword in 
                          ['hajj', 'umrah', 'mosque', 'prayer', 'guide', 'day', 'how', 'what', 'where'])
        
        return (has_space_before or has_space_after) and reasonable_length and (no_period or has_keywords)

    def _detect_topic_boundaries(self, text: str, guide_type: str) -> List[Dict[str, Any]]:
        """Detect topic boundaries using Islamic context and semantic similarity"""
        
        sections = []
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        
        if len(paragraphs) < 3:
            return sections
        
        # Group paragraphs by topic similarity
        if self.embedder is not None:
            try:
                embeddings = self.embedder.encode(paragraphs)
                
                # Find topic boundaries using similarity thresholds
                current_section = [paragraphs[0]]
                
                for i in range(1, len(paragraphs)):
                    similarity = np.dot(embeddings[i-1], embeddings[i])
                    
                    if similarity < 0.7:  # Topic boundary
                        if len('\n\n'.join(current_section)) > 200:
                            sections.append({
                                'title': f"Topic {len(sections) + 1}",
                                'content': '\n\n'.join(current_section),
                                'type': 'topic_boundary'
                            })
                        current_section = [paragraphs[i]]
                    else:
                        current_section.append(paragraphs[i])
                
                # Add final section
                if current_section and len('\n\n'.join(current_section)) > 200:
                    sections.append({
                        'title': f"Topic {len(sections) + 1}",
                        'content': '\n\n'.join(current_section),
                        'type': 'topic_boundary'
                    })
                    
            except Exception as e:
                logger.warning(f"Topic boundary detection failed: {e}")
        
        return sections

    def _fallback_length_split(self, text: str, guide_type: str) -> List[Dict[str, Any]]:
        """Fallback: split text into sections by length"""
        
        sections = []
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
        
        current_section = []
        current_length = 0
        target_length = 1500  # Target section length
        
        for paragraph in paragraphs:
            if current_length + len(paragraph) > target_length and current_section:
                # Create section
                sections.append({
                    'title': f"{guide_type} - Part {len(sections) + 1}",
                    'content': '\n\n'.join(current_section),
                    'type': 'length_based'
                })
                current_section = [paragraph]
                current_length = len(paragraph)
            else:
                current_section.append(paragraph)
                current_length += len(paragraph)
        
        # Add final section
        if current_section:
            sections.append({
                'title': f"{guide_type} - Part {len(sections) + 1}",
                'content': '\n\n'.join(current_section),
                'type': 'length_based'
            })
        
        return sections

    def _create_chunks_from_section(self, section: Dict[str, Any], 
                                   guide_type: str, section_index: int) -> List[Document]:
        """Create optimized chunks from a section"""
        
        chunks = []
        section_content = section['content']
        section_title = section['title']
        
        # Determine chunk parameters based on content analysis
        content_analysis = self._analyze_section_content(section_content, guide_type)
        chunk_params = self._get_chunk_parameters(content_analysis)
        
        # Create text splitter
        if LANGCHAIN_AVAILABLE:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_params['size'],
                chunk_overlap=chunk_params['overlap'],
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            text_chunks = splitter.split_text(section_content)
        else:
            # Simple fallback splitting
            text_chunks = self._simple_text_split(section_content, chunk_params['size'])
        
        # Create Document objects with rich metadata
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < 50:  # Skip very small chunks
                continue
            
            chunk_analysis = self._analyze_chunk_content(chunk_text, guide_type)
            
            metadata = self._create_enhanced_metadata(
                chunk_text=chunk_text,
                section_title=section_title,
                guide_type=guide_type,
                section_index=section_index,
                chunk_index=i,
                content_analysis=content_analysis,
                chunk_analysis=chunk_analysis
            )
            
            doc = Document(
                page_content=chunk_text,
                metadata=metadata
            )
            
            chunks.append(doc)
        
        return chunks

    def _analyze_section_content(self, content: str, guide_type: str) -> Dict[str, Any]:
        """Analyze section content for Islamic context and urgency"""
        
        content_lower = content.lower()
        
        # Detect Islamic concepts
        islamic_concepts = []
        for category, concepts in self.islamic_context.items():
            if isinstance(concepts, dict):
                for concept, keywords in concepts.items():
                    if concept in content_lower or any(kw in content_lower for kw in keywords):
                        islamic_concepts.append(concept)
            else:
                for concept in concepts:
                    if concept in content_lower:
                        islamic_concepts.append(concept)
        
        # Determine content type
        content_type = self._determine_content_type(content_lower)
        
        # Assess urgency
        urgency_level = self._assess_urgency_level(content_lower)
        
        # Detect instructions
        has_instructions = any(pattern in content_lower for pattern in 
                              ['step', 'how to', 'must', 'should', 'perform', 'follow'])
        
        return {
            'islamic_concepts': islamic_concepts,
            'content_type': content_type,
            'urgency_level': urgency_level,
            'has_instructions': has_instructions,
            'word_count': len(content.split()),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()])
        }

    def _determine_content_type(self, content: str) -> str:
        """Determine the type of content"""
        
        if any(word in content for word in ['how to', 'steps', 'perform', 'follow']):
            return 'instruction'
        elif any(word in content for word in ['where', 'location', 'address', 'direction']):
            return 'navigation'
        elif any(word in content for word in ['warning', 'danger', 'prohibited', 'avoid']):
            return 'warning'
        elif any(word in content for word in ['history', 'background', 'story', 'virtue']):
            return 'background'
        elif any(word in content for word in ['prayer', 'tawaf', 'sai', 'ihram']):
            return 'ritual'
        else:
            return 'general'

    def _assess_urgency_level(self, content: str) -> int:
        """Assess urgency level (1-5)"""
        
        if any(word in content for word in ['emergency', 'urgent', 'immediately', 'critical']):
            return 5
        elif any(word in content for word in ['important', 'must', 'required', 'mandatory']):
            return 4
        elif any(word in content for word in ['should', 'recommended', 'advised']):
            return 3
        elif any(word in content for word in ['may', 'can', 'optional']):
            return 2
        else:
            return 1

    def _get_chunk_parameters(self, content_analysis: Dict[str, Any]) -> Dict[str, int]:
        """Get optimal chunk parameters based on content analysis"""
        
        content_type = content_analysis['content_type']
        urgency = content_analysis['urgency_level']
        has_instructions = content_analysis['has_instructions']
        
        if content_type == 'instruction' or urgency >= 4 or has_instructions:
            return {'size': 500, 'overlap': 100}  # Smaller for critical content
        elif content_type in ['navigation', 'warning']:
            return {'size': 700, 'overlap': 140}
        elif content_type == 'background':
            return {'size': 1200, 'overlap': 200}  # Larger for context
        else:
            return {'size': 800, 'overlap': 160}  # Default

    def _analyze_chunk_content(self, chunk_text: str, guide_type: str) -> Dict[str, Any]:
        """Analyze individual chunk content"""
        
        chunk_lower = chunk_text.lower()
        
        # Find relevant Islamic terms
        relevant_terms = []
        for category, items in self.islamic_context.items():
            if isinstance(items, dict):
                for term, keywords in items.items():
                    if term in chunk_lower or any(kw in chunk_lower for kw in keywords):
                        relevant_terms.append(term)
        
        # Extract locations
        locations = []
        for location in self.islamic_context['locations'].keys():
            if location in chunk_lower:
                locations.append(location)
        
        # Extract voice-friendly keywords
        voice_keywords = []
        question_words = ['how', 'what', 'where', 'when', 'why', 'which', 'who']
        for word in question_words:
            if word in chunk_lower:
                voice_keywords.append(word)
        
        return {
            'islamic_terms': relevant_terms,
            'locations': locations,
            'voice_keywords': voice_keywords,
            'has_numbers': bool(re.search(r'\d+', chunk_text)),
            'has_quotes': '"' in chunk_text or '"' in chunk_text,
            'sentence_count': len([s for s in chunk_text.split('.') if s.strip()])
        }

    def _create_enhanced_metadata(self, chunk_text: str, section_title: str, 
                                 guide_type: str, section_index: int, chunk_index: int,
                                 content_analysis: Dict[str, Any], 
                                 chunk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive metadata for optimal retrieval"""
        
        # Generate unique chunk ID
        text_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
        chunk_id = f"{guide_type.lower().replace(' ', '_')}_{section_index}_{chunk_index}_{text_hash}"
        
        metadata = {
            # Core identification
            'chunk_id': chunk_id,
            'guide_type': guide_type,
            'section_title': section_title,
            'section_index': section_index,
            'chunk_index': chunk_index,
            
            # Content classification
            'content_type': content_analysis['content_type'],
            'urgency_level': content_analysis['urgency_level'],
            'has_instructions': content_analysis['has_instructions'],
            
            # Islamic context
            'islamic_terms': chunk_analysis['islamic_terms'],
            'location_contexts': chunk_analysis['locations'],
            'ritual_category': self._get_ritual_category(chunk_text, guide_type),
            
            # Retrieval optimization
            'voice_keywords': chunk_analysis['voice_keywords'],
            'difficulty_level': self._calculate_difficulty(chunk_analysis),
            
            # Quality metrics
            'chunk_length': len(chunk_text),
            'word_count': len(chunk_text.split()),
            'sentence_count': chunk_analysis['sentence_count'],
            'has_numbers': chunk_analysis['has_numbers'],
            'has_quotes': chunk_analysis['has_quotes']
        }
        
        return metadata

    def _get_ritual_category(self, text: str, guide_type: str) -> str:
        """Determine the main ritual category"""
        
        text_lower = text.lower()
        
        if 'umrah' in text_lower and 'hajj' not in text_lower:
            return 'umrah'
        elif 'hajj' in text_lower:
            return 'hajj'
        elif any(term in text_lower for term in ['prepare', 'visa', 'book']):
            return 'preparation'
        elif any(term in text_lower for term in ['return', 'depart', 'farewell']):
            return 'departure'
        else:
            return 'general'

    def _calculate_difficulty(self, chunk_analysis: Dict[str, Any]) -> int:
        """Calculate content difficulty (1-5)"""
        
        factors = 0
        
        # More Islamic terms = higher difficulty
        if len(chunk_analysis['islamic_terms']) > 3:
            factors += 2
        elif len(chunk_analysis['islamic_terms']) > 1:
            factors += 1
        
        # Complex sentences = higher difficulty
        if chunk_analysis['sentence_count'] > 10:
            factors += 2
        elif chunk_analysis['sentence_count'] > 5:
            factors += 1
        
        # Numbers often indicate instructions = moderate difficulty
        if chunk_analysis['has_numbers']:
            factors += 1
        
        return min(5, max(1, factors))

    def _simple_text_split(self, text: str, chunk_size: int) -> List[str]:
        """Simple text splitting when langchain is not available"""
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                current_chunk += sentence + ". "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

def main():
    """Test the fixed chunking system"""
    
    print("🔧 Testing Fixed Advanced Islamic Chunker")
    print("=" * 50)
    
    # Initialize chunker
    chunker = AdvancedIslamicChunker()
    
    # Test with sample text
    sample_text = """
    Financial Awareness
    A Guide for Pilgrims

    Contents
    Click on a title to get to the desired page

    Financial Transactions Before Your Arrival
    Purchasing Hajj and Umrah Services
    The Approved Currency in Saudi Arabia
    
    Financial Transactions Before Your Arrival
    
    The concerned authorities were keen to design a financial path where all fees
    and service costs are paid in advance, including visa fees, accommodation,
    transportation costs, and round-trip tickets.

    Why Pay for the Trip Before Coming?
    
    You won't have to carry a lot of cash when you come.
    You won't be faced with financial fraud attempts.
    You aren't preoccupied with anything that comes in the way of your devotion to worship.
    """
    
    # Test guide info extraction
    guide_info = chunker.extract_guide_info("EN-102.txt")
    print(f"Guide Info: {guide_info}")
    
    # Test chunking
    chunks = chunker.create_hierarchical_chunks(sample_text, guide_info['guide_type'])
    
    print(f"\n📊 Results:")
    print(f"Generated {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n--- Chunk {i+1} ---")
        print(f"Content: {chunk.page_content[:100]}...")
        print(f"Metadata: {chunk.metadata}")

if __name__ == "__main__":
    main()