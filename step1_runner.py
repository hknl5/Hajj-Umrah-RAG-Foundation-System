#!/usr/bin/env python3
"""
Quick Start Runner for Step 1 Foundation System

This script provides an easy way to run the complete Step 1 system
with different configuration options.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    
    required_packages = [
        'sentence_transformers',
        'faiss',
        'langchain',
        'ollama',
        'numpy',
        'transformers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'faiss':
                # Try faiss-cpu
                try:
                    import faiss
                except ImportError:
                    missing_packages.append('faiss-cpu')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is running and has required model"""
    
    try:
        import ollama
        
        # Test connection and list models
        models_response = ollama.list()
        
        # Extract model names from the response
        model_names = []
        
        if hasattr(models_response, 'models'):
            for model in models_response.models:
                if hasattr(model, 'model'):
                    model_names.append(model.model)
                elif hasattr(model, 'name'):
                    model_names.append(model.name)
                else:
                    model_names.append(str(model))
        
        print(f"Found {len(model_names)} models in Ollama: {model_names}")
        
        # Check for qwen2.5:7b model
        qwen_models = [name for name in model_names if 'qwen2.5' in name.lower() and '7b' in name.lower()]
        
        if not qwen_models:
            print("Qwen2.5:7b model not found in Ollama")
            print("Available models:", model_names)
            print("Please run: ollama pull qwen2.5:7b")
            return False
        
        print(f"Ollama connection verified. Found model: {qwen_models[0]}")
        return True
        
    except Exception as e:
        print(f"Ollama check failed: {e}")
        print("Please ensure Ollama is running and has qwen2.5:7b model")
        return False

def setup_data_directory(data_dir: str = "./data"):
    """Setup data directory and check for input files"""
    
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Check for input files
    expected_files = [
        "EN-102.txt", "EN-103.txt", "EN-104.txt", "EN-105.txt", "EN-106.txt",
        "EN-108.txt", "EN-109.txt", "EN-110.txt", "EN-111.txt", "EN-112.txt", 
        "EN-113.txt", "EN-114.txt", "EN-115.txt", "EN-116.txt"
    ]
    
    found_files = []
    for filename in expected_files:
        # Check in data directory
        if (data_path / filename).exists():
            found_files.append(data_path / filename)
        # Check in current directory
        elif Path(filename).exists():
            found_files.append(Path(filename))
    
    if not found_files:
        print(f"No input files found!")
        print(f"Please place EN-*.txt files in:")
        print(f"  - {data_path}")
        print(f"  - Current directory")
        return False, []
    
    print(f"Found {len(found_files)} input files")
    return True, found_files

def run_step1_system(args):
    """Run the Step 1 foundation system"""
    
    # Import here to avoid issues if requirements not met
    from step1_integration import Step1FoundationSystem
    
    print("="*60)
    print("HAJJ/UMRAH RAG SYSTEM - STEP 1 FOUNDATION")
    print("="*60)
    print("Components:")
    print("- Advanced Islamic content chunking")
    print("- Multi-model embeddings (E5-large-v2 + BGE-M3)")
    print("- FAISS HNSW indexing for <100ms retrieval")
    print("- Comprehensive evaluation system")
    print("="*60)
    
    # Initialize system
    system = Step1FoundationSystem(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run complete foundation
    system.run_complete_foundation()
    
    print("\n" + "="*60)
    print("STEP 1 FOUNDATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return system

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(
        description="Step 1 Foundation System for Hajj/Umrah RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step1_runner.py                          # Run with defaults
  python step1_runner.py --data-dir ./my_data    # Custom data directory
  python step1_runner.py --output-dir ./results  # Custom output directory
  python step1_runner.py --skip-checks           # Skip requirement checks
        """
    )
    
    parser.add_argument(
        '--data-dir', 
        default='./data',
        help='Directory containing EN-*.txt files (default: ./data)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./foundation_output', 
        help='Directory for system outputs (default: ./foundation_output)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip requirement and setup checks'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run checks unless skipped
    if not args.skip_checks:
        print("Checking system requirements...")
        
        if not check_requirements():
            print("Requirement check failed. Use --skip-checks to bypass.")
            sys.exit(1)
        
        if not check_ollama():
            print("Ollama check failed. Use --skip-checks to bypass.")
            sys.exit(1)
        
        print("Checking data files...")
        files_ok, found_files = setup_data_directory(args.data_dir)
        if not files_ok:
            print("Data file check failed.")
            sys.exit(1)
        
        print("All checks passed!")
    
    # Run the system
    try:
        system = run_step1_system(args)
        
        print(f"\nSystem outputs saved to: {args.output_dir}")
        print(f"Ready for Step 2: Enhanced Retrieval")
        
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nSystem failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()