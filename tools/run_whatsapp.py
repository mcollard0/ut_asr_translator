#!/usr/bin/env python3
"""
Simple wrapper to run WhatsApp translation with default settings
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wa_asr_translator.cli import main

if __name__ == "__main__":
    # Set default HF token if not already set
    if not os.getenv("HUGGING_FACE_API_KEY"):
        os.environ["HUGGING_FACE_API_KEY"] = "{{HUGGING_FACE_API_KEY}}"
    
    main()