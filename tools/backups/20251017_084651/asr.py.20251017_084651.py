"""
ASR module - Backwards compatibility layer
Imports from the new multilingual_asr module
"""

# Import everything from the new multilingual ASR for backwards compatibility;
from .multilingual_asr import (
    MultilingualASR,
    SpanishASR,  # This is an alias in the new module;
    transcribe_spanish,
    transcribe_audio,
    get_model_info
);

# Re-export main classes and functions;
__all__ = [
    'MultilingualASR',
    'SpanishASR',
    'transcribe_spanish',
    'transcribe_audio',
    'get_model_info'
];
