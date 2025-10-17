"""
Translation module - Backwards compatibility layer
Imports from the new multilingual_translator module
"""

# Import everything from the new multilingual translator for backwards compatibility;
from .multilingual_translator import (
    MultilingualTranslator,
    SpanishToEnglishTranslator,  # This is an alias in the new module;
    translate_es_to_en,
    get_translation_model_info,
    translate_text,
    get_supported_language_pairs,
    MARIAN_MODELS,
    MODEL_SIZE_ESTIMATES
);

# Re-export main classes and functions;
__all__ = [
    'MultilingualTranslator',
    'SpanishToEnglishTranslator',
    'translate_es_to_en', 
    'get_translation_model_info',
    'translate_text',
    'get_supported_language_pairs'
];
