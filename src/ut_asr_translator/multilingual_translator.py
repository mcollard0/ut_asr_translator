"""
Multilingual translation module with intelligent model selection
Supports MarianMT, M2M100, and NLLB models with automatic fallback

This module implements the "modelmap" system for optimal translation model selection.
"""

import time
import re
import warnings
from typing import Dict, Any, Optional, List, Tuple
from transformers import pipeline
from .config import get_hf_token, validate_token
from .langcodes import normalize_language_code, get_nllb_code, get_display_name, validate_language_code, LanguageCodeError


# MarianMT model mapping for common language pairs - the "modelmap";
MARIAN_MODELS: Dict[Tuple[str, str], str] = {
    # English to other languages;
    ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
    ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
    ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
    ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it',
    ('en', 'pt'): 'Helsinki-NLP/opus-mt-en-roa',  # Romance languages;
    ('en', 'ru'): 'Helsinki-NLP/opus-mt-en-ru',
    ('en', 'zh'): 'Helsinki-NLP/opus-mt-en-zh',
    ('en', 'ja'): 'Helsinki-NLP/opus-mt-en-jap',
    ('en', 'ko'): 'Helsinki-NLP/opus-mt-en-ko',
    ('en', 'ar'): 'Helsinki-NLP/opus-mt-en-ar',
    ('en', 'hi'): 'Helsinki-NLP/opus-mt-en-hi',
    ('en', 'tr'): 'Helsinki-NLP/opus-mt-en-tr',
    ('en', 'nl'): 'Helsinki-NLP/opus-mt-en-gmw',  # West Germanic;
    ('en', 'sv'): 'Helsinki-NLP/opus-mt-en-sv',
    ('en', 'pl'): 'Helsinki-NLP/opus-mt-en-pl',
    ('en', 'uk'): 'Helsinki-NLP/opus-mt-en-uk',
    ('en', 'vi'): 'Helsinki-NLP/opus-mt-en-vi',
    ('en', 'fa'): 'Helsinki-NLP/opus-mt-en-fa',
    ('en', 'he'): 'Helsinki-NLP/opus-mt-en-he',
    
    # Other languages to English;
    ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
    ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
    ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
    ('it', 'en'): 'Helsinki-NLP/opus-mt-it-en',
    ('pt', 'en'): 'Helsinki-NLP/opus-mt-roa-en',  # Romance to English;
    ('ru', 'en'): 'Helsinki-NLP/opus-mt-ru-en',
    ('zh', 'en'): 'Helsinki-NLP/opus-mt-zh-en',
    ('ja', 'en'): 'Helsinki-NLP/opus-mt-jap-en',
    ('ko', 'en'): 'Helsinki-NLP/opus-mt-ko-en',
    ('ar', 'en'): 'Helsinki-NLP/opus-mt-ar-en',
    ('hi', 'en'): 'Helsinki-NLP/opus-mt-hi-en',
    ('tr', 'en'): 'Helsinki-NLP/opus-mt-tr-en',
    ('nl', 'en'): 'Helsinki-NLP/opus-mt-gmw-en',  # West Germanic to English;
    ('sv', 'en'): 'Helsinki-NLP/opus-mt-sv-en',
    ('pl', 'en'): 'Helsinki-NLP/opus-mt-pl-en',
    ('uk', 'en'): 'Helsinki-NLP/opus-mt-uk-en',
    ('vi', 'en'): 'Helsinki-NLP/opus-mt-vi-en',
    ('fa', 'en'): 'Helsinki-NLP/opus-mt-fa-en',
    ('he', 'en'): 'Helsinki-NLP/opus-mt-he-en',
    
    # Some bidirectional pairs;
    ('es', 'fr'): 'Helsinki-NLP/opus-mt-es-fr',
    ('fr', 'es'): 'Helsinki-NLP/opus-mt-fr-es',
    ('de', 'fr'): 'Helsinki-NLP/opus-mt-de-fr',
    ('fr', 'de'): 'Helsinki-NLP/opus-mt-fr-de',
    ('es', 'pt'): 'Helsinki-NLP/opus-mt-es-pt',
    ('pt', 'es'): 'Helsinki-NLP/opus-mt-pt-es',
};

# Model size estimates in MB;
MODEL_SIZE_ESTIMATES: Dict[str, int] = {
    # MarianMT models (generally compact);
    'marian': 300,
    
    # M2M100 models;
    'facebook/m2m100_418M': 418,
    'facebook/m2m100_1.2B': 1200,
    
    # NLLB models;
    'facebook/nllb-200-distilled-600M': 600,
    'facebook/nllb-200-1.3B': 1300,
    'facebook/nllb-200-3.3B': 3300,
};

# Default model preferences in order of priority;
DEFAULT_MODEL_PREFERENCE = [ 'marian', 'm2m100', 'nllb' ];


class MultilingualTranslator:
    """Multilingual translation with intelligent model selection"""
    
    def __init__(
        self,
        source_lang: str = 'auto',
        target_lang: str = 'en', 
        device: str = 'cpu',
        hf_token: Optional[str] = None,
        model_preference: Optional[List[str]] = None,
        max_model_size_mb: Optional[int] = None
    ):
        self.source_lang = normalize_language_code( source_lang ) or 'auto';
        self.target_lang = normalize_language_code( target_lang ) or 'en';
        self.device = device;
        self.hf_token = hf_token or get_hf_token();
        self.model_preference = model_preference or DEFAULT_MODEL_PREFERENCE.copy();
        self.max_model_size_mb = max_model_size_mb;
        
        self.pipeline = None;
        self.selected_model = None;
        self.model_info = {};
        
        # Validate target language at initialization;
        if not validate_language_code( self.target_lang )[0]:
            raise LanguageCodeError( f"Invalid target language: {target_lang}" );
    
    def _get_best_model( self, source_lang: str, target_lang: str ) -> Tuple[str, Dict[str, Any]]:
        """
        Select the best translation model for a language pair
        
        Args:
            source_lang: Source language ISO code
            target_lang: Target language ISO code
            
        Returns:
            Tuple of (model_name, model_info)
        """
        # Check for MarianMT models first;
        if 'marian' in self.model_preference:
            marian_model = MARIAN_MODELS.get( (source_lang, target_lang) );
            if marian_model:
                return marian_model, {
                    'type': 'marian',
                    'name': marian_model,
                    'estimated_size_mb': MODEL_SIZE_ESTIMATES['marian'],
                    'description': f"MarianMT {get_display_name(source_lang)} → {get_display_name(target_lang)}",
                    'performance': 'fast',
                    'quality': 'high for this pair'
                };
        
        # Try M2M100 for broader coverage;
        if 'm2m100' in self.model_preference:
            m2m100_model = 'facebook/m2m100_418M';  # Default to smaller variant;
            if self.max_model_size_mb is None or MODEL_SIZE_ESTIMATES[m2m100_model] <= self.max_model_size_mb:
                return m2m100_model, {
                    'type': 'm2m100',
                    'name': m2m100_model,
                    'estimated_size_mb': MODEL_SIZE_ESTIMATES[m2m100_model],
                    'description': f"M2M100 multilingual translation",
                    'performance': 'medium',
                    'quality': 'good for most pairs',
                    'languages_supported': '100+ languages'
                };
        
        # Fallback to NLLB for maximum coverage;
        if 'nllb' in self.model_preference:
            nllb_model = 'facebook/nllb-200-distilled-600M';  # Default to distilled version;
            
            # Check if we have NLLB codes for both languages;
            src_nllb = get_nllb_code( source_lang );
            tgt_nllb = get_nllb_code( target_lang );
            
            if src_nllb and tgt_nllb:
                if self.max_model_size_mb is None or MODEL_SIZE_ESTIMATES[nllb_model] <= self.max_model_size_mb:
                    return nllb_model, {
                        'type': 'nllb',
                        'name': nllb_model,
                        'estimated_size_mb': MODEL_SIZE_ESTIMATES[nllb_model],
                        'description': f"NLLB-200 No Language Left Behind",
                        'performance': 'slower but comprehensive',
                        'quality': 'very good for 200+ languages',
                        'languages_supported': '200+ languages',
                        'src_nllb_code': src_nllb,
                        'tgt_nllb_code': tgt_nllb
                    };
        
        # No suitable model found;
        raise ValueError( 
            f"No translation model available for {get_display_name(source_lang)} → {get_display_name(target_lang)}. "
            f"Supported models: {', '.join(self.model_preference)}"
        );
    
    def _load_pipeline( self, source_lang: str ):
        """Load translation pipeline for specific source language"""
        try:
            if source_lang == self.source_lang and self.pipeline:
                return;  # Already loaded for this language pair;
                
            model_name, model_info = self._get_best_model( source_lang, self.target_lang );
            
            print( f"📥 Loading translation model: {model_name}" );
            if model_info.get( 'estimated_size_mb', 0 ) > 1000:
                print( f"⚠️  Large model ({model_info['estimated_size_mb']}MB) - this may take some time..." );
            
            # Configure pipeline arguments;
            pipeline_kwargs = {
                'task': 'translation',
                'model': model_name,
                'device': 0 if self.device == 'cuda' else -1,
            };
            
            # Add token if available;
            if self.hf_token and validate_token( self.hf_token ):
                pipeline_kwargs['token'] = self.hf_token;
            
            # Special handling for different model types;
            if model_info['type'] == 'nllb':
                # NLLB requires source and target language codes;
                pipeline_kwargs['src_lang'] = model_info['src_nllb_code'];
                pipeline_kwargs['tgt_lang'] = model_info['tgt_nllb_code'];
            
            # Create pipeline;
            self.pipeline = pipeline( **pipeline_kwargs );
            self.selected_model = model_name;
            self.model_info = model_info;
            self.source_lang = source_lang;  # Update current source language;
            
            print( f"✅ Translation model loaded: {model_info['description']}" );
            print( f"   Performance: {model_info['performance']}, Quality: {model_info['quality']}" );
            
        except Exception as e:
            print( f"❌ Failed to load translation model: {e}" );
            raise RuntimeError( f"Translation model loading failed: {e}" );
    
    def translate( self, text: str, source_lang: Optional[str] = None ) -> Dict[str, Any]:
        """
        Translate text from source to target language
        
        Args:
            text: Text to translate
            source_lang: Override source language (uses class default if None)
            
        Returns:
            Translation result dictionary
        """
        start_time = time.time();
        
        try:
            # Determine source language;
            if source_lang:
                src_lang = normalize_language_code( source_lang );
                if not src_lang or src_lang == 'auto':
                    raise ValueError( f"Invalid or auto-detect source language: {source_lang}" );
            else:
                src_lang = self.source_lang;
                if src_lang == 'auto':
                    raise ValueError( "Source language must be specified (cannot be 'auto' for translation)" );
            
            # Check if translation is needed;
            if src_lang == self.target_lang:
                return {
                    'input_text': text,
                    'translated_text': text,
                    'source_language': src_lang,
                    'target_language': self.target_lang,
                    'model_name': 'none (same language)',
                    'translation_time_seconds': 0.0,
                    'skipped': True,
                    'success': True,
                    'note': f"Source and target languages are the same ({get_display_name(src_lang)})"
                };
            
            # Load appropriate model;
            self._load_pipeline( src_lang );
            
            if not text.strip():
                return self._empty_result( text, src_lang, "Empty input text" );
            
            print( f"🌍 Translating {get_display_name(src_lang)} → {get_display_name(self.target_lang)}: '{text[:50]}{'...' if len(text) > 50 else ''}'" );
            
            # Prepare text for translation;
            normalized_text = self._normalize_input( text );
            
            # Perform translation;
            if self.model_info['type'] == 'nllb':
                # NLLB handles language codes internally;
                result = self.pipeline( normalized_text );
            else:
                # MarianMT and M2M100;
                result = self.pipeline( normalized_text );
            
            translation_time = time.time() - start_time;
            
            # Extract translated text;
            if isinstance( result, list ) and len( result ) > 0:
                translated_text = result[0].get( 'translation_text', '' );
            elif isinstance( result, dict ):
                translated_text = result.get( 'translation_text', '' );
            else:
                translated_text = str( result );
            
            # Normalize output;
            normalized_output = self._normalize_output( translated_text );
            
            result_dict = {
                'input_text': text,
                'translated_text': normalized_output,
                'source_language': src_lang,
                'target_language': self.target_lang,
                'model_name': self.selected_model,
                'model_type': self.model_info['type'],
                'model_info': self.model_info,
                'translation_time_seconds': round( translation_time, 2 ),
                'input_length': len( text ),
                'output_length': len( normalized_output ),
                'success': True
            };
            
            print( f"🎯 Translation result: '{normalized_output[:100]}{'...' if len(normalized_output) > 100 else ''}'" );
            print( f"⏱️  Translation time: {translation_time:.2f}s using {self.model_info['type']}" );
            
            return result_dict;
            
        except Exception as e:
            error_time = time.time() - start_time;
            print( f"❌ Translation failed after {error_time:.2f}s: {e}" );
            return self._error_result( text, src_lang if 'src_lang' in locals() else 'unknown', str( e ), error_time );
    
    def _normalize_input( self, text: str ) -> str:
        """Normalize input text for translation"""
        if not text:
            return "";
        
        normalized = text.strip();
        normalized = re.sub( r'\s+', ' ', normalized );
        
        # Ensure text ends with punctuation for better translation;
        if normalized and normalized[-1] not in '.!?':
            normalized += '.';
        
        return normalized;
    
    def _normalize_output( self, text: str ) -> str:
        """Normalize output text"""
        if not text:
            return "";
        
        normalized = text.strip();
        
        # Capitalize first letter;
        if normalized:
            normalized = normalized[0].upper() + normalized[1:];
        
        normalized = re.sub( r'\s+', ' ', normalized );
        return normalized;
    
    def _empty_result( self, text: str, source_lang: str, error: str ) -> Dict[str, Any]:
        """Create result for empty/invalid input"""
        return {
            'input_text': text,
            'translated_text': '',
            'source_language': source_lang,
            'target_language': self.target_lang,
            'model_name': self.selected_model or 'none',
            'translation_time_seconds': 0.0,
            'success': False,
            'error': error
        };
    
    def _error_result( self, text: str, source_lang: str, error: str, error_time: float ) -> Dict[str, Any]:
        """Create result for translation error"""
        return {
            'input_text': text,
            'translated_text': '',
            'source_language': source_lang,
            'target_language': self.target_lang,
            'model_name': self.selected_model or 'none',
            'translation_time_seconds': round( error_time, 2 ),
            'success': False,
            'error': error
        };
    
    def get_model_info( self ) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if not self.model_info:
            return { 'status': 'No model loaded' };
        return self.model_info.copy();
    
    def supports_language_pair( self, source_lang: str, target_lang: str ) -> bool:
        """Check if a language pair is supported"""
        try:
            src = normalize_language_code( source_lang );
            tgt = normalize_language_code( target_lang );
            if not src or not tgt or src == 'auto':
                return False;
                
            # Try to find a model for this pair;
            _, _ = self._get_best_model( src, tgt );
            return True;
        except:
            return False;


# Convenience functions for backwards compatibility;
def translate_text( text: str, source_lang: str, target_lang: str = 'en', device: str = 'cpu', hf_token: Optional[str] = None ) -> Dict[str, Any]:
    """
    Convenience function for single translation
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        device: Device to use
        hf_token: HuggingFace token
        
    Returns:
        Translation result dictionary
    """
    translator = MultilingualTranslator( source_lang, target_lang, device, hf_token );
    return translator.translate( text );


def get_supported_language_pairs() -> List[Tuple[str, str]]:
    """Get list of all supported language pairs"""
    pairs = [];
    
    # Add all MarianMT pairs;
    pairs.extend( MARIAN_MODELS.keys() );
    
    # Add common M2M100 pairs (most combinations);
    m2m100_langs = [ 'ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'nl', 'pl', 'pt', 'ru', 'sv', 'tr', 'uk', 'vi', 'zh' ];
    for src in m2m100_langs:
        for tgt in m2m100_langs:
            if src != tgt and (src, tgt) not in pairs:
                pairs.append( (src, tgt) );
    
    return sorted( set( pairs ) );


# Backwards compatibility;
# Import alias for legacy code;
SpanishToEnglishTranslator = MultilingualTranslator;

def translate_es_to_en( spanish_text: str, model_name: str, device: str = 'cpu', hf_token: Optional[str] = None ) -> Dict[str, Any]:
    """
    Legacy function for Spanish to English translation
    
    DEPRECATED: Use MultilingualTranslator instead
    """
    warnings.warn( 
        "translate_es_to_en is deprecated. Use MultilingualTranslator or translate_text instead.",
        DeprecationWarning,
        stacklevel=2
    );
    
    # Map old model_name to new system;
    translator = MultilingualTranslator( 'es', 'en', device, hf_token );
    return translator.translate( spanish_text );


def get_translation_model_info( model_name: str ) -> Dict[str, Any]:
    """
    Legacy function for model info
    
    DEPRECATED: Use MultilingualTranslator.get_model_info() instead
    """
    warnings.warn(
        "get_translation_model_info is deprecated. Use MultilingualTranslator.get_model_info() instead.",
        DeprecationWarning,
        stacklevel=2
    );
    
    return {
        'name': model_name,
        'source_language': 'Spanish (legacy)',
        'target_language': 'English (legacy)', 
        'estimated_size_mb': MODEL_SIZE_ESTIMATES.get( model_name, 'unknown' ),
        'note': 'This function is deprecated - use MultilingualTranslator instead'
    };