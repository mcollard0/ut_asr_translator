"""
Machine Translation module for Spanish to English conversion
Uses MarianMT models via Hugging Face transformers
"""

import time
import re
from typing import Dict, Any, Optional
from transformers import pipeline
from .config import get_hf_token, validate_token


class SpanishToEnglishTranslator:
    """Spanish to English translation using MarianMT models"""
    
    def __init__( self, model_name: str, device: str = "cpu", hf_token: Optional[str] = None ):
        self.model_name = model_name;
        self.device = device;
        self.hf_token = hf_token or get_hf_token();
        self.pipeline = None;
        self._load_pipeline();
    
    def _load_pipeline( self ):
        """Initialize the translation pipeline with error handling"""
        try:
            print( f"📥 Loading translation model: {self.model_name}" );
            
            # Configure pipeline arguments;
            pipeline_kwargs = {
                "task": "translation",
                "model": self.model_name,
                "device": 0 if self.device == "cuda" else -1
            };
            
            # Add token if available and valid;
            if self.hf_token and validate_token( self.hf_token ):
                pipeline_kwargs["token"] = self.hf_token;
            elif self.hf_token:
                print( f"⚠️  Invalid HF token format, proceeding without authentication" );
            
            # Create pipeline;
            self.pipeline = pipeline( **pipeline_kwargs );
            
            print( f"✅ Translation model loaded successfully on {self.device.upper()}" );
            
        except Exception as e:
            print( f"❌ Failed to load translation model {self.model_name}: {e}" );
            raise RuntimeError( f"Translation model loading failed: {e}" );
    
    def _normalize_input( self, text: str ) -> str:
        """Normalize Spanish input text for better translation"""
        if not text:
            return "";
        
        # Basic cleanup;
        normalized = text.strip();
        
        # Remove excessive whitespace;
        normalized = re.sub( r'\s+', ' ', normalized );
        
        # Ensure text ends with punctuation for better translation;
        if normalized and normalized[-1] not in '.!?':
            normalized += '.';
        
        return normalized;
    
    def _normalize_output( self, text: str ) -> str:
        """Normalize English output text"""
        if not text:
            return "";
        
        # Basic cleanup;
        normalized = text.strip();
        
        # Capitalize first letter;
        if normalized:
            normalized = normalized[0].upper() + normalized[1:];
        
        # Remove excessive whitespace;
        normalized = re.sub( r'\s+', ' ', normalized );
        
        return normalized;
    
    def translate( self, spanish_text: str ) -> Dict[str, Any]:
        """
        Translate Spanish text to English
        
        Args:
            spanish_text: Spanish text to translate
        
        Returns:
            Dictionary with translation results and metadata
        """
        if not self.pipeline:
            raise RuntimeError( "Translation pipeline not initialized" );
        
        start_time = time.time();
        
        try:
            # Normalize input;
            normalized_input = self._normalize_input( spanish_text );
            
            if not normalized_input:
                return {
                    "input_text": spanish_text,
                    "translated_text": "",
                    "model_name": self.model_name,
                    "device": self.device,
                    "translation_time_seconds": 0.0,
                    "success": False,
                    "error": "Empty input text"
                };
            
            print( f"🌍 Translating: '{normalized_input}'" );
            
            # Run translation;
            result = self.pipeline( normalized_input );
            
            translation_time = time.time() - start_time;
            
            # Extract translated text;
            if isinstance( result, list ) and len( result ) > 0:
                translated_text = result[0].get( "translation_text", "" );
            elif isinstance( result, dict ):
                translated_text = result.get( "translation_text", "" );
            else:
                translated_text = str( result );
            
            # Normalize output;
            normalized_output = self._normalize_output( translated_text );
            
            # Build result dictionary;
            result_dict = {
                "input_text": spanish_text,
                "translated_text": normalized_output,
                "model_name": self.model_name,
                "device": self.device,
                "translation_time_seconds": round( translation_time, 2 ),
                "input_length": len( spanish_text ),
                "output_length": len( normalized_output ),
                "success": True
            };
            
            print( f"🎯 English translation: '{normalized_output}'" );
            print( f"⏱️  Translation time: {translation_time:.2f}s" );
            
            return result_dict;
            
        except Exception as e:
            error_time = time.time() - start_time;
            print( f"❌ Translation failed after {error_time:.2f}s: {e}" );
            
            return {
                "input_text": spanish_text,
                "translated_text": "",
                "model_name": self.model_name,
                "device": self.device,
                "translation_time_seconds": round( error_time, 2 ),
                "input_length": len( spanish_text ),
                "output_length": 0,
                "success": False,
                "error": str( e )
            };


def translate_es_to_en( spanish_text: str, model_name: str, device: str = "cpu", hf_token: Optional[str] = None ) -> Dict[str, Any]:
    """
    Convenience function for Spanish to English translation
    
    Args:
        spanish_text: Spanish text to translate
        model_name: Translation model name (e.g., 'Helsinki-NLP/opus-mt-es-en')
        device: Device to use ('cpu', 'cuda', 'mps')
        hf_token: Hugging Face token for model access
    
    Returns:
        Translation results dictionary
    """
    translator = SpanishToEnglishTranslator( model_name, device, hf_token );
    return translator.translate( spanish_text );


def get_translation_model_info( model_name: str ) -> Dict[str, Any]:
    """Get information about a translation model"""
    model_info = {
        "name": model_name,
        "source_language": "Spanish",
        "target_language": "English",
        "estimated_size_mb": "unknown"
    };
    
    # Add size estimates for common models;
    size_estimates = {
        "Helsinki-NLP/opus-mt-es-en": 300,  # MarianMT Spanish-English;
        "facebook/nllb-200-distilled-600M": 600,
        "facebook/nllb-200-1.3B": 1300,
        "facebook/nllb-200-3.3B": 3300
    };
    
    if model_name in size_estimates:
        model_info["estimated_size_mb"] = size_estimates[model_name];
    
    # Detect model type;
    if "opus-mt" in model_name.lower():
        model_info["model_type"] = "MarianMT";
        model_info["description"] = "Fast, efficient neural machine translation";
    elif "nllb" in model_name.lower():
        model_info["model_type"] = "NLLB";
        model_info["description"] = "No Language Left Behind - multilingual translation";
    else:
        model_info["model_type"] = "Unknown";
        model_info["description"] = "Unknown translation model type";
    
    return model_info;