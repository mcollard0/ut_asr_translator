"""
Multilingual Automatic Speech Recognition (ASR) module
Uses Whisper models with automatic language detection support

This module implements the "langauto" detection system for multilingual audio.
"""

import time
import warnings
from typing import Dict, Any, Optional, List
from transformers import pipeline
import librosa
import torch
from .config import get_hf_token, validate_token
from .langcodes import normalize_language_code, get_whisper_language_name, get_display_name, validate_language_code


class MultilingualASR:
    """Multilingual speech recognition using Whisper models with language detection"""
    
    def __init__( self, model_name: str, device: str = "cpu", hf_token: Optional[str] = None ):
        self.model_name = model_name;
        self.device = device;
        self.hf_token = hf_token or get_hf_token();
        self.pipeline = None;
        self._load_pipeline();
    
    def _load_pipeline( self ):
        """Initialize the ASR pipeline with error handling"""
        try:
            print( f"📥 Loading multilingual ASR model: {self.model_name}" );
            
            # Configure pipeline arguments;
            pipeline_kwargs = {
                "task": "automatic-speech-recognition",
                "model": self.model_name,
                "device": 0 if self.device == "cuda" else -1,
                "return_timestamps": False  # Can be enabled per-request;
            };
            
            # Add token if available and valid;
            if self.hf_token and validate_token( self.hf_token ):
                pipeline_kwargs["token"] = self.hf_token;
            elif self.hf_token:
                print( f"⚠️  Invalid HF token format, proceeding without authentication" );
            
            # Create pipeline;
            self.pipeline = pipeline( **pipeline_kwargs );
            
            print( f"✅ Multilingual ASR model loaded successfully on {self.device.upper()}" );
            
        except Exception as e:
            print( f"❌ Failed to load ASR model {self.model_name}: {e}" );
            raise RuntimeError( f"ASR model loading failed: {e}" );
    
    def transcribe( 
        self, 
        audio_path: str, 
        source_language: str = "auto",
        task: str = "transcribe",
        return_segments: bool = False,
        return_language_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio with optional language detection
        
        Args:
            audio_path: Path to audio file
            source_language: Language hint ('auto' for detection, ISO code for forced)
            task: 'transcribe' or 'translate' (translate to English)
            return_segments: Include word-level timestamps
            return_language_detection: Include language detection results
        
        Returns:
            Dictionary with transcription results and metadata
        """
        if not self.pipeline:
            raise RuntimeError( "ASR pipeline not initialized" );
        
        start_time = time.time();
        
        try:
            print( f"🎤 Processing audio: {audio_path}" );
            
            # Load audio using librosa for consistent preprocessing;
            audio, sr = librosa.load( audio_path, sr=16000, mono=True );
            audio_duration = len( audio ) / sr;
            
            # Normalize source language;
            normalized_lang = normalize_language_code( source_language );
            if not normalized_lang:
                normalized_lang = "auto";
            
            # Configure generation parameters;
            generate_kwargs = {
                "task": task,
                "do_sample": False,  # Deterministic output;
                "return_timestamps": return_segments
            };
            
            # Set language parameter based on input;
            if normalized_lang == "auto":
                # Let Whisper auto-detect language;
                generate_kwargs["language"] = None;
                print( f"🔍 Auto-detecting language..." );
            else:
                # Force specific language;
                whisper_lang = get_whisper_language_name( normalized_lang );
                if whisper_lang:
                    generate_kwargs["language"] = whisper_lang;
                    print( f"🎯 Forced language: {get_display_name(normalized_lang)} ({whisper_lang})" );
                else:
                    print( f"⚠️  Language '{normalized_lang}' not supported by Whisper, auto-detecting instead" );
                    generate_kwargs["language"] = None;
                    normalized_lang = "auto";
            
            # Run transcription;
            result = self.pipeline( 
                audio,
                generate_kwargs=generate_kwargs,
                return_timestamps=return_segments
            );
            
            transcription_time = time.time() - start_time;
            
            # Extract transcription text;
            if isinstance( result, dict ):
                transcribed_text = result.get( "text", "" ).strip();
                chunks = result.get( "chunks", [] ) if return_segments else [];
            else:
                transcribed_text = str( result ).strip();
                chunks = [];
            
            # Detect language from pipeline output if auto-detection was used;
            detected_language = normalized_lang;
            language_confidence = None;
            
            if normalized_lang == "auto" and return_language_detection:
                # Try to extract language detection info from the pipeline;
                # Note: This depends on the specific Whisper implementation;
                try:
                    if hasattr( self.pipeline, 'model' ) and hasattr( self.pipeline.model, 'detect_language' ):
                        # Direct detection using Whisper model;
                        with torch.no_grad():
                            audio_tensor = torch.from_numpy( audio ).float();
                            if self.device == "cuda":
                                audio_tensor = audio_tensor.cuda();
                            
                            # Use Whisper's internal language detection;
                            language_tokens = self.pipeline.model.detect_language( 
                                self.pipeline.model.encode( audio_tensor ) 
                            );
                            
                            # Extract most probable language;
                            if isinstance( language_tokens, dict ):
                                detected_lang_full = max( language_tokens.items(), key=lambda x: x[1] );
                                detected_language = detected_lang_full[0];
                                language_confidence = detected_lang_full[1];
                                
                                # Map Whisper language names back to ISO codes;
                                lang_mapping = {
                                    'spanish': 'es', 'english': 'en', 'french': 'fr', 'german': 'de',
                                    'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'chinese': 'zh',
                                    'japanese': 'ja', 'korean': 'ko', 'arabic': 'ar', 'hindi': 'hi',
                                    'dutch': 'nl', 'swedish': 'sv', 'polish': 'pl', 'turkish': 'tr',
                                    'ukrainian': 'uk', 'vietnamese': 'vi', 'persian': 'fa', 'hebrew': 'he',
                                    'indonesian': 'id', 'thai': 'th', 'czech': 'cs', 'danish': 'da',
                                    'greek': 'el', 'finnish': 'fi', 'hungarian': 'hu', 'norwegian': 'no',
                                    'romanian': 'ro', 'slovak': 'sk', 'slovenian': 'sl', 'bulgarian': 'bg',
                                    'croatian': 'hr', 'estonian': 'et', 'latvian': 'lv', 'lithuanian': 'lt',
                                    'catalan': 'ca', 'basque': 'eu', 'galician': 'gl'
                                };
                                
                                detected_language = lang_mapping.get( detected_language, detected_language );
                        
                        print( f"🔍 Detected language: {get_display_name(detected_language)} ({detected_language})" );
                        if language_confidence:
                            print( f"   Confidence: {language_confidence:.3f}" );
                        
                except Exception as lang_detect_error:
                    print( f"⚠️  Language detection failed: {lang_detect_error}" );
                    detected_language = "unknown";
            
            # Fallback language detection using simple heuristics;
            if detected_language == "auto" or detected_language == "unknown":
                detected_language = self._simple_language_detection( transcribed_text );
            
            # Build result dictionary;
            result_dict = {
                "text": transcribed_text,
                "language": detected_language,
                "language_display_name": get_display_name( detected_language ),
                "language_confidence": language_confidence,
                "model_name": self.model_name,
                "device": self.device,
                "task": task,
                "transcription_time_seconds": round( transcription_time, 2 ),
                "audio_duration_seconds": round( audio_duration, 2 ),
                "success": True
            };
            
            # Add segments if requested;
            if return_segments and chunks:
                result_dict["segments"] = chunks;
            
            # Add language detection metadata;
            if return_language_detection:
                result_dict["language_detection"] = {
                    "requested": source_language,
                    "detected": detected_language,
                    "confidence": language_confidence,
                    "method": "whisper" if language_confidence else "heuristic"
                };
            
            print( f"📝 Transcription: '{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}'" );
            print( f"⏱️  Transcription time: {transcription_time:.2f}s for {audio_duration:.1f}s audio" );
            
            return result_dict;
            
        except Exception as e:
            error_time = time.time() - start_time;
            print( f"❌ Transcription failed after {error_time:.2f}s: {e}" );
            
            return {
                "text": "",
                "language": "unknown",
                "language_display_name": "Unknown",
                "language_confidence": None,
                "model_name": self.model_name,
                "device": self.device,
                "task": task,
                "transcription_time_seconds": round( error_time, 2 ),
                "audio_duration_seconds": 0.0,
                "success": False,
                "error": str( e )
            };
    
    def _simple_language_detection( self, text: str ) -> str:
        """
        Simple heuristic-based language detection as fallback
        
        Args:
            text: Transcribed text
            
        Returns:
            Detected language code (best guess)
        """
        if not text or len( text.strip() ) < 3:
            return "unknown";
        
        text_lower = text.lower();
        
        # Simple character-based heuristics;
        if any( char in text for char in ['ñ', 'á', 'é', 'í', 'ó', 'ú'] ):
            return "es";  # Spanish;
        elif any( char in text for char in ['à', 'è', 'ù', 'ç'] ):
            return "fr";  # French;
        elif any( char in text for char in ['ä', 'ö', 'ü', 'ß'] ):
            return "de";  # German;
        elif any( char in text for char in ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з'] ):
            return "ru";  # Russian;
        elif any( char in text for char in ['中', '的', '一', '是', '了', '人', '我', '在'] ):
            return "zh";  # Chinese;
        elif any( char in text for char in ['の', 'に', 'は', 'を', 'が', 'と', 'で', 'た'] ):
            return "ja";  # Japanese;
        elif any( char in text for char in ['한', '이', '그', '을', '를', '에', '는', '가'] ):
            return "ko";  # Korean;
        elif any( char in text for char in ['ا', 'ل', 'م', 'ن', 'ت', 'ر', 'ي', 'و'] ):
            return "ar";  # Arabic;
        
        # Word-based heuristics for Latin scripts;
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'me', 'una', 'para', 'ni', 'si', 'mi', 'todo', 'pero'];
        french_words = ['le', 'de', 'et', 'un', 'à', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand', 'il', 'me', 'ses', 'te', 'comme'];
        english_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from'];
        german_words = ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass'];
        
        word_lists = [
            (spanish_words, 'es'),
            (french_words, 'fr'),
            (english_words, 'en'),
            (german_words, 'de')
        ];
        
        words_in_text = text_lower.split();
        best_match_count = 0;
        best_language = "en";  # Default to English;
        
        for word_list, lang_code in word_lists:
            match_count = sum( 1 for word in words_in_text if word in word_list );
            if match_count > best_match_count:
                best_match_count = match_count;
                best_language = lang_code;
        
        return best_language if best_match_count > 0 else "unknown";


def transcribe_audio( 
    audio_path: str, 
    model_name: str, 
    device: str = "cpu", 
    hf_token: Optional[str] = None,
    source_language: str = "auto",
    task: str = "transcribe"
) -> Dict[str, Any]:
    """
    Convenience function for audio transcription with language detection
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model name (e.g., 'openai/whisper-small')
        device: Device to use ('cpu', 'cuda', 'mps')
        hf_token: Hugging Face token for model access
        source_language: Language hint ('auto' for detection)
        task: 'transcribe' or 'translate'
    
    Returns:
        Transcription results dictionary
    """
    asr = MultilingualASR( model_name, device, hf_token );
    return asr.transcribe( audio_path, source_language, task );


def get_model_info( model_name: str ) -> Dict[str, Any]:
    """Get information about a Whisper model"""
    model_info = {
        "name": model_name,
        "is_multilingual": "whisper" in model_name.lower(),
        "supports_languages": "100+" if "whisper" in model_name.lower() else "unknown",
        "estimated_size_mb": "unknown"
    };
    
    # Add size estimates for common models;
    size_estimates = {
        "openai/whisper-tiny": 39,
        "openai/whisper-base": 74,
        "openai/whisper-small": 244,
        "openai/whisper-medium": 769,
        "openai/whisper-large": 1550,
        "openai/whisper-large-v2": 1550,
        "openai/whisper-large-v3": 1550
    };
    
    if model_name in size_estimates:
        model_info["estimated_size_mb"] = size_estimates[model_name];
    
    # Add performance characteristics;
    if "tiny" in model_name.lower():
        model_info["performance"] = "very fast, lower accuracy";
    elif "base" in model_name.lower():
        model_info["performance"] = "fast, good accuracy";
    elif "small" in model_name.lower():
        model_info["performance"] = "balanced speed/accuracy";
    elif "medium" in model_name.lower():
        model_info["performance"] = "slower, high accuracy";
    elif "large" in model_name.lower():
        model_info["performance"] = "slowest, highest accuracy";
    else:
        model_info["performance"] = "unknown";
    
    return model_info;


# Backwards compatibility;
# Import alias for legacy code;
SpanishASR = MultilingualASR;

def transcribe_spanish( 
    audio_path: str, 
    model_name: str, 
    device: str = "cpu", 
    hf_token: Optional[str] = None 
) -> Dict[str, Any]:
    """
    Legacy function for Spanish transcription
    
    DEPRECATED: Use MultilingualASR or transcribe_audio instead
    """
    warnings.warn( 
        "transcribe_spanish is deprecated. Use MultilingualASR.transcribe() or transcribe_audio() instead.",
        DeprecationWarning,
        stacklevel=2
    );
    
    # Use multilingual ASR with Spanish language hint;
    asr = MultilingualASR( model_name, device, hf_token );
    result = asr.transcribe( audio_path, source_language="es", task="transcribe" );
    
    # Map new format to old format for backwards compatibility;
    legacy_result = {
        "text": result["text"],
        "language": "es",  # Legacy always returned 'es';
        "model_name": result["model_name"],
        "device": result["device"],
        "transcription_time_seconds": result["transcription_time_seconds"],
        "audio_duration_seconds": result["audio_duration_seconds"],
        "success": result["success"]
    };
    
    if not result["success"]:
        legacy_result["error"] = result.get( "error", "Unknown error" );
    
    return legacy_result;