"""
Automatic Speech Recognition (ASR) module for Spanish audio transcription
Uses Whisper models via Hugging Face transformers
"""

import time
from typing import Dict, Any, Optional
from transformers import pipeline
import librosa
import torch
from .config import get_hf_token, validate_token


class SpanishASR:
    """Spanish speech recognition using Whisper models"""
    
    def __init__( self, model_name: str, device: str = "cpu", hf_token: Optional[str] = None ):
        self.model_name = model_name;
        self.device = device;
        self.hf_token = hf_token or get_hf_token();
        self.pipeline = None;
        self._load_pipeline();
    
    def _load_pipeline( self ):
        """Initialize the ASR pipeline with error handling"""
        try:
            print( f"📥 Loading ASR model: {self.model_name}" );
            
            # Configure pipeline arguments;
            pipeline_kwargs = {
                "task": "automatic-speech-recognition",
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
            
            print( f"✅ ASR model loaded successfully on {self.device.upper()}" );
            
        except Exception as e:
            print( f"❌ Failed to load ASR model {self.model_name}: {e}" );
            raise RuntimeError( f"ASR model loading failed: {e}" );
    
    def transcribe( self, audio_path: str ) -> Dict[str, Any]:
        """
        Transcribe Spanish audio to text
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with transcription results and metadata
        """
        if not self.pipeline:
            raise RuntimeError( "ASR pipeline not initialized" );
        
        start_time = time.time();
        
        try:
            print( f"🎤 Transcribing audio: {audio_path}" );
            
            # Load audio using librosa for consistent preprocessing;
            audio, sr = librosa.load( audio_path, sr=16000, mono=True );
            
            # Configure generation parameters for Spanish;
            generate_kwargs = {
                "language": "spanish",  # Force Spanish language;
                "task": "transcribe",   # Transcription (not translation);
                "do_sample": False      # Deterministic output;
            };
            
            # Run transcription;
            result = self.pipeline( 
                audio,
                generate_kwargs=generate_kwargs,
                return_timestamps=False  # We don't need word-level timing for WhatsApp;
            );
            
            transcription_time = time.time() - start_time;
            
            # Extract and clean transcription text;
            transcribed_text = result.get( "text", "" ).strip();
            
            # Build result dictionary;
            result_dict = {
                "text": transcribed_text,
                "language": "es",
                "model_name": self.model_name,
                "device": self.device,
                "transcription_time_seconds": round( transcription_time, 2 ),
                "audio_duration_seconds": round( len( audio ) / sr, 2 ),
                "success": True
            };
            
            print( f"📝 Spanish transcription: '{transcribed_text}'" );
            print( f"⏱️  Transcription time: {transcription_time:.2f}s" );
            
            return result_dict;
            
        except Exception as e:
            error_time = time.time() - start_time;
            print( f"❌ Transcription failed after {error_time:.2f}s: {e}" );
            
            return {
                "text": "",
                "language": "es",
                "model_name": self.model_name,
                "device": self.device,
                "transcription_time_seconds": round( error_time, 2 ),
                "audio_duration_seconds": 0.0,
                "success": False,
                "error": str( e )
            };


def transcribe_spanish( audio_path: str, model_name: str, device: str = "cpu", hf_token: Optional[str] = None ) -> Dict[str, Any]:
    """
    Convenience function for Spanish transcription
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model name (e.g., 'openai/whisper-small')
        device: Device to use ('cpu', 'cuda', 'mps')
        hf_token: Hugging Face token for model access
    
    Returns:
        Transcription results dictionary
    """
    asr = SpanishASR( model_name, device, hf_token );
    return asr.transcribe( audio_path );


def get_model_info( model_name: str ) -> Dict[str, Any]:
    """Get information about a Whisper model"""
    model_info = {
        "name": model_name,
        "is_multilingual": "whisper" in model_name.lower(),
        "supports_spanish": True,  # All Whisper models support Spanish;
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
    
    return model_info;