"""
Configuration module for WhatsApp ASR Translator
Handles device detection, model defaults, and HF token management
"""

import os
import torch
from typing import Optional, Dict, Any


# Default model configurations
DEFAULT_ASR_MODEL = "openai/whisper-small"  # Good balance of speed/accuracy for Spanish;
DEFAULT_MT_MODEL = "Helsinki-NLP/opus-mt-es-en"  # Efficient Spanish-English translation;

# Compute warning thresholds
HIGH_COMPUTE_ASR_MODELS = [ "openai/whisper-large", "openai/whisper-large-v2", "openai/whisper-large-v3" ];
HIGH_COMPUTE_MT_MODELS = [ "facebook/nllb-200-3.3B", "facebook/nllb-200-1.3B" ];


def detect_device() -> str:
    """
    Auto-detect best available device: CUDA > MPS > CPU
    Returns device string for transformers pipelines
    """
    if torch.cuda.is_available():
        return "cuda";
    elif hasattr( torch.backends, "mps" ) and torch.backends.mps.is_available():
        return "mps";
    else:
        return "cpu";


def get_hf_token() -> Optional[str]:
    """
    Get Hugging Face token from environment variables
    Priority: HUGGING_FACE_API_KEY > HF_TOKEN
    """
    token = os.getenv( "HUGGING_FACE_API_KEY" );
    if not token:
        token = os.getenv( "HF_TOKEN" );
    return token;


def validate_token( token: Optional[str] ) -> bool:
    """Validate HF token format (basic check)"""
    if not token:
        return False;
    return token.startswith( "hf_" ) and len( token ) > 10;


def is_high_compute_model( model_name: str, model_type: str = "asr" ) -> bool:
    """
    Check if model requires high compute resources
    model_type: 'asr' or 'mt' (machine translation)
    """
    if model_type == "asr":
        return any( hc_model in model_name.lower() for hc_model in [ "large", "large-v2", "large-v3" ] );
    elif model_type == "mt":
        return any( hc_model in model_name.lower() for hc_model in [ "nllb-200", "3.3b", "1.3b" ] );
    return False;


def get_config( 
    asr_model: Optional[str] = None,
    mt_model: Optional[str] = None,
    device: Optional[str] = None,
    hf_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get complete configuration for the translator
    """
    config = {
        "asr_model": asr_model or DEFAULT_ASR_MODEL,
        "mt_model": mt_model or DEFAULT_MT_MODEL,
        "device": device or detect_device(),
        "hf_token": hf_token or get_hf_token()
    };
    
    # Add compute warnings;
    config["high_compute_asr"] = is_high_compute_model( config["asr_model"], "asr" );
    config["high_compute_mt"] = is_high_compute_model( config["mt_model"], "mt" );
    
    return config;


def print_device_info():
    """Print device and compute information"""
    device = detect_device();
    print( f"🖥️  Device: {device.upper()}" );
    
    if device == "cuda":
        print( f"   GPU: {torch.cuda.get_device_name()}" );
        print( f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" );
    elif device == "mps":
        print( "   Apple Silicon GPU acceleration enabled" );
    else:
        print( "   Using CPU (consider GPU for faster processing)" );