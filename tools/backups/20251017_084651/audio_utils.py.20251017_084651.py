"""
Audio utilities for WhatsApp voice message processing
Handles file validation, format conversion, and audio preprocessing
"""

import os
import subprocess
import tempfile
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Tuple
import librosa
import soundfile as sf


SUPPORTED_EXTENSIONS = { ".mp3", ".ogg", ".wav", ".m4a", ".aac" };
FFMPEG_AVAILABLE = shutil.which( "ffmpeg" ) is not None;


def validate_audio_path( file_path: str ) -> bool:
    """
    Validate audio file exists and has supported extension
    Handles WhatsApp files with spaces in names safely
    """
    if not os.path.exists( file_path ):
        return False;
    
    if not os.path.isfile( file_path ):
        return False;
    
    if not os.access( file_path, os.R_OK ):
        return False;
    
    file_ext = Path( file_path ).suffix.lower();
    return file_ext in SUPPORTED_EXTENSIONS;


def get_audio_info( file_path: str ) -> Optional[dict]:
    """Get basic audio file information using librosa"""
    try:
        # Load just to get info, don't load full audio yet;
        duration = librosa.get_duration( path=file_path );
        return {
            "duration_seconds": duration,
            "file_size_mb": os.path.getsize( file_path ) / ( 1024 * 1024 ),
            "extension": Path( file_path ).suffix.lower()
        };
    except Exception as e:
        print( f"⚠️  Could not analyze audio file: {e}" );
        return None;


def _convert_with_ffmpeg( input_path: str, output_path: str ) -> bool:
    """
    Convert audio file using ffmpeg to 16kHz mono WAV
    Handles filenames with spaces safely using proper quoting
    """
    if not FFMPEG_AVAILABLE:
        return False;
    
    try:
        cmd = [
            "ffmpeg", "-y",  # Overwrite output;
            "-i", input_path,
            "-ar", "16000",  # 16kHz sample rate;
            "-ac", "1",      # Mono;
            "-c:a", "pcm_s16le",  # 16-bit PCM;
            output_path
        ];
        
        result = subprocess.run( 
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30  # Reasonable timeout for short WhatsApp files;
        );
        
        if result.returncode == 0:
            return True;
        else:
            print( f"❌ FFmpeg conversion failed: {result.stderr}" );
            return False;
            
    except subprocess.TimeoutExpired:
        print( "❌ FFmpeg conversion timed out" );
        return False;
    except Exception as e:
        print( f"❌ FFmpeg conversion error: {e}" );
        return False;


def _convert_with_librosa( input_path: str, output_path: str ) -> bool:
    """
    Fallback conversion using librosa when ffmpeg unavailable
    """
    try:
        # Load audio and resample to 16kHz;
        audio, sr = librosa.load( input_path, sr=16000, mono=True );
        
        # Save as 16-bit WAV;
        sf.write( output_path, audio, 16000, subtype='PCM_16' );
        return True;
        
    except Exception as e:
        print( f"❌ Librosa conversion failed: {e}" );
        return False;


def ensure_wav_mono_16k( file_path: str, force_convert: bool = False ) -> str:
    """
    Ensure audio file is in 16kHz mono WAV format suitable for ASR
    Returns path to processed file (original or converted)
    
    Args:
        file_path: Path to input audio file
        force_convert: Always convert even if file seems compatible
    
    Returns:
        Path to audio file ready for ASR processing
    """
    if not validate_audio_path( file_path ):
        raise ValueError( f"Invalid audio file: {file_path}" );
    
    # Check if already WAV format and we don't need to force convert;
    file_ext = Path( file_path ).suffix.lower();
    if file_ext == ".wav" and not force_convert:
        # Try to verify it's actually 16kHz mono, but don't fail if we can't check;
        try:
            y, sr = librosa.load( file_path, sr=None );  # Load with original sample rate;
            if sr == 16000 and len( y.shape ) == 1:  # Mono check;
                return file_path;  # Already correct format;
        except:
            pass;  # Fall through to conversion;
    
    # Generate unique temporary file name;
    file_hash = hashlib.md5( file_path.encode() ).hexdigest()[:8];
    temp_name = f"converted-{file_hash}.wav";
    temp_path = os.path.join( tempfile.gettempdir(), temp_name );
    
    # Try ffmpeg first, then librosa fallback;
    conversion_success = False;
    
    if FFMPEG_AVAILABLE:
        print( f"🔄 Converting with ffmpeg: {Path(file_path).name} -> 16kHz mono WAV" );
        conversion_success = _convert_with_ffmpeg( file_path, temp_path );
    
    if not conversion_success:
        print( f"🔄 Converting with librosa: {Path(file_path).name} -> 16kHz mono WAV" );
        conversion_success = _convert_with_librosa( file_path, temp_path );
    
    if conversion_success and os.path.exists( temp_path ):
        print( f"✅ Converted audio ready: {temp_path}" );
        return temp_path;
    else:
        # Conversion failed, return original and hope ASR pipeline can handle it;
        print( f"⚠️  Conversion failed, using original file: {file_path}" );
        if not FFMPEG_AVAILABLE:
            print( "💡 Install ffmpeg for better audio format support: sudo apt-get install -y ffmpeg" );
        return file_path;


def find_whatsapp_files( search_dir: str = "/tmp" ) -> list:
    """
    Auto-discover WhatsApp voice message files in directory
    Returns list of found audio files
    """
    whatsapp_files = [];
    
    try:
        for file_path in Path( search_dir ).glob( "WhatsApp*" ):
            if file_path.is_file() and validate_audio_path( str( file_path ) ):
                whatsapp_files.append( str( file_path ) );
    except Exception as e:
        print( f"⚠️  Error searching for WhatsApp files: {e}" );
    
    return sorted( whatsapp_files );


def cleanup_temp_files( file_paths: list ):
    """Clean up temporary converted audio files"""
    for path in file_paths:
        if "/tmp/converted-" in path and os.path.exists( path ):
            try:
                os.remove( path );
                print( f"🗑️  Cleaned up: {Path(path).name}" );
            except Exception as e:
                print( f"⚠️  Could not clean up {path}: {e}" );