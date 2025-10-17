"""
Command Line Interface for WhatsApp ASR Translator
Handles user input, compute warnings, and orchestrates the translation pipeline
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .config import get_config, print_device_info, is_high_compute_model
from .audio_utils import find_whatsapp_files, validate_audio_path, ensure_wav_mono_16k, get_audio_info, cleanup_temp_files
from .asr import transcribe_spanish, get_model_info
from .translate import translate_es_to_en, get_translation_model_info


console = Console();


def print_compute_warning( config: Dict[str, Any] ):
    """Print compute usage warnings for selected models"""
    warnings = [];
    
    if config["high_compute_asr"]:
        asr_info = get_model_info( config["asr_model"] );
        warnings.append( f"ASR model {config['asr_model']} is HIGH COMPUTE ({asr_info.get('estimated_size_mb', 'unknown')} MB)" );
    
    if config["high_compute_mt"]:
        mt_info = get_translation_model_info( config["mt_model"] );
        warnings.append( f"Translation model {config['mt_model']} is HIGH COMPUTE ({mt_info.get('estimated_size_mb', 'unknown')} MB)" );
    
    if warnings:
        console.print();
        console.print( "⚠️ [bold red]COMPUTE WARNING[/bold red]", style="red" );
        for warning in warnings:
            console.print( f"   {warning}", style="yellow" );
        console.print( "   Expected processing time: 30s - 5+ minutes depending on hardware", style="yellow" );
        console.print();


def print_model_info( config: Dict[str, Any] ):
    """Print information about selected models"""
    asr_info = get_model_info( config["asr_model"] );
    mt_info = get_translation_model_info( config["mt_model"] );
    
    table = Table( show_header=True, header_style="bold blue" );
    table.add_column( "Component", style="cyan" );
    table.add_column( "Model", style="green" );
    table.add_column( "Size (MB)", justify="right" );
    table.add_column( "Description" );
    
    table.add_row( 
        "Speech-to-Text", 
        config["asr_model"], 
        str( asr_info.get( "estimated_size_mb", "?" ) ),
        "Whisper multilingual ASR"
    );
    
    table.add_row(
        "Translation",
        config["mt_model"],
        str( mt_info.get( "estimated_size_mb", "?" ) ),
        mt_info.get( "description", "Spanish→English translation" )
    );
    
    console.print( table );
    console.print();


def process_audio_file( audio_path: str, config: Dict[str, Any] ) -> Dict[str, Any]:
    """Process a single audio file through ASR and translation pipeline"""
    
    result = {
        "audio_path": audio_path,
        "audio_info": get_audio_info( audio_path ),
        "asr_result": None,
        "translation_result": None,
        "success": False,
        "error": None
    };
    
    temp_files = [];
    
    try:
        console.print( f"🎵 Processing: [blue]{Path(audio_path).name}[/blue]" );
        
        # Convert audio to suitable format;
        processed_audio_path = ensure_wav_mono_16k( audio_path );
        if processed_audio_path != audio_path:
            temp_files.append( processed_audio_path );
        
        # Step 1: Spanish ASR;
        console.print( "📝 Transcribing Spanish audio..." );
        asr_result = transcribe_spanish(
            processed_audio_path,
            config["asr_model"],
            config["device"],
            config["hf_token"]
        );
        result["asr_result"] = asr_result;
        
        if not asr_result["success"]:
            result["error"] = f"ASR failed: {asr_result.get('error', 'Unknown error')}";
            return result;
        
        spanish_text = asr_result["text"];
        if not spanish_text:
            result["error"] = "No speech detected in audio";
            return result;
        
        # Step 2: Spanish to English translation;
        console.print( "🌍 Translating to English..." );
        translation_result = translate_es_to_en(
            spanish_text,
            config["mt_model"],
            config["device"],
            config["hf_token"]
        );
        result["translation_result"] = translation_result;
        
        if not translation_result["success"]:
            result["error"] = f"Translation failed: {translation_result.get('error', 'Unknown error')}";
            return result;
        
        result["success"] = True;
        return result;
        
    except Exception as e:
        result["error"] = str( e );
        return result;
        
    finally:
        # Clean up temporary files;
        if temp_files:
            cleanup_temp_files( temp_files );


def print_results( results: List[Dict[str, Any]] ):
    """Print translation results in a nice format"""
    
    for i, result in enumerate( results ):
        if len( results ) > 1:
            console.print( f"\n{'='*60}" );
            console.print( f"[bold]Result {i+1}/{len(results)}[/bold] - {Path(result['audio_path']).name}" );
            console.print( f"{'='*60}" );
        
        if not result["success"]:
            console.print( f"❌ [red]Processing failed: {result['error']}[/red]" );
            continue;
        
        # Audio info;
        if result["audio_info"]:
            audio_info = result["audio_info"];
            console.print( f"🎵 Audio: {audio_info['duration_seconds']:.1f}s, {audio_info['file_size_mb']:.1f}MB {audio_info['extension']}" );
        
        # Spanish transcription;
        spanish_text = result["asr_result"]["text"];
        console.print();
        console.print( Panel( 
            spanish_text, 
            title="🇪🇸 Spanish Transcription",
            title_align="left",
            style="blue"
        ) );
        
        # English translation;
        english_text = result["translation_result"]["translated_text"];
        console.print();
        console.print( Panel( 
            english_text, 
            title="🇺🇸 English Translation",
            title_align="left",
            style="green"
        ) );
        
        # Performance metrics;
        asr_time = result["asr_result"]["transcription_time_seconds"];
        trans_time = result["translation_result"]["translation_time_seconds"];
        total_time = asr_time + trans_time;
        
        console.print( f"\n⏱️  Processing time: {total_time:.2f}s (ASR: {asr_time:.2f}s, Translation: {trans_time:.2f}s)" );


def save_json_results( results: List[Dict[str, Any]], output_path: str ):
    """Save results to JSON file"""
    try:
        with open( output_path, 'w', encoding='utf-8' ) as f:
            json.dump( results, f, indent=2, ensure_ascii=False );
        console.print( f"💾 Results saved to: [blue]{output_path}[/blue]" );
    except Exception as e:
        console.print( f"❌ Failed to save JSON: {e}", style="red" );


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="Translate Spanish WhatsApp voice messages to English using Hugging Face models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect WhatsApp files in /tmp
  python -m wa_asr_translator.cli
  
  # Process specific file
  python -m wa_asr_translator.cli --audio "/tmp/WhatsApp Ptt 2025-10-16 at 2.38.41 PM.ogg"
  
  # Use different models
  python -m wa_asr_translator.cli --asr-model openai/whisper-base --mt-model Helsinki-NLP/opus-mt-es-en
  
  # Force CPU usage
  python -m wa_asr_translator.cli --device cpu
        """
    );
    
    parser.add_argument(
        "--audio", "-a",
        action="append",
        help="Audio file path (can be used multiple times). If not specified, auto-detects WhatsApp files in /tmp"
    );
    
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-small",
        help="Whisper ASR model (default: openai/whisper-small)"
    );
    
    parser.add_argument(
        "--mt-model", 
        default="Helsinki-NLP/opus-mt-es-en",
        help="Translation model (default: Helsinki-NLP/opus-mt-es-en)"
    );
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use (default: auto)"
    );
    
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token (prefer using HUGGING_FACE_API_KEY env var)"
    );
    
    parser.add_argument(
        "--json-out",
        help="Save results to JSON file"
    );
    
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip audio format conversion"
    );
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    );
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet output (minimal)"
    );
    
    return parser;


def main():
    """Main CLI entry point"""
    parser = create_parser();
    args = parser.parse_args();
    
    # Token security warning;
    if args.hf_token:
        console.print( "⚠️  [yellow]HF token provided via command line (visible in shell history)[/yellow]" );
        console.print( "💡 [blue]Prefer setting HUGGING_FACE_API_KEY environment variable[/blue]\n" );
    
    # Get configuration;
    config = get_config(
        asr_model=args.asr_model,
        mt_model=args.mt_model,
        device=None if args.device == "auto" else args.device,
        hf_token=args.hf_token
    );
    
    if not args.quiet:
        console.print( "[bold blue]🎤 WhatsApp Spanish → English Translator[/bold blue]\n" );
        print_device_info();
        console.print();
        print_model_info( config );
        print_compute_warning( config );
    
    # Find audio files;
    if args.audio:
        audio_files = [];
        for audio_path in args.audio:
            if validate_audio_path( audio_path ):
                audio_files.append( audio_path );
            else:
                console.print( f"❌ Invalid audio file: {audio_path}", style="red" );
                sys.exit( 1 );
    else:
        # Auto-detect WhatsApp files;
        audio_files = find_whatsapp_files( "/tmp" );
        if not audio_files:
            console.print( "❌ [red]No WhatsApp audio files found in /tmp[/red]" );
            console.print( "💡 [blue]Use --audio to specify file path manually[/blue]" );
            sys.exit( 1 );
        
        if not args.quiet:
            console.print( f"🔍 Found {len(audio_files)} WhatsApp audio files:" );
            for f in audio_files:
                console.print( f"   • {Path(f).name}" );
            console.print();
    
    # Process audio files;
    results = [];
    
    for audio_file in audio_files:
        if not args.quiet and len( audio_files ) > 1:
            console.print( f"\n[bold]Processing {Path(audio_file).name}...[/bold]" );
        
        result = process_audio_file( audio_file, config );
        results.append( result );
        
        if args.verbose:
            print_results( [result] );
    
    # Print final results;
    if not args.quiet:
        console.print( "\n" + "="*60 );
        console.print( "[bold green]🎯 FINAL RESULTS[/bold green]" );
        console.print( "="*60 );
    
    print_results( results );
    
    # Save JSON output if requested;
    if args.json_out:
        save_json_results( results, args.json_out );
    
    # Summary;
    successful = sum( 1 for r in results if r["success"] );
    if not args.quiet:
        console.print( f"\n✅ Successfully processed {successful}/{len(results)} files" );
    
    if successful < len( results ):
        sys.exit( 1 );


if __name__ == "__main__":
    main();