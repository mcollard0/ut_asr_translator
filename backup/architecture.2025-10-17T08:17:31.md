# WhatsApp ASR Translator Architecture

## Summary

The WhatsApp ASR Translator is a Python application that transcribes Spanish WhatsApp voice messages and translates them to English using Hugging Face transformers. It provides a complete pipeline from audio preprocessing to final translation with support for multiple audio formats and compute optimization warnings.

## CLI Interface

### Command Line Usage
```bash
# Auto-detect WhatsApp files in /tmp
python tools/run_whatsapp.py

# Process specific file
python tools/run_whatsapp.py --audio "/tmp/WhatsApp Ptt 2025-10-16 at 2.38.41 PM.ogg"

# Use different models
python tools/run_whatsapp.py --asr-model openai/whisper-base --mt-model Helsinki-NLP/opus-mt-es-en

# Save JSON output
python tools/run_whatsapp.py --json-out results.json --verbose
```

### Key Arguments
- `--audio`: Specify audio files (auto-detects WhatsApp files if omitted)
- `--asr-model`: Whisper ASR model (default: openai/whisper-small)
- `--mt-model`: Translation model (default: Helsinki-NLP/opus-mt-es-en)
- `--device`: Compute device (auto-detects CUDA > MPS > CPU)
- `--json-out`: Save results to JSON file
- `--verbose`: Detailed output with step-by-step progress

## Model Choices

### ASR Models (Speech-to-Text)
- **Default**: `openai/whisper-small` (244MB)
  - Good balance of speed and accuracy for Spanish
  - Processing time: ~2-10s for 10s audio on CPU
- **Alternatives**: 
  - `openai/whisper-base` (74MB) - Faster, slightly less accurate
  - `openai/whisper-large-v3` (1550MB) - **HIGH COMPUTE WARNING**

### Translation Models (Spanish→English)
- **Default**: `Helsinki-NLP/opus-mt-es-en` (300MB)
  - MarianMT - Fast and efficient neural translation
  - Processing time: <1s for typical WhatsApp message lengths
- **Alternatives**:
  - `facebook/nllb-200-1.3B` (1300MB) - **HIGH COMPUTE WARNING**

## Error Handling

### Audio Processing
- Validates file existence, permissions, and supported formats (.mp3, .ogg, .wav, .m4a)
- Automatic format conversion using ffmpeg (preferred) or librosa fallback
- Graceful handling of conversion failures with fallback to original file
- Cleanup of temporary converted files

### Model Loading
- HF token validation and secure handling via environment variables
- Device auto-detection with fallback chain: CUDA → MPS → CPU
- Clear error messages for missing models or authentication issues
- Compute warnings for large models that may require significant resources

### Pipeline Errors
- Each pipeline stage (ASR, translation) returns structured results with success flags
- Detailed error reporting with timing information
- Partial results preservation (e.g., successful ASR with failed translation)

## Logging Strategy

### Console Output
- Rich-formatted output with colored panels for Spanish/English results
- Progress indicators during model loading and processing
- Performance metrics (processing times, audio duration)
- Compute usage warnings for resource-intensive models

### File Logging
- Structured JSON output option for programmatic consumption
- Results include: audio info, transcriptions, translations, model names, timings
- Preserves all metadata for reproducibility and analysis

## Known Constraints

### Audio Format Support
- Primary support for WhatsApp formats: .ogg, .mp3
- Requires ffmpeg for robust .ogg handling (warns if missing)
- Limited to mono/stereo audio (automatically converted to mono)

### Model Requirements
- Internet connection required for initial model downloads
- Models cached locally after first use
- Large models (>1GB) may require significant RAM and processing time
- CUDA/MPS acceleration recommended for large models

### Language Support
- Optimized specifically for Spanish→English translation
- Whisper models support other languages but translation pipeline is Spanish-focused
- Input validation assumes Spanish speech content

## Current Feature Status

### ✅ Implemented
- [x] Spanish speech recognition using Whisper
- [x] Spanish to English translation using MarianMT
- [x] Audio format conversion and preprocessing
- [x] Device auto-detection and compute optimization
- [x] CLI with auto-file detection
- [x] Rich console output formatting
- [x] JSON export capabilities
- [x] HF token security handling
- [x] Compute usage warnings
- [x] Error handling and recovery

### 🚧 Potential Enhancements
- [ ] Batch processing optimization for multiple files
- [ ] Web interface for easier access
- [ ] Support for other language pairs
- [ ] Advanced audio preprocessing (noise reduction)
- [ ] Integration with WhatsApp backup formats

## External Requirements

### System Dependencies
- **Python 3.8+**
- **ffmpeg** (recommended): `sudo apt-get install -y ffmpeg`
  - Used for robust audio format conversion
  - Program works without it but may have limited .ogg support

### Python Dependencies
See `requirements.txt` for complete list:
- `transformers>=4.44.0` - HuggingFace model pipeline
- `torch>=2.2.0` - Deep learning framework
- `librosa>=0.10.1` - Audio processing
- `rich>=13.7.0` - Console formatting
- `soundfile>=0.12.1` - Audio I/O

### Authentication
- **Hugging Face Token**: Required for model downloads
- Set via environment variable: `HUGGING_FACE_API_KEY=hf_...`
- Token format validation and secure handling implemented

## File Organization

```
wa-asr-translator/
├── src/wa_asr_translator/        # Main package
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Entry point for -m execution
│   ├── cli.py                   # Command line interface
│   ├── config.py                # Configuration and device detection
│   ├── audio_utils.py           # Audio processing utilities
│   ├── asr.py                   # Spanish speech recognition
│   └── translate.py             # Spanish→English translation
├── tools/                       # Helper scripts
│   └── run_whatsapp.py          # Convenient execution wrapper
├── tests/                       # Test suite (basic smoke tests)
├── logs/                        # Runtime logs (created on execution)
├── backup/                      # Code backups (ISO date versioning)
├── docs/                        # Documentation
│   └── architecture.md          # This document
└── requirements.txt             # Python dependencies
```

## Database Schema
**N/A** - This is a stateless command-line tool with no persistent database.

## API Endpoints  
**N/A** - No HTTP API; pure command-line interface.

## Processing Pipeline

1. **File Discovery**: Auto-detect WhatsApp files or use specified paths
2. **Audio Validation**: Check file existence, permissions, format support
3. **Audio Preprocessing**: Convert to 16kHz mono WAV if needed
4. **Model Loading**: Initialize Whisper ASR and MarianMT translation pipelines
5. **Speech Recognition**: Transcribe Spanish audio to text with language hints
6. **Translation**: Convert Spanish text to English with normalization
7. **Output Formatting**: Display results with rich console formatting
8. **Cleanup**: Remove temporary files and provide summary statistics

---
*Last updated: 2025-10-16*  
*Total processing time for 10s WhatsApp message: ~3-15s depending on device*