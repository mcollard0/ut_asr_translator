# 🎤 UT ASR Translator - Universal Speech Translation

Transcribe and translate voice messages using Hugging Face transformers. Supports multiple languages and audio formats.

## ⚡ Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Hugging Face token**:
   ```bash
   export HUGGING_FACE_API_KEY={{HUGGING_FACE_API_KEY}}
   ```

3. **Run the translator**:
   ```bash
   # Auto-detect audio files in /tmp (recommended)
   python -m ut_asr_translator
   
   # Or specify a file
   python -m ut_asr_translator --audio "/tmp/WhatsApp Ptt 2025-10-16 at 2.38.41 PM.ogg"
   
   # Alternative: Use the wrapper script
   python tools/run_whatsapp.py
   ```

## 📁 Your Files

The program automatically detected these WhatsApp voice messages:
- `/tmp/WhatsApp Ptt 2025-10-16 at 2.38.41 PM.mp3` (81KB)
- `/tmp/WhatsApp Ptt 2025-10-16 at 2.38.41 PM.ogg` (21KB)

Both files are ~10 seconds long and will process quickly (LOW compute usage).

## 🔧 Installation

### System Requirements
```bash
# Ubuntu/Debian (recommended for better audio support)
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage Examples

```bash
# Basic usage - auto-detects audio files
python -m ut_asr_translator

# Verbose output with detailed progress
python -m ut_asr_translator --verbose

# Save results to JSON
python -m ut_asr_translator --json-out results.json

# Use different models (faster but less accurate)
python -m ut_asr_translator --asr-model openai/whisper-base

# Force CPU usage
python -m ut_asr_translator --device cpu

# Alternative: Use the wrapper script (legacy)
python tools/run_whatsapp.py
```

## 🎯 Expected Output

```
🎤 UT ASR Translator - Universal Speech Translation

🖥️  Device: CUDA
   GPU: NVIDIA GeForce RTX 4080

┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Component     ┃ Model                               ┃  Size (MB) ┃ Description                          ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Speech-to-Text │ openai/whisper-small               │       244 │ Whisper multilingual ASR            │
│ Translation    │ Helsinki-NLP/opus-mt-es-en         │       300 │ Fast, efficient neural translation  │
└───────────────┴─────────────────────────────────────┴───────────┴──────────────────────────────────────┘

🔍 Found 2 WhatsApp audio files:
   • WhatsApp Ptt 2025-10-16 at 2.38.41 PM.mp3
   • WhatsApp Ptt 2025-10-16 at 2.38.41 PM.ogg

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 🇪🇸 Spanish Transcription                                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ [Original transcription will appear here]                  │
└────────────────────────────────────────────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 🇺🇸 English Translation                                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ [Source translation will appear here]                      │
└────────────────────────────────────────────────────────────┘

⏱️  Processing time: 4.2s (ASR: 3.8s, Translation: 0.4s)
✅ Successfully processed 2/2 files
```

## ⚠️ Compute Usage

For your ~10 second audio files:
- **Current setup**: LOW compute usage
- **Processing time**: ~3-15 seconds total depending on device
- **Models used**: Efficient models optimized for speed

The program will warn you if you select high-compute models like `whisper-large-v3`.

## 📊 Features

- ✅ Auto-detects WhatsApp voice messages
- ✅ Handles .ogg and .mp3 formats with spaces in filenames  
- ✅ Spanish speech recognition using Whisper
- ✅ Spanish→English translation using MarianMT
- ✅ Device auto-detection (CUDA/MPS/CPU)  <-- FALLBACK!
- ✅ Secure HF token handling
- ✅ Rich console output with progress indicators
- ✅ JSON export for programmatic use
- ✅ Compute usage warnings
- ✅ Error handling and recovery

## 🔒 Security

Your Hugging Face token is handled securely:
- Store in environment variable (recommended): `HUGGING_FACE_API_KEY`
- Never hardcoded in source files
- Token validation and format checking
- Warning if passed via command line (visible in shell history)

## 📋 Requirements

See `requirements.txt` for complete list. Key dependencies:
- `transformers>=4.44.0` - Hugging Face models
- `torch>=2.2.0` - Deep learning framework  
- `librosa>=0.10.1` - Audio processing
- `rich>=13.7.0` - Beautiful console output

## 📚 Documentation

- `docs/architecture.md` - Complete technical documentation
- `tools/run_whatsapp.py` - Simple execution wrapper
- `src/ut_asr_translator/` - Full source code

## 🐛 Troubleshooting

**No audio files found**: Use `--audio` to specify file path manually
**ffmpeg not found**: Install with `sudo apt-get install -y ffmpeg`  
**CUDA errors**: Use `--device cpu` to force CPU usage  
**Token errors**: Check `HUGGING_FACE_API_KEY` environment variable

---
*Processing your specific WhatsApp files: ~10s audio → ~5s total processing time*
