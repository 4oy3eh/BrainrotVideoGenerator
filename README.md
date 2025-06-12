# TikTok Content Factory 🏭

Automated mass production system for TikTok content with multi-language TTS support and GPU-accelerated video processing.

## 🎯 Project Overview

This system produces **36 videos per day** (6 languages × 6 videos) from text input and background videos, with professional TTS narration and optimized rendering.

### Key Features
- **Mass Production**: 36 videos daily with automated pipeline
- **Multi-language TTS**: 6 languages with natural-sounding voices
- **GPU Acceleration**: NVENC hardware encoding for speed
- **Queue Management**: Batch processing with progress tracking
- **Performance Monitoring**: Real-time system metrics
- **Web Interface**: Simple upload and management UI

## 🛠️ Tech Stack

- **Backend**: FastAPI + asyncio
- **TTS**: Edge-TTS (free, TikTok-quality voices)
- **Video Processing**: FFmpeg + NVENC GPU acceleration
- **Frontend**: HTML/JavaScript with real-time updates
- **Performance**: Optimized for Intel 8700K + GTX 1070

## 🌍 Supported Languages

| Language | Voice | Code |
|----------|-------|------|
| English | Jenny Neural | en-US-JennyNeural |
| Spanish | Dalia Neural | es-MX-DaliaNeural |
| Portuguese | Francisca Neural | pt-BR-FranciscaNeural |
| French | Denise Neural | fr-FR-DeniseNeural |
| German | Katja Neural | de-DE-KatjaNeural |
| Russian | Svetlana Neural | ru-RU-SvetlanaNeural |

## 📁 Project Structure

```
content_factory/
├── app.py                     # Main FastAPI application
├── launcher.py               # System launcher with dependency checks
├── requirements.txt          # Python dependencies
├── config/
│   ├── hardware_config.py   # Hardware optimization settings
│   └── languages.json       # Language and voice configuration
├── frontend/
│   ├── index.html           # Web interface
│   └── static/              # CSS, JavaScript assets
├── content/
│   ├── texts/               # Input text files
│   ├── videos/              # Background video files
│   └── output/              # Generated videos (by language)
├── modules/
│   ├── tts_processor.py     # Text-to-speech processing
│   ├── nvenc_processor.py   # GPU-accelerated video rendering
│   ├── queue_manager.py     # Job queue and batch processing
│   ├── performance_monitor.py # System performance tracking
│   └── utils.py             # Utilities and validation
├── temp/                    # Temporary processing files
└── logs/                    # System logs
```

## 🚀 Quick Start

### Prerequisites

1. **Hardware Requirements**:
   - NVIDIA GPU with NVENC support (GTX 1070+ recommended)
   - 8GB+ RAM
   - 50GB+ free disk space

2. **Software Dependencies**:
   - Python 3.8+
   - FFmpeg with NVENC support
   - NVIDIA drivers

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd content_factory
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify FFmpeg with NVENC**:
   ```bash
   ffmpeg -encoders | grep nvenc
   ```

4. **Create required directories**:
   ```bash
   mkdir -p content/{texts,videos,output} temp logs
   ```

### Running the System

1. **Launch with automatic checks**:
   ```bash
   python launcher.py
   ```

2. **Or run directly**:
   ```bash
   python app.py
   ```

3. **Access web interface**:
   ```
   http://localhost:8000
   ```

## 📝 Usage Guide

### 1. Prepare Content

**Text Files** (`content/texts/`):
- Format: `.txt` files with your script
- Length: 50-500 words (optimal for TikTok)
- Encoding: UTF-8

**Background Videos** (`content/videos/`):
- Format: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- Resolution: Any (will be processed to 720p vertical)
- Duration: 15-60 seconds recommended

### 2. Upload via Web Interface

1. Open `http://localhost:8000`
2. Upload text file and select background videos
3. Choose target languages
4. Click "Start Production"

### 3. Monitor Progress

- Real-time queue status
- Processing progress per language
- System performance metrics
- Download completed videos

### 4. Output Files

Generated videos are saved in:
```
content/output/
├── en/          # English videos
├── es/          # Spanish videos
├── pt/          # Portuguese videos
├── fr/          # French videos
├── de/          # German videos
└── ru/          # Russian videos
```

## ⚙️ Configuration

### Hardware Optimization

Edit `config/hardware_config.py`:

```python
# GPU Settings
NVENC_PRESET = "fast"        # fast, medium, slow
NVENC_QUALITY = "medium"     # high, medium, low
MAX_CONCURRENT_JOBS = 3      # Based on GPU memory

# CPU Settings
FFMPEG_THREADS = 6           # CPU cores for non-GPU tasks
TTS_BATCH_SIZE = 4          # Concurrent TTS generations
```

### Language Configuration

Edit `config/languages.json` to add/modify languages:

```json
{
  "languages": {
    "new_lang": {
      "voice": "voice-name-neural",
      "name": "Language Name",
      "enabled": true
    }
  }
}
```

## 📊 Performance Optimization

### Recommended Settings by Hardware

**GTX 1070 (8GB VRAM)**:
```python
MAX_CONCURRENT_JOBS = 2
NVENC_PRESET = "fast"
BATCH_SIZE = 3
```

**RTX 3070+ (10GB+ VRAM)**:
```python
MAX_CONCURRENT_JOBS = 4
NVENC_PRESET = "medium"
BATCH_SIZE = 6
```

### Production Targets

- **Single video**: ~2-3 minutes processing time
- **6-language batch**: ~15-20 minutes total
- **Daily target**: 36 videos in 3-4 hours

## 🔧 Troubleshooting

### Common Issues

**NVENC not available**:
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify FFmpeg NVENC support
ffmpeg -encoders | grep nvenc
```

**Out of GPU memory**:
- Reduce `MAX_CONCURRENT_JOBS`
- Lower video resolution
- Close other GPU applications

**TTS errors**:
- Check internet connection (Edge-TTS requires online access)
- Verify text encoding (must be UTF-8)
- Check text length limits

**Slow processing**:
- Enable GPU acceleration
- Increase `FFMPEG_THREADS`
- Use SSD for temp directory

### Log Files

Check logs for detailed error information:
```bash
tail -f logs/system.log
tail -f logs/queue.log
tail -f logs/performance.log
```

## 🔒 Security Notes

- System designed for local use only
- No authentication implemented
- Keep temp files cleaned regularly
- Monitor disk usage for large batches

## 📈 Scaling

For higher production volumes:

1. **Multiple GPU setup**: Modify hardware config for multi-GPU
2. **Distributed processing**: Run multiple instances
3. **Cloud deployment**: AWS/GCP with GPU instances
4. **Storage optimization**: Network storage for large video libraries

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Test with sample content
4. Submit pull request

## 📄 License

This project is for educational and personal use. Ensure compliance with TikTok's terms of service and content policies.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Review log files
3. Open GitHub issue with:
   - System specifications
   - Error logs
   - Steps to reproduce

---

**Made for efficient TikTok content creation** 🎬✨