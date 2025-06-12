"""
Hardware configuration optimized for Intel 8700K + GTX 1070
Production capacity: 36 videos/day (6 languages × 6 videos)
"""
import json
import os
from typing import Dict, Any

class HardwareConfig:
    """Hardware-specific settings for optimal performance"""
    
    # CPU Configuration (Intel 8700K: 6 cores, 12 threads)
    CPU_CORES = 6
    CPU_THREADS = 12
    
    # Optimal threading for different operations
    TTS_WORKERS = 8          # Parallel TTS generation (leave 4 threads for system)
    FILE_IO_WORKERS = 4      # File operations
    QUEUE_WORKERS = 2        # Queue management
    
    # GPU Configuration (GTX 1070)
    GPU_MEMORY_GB = 8
    NVENC_PRESET = "fast"    # fast/medium/slow - fast for mass production
    NVENC_PROFILE = "high"   # baseline/main/high
    NVENC_LEVEL = "4.1"      # H.264 level
    NVENC_RC_MODE = "cbr"    # cbr/vbr - constant bitrate for consistency
    
    # Video encoding settings optimized for speed
    VIDEO_SETTINGS = {
        "codec": "h264_nvenc",
        "preset": NVENC_PRESET,
        "profile": NVENC_PROFILE,
        "level": NVENC_LEVEL,
        "rc": NVENC_RC_MODE,
        "bitrate": "2M",         # 2 Mbps for 720p
        "maxrate": "2.5M",
        "bufsize": "4M",
        "gpu": 0,                # First GPU
        "surfaces": 32,          # NVENC surfaces for better performance
    }
    
    # Output format settings
    OUTPUT_FORMAT = {
        "width": 720,            # Vertical video for TikTok
        "height": 1280,
        "fps": 30,
        "format": "mp4",
        "audio_bitrate": "128k", # Sufficient for TTS
        "audio_sample_rate": 44100,
    }
    
    # Performance thresholds
    PERFORMANCE_LIMITS = {
        "max_concurrent_videos": 3,    # Max simultaneous video processing
        "max_concurrent_tts": 6,       # Max TTS operations
        "cpu_usage_limit": 85,         # Percentage
        "gpu_usage_limit": 90,         # Percentage
        "memory_usage_limit": 80,      # Percentage
        "temp_limit_cpu": 80,          # Celsius
        "temp_limit_gpu": 83,          # GTX 1070 max safe temp
    }
    
    # Queue management
    QUEUE_CONFIG = {
        "max_queue_size": 100,
        "batch_size": 6,               # Process 6 videos (1 per language) at once
        "retry_attempts": 3,
        "timeout_seconds": 300,        # 5 minutes per video max
        "cleanup_interval": 3600,      # Cleanup temp files every hour
    }
    
    # File paths
    PATHS = {
        "temp_dir": "temp",
        "output_dir": "content/output",
        "logs_dir": "logs",
        "videos_dir": "content/videos",
        "texts_dir": "content/texts",
    }
    
    # FFmpeg optimization flags
    FFMPEG_GLOBAL_ARGS = [
        "-hide_banner",
        "-loglevel", "error",
        "-y",                          # Overwrite output files
        "-threads", str(CPU_THREADS),  # Use all CPU threads for FFmpeg
        "-hwaccel", "cuda",            # Hardware acceleration
        "-hwaccel_output_format", "cuda",
    ]
    
    # TTS configuration
    TTS_CONFIG = {
        "rate": "+20%",                # Slightly faster speech for engagement
        "volume": "+0%",               # Default volume
        "pitch": "+0Hz",               # Default pitch
        "output_format": "audio-16khz-32kbitrate-mono-mp3",
        "timeout": 30,                 # Seconds
    }
    
    @classmethod
    def get_nvenc_args(cls) -> Dict[str, Any]:
        """Get NVENC-specific FFmpeg arguments"""
        return {
            "vcodec": cls.VIDEO_SETTINGS["codec"],
            "preset": cls.VIDEO_SETTINGS["preset"],
            "profile:v": cls.VIDEO_SETTINGS["profile"],
            "level": cls.VIDEO_SETTINGS["level"],
            "rc": cls.VIDEO_SETTINGS["rc"],
            "b:v": cls.VIDEO_SETTINGS["bitrate"],
            "maxrate": cls.VIDEO_SETTINGS["maxrate"],
            "bufsize": cls.VIDEO_SETTINGS["bufsize"],
            "gpu": cls.VIDEO_SETTINGS["gpu"],
            "surfaces": cls.VIDEO_SETTINGS["surfaces"],
        }
    
    @classmethod
    def get_output_args(cls) -> Dict[str, Any]:
        """Get output format arguments"""
        return {
            "vf": f"scale={cls.OUTPUT_FORMAT['width']}:{cls.OUTPUT_FORMAT['height']}",
            "r": cls.OUTPUT_FORMAT["fps"],
            "acodec": "aac",
            "b:a": cls.OUTPUT_FORMAT["audio_bitrate"],
            "ar": cls.OUTPUT_FORMAT["audio_sample_rate"],
            "f": cls.OUTPUT_FORMAT["format"],
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for path in cls.PATHS.values():
            os.makedirs(path, exist_ok=True)

        # Loading JSON with languages
        with open("languages.json", "r", encoding="utf-8") as f:
            language_config = json.load(f)

        # Getting language library
        LANGUAGES = language_config["languages"]

        # Create language-specific output directories
        for lang_code in LANGUAGES.keys():
            os.makedirs(f"{cls.PATHS['output_dir']}/{lang_code}", exist_ok=True)    
        