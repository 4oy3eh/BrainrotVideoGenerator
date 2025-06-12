"""
Content Factory Modules
========================

Core modules for TikTok brainrot content mass production system.
Optimized for Intel 8700K + GTX 1070 hardware configuration.

Modules:
- tts_processor: Edge-TTS text-to-speech generation
- nvenc_processor: GPU-accelerated video encoding with NVENC
- queue_manager: Async job queue management
- performance_monitor: System resource monitoring
- utils: Shared utilities and helpers

Production target: 36 videos/day (6 languages × 6 videos)
Video format: 720x1280 vertical, H.264 NVENC, 30fps
"""

__version__ = "1.0.0"
__author__ = "Content Factory"
__description__ = "Mass TikTok content production system"

# Module imports for easy access
from .tts_processor import TTSProcessor
from .nvenc_processor import NVENCProcessor  
from .queue_manager import QueueManager
from .performance_monitor import PerformanceMonitor

__all__ = [
    "TTSProcessor",
    "NVENCProcessor", 
    "QueueManager",
    "PerformanceMonitor"
]