"""
Hardware configuration optimized for Intel 8700K + GTX 1070
Production capacity: 36 videos/day (6 languages × 6 videos)
Dynamic hardware detection with fallback to optimized defaults
"""

import os
import psutil
from typing import Dict, Any, Optional

try:
    import GPUtil
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False


class HardwareConfig:
    """Hardware configuration with dynamic detection and optimized defaults"""
    
    def __init__(self):
        # Detect hardware
        self.cpu_cores = os.cpu_count() or 6
        self.cpu_threads = self.cpu_cores * 2  # Assuming hyperthreading
        self.ram_gb = self._get_ram_gb()
        self.gpu_info = self._get_gpu_info()
        
        # Set optimal threading based on detected hardware
        self._configure_threading()
        
        # Static optimized settings for target hardware
        self._configure_video_settings()
        self._configure_performance_limits()
        self._configure_paths()
        
    def _get_ram_gb(self) -> float:
        """Get total RAM in GB"""
        try:
            return psutil.virtual_memory().total / (1024**3)
        except:
            return 16.0  # Default assumption
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information with fallback"""
        if not GPU_UTILS_AVAILABLE:
            # Fallback for target hardware
            return {
                "name": "GTX 1070",
                "memory_total": 8192,
                "memory_free": 6144,
                "nvenc_support": True
            }
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_free": gpu.memoryFree,
                    "nvenc_support": self._check_nvenc_support(gpu.name)
                }
        except Exception:
            pass
        
        # Fallback
        return {
            "name": "Unknown GPU",
            "memory_total": 4096,
            "memory_free": 3072,
            "nvenc_support": False
        }
    
    def _check_nvenc_support(self, gpu_name: str) -> bool:
        """Check if GPU supports NVENC"""
        nvenc_gpus = ["GTX 1070", "GTX 1080", "RTX", "GTX 16", "GTX 20", "GTX 30", "GTX 40"]
        return any(gpu in gpu_name for gpu in nvenc_gpus)
    
    def _configure_threading(self):
        """Configure optimal threading based on hardware"""
        # Conservative approach - leave headroom for system
        available_threads = max(1, self.cpu_threads - 2)
        
        # TTS workers (most CPU intensive)
        self.TTS_WORKERS = min(8, max(2, available_threads // 2))
        
        # File I/O workers
        self.FILE_IO_WORKERS = min(4, max(1, available_threads // 4))
        
        # Queue management
        self.QUEUE_WORKERS = 2
        
        # Max concurrent operations
        self.max_concurrent_tts = min(3, max(1, self.cpu_cores // 2))
        self.max_concurrent_video = 1  # NVENC hardware limitation
        
        # FFmpeg threads
        self.ffmpeg_threads = min(6, max(2, available_threads // 2))
    
    def _configure_video_settings(self):
        """Configure video encoding settings"""
        # NVENC settings for GTX 1070
        if self.gpu_info["nvenc_support"]:
            self.VIDEO_SETTINGS = {
                "codec": "h264_nvenc",
                "preset": "p4",          # NVENC preset (p1=fastest, p7=slowest)
                "profile": "high",
                "level": "4.1",
                "rc": "vbr",            # Variable bitrate for better quality
                "cq": 28,               # Constant quality (lower = better quality)
                "bitrate": "2M",
                "maxrate": "2.5M",
                "bufsize": "4M",
                "gpu": 0,
                "surfaces": 32,         # NVENC surfaces for better performance
                "2pass": "0",           # Disable 2-pass for speed
            }
        else:
            # CPU fallback
            self.VIDEO_SETTINGS = {
                "codec": "libx264",
                "preset": "fast",
                "profile": "high",
                "level": "4.1",
                "crf": 28,
                "bitrate": "2M",
                "maxrate": "2.5M",
                "bufsize": "4M",
            }
        
        # Output format settings
        self.OUTPUT_FORMAT = {
            "width": 720,            # Vertical video for TikTok
            "height": 1280,
            "fps": 30,
            "format": "mp4",
            "audio_bitrate": "128k",
            "audio_sample_rate": 44100,
        }
    
    def _configure_performance_limits(self):
        """Configure performance limits based on hardware"""
        # Base limits
        base_queue_size = 50
        base_concurrent_jobs = 2
        
        # Scale based on available resources
        ram_factor = min(2.0, self.ram_gb / 16.0)  # Scale based on 16GB baseline
        cpu_factor = min(2.0, self.cpu_cores / 6.0)  # Scale based on 6-core baseline
        
        self.PERFORMANCE_LIMITS = {
            "max_queue_size": int(base_queue_size * ram_factor),
            "max_concurrent_jobs": max(1, int(base_concurrent_jobs * cpu_factor)),
            "max_concurrent_videos": min(3, self.max_concurrent_video),
            "max_concurrent_tts": self.max_concurrent_tts,
            "cpu_usage_limit": 85,
            "gpu_usage_limit": 90,
            "memory_usage_limit": 80,
            "temp_limit_cpu": 80,
            "temp_limit_gpu": 83,
            "max_video_length": 300,  # 5 minutes
            "max_file_size": 100 * 1024 * 1024,  # 100MB
            "cleanup_interval": 3600,  # 1 hour
        }
        
        # Queue management
        self.QUEUE_CONFIG = {
            "max_queue_size": self.PERFORMANCE_LIMITS["max_queue_size"],
            "batch_size": 6,  # Process 6 videos (1 per language) at once
            "retry_attempts": 3,
            "timeout_seconds": 300,  # 5 minutes per video max
            "cleanup_interval": 3600,
        }
    
    def _configure_paths(self):
        """Configure file paths"""
        self.PATHS = {
            "temp_dir": "temp",
            "output_dir": "content/output",
            "logs_dir": "logs",
            "videos_dir": "content/videos",
            "texts_dir": "content/texts",
        }
    
    # TTS configuration
    TTS_CONFIG = {
        "rate": "+20%",
        "volume": "+0%",
        "pitch": "+0Hz",
        "output_format": "audio-16khz-32kbitrate-mono-mp3",
        "timeout": 30,
        "max_retry_attempts": 3,
        "retry_delay": 1.0,  # seconds
    }
    
    # FFmpeg optimization flags
    @property
    def FFMPEG_GLOBAL_ARGS(self) -> list:
        """Get FFmpeg global arguments"""
        return [
            "-hide_banner",
            "-loglevel", "error",
            "-y",  # Overwrite output files
            "-threads", str(self.ffmpeg_threads),
            "-hwaccel", "cuda" if self.gpu_info["nvenc_support"] else "auto",
            "-hwaccel_output_format", "cuda" if self.gpu_info["nvenc_support"] else "auto",
        ]
    
    def get_nvenc_args(self) -> Dict[str, Any]:
        """Get NVENC-specific FFmpeg arguments"""
        if not self.gpu_info["nvenc_support"]:
            return {}
        
        return {
            "vcodec": self.VIDEO_SETTINGS["codec"],
            "preset": self.VIDEO_SETTINGS["preset"],
            "profile:v": self.VIDEO_SETTINGS["profile"],
            "level": self.VIDEO_SETTINGS["level"],
            "rc": self.VIDEO_SETTINGS["rc"],
            "cq": self.VIDEO_SETTINGS["cq"],
            "b:v": self.VIDEO_SETTINGS["bitrate"],
            "maxrate": self.VIDEO_SETTINGS["maxrate"],
            "bufsize": self.VIDEO_SETTINGS["bufsize"],
            "gpu": self.VIDEO_SETTINGS["gpu"],
            "surfaces": self.VIDEO_SETTINGS["surfaces"],
            "2pass": self.VIDEO_SETTINGS["2pass"],
        }
    
    def get_cpu_args(self) -> Dict[str, Any]:
        """Get CPU encoding arguments (fallback)"""
        return {
            "vcodec": "libx264",
            "preset": "fast",
            "profile:v": "high",
            "level": "4.1",
            "crf": 28,
            "b:v": self.VIDEO_SETTINGS["bitrate"],
            "maxrate": self.VIDEO_SETTINGS["maxrate"],
            "bufsize": self.VIDEO_SETTINGS["bufsize"],
        }
    
    def get_ffmpeg_params(self) -> Dict[str, Any]:
        """Get optimized FFmpeg parameters"""
        if self.gpu_info["nvenc_support"]:
            return self.get_nvenc_args()
        else:
            return self.get_cpu_args()
    
    def get_output_args(self) -> Dict[str, Any]:
        """Get output format arguments"""
        return {
            "vf": f"scale={self.OUTPUT_FORMAT['width']}:{self.OUTPUT_FORMAT['height']}",
            "r": self.OUTPUT_FORMAT["fps"],
            "acodec": "aac",
            "b:a": self.OUTPUT_FORMAT["audio_bitrate"],
            "ar": self.OUTPUT_FORMAT["audio_sample_rate"],
            "f": self.OUTPUT_FORMAT["format"],
        }
    
    def get_performance_limits(self) -> Dict[str, Any]:
        """Get performance limits based on hardware"""
        return self.PERFORMANCE_LIMITS.copy()
    
    def create_directories(self):
        """Create necessary directories"""
        for path in self.PATHS.values():
            os.makedirs(path, exist_ok=True)
        
        # Create language-specific output directories
        languages = ["en", "es", "pt", "fr", "de", "ru"]
        for lang_code in languages:
            os.makedirs(f"{self.PATHS['output_dir']}/{lang_code}", exist_ok=True)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for diagnostics"""
        return {
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "ram_gb": round(self.ram_gb, 2),
            "gpu_name": self.gpu_info["name"],
            "gpu_memory": self.gpu_info["memory_total"],
            "nvenc_support": self.gpu_info["nvenc_support"],
            "tts_workers": self.TTS_WORKERS,
            "max_concurrent_tts": self.max_concurrent_tts,
            "max_concurrent_video": self.max_concurrent_video,
            "ffmpeg_threads": self.ffmpeg_threads,
        }
    
    def validate_hardware(self) -> Dict[str, bool]:
        """Validate hardware meets minimum requirements"""
        return {
            "cpu_cores_ok": self.cpu_cores >= 4,
            "ram_ok": self.ram_gb >= 8,
            "gpu_memory_ok": self.gpu_info["memory_total"] >= 4096,
            "nvenc_available": self.gpu_info["nvenc_support"],
            "overall_ok": (
                self.cpu_cores >= 4 and 
                self.ram_gb >= 8 and 
                self.gpu_info["memory_total"] >= 2048
            )
        }