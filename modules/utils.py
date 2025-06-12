"""
Content Factory Utils
Utilities for TikTok content mass production system
"""

import os
import sys
import json
import shutil
import logging
import hashlib
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import mimetypes
import psutil

# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging system"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger('content_factory')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        log_dir / f"content_factory_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Error file handler
    error_handler = logging.FileHandler(log_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    return logger

logger = setup_logging()

class FileValidator:
    """File validation and safety checks"""
    
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    ALLOWED_TEXT_EXTENSIONS = {'.txt', '.json'}
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    MIN_VIDEO_DURATION = 5  # seconds
    MAX_VIDEO_DURATION = 300  # 5 minutes
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path for security"""
        try:
            path = Path(file_path).resolve()
            # Check for path traversal
            if '..' in str(path) or str(path).startswith('/'):
                logger.warning(f"Suspicious file path detected: {file_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return False
    
    @staticmethod
    def validate_video_file(file_path: str) -> Dict[str, Any]:
        """Comprehensive video file validation"""
        result = {
            'valid': False,
            'error': None,
            'info': {},
            'warnings': []
        }
        
        try:
            path = Path(file_path)
            
            # Check existence
            if not path.exists():
                result['error'] = "File does not exist"
                return result
            
            # Check extension
            if path.suffix.lower() not in FileValidator.ALLOWED_VIDEO_EXTENSIONS:
                result['error'] = f"Invalid extension. Allowed: {FileValidator.ALLOWED_VIDEO_EXTENSIONS}"
                return result
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > FileValidator.MAX_FILE_SIZE:
                result['error'] = f"File too large: {file_size / 1024 / 1024:.1f}MB > {FileValidator.MAX_FILE_SIZE / 1024 / 1024}MB"
                return result
            
            if file_size < 1024:  # Less than 1KB
                result['error'] = "File too small, likely corrupted"
                return result
            
            # Get video info using ffprobe
            video_info = get_video_info(file_path)
            if not video_info:
                result['error'] = "Could not read video information"
                return result
            
            result['info'] = video_info
            
            # Check duration
            duration = video_info.get('duration', 0)
            if duration < FileValidator.MIN_VIDEO_DURATION:
                result['warnings'].append(f"Video very short: {duration}s")
            elif duration > FileValidator.MAX_VIDEO_DURATION:
                result['warnings'].append(f"Video very long: {duration}s")
            
            # Check resolution
            width = video_info.get('width', 0)
            height = video_info.get('height', 0)
            if width < 480 or height < 480:
                result['warnings'].append(f"Low resolution: {width}x{height}")
            
            result['valid'] = True
            logger.info(f"Video validation passed: {file_path}")
            
        except Exception as e:
            result['error'] = f"Validation error: {str(e)}"
            logger.error(f"Video validation failed for {file_path}: {e}")
        
        return result
    
    @staticmethod
    def validate_text_file(file_path: str) -> Dict[str, Any]:
        """Validate text file"""
        result = {
            'valid': False,
            'error': None,
            'content': None,
            'warnings': []
        }
        
        try:
            path = Path(file_path)
            
            if not path.exists():
                result['error'] = "File does not exist"
                return result
            
            if path.suffix.lower() not in FileValidator.ALLOWED_TEXT_EXTENSIONS:
                result['error'] = f"Invalid extension. Allowed: {FileValidator.ALLOWED_TEXT_EXTENSIONS}"
                return result
            
            # Read and validate content
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                result['error'] = "Empty file"
                return result
            
            if len(content) > 10000:  # 10KB text limit
                result['warnings'].append("Very long text, might cause TTS issues")
            
            # Check for non-printable characters
            if any(ord(char) < 32 and char not in '\n\r\t' for char in content):
                result['warnings'].append("Contains non-printable characters")
            
            result['content'] = content
            result['valid'] = True
            
        except UnicodeDecodeError:
            result['error'] = "File encoding not supported (use UTF-8)"
        except Exception as e:
            result['error'] = f"Validation error: {str(e)}"
        
        return result

class TempFileManager:
    """Temporary file management and cleanup"""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.created_files = set()
        self.created_dirs = set()
    
    def create_temp_file(self, suffix: str = "", prefix: str = "cf_") -> str:
        """Create temporary file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}{timestamp}{suffix}"
        temp_path = self.temp_dir / filename
        
        # Ensure unique filename
        counter = 1
        while temp_path.exists():
            filename = f"{prefix}{timestamp}_{counter}{suffix}"
            temp_path = self.temp_dir / filename
            counter += 1
        
        # Create empty file
        temp_path.touch()
        self.created_files.add(str(temp_path))
        logger.debug(f"Created temp file: {temp_path}")
        return str(temp_path)
    
    def create_temp_dir(self, prefix: str = "cf_dir_") -> str:
        """Create temporary directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dirname = f"{prefix}{timestamp}"
        temp_path = self.temp_dir / dirname
        
        counter = 1
        while temp_path.exists():
            dirname = f"{prefix}{timestamp}_{counter}"
            temp_path = self.temp_dir / dirname
            counter += 1
        
        temp_path.mkdir()
        self.created_dirs.add(str(temp_path))
        logger.debug(f"Created temp dir: {temp_path}")
        return str(temp_path)
    
    def cleanup_file(self, file_path: str) -> bool:
        """Remove specific temporary file"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                self.created_files.discard(str(path))
                logger.debug(f"Cleaned up temp file: {file_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")
        return False
    
    def cleanup_all(self) -> Dict[str, int]:
        """Clean up all tracked temporary files and directories"""
        stats = {'files_removed': 0, 'dirs_removed': 0, 'errors': 0}
        
        # Clean up files
        for file_path in list(self.created_files):
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    stats['files_removed'] += 1
                self.created_files.remove(file_path)
            except Exception as e:
                logger.error(f"Failed to cleanup file {file_path}: {e}")
                stats['errors'] += 1
        
        # Clean up directories
        for dir_path in list(self.created_dirs):
            try:
                path = Path(dir_path)
                if path.exists():
                    shutil.rmtree(path)
                    stats['dirs_removed'] += 1
                self.created_dirs.remove(dir_path)
            except Exception as e:
                logger.error(f"Failed to cleanup directory {dir_path}: {e}")
                stats['errors'] += 1
        
        # Clean up old temp files (older than 24 hours)
        self.cleanup_old_temp_files()
        
        logger.info(f"Temp cleanup: {stats}")
        return stats
    
    def cleanup_old_temp_files(self, max_age_hours: int = 24):
        """Remove old temporary files"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            removed_count = 0
            
            for temp_file in self.temp_dir.glob("cf_*"):
                try:
                    file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                        removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove old temp file {temp_file}: {e}")
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old temporary files")
                
        except Exception as e:
            logger.error(f"Error during old temp file cleanup: {e}")

def get_video_info(video_path: str) -> Optional[Dict[str, Any]]:
    """Get video information using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.error(f"ffprobe failed for {video_path}: {result.stderr}")
            return None
        
        data = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        format_info = data.get('format', {})
        
        return {
            'duration': float(format_info.get('duration', 0)),
            'size': int(format_info.get('size', 0)),
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'codec': video_stream.get('codec_name', 'unknown'),
            'fps': eval(video_stream.get('r_frame_rate', '0/1')),
            'bitrate': int(format_info.get('bit_rate', 0))
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timeout for {video_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe output: {e}")
    except Exception as e:
        logger.error(f"Error getting video info for {video_path}: {e}")
    
    return None

def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> Optional[str]:
    """Calculate file hash for integrity checking"""
    try:
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        return None

def check_disk_space(path: str, required_gb: float = 5.0) -> Dict[str, Any]:
    """Check available disk space"""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        
        return {
            'free_gb': round(free_gb, 2),
            'total_gb': round(total_gb, 2),
            'used_gb': round(used_gb, 2),
            'sufficient': free_gb >= required_gb,
            'usage_percent': round((used_gb / total_gb) * 100, 1)
        }
    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")
        return {'sufficient': False, 'error': str(e)}

def check_system_resources() -> Dict[str, Any]:
    """Check system resource availability"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # GPU check (if available)
        gpu_available = check_gpu_availability()
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_total_gb': round(memory_gb, 2),
            'memory_available_gb': round(memory_available_gb, 2),
            'memory_percent': memory.percent,
            'gpu_available': gpu_available,
            'healthy': (
                cpu_percent < 80 and 
                memory.percent < 85 and 
                memory_available_gb > 2
            )
        }
    except Exception as e:
        logger.error(f"Failed to check system resources: {e}")
        return {'healthy': False, 'error': str(e)}

def check_gpu_availability() -> bool:
    """Check if NVIDIA GPU is available for NVENC"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available"""
    dependencies = {
        'ffmpeg': False,
        'ffprobe': False,
        'edge-tts': False,
        'nvidia-smi': False
    }
    
    # Check command-line tools
    for tool in ['ffmpeg', 'ffprobe', 'nvidia-smi']:
        try:
            result = subprocess.run([tool, '-version'], 
                                  capture_output=True, text=True, timeout=10)
            dependencies[tool] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            dependencies[tool] = False
    
    # Check Python packages
    try:
        import edge_tts
        dependencies['edge-tts'] = True
    except ImportError:
        dependencies['edge-tts'] = False
    
    return dependencies

async def test_edge_tts() -> bool:
    """Test Edge-TTS functionality"""
    try:
        import edge_tts
        
        # Test with a simple phrase
        communicate = edge_tts.Communicate("Test", "en-US-JennyNeural")
        
        # Try to generate audio (we'll just check if it doesn't crash)
        audio_generator = communicate.stream()
        first_chunk = await audio_generator.__anext__()
        
        return first_chunk is not None
        
    except Exception as e:
        logger.error(f"Edge-TTS test failed: {e}")
        return False

def create_directory_structure() -> bool:
    """Create required directory structure"""
    directories = [
        "content/texts",
        "content/videos", 
        "content/output",
        "temp",
        "logs",
        "config"
    ]
    
    try:
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create language-specific output directories
        languages = ['en', 'es', 'pt', 'fr', 'de', 'ru']
        for lang in languages:
            Path(f"content/output/{lang}").mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create directory structure: {e}")
        return False

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.last_check = None
        self.check_interval = 300  # 5 minutes
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        self.last_check = datetime.now()
        
        health_report = {
            'timestamp': self.last_check.isoformat(),
            'overall_health': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        # Check dependencies
        deps = check_dependencies()
        health_report['checks']['dependencies'] = deps
        
        missing_deps = [dep for dep, available in deps.items() if not available]
        if missing_deps:
            health_report['errors'].extend([f"Missing dependency: {dep}" for dep in missing_deps])
            health_report['overall_health'] = False
        
        # Check system resources
        resources = check_system_resources()
        health_report['checks']['system_resources'] = resources
        
        if not resources.get('healthy', False):
            health_report['warnings'].append("System resources under stress")
        
        # Check disk space
        disk_info = check_disk_space(".")
        health_report['checks']['disk_space'] = disk_info
        
        if not disk_info.get('sufficient', False):
            health_report['errors'].append("Insufficient disk space")
            health_report['overall_health'] = False
        
        # Test Edge-TTS
        try:
            tts_working = await test_edge_tts()
            health_report['checks']['edge_tts'] = tts_working
            
            if not tts_working:
                health_report['errors'].append("Edge-TTS not functioning")
                health_report['overall_health'] = False
                
        except Exception as e:
            health_report['checks']['edge_tts'] = False
            health_report['errors'].append(f"Edge-TTS test failed: {e}")
            health_report['overall_health'] = False
        
        # Check directory structure
        dirs_ok = create_directory_structure()
        health_report['checks']['directory_structure'] = dirs_ok
        
        if not dirs_ok:
            health_report['errors'].append("Failed to create directory structure")
            health_report['overall_health'] = False
        
        logger.info(f"Health check completed. Overall health: {health_report['overall_health']}")
        
        return health_report

# Global instances
temp_manager = TempFileManager()
health_checker = HealthChecker()

def emergency_cleanup():
    """Emergency cleanup function"""
    try:
        logger.warning("Performing emergency cleanup...")
        temp_manager.cleanup_all()
        logger.info("Emergency cleanup completed")
    except Exception as e:
        logger.error(f"Emergency cleanup failed: {e}")

# Error handling decorators
def handle_errors(fallback_value=None):
    """Decorator for error handling"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return fallback_value
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return fallback_value
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

if __name__ == "__main__":
    # Test utilities
    async def test_utils():
        print("Testing Content Factory Utils...")
        
        # Test health check
        health_report = await health_checker.comprehensive_health_check()
        print(f"System Health: {health_report['overall_health']}")
        
        if health_report['errors']:
            print("Errors found:")
            for error in health_report['errors']:
                print(f"  - {error}")
        
        if health_report['warnings']:
            print("Warnings:")
            for warning in health_report['warnings']:
                print(f"  - {warning}")
    
    asyncio.run(test_utils())