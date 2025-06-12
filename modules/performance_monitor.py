import asyncio
import json
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import GPUtil
from fastapi import WebSocket
import subprocess
import threading
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics data structure"""
    timestamp: str
    cpu_usage: float
    cpu_temp: Optional[float]
    cpu_threads: List[float]
    memory_used: float
    memory_total: float
    memory_percent: float
    gpu_usage: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_memory_percent: float
    gpu_temperature: float
    nvenc_sessions: int
    nvenc_capacity: int
    videos_processed: int
    processing_speed: float  # videos per hour
    queue_size: int
    eta_minutes: Optional[int]
    disk_usage: float

@dataclass
class ProcessingStats:
    """Processing statistics tracking"""
    total_processed: int = 0
    start_time: float = 0
    last_reset: float = 0
    processing_times: deque = None
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = deque(maxlen=100)  # Keep last 100 processing times

class PerformanceMonitor:
    """
    Real-time system performance monitoring for brainrot content factory
    Optimized for Intel 8700K + GTX 1070 + NVENC hardware setup
    """
    
    def __init__(self):
        self.stats = ProcessingStats()
        self.stats.start_time = time.time()
        self.stats.last_reset = time.time()
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        
        # Monitoring intervals
        self.update_interval = 1.0  # 1 second
        self.metrics_history = deque(maxlen=3600)  # Keep 1 hour of history
        
        # NVENC monitoring
        self.max_nvenc_sessions = 2  # GTX 1070 supports max 2 concurrent NVENC sessions
        
        # Performance thresholds
        self.cpu_warning_threshold = 85.0
        self.gpu_warning_threshold = 85.0
        self.memory_warning_threshold = 85.0
        self.temp_warning_threshold = 80.0
        
        self._monitoring_task = None
        self._running = False
        
    async def start_monitoring(self):
        """Start the performance monitoring loop"""
        if self._running:
            logger.warning("Performance monitoring already running")
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
        
    async def stop_monitoring(self):
        """Stop the performance monitoring loop"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
        
    async def add_websocket(self, websocket: WebSocket):
        """Add WebSocket connection for real-time updates"""
        self.websocket_connections.append(websocket)
        logger.info(f"WebSocket connection added. Total connections: {len(self.websocket_connections)}")
        
    async def remove_websocket(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
        logger.info(f"WebSocket connection removed. Total connections: {len(self.websocket_connections)}")
        
    def record_video_processed(self, processing_time: float):
        """Record a completed video processing operation"""
        self.stats.total_processed += 1
        self.stats.processing_times.append(processing_time)
        logger.debug(f"Video processed in {processing_time:.2f}s. Total: {self.stats.total_processed}")
        
    def update_queue_size(self, queue_size: int):
        """Update current queue size"""
        self.current_queue_size = queue_size
        
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats.total_processed = 0
        self.stats.last_reset = time.time()
        self.stats.processing_times.clear()
        logger.info("Processing statistics reset")
        
    def get_cpu_usage(self) -> Dict[str, Any]:
        """Get detailed CPU usage information for 8700K (6 cores, 12 threads)"""
        try:
            # Overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Per-thread usage (8700K has 12 threads)
            cpu_per_thread = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # CPU temperature (if available)
            cpu_temp = None
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temp = max([temp.current for temp in temps['coretemp']])
                elif 'cpu_thermal' in temps:
                    cpu_temp = temps['cpu_thermal'][0].current
            except (AttributeError, KeyError):
                pass
                
            return {
                'usage': cpu_percent,
                'threads': cpu_per_thread,
                'temperature': cpu_temp,
                'cores': psutil.cpu_count(logical=False),
                'threads_total': psutil.cpu_count(logical=True)
            }
        except Exception as e:
            logger.error(f"Error getting CPU usage: {e}")
            return {'usage': 0, 'threads': [], 'temperature': None, 'cores': 6, 'threads_total': 12}
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get system memory usage"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'used': memory.used,
                'available': memory.available,
                'percent': memory.percent
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {'total': 0, 'used': 0, 'available': 0, 'percent': 0}
            
    def get_gpu_usage(self) -> Dict[str, Any]:
        """Get GTX 1070 GPU usage and temperature"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {
                    'usage': 0,
                    'memory_used': 0,
                    'memory_total': 8192,  # GTX 1070 has 8GB VRAM
                    'memory_percent': 0,
                    'temperature': 0
                }
                
            gpu = gpus[0]  # Assuming single GPU setup
            return {
                'usage': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'temperature': gpu.temperature
            }
        except Exception as e:
            logger.error(f"Error getting GPU usage: {e}")
            return {
                'usage': 0,
                'memory_used': 0,
                'memory_total': 8192,
                'memory_percent': 0,
                'temperature': 0
            }
            
    def get_nvenc_sessions(self) -> int:
        """Get current NVENC encoding sessions count"""
        try:
            # Check running FFmpeg processes with NVENC
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq ffmpeg.exe', '/FO', 'CSV'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                # Subtract 1 for header line, count actual processes
                nvenc_processes = max(0, len(lines) - 1) if len(lines) > 1 else 0
                return min(nvenc_processes, self.max_nvenc_sessions)
            return 0
        except Exception as e:
            logger.error(f"Error checking NVENC sessions: {e}")
            return 0
            
    def get_disk_usage(self) -> float:
        """Get disk usage percentage for the main drive"""
        try:
            disk = psutil.disk_usage('/')
            return (disk.used / disk.total) * 100
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return 0
            
    def calculate_processing_speed(self) -> float:
        """Calculate videos processed per hour"""
        if not self.stats.processing_times:
            return 0.0
            
        current_time = time.time()
        time_elapsed = current_time - self.stats.last_reset
        
        if time_elapsed < 60:  # Less than 1 minute, return 0
            return 0.0
            
        # Calculate based on recent processing times
        recent_times = list(self.stats.processing_times)[-50:]  # Last 50 videos
        if not recent_times:
            return 0.0
            
        avg_processing_time = sum(recent_times) / len(recent_times)
        videos_per_hour = 3600 / avg_processing_time if avg_processing_time > 0 else 0
        
        return videos_per_hour
        
    def calculate_eta(self, queue_size: int, processing_speed: float) -> Optional[int]:
        """Calculate ETA in minutes for queue completion"""
        if queue_size <= 0 or processing_speed <= 0:
            return None
            
        hours_remaining = queue_size / processing_speed
        minutes_remaining = int(hours_remaining * 60)
        
        return minutes_remaining
        
    async def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            cpu_info = self.get_cpu_usage()
            memory_info = self.get_memory_usage()
            gpu_info = self.get_gpu_usage()
            nvenc_sessions = self.get_nvenc_sessions()
            disk_usage = self.get_disk_usage()
            
            processing_speed = self.calculate_processing_speed()
            queue_size = getattr(self, 'current_queue_size', 0)
            eta = self.calculate_eta(queue_size, processing_speed)
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage=cpu_info['usage'],
                cpu_temp=cpu_info['temperature'],
                cpu_threads=cpu_info['threads'],
                memory_used=memory_info['used'],
                memory_total=memory_info['total'],
                memory_percent=memory_info['percent'],
                gpu_usage=gpu_info['usage'],
                gpu_memory_used=gpu_info['memory_used'],
                gpu_memory_total=gpu_info['memory_total'],
                gpu_memory_percent=gpu_info['memory_percent'],
                gpu_temperature=gpu_info['temperature'],
                nvenc_sessions=nvenc_sessions,
                nvenc_capacity=self.max_nvenc_sessions,
                videos_processed=self.stats.total_processed,
                processing_speed=processing_speed,
                queue_size=queue_size,
                eta_minutes=eta,
                disk_usage=disk_usage
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return default metrics on error
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage=0, cpu_temp=None, cpu_threads=[],
                memory_used=0, memory_total=0, memory_percent=0,
                gpu_usage=0, gpu_memory_used=0, gpu_memory_total=8192,
                gpu_memory_percent=0, gpu_temperature=0,
                nvenc_sessions=0, nvenc_capacity=2,
                videos_processed=0, processing_speed=0,
                queue_size=0, eta_minutes=None, disk_usage=0
            )
            
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Performance monitoring loop started")
        
        while self._running:
            try:
                # Collect current metrics
                metrics = await self.get_current_metrics()
                
                # Store in history
                self.metrics_history.append(metrics)
                
                # Check for warnings
                await self._check_warnings(metrics)
                
                # Send to WebSocket clients
                await self._broadcast_metrics(metrics)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
                
        logger.info("Performance monitoring loop stopped")
        
    async def _check_warnings(self, metrics: SystemMetrics):
        """Check for performance warnings"""
        warnings = []
        
        if metrics.cpu_usage > self.cpu_warning_threshold:
            warnings.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            
        if metrics.gpu_usage > self.gpu_warning_threshold:
            warnings.append(f"High GPU usage: {metrics.gpu_usage:.1f}%")
            
        if metrics.memory_percent > self.memory_warning_threshold:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            
        if metrics.cpu_temp and metrics.cpu_temp > self.temp_warning_threshold:
            warnings.append(f"High CPU temperature: {metrics.cpu_temp:.1f}°C")
            
        if metrics.gpu_temperature > self.temp_warning_threshold:
            warnings.append(f"High GPU temperature: {metrics.gpu_temperature:.1f}°C")
            
        if metrics.nvenc_sessions >= self.max_nvenc_sessions:
            warnings.append("NVENC capacity maxed out")
            
        if warnings:
            logger.warning("Performance warnings: " + "; ".join(warnings))
            
    async def _broadcast_metrics(self, metrics: SystemMetrics):
        """Broadcast metrics to all WebSocket connections"""
        if not self.websocket_connections:
            return
            
        metrics_json = json.dumps(asdict(metrics))
        
        # Remove disconnected WebSockets
        active_connections = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(metrics_json)
                active_connections.append(websocket)
            except Exception as e:
                logger.debug(f"WebSocket connection lost: {e}")
                
        self.websocket_connections = active_connections
        
    def get_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics history for specified number of minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        filtered_metrics = []
        for metrics in self.metrics_history:
            metrics_time = datetime.fromisoformat(metrics.timestamp)
            if metrics_time >= cutoff_time:
                filtered_metrics.append(metrics)
                
        return filtered_metrics
        
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system summary information"""
        return {
            'hardware': {
                'cpu': 'Intel Core i7-8700K (6C/12T)',
                'gpu': 'NVIDIA GTX 1070 (8GB)',
                'nvenc_sessions': f"{self.get_nvenc_sessions()}/{self.max_nvenc_sessions}"
            },
            'performance': {
                'videos_processed': self.stats.total_processed,
                'uptime_hours': (time.time() - self.stats.start_time) / 3600,
                'processing_speed': self.calculate_processing_speed()
            },
            'thresholds': {
                'cpu_warning': self.cpu_warning_threshold,
                'gpu_warning': self.gpu_warning_threshold,
                'memory_warning': self.memory_warning_threshold,
                'temp_warning': self.temp_warning_threshold
            }
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

async def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return performance_monitor