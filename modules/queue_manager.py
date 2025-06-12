import asyncio
import json
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    TTS_PROCESSING = "tts_processing"
    VIDEO_PROCESSING = "video_processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class TaskData:
    """Task data structure"""
    task_id: str
    language: str
    text: str
    video_files: List[str]
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    output_file: Optional[str] = None
    estimated_duration: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary for JSON serialization"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data

class WorkerStats:
    """Worker statistics tracking"""
    def __init__(self, worker_type: str, worker_id: str):
        self.worker_type = worker_type
        self.worker_id = worker_id
        self.tasks_processed = 0
        self.total_processing_time = 0.0
        self.current_task = None
        self.is_busy = False
        self.last_activity = datetime.now()
    
    def start_task(self, task_id: str):
        self.current_task = task_id
        self.is_busy = True
        self.last_activity = datetime.now()
    
    def complete_task(self, processing_time: float):
        self.tasks_processed += 1
        self.total_processing_time += processing_time
        self.current_task = None
        self.is_busy = False
        self.last_activity = datetime.now()
    
    @property
    def average_processing_time(self) -> float:
        return self.total_processing_time / max(1, self.tasks_processed)

class QueueManager:
    """Main queue management system optimized for 8700K + GTX 1070"""
    
    def __init__(self, 
                 tts_workers: int = 6,  # One per language
                 video_workers: int = 2,  # NVENC dual encoder limit
                 websocket_callback: Optional[Callable] = None):
        
        # Hardware optimization settings
        self.tts_workers = tts_workers
        self.video_workers = video_workers
        self.websocket_callback = websocket_callback
        
        # Queue and task management
        self.tasks: Dict[str, TaskData] = {}
        self.pending_queue = asyncio.PriorityQueue()
        self.tts_queue = asyncio.Queue(maxsize=12)  # Buffer for TTS tasks
        self.video_queue = asyncio.Queue(maxsize=4)  # Buffer for video tasks
        
        # Worker management
        self.tts_worker_stats = [
            WorkerStats("TTS", f"tts_{i}") for i in range(tts_workers)
        ]
        self.video_worker_stats = [
            WorkerStats("VIDEO", f"video_{i}") for i in range(video_workers)
        ]
        
        # System state
        self.is_running = False
        self.is_paused = False
        self.worker_tasks = []
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.queue_start_time = None
        
        # Language configuration
        self.languages = {
            'en': 'en-US-JennyNeural',
            'es': 'es-MX-DaliaNeural',
            'pt': 'pt-BR-FranciscaNeural',
            'fr': 'fr-FR-DeniseNeural',
            'de': 'de-DE-KatjaNeural',
            'ru': 'ru-RU-SvetlanaNeural'
        }
    
    async def add_task(self, 
                      text: str, 
                      video_files: List[str],
                      language: str = None,
                      priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
        """Add task(s) to queue. If language is None, creates tasks for all languages"""
        
        task_ids = []
        languages_to_process = [language] if language else list(self.languages.keys())
        
        for lang in languages_to_process:
            if lang not in self.languages:
                logger.warning(f"Unsupported language: {lang}")
                continue
                
            task_id = str(uuid.uuid4())
            
            # Estimate processing duration
            estimated_duration = self._estimate_task_duration(text, video_files)
            
            task = TaskData(
                task_id=task_id,
                language=lang,
                text=text,
                video_files=video_files.copy(),
                priority=priority,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
                estimated_duration=estimated_duration
            )
            
            self.tasks[task_id] = task
            
            # Add to priority queue (negative priority for correct ordering)
            await self.pending_queue.put((-priority.value, time.time(), task_id))
            
            task_ids.append(task_id)
            
            logger.info(f"Task {task_id} added for language {lang}")
            await self._notify_websocket("task_added", task.to_dict())
        
        return task_ids
    
    def _estimate_task_duration(self, text: str, video_files: List[str]) -> float:
        """Estimate task processing duration based on text length and video count"""
        
        # Base estimates (in seconds)
        tts_time = len(text) * 0.1  # ~0.1s per character for TTS
        video_base_time = 15.0  # Base video processing time
        video_per_file_time = len(video_files) * 2.0  # Additional time per video file
        
        # Hardware-specific adjustments
        nvenc_speedup = 0.7  # NVENC is ~30% faster than CPU encoding
        
        return tts_time + (video_base_time + video_per_file_time) * nvenc_speedup
    
    async def start_processing(self):
        """Start the queue processing system"""
        if self.is_running:
            logger.warning("Queue is already running")
            return
        
        self.is_running = True
        self.is_paused = False
        self.queue_start_time = time.time()
        
        logger.info(f"Starting queue with {self.tts_workers} TTS workers and {self.video_workers} video workers")
        
        # Start worker coroutines
        self.worker_tasks = []
        
        # TTS workers (one per language for optimal distribution)
        for i in range(self.tts_workers):
            worker_task = asyncio.create_task(self._tts_worker(i))
            self.worker_tasks.append(worker_task)
        
        # Video workers (limited by NVENC)
        for i in range(self.video_workers):
            worker_task = asyncio.create_task(self._video_worker(i))
            self.worker_tasks.append(worker_task)
        
        # Queue distributor
        distributor_task = asyncio.create_task(self._queue_distributor())
        self.worker_tasks.append(distributor_task)
        
        # Progress monitor
        monitor_task = asyncio.create_task(self._progress_monitor())
        self.worker_tasks.append(monitor_task)
        
        await self._notify_websocket("queue_started", {
            "tts_workers": self.tts_workers,
            "video_workers": self.video_workers,
            "timestamp": datetime.now().isoformat()
        })
    
    async def stop_processing(self):
        """Stop the queue processing system"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping queue processing...")
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        await self._notify_websocket("queue_stopped", {
            "timestamp": datetime.now().isoformat()
        })
    
    async def pause_processing(self):
        """Pause queue processing"""
        self.is_paused = True
        logger.info("Queue processing paused")
        await self._notify_websocket("queue_paused", {
            "timestamp": datetime.now().isoformat()
        })
    
    async def resume_processing(self):
        """Resume queue processing"""
        self.is_paused = False
        logger.info("Queue processing resumed")
        await self._notify_websocket("queue_resumed", {
            "timestamp": datetime.now().isoformat()
        })
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        
        logger.info(f"Task {task_id} cancelled")
        await self._notify_websocket("task_cancelled", task.to_dict())
        
        return True
    
    async def _queue_distributor(self):
        """Distribute tasks from pending queue to worker queues"""
        while self.is_running:
            try:
                if self.is_paused:
                    await asyncio.sleep(1)
                    continue
                
                # Get next task from priority queue
                try:
                    priority, timestamp, task_id = await asyncio.wait_for(
                        self.pending_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                if task.status != TaskStatus.PENDING:
                    continue
                
                # Add to TTS queue
                await self.tts_queue.put(task_id)
                
            except Exception as e:
                logger.error(f"Error in queue distributor: {e}")
                await asyncio.sleep(1)
    
    async def _tts_worker(self, worker_id: int):
        """TTS processing worker"""
        worker_stats = self.tts_worker_stats[worker_id]
        
        while self.is_running:
            try:
                if self.is_paused:
                    await asyncio.sleep(1)
                    continue
                
                # Get task from TTS queue
                try:
                    task_id = await asyncio.wait_for(
                        self.tts_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                if task.status != TaskStatus.PENDING:
                    continue
                
                # Process TTS
                start_time = time.time()
                worker_stats.start_task(task_id)
                
                task.status = TaskStatus.TTS_PROCESSING
                task.started_at = datetime.now()
                task.progress = 0.1
                
                await self._notify_websocket("task_updated", task.to_dict())
                
                # Simulate TTS processing (replace with actual TTS call)
                success = await self._process_tts(task)
                
                processing_time = time.time() - start_time
                worker_stats.complete_task(processing_time)
                
                if success:
                    task.progress = 0.5
                    await self.video_queue.put(task_id)
                    await self._notify_websocket("task_updated", task.to_dict())
                else:
                    task.status = TaskStatus.FAILED
                    task.error_message = "TTS processing failed"
                    task.completed_at = datetime.now()
                    await self._notify_websocket("task_updated", task.to_dict())
                
            except Exception as e:
                logger.error(f"Error in TTS worker {worker_id}: {e}")
                if 'task' in locals():
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.completed_at = datetime.now()
                    await self._notify_websocket("task_updated", task.to_dict())
    
    async def _video_worker(self, worker_id: int):
        """Video processing worker (NVENC optimized)"""
        worker_stats = self.video_worker_stats[worker_id]
        
        while self.is_running:
            try:
                if self.is_paused:
                    await asyncio.sleep(1)
                    continue
                
                # Get task from video queue
                try:
                    task_id = await asyncio.wait_for(
                        self.video_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                if task.status != TaskStatus.TTS_PROCESSING:
                    continue
                
                # Process video
                start_time = time.time()
                worker_stats.start_task(task_id)
                
                task.status = TaskStatus.VIDEO_PROCESSING
                task.progress = 0.6
                
                await self._notify_websocket("task_updated", task.to_dict())
                
                # Simulate video processing with progress updates
                success = await self._process_video(task)
                
                processing_time = time.time() - start_time
                worker_stats.complete_task(processing_time)
                
                if success:
                    task.status = TaskStatus.COMPLETED
                    task.progress = 1.0
                    task.completed_at = datetime.now()
                    self.total_tasks_processed += 1
                else:
                    task.status = TaskStatus.FAILED
                    task.error_message = "Video processing failed"
                    task.completed_at = datetime.now()
                
                await self._notify_websocket("task_updated", task.to_dict())
                
            except Exception as e:
                logger.error(f"Error in video worker {worker_id}: {e}")
                if 'task' in locals():
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.completed_at = datetime.now()
                    await self._notify_websocket("task_updated", task.to_dict())
    
    async def _process_tts(self, task: TaskData) -> bool:
        
        
        # In real implementation, call TTS processor here
        from modules.tts_processor import process_tts
        return await process_tts(task.text, task.language)
        
        return True  # Simulate success
    
    async def _process_video(self, task: TaskData) -> bool:
                
       
        from modules.nvenc_processor import process_video
        return await process_video(task)
        
    
    async def _progress_monitor(self):
        """Monitor overall progress and send periodic updates"""
        while self.is_running:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                
                stats = self.get_queue_stats()
                await self._notify_websocket("queue_stats", stats)
                
            except Exception as e:
                logger.error(f"Error in progress monitor: {e}")
    
    async def _notify_websocket(self, event_type: str, data: Any):
        """Send WebSocket notification"""
        if self.websocket_callback:
            try:
                await self.websocket_callback({
                    "type": event_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"WebSocket notification error: {e}")
    
    def get_queue_stats(self) -> Dict:
        """Get comprehensive queue statistics"""
        pending_count = sum(1 for task in self.tasks.values() 
                          if task.status == TaskStatus.PENDING)
        processing_count = sum(1 for task in self.tasks.values() 
                             if task.status in [TaskStatus.TTS_PROCESSING, TaskStatus.VIDEO_PROCESSING])
        completed_count = sum(1 for task in self.tasks.values() 
                            if task.status == TaskStatus.COMPLETED)
        failed_count = sum(1 for task in self.tasks.values() 
                         if task.status == TaskStatus.FAILED)
        
        # Worker statistics
        tts_workers_busy = sum(1 for worker in self.tts_worker_stats if worker.is_busy)
        video_workers_busy = sum(1 for worker in self.video_worker_stats if worker.is_busy)
        
        # Performance metrics
        runtime = time.time() - self.queue_start_time if self.queue_start_time else 0
        tasks_per_hour = (completed_count / max(runtime / 3600, 0.001)) if runtime > 0 else 0
        
        return {
            "queue_status": {
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "runtime_seconds": runtime
            },
            "task_counts": {
                "pending": pending_count,
                "processing": processing_count,
                "completed": completed_count,
                "failed": failed_count,
                "total": len(self.tasks)
            },
            "worker_status": {
                "tts_workers": {
                    "total": self.tts_workers,
                    "busy": tts_workers_busy,
                    "idle": self.tts_workers - tts_workers_busy
                },
                "video_workers": {
                    "total": self.video_workers,
                    "busy": video_workers_busy,
                    "idle": self.video_workers - video_workers_busy
                }
            },
            "performance": {
                "tasks_per_hour": round(tasks_per_hour, 2),
                "estimated_daily_capacity": round(tasks_per_hour * 24, 0),
                "total_processed": self.total_tasks_processed
            },
            "queue_sizes": {
                "pending_queue": self.pending_queue.qsize(),
                "tts_queue": self.tts_queue.qsize(),
                "video_queue": self.video_queue.qsize()
            }
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of specific task"""
        if task_id not in self.tasks:
            return None
        return self.tasks[task_id].to_dict()
    
    def get_all_tasks(self) -> List[Dict]:
        """Get all tasks with their status"""
        return [task.to_dict() for task in self.tasks.values()]
    
    def clear_completed_tasks(self):
        """Remove completed and failed tasks from memory"""
        to_remove = [
            task_id for task_id, task in self.tasks.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        logger.info(f"Cleared {len(to_remove)} completed tasks")
        return len(to_remove)

# Usage example and testing
async def main():
    """Example usage of QueueManager"""
    
    # WebSocket callback example
    async def websocket_callback(message):
        print(f"WebSocket: {message['type']} - {message['data']}")
    
    # Initialize queue manager
    queue = QueueManager(websocket_callback=websocket_callback)
    
    # Start processing
    await queue.start_processing()
    
    # Add some test tasks
    task_ids = await queue.add_task(
        text="This is a test brainrot content for TikTok generation!",
        video_files=["video1.mp4", "video2.mp4", "video3.mp4"],
        priority=TaskPriority.HIGH
    )
    
    print(f"Added tasks: {task_ids}")
    
    # Monitor for a while
    await asyncio.sleep(30)
    
    # Print stats
    stats = queue.get_queue_stats()
    print(json.dumps(stats, indent=2))
    
    # Stop processing
    await queue.stop_processing()

if __name__ == "__main__":
    asyncio.run(main())