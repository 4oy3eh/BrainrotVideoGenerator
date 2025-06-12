import asyncio
import logging
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiofiles
import edge_tts
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TTSJob:
    """Single TTS generation job"""
    text: str
    language: str
    voice: str
    output_path: str
    job_id: str
    priority: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class TTSResult:
    """TTS generation result"""
    job_id: str
    language: str
    success: bool
    output_path: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    file_size: Optional[int] = None

class TTSCache:
    """Simple file-based cache for TTS results"""
    
    def __init__(self, cache_dir: str, max_age_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
        self._cache_index = {}
        self._load_cache_index()
    
    def _get_cache_key(self, text: str, voice: str) -> str:
        """Generate cache key from text and voice"""
        content = f"{text}-{voice}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, info in data.items():
                        info['created_at'] = datetime.fromisoformat(info['created_at'])
                    self._cache_index = data
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            data = {}
            for key, info in self._cache_index.items():
                data[key] = {
                    'file_path': info['file_path'],
                    'created_at': info['created_at'].isoformat(),
                    'file_size': info['file_size']
                }
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def get(self, text: str, voice: str) -> Optional[str]:
        """Get cached audio file path if exists and valid"""
        cache_key = self._get_cache_key(text, voice)
        
        if cache_key not in self._cache_index:
            return None
        
        cache_info = self._cache_index[cache_key]
        file_path = Path(cache_info['file_path'])
        
        # Check if file exists and is not too old
        if not file_path.exists():
            del self._cache_index[cache_key]
            self._save_cache_index()
            return None
        
        if datetime.now() - cache_info['created_at'] > self.max_age:
            try:
                file_path.unlink()
                del self._cache_index[cache_key]
                self._save_cache_index()
            except:
                pass
            return None
        
        return str(file_path)
    
    def put(self, text: str, voice: str, file_path: str):
        """Cache audio file"""
        cache_key = self._get_cache_key(text, voice)
        file_path_obj = Path(file_path)
        
        if file_path_obj.exists():
            self._cache_index[cache_key] = {
                'file_path': str(file_path_obj),
                'created_at': datetime.now(),
                'file_size': file_path_obj.stat().st_size
            }
            self._save_cache_index()
    
    def cleanup_old_entries(self):
        """Remove old cache entries"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, info in self._cache_index.items():
            if current_time - info['created_at'] > self.max_age:
                keys_to_remove.append(key)
                try:
                    Path(info['file_path']).unlink(missing_ok=True)
                except:
                    pass
        
        for key in keys_to_remove:
            del self._cache_index[key]
        
        if keys_to_remove:
            self._save_cache_index()
            logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")

class ProgressTracker:
    """Thread-safe progress tracking for UI updates"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._progress = {}
        self._callbacks = []
    
    def add_callback(self, callback: Callable):
        """Add progress update callback"""
        with self._lock:
            self._callbacks.append(callback)
    
    def update_job_progress(self, job_id: str, language: str, status: str, 
                          progress: float = 0.0, error: str = None):
        """Update progress for specific job"""
        with self._lock:
            if job_id not in self._progress:
                self._progress[job_id] = {}
            
            self._progress[job_id][language] = {
                'status': status,
                'progress': progress,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(job_id, self._progress[job_id])
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
    
    def get_job_progress(self, job_id: str) -> Dict:
        """Get progress for specific job"""
        with self._lock:
            return self._progress.get(job_id, {}).copy()
    
    def remove_job(self, job_id: str):
        """Remove job from progress tracking"""
        with self._lock:
            self._progress.pop(job_id, None)

class TTSProcessor:
    """High-performance TTS processor optimized for Intel 8700K"""
    
    def __init__(self, 
                 languages_config: Dict[str, str],
                 output_dir: str = "temp/tts",
                 cache_dir: str = "temp/cache",
                 max_concurrent_jobs: int = 6,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        
        self.languages = languages_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache = TTSCache(cache_dir)
        
        # Progress tracking
        self.progress_tracker = ProgressTracker()
        
        # Concurrency settings optimized for 8700K (6 cores, 12 threads)
        self.max_concurrent_jobs = max_concurrent_jobs
        self.semaphore = asyncio.Semaphore(max_concurrent_jobs)
        
        # Retry settings
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Statistics
        self.stats = {
            'total_generated': 0,
            'cache_hits': 0,
            'total_errors': 0,
            'avg_generation_time': 0.0
        }
        
        logger.info(f"TTS Processor initialized with {max_concurrent_jobs} concurrent jobs")
        logger.info(f"Supported languages: {list(self.languages.keys())}")
    
    async def generate_single_tts(self, job: TTSJob) -> TTSResult:
        """Generate TTS for single language with retry logic"""
        
        # Check cache first
        cached_path = self.cache.get(job.text, job.voice)
        if cached_path:
            self.stats['cache_hits'] += 1
            self.progress_tracker.update_job_progress(
                job.job_id, job.language, "completed", 100.0
            )
            logger.info(f"Cache hit for {job.language}: {job.job_id}")
            return TTSResult(
                job_id=job.job_id,
                language=job.language,
                success=True,
                output_path=cached_path,
                duration=0.0
            )
        
        # Generate new TTS
        for attempt in range(self.max_retries + 1):
            try:
                self.progress_tracker.update_job_progress(
                    job.job_id, job.language, "generating", 
                    (attempt / (self.max_retries + 1)) * 50
                )
                
                start_time = time.time()
                
                # Create Edge-TTS communicate object
                communicate = edge_tts.Communicate(job.text, job.voice)
                
                # Generate audio data
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                
                # Save to file
                os.makedirs(os.path.dirname(job.output_path), exist_ok=True)
                async with aiofiles.open(job.output_path, "wb") as f:
                    await f.write(audio_data)
                
                generation_time = time.time() - start_time
                file_size = len(audio_data)
                
                # Cache the result
                self.cache.put(job.text, job.voice, job.output_path)
                
                # Update statistics
                self.stats['total_generated'] += 1
                self.stats['avg_generation_time'] = (
                    (self.stats['avg_generation_time'] * (self.stats['total_generated'] - 1) + 
                     generation_time) / self.stats['total_generated']
                )
                
                self.progress_tracker.update_job_progress(
                    job.job_id, job.language, "completed", 100.0
                )
                
                logger.info(f"Generated TTS {job.language}: {job.job_id} in {generation_time:.2f}s")
                
                return TTSResult(
                    job_id=job.job_id,
                    language=job.language,
                    success=True,
                    output_path=job.output_path,
                    duration=generation_time,
                    file_size=file_size
                )
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
                logger.warning(f"TTS generation error {job.language}: {error_msg}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    # Final failure
                    self.stats['total_errors'] += 1
                    self.progress_tracker.update_job_progress(
                        job.job_id, job.language, "failed", 0.0, error_msg
                    )
                    
                    return TTSResult(
                        job_id=job.job_id,
                        language=job.language,
                        success=False,
                        error=error_msg
                    )
    
    async def generate_multilingual_tts(self, 
                                      text: str, 
                                      job_id: str,
                                      languages: Optional[List[str]] = None,
                                      output_base_path: Optional[str] = None) -> Dict[str, TTSResult]:
        """Generate TTS for multiple languages concurrently"""
        
        if languages is None:
            languages = list(self.languages.keys())
        
        # Validate languages
        invalid_langs = set(languages) - set(self.languages.keys())
        if invalid_langs:
            raise ValueError(f"Unsupported languages: {invalid_langs}")
        
        # Create jobs
        jobs = []
        for lang in languages:
            if output_base_path:
                output_path = f"{output_base_path}_{lang}.wav"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.output_dir / f"{job_id}_{lang}_{timestamp}.wav"
            
            job = TTSJob(
                text=text,
                language=lang,
                voice=self.languages[lang],
                output_path=str(output_path),
                job_id=job_id
            )
            jobs.append(job)
        
        logger.info(f"Starting multilingual TTS generation: {job_id} ({len(jobs)} languages)")
        
        # Process jobs concurrently with semaphore control
        async def process_job(job):
            async with self.semaphore:
                return await self.generate_single_tts(job)
        
        # Execute all jobs concurrently
        start_time = time.time()
        results = await asyncio.gather(*[process_job(job) for job in jobs])
        total_time = time.time() - start_time
        
        # Organize results by language
        results_dict = {result.language: result for result in results}
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        logger.info(f"Completed multilingual TTS {job_id}: {successful} success, {failed} failed in {total_time:.2f}s")
        
        return results_dict
    
    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates"""
        self.progress_tracker.add_callback(callback)
    
    def get_job_progress(self, job_id: str) -> Dict:
        """Get current progress for job"""
        return self.progress_tracker.get_job_progress(job_id)
    
    def get_statistics(self) -> Dict:
        """Get processor statistics"""
        return self.stats.copy()
    
    def cleanup_old_cache(self):
        """Clean up old cache entries"""
        self.cache.cleanup_old_entries()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of TTS system"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'languages_available': len(self.languages),
            'cache_entries': len(self.cache._cache_index),
            'statistics': self.get_statistics()
        }
        
        # Test TTS generation with short text
        try:
            test_job_id = f"health_check_{int(time.time())}"
            test_results = await self.generate_multilingual_tts(
                "Test", test_job_id, ["en"]
            )
            
            if test_results["en"].success:
                health_status['tts_test'] = 'passed'
                # Clean up test file
                try:
                    os.unlink(test_results["en"].output_path)
                except:
                    pass
            else:
                health_status['status'] = 'degraded'
                health_status['tts_test'] = 'failed'
                health_status['tts_error'] = test_results["en"].error
                
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['tts_test'] = 'error'
            health_status['tts_error'] = str(e)
        
        return health_status

# Example usage and testing
async def main():
    """Example usage of TTS processor"""
    
    # Load languages config
    languages_config = {
        "en": "en-US-JennyNeural",
        "es": "es-MX-DaliaNeural",
        "pt": "pt-BR-FranciscaNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
        "ru": "ru-RU-SvetlanaNeural"
    }
    
    # Initialize processor
    processor = TTSProcessor(languages_config)
    
    # Add progress callback
    def progress_callback(job_id, progress):
        print(f"Progress update for {job_id}: {progress}")
    
    processor.add_progress_callback(progress_callback)
    
    # Test text
    test_text = "Hello everyone! This is a test of our TTS system for creating viral content."
    
    # Generate TTS for all languages
    results = await processor.generate_multilingual_tts(test_text, "test_job_001")
    
    # Print results
    for lang, result in results.items():
        if result.success:
            print(f"✅ {lang}: {result.output_path} ({result.file_size} bytes, {result.duration:.2f}s)")
        else:
            print(f"❌ {lang}: {result.error}")
    
    # Print statistics
    print(f"Statistics: {processor.get_statistics()}")
    
    # Health check
    health = await processor.health_check()
    print(f"Health status: {health}")

if __name__ == "__main__":
    asyncio.run(main())