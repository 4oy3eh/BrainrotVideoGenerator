import asyncio
import subprocess
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import time
import re

logger = logging.getLogger(__name__)

@dataclass
class VideoProcessingJob:
    """Video processing job configuration"""
    job_id: str
    video_files: List[str]
    audio_file: str
    output_path: str
    language: str
    progress_callback: Optional[Callable] = None

@dataclass
class ProcessingResult:
    """Video processing result"""
    success: bool
    output_path: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    error: Optional[str] = None

class NVENCProcessor:
    """
    NVENC-optimized video processor for TikTok content generation
    Handles video concatenation, audio overlay, and GPU-accelerated encoding
    """
    
    def __init__(self, max_concurrent_jobs: int = 2):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs = 0
        self.job_semaphore = asyncio.Semaphore(max_concurrent_jobs)
        
        # TikTok-optimized encoding settings
        self.encoding_params = {
            # NVENC H.264 settings for GTX 1070
            'video_codec': 'h264_nvenc',
            'preset': 'p4',  # Medium preset for GTX 1070
            'profile': 'high',
            'level': '4.1',
            'rc': 'vbr',  # Variable bitrate
            'cq': '23',   # Quality level (lower = better quality)
            'maxrate': '2M',  # Max bitrate for TikTok
            'bufsize': '4M',  # Buffer size
            'spatial_aq': '1',  # Spatial AQ for better quality
            'temporal_aq': '1', # Temporal AQ
            'b_ref_mode': 'middle',  # B-frame reference mode
            'bf': '3',    # B-frames
            
            # Audio settings
            'audio_codec': 'aac',
            'audio_bitrate': '128k',
            'audio_sample_rate': '44100',
            
            # TikTok format
            'width': 720,
            'height': 1280,
            'fps': 30,
            'pixel_format': 'yuv420p'
        }
        
        # Progress tracking regex patterns
        self.duration_pattern = re.compile(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})')
        self.progress_pattern = re.compile(r'time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})')
        
        self._validate_nvenc_support()
    
    def _validate_nvenc_support(self) -> None:
        """Validate NVENC support and FFmpeg availability"""
        try:
            # Check FFmpeg availability
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("FFmpeg not found")
            
            # Check NVENC encoder availability
            result = subprocess.run(['ffmpeg', '-encoders'], 
                                  capture_output=True, text=True, timeout=10)
            if 'h264_nvenc' not in result.stdout:
                raise RuntimeError("NVENC encoder not available")
                
            logger.info("NVENC processor initialized successfully")
            
        except Exception as e:
            logger.error(f"NVENC validation failed: {e}")
            raise RuntimeError(f"NVENC not available: {e}")
    
    async def process_video(self, job: VideoProcessingJob) -> ProcessingResult:
        """
        Process video with NVENC acceleration
        
        Args:
            job: Video processing job configuration
            
        Returns:
            ProcessingResult with success status and details
        """
        async with self.job_semaphore:
            self.active_jobs += 1
            start_time = time.time()
            
            try:
                logger.info(f"Starting video processing job {job.job_id}")
                
                # Validate input files
                await self._validate_input_files(job)
                
                # Create output directory
                os.makedirs(os.path.dirname(job.output_path), exist_ok=True)
                
                # Build FFmpeg command
                ffmpeg_cmd = await self._build_ffmpeg_command(job)
                
                # Execute FFmpeg with progress tracking
                success = await self._execute_ffmpeg(ffmpeg_cmd, job.progress_callback)
                
                if success and os.path.exists(job.output_path):
                    duration = time.time() - start_time
                    file_size = os.path.getsize(job.output_path)
                    
                    logger.info(f"Job {job.job_id} completed successfully in {duration:.2f}s")
                    
                    return ProcessingResult(
                        success=True,
                        output_path=job.output_path,
                        duration=duration,
                        file_size=file_size
                    )
                else:
                    return ProcessingResult(
                        success=False,
                        error="FFmpeg processing failed"
                    )
                    
            except Exception as e:
                logger.error(f"Job {job.job_id} failed: {e}")
                return ProcessingResult(
                    success=False,
                    error=str(e)
                )
            finally:
                self.active_jobs -= 1
    
    async def _validate_input_files(self, job: VideoProcessingJob) -> None:
        """Validate all input files exist and are accessible"""
        for video_file in job.video_files:
            if not os.path.exists(video_file):
                raise FileNotFoundError(f"Video file not found: {video_file}")
        
        if not os.path.exists(job.audio_file):
            raise FileNotFoundError(f"Audio file not found: {job.audio_file}")
    
    async def _build_ffmpeg_command(self, job: VideoProcessingJob) -> List[str]:
        """
        Build optimized FFmpeg command for NVENC processing
        
        Args:
            job: Video processing job
            
        Returns:
            FFmpeg command as list of arguments
        """
        cmd = ['ffmpeg', '-y']  # -y to overwrite output
        
        # GPU acceleration
        cmd.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
        
        # Input videos - concatenate multiple videos
        if len(job.video_files) == 1:
            cmd.extend(['-i', job.video_files[0]])
        else:
            # Create concat filter for multiple videos
            concat_filter = self._create_concat_filter(job.video_files)
            for video_file in job.video_files:
                cmd.extend(['-i', video_file])
            cmd.extend(['-filter_complex', concat_filter])
        
        # Input audio
        cmd.extend(['-i', job.audio_file])
        
        # Video encoding with NVENC
        cmd.extend([
            '-c:v', self.encoding_params['video_codec'],
            '-preset', self.encoding_params['preset'],
            '-profile:v', self.encoding_params['profile'],
            '-level:v', self.encoding_params['level'],
            '-rc:v', self.encoding_params['rc'],
            '-cq:v', self.encoding_params['cq'],
            '-maxrate:v', self.encoding_params['maxrate'],
            '-bufsize:v', self.encoding_params['bufsize'],
            '-spatial_aq:v', self.encoding_params['spatial_aq'],
            '-temporal_aq:v', self.encoding_params['temporal_aq'],
            '-b_ref_mode:v', self.encoding_params['b_ref_mode'],
            '-bf:v', self.encoding_params['bf']
        ])
        
        # Audio encoding
        cmd.extend([
            '-c:a', self.encoding_params['audio_codec'],
            '-b:a', self.encoding_params['audio_bitrate'],
            '-ar', self.encoding_params['audio_sample_rate']
        ])
        
        # Video format and scaling for TikTok
        video_filters = []
        
        # Scale to TikTok format (720x1280)
        video_filters.append(f"scale={self.encoding_params['width']}:{self.encoding_params['height']}:force_original_aspect_ratio=decrease")
        video_filters.append(f"pad={self.encoding_params['width']}:{self.encoding_params['height']}:(ow-iw)/2:(oh-ih)/2:black")
        
        # Set framerate
        video_filters.append(f"fps={self.encoding_params['fps']}")
        
        if len(job.video_files) == 1:
            cmd.extend(['-vf', ','.join(video_filters)])
        
        # Pixel format
        cmd.extend(['-pix_fmt', self.encoding_params['pixel_format']])
        
        # Audio synchronization
        cmd.extend(['-map', '0:v:0', '-map', f"{len(job.video_files)}:a:0"])
        cmd.extend(['-shortest'])  # End when shortest stream ends
        
        # Threading and performance
        cmd.extend(['-threads', '0'])  # Use all available threads
        
        # Output file
        cmd.append(job.output_path)
        
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        return cmd
    
    def _create_concat_filter(self, video_files: List[str]) -> str:
        """Create FFmpeg concat filter for multiple videos"""
        inputs = ''.join(f'[{i}:v:0]' for i in range(len(video_files)))
        return f'{inputs}concat=n={len(video_files)}:v=1:a=0[outv]'
    
    async def _execute_ffmpeg(self, cmd: List[str], progress_callback: Optional[Callable] = None) -> bool:
        """
        Execute FFmpeg command with progress tracking
        
        Args:
            cmd: FFmpeg command
            progress_callback: Optional progress callback function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )
            
            total_duration = None
            
            # Read stderr for progress information
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                
                line_str = line.decode('utf-8', errors='ignore').strip()
                
                # Parse total duration
                if total_duration is None:
                    duration_match = self.duration_pattern.search(line_str)
                    if duration_match:
                        h, m, s, cs = map(int, duration_match.groups())
                        total_duration = h * 3600 + m * 60 + s + cs / 100
                
                # Parse current progress
                if total_duration and progress_callback:
                    progress_match = self.progress_pattern.search(line_str)
                    if progress_match:
                        h, m, s, cs = map(int, progress_match.groups())
                        current_time = h * 3600 + m * 60 + s + cs / 100
                        progress = min(current_time / total_duration * 100, 100)
                        
                        try:
                            await progress_callback(progress)
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}")
                
                # Log important messages
                if any(keyword in line_str.lower() for keyword in ['error', 'failed', 'invalid']):
                    logger.warning(f"FFmpeg warning/error: {line_str}")
            
            # Wait for process completion
            await process.wait()
            
            if process.returncode == 0:
                logger.info("FFmpeg processing completed successfully")
                return True
            else:
                # Read any remaining stderr
                stderr_output = await process.stderr.read()
                logger.error(f"FFmpeg failed with return code {process.returncode}")
                logger.error(f"FFmpeg stderr: {stderr_output.decode('utf-8', errors='ignore')}")
                return False
                
        except Exception as e:
            logger.error(f"FFmpeg execution error: {e}")
            return False
    
    async def get_video_info(self, video_path: str) -> Dict:
        """
        Get video information using FFprobe
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return json.loads(stdout.decode('utf-8'))
            else:
                logger.error(f"FFprobe failed: {stderr.decode('utf-8')}")
                return {}
                
        except Exception as e:
            logger.error(f"Video info extraction error: {e}")
            return {}
    
    async def cleanup_temp_files(self, temp_files: List[str]) -> None:
        """Clean up temporary files"""
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")
    
    def get_processing_stats(self) -> Dict:
        """Get current processing statistics"""
        return {
            'active_jobs': self.active_jobs,
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'available_slots': self.max_concurrent_jobs - self.active_jobs
        }
    
    async def process_batch(self, jobs: List[VideoProcessingJob]) -> List[ProcessingResult]:
        """
        Process multiple video jobs concurrently
        
        Args:
            jobs: List of video processing jobs
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting batch processing of {len(jobs)} jobs")
        
        # Process jobs concurrently with semaphore limiting
        tasks = [self.process_video(job) for job in jobs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Job {jobs[i].job_id} failed with exception: {result}")
                processed_results.append(ProcessingResult(
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.success)
        logger.info(f"Batch processing completed: {successful}/{len(jobs)} successful")
        
        return processed_results

# Utility functions
async def create_test_job(video_files: List[str], audio_file: str, output_path: str, language: str = 'en') -> VideoProcessingJob:
    """Create a test video processing job"""
    return VideoProcessingJob(
        job_id=f"test_{int(time.time())}",
        video_files=video_files,
        audio_file=audio_file,
        output_path=output_path,
        language=language
    )

# Example usage
async def main():
    """Example usage of NVENC processor"""
    processor = NVENCProcessor(max_concurrent_jobs=2)
    
    # Create test job
    job = await create_test_job(
        video_files=['content/videos/clip1.mp4', 'content/videos/clip2.mp4'],
        audio_file='temp/audio_en.wav',
        output_path='content/output/en/final_video.mp4',
        language='en'
    )
    
    # Progress callback
    async def progress_callback(progress: float):
        print(f"Progress: {progress:.1f}%")
    
    job.progress_callback = progress_callback
    
    # Process video
    result = await processor.process_video(job)
    
    if result.success:
        print(f"Video processed successfully: {result.output_path}")
        print(f"File size: {result.file_size / 1024 / 1024:.2f} MB")
    else:
        print(f"Processing failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())