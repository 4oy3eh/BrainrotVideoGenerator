"""
Content Factory - Main FastAPI Application
TikTok mass production system with full integration
"""

import os
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
from modules.utils import (
    setup_logging, FileValidator, temp_manager, health_checker,
    handle_errors, get_video_info
)
from modules.tts_processor import TTSProcessor
from modules.nvenc_processor import NVENCProcessor
from modules.queue_manager import QueueManager
from modules.performance_monitor import PerformanceMonitor
from config.hardware_config import HardwareConfig

# Setup logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Content Factory",
    description="TikTok Mass Production System - 36 videos per day across 6 languages",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Initialize system components
hw_config = HardwareConfig()
tts_processor = TTSProcessor()
nvenc_processor = NVENCProcessor(hw_config)
queue_manager = QueueManager(max_concurrent=hw_config.MAX_CONCURRENT_JOBS)
performance_monitor = PerformanceMonitor()

# Load language configuration
with open("config/languages.json", "r") as f:
    LANGUAGES = json.load(f)

# Pydantic models
class ProductionJob(BaseModel):
    text: str
    languages: List[str]
    video_files: List[str]
    priority: int = 1
    output_format: str = "mp4"

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    created_at: datetime
    estimated_completion: Optional[datetime] = None
    results: Dict[str, Any] = {}
    error: Optional[str] = None

class SystemStats(BaseModel):
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    queue_size: int
    active_jobs: int
    system_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]

# Global job storage (in production, use Redis or database)
active_jobs: Dict[str, JobStatus] = {}
job_counter = 0

@app.on_startup
async def startup_event():
    """Initialize system on startup"""
    logger.info("🚀 Content Factory starting up...")
    
    try:
        # Initialize processors
        await tts_processor.initialize()
        logger.info("✅ TTS Processor initialized")
        
        await nvenc_processor.initialize()
        logger.info("✅ NVENC Processor initialized")
        
        # Start queue manager
        await queue_manager.start()
        logger.info("✅ Queue Manager started")
        
        # Start performance monitoring
        performance_monitor.start()
        logger.info("✅ Performance Monitor started")
        
        logger.info("🎬 Content Factory ready for mass production!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_shutdown
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Content Factory shutting down...")
    
    try:
        # Stop all processors
        await queue_manager.stop()
        performance_monitor.stop()
        
        # Cleanup temporary files
        temp_manager.cleanup_all()
        
        logger.info("👋 Content Factory shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main frontend page"""
    try:
        frontend_file = Path("frontend/index.html")
        if frontend_file.exists():
            return HTMLResponse(content=frontend_file.read_text())
        else:
            return HTMLResponse(content="""
            <html>
                <head><title>Content Factory</title></head>
                <body>
                    <h1>🎬 Content Factory</h1>
                    <p>TikTok Mass Production System</p>
                    <p>Frontend not found. Please check frontend/index.html</p>
                    <p><a href="/docs">API Documentation</a></p>
                </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving root page: {e}")
        return HTMLResponse(content="<h1>Error loading page</h1>", status_code=500)

@app.get("/health")
async def health_check():
    """System health check endpoint"""
    try:
        health_report = await health_checker.comprehensive_health_check()
        return JSONResponse(content=health_report)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={"overall_health": False, "error": str(e)},
            status_code=500
        )

@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get queue statistics
        queue_stats = queue_manager.get_statistics()
        
        # Get performance metrics
        perf_metrics = performance_monitor.get_current_metrics()
        
        # Count job statuses
        total_jobs = len(active_jobs)
        completed_jobs = sum(1 for job in active_jobs.values() if job.status == "completed")
        failed_jobs = sum(1 for job in active_jobs.values() if job.status == "failed")
        active_job_count = sum(1 for job in active_jobs.values() if job.status in ["processing", "queued"])
        
        # Get system health
        health_report = await health_checker.comprehensive_health_check()
        
        stats = SystemStats(
            total_jobs=total_jobs,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            queue_size=queue_stats.get("queue_size", 0),
            active_jobs=active_job_count,
            system_health=health_report,
            performance_metrics=perf_metrics
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/languages")
async def get_available_languages():
    """Get available languages and voices"""
    return {"languages": LANGUAGES}

@app.post("/api/upload-videos")
async def upload_videos(files: List[UploadFile] = File(...)):
    """Upload video files for processing"""
    uploaded_files = []
    errors = []
    
    try:
        videos_dir = Path("content/videos")
        videos_dir.mkdir(exist_ok=True)
        
        for file in files:
            try:
                # Validate file
                if not file.filename:
                    errors.append("File has no name")
                    continue
                
                # Save file
                file_path = videos_dir / file.filename
                
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                # Validate video file
                validation = FileValidator.validate_video_file(str(file_path))
                
                if validation['valid']:
                    uploaded_files.append({
                        "filename": file.filename,
                        "path": str(file_path),
                        "size": len(content),
                        "info": validation.get('info', {}),
                        "warnings": validation.get('warnings', [])
                    })
                    logger.info(f"✅ Uploaded video: {file.filename}")
                else:
                    # Remove invalid file
                    file_path.unlink(missing_ok=True)
                    errors.append(f"{file.filename}: {validation['error']}")
                    logger.warning(f"❌ Invalid video {file.filename}: {validation['error']}")
                
            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")
                logger.error(f"Failed to upload {file.filename}: {e}")
        
        return {
            "uploaded_files": uploaded_files,
            "errors": errors,
            "success_count": len(uploaded_files),
            "error_count": len(errors)
        }
        
    except Exception as e:
        logger.error(f"Upload videos failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start-production")
async def start_production(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    languages: str = Form(...),  # JSON string
    video_files: str = Form(...),  # JSON string
    priority: int = Form(default=1)
):
    """Start mass production job"""
    global job_counter
    
    try:
        # Parse JSON parameters
        selected_languages = json.loads(languages)
        selected_videos = json.loads(video_files)
        
        # Validate languages
        invalid_languages = [lang for lang in selected_languages if lang not in LANGUAGES]
        if invalid_languages:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid languages: {invalid_languages}"
            )
        
        # Validate video files
        videos_dir = Path("content/videos")
        for video_file in selected_videos:
            video_path = videos_dir / video_file
            if not video_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Video file not found: {video_file}"
                )
        
        # Validate text
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        # Create job
        job_counter += 1
        job_id = f"job_{job_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job_status = JobStatus(
            job_id=job_id,
            status="queued",
            progress=0.0,
            created_at=datetime.now()
        )
        
        active_jobs[job_id] = job_status
        
        # Add to background processing
        background_tasks.add_task(
            process_production_job,
            job_id, text, selected_languages, selected_videos, priority
        )
        
        logger.info(f"🎬 Started production job {job_id}")
        logger.info(f"   Languages: {selected_languages}")
        logger.info(f"   Videos: {len(selected_videos)}")
        logger.info(f"   Text length: {len(text)} chars")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Production job started for {len(selected_languages)} languages"
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON parameter: {e}")
    except Exception as e:
        logger.error(f"Failed to start production: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and progress"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]

@app.get("/api/jobs")
async def list_jobs(limit: int = 50, status: Optional[str] = None):
    """List all jobs with optional filtering"""
    jobs = list(active_jobs.values())
    
    # Filter by status if specified
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    # Sort by creation date (newest first)
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    # Limit results
    jobs = jobs[:limit]
    
    return {
        "jobs": jobs,
        "total": len(jobs),
        "filtered": status is not None
    }

@app.delete("/api/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    if job.status in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
    
    # Try to cancel in queue manager
    cancelled = await queue_manager.cancel_job(job_id)
    
    if cancelled:
        job.status = "cancelled"
        logger.info(f"Job {job_id} cancelled")
        return {"message": "Job cancelled successfully"}
    else:
        raise HTTPException(status_code=400, detail="Job could not be cancelled")

@app.get("/api/download/{job_id}/{language}")
async def download_result(job_id: str, language: str):
    """Download production result"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Find the output file
    output_dir = Path(f"content/output/{language}")
    
    # Look for files with job_id in name
    output_files = list(output_dir.glob(f"*{job_id}*"))
    
    if not output_files:
        raise HTTPException(status_code=404, detail="Output file not found")
    
    output_file = output_files[0]  # Take the first match
    
    return FileResponse(
        path=output_file,
        filename=f"{job_id}_{language}.mp4",
        media_type="video/mp4"
    )

@app.post("/api/cleanup")
async def cleanup_system():
    """Manual system cleanup"""
    try:
        # Cleanup temp files
        cleanup_stats = temp_manager.cleanup_all()
        
        # Remove old completed jobs (older than 24 hours)
        cutoff_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        old_jobs = [
            job_id for job_id, job in active_jobs.items()
            if job.created_at < cutoff_time and job.status in ["completed", "failed", "cancelled"]
        ]
        
        for job_id in old_jobs:
            del active_jobs[job_id]
        
        logger.info(f"Cleanup completed: {cleanup_stats}, removed {len(old_jobs)} old jobs")
        
        return {
            "message": "Cleanup completed",
            "temp_files_removed": cleanup_stats['files_removed'],
            "temp_dirs_removed": cleanup_stats['dirs_removed'],
            "old_jobs_removed": len(old_jobs)
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background processing function
@handle_errors()
async def process_production_job(
    job_id: str, 
    text: str, 
    languages: List[str], 
    video_files: List[str], 
    priority: int
):
    """Process production job in background"""
    job = active_jobs[job_id]
    
    try:
        job.status = "processing"
        logger.info(f"🎬 Processing job {job_id}")
        
        total_tasks = len(languages) * len(video_files)
        completed_tasks = 0
        
        results = {}
        
        for language in languages:
            results[language] = []
            
            for video_file in video_files:
                try:
                    # Update progress
                    job.progress = (completed_tasks / total_tasks) * 100
                    
                    # Generate TTS audio
                    logger.info(f"Generating TTS for {language}...")
                    voice = LANGUAGES[language]["voice"]
                    audio_file = await tts_processor.generate_audio(text, voice, language)
                    
                    # Process video with NVENC
                    logger.info(f"Processing video {video_file} for {language}...")
                    video_path = Path("content/videos") / video_file
                    
                    output_filename = f"{job_id}_{language}_{Path(video_file).stem}.mp4"
                    output_path = Path(f"content/output/{language}") / output_filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Add to processing queue
                    task_id = await queue_manager.add_task(
                        nvenc_processor.process_video,
                        {
                            "video_path": str(video_path),
                            "audio_path": audio_file,
                            "output_path": str(output_path),
                            "text": text
                        },
                        priority=priority
                    )
                    
                    # Wait for task completion
                    result = await queue_manager.wait_for_task(task_id)
                    
                    if result["success"]:
                        results[language].append({
                            "video_file": video_file,
                            "output_file": output_filename,
                            "output_path": str(output_path),
                            "duration": result.get("duration", 0),
                            "file_size": result.get("file_size", 0)
                        })
                        logger.info(f"✅ Completed {language}/{video_file}")
                    else:
                        logger.error(f"❌ Failed {language}/{video_file}: {result.get('error')}")
                        results[language].append({
                            "video_file": video_file,
                            "error": result.get("error", "Unknown error")
                        })
                    
                    completed_tasks += 1
                    
                    # Cleanup temp audio file
                    if Path(audio_file).exists():
                        Path(audio_file).unlink()
                    
                except Exception as e:
                    logger.error(f"Task failed {language}/{video_file}: {e}")
                    results[language].append({
                        "video_file": video_file,
                        "error": str(e)
                    })
                    completed_tasks += 1
        
        # Job completed
        job.status = "completed"
        job.progress = 100.0
        job.results = results
        
        # Calculate success statistics
        total_videos = sum(len(results[lang]) for lang in results)
        successful_videos = sum(
            len([r for r in results[lang] if "error" not in r]) 
            for lang in results
        )
        
        logger.info(f"🎉 Job {job_id} completed: {successful_videos}/{total_videos} videos successful")
        
        # Record performance metrics
        performance_monitor.record_job_completion(
            job_id, total_videos, successful_videos, 
            (datetime.now() - job.created_at).total_seconds()
        )
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        logger.error(f"💥 Job {job_id} failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )