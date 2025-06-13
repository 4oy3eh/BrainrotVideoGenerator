import asyncio
import sys
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional
import uvicorn
from contextlib import asynccontextmanager
import signal
import os

# Import our modules
from modules.tts_processor import TTSProcessor, setup_windows_asyncio
from modules.queue_manager import QueueManager
from modules.nvenc_processor import NVENCProcessor
from modules.performance_monitor import PerformanceMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global app state
app_state = {
    "tts_processor": None,
    "video_processor": None,
    "queue_manager": None,
    "performance_monitor": None,
    "shutdown_event": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with proper startup/shutdown"""
    # Setup Windows asyncio
    setup_windows_asyncio()
    
    # Create directories
    directories = [
        "content/texts", "content/videos", "content/output",
        "temp", "logs", "frontend/static"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Load language configuration
    with open("config/languages.json", "r", encoding="utf-8") as f:
        languages = json.load(f)
    
    # Initialize components
    logger.info("Starting Content Factory...")
    
    try:
        # Initialize processors
        app_state["tts_processor"] = TTSProcessor(languages)
        app_state["video_processor"] = NVENCProcessor()
        app_state["queue_manager"] = QueueManager()
        app_state["performance_monitor"] = PerformanceMonitor()
        app_state["shutdown_event"] = asyncio.Event()
        
        # Start background services
        asyncio.create_task(background_processor())
        asyncio.create_task(performance_monitor_task())
        
        logger.info("✓ Content Factory started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down Content Factory...")
        
        if app_state["shutdown_event"]:
            app_state["shutdown_event"].set()
        
        # Cleanup processors
        if app_state["tts_processor"]:
            try:
                await app_state["tts_processor"].cleanup()
            except Exception as e:
                logger.error(f"TTS cleanup error: {e}")
        
        if app_state["video_processor"]:
            try:
                await app_state["video_processor"].cleanup()
            except Exception as e:
                logger.error(f"Video cleanup error: {e}")
        
        # Wait a bit for cleanup
        await asyncio.sleep(0.5)
        logger.info("✓ Shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Content Factory",
    description="Mass TikTok Content Production System",
    version="1.0.0",
    lifespan=lifespan
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
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the main interface"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>Content Factory</h1>
        <p>Frontend not found. Please create frontend/index.html</p>
        </body></html>
        """)

@app.post("/api/create-content")
async def create_content(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    videos: List[UploadFile] = File(...),
    title: str = Form("video")
):
    """Create content for all languages"""
    try:
        if not app_state["queue_manager"]:
            return JSONResponse(
                status_code=503,
                content={"error": "Service not ready"}
            )
        
        # Save uploaded videos
        video_paths = []
        for i, video in enumerate(videos):
            video_path = f"content/videos/{title}_{i}.mp4"
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)
            video_paths.append(video_path)
        
        # Add to queue
        job_id = await app_state["queue_manager"].add_job({
            "text": text,
            "video_paths": video_paths,
            "title": title,
            "languages": ["en", "es", "pt", "fr", "de", "ru"]
        })
        
        return JSONResponse(content={
            "job_id": job_id,
            "status": "queued",
            "message": f"Job queued successfully. ID: {job_id}"
        })
        
    except Exception as e:
        logger.error(f"Content creation error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    try:
        if not app_state["queue_manager"]:
            return JSONResponse(
                status_code=503,
                content={"error": "Service not ready"}
            )
        
        status = await app_state["queue_manager"].get_job_status(job_id)
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/queue")
async def get_queue_status():
    """Get queue status"""
    try:
        if not app_state["queue_manager"]:
            return JSONResponse(
                status_code=503,
                content={"error": "Service not ready"}
            )
        
        queue_status = await app_state["queue_manager"].get_queue_status()
        return JSONResponse(content=queue_status)
        
    except Exception as e:
        logger.error(f"Queue status error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    try:
        if not app_state["performance_monitor"]:
            return JSONResponse(
                status_code=503,
                content={"error": "Service not ready"}
            )
        
        metrics = await app_state["performance_monitor"].get_metrics()
        return JSONResponse(content=metrics)
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

async def background_processor():
    """Background task processor"""
    logger.info("Starting background processor...")
    
    while not app_state["shutdown_event"].is_set():
        try:
            if not app_state["queue_manager"]:
                await asyncio.sleep(1)
                continue
            
            # Get next job
            job = await app_state["queue_manager"].get_next_job()
            if not job:
                await asyncio.sleep(1)
                continue
            
            logger.info(f"Processing job {job['id']}")
            
            # Update job status
            await app_state["queue_manager"].update_job_status(
                job["id"], "processing", "Generating TTS..."
            )
            
            # Generate TTS for all languages
            tts_results = {}
            if app_state["tts_processor"]:
                async with app_state["tts_processor"] as tts:
                    tts_results = await tts.generate_all_languages(
                        job["data"]["text"],
                        job["data"]["title"]
                    )
            
            if not tts_results:
                await app_state["queue_manager"].update_job_status(
                    job["id"], "failed", "TTS generation failed"
                )
                continue
            
            # Update status
            await app_state["queue_manager"].update_job_status(
                job["id"], "processing", "Compositing videos..."
            )
            
            # Generate videos for each language
            video_results = {}
            if app_state["video_processor"]:
                for lang, tts_path in tts_results.items():
                    try:
                        output_path = f"content/output/{lang}/{job['data']['title']}.mp4"
                        success = await app_state["video_processor"].create_video(
                            job["data"]["video_paths"],
                            tts_path,
                            output_path
                        )
                        if success:
                            video_results[lang] = output_path
                    except Exception as e:
                        logger.error(f"Video creation failed for {lang}: {e}")
            
            # Update final status
            if video_results:
                await app_state["queue_manager"].update_job_status(
                    job["id"], "completed", f"Generated {len(video_results)} videos",
                    {"videos": video_results}
                )
                logger.info(f"✓ Job {job['id']} completed: {len(video_results)} videos")
            else:
                await app_state["queue_manager"].update_job_status(
                    job["id"], "failed", "Video generation failed"
                )
                logger.error(f"✗ Job {job['id']} failed")
            
        except Exception as e:
            logger.error(f"Background processor error: {e}")
            await asyncio.sleep(5)  # Wait before retrying

async def performance_monitor_task():
    """Performance monitoring task"""
    while not app_state["shutdown_event"].is_set():
        try:
            if app_state["performance_monitor"]:
                await app_state["performance_monitor"].update_metrics()
        except Exception as e:
            logger.error(f"Performance monitor error: {e}")
        
        await asyncio.sleep(30)  # Update every 30 seconds

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if app_state["shutdown_event"]:
        app_state["shutdown_event"].set()

# Setup signal handlers
if sys.platform != 'win32':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        # Setup Windows asyncio before running
        setup_windows_asyncio()
        
        # Run the server
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload to prevent asyncio issues
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application terminated")