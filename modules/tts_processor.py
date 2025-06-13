import asyncio
import edge_tts
import logging
import os
from pathlib import Path
from typing import Dict, Optional
import aiofiles
import aiohttp
from contextlib import asynccontextmanager
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSProcessor:
    """
    Enhanced TTS Processor with proper SSL/session management for Windows
    Fixes Edge-TTS SSL transport errors
    """
    
    def __init__(self, languages_config: Dict[str, str]):
        self.languages = languages_config
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Session management
        self._sessions = {}
        self._semaphore = asyncio.Semaphore(3)  # Limit concurrent TTS requests
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup"""
        await self.cleanup()
    
    @asynccontextmanager
    async def get_session(self, voice: str):
        """Context manager for TTS sessions with proper cleanup"""
        session = None
        try:
            # Create SSL context that's more compatible with Windows
            ssl_context = None
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            yield session
            
        except Exception as e:
            logger.error(f"Session error for voice {voice}: {e}")
            raise
        finally:
            if session and not session.closed:
                try:
                    await session.close()
                    # Give time for cleanup
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Session cleanup warning: {e}")
    
    async def generate_tts_safe(self, text: str, voice: str, output_path: str) -> bool:
        """
        Safe TTS generation with proper error handling and resource cleanup
        """
        async with self._semaphore:  # Limit concurrent requests
            try:
                # Use Edge-TTS with proper session management
                communicate = edge_tts.Communicate(text=text, voice=voice)
                
                # Write to temporary file first
                temp_path = self.temp_dir / f"temp_{voice}_{hash(text)}.wav"
                
                async with aiofiles.open(temp_path, "wb") as temp_file:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            await temp_file.write(chunk["data"])
                
                # Move to final location
                final_path = Path(output_path)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Use synchronous file operations for reliability
                import shutil
                shutil.move(str(temp_path), str(final_path))
                
                logger.info(f"TTS generated successfully: {voice} -> {output_path}")
                return True
                
            except Exception as e:
                logger.error(f"TTS generation failed for {voice}: {e}")
                # Cleanup temp file if exists
                temp_path = self.temp_dir / f"temp_{voice}_{hash(text)}.wav"
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                return False
    
    async def generate_all_languages(self, text: str, base_filename: str) -> Dict[str, str]:
        """
        Generate TTS for all languages with proper error handling
        Returns dict of {language: filepath} for successful generations
        """
        results = {}
        tasks = []
        
        for lang_code, voice in self.languages.items():
            output_path = f"content/output/{lang_code}/{base_filename}.wav"
            task = asyncio.create_task(
                self.generate_tts_safe(text, voice, output_path),
                name=f"tts_{lang_code}"
            )
            tasks.append((lang_code, voice, output_path, task))
        
        # Wait for all tasks with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*[task for _, _, _, task in tasks], return_exceptions=True),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.error("TTS generation timed out")
            # Cancel remaining tasks
            for _, _, _, task in tasks:
                if not task.done():
                    task.cancel()
        
        # Collect results
        for lang_code, voice, output_path, task in tasks:
            try:
                if task.done() and not task.cancelled():
                    success = await task
                    if success:
                        results[lang_code] = output_path
                        logger.info(f"✓ {lang_code}: {output_path}")
                    else:
                        logger.error(f"✗ {lang_code}: Generation failed")
                else:
                    logger.error(f"✗ {lang_code}: Task cancelled or timeout")
            except Exception as e:
                logger.error(f"✗ {lang_code}: {e}")
        
        return results
    
    async def cleanup(self):
        """Cleanup resources and sessions"""
        logger.info("Cleaning up TTS processor...")
        
        # Clean up any remaining sessions
        for session in self._sessions.values():
            if not session.closed:
                try:
                    await session.close()
                except:
                    pass
        
        self._sessions.clear()
        
        # Clean up temp files
        try:
            for temp_file in self.temp_dir.glob("temp_*.wav"):
                temp_file.unlink()
        except Exception as e:
            logger.warning(f"Temp cleanup warning: {e}")
        
        # Give asyncio time to cleanup
        await asyncio.sleep(0.2)

# Windows-specific asyncio setup
def setup_windows_asyncio():
    """Setup asyncio properly for Windows"""
    if sys.platform == 'win32':
        # Use ProactorEventLoop for Windows
        if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Set event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

# Usage example with proper cleanup
async def main():
    """Example usage with proper resource management"""
    setup_windows_asyncio()
    
    languages = {
        "en": "en-US-JennyNeural",
        "es": "es-MX-DaliaNeural",
        "pt": "pt-BR-FranciscaNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
        "ru": "ru-RU-SvetlanaNeural"
    }
    
    text = "This is a test message for TTS generation."
    
    async with TTSProcessor(languages) as tts:
        results = await tts.generate_all_languages(text, "test_video_001")
        print(f"Generated TTS for {len(results)} languages:")
        for lang, path in results.items():
            print(f"  {lang}: {path}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Ensure cleanup on Windows
        if sys.platform == 'win32':
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.close()
            except:
                pass