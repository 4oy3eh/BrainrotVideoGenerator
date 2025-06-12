#!/usr/bin/env python3
"""
Content Factory Launcher
Launch script for TikTok mass production system with comprehensive checks
"""

import os
import sys
import asyncio
import signal
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from modules.utils import (
        setup_logging, health_checker, temp_manager, 
        create_directory_structure, emergency_cleanup
    )
    from config.hardware_config import HardwareConfig
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    print("Make sure all required files are in place and dependencies are installed.")
    sys.exit(1)

class ContentFactoryLauncher:
    """Main launcher for the content factory system"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.app_process = None
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True
        if self.app_process:
            self.app_process.terminate()
    
    def print_banner(self):
        """Print startup banner"""
        banner = """
    ╔══════════════════════════════════════════════════════╗
    ║              🎬 CONTENT FACTORY v1.0 🎬              ║
    ║                                                      ║
    ║          TikTok Mass Production System               ║
    ║          36 videos/day • 6 languages                ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
        """
        print(banner)
    
    async def pre_flight_checks(self) -> Dict[str, Any]:
        """Comprehensive pre-flight system checks"""
        print("\n🔍 Running pre-flight checks...")
        
        # Run comprehensive health check
        health_report = await health_checker.comprehensive_health_check()
        
        print(f"\n📊 System Health Report:")
        print(f"   Overall Status: {'✅ HEALTHY' if health_report['overall_health'] else '❌ ISSUES DETECTED'}")
        
        # Dependencies check
        deps = health_report['checks'].get('dependencies', {})
        print(f"\n📦 Dependencies:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"   {status} {dep}")
        
        # System resources
        resources = health_report['checks'].get('system_resources', {})
        if resources:
            print(f"\n💻 System Resources:")
            print(f"   CPU Usage: {resources.get('cpu_percent', 0):.1f}%")
            print(f"   Memory Usage: {resources.get('memory_percent', 0):.1f}%")
            print(f"   Available Memory: {resources.get('memory_available_gb', 0):.1f} GB")
            print(f"   GPU Available: {'✅' if resources.get('gpu_available') else '❌'}")
        
        # Disk space
        disk_info = health_report['checks'].get('disk_space', {})
        if disk_info:
            print(f"\n💾 Disk Space:")
            print(f"   Free Space: {disk_info.get('free_gb', 0):.1f} GB")
            print(f"   Usage: {disk_info.get('usage_percent', 0):.1f}%")
            print(f"   Status: {'✅ Sufficient' if disk_info.get('sufficient') else '❌ Low Space'}")
        
        # Edge-TTS test
        tts_status = health_report['checks'].get('edge_tts', False)
        print(f"\n🎤 Text-to-Speech:")
        print(f"   Edge-TTS: {'✅ Working' if tts_status else '❌ Not Working'}")
        
        # Show errors and warnings
        if health_report['errors']:
            print(f"\n❌ Critical Errors:")
            for error in health_report['errors']:
                print(f"   • {error}")
        
        if health_report['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in health_report['warnings']:
                print(f"   • {warning}")
        
        return health_report
    
    def check_configuration(self) -> bool:
        """Check configuration files"""
        print("\n🔧 Checking configuration...")
        
        config_files = [
            "config/languages.json",
            "config/hardware_config.py"
        ]
        
        missing_configs = []
        for config_file in config_files:
            if not Path(config_file).exists():
                missing_configs.append(config_file)
        
        if missing_configs:
            print("❌ Missing configuration files:")
            for config in missing_configs:
                print(f"   • {config}")
            return False
        
        # Test hardware config import
        try:
            hw_config = HardwareConfig()
            print(f"✅ Hardware configuration loaded")
            print(f"   NVENC Encoder: {hw_config.VIDEO_SETTINGS['codec']}")
            print(f"   Output Resolution: {hw_config.OUTPUT_FORMAT['width']}x{hw_config.OUTPUT_FORMAT['height']}")
            print(f"   Max Concurrent Jobs: {hw_config.PERFORMANCE_LIMITS['max_concurrent_videos']}")
        except Exception as e:
            print(f"❌ Hardware configuration error: {e}")
            return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install missing Python dependencies"""
        print("\n📥 Checking Python dependencies...")
        
        try:
            # Check if requirements.txt exists
            if not Path("requirements.txt").exists():
                print("⚠️  requirements.txt not found, skipping dependency installation")
                return True
            
            # Try to install requirements
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Python dependencies installed/verified")
                return True
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Dependency installation timed out")
            return False
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_directories(self) -> bool:
        """Setup required directory structure"""
        print("\n📁 Setting up directories...")
        
        success = create_directory_structure()
        if success:
            print("✅ Directory structure created")
        else:
            print("❌ Failed to create directory structure")
        
        return success
    
    async def start_application(self) -> bool:
        """Start the main FastAPI application"""
        print("\n🚀 Starting Content Factory application...")
        
        try:
            # Import and start the FastAPI app
            import uvicorn
            from app import app
            
            # Configure uvicorn
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                access_log=True,
                reload=False  # Disable reload in production
            )
            
            server = uvicorn.Server(config)
            
            print("✅ FastAPI server starting on http://localhost:8000")
            print("   Frontend available at: http://localhost:8000")
            print("   API docs at: http://localhost:8000/docs")
            
            # Start server
            await server.serve()
            
            return True
            
        except ImportError as e:
            print(f"❌ Failed to import application: {e}")
            return False
        except Exception as e:
            print(f"❌ Failed to start application: {e}")
            return False
    
    def cleanup_on_exit(self):
        """Cleanup resources on exit"""
        print("\n🧹 Performing cleanup...")
        
        try:
            # Cleanup temporary files
            cleanup_stats = temp_manager.cleanup_all()
            print(f"✅ Cleaned up {cleanup_stats['files_removed']} temp files, {cleanup_stats['dirs_removed']} temp directories")
            
            if cleanup_stats['errors'] > 0:
                print(f"⚠️  {cleanup_stats['errors']} cleanup errors occurred")
        
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
        
        print("👋 Content Factory shutdown complete")
    
    async def run(self):
        """Main run method"""
        try:
            self.print_banner()
            
            # Step 1: Install dependencies
            if not self.install_dependencies():
                print("\n❌ Failed to install dependencies. Please install manually:")
                print("   pip install -r requirements.txt")
                return False
            
            # Step 2: Setup directories
            if not self.setup_directories():
                print("\n❌ Failed to setup directories")
                return False
            
            # Step 3: Check configuration
            if not self.check_configuration():
                print("\n❌ Configuration check failed")
                return False
            
            # Step 4: Pre-flight checks
            health_report = await self.pre_flight_checks()
            
            if not health_report['overall_health']:
                print("\n❌ Pre-flight checks failed!")
                print("\nCritical issues must be resolved before starting:")
                
                for error in health_report.get('errors', []):
                    print(f"   • {error}")
                
                print("\nPlease fix these issues and try again.")
                return False
            
            # Show warnings but continue
            if health_report.get('warnings'):
                print("\n⚠️  System has warnings but will continue:")
                for warning in health_report['warnings']:
                    print(f"   • {warning}")
                
                print("\nPress Enter to continue or Ctrl+C to abort...")
                try:
                    input()
                except KeyboardInterrupt:
                    print("\n👋 Startup aborted by user")
                    return False
            
            print("\n✅ All checks passed! Starting application...")
            
            # Step 5: Start the application
            success = await self.start_application()
            
            return success
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Startup interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Launcher error: {e}")
            print(f"\n❌ Unexpected error: {e}")
            return False
        finally:
            self.cleanup_on_exit()

def print_help():
    """Print help information"""
    help_text = """
Content Factory Launcher

Usage:
    python launcher.py [options]

Options:
    --help, -h          Show this help message
    --check-only        Run health checks only, don't start server
    --force-start       Skip pre-flight checks and start anyway
    --install-deps      Install dependencies and exit
    --cleanup           Clean up temporary files and exit

Examples:
    python launcher.py                 # Normal startup
    python launcher.py --check-only    # Health check only
    python launcher.py --install-deps  # Install dependencies
    python launcher.py --cleanup       # Clean temp files

System Requirements:
    - Python 3.8+
    - FFmpeg with NVENC support
    - NVIDIA GPU (GTX 1070 or better)
    - 8GB+ RAM
    - 5GB+ free disk space

For more information, check the README.md file.
    """
    print(help_text)

async def main():
    """Main entry point"""
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        print_help()
        return
    
    if '--cleanup' in args:
        print("🧹 Cleaning up temporary files...")
        temp_manager.cleanup_all()
        print("✅ Cleanup complete")
        return
    
    if '--install-deps' in args:
        launcher = ContentFactoryLauncher()
        if launcher.install_dependencies():
            print("✅ Dependencies installed successfully")
        else:
            print("❌ Failed to install dependencies")
        return
    
    launcher = ContentFactoryLauncher()
    
    if '--check-only' in args:
        print("🔍 Running health checks only...")
        health_report = await launcher.pre_flight_checks()
        
        if health_report['overall_health']:
            print("\n✅ System is ready for operation!")
            sys.exit(0)
        else:
            print("\n❌ System has critical issues")
            sys.exit(1)
    
    # Normal startup or force start
    force_start = '--force-start' in args
    
    if force_start:
        print("⚠️  FORCE START mode - skipping some safety checks")
        launcher.setup_directories()
        success = await launcher.start_application()
    else:
        success = await launcher.run()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)