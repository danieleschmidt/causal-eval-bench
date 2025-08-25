"""Production-ready entry point for the Quantum Causal Evaluation Bench API."""

import sys
import os
import asyncio
import signal
from typing import Optional
import uvicorn
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causal_eval.api.app import create_app
from causal_eval.core.logging_config import setup_logging

# Configure logging for production
logger = logging.getLogger(__name__)


class ProductionServer:
    """Production-grade server with graceful shutdown and health monitoring."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = None
        self.server_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
    async def startup(self) -> None:
        """Initialize the application with all components."""
        logger.info("🚀 Starting Quantum Causal Evaluation Bench API...")
        
        # Create the FastAPI application
        self.app = create_app()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info("✅ Application startup completed")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_server(self) -> None:
        """Run the server with production configuration."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            reload=False,  # Never reload in production
            workers=1,     # Single worker for now, can be scaled with process manager
            # Performance tuning
            backlog=2048,
            limit_concurrency=1000,
            limit_max_requests=10000,
            timeout_keep_alive=30,
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"🌟 Quantum Causal Evaluation API starting on {self.host}:{self.port}")
        logger.info("📊 Features enabled:")
        logger.info("  • Advanced Causal Reasoning Evaluation")
        logger.info("  • Quantum-Leap Performance Optimization")
        logger.info("  • Intelligent Adaptive Caching")
        logger.info("  • Real-time Resource Monitoring")
        logger.info("  • Enterprise-Grade Security")
        logger.info("  • Research-Grade Analytics")
        
        # Start server
        try:
            await server.serve()
        except Exception as e:
            logger.error(f"Server failed to start: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        logger.info("🔄 Initiating graceful shutdown...")
        
        # Signal the shutdown event
        self.shutdown_event.set()
        
        # Cancel the server task if running
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Graceful shutdown completed")
    
    async def run(self) -> None:
        """Run the complete server lifecycle."""
        try:
            await self.startup()
            self.server_task = asyncio.create_task(self.run_server())
            
            # Wait for shutdown signal or server completion
            done, pending = await asyncio.wait(
                [self.server_task, asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            await self.shutdown()


def get_config_from_env() -> dict:
    """Get server configuration from environment variables."""
    return {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "8000")),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "info"),
        "workers": int(os.getenv("WORKERS", "1")),
    }


async def main_async():
    """Async main function for production server."""
    config = get_config_from_env()
    
    # Create and run the production server
    server = ProductionServer(
        host=config["host"],
        port=config["port"]
    )
    
    await server.run()


def main():
    """Main entry point for the application."""
    try:
        # Check Python version
        if sys.version_info < (3, 9):
            print("Error: Python 3.9+ required")
            sys.exit(1)
        
        # Set event loop policy for better performance on Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Run the async main function
        asyncio.run(main_async())
        
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)


def dev_main():
    """Development server with hot reload."""
    config = get_config_from_env()
    
    logger.info("🔧 Starting development server with hot reload...")
    
    app = create_app()
    
    uvicorn.run(
        "causal_eval.main:dev_app",
        host=config["host"],
        port=config["port"],
        reload=True,
        reload_dirs=["causal_eval"],
        log_level=config["log_level"],
    )


# Development app instance for hot reload
dev_app = create_app()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum Causal Evaluation Bench API")
    parser.add_argument("--dev", action="store_true", help="Run development server")
    
    args = parser.parse_args()
    
    if args.dev:
        dev_main()
    else:
        main()