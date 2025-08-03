"""
Database connection management for causal evaluation benchmark.
"""

import os
from typing import AsyncGenerator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging

from causal_eval.database.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager with connection URL."""
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "sqlite:///./causal_eval_bench.db"
        )
        
        # Determine if async or sync based on URL
        self.is_async = self.database_url.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://"))
        
        if self.is_async:
            self._init_async_engine()
        else:
            self._init_sync_engine()
    
    def _init_async_engine(self):
        """Initialize async database engine."""
        connect_args = {}
        
        if "sqlite" in self.database_url:
            # SQLite-specific configuration for async
            connect_args = {
                "check_same_thread": False,
                "poolclass": StaticPool,
            }
            # Convert to async SQLite URL if needed
            if not self.database_url.startswith("sqlite+aiosqlite://"):
                self.database_url = self.database_url.replace("sqlite://", "sqlite+aiosqlite://")
        
        self.async_engine = create_async_engine(
            self.database_url,
            connect_args=connect_args,
            echo=os.getenv("DEBUG", "false").lower() == "true",
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info(f"Initialized async database engine: {self.database_url}")
    
    def _init_sync_engine(self):
        """Initialize synchronous database engine."""
        connect_args = {}
        
        if "sqlite" in self.database_url:
            connect_args = {"check_same_thread": False}
        
        self.engine = create_engine(
            self.database_url,
            connect_args=connect_args,
            echo=os.getenv("DEBUG", "false").lower() == "true",
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        self.session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
        
        logger.info(f"Initialized sync database engine: {self.database_url}")
    
    async def create_tables(self):
        """Create all database tables."""
        if self.is_async:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        else:
            Base.metadata.create_all(bind=self.engine)
        
        logger.info("Created all database tables")
    
    async def drop_tables(self):
        """Drop all database tables."""
        if self.is_async:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        else:
            Base.metadata.drop_all(bind=self.engine)
        
        logger.info("Dropped all database tables")
    
    async def get_async_session(self) -> AsyncSession:
        """Get async database session."""
        if not self.is_async:
            raise RuntimeError("Database not configured for async operations")
        
        return self.async_session_factory()
    
    def get_sync_session(self) -> Session:
        """Get synchronous database session."""
        if self.is_async:
            raise RuntimeError("Database configured for async operations, use get_async_session()")
        
        return self.session_factory()
    
    async def check_connection(self) -> bool:
        """Check database connection health."""
        try:
            if self.is_async:
                async with self.async_engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
            else:
                with self.engine.begin() as conn:
                    conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    async def close(self):
        """Close database connections."""
        if self.is_async and hasattr(self, 'async_engine'):
            await self.async_engine.dispose()
        elif hasattr(self, 'engine'):
            self.engine.dispose()
        
        logger.info("Closed database connections")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database(database_url: Optional[str] = None) -> DatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager
    
    if _db_manager is None or (database_url and database_url != _db_manager.database_url):
        _db_manager = DatabaseManager(database_url)
    
    return _db_manager


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get async database session."""
    db_manager = get_database()
    
    if not db_manager.is_async:
        # For sync operations, we'll need to adapt
        session = db_manager.get_sync_session()
        try:
            yield session
        finally:
            session.close()
    else:
        session = await db_manager.get_async_session()
        try:
            yield session
        finally:
            await session.close()


def get_sync_session() -> Session:
    """Get synchronous database session for non-async contexts."""
    db_manager = get_database()
    return db_manager.get_sync_session()


async def init_database(database_url: Optional[str] = None):
    """Initialize database with tables."""
    db_manager = get_database(database_url)
    await db_manager.create_tables()
    
    # Run initial data setup
    await _setup_initial_data(db_manager)


async def _setup_initial_data(db_manager: DatabaseManager):
    """Set up initial database data."""
    try:
        if db_manager.is_async:
            session = await db_manager.get_async_session()
            try:
                # Add any initial data setup here
                await session.commit()
            finally:
                await session.close()
        else:
            session = db_manager.get_sync_session()
            try:
                # Add any initial data setup here
                session.commit()
            finally:
                session.close()
        
        logger.info("Initial database data setup completed")
    except Exception as e:
        logger.error(f"Failed to set up initial data: {e}")
        raise