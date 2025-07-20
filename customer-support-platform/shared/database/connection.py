"""Database connection and session management."""

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from shared.config.settings import get_settings

# Configuration
settings = get_settings()

# Create async engine
engine = create_async_engine(
    settings.database.async_database_url,
    echo=settings.debug,
    poolclass=StaticPool,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={
        "server_settings": {
            "application_name": settings.service_name,
        }
    }
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logging.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables():
    """Create database tables."""
    from shared.database.models import Base
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logging.info("Database tables created successfully")


async def drop_tables():
    """Drop database tables."""
    from shared.database.models import Base
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        logging.info("Database tables dropped successfully")


async def health_check() -> bool:
    """Check database connectivity."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            return True
    except Exception as e:
        logging.error(f"Database health check failed: {e}")
        return False