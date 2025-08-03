"""
Base repository implementation with common CRUD operations.
"""

from typing import Generic, TypeVar, Type, List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.exc import IntegrityError
import logging

from causal_eval.database.models import Base

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations."""
    
    def __init__(self, model: Type[ModelType], session: AsyncSession):
        """Initialize repository with model class and session."""
        self.model = model
        self.session = session
    
    async def create(self, **kwargs) -> ModelType:
        """Create a new record."""
        try:
            instance = self.model(**kwargs)
            self.session.add(instance)
            await self.session.commit()
            await self.session.refresh(instance)
            logger.debug(f"Created {self.model.__name__} with id {instance.id}")
            return instance
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Failed to create {self.model.__name__}: {e}")
            raise
    
    async def get_by_id(self, id: int) -> Optional[ModelType]:
        """Get record by ID."""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_field(self, field_name: str, value: Any) -> Optional[ModelType]:
        """Get record by a specific field."""
        field = getattr(self.model, field_name)
        result = await self.session.execute(
            select(self.model).where(field == value)
        )
        return result.scalar_one_or_none()
    
    async def get_all(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        **filters
    ) -> List[ModelType]:
        """Get all records with optional filtering and pagination."""
        query = select(self.model)
        
        # Apply filters
        if filters:
            conditions = []
            for field_name, value in filters.items():
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    if isinstance(value, list):
                        conditions.append(field.in_(value))
                    else:
                        conditions.append(field == value)
            
            if conditions:
                query = query.where(and_(*conditions))
        
        # Apply ordering
        if order_by:
            if order_by.startswith("-"):
                # Descending order
                field_name = order_by[1:]
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    query = query.order_by(field.desc())
            else:
                # Ascending order
                if hasattr(self.model, order_by):
                    field = getattr(self.model, order_by)
                    query = query.order_by(field)
        
        # Apply pagination
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def update(self, id: int, **kwargs) -> Optional[ModelType]:
        """Update record by ID."""
        try:
            # Remove None values and fields that don't exist in the model
            update_data = {
                k: v for k, v in kwargs.items() 
                if v is not None and hasattr(self.model, k)
            }
            
            if not update_data:
                # Nothing to update
                return await self.get_by_id(id)
            
            await self.session.execute(
                update(self.model)
                .where(self.model.id == id)
                .values(**update_data)
            )
            await self.session.commit()
            
            logger.debug(f"Updated {self.model.__name__} id {id}")
            return await self.get_by_id(id)
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Failed to update {self.model.__name__} id {id}: {e}")
            raise
    
    async def delete(self, id: int) -> bool:
        """Delete record by ID."""
        try:
            result = await self.session.execute(
                delete(self.model).where(self.model.id == id)
            )
            await self.session.commit()
            
            deleted = result.rowcount > 0
            if deleted:
                logger.debug(f"Deleted {self.model.__name__} id {id}")
            return deleted
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to delete {self.model.__name__} id {id}: {e}")
            raise
    
    async def count(self, **filters) -> int:
        """Count records with optional filtering."""
        query = select(self.model.id)
        
        # Apply filters
        if filters:
            conditions = []
            for field_name, value in filters.items():
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    if isinstance(value, list):
                        conditions.append(field.in_(value))
                    else:
                        conditions.append(field == value)
            
            if conditions:
                query = query.where(and_(*conditions))
        
        result = await self.session.execute(query)
        return len(result.scalars().all())
    
    async def exists(self, **filters) -> bool:
        """Check if record exists with given filters."""
        count = await self.count(**filters)
        return count > 0
    
    async def bulk_create(self, instances: List[Dict[str, Any]]) -> List[ModelType]:
        """Create multiple records in bulk."""
        try:
            objects = [self.model(**data) for data in instances]
            self.session.add_all(objects)
            await self.session.commit()
            
            # Refresh to get IDs
            for obj in objects:
                await self.session.refresh(obj)
            
            logger.debug(f"Bulk created {len(objects)} {self.model.__name__} records")
            return objects
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Failed to bulk create {self.model.__name__}: {e}")
            raise
    
    async def search(
        self, 
        search_fields: List[str], 
        search_term: str,
        limit: Optional[int] = None,
        **filters
    ) -> List[ModelType]:
        """Search records by text in specified fields."""
        query = select(self.model)
        
        # Build search conditions
        search_conditions = []
        for field_name in search_fields:
            if hasattr(self.model, field_name):
                field = getattr(self.model, field_name)
                # Case-insensitive search
                search_conditions.append(field.ilike(f"%{search_term}%"))
        
        if search_conditions:
            query = query.where(or_(*search_conditions))
        
        # Apply additional filters
        if filters:
            conditions = []
            for field_name, value in filters.items():
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    conditions.append(field == value)
            
            if conditions:
                query = query.where(and_(*conditions))
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()


class SyncBaseRepository(Generic[ModelType]):
    """Synchronous base repository for non-async contexts."""
    
    def __init__(self, model: Type[ModelType], session: Session):
        """Initialize repository with model class and session."""
        self.model = model
        self.session = session
    
    def create(self, **kwargs) -> ModelType:
        """Create a new record."""
        try:
            instance = self.model(**kwargs)
            self.session.add(instance)
            self.session.commit()
            self.session.refresh(instance)
            logger.debug(f"Created {self.model.__name__} with id {instance.id}")
            return instance
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Failed to create {self.model.__name__}: {e}")
            raise
    
    def get_by_id(self, id: int) -> Optional[ModelType]:
        """Get record by ID."""
        return self.session.query(self.model).filter(self.model.id == id).first()
    
    def get_all(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        **filters
    ) -> List[ModelType]:
        """Get all records with optional filtering and pagination."""
        query = self.session.query(self.model)
        
        # Apply filters
        for field_name, value in filters.items():
            if hasattr(self.model, field_name):
                field = getattr(self.model, field_name)
                query = query.filter(field == value)
        
        # Apply pagination
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def update(self, id: int, **kwargs) -> Optional[ModelType]:
        """Update record by ID."""
        try:
            update_data = {
                k: v for k, v in kwargs.items() 
                if v is not None and hasattr(self.model, k)
            }
            
            if update_data:
                self.session.query(self.model).filter(self.model.id == id).update(update_data)
                self.session.commit()
                logger.debug(f"Updated {self.model.__name__} id {id}")
            
            return self.get_by_id(id)
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Failed to update {self.model.__name__} id {id}: {e}")
            raise
    
    def delete(self, id: int) -> bool:
        """Delete record by ID."""
        try:
            result = self.session.query(self.model).filter(self.model.id == id).delete()
            self.session.commit()
            
            deleted = result > 0
            if deleted:
                logger.debug(f"Deleted {self.model.__name__} id {id}")
            return deleted
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to delete {self.model.__name__} id {id}: {e}")
            raise