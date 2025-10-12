"""
API endpoints para Biometric Database
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import numpy as np

from app.core.biometric_database import (
    get_biometric_database,
    TemplateType,
    BiometricQuality,
    SearchStrategy
)

router = APIRouter(prefix="/biometric-database", tags=["Biometric Database"])


class DatabaseStatsResponse(BaseModel):
    """Respuesta con estadísticas de la base de datos"""
    total_users: int
    total_templates: int
    anatomical_templates: int
    dynamic_templates: int
    multimodal_templates: int
    database_size_mb: float


class UserProfileResponse(BaseModel):
    """Respuesta con perfil de usuario"""
    user_id: str
    username: str
    total_templates: int
    gesture_sequence: List[str]
    total_enrollments: int
    verification_success_rate: float


@router.get("/health")
async def biometric_database_health_check():
    """Verifica que Biometric Database esté funcionando"""
    try:
        db = get_biometric_database()
        
        return {
            "status": "healthy",
            "module": "Biometric Database",
            "initialized": True,
            "message": "✅ Módulo 13 cargado correctamente",
            "total_users": len(db.users),
            "total_templates": len(db.templates),
            "database_path": str(db.db_path),
            "encryption_enabled": db.config.get('encryption_enabled', False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Biometric Database: {str(e)}")


@router.get("/stats", response_model=DatabaseStatsResponse)
async def get_database_stats():
    """Obtiene estadísticas de la base de datos"""
    try:
        db = get_biometric_database()
        stats = db.get_database_stats()
        
        return {
            "total_users": stats.total_users,
            "total_templates": stats.total_templates,
            "anatomical_templates": stats.anatomical_templates,
            "dynamic_templates": stats.dynamic_templates,
            "multimodal_templates": stats.multimodal_templates,
            "database_size_mb": stats.total_size_mb
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/summary")
async def get_database_summary():
    """Obtiene resumen completo de la base de datos"""
    try:
        db = get_biometric_database()
        summary = db.get_summary()
        
        return {
            "status": "success",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/users")
async def list_all_users():
    """Lista todos los usuarios registrados"""
    try:
        db = get_biometric_database()
        users = db.list_users()
        
        users_data = []
        for user in users:
            users_data.append({
                "user_id": user.user_id,
                "username": user.username,
                "total_templates": user.total_templates,
                "gesture_sequence": user.gesture_sequence or [],
                "total_enrollments": user.total_enrollments,
                "verification_success_rate": user.verification_success_rate,
                "created_at": user.created_at,
                "last_activity": user.last_activity
            })
        
        return {
            "status": "success",
            "total_users": len(users_data),
            "users": users_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/users/{user_id}")
async def get_user_profile(user_id: str):
    """Obtiene perfil detallado de un usuario"""
    try:
        db = get_biometric_database()
        user = db.get_user(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado")
        
        return {
            "status": "success",
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "total_templates": user.total_templates,
                "anatomical_templates": len(user.anatomical_templates),
                "dynamic_templates": len(user.dynamic_templates),
                "multimodal_templates": len(user.multimodal_templates),
                "gesture_sequence": user.gesture_sequence or [],
                "total_enrollments": user.total_enrollments,
                "total_verifications": user.total_verifications,
                "successful_verifications": user.successful_verifications,
                "verification_success_rate": user.verification_success_rate,
                "created_at": user.created_at,
                "updated_at": user.updated_at,
                "last_activity": user.last_activity,
                "metadata": user.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/users/{user_id}/templates")
async def get_user_templates(user_id: str):
    """Obtiene templates de un usuario específico"""
    try:
        db = get_biometric_database()
        
        user = db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado")
        
        templates = db.list_user_templates(user_id)
        
        templates_data = []
        for template in templates:
            templates_data.append({
                "template_id": template.template_id,
                "template_type": template.template_type.value,
                "gesture_name": template.gesture_name,
                "quality_score": template.quality_score,
                "quality_level": template.quality_level.value,
                "confidence": template.confidence,
                "has_anatomical": template.anatomical_embedding is not None,
                "has_dynamic": template.dynamic_embedding is not None,
                "created_at": template.created_at,
                "verification_count": template.verification_count,
                "success_rate": template.success_rate,
                "is_bootstrap": template.metadata.get('bootstrap_mode', False)
            })
        
        return {
            "status": "success",
            "user_id": user_id,
            "total_templates": len(templates_data),
            "templates": templates_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/templates/{template_id}")
async def get_template_details(template_id: str):
    """Obtiene detalles de un template específico"""
    try:
        db = get_biometric_database()
        template = db.get_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} no encontrado")
        
        return {
            "status": "success",
            "template": {
                "template_id": template.template_id,
                "user_id": template.user_id,
                "template_type": template.template_type.value,
                "gesture_name": template.gesture_name,
                "hand_side": template.hand_side,
                "quality_score": template.quality_score,
                "quality_level": template.quality_level.value,
                "confidence": template.confidence,
                "has_anatomical_embedding": template.anatomical_embedding is not None,
                "has_dynamic_embedding": template.dynamic_embedding is not None,
                "anatomical_embedding_shape": template.anatomical_embedding.shape if template.anatomical_embedding is not None else None,
                "dynamic_embedding_shape": template.dynamic_embedding.shape if template.dynamic_embedding is not None else None,
                "created_at": template.created_at,
                "updated_at": template.updated_at,
                "last_used": template.last_used,
                "enrollment_session": template.enrollment_session,
                "verification_count": template.verification_count,
                "success_count": template.success_count,
                "success_rate": template.success_rate,
                "is_encrypted": template.is_encrypted,
                "checksum": template.checksum,
                "is_bootstrap": template.metadata.get('bootstrap_mode', False),
                "metadata": template.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """Elimina un usuario y todos sus templates"""
    try:
        db = get_biometric_database()
        
        if user_id not in db.users:
            raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado")
        
        success = db.delete_user(user_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Usuario {user_id} eliminado exitosamente"
            }
        else:
            raise HTTPException(status_code=500, detail="Error eliminando usuario")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """Elimina un template específico"""
    try:
        db = get_biometric_database()
        
        if template_id not in db.templates:
            raise HTTPException(status_code=404, detail=f"Template {template_id} no encontrado")
        
        success = db.delete_template(template_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Template {template_id} eliminado exitosamente"
            }
        else:
            raise HTTPException(status_code=500, detail="Error eliminando template")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/bootstrap/stats")
async def get_bootstrap_stats():
    """Obtiene estadísticas de templates Bootstrap"""
    try:
        db = get_biometric_database()
        stats = db.get_bootstrap_stats()
        
        return {
            "status": "success",
            "bootstrap_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/bootstrap/templates")
async def get_bootstrap_templates(user_id: Optional[str] = None):
    """Obtiene templates en modo Bootstrap"""
    try:
        db = get_biometric_database()
        templates = db.get_bootstrap_templates(user_id)
        
        templates_data = []
        for template in templates:
            templates_data.append({
                "template_id": template.template_id,
                "user_id": template.user_id,
                "gesture_name": template.gesture_name,
                "quality_score": template.quality_score,
                "has_anatomical_raw": template.metadata.get('has_anatomical_raw', False),
                "has_temporal_data": template.metadata.get('has_temporal_data', False),
                "created_at": template.created_at
            })
        
        return {
            "status": "success",
            "total_bootstrap_templates": len(templates_data),
            "user_filter": user_id,
            "templates": templates_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/integrity/verify")
async def verify_database_integrity():
    """Verifica integridad de la base de datos"""
    try:
        db = get_biometric_database()
        integrity_report = db.verify_integrity()
        
        return {
            "status": "success",
            "integrity_report": integrity_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/backup/create")
async def create_database_backup():
    """Crea un backup de la base de datos"""
    try:
        db = get_biometric_database()
        success = db.create_backup()
        
        if success:
            return {
                "status": "success",
                "message": "Backup creado exitosamente"
            }
        else:
            raise HTTPException(status_code=500, detail="Error creando backup")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/indices/stats")
async def get_indices_stats():
    """Obtiene estadísticas de índices vectoriales"""
    try:
        db = get_biometric_database()
        
        anatomical_stats = db.anatomical_index.get_stats()
        dynamic_stats = db.dynamic_index.get_stats()
        
        return {
            "status": "success",
            "anatomical_index": anatomical_stats,
            "dynamic_index": dynamic_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/config")
async def get_database_config():
    """Obtiene configuración de la base de datos"""
    try:
        db = get_biometric_database()
        
        return {
            "status": "success",
            "config": {
                "encryption_enabled": db.config.get('encryption_enabled', False),
                "auto_backup": db.config.get('auto_backup', True),
                "search_strategy": db.config.get('search_strategy', 'linear'),
                "max_templates_per_user": db.config.get('max_templates_per_user', 50),
                "debug_mode": db.config.get('debug_mode', False),
                "database_path": str(db.db_path)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")