"""
API Endpoints para gestión del sistema biométrico
VERSIÓN 100% CORREGIDA
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from app.core.system_manager import get_system_manager

router = APIRouter()


class SystemStatusResponse(BaseModel):
    """Response de estado del sistema"""
    initialization_level: str
    users_count: int
    networks_trained: bool
    database_ready: bool
    enrollment_active: bool
    authentication_active: bool
    bootstrap_mode: bool
    uptime: str
    version: str
    status: str
    error_message: Optional[str] = None


@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Obtiene el estado actual del sistema.
    
    Returns:
        SystemStatusResponse con información completa del estado
    """
    try:
        manager = get_system_manager()
        status_data = manager.get_system_status()
        
        # Convertir uptime_seconds a string legible
        uptime_seconds = status_data.get('uptime_seconds', 0)
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"
        
        # Construir response con todos los campos requeridos
        response = SystemStatusResponse(
            status=status_data.get('status', 'unknown'),
            initialization_level=status_data.get('initialization_level', 'NONE'),
            users_count=status_data.get('users_count', 0),
            networks_trained=status_data.get('networks_trained', False),
            database_ready=status_data.get('database_ready', False),
            enrollment_active=status_data.get('enrollment_active', False),
            authentication_active=status_data.get('authentication_active', False),
            bootstrap_mode=status_data.get('bootstrap_mode', False),
            uptime=uptime_str,
            version="2.0.0",  # Versión del sistema
            error_message=status_data.get('error_message')
        )
        
        return response
        
    except Exception as e:
        import traceback
        error_detail = f"Error obteniendo estado: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/system/status/detailed")
async def get_detailed_system_status():
    """
    Obtiene el estado detallado del sistema (sin validación Pydantic).
    Útil para debugging.
    
    Returns:
        Dict con toda la información del sistema
    """
    try:
        manager = get_system_manager()
        status_data = manager.get_system_status()
        
        # Agregar información adicional
        status_data['version'] = "2.0.0"
        status_data['component_status'] = {
            'camera': 'initialized' if hasattr(manager, 'camera_manager') else 'not_initialized',
            'mediapipe': 'initialized' if hasattr(manager, 'mediapipe_processor') else 'not_initialized',
            'anatomical_network': 'trained' if status_data.get('networks_trained') else 'not_trained',
            'dynamic_network': 'trained' if status_data.get('networks_trained') else 'not_trained',
            'database': 'ready' if status_data.get('database_ready') else 'not_ready'
        }
        
        return {
            "success": True,
            "data": status_data
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.post("/system/initialize")
async def initialize_system():
    """
    Inicializa el sistema biométrico.
    
    Returns:
        Dict con resultado de la inicialización
    """
    try:
        manager = get_system_manager()
        
        # Verificar si ya está inicializado
        if manager.state.initialization_level.value > 0:
            return {
                "initialized": True,
                "message": "Sistema ya inicializado",
                "level": manager.state.initialization_level.name,
                "level_value": manager.state.initialization_level.value
            }
        
        # Inicializar sistema
        success = manager.initialize_system()
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=manager.state.error_message or "Error inicializando sistema"
            )
        
        return {
            "initialized": True,
            "message": "Sistema inicializado exitosamente",
            "level": manager.state.initialization_level.name,
            "level_value": manager.state.initialization_level.value,
            "users_count": manager.state.users_count,
            "enrollment_active": manager.state.enrollment_active,
            "bootstrap_mode": manager.state.bootstrap_mode
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error en inicialización: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/system/train")
async def train_networks():
    """
    Entrena las redes neuronales del sistema.
    
    Returns:
        Dict con resultado del entrenamiento
    """
    try:
        manager = get_system_manager()
        
        # Verificar que haya suficientes usuarios
        if manager.state.users_count < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Se necesitan al menos 2 usuarios registrados. Actualmente: {manager.state.users_count}"
            )
        
        # Entrenar redes
        result = manager.train_networks()
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=500,
                detail=result.get('message', 'Error entrenando redes')
            )
        
        return {
            "success": True,
            "trained": True,
            "message": result.get('message', 'Redes entrenadas exitosamente'),
            "anatomical_trained": result.get('anatomical_trained', False),
            "dynamic_trained": result.get('dynamic_trained', False),
            "networks_trained": manager.state.networks_trained,
            "authentication_active": manager.state.authentication_active,
            "bootstrap_mode": manager.state.bootstrap_mode
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error en entrenamiento: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/system/retrain")
async def retrain_networks(force: bool = False):
    """
    Reentrena las redes neuronales.
    
    Args:
        force: Si True, fuerza el reentrenamiento aunque ya estén entrenadas
    
    Returns:
        Dict con resultado del reentrenamiento
    """
    try:
        manager = get_system_manager()
        
        # Verificar que haya suficientes usuarios
        if manager.state.users_count < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Se necesitan al menos 2 usuarios. Actualmente: {manager.state.users_count}"
            )
        
        # Reentrenar
        result = manager.train_networks(force=force)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=500,
                detail=result.get('message', 'Error reentrenando redes')
            )
        
        return {
            "success": True,
            "message": result.get('message', 'Redes reentrenadas exitosamente'),
            "anatomical_trained": result.get('anatomical_trained', False),
            "dynamic_trained": result.get('dynamic_trained', False),
            "networks_trained": manager.state.networks_trained,
            "authentication_active": manager.state.authentication_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error en reentrenamiento: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/system/modules")
async def get_modules_status():
    """
    Obtiene el estado de todos los módulos del sistema.
    
    Returns:
        Dict con estado de cada módulo
    """
    try:
        manager = get_system_manager()
        
        return {
            "success": True,
            "modules": manager.state.modules_loaded,
            "total_modules": len(manager.state.modules_loaded),
            "all_loaded": all(manager.state.modules_loaded.values())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/statistics")
async def get_system_statistics():
    """
    Obtiene estadísticas del sistema.
    
    Returns:
        Dict con estadísticas de uso
    """
    try:
        manager = get_system_manager()
        status = manager.get_system_status()
        
        return {
            "success": True,
            "statistics": status.get('statistics', {}),
            "users_count": status.get('users_count', 0),
            "networks_trained": status.get('networks_trained', False),
            "enrollment_active": status.get('enrollment_active', False),
            "authentication_active": status.get('authentication_active', False)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/cleanup")
async def cleanup_resources():
    """
    Limpia recursos del sistema (cámara, MediaPipe, etc).
    
    Returns:
        Dict con resultado de la limpieza
    """
    try:
        manager = get_system_manager()
        manager.cleanup_resources()
        
        return {
            "success": True,
            "message": "Recursos limpiados exitosamente"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/health")
async def system_health_check():
    """
    Health check básico del sistema.
    
    Returns:
        Dict con estado de salud
    """
    try:
        manager = get_system_manager()
        
        is_healthy = (
            manager.state.initialization_level.value >= 2 and  # Al menos nivel 2
            manager.state.database_ready and
            manager.state.enrollment_active
        )
        
        return {
            "healthy": is_healthy,
            "status": "healthy" if is_healthy else "degraded",
            "initialization_level": manager.state.initialization_level.name,
            "enrollment_active": manager.state.enrollment_active,
            "authentication_active": manager.state.authentication_active,
            "version": "2.0.0"
        }
        
    except Exception as e:
        return {
            "healthy": False,
            "status": "error",
            "error": str(e)
        }