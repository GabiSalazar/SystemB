"""
API endpoints para Config Manager
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from app.core.config_manager import get_config_manager, get_config

router = APIRouter(prefix="/config", tags=["Configuration"])


class ConfigUpdateRequest(BaseModel):
    """Request para actualizar configuración"""
    key: str
    value: Any


class ConfigResponse(BaseModel):
    """Respuesta con valor de configuración"""
    key: str
    value: Any
    exists: bool


@router.get("/health")
async def config_health_check():
    """Verifica que el Config Manager esté funcionando"""
    try:
        config_mgr = get_config_manager()
        return {
            "status": "healthy",
            "module": "Config Manager",
            "initialized": True,
            "config_file": config_mgr.config_file,
            "logging_enabled": config_mgr.logger is not None,
            "message": "✅ Módulo 1 cargado correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Config Manager: {str(e)}")


@router.get("/system-info")
async def get_system_info():
    """Obtiene información completa del sistema"""
    try:
        config_mgr = get_config_manager()
        system_info = config_mgr.get_system_info()
        return {
            "status": "success",
            "system_info": system_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo información: {str(e)}")


@router.get("/all")
async def get_all_config():
    """Obtiene toda la configuración actual"""
    try:
        config_mgr = get_config_manager()
        return {
            "status": "success",
            "config": config_mgr._config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo configuración: {str(e)}")


@router.get("/get/{key:path}", response_model=ConfigResponse)
async def get_config_value(key: str):
    """
    Obtiene un valor específico de configuración
    
    Ejemplo: /api/v1/config/get/thresholds.hand_confidence
    """
    try:
        value = get_config(key)
        return ConfigResponse(
            key=key,
            value=value,
            exists=value is not None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo configuración: {str(e)}")


@router.post("/set")
async def set_config_value(request: ConfigUpdateRequest):
    """
    Actualiza un valor de configuración
    
    Body:
    {
        "key": "thresholds.hand_confidence",
        "value": 0.95
    }
    """
    try:
        config_mgr = get_config_manager()
        old_value = config_mgr.get(request.key)
        config_mgr.set(request.key, request.value)
        
        return {
            "status": "success",
            "message": f"Configuración actualizada: {request.key}",
            "old_value": old_value,
            "new_value": request.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando configuración: {str(e)}")


@router.get("/capture-settings")
async def get_capture_settings():
    """Obtiene configuración de captura"""
    try:
        config_mgr = get_config_manager()
        return {
            "samples_per_gesture": config_mgr.get('capture.samples_per_gesture'),
            "gestures_per_user": config_mgr.get('capture.gestures_per_user'),
            "total_captures": config_mgr.get_total_captures(),
            "required_stable_frames": config_mgr.get('capture.required_stable_frames'),
            "capture_delay_seconds": config_mgr.get('capture.capture_delay_seconds')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/thresholds")
async def get_thresholds():
    """Obtiene todos los umbrales de calidad"""
    try:
        return {
            "hand_confidence": get_config('thresholds.hand_confidence'),
            "gesture_confidence": get_config('thresholds.gesture_confidence'),
            "movement_threshold": get_config('thresholds.movement_threshold'),
            "target_hand_size": get_config('thresholds.target_hand_size'),
            "size_tolerance": get_config('thresholds.size_tolerance'),
            "visibility_margin": get_config('thresholds.visibility_margin')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/camera-settings")
async def get_camera_settings():
    """Obtiene configuración de cámara"""
    try:
        return {
            "width": get_config('camera.width'),
            "height": get_config('camera.height'),
            "fps_target": get_config('camera.fps_target'),
            "autofocus": get_config('camera.autofocus'),
            "brightness": get_config('camera.brightness'),
            "contrast": get_config('camera.contrast'),
            "warmup_frames": get_config('camera.warmup_frames')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/available-gestures")
async def get_available_gestures():
    """Obtiene lista de gestos disponibles"""
    try:
        gestures = get_config('available_gestures', [])
        return {
            "count": len(gestures),
            "gestures": gestures
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/gesture-requirements/{gesture_name}")
async def get_gesture_requirements(gesture_name: str):
    """Obtiene requisitos específicos de un gesto"""
    try:
        config_mgr = get_config_manager()
        requirements = config_mgr.get_gesture_requirements(gesture_name)
        
        # Obtener área de referencia
        area_config = get_config(f'reference_area.gesture_areas.{gesture_name}')
        if not area_config:
            area_config = get_config('reference_area.gesture_areas.default')
        
        return {
            "gesture_name": gesture_name,
            "requirements": requirements,
            "area_config": area_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/backup")
async def create_backup():
    """Crea un backup de la configuración actual"""
    try:
        config_mgr = get_config_manager()
        backup_file = config_mgr.backup_config()
        
        if backup_file:
            return {
                "status": "success",
                "message": "Backup creado exitosamente",
                "backup_file": backup_file
            }
        else:
            raise HTTPException(status_code=500, detail="Error creando backup")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/save")
async def save_config():
    """Guarda la configuración actual en el archivo"""
    try:
        config_mgr = get_config_manager()
        config_mgr.save_config()
        return {
            "status": "success",
            "message": "Configuración guardada exitosamente",
            "config_file": config_mgr.config_file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/validate")
async def validate_config():
    """Valida la configuración actual"""
    try:
        config_mgr = get_config_manager()
        is_valid = config_mgr.validate_config()
        
        return {
            "status": "success",
            "valid": is_valid,
            "message": "Configuración válida" if is_valid else "Configuración inválida"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/paths")
async def get_paths():
    """Obtiene todas las rutas configuradas"""
    try:
        paths = get_config('paths', {})
        return {
            "status": "success",
            "paths": paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/model-path")
async def get_model_path():
    """Obtiene la ruta completa del modelo MediaPipe"""
    try:
        config_mgr = get_config_manager()
        model_path = config_mgr.get_model_path()
        
        import os
        exists = os.path.exists(model_path)
        
        return {
            "model_path": model_path,
            "exists": exists,
            "message": "Modelo encontrado" if exists else "Modelo no encontrado"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")