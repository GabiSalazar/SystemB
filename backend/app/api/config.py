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
            "message": "Módulo 1 cargado correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Config Manager: {str(e)}")


@router.get("/system-info")
async def get_system_info():
    """
    Obtiene información del sistema y su configuración actual

    Args:
        None

    Returns:
        dict:
            - status (str): resultado de la operación
            - system_info (dict): información del sistema
    """
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
    """
    Obtiene la configuración del sistema

    Args:
        None

    Returns:
        dict:
            - status (str): resultado de la operación
            - config (dict): configuración del sistema
    """
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
    Obtiene un valor específico de configuración a partir de su clave

    Args:
        key (str): clave de la configuración

    Returns:
        ConfigResponse:
            - key (str): clave solicitada
            - value (Any): valor asociado a la clave
            - exists (bool): indica si la clave existe en la configuración
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
    Actualiza un valor específico de la configuración del sistema

    Args:
        request (ConfigUpdateRequest):
            - key (str): clave de la configuración a actualizar
            - value (Any): nuevo valor a asignar

    Returns:
        dict:
            - status (str): estado de la operación
            - message (str): descripción del resultado
            - old_value (Any): valor anterior de la configuración
            - new_value (Any): nuevo valor asignado
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
    """
    Obtiene la configuración relacionada con el proceso de captura biométrica.

    Returns:
        dict:
            - samples_per_gesture (int): número de muestras requeridas por gesto
            - gestures_per_user (int): número de gestos requeridos por usuario
            - total_captures (int): total de capturas necesarias en el proceso
            - required_stable_frames (int): cantidad de frames estables requeridos
            - capture_delay_seconds (float): retardo en segundos antes de la captura
    """
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
    """
    Obtiene los umbrales de calidad utilizados en el sistema biométrico.

    Returns:
        dict:
            - hand_confidence (float): umbral mínimo de confianza para detección de mano
            - gesture_confidence (float): umbral mínimo de confianza del gesto reconocido
            - movement_threshold (float): umbral de movimiento permitido durante la captura
            - target_hand_size (float): tamaño objetivo de la mano en el frame
            - size_tolerance (float): tolerancia permitida respecto al tamaño objetivo
            - visibility_margin (float): margen mínimo de visibilidad requerido
    """
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
    """
    Obtiene la configuración de la cámara

    Returns:
        dict:
            - width (int): ancho de captura de la cámara en píxeles
            - height (int): alto de captura de la cámara en píxeles
            - fps_target (int | float): frames por segundo objetivo
            - autofocus (bool): indica si el autofocus está habilitado
            - brightness (int | float): nivel de brillo configurado
            - contrast (int | float): nivel de contraste configurado
            - warmup_frames (int): número de frames de calentamiento antes de captura
    """
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
    """
    Obtiene la lista de gestos disponibles

    Returns:
        dict:
            - count (int): número total de gestos disponibles
            - gestures (list[str]): lista de nombres de gestos configurados en el sistema
    """
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
    """
    Obtiene los requisitos específicos asociados a un gesto biométrico.

    Args:
        gesture_name (str): nombre del gesto a consultar

    Returns:
        dict:
            - gesture_name (str): nombre del gesto consultado
            - requirements (dict): requisitos configurados para el gesto
            - area_config (dict): configuración del área de referencia asociada al gesto
    """
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
    """
    Crea un respaldo de la configuración actual del sistema

    Returns:
        dict:
            - status (str): resultado de la operación
            - message (str): mensaje descriptivo del resultado
            - backup_file (str): ruta o nombre del archivo de respaldo generado
    """
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
    """
    Guarda la configuración actual en el archivo de configuración persistente

    Returns:
        dict:
            - status (str): resultado de la operación
            - message (str): mensaje descriptivo del resultado
            - config_file (str): ruta del archivo de configuración guardado
    """
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