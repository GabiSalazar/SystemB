"""
API endpoints para Reference Area Manager
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Tuple

from app.core.reference_area_manager import (
    get_reference_area_manager,
    AreaType
)

router = APIRouter(prefix="/reference-area", tags=["Reference Area"])


class AreaDimensionsResponse(BaseModel):
    """Respuesta con dimensiones de área"""
    width_ratio: float
    height_ratio: float
    center_y_offset: float
    area_type: str


class AreaCoordinatesResponse(BaseModel):
    """Respuesta con coordenadas de área"""
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    width: int
    height: int


class VisualFeedbackResponse(BaseModel):
    """Respuesta con feedback visual"""
    instruction_text: str
    text_offset: int
    area_color: Tuple[int, int, int]
    requirements_text: str


@router.get("/health")
async def reference_area_health_check():
    """Verifica que Reference Area Manager esté funcionando"""
    try:
        area_mgr = get_reference_area_manager()
        
        return {
            "status": "healthy",
            "module": "Reference Area Manager",
            "initialized": True,
            "message": "✅ Módulo 5 cargado correctamente",
            "use_single_area": area_mgr.use_single_area
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Reference Area Manager: {str(e)}")


@router.get("/stats")
async def get_area_stats():
    """Obtiene estadísticas del gestor de áreas"""
    try:
        area_mgr = get_reference_area_manager()
        stats = area_mgr.get_area_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/gestures")
async def get_configured_gestures():
    """Obtiene lista de gestos configurados"""
    try:
        area_mgr = get_reference_area_manager()
        
        # Filtrar keys que no son gestos
        gestures = [key for key in area_mgr.area_config.keys() 
                   if key not in ['default', 'corner_size', 'line_thickness']]
        
        return {
            "status": "success",
            "count": len(gestures),
            "gestures": gestures,
            "default_available": "default" in area_mgr.area_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/dimensions/{gesture_name}", response_model=AreaDimensionsResponse)
async def get_gesture_dimensions(gesture_name: str, width: int = 1280, height: int = 720):
    """
    Obtiene las dimensiones del área para un gesto específico
    
    Query params:
    - width: Ancho del frame (default: 1280)
    - height: Alto del frame (default: 720)
    
    Ejemplo: /api/v1/reference-area/dimensions/Open_Palm?width=1280&height=720
    """
    try:
        area_mgr = get_reference_area_manager()
        
        frame_shape = (height, width)
        dimensions = area_mgr.get_area_dimensions(gesture_name, frame_shape)
        
        return AreaDimensionsResponse(
            width_ratio=dimensions.width_ratio,
            height_ratio=dimensions.height_ratio,
            center_y_offset=dimensions.center_y_offset,
            area_type=dimensions.area_type.value
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/coordinates/{gesture_name}", response_model=AreaCoordinatesResponse)
async def get_gesture_coordinates(gesture_name: str, width: int = 1280, height: int = 720):
    """
    Obtiene las coordenadas exactas del área para un gesto
    
    Query params:
    - width: Ancho del frame (default: 1280)
    - height: Alto del frame (default: 720)
    
    Ejemplo: /api/v1/reference-area/coordinates/Victory?width=1280&height=720
    """
    try:
        area_mgr = get_reference_area_manager()
        
        frame_shape = (height, width)
        coords = area_mgr.calculate_area_coordinates(gesture_name, frame_shape)
        
        return coords
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/requirements/{gesture_name}")
async def get_gesture_requirements(gesture_name: str):
    """
    Obtiene los requisitos de área para un gesto específico
    
    Ejemplo: /api/v1/reference-area/requirements/Pointing_Up
    """
    try:
        area_mgr = get_reference_area_manager()
        requirements = area_mgr.get_gesture_requirements(gesture_name)
        
        return {
            "status": "success",
            "gesture_name": gesture_name,
            "requirements": requirements
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/visual-feedback/{gesture_name}", response_model=VisualFeedbackResponse)
async def get_visual_feedback(gesture_name: str):
    """
    Obtiene la configuración de feedback visual para un gesto
    
    Ejemplo: /api/v1/reference-area/visual-feedback/Thumb_Up
    """
    try:
        area_mgr = get_reference_area_manager()
        feedback = area_mgr.get_visual_feedback(gesture_name)
        
        return feedback
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/area-types")
async def get_area_types():
    """Obtiene los tipos de área disponibles"""
    return {
        "types": [area_type.value for area_type in AreaType],
        "descriptions": {
            "narrow_high": "Área estrecha y alta (para Pointing_Up)",
            "wide_high": "Área ancha y alta (para Victory, ILoveYou)",
            "medium_high": "Área media y alta (para Thumb_Up/Down)",
            "standard": "Área estándar (para Open_Palm, Closed_Fist)"
        }
    }


@router.get("/colors")
async def get_color_config():
    """Obtiene la configuración de colores"""
    try:
        area_mgr = get_reference_area_manager()
        
        return {
            "status": "success",
            "colors": area_mgr.color_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/text-config")
async def get_text_config():
    """Obtiene la configuración de texto"""
    try:
        area_mgr = get_reference_area_manager()
        
        # Convertir font a string legible
        text_config = area_mgr.text_config.copy()
        text_config['font'] = 'FONT_HERSHEY_SIMPLEX'
        
        return {
            "status": "success",
            "text_config": text_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/clear-cache")
async def clear_dimensions_cache():
    """Limpia el cache de dimensiones calculadas"""
    try:
        area_mgr = get_reference_area_manager()
        area_mgr.clear_cache()
        
        return {
            "status": "success",
            "message": "Cache de dimensiones limpiado correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/single-area-config")
async def get_single_area_config():
    """Obtiene la configuración del área única (cuando está habilitada)"""
    try:
        area_mgr = get_reference_area_manager()
        
        return {
            "status": "success",
            "use_single_area": area_mgr.use_single_area,
            "single_area_config": area_mgr.single_area_config if area_mgr.use_single_area else None,
            "message": "Área única habilitada" if area_mgr.use_single_area else "Áreas específicas por gesto"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")