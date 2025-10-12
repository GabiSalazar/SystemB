"""
API endpoints para ROI Normalization System
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.core.roi_normalization import get_roi_normalization_system, ROIDistanceStatus

router = APIRouter(prefix="/roi", tags=["ROI Normalization"])


class ROIStatsResponse(BaseModel):
    """Respuesta con estadísticas del sistema ROI"""
    total_extractions: int
    valid_extractions: int
    success_rate_percent: float
    too_far_count: int
    too_close_count: int
    avg_processing_time_ms: float
    total_processing_time_ms: float
    config: Dict[str, Any]


@router.get("/health")
async def roi_health_check():
    """
    Verifica que el sistema ROI esté funcionando
    """
    try:
        roi_system = get_roi_normalization_system()
        return {
            "status": "healthy",
            "module": "ROI Normalization System",
            "initialized": True,
            "message": "✅ Módulo 0 cargado correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en ROI System: {str(e)}")


@router.get("/statistics", response_model=ROIStatsResponse)
async def get_roi_statistics():
    """
    Obtiene estadísticas del sistema ROI
    """
    try:
        roi_system = get_roi_normalization_system()
        stats = roi_system.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@router.post("/reset-statistics")
async def reset_roi_statistics():
    """
    Resetea las estadísticas del sistema ROI
    """
    try:
        roi_system = get_roi_normalization_system()
        roi_system.reset_statistics()
        return {
            "status": "success",
            "message": "Estadísticas reseteadas correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reseteando estadísticas: {str(e)}")


@router.get("/config")
async def get_roi_config():
    """
    Obtiene la configuración actual del sistema ROI
    """
    try:
        roi_system = get_roi_normalization_system()
        return {
            "min_roi_width": roi_system.min_roi_width,
            "max_roi_width": roi_system.max_roi_width,
            "target_size": roi_system.target_size,
            "roi_padding": roi_system.roi_padding,
            "apply_sharpening": roi_system.apply_sharpening,
            "apply_contrast_enhancement": roi_system.apply_contrast_enhancement,
            "sharpening_threshold": roi_system.sharpening_threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo configuración: {str(e)}")


@router.get("/distance-statuses")
async def get_distance_statuses():
    """
    Obtiene los estados de distancia disponibles
    """
    return {
        "statuses": [status.value for status in ROIDistanceStatus],
        "descriptions": {
            "too_far": "Usuario muy lejos (ROI < 150px)",
            "too_close": "Usuario muy cerca (ROI > 600px)",
            "acceptable": "Distancia correcta (150-600px)",
            "unknown": "No detectado"
        }
    }
    
@router.get("/feedback-levels")
async def get_feedback_levels():
    """
    Obtiene los niveles de feedback disponibles
    """
    from app.core.visual_feedback import FeedbackLevel
    
    return {
        "levels": [level.value for level in FeedbackLevel],
        "descriptions": {
            "success": "Acción exitosa - Verde",
            "warning": "Advertencia - Amarillo",
            "error": "Error o requisito no cumplido - Rojo",
            "info": "Información general - Azul",
            "bootstrap": "Modo inicial de entrenamiento - Magenta"
        },
        "colors_bgr": {
            "success": [0, 255, 0],
            "warning": [0, 255, 255],
            "error": [0, 0, 255],
            "info": [255, 200, 0],
            "bootstrap": [255, 0, 255]
        }
    }


@router.post("/test-feedback")
async def test_feedback_generation():
    """
    Prueba la generación de feedback visual (sin frame real)
    """
    try:
        from app.core.visual_feedback import get_visual_feedback_manager
        
        feedback_manager = get_visual_feedback_manager()
        
        # Simular sesión de prueba
        session_info = {
            'bootstrap_mode': False,
            'samples_captured': 3,
            'samples_needed': 8
        }
        
        # Generar feedback sin assessment (debería pedir mostrar mano)
        messages = feedback_manager.generate_real_time_feedback(
            quality_assessment=None,
            target_gesture="Open_Palm",
            session_info=session_info
        )
        
        return {
            "status": "success",
            "message": "Feedback generado correctamente",
            "messages_count": len(messages),
            "messages": [
                {
                    "text": msg.text,
                    "level": msg.level.value,
                    "priority": msg.priority,
                    "icon": msg.icon,
                    "action": msg.action,
                    "details": msg.details,
                    "progress": msg.progress
                }
                for msg in messages
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando feedback: {str(e)}")