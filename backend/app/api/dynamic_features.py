"""
API endpoints para Dynamic Features Extractor
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from app.core.dynamic_features_extractor import (
    get_real_dynamic_features_extractor,
    TransitionPhase,
    MotionType
)

router = APIRouter(prefix="/dynamic-features", tags=["Dynamic Features"])


class DynamicStatsResponse(BaseModel):
    """Respuesta con estadísticas de extracción dinámica"""
    frames_processed: int
    transitions_detected: int
    successful_extractions: int
    success_rate_percent: float
    feature_dimension: int
    sequence_length: int
    current_gesture: str
    transition_active: bool
    buffer_size: int
    detected_transitions_count: int
    extractor_type: str
    version: str


@router.get("/health")
async def dynamic_features_health_check():
    """Verifica que Dynamic Features Extractor esté funcionando"""
    try:
        extractor = get_real_dynamic_features_extractor()
        
        return {
            "status": "healthy",
            "module": "Dynamic Features Extractor",
            "initialized": True,
            "message": "✅ Módulo 7 cargado correctamente",
            "feature_dimension": 320,
            "version": "2.0 - 100% Real",
            "type": "REAL - Sin simulación"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Dynamic Features Extractor: {str(e)}")


@router.get("/stats", response_model=DynamicStatsResponse)
async def get_extraction_stats():
    """Obtiene estadísticas de extracción de características dinámicas"""
    try:
        extractor = get_real_dynamic_features_extractor()
        stats = extractor.get_extraction_stats_real()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reset-stats")
async def reset_extraction_stats():
    """Reinicia estadísticas de extracción"""
    try:
        extractor = get_real_dynamic_features_extractor()
        extractor.reset_stats()
        
        return {
            "status": "success",
            "message": "Estadísticas de extracción dinámica reiniciadas correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reset-state")
async def reset_extractor_state():
    """Reinicia el estado del extractor (buffer temporal)"""
    try:
        extractor = get_real_dynamic_features_extractor()
        extractor.reset_state()
        
        return {
            "status": "success",
            "message": "Estado del extractor reiniciado correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/transition-phases")
async def get_transition_phases():
    """Obtiene las fases de transición disponibles"""
    return {
        "status": "success",
        "phases": [
            {"name": phase.name, "value": phase.value, "description": _get_phase_description(phase.value)}
            for phase in TransitionPhase
        ]
    }


@router.get("/motion-types")
async def get_motion_types():
    """Obtiene los tipos de movimiento disponibles"""
    return {
        "status": "success",
        "motion_types": [
            {"name": mtype.name, "value": mtype.value, "description": _get_motion_description(mtype.value)}
            for mtype in MotionType
        ]
    }


@router.get("/dimensions")
async def get_feature_dimensions():
    """Obtiene las dimensiones de cada categoría de características dinámicas"""
    return {
        "status": "success",
        "total_dimension": 320,
        "categories": {
            "velocity_features": {
                "dimension": 70,
                "description": "Características de velocidad (picos, promedios, patrones)"
            },
            "acceleration_features": {
                "dimension": 65,
                "description": "Características de aceleración (jerk, suavidad)"
            },
            "trajectory_features": {
                "dimension": 85,
                "description": "Características de trayectoria (longitud, curvatura, eficiencia)"
            },
            "timing_features": {
                "dimension": 40,
                "description": "Características temporales (duración, intervalos, fases)"
            },
            "rhythm_features": {
                "dimension": 35,
                "description": "Características de ritmo (periodicidad, cambios)"
            },
            "transition_features": {
                "dimension": 25,
                "description": "Características de transición (suavidad, complejidad)"
            }
        }
    }


@router.get("/config")
async def get_dynamic_config():
    """Obtiene la configuración del extractor dinámico"""
    try:
        extractor = get_real_dynamic_features_extractor()
        
        return {
            "status": "success",
            "config": extractor.dynamic_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/buffer-info")
async def get_buffer_info():
    """Obtiene información del buffer temporal"""
    try:
        extractor = get_real_dynamic_features_extractor()
        
        return {
            "status": "success",
            "sequence_length": extractor.sequence_length,
            "current_buffer_size": len(extractor.temporal_buffer),
            "buffer_utilization": len(extractor.temporal_buffer) / extractor.sequence_length * 100,
            "has_previous_frame": extractor.previous_frame is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/last-transition")
async def get_last_transition():
    """Obtiene información de la última transición detectada"""
    try:
        extractor = get_real_dynamic_features_extractor()
        last_transition = extractor.get_last_transition_real()
        
        if last_transition:
            return {
                "status": "success",
                "has_transition": True,
                "transition": {
                    "start_frame": last_transition.start_frame,
                    "end_frame": last_transition.end_frame,
                    "start_gesture": last_transition.start_gesture,
                    "end_gesture": last_transition.end_gesture,
                    "transition_type": last_transition.transition_type,
                    "duration_ms": last_transition.duration_ms,
                    "motion_type": last_transition.motion_type.value,
                    "confidence": last_transition.confidence,
                    "frame_count": len(last_transition.transition_frames)
                }
            }
        else:
            return {
                "status": "success",
                "has_transition": False,
                "message": "No hay transiciones detectadas aún"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def _get_phase_description(phase_value: str) -> str:
    """Obtiene descripción de una fase de transición"""
    descriptions = {
        "stable": "Gesto estable sin cambios",
        "preparing": "Preparando cambio de gesto",
        "transitioning": "En proceso de transición",
        "completing": "Completando transición",
        "stabilizing": "Estabilizando nuevo gesto"
    }
    return descriptions.get(phase_value, "Sin descripción")


def _get_motion_description(motion_value: str) -> str:
    """Obtiene descripción de un tipo de movimiento"""
    descriptions = {
        "smooth": "Movimiento suave y continuo",
        "abrupt": "Movimiento abrupto con cambios bruscos",
        "curved": "Movimiento curvo con trayectoria arqueada",
        "linear": "Movimiento lineal directo",
        "oscillatory": "Movimiento oscilatorio con vaivenes"
    }
    return descriptions.get(motion_value, "Sin descripción")