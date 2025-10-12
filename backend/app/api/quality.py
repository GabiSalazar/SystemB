"""
API endpoints para Quality Validator
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.core.quality_validator import (
    get_quality_validator,
    DistanceStatus,
    ValidationStatus
)

router = APIRouter(prefix="/quality", tags=["Quality Validation"])


class ThresholdsUpdate(BaseModel):
    """Request para actualizar umbrales"""
    hand_confidence: Optional[float] = None
    gesture_confidence: Optional[float] = None
    movement_threshold: Optional[float] = None
    target_hand_size: Optional[float] = None
    size_tolerance: Optional[float] = None
    required_stable_frames: Optional[int] = None


class ValidationStats(BaseModel):
    """Estadísticas de validación"""
    validations_performed: int
    valid_captures: int
    success_rate_percent: float
    current_stable_frames: int
    landmark_history_size: int
    thresholds: Dict[str, float]


@router.get("/health")
async def quality_health_check():
    """Verifica que Quality Validator esté funcionando"""
    try:
        validator = get_quality_validator()
        
        return {
            "status": "healthy",
            "module": "Quality Validator",
            "initialized": True,
            "message": "✅ Módulo 4 cargado correctamente",
            "roi_normalization_enabled": validator.use_roi_normalization
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Quality Validator: {str(e)}")


@router.get("/stats", response_model=ValidationStats)
async def get_validation_stats():
    """Obtiene estadísticas de validación"""
    try:
        validator = get_quality_validator()
        stats = validator.get_validation_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reset-stats")
async def reset_validation_stats():
    """Reinicia las estadísticas de validación"""
    try:
        validator = get_quality_validator()
        validator.reset_stats()
        
        return {
            "status": "success",
            "message": "Estadísticas de validación reiniciadas"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reset-stability")
async def reset_stability_counter():
    """Reinicia el contador de estabilidad"""
    try:
        validator = get_quality_validator()
        validator.reset_stability_counter()
        
        return {
            "status": "success",
            "message": "Contador de estabilidad reiniciado"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/thresholds")
async def get_thresholds():
    """Obtiene los umbrales actuales de validación"""
    try:
        validator = get_quality_validator()
        
        return {
            "status": "success",
            "thresholds": validator.thresholds,
            "visibility_config": validator.visibility_config,
            "roi_normalization_enabled": validator.use_roi_normalization
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/thresholds")
async def update_thresholds(update: ThresholdsUpdate):
    """
    Actualiza los umbrales de validación
    
    Body:
    {
        "hand_confidence": 0.95,
        "gesture_confidence": 0.65,
        "movement_threshold": 0.02,
        "target_hand_size": 0.25,
        "size_tolerance": 0.08,
        "required_stable_frames": 2
    }
    """
    try:
        validator = get_quality_validator()
        
        # Construir diccionario solo con valores no None
        new_thresholds = {}
        if update.hand_confidence is not None:
            new_thresholds['hand_confidence'] = update.hand_confidence
        if update.gesture_confidence is not None:
            new_thresholds['gesture_confidence'] = update.gesture_confidence
        if update.movement_threshold is not None:
            new_thresholds['movement_threshold'] = update.movement_threshold
        if update.target_hand_size is not None:
            new_thresholds['target_hand_size'] = update.target_hand_size
        if update.size_tolerance is not None:
            new_thresholds['size_tolerance'] = update.size_tolerance
        if update.required_stable_frames is not None:
            new_thresholds['required_stable_frames'] = update.required_stable_frames
        
        if not new_thresholds:
            raise HTTPException(status_code=400, detail="No se proporcionaron umbrales para actualizar")
        
        validator.update_thresholds(new_thresholds)
        
        return {
            "status": "success",
            "message": "Umbrales actualizados correctamente",
            "updated_thresholds": new_thresholds,
            "current_thresholds": validator.thresholds
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/distance-statuses")
async def get_distance_statuses():
    """Obtiene los estados de distancia disponibles"""
    return {
        "statuses": [status.value for status in DistanceStatus],
        "descriptions": {
            "muy_lejos": "La mano está muy lejos de la cámara",
            "muy_cerca": "La mano está muy cerca de la cámara",
            "correcta": "La mano está a la distancia correcta"
        }
    }


@router.get("/validation-statuses")
async def get_validation_statuses():
    """Obtiene los estados de validación disponibles"""
    return {
        "statuses": [status.value for status in ValidationStatus],
        "descriptions": {
            "valid": "Validación exitosa",
            "invalid": "Validación fallida",
            "pending": "Validación pendiente"
        }
    }


@router.get("/gesture-validation-points/{gesture_name}")
async def get_gesture_validation_points(gesture_name: str):
    """
    Obtiene los puntos de validación para un gesto específico
    
    Ejemplo: /api/v1/quality/gesture-validation-points/Open_Palm
    """
    try:
        validator = get_quality_validator()
        
        important_points, core_points, tolerance = validator._get_gesture_validation_points(gesture_name)
        
        return {
            "status": "success",
            "gesture_name": gesture_name,
            "important_points": important_points,
            "important_points_count": len(important_points),
            "core_points": core_points,
            "core_points_count": len(core_points),
            "tolerance": tolerance,
            "description": f"Se requiere {int(tolerance*100)}% de puntos importantes dentro del área"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/quality-criteria")
async def get_quality_criteria():
    """Obtiene los criterios de calidad que se evalúan"""
    return {
        "criteria": [
            {
                "name": "hand_confidence",
                "weight": 25,
                "description": "Confianza de detección de la mano"
            },
            {
                "name": "gesture_confidence",
                "weight": 20,
                "description": "Confianza del gesto detectado"
            },
            {
                "name": "visibility",
                "weight": 15,
                "description": "Todos los puntos visibles en el frame"
            },
            {
                "name": "area_coverage",
                "weight": 15,
                "description": "Mano dentro del área de referencia"
            },
            {
                "name": "size_quality",
                "weight": 10,
                "description": "Tamaño de mano correcto (distancia)"
            },
            {
                "name": "stability",
                "weight": 10,
                "description": "Mano estable sin movimiento"
            },
            {
                "name": "extension",
                "weight": 5,
                "description": "Mano extendida (para gestos que lo requieren)"
            }
        ],
        "total_weight": 100,
        "min_passing_score": 80,
        "note": "Un score de 80+ generalmente indica captura lista"
    }


@router.get("/config")
async def get_quality_config():
    """Obtiene la configuración completa del validador"""
    try:
        validator = get_quality_validator()
        
        return {
            "status": "success",
            "thresholds": validator.thresholds,
            "visibility_config": validator.visibility_config,
            "area_config": validator.area_config,
            "roi_normalization_enabled": validator.use_roi_normalization,
            "landmark_history_size": len(validator.landmark_history),
            "max_history_size": validator.landmark_history.maxlen
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")