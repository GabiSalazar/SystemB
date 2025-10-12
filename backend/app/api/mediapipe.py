"""
API endpoints para MediaPipe Processor
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.core.mediapipe_processor import (
    get_mediapipe_processor, 
    release_mediapipe,
    HandSide
)

router = APIRouter(prefix="/mediapipe", tags=["MediaPipe"])


class GestureInfo(BaseModel):
    """Información de un gesto"""
    name: str
    is_available: bool
    requirements: str
    index: int


class ProcessingStats(BaseModel):
    """Estadísticas de procesamiento"""
    frames_processed: int
    hands_detected: int
    gestures_recognized: int
    hand_detection_rate_percent: float
    gesture_recognition_rate_percent: float
    is_initialized: bool
    available_gestures_count: int
    model_path: str


@router.get("/health")
async def mediapipe_health_check():
    """Verifica que MediaPipe esté funcionando"""
    try:
        processor = get_mediapipe_processor()
        
        if processor is None:
            return {
                "status": "error",
                "module": "MediaPipe Processor",
                "initialized": False,
                "message": "❌ MediaPipe no disponible (modelo no encontrado)"
            }
        
        return {
            "status": "healthy",
            "module": "MediaPipe Processor",
            "initialized": processor.is_initialized,
            "message": "✅ Módulo 3 cargado correctamente",
            "gestures_count": len(processor.available_gestures)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en MediaPipe: {str(e)}")


@router.get("/stats", response_model=ProcessingStats)
async def get_mediapipe_stats():
    """Obtiene estadísticas de procesamiento"""
    try:
        processor = get_mediapipe_processor()
        
        if processor is None:
            raise HTTPException(status_code=503, detail="MediaPipe no disponible")
        
        stats = processor.get_processing_stats()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reset-stats")
async def reset_mediapipe_stats():
    """Reinicia las estadísticas de procesamiento"""
    try:
        processor = get_mediapipe_processor()
        
        if processor is None:
            raise HTTPException(status_code=503, detail="MediaPipe no disponible")
        
        processor.reset_stats()
        
        return {
            "status": "success",
            "message": "Estadísticas reiniciadas correctamente"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/gestures")
async def get_available_gestures():
    """Obtiene lista de gestos disponibles"""
    try:
        processor = get_mediapipe_processor()
        
        if processor is None:
            raise HTTPException(status_code=503, detail="MediaPipe no disponible")
        
        return {
            "status": "success",
            "count": len(processor.available_gestures),
            "gestures": processor.available_gestures
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/gestures/{gesture_name}", response_model=GestureInfo)
async def get_gesture_info(gesture_name: str):
    """
    Obtiene información detallada de un gesto específico
    
    Ejemplo: /api/v1/mediapipe/gestures/Open_Palm
    """
    try:
        processor = get_mediapipe_processor()
        
        if processor is None:
            raise HTTPException(status_code=503, detail="MediaPipe no disponible")
        
        gesture_info = processor.get_gesture_info(gesture_name)
        return gesture_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/config")
async def get_mediapipe_config():
    """Obtiene la configuración actual de MediaPipe"""
    try:
        processor = get_mediapipe_processor()
        
        if processor is None:
            raise HTTPException(status_code=503, detail="MediaPipe no disponible")
        
        return {
            "status": "success",
            "hands_config": processor.hands_config,
            "gesture_config": processor.gesture_config,
            "model_path": processor.model_path,
            "model_exists": processor.is_initialized
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/validate/hand-confidence")
async def validate_hand_confidence(confidence: float, threshold: Optional[float] = None):
    """
    Valida si una confianza de mano es suficiente
    
    Body: { "confidence": 0.95, "threshold": 0.90 }
    """
    try:
        processor = get_mediapipe_processor()
        
        if processor is None:
            raise HTTPException(status_code=503, detail="MediaPipe no disponible")
        
        is_valid = processor.validate_hand_confidence(confidence, threshold)
        
        return {
            "status": "success",
            "confidence": confidence,
            "threshold": threshold or 0.90,
            "is_valid": is_valid,
            "message": "Confianza suficiente" if is_valid else "Confianza insuficiente"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/validate/gesture-match")
async def validate_gesture_match(detected: str, target: str):
    """
    Valida si un gesto detectado coincide con el objetivo
    
    Body: { "detected": "Open_Palm", "target": "Open_Palm" }
    """
    try:
        processor = get_mediapipe_processor()
        
        if processor is None:
            raise HTTPException(status_code=503, detail="MediaPipe no disponible")
        
        is_valid = processor.validate_gesture_match(detected, target)
        
        return {
            "status": "success",
            "detected_gesture": detected,
            "target_gesture": target,
            "is_match": is_valid,
            "message": "Gesto correcto" if is_valid else "Gesto incorrecto"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/hand-sides")
async def get_hand_sides():
    """Obtiene los lados de mano disponibles"""
    return {
        "sides": [side.value for side in HandSide],
        "note": "MediaPipe corrige automáticamente la lateralidad (efecto espejo de la cámara)"
    }


@router.post("/release")
async def release_mediapipe_endpoint():
    """Libera los recursos de MediaPipe"""
    try:
        release_mediapipe()
        
        return {
            "status": "success",
            "message": "Recursos de MediaPipe liberados"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/model-info")
async def get_model_info():
    """Obtiene información del modelo MediaPipe"""
    try:
        processor = get_mediapipe_processor()
        
        if processor is None:
            return {
                "status": "error",
                "model_path": "Unknown",
                "model_exists": False,
                "download_url": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
                "message": "❌ Modelo no encontrado. Descárgalo del URL proporcionado."
            }
        
        import os
        model_exists = os.path.exists(processor.model_path)
        model_size = os.path.getsize(processor.model_path) if model_exists else 0
        
        return {
            "status": "success",
            "model_path": processor.model_path,
            "model_exists": model_exists,
            "model_size_mb": round(model_size / (1024 * 1024), 2) if model_exists else 0,
            "message": "✅ Modelo cargado correctamente" if model_exists else "❌ Modelo no encontrado"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")