"""
API Endpoints para Enrollment System
Integración completa con RealEnrollmentSystem del core
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from app.core.enrollment_system import (
    get_real_enrollment_system,
    EnrollmentPhase,
    EnrollmentStatus
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ====================================================================
# MODELOS PYDANTIC
# ====================================================================

class EnrollmentStartRequest(BaseModel):
    """Request para iniciar enrollment"""
    user_id: str = Field(..., description="ID único del usuario")
    username: str = Field(..., description="Nombre del usuario")
    gesture_sequence: List[str] = Field(
        ..., 
        description="Secuencia de gestos",
        example=["Victory", "Thumb_Up", "Open_Palm"]
    )


class EnrollmentStartResponse(BaseModel):
    """Response de inicio de enrollment"""
    session_id: str
    user_id: str
    username: str
    gesture_sequence: List[str]
    total_samples_needed: int
    samples_per_gesture: int
    bootstrap_mode: bool
    status: str
    message: str


class FrameProcessResponse(BaseModel):
    """Response de procesamiento de frame"""
    session_id: str
    status: str
    phase: str
    progress: float
    current_gesture: str
    current_gesture_index: int
    total_gestures: int
    samples_collected: int
    samples_needed: int
    sample_captured: bool
    is_real_processing: bool
    bootstrap_mode: bool
    visual_feedback: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    """Response de estado de enrollment"""
    session_id: str
    user_id: str
    username: str
    status: str
    phase: str
    progress_percentage: float
    duration: float
    is_real_session: bool
    bootstrap_mode: bool


class StatsResponse(BaseModel):
    """Response de estadísticas del sistema"""
    enrollment_stats: Dict[str, Any]
    active_sessions: int
    total_users_in_db: int
    database_stats: Dict[str, Any]
    config: Dict[str, Any]
    system_status: Dict[str, Any]


# ====================================================================
# ENDPOINTS
# ====================================================================

@router.post("/enrollment/start", response_model=EnrollmentStartResponse)
async def start_enrollment(request: EnrollmentStartRequest):
    """
    Inicia proceso de enrollment para un nuevo usuario.
    Soporta modo Bootstrap automático.
    """
    try:
        logger.info(f"API: Iniciando enrollment para {request.user_id}")
        
        enrollment_system = get_real_enrollment_system()
        
        session_id = enrollment_system.start_real_enrollment(
            user_id=request.user_id,
            username=request.username,
            gesture_sequence=request.gesture_sequence
        )
        
        session = enrollment_system.active_sessions.get(session_id)
        
        if not session:
            raise HTTPException(status_code=500, detail="Error creando sesión")
        
        return EnrollmentStartResponse(
            session_id=session_id,
            user_id=request.user_id,
            username=request.username,
            gesture_sequence=request.gesture_sequence,
            total_samples_needed=session.total_samples_needed,
            samples_per_gesture=enrollment_system.config.samples_per_gesture,
            bootstrap_mode=enrollment_system.bootstrap_mode,
            status=session.status.value,
            message=f"Enrollment iniciado {'(Modo Bootstrap)' if enrollment_system.bootstrap_mode else ''}"
        )
        
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error iniciando enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/enrollment/{session_id}/frame", response_model=FrameProcessResponse)
async def process_enrollment_frame(session_id: str):
    """
    Procesa un frame para la sesión de enrollment.
    Incluye feedback visual en tiempo real.
    """
    try:
        enrollment_system = get_real_enrollment_system()
        
        result = enrollment_system.process_enrollment_frame(session_id)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return FrameProcessResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrollment/{session_id}/status", response_model=StatusResponse)
async def get_enrollment_status(session_id: str):
    """
    Obtiene el estado actual de una sesión de enrollment.
    """
    try:
        enrollment_system = get_real_enrollment_system()
        
        status = enrollment_system.get_enrollment_status(session_id)
        
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        
        return StatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enrollment/{session_id}/cancel")
async def cancel_enrollment(session_id: str):
    """
    Cancela una sesión de enrollment en curso.
    """
    try:
        enrollment_system = get_real_enrollment_system()
        
        success = enrollment_system.cancel_enrollment(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        return {
            "cancelled": True,
            "session_id": session_id,
            "message": "Enrollment cancelado exitosamente"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelando enrollment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrollment/stats", response_model=StatsResponse)
async def get_enrollment_stats():
    """
    Obtiene estadísticas completas del sistema de enrollment.
    """
    try:
        enrollment_system = get_real_enrollment_system()
        
        stats = enrollment_system.get_system_stats()
        
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enrollment/force-training")
async def force_bootstrap_training(background_tasks: BackgroundTasks):
    """
    Fuerza el entrenamiento de redes (para testing/debugging).
    Solo funciona si hay suficientes datos (2+ usuarios).
    """
    try:
        enrollment_system = get_real_enrollment_system()
        
        result = enrollment_system.force_bootstrap_training()
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "training_initiated": result.get('overall_success', False),
            "bootstrap_disabled": result.get('bootstrap_disabled', False),
            "message": "Entrenamiento completado" if result.get('overall_success') else "Datos insuficientes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forzando entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrollment/bootstrap/status")
async def get_bootstrap_status():
    """
    Obtiene el estado del modo bootstrap.
    """
    try:
        enrollment_system = get_real_enrollment_system()
        
        return {
            "bootstrap_mode": enrollment_system.bootstrap_mode,
            "bootstrap_enrollments": enrollment_system.stats.get('bootstrap_enrollments', 0),
            "networks_trained": enrollment_system.stats.get('networks_trained', False),
            "total_users": len(enrollment_system.database.list_users()),
            "message": "Sistema en modo bootstrap" if enrollment_system.bootstrap_mode else "Sistema en modo normal"
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estado bootstrap: {e}")
        raise HTTPException(status_code=500, detail=str(e))