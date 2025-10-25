"""
API Endpoints para Enrollment System
Integración completa con RealEnrollmentSystem del core
VERSIÓN 100% CORREGIDA - Manejo correcto de start_real_enrollment
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import cv2
import numpy as np

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
    progress: float
    current_gesture: str
    current_gesture_index: int
    total_gestures: int
    samples_collected: int
    samples_needed: int
    sample_captured: bool
    session_completed: bool
    message: Optional[str] = None


class StatusResponse(BaseModel):
    """Response de estado de enrollment"""
    session_id: str
    user_id: str
    username: str
    status: str
    progress_percentage: float
    current_gesture: str
    samples_collected: int
    samples_needed: int
    session_completed: bool


class StatsResponse(BaseModel):
    """Response de estadísticas del sistema"""
    enrollment_stats: Dict[str, Any]
    active_sessions: int
    total_users_in_db: int
    bootstrap_mode: bool


# ====================================================================
# ENDPOINTS
# ====================================================================

@router.get("/enrollment/health")
async def enrollment_health_check():
    """
    Verifica que el módulo de Enrollment System esté operativo.
    """
    try:
        enrollment_system = get_real_enrollment_system()
        return {
            "status": "healthy",
            "module": "Enrollment System",
            "initialized": True,
            "bootstrap_mode": enrollment_system.bootstrap_mode,
            "active_sessions": len(enrollment_system.active_sessions),
            "total_users_in_db": len(enrollment_system.database.list_users()),
            "networks_trained": enrollment_system.stats.get('networks_trained', False),
            "message": "✅ Módulo 14 (Enrollment System) cargado correctamente"
        }
    except Exception as e:
        import traceback
        logger.error(f"Error en health check: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error verificando módulo: {str(e)}")


@router.post("/enrollment/start", response_model=EnrollmentStartResponse)
async def start_enrollment(request: EnrollmentStartRequest):
    """
    Inicia proceso de enrollment para un nuevo usuario.
    Soporta modo Bootstrap automático.
    """
    try:
        logger.info(f"API: Iniciando enrollment para {request.user_id}")
        logger.info(f"  - Username: {request.username}")
        logger.info(f"  - Gestos: {request.gesture_sequence}")
        
        enrollment_system = get_real_enrollment_system()
        
        # ✅ CORRECCIÓN CRÍTICA: start_real_enrollment retorna session_id (str)
        session_id = enrollment_system.start_real_enrollment(
            user_id=request.user_id,
            username=request.username,
            gesture_sequence=request.gesture_sequence
        )
        
        # ✅ Verificar el tipo de retorno
        logger.info(f"  - Retorno de start_real_enrollment: {type(session_id)}")
        logger.info(f"  - Session ID: {session_id}")
        
        # ✅ Si retorna str, buscar en active_sessions
        if isinstance(session_id, str):
            logger.info(f"  - Buscando sesión en active_sessions...")
            logger.info(f"  - Active sessions: {list(enrollment_system.active_sessions.keys())}")
            
            session = enrollment_system.active_sessions.get(session_id)
            
            if not session:
                logger.error(f"❌ Sesión {session_id} NO encontrada en active_sessions")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error: Sesión creada pero no encontrada. ID: {session_id}"
                )
            
            logger.info(f"✅ Sesión encontrada en active_sessions")
        
        # ✅ Si retorna objeto session directamente
        else:
            logger.info(f"  - start_real_enrollment retornó objeto session directamente")
            session = session_id
            session_id = session.session_id
            
            # Agregar a active_sessions si no está
            if session_id not in enrollment_system.active_sessions:
                enrollment_system.active_sessions[session_id] = session
                logger.info(f"  - Sesión agregada a active_sessions")
        
        logger.info(f"✅ Preparando respuesta con session_id: {session_id}")
        
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
        logger.error(f"❌ Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error iniciando enrollment: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.post("/enrollment/{session_id}/frame", response_model=FrameProcessResponse)
async def process_enrollment_frame(
    session_id: str,
    frame: UploadFile = File(..., description="Frame de cámara en formato JPEG")
):
    """
    Procesa un frame enviado desde el frontend.
    El frame debe ser una imagen JPEG capturada desde la cámara del navegador.
    """
    try:
        enrollment_system = get_real_enrollment_system()
        
        # Verificar que la sesión existe
        if session_id not in enrollment_system.active_sessions:
            logger.error(f"Sesión {session_id} no encontrada")
            logger.error(f"Active sessions: {list(enrollment_system.active_sessions.keys())}")
            raise HTTPException(
                status_code=404,
                detail=f"Sesión {session_id} no encontrada"
            )
        
        # Leer el frame del archivo
        frame_data = await frame.read()
        
        # Convertir bytes a imagen OpenCV
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Frame inválido o corrupto")
        
        logger.debug(f"Frame recibido: shape={img.shape}, dtype={img.dtype}")
        
        # Procesar frame CON la imagen recibida
        result = enrollment_system.process_enrollment_frame_with_image(session_id, img)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Construir response
        session = enrollment_system.active_sessions.get(session_id)
        
        return FrameProcessResponse(
            session_id=session_id,
            status=result.get('status', 'processing'),
            progress=result.get('progress', 0.0),
            current_gesture=result.get('current_gesture', ''),
            current_gesture_index=result.get('current_gesture_index', 0),
            total_gestures=len(session.gesture_sequence),
            samples_collected=result.get('samples_collected', 0),
            samples_needed=result.get('samples_needed', 0),
            sample_captured=result.get('sample_captured', False),
            session_completed=result.get('session_completed', False),
            message=result.get('message')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando frame: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrollment/{session_id}/status", response_model=StatusResponse)
async def get_enrollment_status(session_id: str):
    """
    Obtiene el estado actual de una sesión de enrollment.
    """
    try:
        enrollment_system = get_real_enrollment_system()
        
        if session_id not in enrollment_system.active_sessions:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        session = enrollment_system.active_sessions[session_id]
        
        # Calcular samples collected
        valid_samples = [s for s in session.samples if s.is_valid]
        samples_collected = len(valid_samples)
        
        return StatusResponse(
            session_id=session_id,
            user_id=session.user_id,
            username=session.username,
            status=session.status.value,
            progress_percentage=(samples_collected / session.total_samples_needed * 100) if session.total_samples_needed > 0 else 0,
            current_gesture=session.current_gesture,
            samples_collected=samples_collected,
            samples_needed=session.total_samples_needed,
            session_completed=session.status in [EnrollmentStatus.COMPLETED, EnrollmentStatus.FAILED]
        )
        
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
            raise HTTPException(status_code=404, detail="Sesión no encontrada o ya finalizada")
        
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
        
        return StatsResponse(
            enrollment_stats=enrollment_system.stats,
            active_sessions=len(enrollment_system.active_sessions),
            total_users_in_db=len(enrollment_system.database.list_users()),
            bootstrap_mode=enrollment_system.bootstrap_mode
        )
        
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
        # Usar system_manager para entrenamiento
        from app.core.system_manager import get_system_manager
        
        manager = get_system_manager()
        enrollment_system = get_real_enrollment_system()
        
        if manager.state.users_count < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Se necesitan al menos 2 usuarios. Actualmente: {manager.state.users_count}"
            )
        
        # Entrenar redes
        result = manager.train_networks(force=True)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=500,
                detail=result.get('message', 'Error entrenando redes')
            )
        
        # Actualizar bootstrap mode
        enrollment_system.bootstrap_mode = enrollment_system._check_bootstrap_needed()
        
        return {
            "training_initiated": True,
            "bootstrap_disabled": not enrollment_system.bootstrap_mode,
            "networks_trained": manager.state.networks_trained,
            "message": "Entrenamiento completado exitosamente"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forzando entrenamiento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrollment/bootstrap/status")
async def get_bootstrap_status():
    """
    Obtiene el estado del modo bootstrap.
    """
    try:
        enrollment_system = get_real_enrollment_system()
        
        users = enrollment_system.database.list_users()
        users_count = len(users)
        
        return {
            "bootstrap_mode": enrollment_system.bootstrap_mode,
            "bootstrap_enrollments": enrollment_system.stats.get('bootstrap_enrollments', 0),
            "networks_trained": enrollment_system.stats.get('networks_trained', False),
            "total_users": users_count,
            "min_users_required": 2,
            "can_train": users_count >= 2,
            "message": "Sistema en modo bootstrap - Primeros usuarios" if enrollment_system.bootstrap_mode else "Sistema en modo normal - Redes disponibles"
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estado bootstrap: {e}")
        raise HTTPException(status_code=500, detail=str(e))