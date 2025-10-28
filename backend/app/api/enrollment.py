"""
API Endpoints para gestión de enrollment/registro biométrico
VERSIÓN CORREGIDA CON BOOTSTRAP STATUS
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import base64
import cv2
import numpy as np

from app.core.system_manager import get_system_manager

router = APIRouter()


# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class EnrollmentStartRequest(BaseModel):
    """Request para iniciar enrollment"""
    user_id: str
    username: str
    gesture_sequence: Optional[List[str]] = None


class EnrollmentStartResponse(BaseModel):
    """Response de inicio de enrollment"""
    success: bool
    session_id: str
    message: str
    user_id: str
    username: str
    gesture_sequence: List[str]
    total_gestures: int
    samples_per_gesture: int
    total_samples_needed: int


class ProcessFrameRequest(BaseModel):
    """Request para procesar frame"""
    session_id: str
    frame_data: str  # Base64 encoded image
    current_gesture_index: int


class ProcessFrameResponse(BaseModel):
    """Response de procesamiento de frame"""
    success: bool
    message: str
    current_gesture: str
    current_gesture_index: int
    samples_captured: int
    samples_needed: int
    gesture_completed: bool
    all_gestures_completed: bool
    quality_score: Optional[float] = None
    feedback: Optional[str] = None
    error: Optional[str] = None


class EnrollmentStatusResponse(BaseModel):
    """Response de estado de enrollment"""
    success: bool
    session_active: bool
    user_id: Optional[str] = None
    username: Optional[str] = None
    current_gesture: Optional[str] = None
    current_gesture_index: int
    total_gestures: int
    samples_captured: int
    samples_needed: int
    progress_percentage: float
    message: str


class EnrollmentCompleteResponse(BaseModel):
    """Response de finalización de enrollment"""
    success: bool
    message: str
    user_id: str
    username: str
    templates_created: int
    enrollment_time: float


class BootstrapStatusResponse(BaseModel):
    """Response de estado de bootstrap"""
    bootstrap_active: bool
    users_count: int
    min_users_required: int
    templates_count: int
    can_train: bool
    needs_bootstrap: bool
    message: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/enrollment/start", response_model=EnrollmentStartResponse)
async def start_enrollment(request: EnrollmentStartRequest):
    """
    Inicia una nueva sesión de enrollment.
    
    Args:
        request: Datos del usuario y secuencia de gestos
    
    Returns:
        EnrollmentStartResponse con información de la sesión
    """
    try:
        manager = get_system_manager()
        
        # Verificar que el sistema esté listo
        if not manager.state.enrollment_active:
            raise HTTPException(
                status_code=503,
                detail="Sistema de enrollment no está activo"
            )
        
        print(f"🎬 Iniciando enrollment - User: {request.user_id}, Username: {request.username}")
        
        # Iniciar sesión de enrollment
        result = manager.start_enrollment_session(
            user_id=request.user_id,
            username=request.username,
            gesture_sequence=request.gesture_sequence
        )
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Error iniciando enrollment')
            )
        
        session = result['session']
        
        print(f"✅ Sesión creada: {session['session_id']}")
        print(f"   Gestos: {session['gesture_sequence']}")
        print(f"   Total muestras: {session['total_samples_needed']}")
        
        return EnrollmentStartResponse(
            success=True,
            session_id=session['session_id'],
            message=result.get('message', 'Sesión de enrollment iniciada'),
            user_id=session['user_id'],
            username=session['username'],
            gesture_sequence=session['gesture_sequence'],
            total_gestures=session['total_gestures'],
            samples_per_gesture=session['samples_per_gesture'],
            total_samples_needed=session['total_samples_needed']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error iniciando enrollment: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/enrollment/process-frame", response_model=ProcessFrameResponse)
async def process_enrollment_frame(request: ProcessFrameRequest):
    """
    Procesa un frame durante el enrollment.
    
    Args:
        request: Session ID, frame en base64 y gesture index
    
    Returns:
        ProcessFrameResponse con resultado del procesamiento
    """
    try:
        manager = get_system_manager()
        
        # Decodificar imagen base64
        try:
            # Remover prefijo data:image si existe
            frame_data = request.frame_data
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_bytes = base64.b64decode(frame_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("No se pudo decodificar la imagen")
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error decodificando imagen: {str(e)}"
            )
        
        # Procesar frame
        result = manager.process_enrollment_frame(
            session_id=request.session_id,
            frame=frame,
            current_gesture_index=request.current_gesture_index
        )
        
        return ProcessFrameResponse(
            success=result.get('success', False),
            message=result.get('message', ''),
            current_gesture=result.get('current_gesture', ''),
            current_gesture_index=result.get('current_gesture_index', 0),
            samples_captured=result.get('samples_captured', 0),
            samples_needed=result.get('samples_needed', 0),
            gesture_completed=result.get('gesture_completed', False),
            all_gestures_completed=result.get('all_gestures_completed', False),
            quality_score=result.get('quality_score'),
            feedback=result.get('feedback'),
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error procesando frame: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/enrollment/status/{session_id}", response_model=EnrollmentStatusResponse)
async def get_enrollment_status(session_id: str):
    """
    Obtiene el estado actual de una sesión de enrollment.
    
    Args:
        session_id: ID de la sesión
    
    Returns:
        EnrollmentStatusResponse con estado actual
    """
    try:
        manager = get_system_manager()
        
        result = manager.get_enrollment_session_status(session_id)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=404,
                detail=result.get('message', 'Sesión no encontrada')
            )
        
        session = result['session']
        
        return EnrollmentStatusResponse(
            success=True,
            session_active=session['active'],
            user_id=session.get('user_id'),
            username=session.get('username'),
            current_gesture=session.get('current_gesture'),
            current_gesture_index=session['current_gesture_index'],
            total_gestures=session['total_gestures'],
            samples_captured=session['samples_captured'],
            samples_needed=session['samples_needed'],
            progress_percentage=session['progress_percentage'],
            message=result.get('message', '')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error obteniendo estado: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/enrollment/complete/{session_id}", response_model=EnrollmentCompleteResponse)
async def complete_enrollment(session_id: str):
    """
    Completa una sesión de enrollment y genera templates.
    
    Args:
        session_id: ID de la sesión a completar
    
    Returns:
        EnrollmentCompleteResponse con resultado
    """
    try:
        manager = get_system_manager()
        
        print(f"🎯 Completando enrollment - Session: {session_id}")
        
        result = manager.complete_enrollment_session(session_id)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Error completando enrollment')
            )
        
        print(f"✅ Enrollment completado")
        print(f"   User: {result['user_id']}")
        print(f"   Templates: {result['templates_created']}")
        print(f"   Tiempo: {result['enrollment_time']:.2f}s")
        
        return EnrollmentCompleteResponse(
            success=True,
            message=result.get('message', 'Enrollment completado exitosamente'),
            user_id=result['user_id'],
            username=result['username'],
            templates_created=result['templates_created'],
            enrollment_time=result['enrollment_time']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error completando enrollment: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@router.delete("/enrollment/cancel/{session_id}")
async def cancel_enrollment(session_id: str):
    """
    Cancela una sesión de enrollment.
    
    Args:
        session_id: ID de la sesión a cancelar
    
    Returns:
        Dict con resultado de la cancelación
    """
    try:
        manager = get_system_manager()
        
        print(f"🚫 Cancelando enrollment - Session: {session_id}")
        
        result = manager.cancel_enrollment_session(session_id)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=404,
                detail=result.get('message', 'Sesión no encontrada')
            )
        
        print(f"✅ Sesión cancelada")
        
        return {
            "success": True,
            "message": result.get('message', 'Sesión cancelada exitosamente')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error cancelando enrollment: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/enrollment/bootstrap/status", response_model=BootstrapStatusResponse)
async def get_bootstrap_status():
    """
    Obtiene el estado del modo bootstrap.
    CORREGIDO: Cuenta templates directamente desde archivos
    
    Returns:
        BootstrapStatusResponse con información de bootstrap
    """
    try:
        manager = get_system_manager()
        status = manager.get_system_status()
        
        users_count = status.get('users_count', 0)
        networks_trained = status.get('networks_trained', False)
        bootstrap_active = status.get('bootstrap_mode', False)
        min_users = 2
        
        # Calcular templates totales - CORREGIDO v2
        templates_count = 0
        if hasattr(manager, 'database') and manager.database:
            try:
                import os
                templates_dir = os.path.join(manager.database.db_path, 'templates')
                if os.path.exists(templates_dir):
                    # Contar archivos .json directamente
                    templates_count = len([
                        f for f in os.listdir(templates_dir) 
                        if f.endswith('.json')
                    ])
            except Exception as e:
                print(f"Error contando templates: {e}")
                templates_count = 0
        
        can_train = users_count >= min_users and not networks_trained
        needs_bootstrap = users_count < min_users
        
        message = ""
        if needs_bootstrap:
            message = f"Se necesitan {min_users - users_count} usuario(s) más para entrenar"
        elif not networks_trained:
            message = "Sistema listo para entrenar redes neuronales"
        else:
            message = "Redes ya entrenadas - Sistema completamente operativo"
        
        print(f"Bootstrap Status: users={users_count}, trained={networks_trained}, can_train={can_train}, templates={templates_count}")
        
        return BootstrapStatusResponse(
            bootstrap_active=bootstrap_active,
            users_count=users_count,
            min_users_required=min_users,
            templates_count=templates_count,
            can_train=can_train,
            needs_bootstrap=needs_bootstrap,
            message=message
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Error obteniendo bootstrap status: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR en get_bootstrap_status: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/enrollment/sessions")
async def list_enrollment_sessions():
    """
    Lista todas las sesiones de enrollment activas.
    
    Returns:
        Dict con lista de sesiones
    """
    try:
        manager = get_system_manager()
        
        sessions = manager.list_enrollment_sessions()
        
        return {
            "success": True,
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Error listando sesiones: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/enrollment/available-gestures")
async def get_available_gestures():
    """
    Obtiene la lista de gestos disponibles para enrollment.
    
    Returns:
        Dict con gestos disponibles
    """
    try:
        manager = get_system_manager()
        
        gestures = manager.get_available_gestures()
        
        return {
            "success": True,
            "gestures": gestures,
            "total_gestures": len(gestures)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrollment/config")
async def get_enrollment_config():
    """
    Obtiene la configuración actual del sistema de enrollment.
    
    Returns:
        Dict con configuración
    """
    try:
        manager = get_system_manager()
        
        config = manager.get_enrollment_config()
        
        return {
            "success": True,
            "config": config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))