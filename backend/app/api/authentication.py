"""
API Endpoints para Authentication System
Integración completa con RealAuthenticationSystem
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from app.core.authentication_system import (
    get_real_authentication_system,
    AuthenticationMode,
    AuthenticationStatus,
    SecurityLevel
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ====================================================================
# MODELOS PYDANTIC
# ====================================================================

class VerificationStartRequest(BaseModel):
    """Request para iniciar verificación 1:1"""
    user_id: str = Field(..., description="ID del usuario a verificar")
    security_level: str = Field(default="standard", description="Nivel de seguridad")
    required_sequence: Optional[List[str]] = Field(None, description="Secuencia de gestos")
    ip_address: str = Field(default="localhost", description="IP del cliente")


class IdentificationStartRequest(BaseModel):
    """Request para iniciar identificación 1:N"""
    security_level: str = Field(default="standard", description="Nivel de seguridad")
    ip_address: str = Field(default="localhost", description="IP del cliente")


class AuthenticationStartResponse(BaseModel):
    """Response de inicio de autenticación"""
    session_id: str
    mode: str
    user_id: Optional[str]
    security_level: str
    message: str


class FrameProcessResponse(BaseModel):
    """Response de procesamiento de frame"""
    session_id: str
    status: str
    phase: str
    progress: float
    message: str
    frames_processed: int
    frame_processed: bool
    is_real_processing: bool


class AuthenticationStatusResponse(BaseModel):
    """Response de estado de autenticación"""
    session_id: str
    mode: str
    user_id: Optional[str]
    status: str
    phase: str
    duration: float
    progress: float
    is_real_session: bool


# ====================================================================
# ENDPOINTS
# ====================================================================

@router.post("/authentication/verify/start", response_model=AuthenticationStartResponse)
async def start_verification(request: VerificationStartRequest):
    """
    Inicia proceso de verificación 1:1.
    """
    try:
        logger.info(f"API: Iniciando verificación para {request.user_id}")
        
        auth_system = get_real_authentication_system()
        
        # Inicializar si es necesario
        if not auth_system.is_initialized:
            if not auth_system.initialize_real_system():
                raise HTTPException(status_code=500, detail="Error inicializando sistema")
        
        # Convertir security_level
        security_level = SecurityLevel[request.security_level.upper()]
        
        session_id = auth_system.start_real_verification(
            user_id=request.user_id,
            security_level=security_level,
            required_sequence=request.required_sequence,
            ip_address=request.ip_address
        )
        
        return AuthenticationStartResponse(
            session_id=session_id,
            mode="verification",
            user_id=request.user_id,
            security_level=request.security_level,
            message="Verificación iniciada"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error iniciando verificación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/authentication/identify/start", response_model=AuthenticationStartResponse)
async def start_identification(request: IdentificationStartRequest):
    """
    Inicia proceso de identificación 1:N.
    """
    try:
        logger.info("API: Iniciando identificación 1:N")
        
        auth_system = get_real_authentication_system()
        
        # Inicializar si es necesario
        if not auth_system.is_initialized:
            if not auth_system.initialize_real_system():
                raise HTTPException(status_code=500, detail="Error inicializando sistema")
        
        security_level = SecurityLevel[request.security_level.upper()]
        
        session_id = auth_system.start_real_identification(
            security_level=security_level,
            ip_address=request.ip_address
        )
        
        return AuthenticationStartResponse(
            session_id=session_id,
            mode="identification",
            user_id=None,
            security_level=request.security_level,
            message="Identificación iniciada"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error iniciando identificación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/authentication/{session_id}/frame", response_model=FrameProcessResponse)
async def process_authentication_frame(session_id: str):
    """
    Procesa un frame para la sesión de autenticación.
    """
    try:
        auth_system = get_real_authentication_system()
        
        result = auth_system.process_real_authentication_frame(session_id)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return FrameProcessResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/authentication/{session_id}/status", response_model=AuthenticationStatusResponse)
async def get_authentication_status(session_id: str):
    """
    Obtiene el estado de una sesión de autenticación.
    """
    try:
        auth_system = get_real_authentication_system()
        
        status = auth_system.get_real_authentication_status(session_id)
        
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        
        return AuthenticationStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/authentication/{session_id}/cancel")
async def cancel_authentication(session_id: str):
    """
    Cancela una sesión de autenticación.
    """
    try:
        auth_system = get_real_authentication_system()
        
        success = auth_system.cancel_real_authentication(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        return {
            "cancelled": True,
            "session_id": session_id,
            "message": "Autenticación cancelada"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/authentication/users")
async def get_available_users():
    """
    Obtiene usuarios disponibles para autenticación.
    """
    try:
        auth_system = get_real_authentication_system()
        
        users = auth_system.get_real_available_users()
        
        return {
            "users": users,
            "total": len(users)
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo usuarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/authentication/stats")
async def get_authentication_stats():
    """
    Obtiene estadísticas del sistema de autenticación.
    """
    try:
        auth_system = get_real_authentication_system()
        
        stats = auth_system.get_real_system_statistics()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/authentication/initialize")
async def initialize_authentication_system():
    """
    Inicializa el sistema de autenticación.
    """
    try:
        auth_system = get_real_authentication_system()
        
        if auth_system.is_initialized:
            return {
                "initialized": True,
                "message": "Sistema ya inicializado"
            }
        
        success = auth_system.initialize_real_system()
        
        if not success:
            raise HTTPException(status_code=500, detail="Error inicializando sistema")
        
        return {
            "initialized": True,
            "message": "Sistema inicializado exitosamente"
        }
        
    except Exception as e:
        logger.error(f"Error inicializando: {e}")
        raise HTTPException(status_code=500, detail=str(e))