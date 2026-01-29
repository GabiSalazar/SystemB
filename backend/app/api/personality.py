"""
API Endpoints para gestión de perfiles de personalidad
Integración con BiometricDatabase
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# from app.core.biometric_database import get_biometric_database, PersonalityProfile
from app.core.supabase_biometric_storage import get_biometric_database, PersonalityProfile
from app.services.plugin_webhook_service import get_plugin_webhook_service
from app.core.system_manager import get_system_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class PersonalityQuestionnaireRequest(BaseModel):
    """Request para guardar respuestas del cuestionario"""
    user_id: str = Field(..., description="ID del usuario")
    responses: List[int] = Field(..., description="Lista de 10 respuestas (1-5)")
    
    @validator('responses')
    def validate_responses(cls, v):
        if len(v) != 10:
            raise ValueError('Se requieren exactamente 10 respuestas')
        
        for response in v:
            if not isinstance(response, int) or response < 1 or response > 5:
                raise ValueError('Cada respuesta debe ser un número entre 1 y 5')
        
        return v


class PersonalityQuestionnaireResponse(BaseModel):
    """Response de guardado del cuestionario"""
    success: bool
    message: str
    user_id: str
    raw_responses: str


class PersonalityProfileResponse(BaseModel):
    """Response con el perfil de personalidad"""
    success: bool
    user_id: str
    has_profile: bool
    profile: Optional[Dict[str, Any]] = None


class PersonalityCheckResponse(BaseModel):
    """Response de verificación de perfil"""
    user_id: str
    has_personality_profile: bool
    message: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/personality/submit", response_model=PersonalityQuestionnaireResponse)
async def submit_personality_questionnaire(request: PersonalityQuestionnaireRequest):
    """
    Guarda las respuestas del cuestionario de personalidad.
    
    Args:
        request: Datos del cuestionario (user_id, responses)
        
    Returns:
        Confirmación de guardado
    """
    try:
        database = get_biometric_database()
        
        # VERIFICAR QUE EL USUARIO EXISTE (método correcto)
        user_profile = database.get_user(request.user_id)
        if not user_profile:
            raise HTTPException(
                status_code=404,
                detail=f"Usuario {request.user_id} no encontrado"
            )
        
        # VERIFICAR SI YA TIENE UN PERFIL DE PERSONALIDAD
        if database.has_personality_profile(request.user_id):
            raise HTTPException(
                status_code=400,
                detail="Este usuario ya completó el cuestionario de personalidad"
            )
        
        # CREAR PERFIL DE PERSONALIDAD
        personality_profile = PersonalityProfile.from_responses(
            user_id=request.user_id,
            responses=request.responses
        )
        
        # GUARDAR EN LA BASE DE DATOS
        success = database.store_personality_profile(personality_profile)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Error guardando el perfil de personalidad"
            )
        
        # logger.info(f"Cuestionario de personalidad guardado para usuario: {request.user_id}")
        # logger.info(f"   Respuestas: {personality_profile.raw_responses}")
        
        # return PersonalityQuestionnaireResponse(
        #     success=True,
        #     message="Cuestionario de personalidad guardado exitosamente",
        #     user_id=request.user_id,
        #     raw_responses=personality_profile.raw_responses
        # )
        
        logger.info(f"Cuestionario de personalidad guardado para usuario: {request.user_id}")
        logger.info(f"   Respuestas: {personality_profile.raw_responses}")
        
        # ENVIAR RESULTADO AL PLUGIN (si tiene callback_url configurado)
        try:
            # Buscar sesión de enrollment activa para obtener callback_url
            manager = get_system_manager()
            
            # Buscar en sesiones activas
            callback_url = None
            session_token = None
            
            if manager.enrollment_system and manager.enrollment_system.active_sessions:
                for session_id, session in manager.enrollment_system.active_sessions.items():
                    if session.user_id == request.user_id:
                        callback_url = session.callback_url
                        session_token = session.session_token
                        break
            
            # Si encontramos callback_url, enviar al Plugin
            if callback_url and session_token:
                logger.info(f"Enviando resultado de registro al Plugin")
                logger.info(f"   Callback URL: {callback_url}")
                
                webhook_service = get_plugin_webhook_service()
                
                plugin_api_key = os.getenv('PLUGIN_API_KEY')
                
                if not plugin_api_key:
                    logger.error(f"PLUGIN_API_KEY no configurado en .env")
                    logger.warning(f"No se puede enviar resultado al Plugin")
                else:
                    webhook_service.set_api_key(plugin_api_key)
                    
                    success = webhook_service.send_registration_result(
                        callback_url=callback_url,
                        user_id=request.user_id,
                        email=user_profile.email,
                        session_token=session_token,
                        raw_responses=personality_profile.raw_responses
                    )
                
                if success:
                    logger.info(f"Resultado enviado exitosamente al Plugin")
                else:
                    logger.warning(f"NONo se pudo enviar resultado al Plugin")
            else:
                logger.info(f"No hay callback_url configurado - No se envía al Plugin")
                
        except Exception as e:
            logger.error(f"Error enviando resultado al Plugin: {e}")
            # No fallar el guardado si falla el envío al Plugin
        
        return PersonalityQuestionnaireResponse(
            success=True,
            message="Cuestionario de personalidad guardado exitosamente",
            user_id=request.user_id,
            raw_responses=personality_profile.raw_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error guardando cuestionario: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )


@router.get("/personality/profile/{user_id}", response_model=PersonalityProfileResponse)
async def get_personality_profile(user_id: str):
    """
    Obtiene el perfil de personalidad de un usuario.
    
    Args:
        user_id: ID del usuario
        
    Returns:
        Perfil de personalidad
    """
    try:
        database = get_biometric_database()
        
        # VERIFICAR QUE EL USUARIO EXISTE
        user_profile = database.get_user(user_id)
        if not user_profile:
            raise HTTPException(
                status_code=404,
                detail=f"Usuario {user_id} no encontrado"
            )
        
        # OBTENER PERFIL DE PERSONALIDAD
        personality_profile = database.get_personality_profile(user_id)
        
        if personality_profile:
            return PersonalityProfileResponse(
                success=True,
                user_id=user_id,
                has_profile=True,
                profile=personality_profile.to_dict()
            )
        else:
            return PersonalityProfileResponse(
                success=True,
                user_id=user_id,
                has_profile=False,
                profile=None
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo perfil: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )


@router.get("/personality/check/{user_id}", response_model=PersonalityCheckResponse)
async def check_personality_profile(user_id: str):
    """
    Verifica si un usuario tiene perfil de personalidad.
    
    Args:
        user_id: ID del usuario
        
    Returns:
        Estado de completitud del cuestionario
    """
    try:
        database = get_biometric_database()
        
        # VERIFICAR EXISTENCIA DEL PERFIL
        has_profile = database.has_personality_profile(user_id)
        
        return PersonalityCheckResponse(
            user_id=user_id,
            has_personality_profile=has_profile,
            message="Cuestionario completado" if has_profile else "Cuestionario pendiente"
        )
        
    except Exception as e:
        logger.error(f"Error verificando perfil: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )


@router.get("/personality/health")
async def personality_health_check():
    """
    Verifica que el módulo de personalidad esté funcionando.
    
    Returns:
        Estado del módulo
    """
    try:
        database = get_biometric_database()
        
        # Verificar que el directorio existe
        personality_dir = database.db_path / "personality_profiles"
        
        return {
            "status": "healthy",
            "module": "Personality Profiles",
            "initialized": True,
            "message": "Módulo de personalidad funcionando correctamente",
            "storage_path": str(personality_dir),
            "directory_exists": personality_dir.exists()
        }
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en módulo de personalidad: {str(e)}"
        )