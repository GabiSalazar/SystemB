"""
API endpoints para gestión de API Keys

Permite generar, regenerar, validar y consultar estadísticas de API Keys para la comunicación entre el Plugin y el Sistema biométrico
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional

from app.dependencies.auth import require_admin_token

from app.services.api_key_service import get_api_key_service

router = APIRouter(prefix="/api-keys", tags=["API Keys"])


# ============================================
# MODELOS DE DATOS
# ============================================

class APIKeyResponse(BaseModel):
    """Response con API Key generada"""
    success: bool
    key: str
    created_at: str
    message: str


class APIKeyCurrentResponse(BaseModel):
    """Response con API Key actual"""
    exists: bool
    key: Optional[str] = None
    created_at: Optional[str] = None
    usage_count: Optional[int] = None
    last_used_at: Optional[str] = None


class APIKeyValidateRequest(BaseModel):
    """Request de validación de API Key"""
    api_key: str


class APIKeyValidateResponse(BaseModel):
    """Response de validación"""
    valid: bool
    message: str


class APIKeyStatsResponse(BaseModel):
    """Response con estadísticas de API Key"""
    has_active_key: bool
    key_preview: Optional[str] = None
    created_at: Optional[str] = None
    total_usage: Optional[int] = None
    last_used: Optional[str] = None
    created_by: Optional[str] = None


# ============================================
# ENDPOINTS
# ============================================

@router.get("/current", response_model=APIKeyCurrentResponse, dependencies=[Depends(require_admin_token)])
async def get_current_api_key():
    """
    Obtiene la API Key activa actual.

    Returns:
        APIKeyCurrentResponse: información de la clave activa
    """
    try:
        service = get_api_key_service()
        current = service.get_current_api_key()
        
        if current:
            return APIKeyCurrentResponse(
                exists=True,
                key=current['key'],
                created_at=current['created_at'],
                usage_count=current.get('usage_count', 0),
                last_used_at=current.get('last_used_at')
            )
        else:
            return APIKeyCurrentResponse(exists=False)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo API Key: {str(e)}")


@router.post("/generate", response_model=APIKeyResponse, dependencies=[Depends(require_admin_token)])

async def generate_api_key():
    """
    Genera una nueva API Key e invalida la anterior si existe.

    Returns:
        APIKeyResponse: API Key generada y fecha de creación
    """
    try:
        service = get_api_key_service()
        result = service.create_new_api_key(created_by="admin")
        
        return APIKeyResponse(
            success=True,
            key=result['key'],
            created_at=result['created_at'],
            message="API Key generada exitosamente"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando API Key: {str(e)}")


@router.post("/regenerate", response_model=APIKeyResponse, dependencies=[Depends(require_admin_token)])

async def regenerate_api_key():
    """
    Regenera la API Key activa.

    Invalida la clave anterior y genera una nueva.

    Returns:
        APIKeyResponse: nueva API Key generada
    """
    try:
        service = get_api_key_service()
        result = service.regenerate_api_key(created_by="admin")
        
        return APIKeyResponse(
            success=True,
            key=result['key'],
            created_at=result['created_at'],
            message="API Key regenerada exitosamente. ADVERTENCIA: Actualiza la clave en el Plugin."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error regenerando API Key: {str(e)}")


@router.post("/validate", response_model=APIKeyValidateResponse)
async def validate_api_key(request: APIKeyValidateRequest):
    """
    Valida si una API Key es válida y está activa.

    Args:
        request (APIKeyValidateRequest): API Key a validar

    Returns:
        APIKeyValidateResponse: resultado de la validación
    """
    try:
        service = get_api_key_service()
        is_valid = service.validate_api_key(request.api_key)
        
        if is_valid:
            return APIKeyValidateResponse(
                valid=True,
                message="API Key válida"
            )
        else:
            return APIKeyValidateResponse(
                valid=False,
                message="API Key inválida o inactiva"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validando API Key: {str(e)}")


@router.get("/stats", response_model=APIKeyStatsResponse, dependencies=[Depends(require_admin_token)])

async def get_api_key_stats():
    """
    Obtiene estadísticas de uso de la API Key actual.

    Returns:
        APIKeyStatsResponse: métricas de uso de la API Key
    """
    try:
        service = get_api_key_service()
        stats = service.get_api_key_stats()
        
        return APIKeyStatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@router.get("/health")
async def api_keys_health_check():
    """
    Verifica el estado del módulo de API Keys.
    """
    return {
        "status": "healthy",
        "module": "API Keys Management",
        "initialized": True,
        "message": "Módulo de API Keys disponible"
    }