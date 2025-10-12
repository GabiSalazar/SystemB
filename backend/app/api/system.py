"""
API Endpoints para gestión del sistema biométrico
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.core.system_manager import get_system_manager

router = APIRouter()


class SystemStatusResponse(BaseModel):
    """Response de estado del sistema"""
    initialization_level: str
    users_count: int
    networks_trained: bool
    database_ready: bool
    enrollment_active: bool
    authentication_active: bool
    uptime: str
    version: str
    status: str


@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Obtiene el estado actual del sistema."""
    try:
        manager = get_system_manager()
        status = manager.get_system_status()
        return SystemStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/initialize")
async def initialize_system():
    """Inicializa el sistema biométrico."""
    try:
        manager = get_system_manager()
        
        if manager.state.initialization_level.value > 0:
            return {
                "initialized": True,
                "message": "Sistema ya inicializado",
                "level": manager.state.initialization_level.name
            }
        
        success = manager.initialize_system()
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=manager.state.error_message or "Error inicializando sistema"
            )
        
        return {
            "initialized": True,
            "message": "Sistema inicializado exitosamente",
            "level": manager.state.initialization_level.name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/train")
async def train_networks():
    """Entrena las redes neuronales."""
    try:
        manager = get_system_manager()
        
        if manager.state.users_count < 2:
            raise HTTPException(
                status_code=400,
                detail="Se necesitan al menos 2 usuarios registrados"
            )
        
        success = manager.train_networks()
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Error entrenando redes"
            )
        
        return {
            "trained": True,
            "message": "Redes entrenadas exitosamente",
            "networks_trained": manager.state.networks_trained,
            "authentication_active": manager.state.authentication_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))