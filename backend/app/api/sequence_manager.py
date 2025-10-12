"""
API endpoints para Sequence Manager
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.core.sequence_manager import (
    get_sequence_manager,
    SequenceState,
    GestureValidation,
    SequenceEventType
)

router = APIRouter(prefix="/sequence-manager", tags=["Sequence Manager"])


class CreateSequenceRequest(BaseModel):
    """Request para crear secuencia"""
    user_id: str
    gesture_names: List[str]
    sequence_name: Optional[str] = None


class UpdateSequenceRequest(BaseModel):
    """Request para actualizar secuencia"""
    gesture_names: List[str]


class ProcessGestureRequest(BaseModel):
    """Request para procesar gesto"""
    gesture_name: str
    confidence: float
    additional_data: Optional[Dict[str, Any]] = None


class SequenceStatsResponse(BaseModel):
    """Respuesta con estadísticas"""
    total_attempts: int
    successful_sequences: int
    success_rate_percent: float
    registered_users: int
    active_sequence: bool
    sequence_history_size: int
    event_log_size: int


@router.get("/health")
async def sequence_manager_health_check():
    """Verifica que Sequence Manager esté funcionando"""
    try:
        manager = get_sequence_manager()
        
        return {
            "status": "healthy",
            "module": "Sequence Manager",
            "initialized": True,
            "message": "✅ Módulo 8 cargado correctamente",
            "registered_users": len(manager.user_sequences),
            "active_sequence": manager.current_attempt is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Sequence Manager: {str(e)}")


@router.get("/stats", response_model=SequenceStatsResponse)
async def get_statistics():
    """Obtiene estadísticas del gestor de secuencias"""
    try:
        manager = get_sequence_manager()
        stats = manager.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/sequences")
async def create_user_sequence(request: CreateSequenceRequest):
    """Crea una nueva secuencia personalizada para un usuario"""
    try:
        manager = get_sequence_manager()
        
        sequence = manager.create_user_sequence(
            user_id=request.user_id,
            gesture_names=request.gesture_names,
            sequence_name=request.sequence_name
        )
        
        return {
            "status": "success",
            "message": f"Secuencia creada para usuario {request.user_id}",
            "sequence": {
                "user_id": sequence.user_id,
                "sequence_name": sequence.sequence_name,
                "gesture_names": sequence.gesture_names,
                "created_at": sequence.created_at
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/sequences")
async def get_all_sequences():
    """Obtiene todas las secuencias de usuarios"""
    try:
        manager = get_sequence_manager()
        sequences = manager.get_user_sequences()
        
        return {
            "status": "success",
            "count": len(sequences),
            "sequences": sequences
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/sequences/{user_id}")
async def get_user_sequence(user_id: str):
    """Obtiene la secuencia de un usuario específico"""
    try:
        manager = get_sequence_manager()
        sequence = manager.get_user_sequence(user_id)
        
        if sequence is None:
            raise HTTPException(status_code=404, detail=f"No existe secuencia para usuario {user_id}")
        
        return {
            "status": "success",
            "sequence": sequence
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.put("/sequences/{user_id}")
async def update_user_sequence(user_id: str, request: UpdateSequenceRequest):
    """Actualiza la secuencia de un usuario existente"""
    try:
        manager = get_sequence_manager()
        
        success = manager.update_user_sequence(user_id, request.gesture_names)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"No existe secuencia para usuario {user_id}")
        
        return {
            "status": "success",
            "message": f"Secuencia actualizada para usuario {user_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.delete("/sequences/{user_id}")
async def delete_user_sequence(user_id: str):
    """Elimina la secuencia de un usuario"""
    try:
        manager = get_sequence_manager()
        
        success = manager.delete_user_sequence(user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"No existe secuencia para usuario {user_id}")
        
        return {
            "status": "success",
            "message": f"Secuencia eliminada para usuario {user_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/sequences/{user_id}/start")
async def start_sequence(user_id: str):
    """Inicia una secuencia para un usuario"""
    try:
        manager = get_sequence_manager()
        
        success = manager.start_sequence(user_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="No se pudo iniciar la secuencia")
        
        state = manager.get_current_state()
        
        return {
            "status": "success",
            "message": f"Secuencia iniciada para usuario {user_id}",
            "state": state
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/process-gesture")
async def process_gesture(request: ProcessGestureRequest):
    """Procesa un gesto en el contexto de la secuencia actual"""
    try:
        manager = get_sequence_manager()
        
        state = manager.process_gesture(
            gesture_name=request.gesture_name,
            confidence=request.confidence,
            additional_data=request.additional_data
        )
        
        current_state = manager.get_current_state()
        
        return {
            "status": "success",
            "sequence_state": state.value,
            "current_state": current_state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reset")
async def reset_sequence():
    """Reinicia la secuencia actual"""
    try:
        manager = get_sequence_manager()
        
        success = manager.reset_sequence()
        
        return {
            "status": "success" if success else "error",
            "message": "Secuencia reiniciada" if success else "No se pudo reiniciar"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/state")
async def get_current_state():
    """Obtiene el estado actual de la secuencia"""
    try:
        manager = get_sequence_manager()
        state = manager.get_current_state()
        
        return {
            "status": "success",
            "state": state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/history")
async def get_sequence_history(user_id: Optional[str] = None, limit: int = 10):
    """Obtiene historial de secuencias"""
    try:
        manager = get_sequence_manager()
        history = manager.get_sequence_history(user_id=user_id, limit=limit)
        
        return {
            "status": "success",
            "count": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/available-gestures")
async def get_available_gestures():
    """Lista los gestos disponibles para crear secuencias"""
    try:
        manager = get_sequence_manager()
        gestures = manager.list_available_gestures()
        
        return {
            "status": "success",
            "count": len(gestures),
            "gestures": gestures
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/validate-sequence")
async def validate_gesture_sequence(gesture_names: List[str]):
    """Valida si una secuencia de gestos es válida"""
    try:
        manager = get_sequence_manager()
        is_valid, error_message = manager.validate_gesture_sequence(gesture_names)
        
        if is_valid:
            return {
                "status": "success",
                "valid": True,
                "message": "Secuencia válida"
            }
        else:
            return {
                "status": "error",
                "valid": False,
                "message": error_message
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/config")
async def get_sequence_config():
    """Obtiene la configuración del gestor de secuencias"""
    try:
        manager = get_sequence_manager()
        
        return {
            "status": "success",
            "config": manager.config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/states")
async def get_sequence_states():
    """Obtiene los estados posibles de secuencia"""
    return {
        "status": "success",
        "states": [
            {"name": state.name, "value": state.value, "description": _get_state_description(state.value)}
            for state in SequenceState
        ]
    }


@router.get("/validation-modes")
async def get_validation_modes():
    """Obtiene los modos de validación disponibles"""
    return {
        "status": "success",
        "modes": [
            {"name": mode.name, "value": mode.value, "description": _get_validation_description(mode.value)}
            for mode in GestureValidation
        ]
    }


@router.get("/event-types")
async def get_event_types():
    """Obtiene los tipos de eventos disponibles"""
    return {
        "status": "success",
        "events": [
            {"name": event.name, "value": event.value}
            for event in SequenceEventType
        ]
    }


def _get_state_description(state_value: str) -> str:
    """Obtiene descripción de un estado"""
    descriptions = {
        "idle": "Sin secuencia activa",
        "waiting_start": "Esperando inicio de secuencia",
        "in_progress": "Secuencia en progreso",
        "completed": "Secuencia completada exitosamente",
        "failed": "Secuencia falló",
        "timeout": "Secuencia expiró por timeout",
        "interrupted": "Secuencia interrumpida"
    }
    return descriptions.get(state_value, "Sin descripción")


def _get_validation_description(mode_value: str) -> str:
    """Obtiene descripción de un modo de validación"""
    descriptions = {
        "strict": "Validación estricta - debe ser exactamente el gesto esperado",
        "moderate": "Validación moderada - permite pequeñas variaciones",
        "flexible": "Validación flexible - acepta gestos similares"
    }
    return descriptions.get(mode_value, "Sin descripción")