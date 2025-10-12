"""
API endpoints para Camera Manager
"""

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Dict, Any, Optional
import cv2
import numpy as np
import base64

from app.core.camera_manager import get_camera_manager, release_camera

router = APIRouter(prefix="/camera", tags=["Camera"])


class CameraStatsResponse(BaseModel):
    """Respuesta con estadísticas de cámara"""
    frame_count: int
    last_frame_time: float
    is_initialized: bool
    width: int
    height: int
    fps: float
    brightness: float
    contrast: float
    timestamp: str


@router.get("/health")
async def camera_health_check():
    """Verifica que la cámara esté funcionando"""
    try:
        camera_mgr = get_camera_manager()
        
        if camera_mgr is None:
            return {
                "status": "error",
                "module": "Camera Manager",
                "initialized": False,
                "message": "❌ Cámara no disponible"
            }
        
        is_healthy = camera_mgr.check_camera_health()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "module": "Camera Manager",
            "initialized": camera_mgr.is_initialized,
            "healthy": is_healthy,
            "message": "✅ Módulo 2 cargado correctamente" if is_healthy else "⚠️ Cámara con problemas"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Camera Manager: {str(e)}")


@router.get("/stats", response_model=CameraStatsResponse)
async def get_camera_stats():
    """Obtiene estadísticas de la cámara"""
    try:
        camera_mgr = get_camera_manager()
        
        if camera_mgr is None:
            raise HTTPException(status_code=503, detail="Cámara no disponible")
        
        stats = camera_mgr.get_camera_stats()
        
        if 'error' in stats:
            raise HTTPException(status_code=500, detail=stats['error'])
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@router.get("/config")
async def get_camera_config():
    """Obtiene la configuración actual de la cámara"""
    try:
        camera_mgr = get_camera_manager()
        
        if camera_mgr is None:
            raise HTTPException(status_code=503, detail="Cámara no disponible")
        
        return {
            "status": "success",
            "config": camera_mgr.config
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/initialize")
async def initialize_camera():
    """Inicializa la cámara manualmente"""
    try:
        camera_mgr = get_camera_manager()
        
        if camera_mgr is None:
            raise HTTPException(status_code=500, detail="No se pudo crear instancia de cámara")
        
        if camera_mgr.is_initialized:
            return {
                "status": "success",
                "message": "Cámara ya estaba inicializada",
                "initialized": True
            }
        
        success = camera_mgr.initialize()
        
        return {
            "status": "success" if success else "error",
            "message": "Cámara inicializada correctamente" if success else "Error inicializando cámara",
            "initialized": success
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reset")
async def reset_camera():
    """Reinicia la cámara"""
    try:
        camera_mgr = get_camera_manager()
        
        if camera_mgr is None:
            raise HTTPException(status_code=503, detail="Cámara no disponible")
        
        success = camera_mgr.reset_camera()
        
        return {
            "status": "success" if success else "error",
            "message": "Cámara reiniciada correctamente" if success else "Error reiniciando cámara",
            "initialized": success
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/release")
async def release_camera_endpoint():
    """Libera los recursos de la cámara"""
    try:
        release_camera()
        
        return {
            "status": "success",
            "message": "Recursos de cámara liberados"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/capture/test")
async def capture_test_frame():
    """
    Captura un frame de prueba y lo devuelve como base64
    
    NOTA: Solo para testing. Para streaming en producción usar WebSocket.
    """
    try:
        camera_mgr = get_camera_manager()
        
        if camera_mgr is None:
            raise HTTPException(status_code=503, detail="Cámara no disponible")
        
        ret, frame = camera_mgr.capture_frame()
        
        if not ret or frame is None:
            raise HTTPException(status_code=500, detail="Error capturando frame")
        
        # Convertir a JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Convertir a base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "frame": f"data:image/jpeg;base64,{frame_base64}",
            "width": frame.shape[1],
            "height": frame.shape[0],
            "frame_count": camera_mgr.frame_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/capture/high-quality")
async def capture_high_quality_frame():
    """
    Captura un frame de alta calidad
    
    Usa estabilización y selección del mejor frame de 3 capturas.
    """
    try:
        camera_mgr = get_camera_manager()
        
        if camera_mgr is None:
            raise HTTPException(status_code=503, detail="Cámara no disponible")
        
        ret, frame = camera_mgr.capture_high_quality_frame()
        
        if not ret or frame is None:
            raise HTTPException(status_code=500, detail="Error capturando frame de alta calidad")
        
        # Convertir a JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Convertir a base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "frame": f"data:image/jpeg;base64,{frame_base64}",
            "width": frame.shape[1],
            "height": frame.shape[0],
            "quality": "high"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/test-image")
async def get_test_image():
    """
    Devuelve una imagen de prueba directamente (para verificar que endpoints funcionan)
    """
    try:
        # Crear imagen de prueba (640x480, gradiente de colores)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Gradiente horizontal (azul a verde)
        for x in range(640):
            test_image[:, x, 0] = int(255 * (1 - x/640))  # Azul
            test_image[:, x, 1] = int(255 * x/640)        # Verde
        
        # Agregar texto
        cv2.putText(test_image, "Camera Test Image", (180, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Convertir a JPEG
        _, buffer = cv2.imencode('.jpg', test_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Devolver como imagen directamente
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")