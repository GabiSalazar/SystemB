from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.config import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API para Sistema Biométrico de Reconocimiento de Gestos",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Biometric Gesture System API",
        "version": settings.VERSION,
        "status": "online"
    }

@app.get(f"{settings.API_V1_STR}/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION
    }

# ===== IMPORTAR ROUTERS =====
from app.api import (
    roi, 
    config, 
    camera, 
    mediapipe, 
    quality, 
    reference_area, 
    anatomical_features, 
    dynamic_features,
    sequence_manager,
    siamese_anatomical,
    siamese_dynamic,
    feature_preprocessor,
    score_fusion,
    biometric_database,
    enrollment,
    authentication,
    system
)

# ===== REGISTRAR ROUTERS =====
app.include_router(roi.router, prefix=settings.API_V1_STR)
app.include_router(config.router, prefix=settings.API_V1_STR)
app.include_router(camera.router, prefix=settings.API_V1_STR)
app.include_router(mediapipe.router, prefix=settings.API_V1_STR)
app.include_router(quality.router, prefix=settings.API_V1_STR)
app.include_router(reference_area.router, prefix=settings.API_V1_STR)
app.include_router(anatomical_features.router, prefix=settings.API_V1_STR)
app.include_router(dynamic_features.router, prefix=settings.API_V1_STR)
app.include_router(sequence_manager.router, prefix=settings.API_V1_STR)
app.include_router(siamese_anatomical.router, prefix=settings.API_V1_STR)
app.include_router(siamese_dynamic.router, prefix=settings.API_V1_STR)
app.include_router(feature_preprocessor.router, prefix=settings.API_V1_STR)
app.include_router(score_fusion.router, prefix=settings.API_V1_STR)
app.include_router(biometric_database.router, prefix=settings.API_V1_STR)
app.include_router(enrollment.router, prefix=settings.API_V1_STR)
app.include_router(authentication.router, prefix=settings.API_V1_STR)
app.include_router(system.router, prefix=settings.API_V1_STR)



# Evento de inicio
@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"📁 Base directory: {settings.BASE_DIR}")
    logger.info(f"🔧 Debug mode: {settings.DEBUG}")
    
    # Crear directorios necesarios
    settings.BIOMETRIC_DATA_DIR.mkdir(exist_ok=True)
    settings.MODELS_DIR.mkdir(exist_ok=True)
    logger.info("✅ Directories initialized")
    
    logger.info("=" * 80)
    logger.info("INICIALIZANDO MÓDULOS DEL SISTEMA")
    logger.info("=" * 80)
    
    # 1. Config Manager (Módulo 1)
    try:
        from app.core.config_manager import get_config_manager
        config_mgr = get_config_manager()
        logger.info("✅ MÓDULO 1: Config Manager - OK")
    except Exception as e:
        logger.error(f"❌ MÓDULO 1: Config Manager - FAILED: {e}")
    
    # 2. ROI Normalization (Módulo 0)
    try:
        from app.core.roi_normalization import get_roi_normalization_system
        roi_system = get_roi_normalization_system()
        logger.info("✅ MÓDULO 0: ROI Normalization System - OK")
    except Exception as e:
        logger.error(f"❌ MÓDULO 0: ROI Normalization - FAILED: {e}")
    
    # 3. Visual Feedback Manager (Módulo 0.5)
    try:
        from app.core.visual_feedback import get_visual_feedback_manager
        feedback_manager = get_visual_feedback_manager()
        logger.info("✅ MÓDULO 0.5: Visual Feedback Manager - OK")
    except Exception as e:
        logger.error(f"❌ MÓDULO 0.5: Visual Feedback - FAILED: {e}")
    
    # ✅ NUEVO: Inicialización del Sistema Biométrico Completo
    logger.info("=" * 80)
    logger.info("🧠 INICIALIZANDO SISTEMA BIOMÉTRICO COMPLETO")
    logger.info("=" * 80)
    
    try:
        from app.core.system_manager import get_system_manager
        
        manager = get_system_manager()
        success = manager.initialize_system()
        
        if success:
            status = manager.get_system_status()
            logger.info("✅ Sistema biométrico inicializado correctamente")
            logger.info(f"📊 Nivel de inicialización: {status['initialization_level']}")
            logger.info(f"👥 Usuarios registrados: {status['users_count']}")
            logger.info(f"🧠 Redes entrenadas: {'✅ SÍ' if status['networks_trained'] else '❌ NO'}")
            logger.info(f"📝 Enrollment activo: {'✅ SÍ' if status['enrollment_active'] else '❌ NO'}")
            logger.info(f"🔐 Autenticación activa: {'✅ SÍ' if status['authentication_active'] else '❌ NO'}")
        else:
            logger.error("❌ Error inicializando sistema biométrico")
            if manager.state.error_message:
                logger.error(f"🔍 Detalle del error: {manager.state.error_message}")
    
    except Exception as e:
        logger.error(f"❌ Error crítico en sistema biométrico: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info("=" * 80)
    logger.info("SISTEMA COMPLETAMENTE INICIALIZADO")
    logger.info("=" * 80)

# Evento de cierre
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Shutting down Biometric Gesture System")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )