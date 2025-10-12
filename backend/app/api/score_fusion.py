"""
API endpoints para Score Fusion System
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.core.score_fusion_system import (
    get_real_score_fusion_system,
    RealFusionStrategy,
    RealScoreCalibration,
    RealWeightOptimization
)

router = APIRouter(prefix="/score-fusion", tags=["Score Fusion System"])


class FusionStatsResponse(BaseModel):
    """Respuesta con estadísticas del sistema de fusión"""
    is_trained: bool
    is_calibrated: bool
    is_initialized: bool
    anatomical_weight: float
    dynamic_weight: float
    optimal_threshold: float
    available_models: int


@router.get("/health")
async def score_fusion_health_check():
    """Verifica que Score Fusion System esté funcionando"""
    try:
        fusion_system = get_real_score_fusion_system()
        
        return {
            "status": "healthy",
            "module": "Score Fusion System",
            "initialized": True,
            "message": "✅ Módulo 12 cargado correctamente",
            "sklearn_available": True,
            "is_trained": fusion_system.is_trained,
            "is_calibrated": fusion_system.is_calibrated,
            "is_initialized": fusion_system.is_initialized
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Scikit-learn no disponible: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Score Fusion: {str(e)}")


@router.get("/stats", response_model=FusionStatsResponse)
async def get_fusion_stats():
    """Obtiene estadísticas del sistema de fusión"""
    try:
        fusion_system = get_real_score_fusion_system()
        
        return {
            "is_trained": fusion_system.is_trained,
            "is_calibrated": fusion_system.is_calibrated,
            "is_initialized": fusion_system.is_initialized,
            "anatomical_weight": fusion_system.optimal_weights['anatomical'],
            "dynamic_weight": fusion_system.optimal_weights['dynamic'],
            "optimal_threshold": fusion_system.optimal_threshold,
            "available_models": len(fusion_system.real_fusion_models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/summary")
async def get_fusion_summary():
    """Obtiene resumen completo del sistema de fusión"""
    try:
        fusion_system = get_real_score_fusion_system()
        summary = fusion_system.get_real_fusion_summary()
        
        return {
            "status": "success",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/config")
async def get_fusion_config():
    """Obtiene la configuración del sistema de fusión"""
    try:
        fusion_system = get_real_score_fusion_system()
        
        config = {
            "fusion_strategy": fusion_system.config.fusion_strategy.value,
            "calibration_method": fusion_system.config.calibration_method.value,
            "weight_optimization": fusion_system.config.weight_optimization.value,
            "anatomical_weight": fusion_system.config.anatomical_weight,
            "dynamic_weight": fusion_system.config.dynamic_weight,
            "decision_threshold": fusion_system.config.decision_threshold,
            "use_confidence_weighting": fusion_system.config.use_confidence_weighting,
            "adaptive_threshold": fusion_system.config.adaptive_threshold,
            "optimization_metric": fusion_system.config.optimization_metric
        }
        
        return {
            "status": "success",
            "config": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/weights")
async def get_fusion_weights():
    """Obtiene los pesos optimizados de fusión"""
    try:
        fusion_system = get_real_score_fusion_system()
        
        return {
            "status": "success",
            "weights": {
                "anatomical": fusion_system.optimal_weights['anatomical'],
                "dynamic": fusion_system.optimal_weights['dynamic']
            },
            "optimal_threshold": fusion_system.optimal_threshold,
            "is_optimized": fusion_system.is_trained
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/metrics")
async def get_fusion_metrics():
    """Obtiene métricas de rendimiento del sistema de fusión"""
    try:
        fusion_system = get_real_score_fusion_system()
        
        if not fusion_system.is_trained:
            raise HTTPException(status_code=400, detail="Sistema no entrenado")
        
        if fusion_system.fusion_metrics is None:
            raise HTTPException(status_code=400, detail="No hay métricas disponibles")
        
        metrics = {
            "far": fusion_system.fusion_metrics.far,
            "frr": fusion_system.fusion_metrics.frr,
            "eer": fusion_system.fusion_metrics.eer,
            "auc_score": fusion_system.fusion_metrics.auc_score,
            "accuracy": fusion_system.fusion_metrics.accuracy,
            "precision": fusion_system.fusion_metrics.precision,
            "recall": fusion_system.fusion_metrics.recall,
            "f1_score": fusion_system.fusion_metrics.f1_score,
            "fusion_improvement": fusion_system.fusion_metrics.fusion_improvement,
            "calibration_quality": fusion_system.fusion_metrics.calibration_quality,
            "fusion_consistency": fusion_system.fusion_metrics.fusion_consistency,
            "decision_confidence_avg": fusion_system.fusion_metrics.decision_confidence_avg
        }
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/fusion-models")
async def get_fusion_models():
    """Obtiene información de los modelos de fusión entrenados"""
    try:
        fusion_system = get_real_score_fusion_system()
        
        models_info = {
            "available_models": list(fusion_system.real_fusion_models.keys()),
            "model_count": len(fusion_system.real_fusion_models),
            "is_trained": fusion_system.is_trained
        }
        
        return {
            "status": "success",
            "models": models_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/calibrators")
async def get_calibrators():
    """Obtiene información de los calibradores de scores"""
    try:
        fusion_system = get_real_score_fusion_system()
        
        calibrators_info = {
            "calibrated_modalities": list(fusion_system.real_score_calibrators.keys()),
            "calibrator_count": len(fusion_system.real_score_calibrators),
            "is_calibrated": fusion_system.is_calibrated
        }
        
        return {
            "status": "success",
            "calibrators": calibrators_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/fusion-strategies")
async def get_fusion_strategies():
    """Obtiene estrategias de fusión disponibles"""
    return {
        "status": "success",
        "strategies": [
            {
                "name": strategy.name, 
                "value": strategy.value,
                "description": _get_strategy_description(strategy.value)
            }
            for strategy in RealFusionStrategy
        ]
    }


@router.get("/calibration-methods")
async def get_calibration_methods():
    """Obtiene métodos de calibración disponibles"""
    return {
        "status": "success",
        "methods": [
            {
                "name": method.name, 
                "value": method.value,
                "description": _get_calibration_description(method.value)
            }
            for method in RealScoreCalibration
        ]
    }


@router.get("/weight-optimization-methods")
async def get_weight_optimization_methods():
    """Obtiene métodos de optimización de pesos disponibles"""
    return {
        "status": "success",
        "methods": [
            {
                "name": method.name, 
                "value": method.value,
                "description": _get_optimization_description(method.value)
            }
            for method in RealWeightOptimization
        ]
    }


@router.get("/voting-config")
async def get_voting_config():
    """Obtiene configuración del mecanismo de voting"""
    return {
        "status": "success",
        "voting_config": {
            "vote_threshold": 0.85,
            "min_vote_ratio": 0.5,
            "description": "Mecanismo de voting para reducir falsos positivos"
        }
    }


def _get_strategy_description(value: str) -> str:
    """Obtiene descripción de estrategia de fusión"""
    descriptions = {
        "weighted_average": "Promedio ponderado de scores anatómico y dinámico",
        "product_rule": "Regla del producto para combinar probabilidades",
        "max_rule": "Selecciona el score máximo entre modalidades",
        "min_rule": "Selecciona el score mínimo entre modalidades",
        "svm_fusion": "Fusión usando SVM entrenado con datos reales",
        "neural_fusion": "Fusión usando red neuronal entrenada",
        "logistic_fusion": "Fusión usando regresión logística",
        "adaptive_fusion": "Fusión adaptativa basada en confianza",
        "ensemble_fusion": "Ensemble de múltiples estrategias"
    }
    return descriptions.get(value, "Sin descripción")


def _get_calibration_description(value: str) -> str:
    """Obtiene descripción de método de calibración"""
    descriptions = {
        "none": "Sin calibración de scores",
        "min_max": "Normalización Min-Max a rango [0,1]",
        "z_score": "Normalización Z-score (media=0, std=1)",
        "sigmoid": "Calibración mediante función sigmoide",
        "isotonic": "Regresión isotónica no paramétrica"
    }
    return descriptions.get(value, "Sin descripción")


def _get_optimization_description(value: str) -> str:
    """Obtiene descripción de método de optimización"""
    descriptions = {
        "fixed": "Pesos fijos predefinidos",
        "grid_search": "Búsqueda exhaustiva en grilla",
        "gradient_descent": "Optimización por gradiente descendente",
        "genetic_algorithm": "Algoritmo genético evolutivo",
        "confidence_based": "Basado en confianza de modalidades"
    }
    return descriptions.get(value, "Sin descripción")