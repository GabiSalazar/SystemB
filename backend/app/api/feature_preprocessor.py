"""
API endpoints para Feature Preprocessor
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.core.feature_preprocessor import (
    get_real_feature_preprocessor,
    NormalizationMethod,
    BalancingMethod,
    AugmentationStrategy
)

router = APIRouter(prefix="/feature-preprocessor", tags=["Feature Preprocessor"])


class PreprocessorStatsResponse(BaseModel):
    """Respuesta con estadísticas del preprocesador"""
    is_fitted: bool
    total_users: int
    total_samples: int
    anatomical_dim: int
    dynamic_sequence_shape: tuple


@router.get("/health")
async def feature_preprocessor_health_check():
    """Verifica que Feature Preprocessor esté funcionando"""
    try:
        preprocessor = get_real_feature_preprocessor()
        
        return {
            "status": "healthy",
            "module": "Feature Preprocessor",
            "initialized": True,
            "message": "✅ Módulo 11 cargado correctamente",
            "sklearn_available": True,
            "is_fitted": preprocessor.is_fitted
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Scikit-learn no disponible: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Feature Preprocessor: {str(e)}")


@router.get("/stats", response_model=PreprocessorStatsResponse)
async def get_preprocessor_stats():
    """Obtiene estadísticas del preprocesador"""
    try:
        preprocessor = get_real_feature_preprocessor()
        
        if not preprocessor.is_fitted:
            return {
                "is_fitted": False,
                "total_users": 0,
                "total_samples": 0,
                "anatomical_dim": 0,
                "dynamic_sequence_shape": (0,)
            }
        
        dataset = preprocessor.processed_dataset
        
        return {
            "is_fitted": True,
            "total_users": len(preprocessor.user_encoders),
            "total_samples": len(dataset.anatomical_features),
            "anatomical_dim": int(dataset.anatomical_features.shape[1]),
            "dynamic_sequence_shape": tuple(int(x) for x in dataset.dynamic_sequences.shape[1:]) if len(dataset.dynamic_sequences) > 0 else (0,)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/summary")
async def get_preprocessing_summary():
    """Obtiene resumen completo del preprocesamiento"""
    try:
        preprocessor = get_real_feature_preprocessor()
        summary = preprocessor.get_real_preprocessing_summary()
        
        return {
            "status": "success",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/config")
async def get_preprocessing_config():
    """Obtiene la configuración del preprocesador"""
    try:
        preprocessor = get_real_feature_preprocessor()
        
        config = {
            "anatomical_normalization": preprocessor.config.anatomical_normalization.value,
            "dynamic_normalization": preprocessor.config.dynamic_normalization.value,
            "normalize_per_user": preprocessor.config.normalize_per_user,
            "balancing_method": preprocessor.config.balancing_method.value,
            "target_balance_ratio": preprocessor.config.target_balance_ratio,
            "augmentation_strategy": preprocessor.config.augmentation_strategy.value,
            "augmentation_factor": preprocessor.config.augmentation_factor,
            "outlier_threshold": preprocessor.config.outlier_threshold,
            "min_samples_per_user": preprocessor.config.min_samples_per_user,
            "test_size": preprocessor.config.test_size,
            "validation_size": preprocessor.config.validation_size
        }
        
        return {
            "status": "success",
            "config": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/pipelines")
async def get_pipeline_info():
    """Obtiene información de los pipelines"""
    try:
        preprocessor = get_real_feature_preprocessor()
        
        if not preprocessor.is_fitted:
            raise HTTPException(status_code=400, detail="Preprocesador no ajustado")
        
        anatomical_steps = [step[0] for step in preprocessor.anatomical_pipeline.steps]
        dynamic_steps = [step[0] for step in preprocessor.dynamic_pipeline.steps]
        
        return {
            "status": "success",
            "anatomical_pipeline": {
                "steps": anatomical_steps,
                "step_count": len(anatomical_steps)
            },
            "dynamic_pipeline": {
                "steps": dynamic_steps,
                "step_count": len(dynamic_steps)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/quality-metrics")
async def get_quality_metrics():
    """Obtiene métricas de calidad de los datos"""
    try:
        preprocessor = get_real_feature_preprocessor()
        
        if not preprocessor.is_fitted or not preprocessor.processed_dataset:
            raise HTTPException(status_code=400, detail="Preprocesador no ajustado")
        
        metrics = preprocessor.processed_dataset.quality_metrics
        
        return {
            "status": "success",
            "quality_metrics": {
                "total_samples": metrics.total_samples,
                "total_users": metrics.total_users,
                "data_quality_score": metrics.data_quality_score,
                "outlier_percentage": metrics.outlier_percentage,
                "missing_data_percentage": metrics.missing_data_percentage,
                "samples_per_user": metrics.samples_per_user,
                "gesture_distribution": metrics.gesture_distribution,
                "recommendations": metrics.recommendations
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/splits")
async def get_data_splits():
    """Obtiene información de los splits de datos"""
    try:
        preprocessor = get_real_feature_preprocessor()
        
        if not preprocessor.is_fitted or not preprocessor.processed_dataset:
            raise HTTPException(status_code=400, detail="Preprocesador no ajustado")
        
        splits = preprocessor.processed_dataset.splits
        
        return {
            "status": "success",
            "splits": {
                "train": {
                    "users": splits['users']['train'],
                    "user_count": len(splits['users']['train']),
                    "sample_count": len(splits['samples']['train'])
                },
                "validation": {
                    "users": splits['users']['validation'],
                    "user_count": len(splits['users']['validation']),
                    "sample_count": len(splits['samples']['validation'])
                },
                "test": {
                    "users": splits['users']['test'],
                    "user_count": len(splits['users']['test']),
                    "sample_count": len(splits['samples']['test'])
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/normalization-methods")
async def get_normalization_methods():
    """Obtiene métodos de normalización disponibles"""
    return {
        "status": "success",
        "methods": [
            {"name": method.name, "value": method.value, "description": _get_normalization_description(method.value)}
            for method in NormalizationMethod
        ]
    }


@router.get("/balancing-methods")
async def get_balancing_methods():
    """Obtiene métodos de balanceo disponibles"""
    return {
        "status": "success",
        "methods": [
            {"name": method.name, "value": method.value, "description": _get_balancing_description(method.value)}
            for method in BalancingMethod
        ]
    }


@router.get("/augmentation-strategies")
async def get_augmentation_strategies():
    """Obtiene estrategias de augmentación disponibles"""
    return {
        "status": "success",
        "strategies": [
            {"name": strategy.name, "value": strategy.value}
            for strategy in AugmentationStrategy
        ]
    }


def _get_normalization_description(value: str) -> str:
    """Obtiene descripción de método de normalización"""
    descriptions = {
        "standard": "StandardScaler - normalización con media y desviación estándar",
        "robust": "RobustScaler - robusto a outliers usando mediana y MAD",
        "minmax": "MinMaxScaler - escala a rango [0, 1]",
        "quantile": "QuantileTransformer - transformación basada en cuantiles",
        "none": "Sin normalización"
    }
    return descriptions.get(value, "Sin descripción")


def _get_balancing_description(value: str) -> str:
    """Obtiene descripción de método de balanceo"""
    descriptions = {
        "none": "Sin balanceo de clases",
        "undersample": "Submuestreo de clase mayoritaria",
        "balanced_subsample": "Submuestreo balanceado con ratio objetivo",
        "weighted": "Pesos por clase sin modificar datos",
        "stratified_split": "Split estratificado por clase"
    }
    return descriptions.get(value, "Sin descripción")