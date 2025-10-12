"""
API endpoints para Anatomical Features Extractor
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from app.core.anatomical_features_extractor import (
    get_anatomical_features_extractor,
    FeatureCategory
)

router = APIRouter(prefix="/anatomical-features", tags=["Anatomical Features"])


class FeatureStatsResponse(BaseModel):
    """Respuesta con estadísticas de extracción"""
    extractions_performed: int
    successful_extractions: int
    success_rate_percent: float
    feature_dimension: int
    feature_categories: int
    use_world_landmarks: bool
    normalize_features: bool


@router.get("/health")
async def anatomical_features_health_check():
    """Verifica que Anatomical Features Extractor esté funcionando"""
    try:
        extractor = get_anatomical_features_extractor()
        
        return {
            "status": "healthy",
            "module": "Anatomical Features Extractor",
            "initialized": True,
            "message": "✅ Módulo 6 cargado correctamente",
            "feature_dimension": 180,
            "feature_categories": len(FeatureCategory)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Anatomical Features Extractor: {str(e)}")


@router.get("/stats", response_model=FeatureStatsResponse)
async def get_extraction_stats():
    """Obtiene estadísticas de extracción de características"""
    try:
        extractor = get_anatomical_features_extractor()
        stats = extractor.get_extraction_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reset-stats")
async def reset_extraction_stats():
    """Reinicia estadísticas de extracción"""
    try:
        extractor = get_anatomical_features_extractor()
        extractor.reset_stats()
        
        return {
            "status": "success",
            "message": "Estadísticas de extracción reiniciadas correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/categories")
async def get_feature_categories():
    """Obtiene las categorías de características disponibles"""
    try:
        categories = [
            {
                "name": cat.name,
                "value": cat.value,
                "description": _get_category_description(cat.value)
            }
            for cat in FeatureCategory
        ]
        
        return {
            "status": "success",
            "count": len(categories),
            "categories": categories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/dimensions")
async def get_feature_dimensions():
    """Obtiene las dimensiones de cada categoría de características"""
    return {
        "status": "success",
        "total_dimension": 180,
        "categories": {
            "finger_features": {
                "dimension": 50,
                "description": "Características de dedos (5 dedos × 10 características)"
            },
            "palm_features": {
                "dimension": 20,
                "description": "Características de la palma"
            },
            "proportion_features": {
                "dimension": 30,
                "description": "Proporciones generales de la mano"
            },
            "angle_features": {
                "dimension": 25,
                "description": "Ángulos articulares"
            },
            "distance_features": {
                "dimension": 35,
                "description": "Distancias normalizadas entre landmarks"
            },
            "curvature_features": {
                "dimension": 20,
                "description": "Características de curvatura"
            }
        }
    }


@router.get("/landmark-structure")
async def get_landmark_structure():
    """Obtiene la estructura de landmarks MediaPipe"""
    try:
        extractor = get_anatomical_features_extractor()
        
        structure_info = {
            "total_landmarks": 21,
            "structure": {
                "wrist": {"indices": [0], "description": "Muñeca"},
                "thumb": {"indices": [1, 2, 3, 4], "description": "Pulgar (4 puntos)"},
                "index": {"indices": [5, 6, 7, 8], "description": "Índice (4 puntos)"},
                "middle": {"indices": [9, 10, 11, 12], "description": "Medio (4 puntos)"},
                "ring": {"indices": [13, 14, 15, 16], "description": "Anular (4 puntos)"},
                "pinky": {"indices": [17, 18, 19, 20], "description": "Meñique (4 puntos)"}
            }
        }
        
        return {
            "status": "success",
            "landmark_structure": structure_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/config")
async def get_feature_config():
    """Obtiene la configuración del extractor de características"""
    try:
        extractor = get_anatomical_features_extractor()
        
        return {
            "status": "success",
            "config": extractor.feature_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/finger-names")
async def get_finger_names():
    """Obtiene los nombres de los dedos procesados"""
    return {
        "status": "success",
        "fingers": [
            {"name": "thumb", "spanish": "Pulgar"},
            {"name": "index", "spanish": "Índice"},
            {"name": "middle", "spanish": "Medio"},
            {"name": "ring", "spanish": "Anular"},
            {"name": "pinky", "spanish": "Meñique"}
        ]
    }


@router.get("/palm-regions")
async def get_palm_regions():
    """Obtiene las regiones de la palma analizadas"""
    return {
        "status": "success",
        "regions": {
            "boundary": {
                "landmarks": [0, 1, 5, 9, 13, 17],
                "description": "Contorno de la palma"
            },
            "center_region": {
                "landmarks": [0, 2, 5, 9, 13, 17],
                "description": "Región central de la palma"
            }
        }
    }


@router.get("/quality-thresholds")
async def get_quality_thresholds():
    """Obtiene los umbrales de calidad para validación de características"""
    try:
        extractor = get_anatomical_features_extractor()
        
        return {
            "status": "success",
            "thresholds": {
                "outlier_threshold": extractor.feature_config.get('outlier_threshold', 3.0),
                "min_hand_size": extractor.feature_config.get('min_hand_size', 0.05),
                "max_outlier_ratio": 0.1,
                "min_std_deviation": 1e-6
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def _get_category_description(category_value: str) -> str:
    """Obtiene descripción de una categoría"""
    descriptions = {
        "finger_lengths": "Longitudes y proporciones de dedos",
        "palm_dimensions": "Dimensiones y forma de la palma",
        "joint_angles": "Ángulos de articulaciones de dedos",
        "finger_spreads": "Separación entre dedos adyacentes",
        "palm_curvature": "Curvatura de la palma y dedos",
        "hand_proportions": "Proporciones generales de la mano",
        "landmark_distances": "Distancias entre landmarks clave",
        "geometric_ratios": "Ratios geométricos de la mano"
    }
    return descriptions.get(category_value, "Sin descripción")