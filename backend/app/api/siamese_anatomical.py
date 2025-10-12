"""
API endpoints para Siamese Anatomical Network
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.core.siamese_anatomical_network import (
    get_real_siamese_anatomical_network,
    DistanceMetric,
    LossFunction,
    TrainingMode
)

router = APIRouter(prefix="/siamese-anatomical", tags=["Siamese Anatomical Network"])


class ModelStatsResponse(BaseModel):
    """Respuesta con estadísticas del modelo"""
    is_trained: bool
    users_trained: int
    genuine_pairs: int
    impostor_pairs: int
    optimal_threshold: float
    embedding_dim: int
    input_dim: int


class TrainModelRequest(BaseModel):
    """Request para entrenar modelo"""
    validation_split: float = 0.2


@router.get("/health")
async def siamese_anatomical_health_check():
    """Verifica que Siamese Anatomical Network esté funcionando"""
    try:
        network = get_real_siamese_anatomical_network()
        
        return {
            "status": "healthy",
            "module": "Siamese Anatomical Network",
            "initialized": True,
            "message": "✅ Módulo 9 cargado correctamente",
            "tensorflow_available": True,
            "is_trained": network.is_trained,
            "embedding_dim": network.embedding_dim,
            "input_dim": network.input_dim
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"TensorFlow no disponible: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Siamese Anatomical: {str(e)}")


@router.get("/stats", response_model=ModelStatsResponse)
async def get_model_stats():
    """Obtiene estadísticas del modelo siamés"""
    try:
        network = get_real_siamese_anatomical_network()
        
        return {
            "is_trained": network.is_trained,
            "users_trained": network.users_trained_count,
            "genuine_pairs": network.total_genuine_pairs,
            "impostor_pairs": network.total_impostor_pairs,
            "optimal_threshold": network.optimal_threshold,
            "embedding_dim": network.embedding_dim,
            "input_dim": network.input_dim
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/summary")
async def get_model_summary():
    """Obtiene resumen completo del modelo"""
    try:
        network = get_real_siamese_anatomical_network()
        summary = network.get_real_model_summary()
        
        return {
            "status": "success",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/config")
async def get_model_config():
    """Obtiene la configuración del modelo"""
    try:
        network = get_real_siamese_anatomical_network()
        
        return {
            "status": "success",
            "config": network.config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/architecture")
async def get_model_architecture():
    """Obtiene detalles de la arquitectura"""
    try:
        network = get_real_siamese_anatomical_network()
        
        architecture = {
            "embedding_dim": network.embedding_dim,
            "input_dim": network.input_dim,
            "hidden_layers": network.config['hidden_layers'],
            "activation": network.config['activation'],
            "dropout_rate": network.config['dropout_rate'],
            "batch_normalization": network.config['batch_normalization'],
            "l2_regularization": network.config['l2_regularization'],
            "distance_metric": network.config['distance_metric'],
            "loss_function": network.config['loss_function'],
            "total_parameters": network.siamese_model.count_params() if network.siamese_model else 0
        }
        
        return {
            "status": "success",
            "architecture": architecture
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/metrics")
async def get_model_metrics():
    """Obtiene métricas de rendimiento del modelo"""
    try:
        network = get_real_siamese_anatomical_network()
        
        if not network.is_trained:
            raise HTTPException(status_code=400, detail="Modelo no entrenado")
        
        if network.current_metrics is None:
            raise HTTPException(status_code=400, detail="No hay métricas disponibles")
        
        metrics = {
            "far": network.current_metrics.far,
            "frr": network.current_metrics.frr,
            "eer": network.current_metrics.eer,
            "auc_score": network.current_metrics.auc_score,
            "accuracy": network.current_metrics.accuracy,
            "threshold": network.current_metrics.threshold,
            "precision": network.current_metrics.precision,
            "recall": network.current_metrics.recall,
            "f1_score": network.current_metrics.f1_score,
            "total_genuine_pairs": network.current_metrics.total_genuine_pairs,
            "total_impostor_pairs": network.current_metrics.total_impostor_pairs,
            "users_in_test": network.current_metrics.users_in_test
        }
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/training-history")
async def get_training_history():
    """Obtiene historial de entrenamiento"""
    try:
        network = get_real_siamese_anatomical_network()
        
        if not network.is_trained:
            raise HTTPException(status_code=400, detail="Modelo no entrenado")
        
        history = {
            "loss": network.training_history.loss,
            "val_loss": network.training_history.val_loss,
            "far_history": network.training_history.far_history,
            "frr_history": network.training_history.frr_history,
            "eer_history": network.training_history.eer_history,
            "best_epoch": network.training_history.best_epoch,
            "total_training_time": network.training_history.total_training_time,
            "epochs_trained": len(network.training_history.loss)
        }
        
        return {
            "status": "success",
            "history": history
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/distance-metrics")
async def get_distance_metrics():
    """Obtiene métricas de distancia disponibles"""
    return {
        "status": "success",
        "metrics": [
            {"name": metric.name, "value": metric.value}
            for metric in DistanceMetric
        ]
    }


@router.get("/loss-functions")
async def get_loss_functions():
    """Obtiene funciones de pérdida disponibles"""
    return {
        "status": "success",
        "functions": [
            {"name": func.name, "value": func.value}
            for func in LossFunction
        ]
    }


@router.get("/training-modes")
async def get_training_modes():
    """Obtiene modos de entrenamiento disponibles"""
    return {
        "status": "success",
        "modes": [
            {"name": mode.name, "value": mode.value}
            for mode in TrainingMode
        ]
    }


@router.get("/is-trained")
async def check_if_trained():
    """Verifica si el modelo está entrenado"""
    try:
        network = get_real_siamese_anatomical_network()
        
        return {
            "status": "success",
            "is_trained": network.is_trained,
            "is_compiled": network.is_compiled,
            "ready_for_inference": network.is_trained and network.is_compiled,
            "users_trained": network.users_trained_count if network.is_trained else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/save-model")
async def save_model():
    """Guarda el modelo entrenado"""
    try:
        network = get_real_siamese_anatomical_network()
        
        if not network.is_trained:
            raise HTTPException(status_code=400, detail="Modelo no entrenado")
        
        success = network.save_real_model()
        
        if success:
            return {
                "status": "success",
                "message": "Modelo guardado correctamente",
                "path": network.model_save_path
            }
        else:
            raise HTTPException(status_code=500, detail="Error guardando modelo")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/load-model")
async def load_model():
    """Carga un modelo previamente entrenado"""
    try:
        network = get_real_siamese_anatomical_network()
        
        success = network.load_real_model()
        
        if success:
            return {
                "status": "success",
                "message": "Modelo cargado correctamente",
                "is_trained": network.is_trained,
                "users_trained": network.users_trained_count
            }
        else:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/requirements")
async def get_training_requirements():
    """Obtiene requisitos para entrenamiento"""
    try:
        network = get_real_siamese_anatomical_network()
        
        return {
            "status": "success",
            "requirements": {
                "min_users_for_training": network.config['min_users_for_training'],
                "min_samples_per_user": network.config['min_samples_per_user'],
                "min_sessions_per_user": network.config['min_sessions_per_user'],
                "input_dimension": network.input_dim,
                "embedding_dimension": network.embedding_dim,
                "training_epochs": network.config['epochs'],
                "batch_size": network.config['batch_size']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")