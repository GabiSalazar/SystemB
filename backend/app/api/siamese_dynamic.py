"""
API endpoints para Siamese Dynamic Network
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.core.siamese_dynamic_network import (
    get_real_siamese_dynamic_network,
)

router = APIRouter(prefix="/siamese-dynamic", tags=["Siamese Dynamic Network"])


class ModelStatsResponse(BaseModel):
    """Respuesta con estadísticas del modelo"""
    is_trained: bool
    users_trained: int
    training_samples: int
    validation_samples: int
    optimal_threshold: float
    embedding_dim: int
    sequence_length: int
    feature_dim: int


@router.get("/health")
async def siamese_dynamic_health_check():
    """Verifica que Siamese Dynamic Network esté funcionando"""
    try:
        network = get_real_siamese_dynamic_network()
        
        return {
            "status": "healthy",
            "module": "Siamese Dynamic Network",
            "initialized": True,
            "message": "✅ Módulo 10 cargado correctamente",
            "tensorflow_available": True,
            "is_trained": network.is_trained,
            "embedding_dim": network.embedding_dim,
            "sequence_length": network.sequence_length,
            "feature_dim": network.feature_dim
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"TensorFlow no disponible: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Siamese Dynamic: {str(e)}")


@router.get("/stats", response_model=ModelStatsResponse)
async def get_model_stats():
    """Obtiene estadísticas del modelo siamés dinámico"""
    try:
        network = get_real_siamese_dynamic_network()
        
        return {
            "is_trained": network.is_trained,
            "users_trained": network.users_trained_count,
            "training_samples": len(network.real_training_samples),
            "validation_samples": len(network.real_validation_samples),
            "optimal_threshold": network.optimal_threshold,
            "embedding_dim": network.embedding_dim,
            "sequence_length": network.sequence_length,
            "feature_dim": network.feature_dim
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/summary")
async def get_model_summary():
    """Obtiene resumen completo del modelo"""
    try:
        network = get_real_siamese_dynamic_network()
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
        network = get_real_siamese_dynamic_network()
        
        return {
            "status": "success",
            "config": network.config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/architecture")
async def get_model_architecture():
    """Obtiene detalles de la arquitectura temporal"""
    try:
        network = get_real_siamese_dynamic_network()
        
        architecture = {
            "embedding_dim": network.embedding_dim,
            "sequence_length": network.sequence_length,
            "feature_dim": network.feature_dim,
            "lstm_units": network.config['lstm_units'],
            "sequence_processing": network.config['sequence_processing'],
            "temporal_pooling": network.config['temporal_pooling'],
            "dropout_rate": network.config['dropout_rate'],
            "recurrent_dropout": network.config['recurrent_dropout'],
            "dense_layers": network.config['dense_layers'],
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
    """Obtiene métricas de rendimiento del modelo temporal"""
    try:
        network = get_real_siamese_dynamic_network()
        
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
            "sequence_correlation": network.current_metrics.sequence_correlation,
            "temporal_consistency": network.current_metrics.temporal_consistency,
            "rhythm_similarity": network.current_metrics.rhythm_similarity,
            "validation_samples": network.current_metrics.validation_samples
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
    """Obtiene historial de entrenamiento temporal"""
    try:
        network = get_real_siamese_dynamic_network()
        
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


@router.get("/is-trained")
async def check_if_trained():
    """Verifica si el modelo temporal está entrenado"""
    try:
        network = get_real_siamese_dynamic_network()
        
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
    """Guarda el modelo temporal entrenado"""
    try:
        network = get_real_siamese_dynamic_network()
        
        if not network.is_trained:
            raise HTTPException(status_code=400, detail="Modelo no entrenado")
        
        success = network.save_real_model()
        
        if success:
            return {
                "status": "success",
                "message": "Modelo temporal guardado correctamente",
                "path": str(network.model_save_path)
            }
        else:
            raise HTTPException(status_code=500, detail="Error guardando modelo")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/load-model")
async def load_model():
    """Carga un modelo temporal previamente entrenado"""
    try:
        network = get_real_siamese_dynamic_network()
        
        success = network.load_real_model()
        
        if success:
            return {
                "status": "success",
                "message": "Modelo temporal cargado correctamente",
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
    """Obtiene requisitos para entrenamiento temporal"""
    try:
        network = get_real_siamese_dynamic_network()
        
        return {
            "status": "success",
            "requirements": {
                "min_users_for_training": network.config['min_users_for_training'],
                "min_samples_per_user": network.config['min_samples_per_user'],
                "sequence_length": network.sequence_length,
                "feature_dim": network.feature_dim,
                "embedding_dimension": network.embedding_dim,
                "training_epochs": network.config['epochs'],
                "batch_size": network.config['batch_size'],
                "min_sequence_length": network.config['min_sequence_length'],
                "max_sequence_length": network.config['max_sequence_length']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/temporal-config")
async def get_temporal_config():
    """Obtiene configuración específica temporal"""
    try:
        network = get_real_siamese_dynamic_network()
        
        return {
            "status": "success",
            "temporal_config": {
                "sequence_processing": network.config['sequence_processing'],
                "lstm_units": network.config['lstm_units'],
                "temporal_pooling": network.config['temporal_pooling'],
                "use_masking": network.config['use_masking'],
                "sequence_normalization": network.config['sequence_normalization'],
                "recurrent_dropout": network.config['recurrent_dropout'],
                "use_temporal_augmentation": network.config['use_temporal_augmentation']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")