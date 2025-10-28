# =============================================================================
# MÓDULO 9: SIAMESE_ANATOMICAL_NETWORK
# Red Siamesa para características anatómicas
# =============================================================================
import os
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, optimizers, callbacks
    from tensorflow.keras.metrics import binary_accuracy
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow no disponible - red siamesa anatómica limitada")

# Scikit-learn imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn no disponible - métricas limitadas")

# Importar módulos anteriores
try:
    from app.core.config_manager import get_config, get_logger, log_error, log_info
    from app.core.anatomical_features_extractor import AnatomicalFeatureVector, get_anatomical_features_extractor
except ImportError:
    def get_config(key, default=None): 
        return default
    def get_logger(): 
        return print
    def log_error(msg, exc=None): 
        logging.error(f"ERROR: {msg}")
    def log_info(msg): 
        logging.info(f"INFO: {msg}")

# Logger
logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Métricas de distancia para redes siamesas."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    MINKOWSKI = "minkowski"


class LossFunction(Enum):
    """Funciones de pérdida para entrenamiento."""
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    BINARY_CROSSENTROPY = "binary_crossentropy"


class TrainingMode(Enum):
    """Modos de entrenamiento."""
    GENUINE_IMPOSTOR = "genuine_impostor"
    TRIPLET_LOSS = "triplet_loss"
    CLASSIFICATION = "classification"


@dataclass
class RealBiometricSample:
    """Muestra biométrica con características anatómicas."""
    user_id: str
    sample_id: str
    features: np.ndarray
    gesture_name: str
    confidence: float
    timestamp: float
    hand_side: str = "unknown"
    quality_score: float = 1.0
    session_id: str = "default"
    capture_conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealTrainingPair:
    """Par de entrenamiento para red siamesa."""
    sample1: RealBiometricSample
    sample2: RealBiometricSample
    is_genuine: bool
    distance: Optional[float] = None


@dataclass
class RealModelMetrics:
    """Métricas de evaluación del modelo."""
    far: float
    frr: float
    eer: float
    auc_score: float
    accuracy: float
    threshold: float
    precision: float
    recall: float
    f1_score: float
    
    total_genuine_pairs: int
    total_impostor_pairs: int
    users_in_test: int
    cross_validation_score: float


@dataclass
class RealTrainingHistory:
    """Historial de entrenamiento."""
    loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    far_history: List[float] = field(default_factory=list)
    frr_history: List[float] = field(default_factory=list)
    eer_history: List[float] = field(default_factory=list)
    best_epoch: int = 0
    total_training_time: float = 0.0

class RealSiameseAnatomicalNetwork:
    """
    Red Siamesa para autenticación biométrica basada en características anatómicas.
    Implementa arquitectura twin network para comparar características únicas de manos.
    """
    
    def __init__(self, embedding_dim: int = 64, input_dim: int = 180):
        """Inicializa la red siamesa anatómica REAL."""
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow no disponible - no se puede usar red siamesa")
        
        self.logger = get_logger()
        
        # Configuración del modelo
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.config = self._load_real_siamese_config()
        
        # Arquitectura del modelo
        self.base_network = None
        self.siamese_model = None
        self.is_compiled = False
        
        # Estado de entrenamiento
        self.training_history = RealTrainingHistory()
        self.is_trained = False
        self.optimal_threshold = 0.5
        
        # Dataset y métricas
        self.real_training_samples: List[RealBiometricSample] = []
        self.real_validation_samples: List[RealBiometricSample] = []
        self.current_metrics: Optional[RealModelMetrics] = None
        
        # Rutas de guardado
        self.model_save_path = self._get_real_model_save_path()
        
        # Estadísticas de entrenamiento
        self.users_trained_count = 0
        self.total_genuine_pairs = 0
        self.total_impostor_pairs = 0
        
        logger.info("RealSiameseAnatomicalNetwork inicializada")
    
    def _load_real_siamese_config(self) -> Dict[str, Any]:
        """Carga configuración de la red siamesa anatómica."""
        default_config = {
            # Arquitectura de red
            'hidden_layers': [128, 64],
            'activation': 'relu',
            'dropout_rate': 0.2,
            'batch_normalization': True,
            'l2_regularization': 0.001,
            
            # Entrenamiento
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 15,
            'validation_split': 0.2,
            
            # Requisitos para datos 
            'min_users_for_training': 2,
            'min_samples_per_user': 15,
            'max_samples_per_user': 50,
            'min_sessions_per_user': 1,
            
            # Función de pérdida y optimización
            'loss_function': 'contrastive',
            'distance_metric': 'euclidean',
            'margin': 1.5,
            'alpha': 0.2,
            
            # Validación
            'use_stratified_split': True,
            'cross_validation_folds': 5,
            'threshold_optimization': 'eer',
            'quality_threshold': 80.0,
            
            # Augmentación
            'use_real_augmentation': True,
            'temporal_jitter': 0.02,
            'noise_from_real_variance': True,
            
            # Evaluación
            'require_independent_test': True,
            'min_test_users': 1,
            'performance_monitoring': True,
        }
        
        return get_config('biometric.siamese_anatomical', default_config)
    
    def _get_real_model_save_path(self) -> str:
        """Obtiene ruta REAL para guardar modelo entrenado."""
        models_dir = get_config('paths.models', 'biometric_data/models')
        return str(Path(models_dir) / 'anatomical_model.h5')
    
    def load_real_training_data_from_database(self, database) -> bool:
        """
        Carga datos anatómicos desde la base de datos biométrica.
        Procesa templates anatómicos y extrae características de 180D.
        """
        try:
            logger.info("=== CARGANDO DATOS ANATÓMICOS DESDE BASE DE DATOS ===")
            
            # Obtener todos los usuarios
            real_users = database.list_users()
            
            if len(real_users) < self.config.get('min_users_for_training', 2):
                logger.error(f"Insuficientes usuarios: {len(real_users)} < 2")
                return False
            
            logger.info(f"📊 Usuarios encontrados: {len(real_users)}")
            
            # Limpiar muestras existentes
            self.real_training_samples.clear()
            
            users_with_sufficient_data = 0
            total_samples_loaded = 0
            
            for user in real_users:
                try:
                    logger.info(f"📂 Procesando usuario: {user.username} ({user.user_id})")
                    
                    # Obtener todos los templates del usuario
                    user_templates_list = []
                    for template_id, template in database.templates.items():
                        if template.user_id == user.user_id:
                            user_templates_list.append(template)
                    
                    if not user_templates_list:
                        logger.info(f"   ⚠️ Usuario {user.user_id} sin templates")
                        continue
                    
                    logger.info(f"   📊 Templates encontrados: {len(user_templates_list)}")
                    
                    # Filtrar templates anatómicos
                    anatomical_templates = []
                    dynamic_templates = []
                    for template in user_templates_list:
                        template_type_str = str(template.template_type)
                        template_id = template.template_id
                        
                        if ('anatomical' in template_type_str.lower() and 
                            '_bootstrap_dynamic_' not in template_id):
                            anatomical_templates.append(template)
                        elif 'dynamic' in template_type_str.lower():
                            dynamic_templates.append(template)
                            
                    logger.info(f"   📊 Templates anatómicos: {len(anatomical_templates)}")
                    logger.info(f"   📊 Templates dinámicos: {len(dynamic_templates)} (omitidos - red anatómica)")

                    
                    # Procesar templates anatómicos
                    user_anatomical_samples = []
                    
                    for template in anatomical_templates:
                        try:
                            bootstrap_features = template.metadata.get('bootstrap_features', None)
                            
                            if bootstrap_features is not None:
                                features_to_process = []
                                
                                if isinstance(bootstrap_features, list) and len(bootstrap_features) > 0:
                                    if isinstance(bootstrap_features[0], list):
                                        features_to_process = bootstrap_features
                                    elif isinstance(bootstrap_features[0], (int, float)):
                                        if len(bootstrap_features) == 180:
                                            features_to_process = [bootstrap_features]
                                
                                for idx, anatomical_features in enumerate(features_to_process):
                                    if len(anatomical_features) == 180:
                                        anatomical_sample = RealBiometricSample(
                                            user_id=user.user_id,
                                            sample_id=f"{template.template_id}_{idx}",
                                            features=np.array(anatomical_features, dtype=np.float32),
                                            gesture_name=template.gesture_name,
                                            confidence=template.confidence,
                                            timestamp=getattr(template, 'created_at', time.time()),
                                            quality_score=template.quality_score,
                                            metadata={
                                                'data_source': template.metadata.get('data_source', 'enrollment_capture'),
                                                'bootstrap_mode': template.metadata.get('bootstrap_mode', True),
                                                'feature_dimension': len(anatomical_features),
                                                'template_id': template.template_id,
                                                'sample_index': idx
                                            }
                                        )
                                        
                                        user_anatomical_samples.append(anatomical_sample)
                        
                        except Exception as e:
                            logger.error(f"   ❌ Error procesando template {template.template_id}: {e}")
                            continue
                    
                    # Validar usuario con datos suficientes
                    min_anatomical_samples = max(3, self.config.get('min_samples_per_user', 15) // 5)
                    
                    if len(user_anatomical_samples) >= min_anatomical_samples:
                        users_with_sufficient_data += 1
                        total_samples_loaded += len(user_anatomical_samples)
                        self.real_training_samples.extend(user_anatomical_samples)
                        
                        # Calcular estadísticas por gesto
                        gesture_counts = {}
                        for sample in user_anatomical_samples:
                            gesture_name = sample.gesture_name
                            if gesture_name not in gesture_counts:
                                gesture_counts[gesture_name] = 0
                            gesture_counts[gesture_name] += 1
                            
                        logger.info(f"✅ Usuario anatómico válido: {user.username}")
                        logger.info(f"   📊 Muestras anatómicas: {len(user_anatomical_samples)}")
                        logger.info(f"   🎯 Gestos únicos: {len(gesture_counts)}")
                        for gesture, count in gesture_counts.items():
                            logger.info(f"      • {gesture}: {count} muestras anatómicas")
                    else:
                        logger.warning(f"   ⚠️ Usuario {user.user_id} con pocas muestras anatómicas: {len(user_anatomical_samples)} < {min_anatomical_samples}")
                    
                except Exception as e:
                    logger.error(f"Error procesando usuario {user.user_id}: {e}")
                    continue
            
            # Validación final
            min_users_required = self.config.get('min_users_for_training', 2)
            min_total_samples = 6
            
            if users_with_sufficient_data < min_users_required:
                logger.error("❌ USUARIOS INSUFICIENTES PARA ENTRENAMIENTO")
                return False
            
            if total_samples_loaded < min_total_samples:
                logger.error("❌ MUESTRAS ANATÓMICAS INSUFICIENTES")
                return False
            
            logger.info(f"Total muestras cargadas: {len(self.real_training_samples)} (sin dividir)")

            # Actualizar contador de usuarios
            self.users_trained_count = users_with_sufficient_data
            
            logger.info("=" * 60)
            logger.info("✅ DATOS ANATÓMICOS REALES CARGADOS")
            logger.info("=" * 60)
            logger.info(f"👥 Usuarios: {users_with_sufficient_data}")
            logger.info(f"🧬 Total muestras: {total_samples_loaded}")
            logger.info(f"📊 Promedio por usuario: {total_samples_loaded/users_with_sufficient_data:.1f}")
            logger.info("=" * 60)
            
            # Estadísticas detalladas por gesto
            gesture_stats = {}
            all_samples = self.real_training_samples + self.real_validation_samples
            for sample in all_samples:
                gesture_name = sample.gesture_name
                if gesture_name not in gesture_stats:
                    gesture_stats[gesture_name] = 0
                gesture_stats[gesture_name] += 1
            
            logger.info(f"📈 DISTRIBUCIÓN POR GESTO:")
            for gesture, count in gesture_stats.items():
                logger.info(f"   • {gesture}: {count} muestras anatómicas")
            
            # Estadísticas por usuario
            user_stats = {}
            for sample in all_samples:
                if sample.user_id not in user_stats:
                    user_stats[sample.user_id] = 0
                user_stats[sample.user_id] += 1
            
            logger.info(f"📈 DISTRIBUCIÓN POR USUARIO:")
            for user_id, count in user_stats.items():
                user_name = next((u.username for u in real_users if u.user_id == user_id), user_id)
                logger.info(f"   • {user_name} ({user_id}): {count} muestras")

            # ✅ CORRECCIÓN: Actualizar contador de usuarios entrenados
            self.users_trained_count = len(user_stats)
            logger.info(f"📊 Usuarios con datos suficientes registrados: {self.users_trained_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ERROR CARGANDO DATOS: {e}")
            return False
        
    def validate_real_data_quality(self) -> bool:
        """Valida calidad de los datos cargados."""
        try:
            if not self.real_training_samples:
                logger.error("No hay datos para validar")
                return False
            
            # Agrupar por usuario
            users_data = {}
            for sample in self.real_training_samples:
                if sample.user_id not in users_data:
                    users_data[sample.user_id] = []
                users_data[sample.user_id].append(sample)
            
            quality_issues = []
            
            # 1. Verificar variabilidad inter-usuario
            all_features = np.array([sample.features for sample in self.real_training_samples])
            user_means = {}
            for user_id, samples in users_data.items():
                user_features = np.array([s.features for s in samples])
                user_means[user_id] = np.mean(user_features, axis=0)
            
            user_ids = list(user_means.keys())
            min_inter_user_distance = float('inf')
            
            for i in range(len(user_ids)):
                for j in range(i + 1, len(user_ids)):
                    distance = np.linalg.norm(user_means[user_ids[i]] - user_means[user_ids[j]])
                    min_inter_user_distance = min(min_inter_user_distance, distance)
            
            if min_inter_user_distance < 0.1:
                quality_issues.append(f"Usuarios muy similares (distancia: {min_inter_user_distance:.4f})")
            
            # 2. Verificar variabilidad intra-usuario
            for user_id, samples in users_data.items():
                if len(samples) > 1:
                    user_features = np.array([s.features for s in samples])
                    user_std = np.std(user_features, axis=0)
                    mean_std = np.mean(user_std)
                    
                    num_users = len(users_data)
                    if num_users <= 2:
                        variability_threshold = 6.0
                    elif num_users <= 5:
                        variability_threshold = 5.0
                    else:
                        variability_threshold = 3.5
                    
                    if mean_std > variability_threshold:
                        quality_issues.append(f"Usuario {user_id} con alta variabilidad: {mean_std:.4f}")
                    elif mean_std < 0.001:
                        quality_issues.append(f"Usuario {user_id} con baja variabilidad: {mean_std:.6f}")
            
            # 3. Verificar distribución de gestos
            gesture_distribution = {}
            for sample in self.real_training_samples:
                gesture = sample.gesture_name
                if gesture not in gesture_distribution:
                    gesture_distribution[gesture] = 0
                gesture_distribution[gesture] += 1
            
            if len(gesture_distribution) < 3:
                quality_issues.append(f"Pocos tipos de gestos: {len(gesture_distribution)}")
            
            quality_scores = [getattr(s, 'quality_score', 1.0) for s in self.real_training_samples]

            # 4. Verificar calidad de muestras individuales
            low_quality_samples = []
            
            for sample in self.real_training_samples:
                quality = getattr(sample, 'quality_score', 1.0)
                if quality <= 1.5:
                    if quality < 0.8:
                        low_quality_samples.append(sample)
                else:
                    if quality < 80.0:
                        low_quality_samples.append(sample)
            
            if len(low_quality_samples) > len(self.real_training_samples) * 0.2:
                quality_issues.append(f"Muchas muestras de baja calidad: {len(low_quality_samples)}/{len(self.real_training_samples)}")
            
            # Verificar sesiones por usuario (relajado para few-shot learning)
            session_counts = {}
            for user_id, samples in users_data.items():
                sessions = set(getattr(s, 'session_id', 'default') for s in samples)
                session_counts[user_id] = len(sessions)
                
                if len(sessions) < 1:
                    # Solo advertencia, no error crítico para few-shot learning
                    logger.info(f"Usuario {user_id} con {len(sessions)} sesión(es) - OK para redes siamesas")
                    
            # Reportar resultados
            if quality_issues:
                logger.error("Problemas de calidad detectados:")
                for issue in quality_issues:
                    logger.error(f"  - {issue}")
                return False
            
            logger.info("✓ Validación de calidad de datos: EXITOSA")
            logger.info(f"  - Usuarios: {len(users_data)}")
            logger.info(f"  - Distancia mínima inter-usuario: {min_inter_user_distance:.4f}")
            logger.info(f"  - Tipos de gestos: {len(gesture_distribution)}")
            logger.info(f"  - Distribución de gestos: {gesture_distribution}")
            logger.info(f"  - Sesiones promedio por usuario: {np.mean(list(session_counts.values())):.1f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando calidad: {e}")
            return False
    
    def create_real_training_pairs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crea pares de entrenamiento genuinos e impostores."""
        try:
            if not self.real_training_samples:
                raise ValueError("No hay muestras para crear pares")
            
            logger.info("Creando pares de entrenamiento...")
            
            # Agrupar muestras por usuario
            real_user_samples = {}
            for sample in self.real_training_samples:
                if sample.user_id not in real_user_samples:
                    real_user_samples[sample.user_id] = []
                real_user_samples[sample.user_id].append(sample)
            
            # Filtrar usuarios con suficientes muestras
            min_samples = self.config['min_samples_per_user']
            valid_real_users = {uid: samples for uid, samples in real_user_samples.items() 
                               if len(samples) >= min_samples}
            
            if len(valid_real_users) < 2:
                raise ValueError(f"Redes siamesas necesitan mínimo 2 usuarios con {min_samples}+ muestras")
            
            real_pairs = []
            
            # Crear pares genuinos (misma persona)
            genuine_pairs_created = 0
            for user_id, samples in valid_real_users.items():
                user_genuine_pairs = 0
                
                for i in range(len(samples)):
                    for j in range(i + 1, len(samples)):
                        session_i = getattr(samples[i], 'session_id', 'default')
                        session_j = getattr(samples[j], 'session_id', 'default')
                        
                        if session_i != session_j or (session_i == 'default' and session_j == 'default'):
                            real_pairs.append(RealTrainingPair(samples[i], samples[j], is_genuine=True))
                            user_genuine_pairs += 1
                            genuine_pairs_created += 1
                
                logger.info(f"Usuario {user_id}: {user_genuine_pairs} pares genuinos")
            
            # Crear pares impostores (personas diferentes)
            user_ids = list(valid_real_users.keys())
            impostor_pairs_created = 0
            
            if len(user_ids) == 2:
                # Caso especial: 2 usuarios
                user_id1, user_id2 = user_ids[0], user_ids[1]
                samples1 = valid_real_users[user_id1]
                samples2 = valid_real_users[user_id2]
                
                max_possible_impostors = len(samples1) * len(samples2)
                target_impostor_pairs = min(
                    max_possible_impostors,
                    int(genuine_pairs_created * 0.7)
                )
                
                logger.info(f"Modo 2 usuarios: Creando {target_impostor_pairs} pares impostores de {max_possible_impostors} posible")
                
                pairs_created = 0
                for s1 in samples1:
                    for s2 in samples2:
                        if pairs_created < target_impostor_pairs:
                            real_pairs.append(RealTrainingPair(s1, s2, is_genuine=False))
                            pairs_created += 1
                            impostor_pairs_created += 1
                        else:
                            break
                    if pairs_created >= target_impostor_pairs:
                        break
            else:
                # Caso normal: 3+ usuarios
                target_impostor_pairs = max(
                    int(genuine_pairs_created * 0.4),
                    min(genuine_pairs_created, 200)
                )
                
                for i, user_id1 in enumerate(user_ids):
                    for j, user_id2 in enumerate(user_ids[i + 1:], i + 1):
                        samples1 = valid_real_users[user_id1]
                        samples2 = valid_real_users[user_id2]
                        
                        max_pairs_between = min(50, len(samples1) * len(samples2) // 2)
                        pairs_between = 0
                        
                        for s1 in samples1:
                            for s2 in samples2:
                                if impostor_pairs_created < target_impostor_pairs and pairs_between < max_pairs_between:
                                    real_pairs.append(RealTrainingPair(s1, s2, is_genuine=False))
                                    impostor_pairs_created += 1
                                    pairs_between += 1
                                else:
                                    break
                            if pairs_between >= max_pairs_between:
                                break
                        
                        if impostor_pairs_created >= target_impostor_pairs:
                            break
                    if impostor_pairs_created >= target_impostor_pairs:
                        break
            
            # Validación
            min_impostor_ratio = 0.15 if len(user_ids) == 2 else 0.2
            
            if impostor_pairs_created < genuine_pairs_created * min_impostor_ratio:
                logger.warning(f"Balance subóptimo: {impostor_pairs_created} impostores vs {genuine_pairs_created} genuinos")
                logger.warning(f"Ratio: {impostor_pairs_created/(genuine_pairs_created + impostor_pairs_created):.1%}")

                if impostor_pairs_created < 10:
                    raise ValueError("Balance inadecuado para entrenamiento")
            else:
                logger.info(f"Balance aceptable: {impostor_pairs_created} impostores ({impostor_pairs_created/(genuine_pairs_created + impostor_pairs_created):.1%})")
                
            # Convertir a arrays numpy
            features_a = np.array([pair.sample1.features for pair in real_pairs])
            features_b = np.array([pair.sample2.features for pair in real_pairs])
            labels = np.array([1.0 if pair.is_genuine else 0.0 for pair in real_pairs])
            
            # Shuffle
            indices = np.random.permutation(len(labels))
            features_a = features_a[indices]
            features_b = features_b[indices]
            labels = labels[indices]
            
            self.total_genuine_pairs = genuine_pairs_created
            self.total_impostor_pairs = impostor_pairs_created
            
            logger.info(f"Pares creados exitosamente:")
            logger.info(f"  - Genuinos: {genuine_pairs_created}")
            logger.info(f"  - Impostores: {impostor_pairs_created}")
            logger.info(f"  - Total: {len(real_pairs)}")
            logger.info(f"  - Usuarios involucrados: {len(valid_real_users)}")
            logger.info(f"  - Ratio genuinos/impostores: {genuine_pairs_created/impostor_pairs_created:.2f}" if impostor_pairs_created > 0 else "  - Solo pares genuinos")
            
            return features_a, features_b, labels
            
        except Exception as e:
            logger.error(f"Error creando pares de entrenamiento: {e}")
            raise
        
    def build_real_base_network(self) -> Model:
        """Construye la red base para embeddings anatómicos."""
        try:
            logger.info("Construyendo red base para características anatómicas...")
            
            # Input layer
            input_layer = layers.Input(shape=(self.input_dim,), name='anatomical_features_real')
            
            x = input_layer
            
            # Normalización de entrada
            x = layers.BatchNormalization(name='input_normalization')(x)
            
            # Capas ocultas progresivas
            for i, units in enumerate(self.config['hidden_layers']):
                x = layers.Dense(
                    units,
                    activation=self.config['activation'],
                    kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']),
                    name=f'dense_real_{i+1}'
                )(x)
                
                if self.config['batch_normalization']:
                    x = layers.BatchNormalization(name=f'batch_norm_real_{i+1}')(x)
                
                x = layers.Dropout(self.config['dropout_rate'], name=f'dropout_real_{i+1}')(x)
            
            # Capa de embedding final
            embedding = layers.Dense(
                self.embedding_dim,
                activation='linear',
                name='embedding_real'
            )(x)
            
            # Normalización L2 del embedding
            embedding_normalized = layers.Lambda(
                lambda x: tf.nn.l2_normalize(x, axis=1),
                name='l2_normalize_real'
            )(embedding)
            
            # Crear modelo
            base_model = Model(inputs=input_layer, outputs=embedding_normalized, name='base_network_real')
            
            self.base_network = base_model
            
            total_params = base_model.count_params()
            logger.info(f"Red base construida: {self.input_dim} → {self.embedding_dim}")
            logger.info(f"  - Parámetros: {total_params:,}")
            logger.info(f"  - Capas ocultas: {self.config['hidden_layers']}")
            logger.info(f"  - Regularización L2: {self.config['l2_regularization']}")
            logger.info(f"  - Dropout: {self.config['dropout_rate']}")
            
            return base_model
            
        except Exception as e:
            logger.error(f"Error construyendo red base: {e}")
            raise
    
    def build_real_siamese_model(self) -> Model:
        """Construye el modelo siamés completo."""
        try:
            if self.base_network is None:
                self.build_real_base_network()
            
            logger.info("Construyendo modelo siamés...")
            
            # Inputs para las dos ramas
            input_a = layers.Input(shape=(self.input_dim,), name='input_a_real')
            input_b = layers.Input(shape=(self.input_dim,), name='input_b_real')
            
            # Procesar con red base (pesos compartidos)
            embedding_a = self.base_network(input_a)
            embedding_b = self.base_network(input_b)
            
            # Calcular distancia entre embeddings
            if self.config['distance_metric'] == 'euclidean':
                distance = layers.Lambda(
                    lambda embeddings: tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)),
                    name='euclidean_distance_real'
                )([embedding_a, embedding_b])
            elif self.config['distance_metric'] == 'cosine':
                distance = layers.Lambda(
                    lambda embeddings: 1.0 - tf.reduce_sum(embeddings[0] * embeddings[1], axis=1, keepdims=True),
                    name='cosine_distance_real'
                )([embedding_a, embedding_b])
            else:
                distance = layers.Lambda(
                    lambda embeddings: tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)),
                    name='euclidean_distance_real'
                )([embedding_a, embedding_b])
            
            # Crear modelo siamés
            siamese_model = Model(
                inputs=[input_a, input_b], 
                outputs=distance, 
                name='siamese_anatomical_real'
            )
            
            self.siamese_model = siamese_model
            
            total_params = siamese_model.count_params()
            logger.info(f"Modelo siamés construido: {total_params:,} parámetros")
            logger.info(f"  - Métrica: {self.config['distance_metric']}")
            
            return siamese_model
            
        except Exception as e:
            logger.error(f"Error construyendo modelo siamés: {e}")
            raise
    
    def _contrastive_loss_real(self, y_true, y_pred):
        """Función de pérdida contrastiva REAL."""
        margin = self.config['margin']
        
        loss_genuine = y_true * tf.square(y_pred)
        loss_impostor = (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
        
        return tf.reduce_mean(loss_genuine + loss_impostor)
    
    def _far_metric_real(self, y_true, y_pred):
        """Métrica FAR REAL con threshold dinámico."""
        y_pred_flat = tf.reshape(y_pred, [-1])
        y_true_flat = tf.reshape(y_true, [-1])
        
        threshold = tf.reduce_mean(y_pred_flat)
        
        predictions = tf.cast(y_pred_flat < threshold, tf.float32)
        
        impostor_mask = tf.cast(y_true_flat == 0, tf.float32)
        false_accepts = tf.reduce_sum(predictions * impostor_mask)
        total_impostors = tf.reduce_sum(impostor_mask)
        
        return tf.cond(
            total_impostors > 0,
            lambda: false_accepts / total_impostors,
            lambda: 0.0
        )
    
    def _frr_metric_real(self, y_true, y_pred):
        """Métrica FRR REAL con threshold dinámico."""
        y_pred_flat = tf.reshape(y_pred, [-1])
        y_true_flat = tf.reshape(y_true, [-1])
        
        threshold = tf.reduce_mean(y_pred_flat)
        
        predictions = tf.cast(y_pred_flat < threshold, tf.float32)
        
        genuine_mask = tf.cast(y_true_flat == 1, tf.float32)
        false_rejects = tf.reduce_sum((1 - predictions) * genuine_mask)
        total_genuines = tf.reduce_sum(genuine_mask)
        
        return tf.cond(
            total_genuines > 0,
            lambda: false_rejects / total_genuines,
            lambda: 0.0
        )
    
    def compile_real_model(self):
        """Compila el modelo siamés."""
        try:
            if self.siamese_model is None:
                self.build_real_siamese_model()
            
            logger.info("Compilando modelo siamés...")
            
            optimizer = optimizers.Adam(learning_rate=self.config['learning_rate'])
            
            if self.config['loss_function'] == 'contrastive':
                loss_function = self._contrastive_loss_real
            elif self.config['loss_function'] == 'binary_crossentropy':
                loss_function = 'binary_crossentropy'
            else:
                loss_function = self._contrastive_loss_real
            
            self.siamese_model.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=[self._far_metric_real, self._frr_metric_real]
            )
            
            self.is_compiled = True
            
            logger.info(f"Modelo compilado:")
            logger.info(f"  - Optimizador: Adam (lr={self.config['learning_rate']})")
            logger.info(f"  - Pérdida: {self.config['loss_function']}")
            
        except Exception as e:
            logger.error(f"Error compilando modelo: {e}")
            raise
    
    #NUEVO
    def _create_real_training_callbacks(self) -> List:
        """Crea callbacks REALES para el entrenamiento."""
        callback_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Reduce learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['patience'] // 2,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = str(self.model_save_path)
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callback_list.append(checkpoint)
        
        return callback_list
    #NUEVO
    def _update_real_training_history(self, history, training_time: float):
        """Actualiza el historial de entrenamiento REAL."""
        try:
            self.training_history.loss = history.history['loss']
            self.training_history.val_loss = history.history['val_loss']
            
            if 'far_metric_real' in history.history:
                self.training_history.far_history = history.history['far_metric_real']
            if 'frr_metric_real' in history.history:
                self.training_history.frr_history = history.history['frr_metric_real']
            
            self.training_history.total_training_time = training_time
            self.training_history.best_epoch = np.argmin(self.training_history.val_loss)
            
            logger.info("Historial actualizado")
            
        except Exception as e:
            logger.error(f"Error actualizando historial: {e}")
        
    def train_with_real_data(self, database, validation_split: float = 0.2) -> RealTrainingHistory:
        """Entrena el modelo con datos REALES de usuarios."""
        try:
            logger.info("=== INICIANDO ENTRENAMIENTO ===")
            
            # 1. Cargar datos
            if not self.load_real_training_data_from_database(database):
                raise ValueError("No se pudieron cargar datos suficientes")
            
            # 2. Validar calidad
            if not self.validate_real_data_quality():
                raise ValueError("Datos no cumplen criterios de calidad")
            
            # 3. Crear pares
            features_a, features_b, labels = self.create_real_training_pairs()
            
            # 4. División estratificada
            logger.info(f"Dividiendo {len(labels)} pares de entrenamiento...")
            
            genuine_indices = np.where(labels == 1)[0]
            impostor_indices = np.where(labels == 0)[0]
            
            validation_split = 0.15
            n_val_genuine = max(5, int(len(genuine_indices) * validation_split))
            n_val_impostor = max(5, int(len(impostor_indices) * validation_split))
            
            logger.info(f"Pares disponibles: {len(genuine_indices)} genuinos, {len(impostor_indices)} impostores")
            logger.info(f"Para validación: {n_val_genuine} genuinos, {n_val_impostor} impostores")

            np.random.seed(42)
            val_genuine = np.random.choice(genuine_indices, n_val_genuine, replace=False)
            val_impostor = np.random.choice(impostor_indices, n_val_impostor, replace=False)
            
            val_indices = np.concatenate([val_genuine, val_impostor])
            train_indices = np.setdiff1d(np.arange(len(labels)), val_indices)
            
            logger.info(f"División: {len(train_indices)} entrenamiento, {len(val_indices)} validación")
            logger.info(f"TOTAL USADO: {len(train_indices) + len(val_indices)} de {len(labels)} pares disponibles")
            
            train_a, train_b, train_labels = features_a[train_indices], features_b[train_indices], labels[train_indices]
            val_a, val_b, val_labels = features_a[val_indices], features_b[val_indices], labels[val_indices]
            
            logger.info(f"División de datos REALES:")
            logger.info(f"  - Entrenamiento: {len(train_labels)} pares")
            logger.info(f"  - Validación: {len(val_labels)} pares")
            logger.info(f"  - Genuinos entrenamiento: {np.sum(train_labels)}")
            logger.info(f"  - Impostores entrenamiento: {np.sum(1-train_labels)}")
            
            # 5. Compilar
            if not self.is_compiled:
                self.compile_real_model()
            
            # 6. Callbacks
            callbacks_list = self._create_real_training_callbacks()
            
            # 7. Entrenar
            logger.info("Iniciando entrenamiento con datos...")
            start_time = time.time()
            
            history = self.siamese_model.fit(
                [train_a, train_b], train_labels,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_data=([val_a, val_b], val_labels),
                callbacks=callbacks_list,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # 8. Actualizar historial
            self._update_real_training_history(history, training_time)
            self.is_trained = True
            
            # 9. Evaluar
            final_metrics = self.evaluate_real_model(val_a, val_b, val_labels)
            self.current_metrics = final_metrics
            
            logger.info("=== ENTRENAMIENTO COMPLETADO ===")
            logger.info(f"  - Tiempo: {training_time:.2f}s")
            logger.info(f"  - Épocas: {len(history.history['loss'])}")
            logger.info(f"  - EER: {final_metrics.eer:.4f}")
            logger.info(f"  - AUC: {final_metrics.auc_score:.4f}")
            logger.info(f"  - Threshold óptimo: {final_metrics.threshold:.4f}")
            
            self.is_trained = True
            logger.info("✓ Red anatómica marcada como entrenada")
            
            # Guardar modelo
            if self.save_real_model():
                logger.info("✓ Modelo anatómico guardado con metadatos")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Error durante entrenamiento: {e}")
            raise
    
    def _create_user_stratified_split(self, validation_split: float) -> Tuple[np.ndarray, np.ndarray]:
        """Crea división estratificada por usuarios REALES."""
        try:
            logger.info(f"Iniciando división con {len(self.real_training_samples)} pares totales")
            
            # Agrupar índices por usuario
            user_indices = {}
            for i, sample in enumerate(self.real_training_samples):
                if sample.user_id not in user_indices:
                    user_indices[sample.user_id] = []
                user_indices[sample.user_id].append(i)
            
            # Dividir usuarios (no muestras)
            user_ids = list(user_indices.keys())
            
            # VALIDACIÓN PREVIA
            if len(user_ids) < 3:
                logger.warning(f"Solo {len(user_ids)} usuarios - usando división MANUAL por muestras")
                
                # DIVISIÓN MANUAL QUE USA TODOS LOS DATOS
                all_indices = np.arange(len(self.real_training_samples))
                labels_for_split = np.array([1.0 if pair.is_genuine else 0.0 for pair in self.real_training_samples])
                
                # Separar por tipo de par
                genuine_indices = all_indices[labels_for_split == 1]
                impostor_indices = all_indices[labels_for_split == 0]
                
                # Dividir cada tipo proporcionalmente
                n_val_genuine = max(3, int(len(genuine_indices) * validation_split))
                n_val_impostor = max(3, int(len(impostor_indices) * validation_split))
                
                # Selección aleatoria estratificada
                np.random.seed(42)
                val_genuine = np.random.choice(genuine_indices, n_val_genuine, replace=False)
                val_impostor = np.random.choice(impostor_indices, n_val_impostor, replace=False)
                
                val_indices = np.concatenate([val_genuine, val_impostor])
                train_indices = np.setdiff1d(all_indices, val_indices)
                
                logger.info(f"División manual exitosa:")
                logger.info(f"  - Entrenamiento: {len(train_indices)} pares")
                logger.info(f"  - Validación: {len(val_indices)} pares") 
                logger.info(f"  - TOTAL USADO: {len(train_indices) + len(val_indices)} de {len(all_indices)}")
                
                return train_indices, val_indices
            
            else:
                # CON SUFICIENTES USUARIOS: DIVISIÓN POR USUARIOS
                train_users, val_users = train_test_split(
                    user_ids, 
                    test_size=validation_split,
                    random_state=42
                )
    
                # Obtener índices de muestras para cada conjunto
                train_sample_indices = []
                val_sample_indices = []
                
                for user_id in train_users:
                    train_sample_indices.extend(user_indices[user_id])
                
                for user_id in val_users:
                    val_sample_indices.extend(user_indices[user_id])
                
                logger.info(f"División estratificada por usuarios REALES:")
                logger.info(f"  - Usuarios entrenamiento: {len(train_users)}")
                logger.info(f"  - Usuarios validación: {len(val_users)}")
                
                return np.array(train_sample_indices), np.array(val_sample_indices)
                
        except Exception as e:
            logger.error("Error en división estratificada por usuarios", e)
            # Fallback que usa TODOS los datos
            total_samples = len(self.real_training_samples)
            val_size = int(total_samples * validation_split)
            train_size = total_samples - val_size
            
            return np.arange(train_size), np.arange(train_size, total_samples)
    
    def _create_real_training_callbacks(self) -> List:
        """Crea callbacks REALES para el entrenamiento."""
        callback_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['patience'] // 2,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpoint
        #checkpoint_path = os.path.join(self.model_save_path, 'anatomical_model.h5')
        checkpoint_path = str(self.model_save_path)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callback_list.append(checkpoint)
        
        return callback_list
    
    def _update_real_training_history(self, history, training_time: float):
        """Actualiza el historial de entrenamiento."""
        try:
            self.training_history.loss = history.history['loss']
            self.training_history.val_loss = history.history['val_loss']
            
            # Métricas adicionales si están disponibles
            if 'far_metric_real' in history.history:
                self.training_history.far_history = history.history['far_metric_real']
            if 'frr_metric_real' in history.history:
                self.training_history.frr_history = history.history['frr_metric_real']
            
            # Información de entrenamiento
            self.training_history.total_training_time = training_time
            self.training_history.best_epoch = np.argmin(self.training_history.val_loss)
            
            logger.info("Historial de entrenamiento actualizado")
            
        except Exception as e:
            log_error("Error actualizando historial", e)
    
    def evaluate_real_model(self, features_a: np.ndarray, features_b: np.ndarray, 
                       labels: np.ndarray) -> RealModelMetrics:
        """Evalúa el modelo."""
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                raise ValueError("Modelo no entrenado")
            
            logger.info("Evaluando modelo...")
            
            # Predecir distancias
            distances = self.siamese_model.predict([features_a, features_b])
            distances = distances.flatten()
            
            total_samples = len(labels)
            genuine_count = int(np.sum(labels == 1))
            impostor_count = int(np.sum(labels == 0))
            
            if total_samples == 0:
                raise ValueError("No hay datos para evaluar")
            
            use_robust_method = total_samples < 20
            
            if use_robust_method:
                logger.info(f"Dataset pequeño ({total_samples}) - método robusto")
                
                if genuine_count > 0 and impostor_count > 0:
                    genuine_distances = distances[labels == 1]
                    impostor_distances = distances[labels == 0]
                    
                    genuine_median = np.median(genuine_distances)
                    impostor_median = np.median(impostor_distances)
                    
                    eer_threshold = (genuine_median + impostor_median) / 2.0
                    
                    logger.info(f"  Threshold calculado: mediana genuinos={genuine_median:.4f}, impostores={impostor_median:.4f}")

                elif genuine_count > 0:
                    eer_threshold = np.percentile(distances[labels == 1], 75)
                    logger.info("  Solo genuinos disponibles - usando percentil 75")

                elif impostor_count > 0:
                    eer_threshold = np.percentile(distances[labels == 0], 25)
                    logger.info("  Solo impostores disponibles - usando percentil 25")

                else:
                    eer_threshold = np.mean(distances)
                    logger.info("  Fallback - usando promedio de distancias")

                
                predictions = distances < eer_threshold
                
                if impostor_count > 0:
                    false_accepts = np.sum((predictions == 1) & (labels == 0))
                    far = false_accepts / impostor_count
                else:
                    far = 0.0
                
                if genuine_count > 0:
                    false_rejects = np.sum((predictions == 0) & (labels == 1))
                    frr = false_rejects / genuine_count
                else:
                    frr = 0.0
                
                eer = (far + frr) / 2.0
                
                try:
                    if genuine_count > 0 and impostor_count > 0:
                        fpr, tpr, _ = roc_curve(labels, 1 - distances)
                        auc_score = auc(fpr, tpr)
                    else:
                        auc_score = 0.5
                except Exception:
                    auc_score = 0.5
            else:
                logger.info(f"Dataset grande ({total_samples} muestras) - método estándar")
                
                try:
                    fpr, tpr, thresholds = roc_curve(labels, 1 - distances)
                    auc_score = auc(fpr, tpr)
                    
                    fnr = 1 - tpr
                    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
                    eer_threshold = thresholds[eer_idx]
                    eer = fpr[eer_idx]
                    
                    logger.info(f"  EER calculado: {eer:.4f} con threshold: {eer_threshold:.4f}")

                except Exception:
                    logger.error(f"Error en cálculo ROC estándar: {e}")

                    genuine_distances = distances[labels == 1] if genuine_count > 0 else []
                    impostor_distances = distances[labels == 0] if impostor_count > 0 else []
                    
                    if len(genuine_distances) > 0 and len(impostor_distances) > 0:
                        eer_threshold = (np.median(genuine_distances) + np.median(impostor_distances)) / 2.0
                    else:
                        eer_threshold = np.mean(distances)
                    
                    auc_score = 0.5
                    eer = 0.5
                
                predictions = distances < eer_threshold
                
                try:
                    cm = confusion_matrix(labels, predictions)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    else:
                        if impostor_count > 0:
                            far = np.sum((predictions == 1) & (labels == 0)) / impostor_count
                        else:
                            far = 0.0
                        if genuine_count > 0:
                            frr = np.sum((predictions == 0) & (labels == 1)) / genuine_count
                        else:
                            frr = 0.0
                except Exception:
                    far = np.sum((predictions == 1) & (labels == 0)) / max(1, impostor_count)
                    frr = np.sum((predictions == 0) & (labels == 1)) / max(1, genuine_count)
            
            accuracy = accuracy_score(labels, predictions)
            
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            tn = np.sum((predictions == 0) & (labels == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if genuine_count > 0:
                estimated_users = max(2, int((1 + np.sqrt(1 + 8 * genuine_count)) / 2))
            else:
                estimated_users = max(2, total_samples // 8)
            
            metrics = RealModelMetrics(
                far=float(far),
                frr=float(frr),
                eer=float(eer),
                auc_score=float(auc_score),
                accuracy=float(accuracy),
                threshold=float(eer_threshold),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1_score),
                total_genuine_pairs=genuine_count,
                total_impostor_pairs=impostor_count,
                users_in_test=estimated_users,
                cross_validation_score=0.0
            )
            
            self.optimal_threshold = eer_threshold
            
            # Logging detallado y adaptativo
            method_used = "Robusto (dataset pequeño)" if use_robust_method else "Estándar (dataset grande)"
            logger.info("Evaluación REAL completada:")
            logger.info(f"  - Método utilizado: {method_used}")
            logger.info(f"  - FAR: {far:.4f} ({fp} falsos positivos de {impostor_count} impostores)")
            logger.info(f"  - FRR: {frr:.4f} ({fn} falsos negativos de {genuine_count} genuinos)")
            logger.info(f"  - EER: {eer:.4f}")
            logger.info(f"  - AUC: {auc_score:.4f}")
            logger.info(f"  - Accuracy: {accuracy:.4f}")
            logger.info(f"  - Threshold óptimo: {eer_threshold:.4f}")
            logger.info(f"  - Pares genuinos evaluados: {genuine_count}")
            logger.info(f"  - Pares impostores evaluados: {impostor_count}")
            logger.info(f"  - Usuarios estimados en test: {estimated_users}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluando modelo: {e}")
            raise
        
    def predict_similarity_real(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Predice similitud entre dos vectores."""
        try:
            if not self.is_trained:
                raise ValueError("Modelo no entrenado")
            
            if self.siamese_model is None:
                raise ValueError("Modelo no inicializado")
            
            if len(features1) != self.input_dim or len(features2) != self.input_dim:
                raise ValueError(f"Dimensiones incorrectas")
            
            features1 = np.array(features1, dtype=np.float32).reshape(1, -1)
            features2 = np.array(features2, dtype=np.float32).reshape(1, -1)
            
            distance = self.siamese_model.predict([features1, features2])[0][0]
            
            similarity = 1.0 / (1.0 + distance)
            similarity = np.clip(similarity, 0.0, 1.0)
            
            logger.info(f"Predicción: distancia={distance:.4f}, similitud={similarity:.4f}")
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return 0.0
    
    def authenticate_real(self, query_features: np.ndarray, 
                         reference_templates: List[np.ndarray]) -> Tuple[bool, float, Dict[str, Any]]:
        """Autentica usuario comparando con templates."""
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado para autenticación")
                return False, 0.0, {'error': 'Modelo no entrenado'}
            
            if not reference_templates:
                logger.error("No hay templates de referencia")
                return False, 0.0, {'error': 'Sin templates'}
            
            logger.info(f"Autenticación: comparando con {len(reference_templates)} templates")
            
            similarities = []
            for i, template in enumerate(reference_templates):
                try:
                    similarity = self.predict_similarity_real(query_features, template)
                    similarities.append(similarity)
                    logger.info(f"  Template {i+1}: {similarity:.4f}")
                except Exception as e:
                    logger.error(f"Error con template {i+1}: {e}")
                    continue
            
            if not similarities:
                return False, 0.0, {'error': 'Error en similitudes'}
            
            max_similarity = np.max(similarities)
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            threshold_decision = max_similarity > self.optimal_threshold
            
            consistency_bonus = 0.0
            if len(similarities) > 1:
                high_similarities = [s for s in similarities if s > self.optimal_threshold]
                consistency_bonus = len(high_similarities) / len(similarities) * 0.1
            
            final_score = min(1.0, max_similarity + consistency_bonus)
            is_authentic = threshold_decision and final_score > self.optimal_threshold
            
            details = {
                'max_similarity': max_similarity,
                'mean_similarity': mean_similarity,
                'std_similarity': std_similarity,
                'num_references': len(reference_templates),
                'threshold_used': self.optimal_threshold,
                'consistency_bonus': consistency_bonus,
                'final_score': final_score,
                'similarities': similarities,
                'model_trained': self.is_trained,
                'authentication_method': 'real_siamese_anatomical'
            }
                        
            logger.info(f"Resultado autenticación:")
            logger.info(f"  - Auténtico: {is_authentic}")
            logger.info(f"  - Score máximo: {max_similarity:.4f}")
            logger.info(f"  - Score final: {final_score:.4f}")
            logger.info(f"  - Threshold: {self.optimal_threshold:.4f}")
            logger.info(f"  - Templates consultados: {len(reference_templates)}")
            
            return is_authentic, final_score, details
            
        except Exception as e:
            logger.error(f"Error en autenticación: {e}")
            return False, 0.0, {'error': str(e)}
    
    def save_real_model(self, filepath: Optional[str] = None) -> bool:
        """Guarda el modelo REAL entrenado."""
        try:
            if not self.is_trained:
                logger.error("No hay modelo entrenado para guardar")
                return False
            
            if filepath is None:
                models_dir = Path(get_config('paths.models', 'biometric_data/models'))
                filepath = models_dir / 'anatomical_model.h5'
            
            model_path = Path(filepath)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar modelo
            self.siamese_model.save(str(model_path))
            
            # Guardar metadatos
            from datetime import datetime
            metadata = {
                'embedding_dim': int(self.embedding_dim),
                'input_dim': int(self.input_dim),
                'optimal_threshold': float(self.optimal_threshold),
                'is_trained': True,
                'training_samples': int(self.total_genuine_pairs + self.total_impostor_pairs),
                'users_trained_count': int(self.users_trained_count),
                'total_genuine_pairs': int(self.total_genuine_pairs),
                'total_impostor_pairs': int(self.total_impostor_pairs),
                'save_timestamp': datetime.now().isoformat(),
                'version': '2.0',
                'config': self.config
            }
            
            if self.current_metrics:
                metadata['metrics'] = {
                    'far': float(self.current_metrics.far),
                    'frr': float(self.current_metrics.frr),
                    'eer': float(self.current_metrics.eer),
                    'auc_score': float(self.current_metrics.auc_score),
                    'accuracy': float(self.current_metrics.accuracy),
                    'threshold': float(self.current_metrics.threshold),
                    'precision': float(self.current_metrics.precision),
                    'recall': float(self.current_metrics.recall),
                    'f1_score': float(self.current_metrics.f1_score)
                }
            
            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Modelo anatómico guardado: {model_path}")
            logger.info(f"Metadatos: {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            return False
    
    def load_real_model(self, filepath: Optional[str] = None) -> bool:
        """Carga un modelo REAL previamente entrenado."""
        try:
            if filepath is None:
                models_dir = Path(get_config('paths.models', 'biometric_data/models'))
                filepath = models_dir / 'anatomical_model.h5'
            
            model_path = Path(filepath)
            
            if not model_path.exists():
                logger.error(f"Modelo no existe: {model_path}")
                return False
            
            # Construir arquitectura
            if not self.base_network:
                self.build_real_base_network()
            
            if not self.siamese_model:
                self.build_real_siamese_model()
            
            if not self.is_compiled:
                self.compile_real_model()
            
            # Cargar pesos
            self.siamese_model.load_weights(str(model_path))
            self.is_trained = True
            self.is_compiled = True
            
            logger.info(f"Modelo anatómico cargado: {model_path}")
            logger.info(f"Parámetros: {self.siamese_model.count_params():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.is_trained = False
            return False
    
    def get_real_model_summary(self) -> Dict[str, Any]:
        """Obtiene resumen completo del modelo REAL."""
        summary = {
            "architecture": {
                "embedding_dim": self.embedding_dim,
                "input_dim": self.input_dim,
                "hidden_layers": self.config['hidden_layers'],
                "total_parameters": self.siamese_model.count_params() if self.siamese_model else 0,
                "distance_metric": self.config['distance_metric'],
                "model_type": "Real Siamese Anatomical Network"
            },
            "training": {
                "is_trained": self.is_trained,
                "users_trained": self.users_trained_count,
                "genuine_pairs": self.total_genuine_pairs,
                "impostor_pairs": self.total_impostor_pairs,
                "optimal_threshold": self.optimal_threshold,
                "training_time": getattr(self.training_history, 'total_training_time', 0),
                "data_source": "real_users_database"
            },
            "performance": {},
            "status": {
                "model_compiled": self.is_compiled,
                "base_network_built": self.base_network is not None,
                "siamese_model_built": self.siamese_model is not None,
                "ready_for_inference": self.is_trained and self.is_compiled,
                "version": "2.0"
            }
        }
        
        if self.current_metrics:
            summary["performance"] = {
                "far": self.current_metrics.far,
                "frr": self.current_metrics.frr,
                "eer": self.current_metrics.eer,
                "auc_score": self.current_metrics.auc_score,
                "accuracy": self.current_metrics.accuracy,
                "optimal_threshold": self.current_metrics.threshold
            }
        
        return summary


# ===== INSTANCIA GLOBAL =====
_real_siamese_anatomical_instance = None

def get_real_siamese_anatomical_network(embedding_dim: int = 64, 
                                       input_dim: int = 180) -> RealSiameseAnatomicalNetwork:
    """Obtiene instancia global de la red siamesa anatómica REAL."""
    global _real_siamese_anatomical_instance
    
    if _real_siamese_anatomical_instance is None:
        _real_siamese_anatomical_instance = RealSiameseAnatomicalNetwork(embedding_dim, input_dim)
    
    # Verificar modelo guardado
    if not _real_siamese_anatomical_instance.is_trained:
        try:
            models_dir = Path('biometric_data/models')
            model_path = models_dir / 'anatomical_model.h5'
            
            if model_path.exists():
                logger.info(f"Cargando modelo anatómico: {model_path}")
                try:
                    if _real_siamese_anatomical_instance.siamese_model is None:
                        _real_siamese_anatomical_instance.build_real_base_network()
                        _real_siamese_anatomical_instance.build_real_siamese_model()
                        _real_siamese_anatomical_instance.compile_real_model()
                    
                    _real_siamese_anatomical_instance.siamese_model.load_weights(str(model_path))
                    _real_siamese_anatomical_instance.is_trained = True
                    
                    logger.info(f"✅ Red anatómica cargada: {model_path}")
                    logger.info(f"✅ Estado: is_trained = {_real_siamese_anatomical_instance.is_trained}")
                    
                except Exception as load_error:
                    logger.warning(f"Error cargando modelo: {load_error}")
            else:
                logger.info(f"No se encontró modelo guardado: {model_path}")
        
        except Exception as e:
            logger.warning(f"Error verificando modelo: {e}")
    
    return _real_siamese_anatomical_instance


# Alias para compatibilidad
SiameseAnatomicalNetwork = RealSiameseAnatomicalNetwork
get_siamese_anatomical_network = get_real_siamese_anatomical_network