"""
MÓDULO 10: SIAMESE_DYNAMIC_NETWORK
Red Siamesa REAL para características dinámicas temporales (100% SIN SIMULACIÓN)
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
from datetime import datetime

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, optimizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow no disponible - red siamesa dinámica limitada")

# Scikit-learn imports
try:
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, roc_curve, auc
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn no disponible - métricas limitadas")

# Importar módulos anteriores
try:
    from app.core.config_manager import get_config, get_logger, log_error, log_info
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


def log_warning(message: str):
    """Función de conveniencia para logging de warnings."""
    try:
        logger.warning(message)
    except:
        print(f"WARNING: {message}")


@dataclass
class RealDynamicSample:
    """Muestra de secuencia dinámica temporal de usuario."""
    user_id: str
    sequence_id: str
    temporal_features: np.ndarray
    gesture_sequence: List[str]
    transition_types: List[str]
    timestamp: float
    duration: float
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealTemporalPair:
    """Par de secuencias temporales para entrenamiento."""
    sample1: RealDynamicSample
    sample2: RealDynamicSample
    is_genuine: bool
    temporal_distance: Optional[float] = None
    confidence: float = 1.0


@dataclass
class RealTemporalMetrics:
    """Métricas específicas para evaluación temporal."""
    far: float
    frr: float
    eer: float
    auc_score: float
    accuracy: float
    threshold: float
    precision: float
    recall: float
    f1_score: float
    sequence_correlation: float
    temporal_consistency: float
    rhythm_similarity: float
    validation_samples: int


@dataclass
class RealTemporalTrainingHistory:
    """Historial de entrenamiento para modelo temporal."""
    loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    sequence_accuracy: List[float] = field(default_factory=list)
    temporal_loss: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    far_history: List[float] = field(default_factory=list)
    frr_history: List[float] = field(default_factory=list)
    eer_history: List[float] = field(default_factory=list)
    best_epoch: int = 0
    total_training_time: float = 0.0


class RealSiameseDynamicNetwork:
    """
    Red Siamesa para autenticación biométrica basada en características dinámicas temporales.
    Implementa arquitectura twin network con LSTM/BiLSTM para procesar secuencias.
    """
    
    def __init__(self, embedding_dim: int = 128, sequence_length: int = 50, feature_dim: int = 320):
        """Inicializa la red siamesa dinámica."""
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow no disponible")
        
        self.logger = get_logger()
        
        # Configuración del modelo
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.config = self._load_real_dynamic_config()
        
        # Arquitectura del modelo
        self.base_network = None
        self.siamese_model = None
        self.is_compiled = False
        
        # Estado de entrenamiento
        self.training_history = RealTemporalTrainingHistory()
        self.is_trained = False
        self.optimal_threshold = 0.5
        
        # Dataset REAL y métricas
        self.real_training_samples: List[RealDynamicSample] = []
        self.real_validation_samples: List[RealDynamicSample] = []
        self.current_metrics: Optional[RealTemporalMetrics] = None
        
        # Rutas de guardado
        self.model_save_path = self._get_real_model_save_path()
        
        # Estadísticas
        self.users_trained_count = 0
        
        logger.info("RealSiameseDynamicNetwork inicializada")
    
    def _load_real_dynamic_config(self) -> Dict[str, Any]:
        """Carga configuración de la red siamesa dinámica."""
        real_config = {
            'sequence_processing': 'bidirectional_lstm',
            'lstm_units': [128, 64],
            'dropout_rate': 0.3,
            'recurrent_dropout': 0.2,
            'dense_layers': [256, 128],
            'temporal_pooling': 'attention',
            'sequence_normalization': 'layer_norm',
            
            'use_masking': True,
            'return_sequences': False,
            'stateful': False,
            'max_sequence_length': 50,
            'min_sequence_length': 5,
            
            'learning_rate': 5e-4,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 8,
            'min_lr': 1e-6,
            
            'loss_function': 'contrastive',
            'margin': 0.8,
            'distance_metric': 'euclidean',
            
            'min_samples_per_user': 15,
            'min_users_for_training': 2,
            'quality_threshold': 80.0,
            'temporal_consistency_threshold': 0.7,
            
            'use_temporal_augmentation': True,
            'time_shift_range': 0.1,
            'speed_variation_range': 0.2,
            'noise_level': 0.01,
        }
        
        logger.info("Configuración de red dinámica cargada")
        return real_config
    
    def _get_real_model_save_path(self) -> Path:
        """Obtiene ruta para guardar modelos."""
        models_dir = Path(get_config('paths.models', 'biometric_data/models'))
        return models_dir / 'dynamic_model.h5'
    
    def build_real_base_network(self) -> Model:
        """Construye la red base temporal con BiLSTM."""
        try:
            logger.info("Construyendo red base temporal...")
            
            # Input layer para secuencias temporales
            input_layer = layers.Input(
                shape=(self.sequence_length, self.feature_dim), 
                name='real_dynamic_sequence'
            )
            
            x = input_layer
            
            # Masking para secuencias de longitud variable
            if self.config['use_masking']:
                x = layers.Masking(mask_value=0.0, name='real_sequence_masking')(x)
                logger.info("  - Masking aplicado para secuencias variables")
            
            # Normalización de secuencias
            if self.config['sequence_normalization'] == 'layer_norm':
                x = layers.LayerNormalization(name='real_sequence_layer_norm')(x)
                logger.info("  - Layer normalization aplicada")
            
            # Capas temporales
            x = self._build_real_temporal_layers(x)
            
            # Pooling temporal
            x = self._build_real_temporal_pooling(x)
            
            # Capas densas finales
            for i, units in enumerate(self.config['dense_layers']):
                x = layers.Dense(
                    units, 
                    activation='relu',
                    name=f'real_dense_temporal_{i+1}',
                    kernel_regularizer=keras.regularizers.l2(0.001)
                )(x)
                
                if self.config['dropout_rate'] > 0:
                    x = layers.Dropout(
                        self.config['dropout_rate'], 
                        name=f'real_dropout_temporal_{i+1}'
                    )(x)
            
            # Embedding final
            embedding = layers.Dense(
                self.embedding_dim, 
                activation='linear',
                name='real_temporal_embedding',
                kernel_regularizer=keras.regularizers.l2(0.001)
            )(x)
            
            # Normalización L2
            embedding_normalized = layers.Lambda(
                lambda x: tf.nn.l2_normalize(x, axis=1), 
                name='real_l2_normalize_temporal'
            )(embedding)
            
            # Crear modelo base
            base_model = Model(
                inputs=input_layer, 
                outputs=embedding_normalized, 
                name='real_temporal_base_network'
            )
            
            self.base_network = base_model
            
            total_params = base_model.count_params()
            logger.info(f"Red base temporal construida: ({self.sequence_length}, {self.feature_dim}) → {self.embedding_dim}")
            logger.info(f"  - Parámetros totales: {total_params:,}")
            logger.info(f"  - Arquitectura: {self.config['sequence_processing']}")
            logger.info(f"  - LSTM units: {self.config['lstm_units']}")
            logger.info(f"  - Dropout: {self.config['dropout_rate']}")
            logger.info(f"  - Pooling: {self.config['temporal_pooling']}")
            
            return base_model
            
        except Exception as e:
            logger.error(f"Error construyendo red base temporal: {e}")
            raise
    
    def _build_real_temporal_layers(self, x):
        """Construye las capas temporales (LSTM/BiLSTM)."""
        try:
            lstm_units = self.config['lstm_units']
            processing_type = self.config['sequence_processing']
            
            logger.info(f"=== CONSTRUYENDO CAPAS TEMPORALES ===")
            logger.info(f"Input tensor shape: {x.shape}")
            logger.info(f"Processing type: {processing_type}")
            logger.info(f"LSTM units: {lstm_units}")
            logger.info(f"Temporal pooling: {self.config['temporal_pooling']}")
            
            for i, units in enumerate(lstm_units):
                logger.info(f"--- Capa LSTM {i+1}/{len(lstm_units)} ---")
                
                # IMPORTANTE: Si usamos pooling personalizado, todas las capas retornan secuencias
                if self.config['temporal_pooling'] in ['attention', 'last']:
                    return_sequences = True
                    logger.info(f"Capa {i+1}: return_sequences=True (pooling personalizado detectado)")
                else:
                    return_sequences = i < len(lstm_units) - 1
                    logger.info(f"Capa {i+1}: return_sequences={return_sequences} (pooling estándar)")
                
                logger.info(f"Antes de construir capa {i+1}: tensor shape = {x.shape}")
                
                if processing_type == 'bidirectional_lstm':
                    logger.info(f"Construyendo Bidirectional LSTM con {units} unidades...")
                    try:
                        x = layers.Bidirectional(
                            layers.LSTM(
                                units,
                                return_sequences=return_sequences,
                                dropout=self.config['dropout_rate'],
                                recurrent_dropout=self.config['recurrent_dropout'],
                                kernel_regularizer=keras.regularizers.l2(0.001),
                                name=f'real_lstm_{i+1}'
                            ),
                            name=f'real_bidirectional_lstm_{i+1}'
                        )(x)
                        logger.info(f"✓ Bidirectional LSTM {i+1} construido exitosamente")
                        logger.info(f"Después de BiLSTM {i+1}: tensor shape = {x.shape}")
                    except Exception as lstm_error:
                        logger.error(f"ERROR en BiLSTM {i+1}: {lstm_error}")
                        logger.error(f"Config dropout: {self.config['dropout_rate']}")
                        logger.error(f"Config recurrent_dropout: {self.config['recurrent_dropout']}")
                        raise
                        
                elif processing_type == 'lstm':
                    logger.info(f"Construyendo LSTM simple con {units} unidades...")
                    try:
                        x = layers.LSTM(
                            units,
                            return_sequences=return_sequences,
                            dropout=self.config['dropout_rate'],
                            recurrent_dropout=self.config['recurrent_dropout'],
                            kernel_regularizer=keras.regularizers.l2(0.001),
                            name=f'real_lstm_{i+1}'
                        )(x)
                        logger.info(f"✓ LSTM {i+1} construido exitosamente")
                        logger.info(f"Después de LSTM {i+1}: tensor shape = {x.shape}")
                    except Exception as lstm_error:
                        logger.error(f"ERROR en LSTM {i+1}: {lstm_error}")
                        raise
                        
                elif processing_type == 'gru':
                    logger.info(f"Construyendo GRU con {units} unidades...")
                    try:
                        x = layers.GRU(
                            units,
                            return_sequences=return_sequences,
                            dropout=self.config['dropout_rate'],
                            recurrent_dropout=self.config['recurrent_dropout'],
                            kernel_regularizer=keras.regularizers.l2(0.001),
                            name=f'real_gru_{i+1}'
                        )(x)
                        logger.info(f"✓ GRU {i+1} construido exitosamente")
                        logger.info(f"Después de GRU {i+1}: tensor shape = {x.shape}")
                    except Exception as gru_error:
                        logger.error(f"ERROR en GRU {i+1}: {gru_error}")
                        raise
                
                # Normalización entre capas
                if i < len(lstm_units) - 1:
                    logger.info(f"Aplicando LayerNormalization después de capa {i+1}...")
                    try:
                        x = layers.LayerNormalization(name=f'real_layer_norm_{i+1}')(x)
                        logger.info(f"✓ LayerNormalization {i+1} aplicada exitosamente")
                        logger.info(f"Después de LayerNorm {i+1}: tensor shape = {x.shape}")
                    except Exception as norm_error:
                        logger.error(f"ERROR en LayerNormalization {i+1}: {norm_error}")
                        raise
                else:
                    logger.info(f"Última capa: omitiendo LayerNormalization")
            
            logger.info(f"=== CAPAS TEMPORALES COMPLETADAS ===")
            logger.info(f"Shape final: {x.shape}")
            logger.info(f"Total capas construidas: {len(lstm_units)}")

            
            return x
            
        except Exception as e:
            logger.error(f"=== ERROR CONSTRUYENDO CAPAS TEMPORALES ===")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _build_real_temporal_pooling(self, x):
        """Construye el pooling temporal con attention mechanism."""
        try:
            pooling_type = self.config['temporal_pooling']
            logger.info(f"=== CONSTRUYENDO POOLING TEMPORAL ===")
            logger.info(f"Input tensor shape: {x.shape}")
            logger.info(f"Pooling type: {pooling_type}")
            logger.info(f"self.sequence_length: {self.sequence_length}")

            
            if pooling_type == 'attention':
                logger.info("--- CONSTRUYENDO ATTENTION MECHANISM ---")
                
                try:
                    # 1. Crear contexto global
                    logger.info("Paso 1: Global context...")
                    global_context = layers.GlobalAveragePooling1D(name='real_global_context')(x)
                    logger.info(f"Global context shape: {global_context.shape}")
                    
                    # 2. Expandir contexto
                    logger.info("Paso 2: Expandiendo context...")
                    logger.info(f"Usando self.sequence_length = {self.sequence_length}")

                    global_context_expanded = layers.RepeatVector(
                        self.sequence_length, 
                        name='real_context_expanded'
                    )(global_context)
                    logger.info(f"Context expanded shape: {global_context_expanded.shape}")
                    
                    # 3. Concatenar
                    logger.info("Paso 3: Concatenando...")
                    logger.info(f"Shapes para concatenar: x={x.shape}, context_expanded={global_context_expanded.shape}")

                    combined = layers.Concatenate(
                        axis=-1, 
                        name='real_combined_features'
                    )([x, global_context_expanded])
                    logger.info(f"Combined features shape: {combined.shape}")
                    
                    # 4. Attention scores
                    logger.info("Paso 4: Attention scores...")
                    attention_scores = layers.Dense(
                        1, 
                        activation='tanh', 
                        name='real_attention_scores'
                    )(combined)
                    logger.info(f"Attention scores shape: {attention_scores.shape}")
                    logger.info(f"Dense output dtype: {attention_scores.dtype}")
                    
                    # 5. Normalizar con softmax
                    logger.info("Paso 5: Normalizado con softmax...")
                    logger.info(f"ANTES del Softmax - attention_scores shape: {attention_scores.shape}")

                    attention_scores_squeezed = layers.Lambda(
                        lambda x: tf.squeeze(x, axis=-1),
                        name='real_attention_squeeze'
                    )(attention_scores)
                    logger.info(f"Squeezed shape: {attention_scores_squeezed.shape}")
                    
                    attention_weights = layers.Softmax(
                        axis=1, 
                        name='real_attention_weights'
                    )(attention_scores_squeezed)
                    logger.info(f"Attention weights shape: {attention_weights.shape}")
                    
                    # 6. Weighted average
                    logger.info("Paso 6: Weighted average...")
                    attention_weights_expanded = layers.Lambda(
                        lambda x: tf.expand_dims(x, axis=-1),
                        name='real_attention_expand'
                    )(attention_weights)
                    logger.info(f"Weights expanded shape: {attention_weights_expanded.shape}")
                    
                    weighted_output = layers.Lambda(
                        lambda inputs: tf.reduce_sum(inputs[0] * inputs[1], axis=1),
                        name='real_weighted_sum'
                    )([x, attention_weights_expanded])
                    logger.info(f"Weighted output shape: {weighted_output.shape}")
                    
                    x = weighted_output
                    logger.info("✓ Attention mechanism completado")
                    
                except Exception as attention_error:
                    logger.error(f"ERROR EN ATTENTION: {attention_error}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
                    
            elif pooling_type == 'max':
                logger.info("Aplicando GlobalMaxPooling1D...")
                x = layers.GlobalMaxPooling1D(name='real_max_pooling')(x)
                logger.info(f"Max pooling shape: {x.shape}")
                
            elif pooling_type == 'average':
                logger.info("Aplicando GlobalAveragePooling1D...")
                x = layers.GlobalAveragePooling1D(name='real_avg_pooling')(x)
                logger.info(f"Average pooling shape: {x.shape}")
                
            elif pooling_type == 'last':
                logger.info("Tomando último timestep...")
                x = layers.Lambda(
                    lambda inputs: inputs[:, -1, :], 
                    name='real_last_timestep'
                )(x)
                logger.info(f"Last timestep shape: {x.shape}")
                
            else:
                logger.warning(f"Pooling desconocido: {pooling_type}, usando average")
                x = layers.GlobalAveragePooling1D(name='real_default_pooling')(x)
                logger.info(f"Default pooling shape: {x.shape}")
            
            logger.info(f"=== POOLING COMPLETADO ===")
            logger.info(f"Output shape: {x.shape}")
            return x
            
        except Exception as e:
            logger.error(f"=== ERROR EN POOLING ===")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback
            logger.warning("Aplicando pooling de emergencia")
            try:
                x = layers.GlobalAveragePooling1D(name='real_emergency_pooling')(x)
                logger.info(f"Emergency pooling shape: {x.shape}")
                return x
            except Exception as fallback_error:
                logger.error(f"Error en fallback: {fallback_error}")
                raise
    
    def build_real_siamese_model(self) -> Model:
        """Construye el modelo siamés temporal completo."""
        try:
            if self.base_network is None:
                self.build_real_base_network()
            
            logger.info("Construyendo modelo siamés temporal...")
            
            # Inputs para las dos ramas
            input_a = layers.Input(
                shape=(self.sequence_length, self.feature_dim), 
                name='real_input_sequence_a'
            )
            input_b = layers.Input(
                shape=(self.sequence_length, self.feature_dim), 
                name='real_input_sequence_b'
            )
            
            # Procesar con red base (pesos compartidos)
            embedding_a = self.base_network(input_a)
            embedding_b = self.base_network(input_b)
            
            # Calcular distancia
            if self.config['distance_metric'] == 'euclidean':
                distance = layers.Lambda(
                    lambda embeddings: tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)),
                    name='real_euclidean_distance'
                )([embedding_a, embedding_b])
                
            elif self.config['distance_metric'] == 'manhattan':
                distance = layers.Lambda(
                    lambda embeddings: tf.reduce_sum(tf.abs(embeddings[0] - embeddings[1]), axis=1, keepdims=True),
                    name='real_manhattan_distance'
                )([embedding_a, embedding_b])
                
            elif self.config['distance_metric'] == 'cosine':
                distance = layers.Lambda(
                    lambda embeddings: 1 - tf.reduce_sum(embeddings[0] * embeddings[1], axis=1, keepdims=True),
                    name='real_cosine_distance'
                )([embedding_a, embedding_b])
            
            # Crear modelo siamés
            siamese_model = Model(
                inputs=[input_a, input_b], 
                outputs=distance,
                name='real_siamese_dynamic_network'
            )
            
            self.siamese_model = siamese_model
            
            total_params = siamese_model.count_params()
            logger.info(f"Modelo siamés temporal construido: {total_params:,} parámetros")
            logger.info(f"  - Métrica de distancia: {self.config['distance_metric']}")
            logger.info(f"  - Arquitectura: Twin network con pesos compartidos")
            logger.info(f"  - Base network: {self.base_network.count_params():,} parámetros")
            
            return siamese_model
            
        except Exception as e:
            logger.error(f"Error construyendo modelo siamés: {e}")
            raise
    
    def compile_real_model(self):
        """Compila el modelo siamés temporal."""
        try:
            if self.siamese_model is None:
                self.build_real_siamese_model()
            
            logger.info("Compilando modelo siamés temporal...")
            
            optimizer = optimizers.Adam(
                learning_rate=5e-4,
                clipnorm=1.0,
                clipvalue=0.5
            )
            
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
            logger.info(f"  - Funcion de pérdida: {self.config['loss_function']}")
            
        except Exception as e:
            logger.error(f"Error compilando modelo temporal: {e}")
            raise
    
    def _contrastive_loss_real(self, y_true, y_pred):
        """Función de pérdida contrastiva."""
        epsilon = 1e-8
        margin = tf.constant(self.config.get('margin', 1.0), dtype=tf.float32)
        
        distance = tf.sqrt(tf.square(y_pred) + epsilon)
        
        square_pred = tf.square(distance)
        margin_square = tf.square(tf.maximum(margin - distance, 0.0))
        
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    def _far_metric_real(self, y_true, y_pred):
        """Métrica FAR (False Accept Rate) con threshold dinámico."""
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
        """Métrica FRR (False Reject Rate) con threshold dinámico."""
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
    
    def load_real_temporal_data_from_database(self, database) -> bool:
        """
        Carga datos temporales desde la base de datos biométrica.
        Maneja usuarios Bootstrap y Normales.
        """
        try:
            logger.info("=== CARGANDO DATOS TEMPORALES===")
            logger.info("🔄 Buscando templates con datos temporales para red dinámica...")
            
            # Obtener usuarios
            real_users = database.list_users()
            
            if len(real_users) < self.config.get('min_users_for_training', 2):
                logger.error(f"Insuficientes usuarios: {len(real_users)} < {self.config.get('min_users_for_training', 2)}")
                return False
            
            logger.info(f"📊 Usuarios encontrados: {len(real_users)}")
            
            # Limpiar muestras
            self.real_training_samples.clear()
            
            users_with_sufficient_data = 0
            total_samples_loaded = 0
            
            for user in real_users:
                try:
                    logger.info(f"📂 Procesando usuario: {user.username} ({user.user_id})")
                    
                    # Obtener templates
                    user_templates_list = []
                    for template_id, template in database.templates.items():
                        if template.user_id == user.user_id:
                            user_templates_list.append(template)
                    
                    if not user_templates_list:
                        logger.info(f"  ⚠️ Usuario {user.user_id} sin templates, omitiendo")
                        continue
                    
                    logger.info(f"   📊 Templates encontrados: {len(user_templates_list)}")
                    
                    # Filtrar templates con datos temporales
                    temporal_templates = []
                    for template in user_templates_list:
                        template_type_str = str(template.template_type).lower()
                        has_temporal_sequence = (template.metadata.get('temporal_sequence') is not None and 
                                               isinstance(template.metadata.get('temporal_sequence'), list) and
                                               len(template.metadata.get('temporal_sequence', [])) >= 5)
                        has_individual_sequences = (template.metadata.get('individual_temporal_sequences') is not None and
                                                  isinstance(template.metadata.get('individual_temporal_sequences'), list) and
                                                  len(template.metadata.get('individual_temporal_sequences', [])) > 1)
                        
                        if 'dynamic' in template_type_str or has_temporal_sequence or has_individual_sequences:
                            temporal_templates.append(template)
                    
                    logger.info(f"   📊 Templates con datos temporales: {len(temporal_templates)}")
                    
                    # Procesar templates temporales
                    user_temporal_samples = []
                    
                    for template in temporal_templates:
                        try:
                            temporal_sequence = template.metadata.get('temporal_sequence', None)
                            individual_sequences = template.metadata.get('individual_temporal_sequences', [])
                            
                            has_individual_data = individual_sequences and len(individual_sequences) > 1
                            has_temporal_sequence = temporal_sequence and len(temporal_sequence) >= 5
                            
                            if has_temporal_sequence or has_individual_data:
                                logger.info(f"   🔧 Procesando template: {template.gesture_name}")
                                logger.info(f"       Tipo: {template.template_type}")

                                
                                # PROCESAR SECUENCIAS INDIVIDUALES (USUARIOS NORMALES)
                                if has_individual_data:
                                    logger.info(f"       🎯 {len(individual_sequences)} secuencias individuales")
                                elif has_temporal_sequence:
                                    logger.info(f"       Secuencia: {len(temporal_sequence)} frames")
                                    
                                    sequences_loaded = 0
                                    for seq_idx, sequence in enumerate(individual_sequences):
                                        if len(sequence) >= 5:
                                            sequence_array = np.array(sequence, dtype=np.float32)
                                            
                                            if len(sequence_array.shape) == 2 and sequence_array.shape[1] == self.feature_dim:
                                                dynamic_sample = RealDynamicSample(
                                                    user_id=user.user_id,
                                                    sequence_id=f"{template.template_id}_preserved_{seq_idx}",
                                                    temporal_features=sequence_array,
                                                    gesture_sequence=[template.gesture_name] * len(sequence_array),
                                                    transition_types=['hold'] * max(1, len(sequence_array)-1),
                                                    timestamp=getattr(template, 'created_at', time.time()) + (seq_idx * 0.1),
                                                    duration=len(sequence_array) * 0.033,
                                                    quality_score=template.quality_score,
                                                    metadata={
                                                        'data_source': template.metadata.get('data_source', 'enrollment_capture'),
                                                        'bootstrap_mode': False,
                                                        'sequence_length': len(sequence_array),
                                                        'feature_dim': sequence_array.shape[1],
                                                        'user_type': 'Normal_Preserved',
                                                        'original_samples_used': len(individual_sequences),
                                                        'sequence_index': seq_idx,
                                                        'confidence': template.confidence,
                                                        'gesture_name': template.gesture_name,
                                                        'parent_template_id': template.template_id
                                                    }
                                                )
                                                user_temporal_samples.append(dynamic_sample)
                                                sequences_loaded += 1
                                    
                                    genuine_pairs = sequences_loaded * (sequences_loaded - 1) // 2 if sequences_loaded >= 2 else 0
                                    
                                    logger.info(f"       ✅ Secuencias preservadas cargadas: {sequences_loaded}")
                                    logger.info(f"       📊 Pares genuinos generados: {genuine_pairs}")
                                    logger.info(f"       🎯 Usuario incluido en entrenamiento dinámico")
                                
                                # PROCESAR SECUENCIA TEMPORAL (BOOTSTRAP)
                                elif has_temporal_sequence:
                                    temporal_array = np.array(temporal_sequence, dtype=np.float32)
                                    
                                    if len(temporal_array.shape) == 2 and temporal_array.shape[1] == self.feature_dim:
                                        samples_used = template.metadata.get('samples_used', 1)
                                        
                                        dynamic_sample = RealDynamicSample(
                                            user_id=user.user_id,
                                            sequence_id=template.template_id,
                                            temporal_features=temporal_array,
                                            gesture_sequence=[template.gesture_name] * len(temporal_sequence),
                                            transition_types=['hold'] * max(1, len(temporal_sequence)-1),
                                            timestamp=getattr(template, 'created_at', time.time()),
                                            duration=len(temporal_sequence) * 0.033,
                                            quality_score=template.quality_score,
                                            metadata={
                                                'data_source': template.metadata.get('data_source', 'enrollment_capture'),
                                                'bootstrap_mode': template.metadata.get('bootstrap_mode', True),
                                                'sequence_length': len(temporal_sequence),
                                                'feature_dim': temporal_array.shape[1],
                                                'user_type': 'Bootstrap',
                                                'confidence': template.confidence,
                                                'gesture_name': template.gesture_name
                                            }
                                        )
                                        user_temporal_samples.append(dynamic_sample)
                                        logger.info(f"       ✅ Bootstrap: {len(temporal_sequence)} frames")
                                    else:
                                        logger.warning(f"   ⚠️ Dimensiones incorrectas: {temporal_array.shape} (esperado: [N, {self.feature_dim}])")
                            else:
                                logger.warning(f"   ⚠️ Template sin datos temporales válidos")
                        
                        except Exception as e:
                            logger.error(f"   ❌ Error procesando template: {e}")
                            continue
                    
                    # Validar usuario
                    min_temporal_samples = 1
                    
                    if len(user_temporal_samples) >= min_temporal_samples:
                        users_with_sufficient_data += 1
                        total_samples_loaded += len(user_temporal_samples)
                        self.real_training_samples.extend(user_temporal_samples)
                        
                        gesture_counts = {}
                        for sample in user_temporal_samples:
                            gesture_name = sample.metadata.get('gesture_name', 'Unknown')
                            gesture_counts[gesture_name] = gesture_counts.get(gesture_name, 0) + 1
                        
                        logger.info(f"✅ Usuario temporal válido: {user.username}")
                        logger.info(f"   📊 Secuencias temporales cargadas: {len(user_temporal_samples)}")
                        logger.info(f"   🎯 Gestos únicos: {len(gesture_counts)}")
                        for gesture, count in gesture_counts.items():
                            logger.info(f"      • {gesture}: {count} secuencias temporales")
                    else:
                        logger.warning(f"   ⚠️ Usuario {user.user_id} con pocas secuencias temporales: {len(user_temporal_samples)} < {min_temporal_samples}")
                    
                except Exception as e:
                    logger.error(f"Error procesando usuario {user.user_id}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            # Validación final
            min_users_required = 2
            min_total_samples = 10
            
            if users_with_sufficient_data < min_users_required:
                logger.error("=" * 60)
                logger.error("❌ USUARIOS INSUFICIENTES")
                logger.error("=" * 60)
                logger.error(f"Válidos: {users_with_sufficient_data} < {min_users_required}")
                return False
            
            if total_samples_loaded < min_total_samples:
                logger.error("=" * 60)
                logger.error("❌ MUESTRAS INSUFICIENTES")
                logger.error("=" * 60)
                logger.error(f"Muestras cargadas: {total_samples_loaded} < {min_total_samples}")
                return False
            
            # División train/validation
            try:
                user_ids = [sample.user_id for sample in self.real_training_samples]
                
                train_samples, val_samples = train_test_split(
                    self.real_training_samples,
                    test_size=0.2,
                    random_state=42,
                    stratify=user_ids
                )
                
                self.real_training_samples = train_samples
                self.real_validation_samples = val_samples
                
                logger.info(f"División estratificada: Train {len(train_samples)}, Val {len(val_samples)}")
                
            except Exception as e:
                logger.warning(f"División simple aplicada: {e}")
                split_idx = int(0.8 * len(self.real_training_samples))
                self.real_validation_samples = self.real_training_samples[split_idx:]
                self.real_training_samples = self.real_training_samples[:split_idx]
            
            logger.info("=" * 60)
            logger.info("✅ DATOS TEMPORALES REALES CARGADOS EXITOSAMENTE")
            logger.info("=" * 60)
            logger.info(f"👥 Usuarios con datos temporales suficientes: {users_with_sufficient_data}")
            logger.info(f"🧬 Total secuencias temporales REALES cargadas: {total_samples_loaded}")
            logger.info(f"📊 Promedio secuencias por usuario: {total_samples_loaded/users_with_sufficient_data:.1f}")
            logger.info(f"📐 Dimensiones por frame: {self.feature_dim}")
            
            # Actualizar contador
            all_samples = self.real_training_samples + self.real_validation_samples
            user_stats = {}
            for sample in all_samples:
                user_stats[sample.user_id] = user_stats.get(sample.user_id, 0) + 1
            
            logger.info(f"📈 DISTRIBUCIÓN POR USUARIO:")
            for user_id, count in user_stats.items():
                user_name = next((u.username for u in real_users if u.user_id == user_id), user_id)
                log_info(f"   • {user_name} ({user_id}): {count} secuencias")
                
            self.users_trained_count = len(user_stats)
            
            # Reporte final
            logger.info("=" * 60)
            logger.info("✅ DATOS TEMPORALES CARGADOS")
            logger.info("=" * 60)
            logger.info(f"👥 Usuarios: {users_with_sufficient_data}")
            logger.info(f"🧬 Total secuencias: {total_samples_loaded}")
            logger.info(f"📊 Promedio/usuario: {total_samples_loaded/users_with_sufficient_data:.1f}")
            logger.info(f"📐 Dimensiones: {self.feature_dim}")
            logger.info(f"📊 Usuarios registrados: {self.users_trained_count}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error("❌ ERROR CARGANDO DATOS TEMPORALES")
            logger.error("=" * 60)
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error("=" * 60)
            return False
        
    def validate_real_temporal_data_quality(self) -> bool:
        """Valida la calidad de los datos temporales."""
        try:
            logger.info("Validando calidad de datos temporales...")
            
            if len(self.real_training_samples) == 0:
                logger.error("No hay muestras de entrenamiento")
                return False
            
            # Validar calidad mínima
            high_quality_samples = [
                s for s in self.real_training_samples 
                if getattr(s, 'quality_score', 100.0) >= 80.0
            ]
            
            quality_ratio = len(high_quality_samples) / len(self.real_training_samples)
            logger.info(f"Alta calidad: {len(high_quality_samples)}/{len(self.real_training_samples)} ({quality_ratio:.1%})")
            
            if quality_ratio < 0.7:
                logger.warning("Baja proporción de alta calidad")
            
            # Validar dimensiones
            for i, sample in enumerate(self.real_training_samples[:10]):
                if sample.temporal_features.shape[1] != self.feature_dim:
                    logger.error(f"Dimensión incorrecta en muestra {i}: esperado {self.feature_dim}, obtenido {sample.temporal_features.shape[1]}")
                    return False
            
            # Validar longitudes
            sequence_lengths = [sample.temporal_features.shape[0] for sample in self.real_training_samples]
            min_length = min(sequence_lengths)
            max_length = max(sequence_lengths)
            avg_length = sum(sequence_lengths) / len(sequence_lengths)
            
            logger.info(f"Longitudes de secuencia - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.1f}")
            
            if min_length < 5:
                logger.error(f"Secuencia muy corta: {min_length} frames < 5 mínimo")
                return False
            
            # Validar usuarios
            unique_users = set(sample.user_id for sample in self.real_training_samples)
            min_users_required = self.config.get('min_users_for_training', 2)
            
            if len(unique_users) < min_users_required:
                logger.error(f"Insuficientes usuarios: {len(unique_users)} < {min_users_required}")
                return False
            
            logger.info(f"Usuarios únicos: {len(unique_users)}")
            logger.info("✓ Validación de calidad completada")
            return True
            
        except Exception as e:
            logger.error(f"Error validando calidad: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def create_real_temporal_pairs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crea pares de secuencias temporales."""
        try:
            logger.info("Creando pares temporales...")
            
            pairs_a = []
            pairs_b = []
            labels = []
            
            samples = self.real_training_samples
            n_samples = len(samples)
            
            # Crear pares genuinos
            user_samples = {}
            for sample in samples:
                if sample.user_id not in user_samples:
                    user_samples[sample.user_id] = []
                user_samples[sample.user_id].append(sample)
            
            genuine_pairs = 0
            for user_id, user_sample_list in user_samples.items():
                if len(user_sample_list) >= 2:
                    for i in range(len(user_sample_list)):
                        for j in range(i + 1, len(user_sample_list)):
                            sample1 = user_sample_list[i]
                            sample2 = user_sample_list[j]
                            
                            seq1 = self._pad_or_truncate_sequence(sample1.temporal_features)
                            seq2 = self._pad_or_truncate_sequence(sample2.temporal_features)
                            
                            pairs_a.append(seq1)
                            pairs_b.append(seq2)
                            labels.append(1)
                            genuine_pairs += 1
            
            logger.info(f"Pares genuinos creados: {genuine_pairs}")
            
            # Crear pares impostores
            impostor_pairs = 0
            target_impostor_pairs = genuine_pairs #Balancear clases
            
            users_list = list(user_samples.keys())
            while impostor_pairs < target_impostor_pairs:
                user1_idx = np.random.randint(0, len(users_list))
                user2_idx = np.random.randint(0, len(users_list))
                
                if user1_idx != user2_idx:
                    user1_id = users_list[user1_idx]
                    user2_id = users_list[user2_idx]
                    
                    sample1 = np.random.choice(user_samples[user1_id])
                    sample2 = np.random.choice(user_samples[user2_id])
                    
                    seq1 = self._pad_or_truncate_sequence(sample1.temporal_features)
                    seq2 = self._pad_or_truncate_sequence(sample2.temporal_features)
                    
                    pairs_a.append(seq1)
                    pairs_b.append(seq2)
                    labels.append(0)
                    impostor_pairs += 1
            
            logger.info(f"Pares impostores creados: {impostor_pairs}")
            
            # Convertir a arrays
            pairs_a = np.array(pairs_a, dtype=np.float32)
            pairs_b = np.array(pairs_b, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            
            # Mezclar
            indices = np.random.permutation(len(labels))
            pairs_a = pairs_a[indices]
            pairs_b = pairs_b[indices]
            labels = labels[indices]
            
            logger.info(f"Pares temporales creados: {len(labels)}")
            logger.info(f"  - Genuinos: {np.sum(labels)} ({np.mean(labels):.1%})")
            logger.info(f"  - Impostores: {np.sum(1-labels)} ({1-np.mean(labels):.1%})")
            logger.info(f"  - Shape: {pairs_a.shape}, {pairs_b.shape}")
            
            return pairs_a, pairs_b, labels
            
        except Exception as e:
            logger.error(f"Error creando pares: {e}")
            raise
    
    def _pad_or_truncate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Ajusta la secuencia a la longitud fija requerida."""
        current_length = sequence.shape[0]
        
        if current_length >= self.sequence_length:
            return sequence[:self.sequence_length]
        else:
            padding = np.zeros((self.sequence_length - current_length, self.feature_dim))
            return np.vstack([sequence, padding])
        
    def train_with_real_data(self, database, validation_split: float = 0.2) -> RealTemporalTrainingHistory:
        """Entrena el modelo con datos temporales."""
        try:
            start_time = time.time()
            logger.info("=== INICIANDO ENTRENAMIENTO TEMPORA ===")
            
            # 1. Cargar datos
            if not self.load_real_temporal_data_from_database(database):
                raise ValueError("No se pudieron cargar datos temporales suficientes")
            
            # 2. Validar calidad
            if not self.validate_real_temporal_data_quality():
                raise ValueError("Datos temporales no cumplen criterios de calidad")
            
            # 3. Compilar modelo
            if not self.is_compiled:
                self.compile_real_model()
            
            # 4. Crear pares
            pairs_a, pairs_b, labels = self.create_real_temporal_pairs()
            
            # 5. Callbacks
            callbacks_list = self._setup_real_training_callbacks()
            
            # 6. Entrenar
            logger.info("Iniciando entrenamiento temporal...")
            history = self.siamese_model.fit(
                [pairs_a, pairs_b],
                labels,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_split=validation_split,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # 7. Actualizar estado
            self.is_trained = True
            self.training_history.loss = history.history['loss']
            self.training_history.val_loss = history.history['val_loss']
            
            # 8. Evaluar
            metrics = self.evaluate_real_model(pairs_a, pairs_b, labels)
            self.current_metrics = metrics
            
            # 9. Guardar modelo entrenado
            self.save_real_model()
            
            total_time = time.time() - start_time
            self.training_history.total_training_time = total_time
            
            # 6. Logs pre-entrenamiento
            logger.info(f"=== CONFIGURACIÓN PRE-ENTRENAMIENTO ===")
            logger.info(f"Learning rate: {self.siamese_model.optimizer.learning_rate.numpy()}")
            logger.info(f"Margen contrastive loss: {self.config.get('margin', 'NO DEFINIDO')}")
            logger.info(f"Batch size: {self.config['batch_size']}")
            logger.info(f"Total parámetros: {self.siamese_model.count_params()}")
            logger.info(f"Pares entrenamiento: {len(labels)}")
            logger.info(f"Shape: {pairs_a.shape}, {pairs_b.shape}")
            
            logger.info("Iniciando entrenamiento temporal REAL...")
            logger.info(f"✓ Entrenamiento temporal REAL completado en {total_time:.1f}s")
            logger.info(f"  - Épocas entrenadas: {len(history.history['loss'])}")
            logger.info(f"  - Mejor pérdida: {min(history.history['val_loss']):.4f}")
            logger.info(f"  - EER final: {metrics.eer:.3f}")
            logger.info(f"  - AUC final: {metrics.auc_score:.3f}")
            
            # Marcar como entrenado
            self.is_trained = True
            logger.info("✓ Red dinámica marcada como entrenada")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            raise
    
    def _setup_real_training_callbacks(self) -> List[callbacks.Callback]:
        """Configura callbacks para entrenamiento REAL."""
        callbacks_list = []
        
        # LOG: Configuración inicial
        logger.info(f"=== CONFIGURANDO CALLBACKS DE ENTRENAMIENTO ===")
        logger.info(f"Early stopping patience: {self.config['early_stopping_patience']}")
        logger.info(f"Reduce LR patience: {self.config['reduce_lr_patience']}")
        logger.info(f"Min LR: {self.config['min_lr']}")
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        logger.info(f"✓ Early stopping configurado: patience={self.config['early_stopping_patience']}")
        
        # Reduce learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['reduce_lr_patience'],
            min_lr=self.config['min_lr'],
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        logger.info(f"✓ ReduceLROnPlateau configurado: patience={self.config['reduce_lr_patience']}, factor=0.5")
        
        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(self.model_save_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks_list.append(checkpoint)
        logger.info(f"✓ ModelCheckpoint configurado: {self.model_save_path}")
        
        # CALLBACK CRÍTICO: Monitor de Learning Rate y métricas
        class LRAndMetricsMonitor(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                
                current_lr = float(self.model.optimizer.learning_rate)
                train_loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                
                logger.info(f"MONITOR ÉPOCA {epoch+1}:")
                logger.info(f"  ┣━ Learning Rate: {current_lr:.2e}")
                logger.info(f"  ┣━ Train Loss: {train_loss:.6f}")
                logger.info(f"  ┗━ Val Loss: {val_loss:.6f}")
                
                # ALERTA si hay cambio súbito
                if hasattr(self, 'prev_val_loss') and self.prev_val_loss is not None:
                    loss_change = abs(val_loss - self.prev_val_loss)
                    if loss_change > 0.1:
                        logger.info(f"  🚨 ALERTA: Cambio súbito val_loss = {loss_change:.6f}")
                
                self.prev_val_loss = val_loss
                
            def on_train_begin(self, logs=None):
                initial_lr = float(self.model.optimizer.learning_rate)
                logger.info(f"🚀 INICIO ENTRENAMIENTO: LR inicial = {initial_lr:.2e}")
                self.prev_val_loss = None
        
        callbacks_list.append(LRAndMetricsMonitor())
        logger.info(f"✓ Monitor de LR configurado")
        
        # Monitor de gradientes para detectar explosión
        class GradientMonitor(callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                # Solo monitorear después de época 15 (cerca del colapso)
                if epoch >= 15:
                    logger.info(f"🔍 MONITORING GRADIENTES - Época {epoch+1}")
            
            def on_epoch_end(self, epoch, logs=None):
                if epoch >= 15 and logs is not None:  # Solo épocas críticas
                    try:
                        # Obtener pesos del modelo
                        weights = self.model.get_weights()
                        weight_norms = [np.linalg.norm(w) for w in weights if w.size > 0]
                        max_weight_norm = max(weight_norms) if weight_norms else 0.0
                        
                        current_lr = float(self.model.optimizer.learning_rate)
                        train_loss = logs.get('loss', 0)
                        
                        logger.info(f"GRADIENTES ÉPOCA {epoch+1}:")
                        logger.info(f"  ┣━ Max Weight Norm: {max_weight_norm:.6f}")
                        logger.info(f"  ┣━ LR × Loss: {current_lr * train_loss:.8f}")
                        logger.info(f"  ┗━ Estabilidad: {'🟢 OK' if max_weight_norm < 50.0 else '🔴 INESTABLE'}")
                        
                        if max_weight_norm > 100.0:
                            logger.info(f"🚨 PESOS EXPLOSIVOS DETECTADOS: {max_weight_norm:.6f}")
                            
                    except Exception as e:
                        logger.info(f"Error monitoreando gradientes: {e}")
        
        callbacks_list.append(GradientMonitor())
        logger.info(f"✓ Monitor de gradientes configurado")
        
        # NUEVO: Monitor de estabilidad de gradientes (solo logging)
        class GradientStabilityMonitor(callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.prev_loss = None
                
            def on_batch_end(self, batch, logs=None):
                if logs and self.prev_loss is not None:
                    current_loss = logs.get('loss', 0)
                    if current_loss > self.prev_loss * 3.0:  # Spike 3x o más
                        logger.warning(f"Batch {batch}: Loss spike detected - Gradient clipping should handle this")
                        logger.info(f"Loss: {self.prev_loss:.6f} → {current_loss:.6f}")
                        logger.info(f"Ratio: {current_loss/self.prev_loss:.2f}x")
                self.prev_loss = logs.get('loss', 0) if logs else 0
        
        callbacks_list.append(GradientStabilityMonitor())
        logger.info(f"✓ Monitor de estabilidad de gradientes configurado")
        
        # Monitor detallado de colapso por batch
        class DetailedCollapseMonitor(callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.epoch = 0
                self.prev_batch_loss = None
                self.stable_batches = 0
                self.total_batches = 0
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch
                self.prev_batch_loss = None
                self.batch_losses = []
                if epoch >= 20:
                    logger.info(f"🔍 MONITOREANDO COLAPSO ÉPOCA {epoch+1} - Análisis por batch")
            
            def on_batch_end(self, batch, logs=None):
                self.total_batches += 1
                
                if self.epoch >= 20:  # Solo monitorear épocas críticas
                    if logs is None:
                        logs = {}
                        
                    current_loss = logs.get('loss', 0)
                    current_lr = float(self.model.optimizer.learning_rate)
                    self.batch_losses.append(current_loss)
                    
                    # Log cada 10 batches o si hay cambio súbito
                    if batch % 10 == 0 or (self.prev_batch_loss and current_loss > self.prev_batch_loss * 3.0):
                        logger.info(f"Época {self.epoch+1}, Batch {batch}: loss={current_loss:.6f}, lr={current_lr:.2e}")
                    
                    # Detectar salto súbito entre batches
                    if self.prev_batch_loss is not None:
                        loss_ratio = current_loss / self.prev_batch_loss if self.prev_batch_loss > 0 else 1.0
                        
                        if loss_ratio > 5.0:  # Loss se multiplica por 5+
                            logger.error(f"🚨 COLAPSO SÚBITO DETECTADO:")
                            logger.error(f"   Época {self.epoch+1}, Batch {batch}")
                            logger.error(f"   Loss salto: {self.prev_batch_loss:.6f} → {current_loss:.6f}")
                            logger.error(f"   Ratio: {loss_ratio:.2f}x")
                            logger.error(f"   Learning Rate: {current_lr:.2e}")
                            
                            # Información del contexto
                            if len(self.batch_losses) > 5:
                                recent_losses = self.batch_losses[-5:]
                                logger.error(f"   Últimos 5 losses: {[f'{l:.6f}' for l in recent_losses]}")
                    
                    self.prev_batch_loss = current_loss
            
            def on_epoch_end(self, epoch, logs=None):
                if epoch >= 20 and self.batch_losses:
                    # Estadísticas de la época
                    min_loss = min(self.batch_losses)
                    max_loss = max(self.batch_losses)
                    avg_loss = sum(self.batch_losses) / len(self.batch_losses)
                    
                    logger.info(f"ESTADÍSTICAS ÉPOCA {epoch+1}:")
                    logger.info(f"  ┣━ Loss mínimo: {min_loss:.6f}")
                    logger.info(f"  ┣━ Loss máximo: {max_loss:.6f}")
                    logger.info(f"  ┣━ Loss promedio: {avg_loss:.6f}")
                    logger.info(f"  ┗━ Variabilidad: {max_loss/min_loss:.2f}x")
        
        callbacks_list.append(DetailedCollapseMonitor())
        logger.info(f"✓ Monitor de colapso por batch configurado")
        
        # Monitor de métricas de validación por epoch
        class ValidationMonitor(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch >= 20 and logs is not None:
                    val_far = logs.get('val__far_metric_real', 0)
                    val_frr = logs.get('val__frr_metric_real', 0)
                    train_far = logs.get('_far_metric_real', 0)
                    train_frr = logs.get('_frr_metric_real', 0)
                    
                    logger.info(f"MÉTRICAS DETALLADAS ÉPOCA {epoch+1}:")
                    logger.info(f"  ┣━ Train FAR: {train_far:.6f}, FRR: {train_frr:.6f}")
                    logger.info(f"  ┗━ Val FAR: {val_far:.6f}, FRR: {val_frr:.6f}")
        
        callbacks_list.append(ValidationMonitor())
        logger.info(f"✓ Monitor de métricas de validación configurado")
        
        # Monitor anti-NaN
        class NaNStoppingCallback(callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                if logs:
                    import tensorflow as tf
                    current_loss = logs.get('loss', 0)
                    if tf.math.is_nan(current_loss) or tf.math.is_inf(current_loss):
                        logger.error(f"NaN/Inf detectado en batch {batch} - Deteniendo entrenamiento")
                        self.model.stop_training = True
            
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    import tensorflow as tf
                    val_loss = logs.get('val_loss', 0)
                    if tf.math.is_nan(val_loss) or tf.math.is_inf(val_loss):
                        logger.error(f"NaN/Inf en validación época {epoch} - Deteniendo entrenamiento")
                        self.model.stop_training = True
        
        callbacks_list.append(NaNStoppingCallback())
        logger.info(f"✓ Monitor anti-NaN configurado")
        
        logger.info(f"=== TOTAL CALLBACKS: {len(callbacks_list)} configurados ===")
        
        return callbacks_list


    def evaluate_real_model(self, sequences_a: np.ndarray, sequences_b: np.ndarray, 
                           labels: np.ndarray) -> RealTemporalMetrics:
        """Evalúa el modelo temporal con métricas específicas."""
        try:
            logger.info("Evaluando modelo temporal...")
            
            # Predicciones
            distances = self.siamese_model.predict([sequences_a, sequences_b])
            distances = distances.flatten()
            
            # Convertir a similitudes
            similarities = 1.0 / (1.0 + distances)
            
            # Calcular métricas a diferentes umbrales
            thresholds = np.linspace(0, 1, 1000)
            fars = []
            frrs = []
            
            for threshold in thresholds:
                predictions = (similarities >= threshold).astype(int)
                
                genuine_mask = (labels == 1)
                impostor_mask = (labels == 0)
                
                # FAR: falsos aceptados / total impostores
                false_accepts = np.sum((predictions == 1) & impostor_mask)
                total_impostors = np.sum(impostor_mask)
                far = false_accepts / total_impostors if total_impostors > 0 else 0
                
                # FRR: falsos rechazados / total genuinos
                false_rejects = np.sum((predictions == 0) & genuine_mask)
                total_genuines = np.sum(genuine_mask)
                frr = false_rejects / total_genuines if total_genuines > 0 else 0
                
                fars.append(far)
                frrs.append(frr)
            
            # Encontrar EER
            fars = np.array(fars)
            frrs = np.array(frrs)
            eer_idx = np.argmin(np.abs(fars - frrs))
            eer = (fars[eer_idx] + frrs[eer_idx]) / 2
            optimal_threshold = thresholds[eer_idx]
            
            # Otras métricas
            optimal_predictions = (similarities >= optimal_threshold).astype(int)
            accuracy = accuracy_score(labels, optimal_predictions)
            auc_score_val = roc_auc_score(labels, similarities)
            
            # Precision, recall, F1
            precision, recall, _ = precision_recall_curve(labels, similarities)
            f1_score = 2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]) if (precision[-1] + recall[-1]) > 0 else 0
            
            # Métricas temporales específicas
            sequence_correlation = self._calculate_sequence_correlation_real(sequences_a, sequences_b, labels)
            temporal_consistency = self._calculate_temporal_consistency_real(similarities, labels)
            rhythm_similarity = self._calculate_rhythm_similarity_real(sequences_a, sequences_b, labels)
            
            # Crear objeto de métricas
            metrics = RealTemporalMetrics(
                far=fars[eer_idx],
                frr=frrs[eer_idx],
                eer=eer,
                auc_score=auc_score_val,
                accuracy=accuracy,
                threshold=optimal_threshold,
                precision=precision[-1],
                recall=recall[-1],
                f1_score=f1_score,
                sequence_correlation=sequence_correlation,
                temporal_consistency=temporal_consistency,
                rhythm_similarity=rhythm_similarity,
                validation_samples=len(labels)
            )
            
            # Actualizar threshold
            self.optimal_threshold = optimal_threshold
            
            logger.info("✓ Evaluación temporal completada:")
            logger.info(f"  - EER: {eer:.3f}")
            logger.info(f"  - AUC: {auc_score_val:.3f}")
            logger.info(f"  - Accuracy: {accuracy:.3f}")
            logger.info(f"  - Threshold óptimo: {optimal_threshold:.3f}")
            logger.info(f"  - Correlación secuencial: {sequence_correlation:.3f}")
            logger.info(f"  - Consistencia temporal: {temporal_consistency:.3f}")
            logger.info(f"  - Genuinos evaluados: {int(np.sum(labels))}")
            logger.info(f"  - Impostores evaluados: {int(np.sum(1 - labels))}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluando modelo: {e}")
            raise
    
    def _calculate_sequence_correlation_real(self, sequences_a: np.ndarray, 
                                           sequences_b: np.ndarray, labels: np.ndarray) -> float:
        """Calcula correlación promedio entre secuencias genuinas."""
        try:
            genuine_mask = (labels == 1)
            if np.sum(genuine_mask) == 0:
                return 0.0
            
            genuine_a = sequences_a[genuine_mask]
            genuine_b = sequences_b[genuine_mask]
            
            correlations = []
            for seq_a, seq_b in zip(genuine_a, genuine_b):
                flat_a = seq_a.flatten()
                flat_b = seq_b.flatten()
                corr = np.corrcoef(flat_a, flat_b)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_temporal_consistency_real(self, similarities: np.ndarray, labels: np.ndarray) -> float:
        """Calcula consistencia temporal en predicciones."""
        try:
            genuine_similarities = similarities[labels == 1]
            impostor_similarities = similarities[labels == 0]
            
            if len(genuine_similarities) == 0 or len(impostor_similarities) == 0:
                return 0.0
            
            genuine_mean = np.mean(genuine_similarities)
            impostor_mean = np.mean(impostor_similarities)
            separation = abs(genuine_mean - impostor_mean)
            
            return min(separation, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_rhythm_similarity_real(self, sequences_a: np.ndarray, 
                                        sequences_b: np.ndarray, labels: np.ndarray) -> float:
        """Calcula similitud en patrones de ritmo temporal."""
        try:
            genuine_mask = (labels == 1)
            if np.sum(genuine_mask) == 0:
                return 0.0
            
            genuine_a = sequences_a[genuine_mask]
            genuine_b = sequences_b[genuine_mask]
            
            rhythm_similarities = []
            for seq_a, seq_b in zip(genuine_a, genuine_b):
                rhythm_a = np.std(seq_a, axis=1)
                rhythm_b = np.std(seq_b, axis=1)
                
                rhythm_sim = np.corrcoef(rhythm_a, rhythm_b)[0, 1]
                if not np.isnan(rhythm_sim):
                    rhythm_similarities.append(rhythm_sim)
            
            return np.mean(rhythm_similarities) if rhythm_similarities else 0.0
            
        except Exception:
            return 0.0
    
    def predict_temporal_similarity_real(self, sequence1: np.ndarray, sequence2: np.ndarray) -> float:
        """Predice similitud temporal entre dos secuencias."""
        try:
            if not self.is_trained:
                logger.error("Modelo temporal no está entrenado")
                raise ValueError("Modelo no entrenado")
            
            if self.siamese_model is None:
                logger.error("Modelo siamés temporal no inicializado")
                raise ValueError("Modelo no inicializado")
            
            # Validar dimensiones
            if sequence1.shape[1] != self.feature_dim or sequence2.shape[1] != self.feature_dim:
                logger.error(f"Dimensiones incorrectas: esperado (*, {self.feature_dim}), "
                         f"recibido {sequence1.shape}, {sequence2.shape}")
                raise ValueError("Dimensiones incorrectas")
            
            # Ajustar secuencias
            seq1_padded = self._pad_or_truncate_sequence(sequence1)
            seq2_padded = self._pad_or_truncate_sequence(sequence2)
            
            # Preparar datos
            seq1_batch = np.array([seq1_padded], dtype=np.float32)
            seq2_batch = np.array([seq2_padded], dtype=np.float32)
            
            # Predecir distancia
            distance = self.siamese_model.predict([seq1_batch, seq2_batch])[0][0]
            
            # Convertir a similitud
            similarity = 1.0 / (1.0 + distance)
            similarity = np.clip(similarity, 0.0, 1.0)
            
            logger.info(f"Predicción temporal: distancia={distance:.4f}, similitud={similarity:.4f}")
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
    
    def save_real_model(self) -> bool:
        """Guarda el modelo temporal entrenado."""
        try:
            if not self.is_trained or self.siamese_model is None:
                logger.warning("Modelo no entrenado")
                return False
            
            # Guardar modelo
            self.siamese_model.save(str(self.model_save_path))
            
            # Guardar metadatos
            metadata = {
                'embedding_dim': self.embedding_dim,
                'sequence_length': self.sequence_length,
                'feature_dim': self.feature_dim,
                'config': self.config,
                'optimal_threshold': self.optimal_threshold,
                'is_trained': self.is_trained,
                'training_samples': len(self.real_training_samples),
                'users_trained_count': self.users_trained_count,
                'save_timestamp': datetime.now().isoformat(),
                'version': '2.0'
            }
            
            if self.current_metrics:
                metadata['metrics'] = {
                    'eer': self.current_metrics.eer,
                    'auc_score': self.current_metrics.auc_score,
                    'accuracy': self.current_metrics.accuracy,
                    'far': self.current_metrics.far,
                    'frr': self.current_metrics.frr
                }
            
            metadata_path = self.model_save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Modelo guardado: {self.model_save_path}")
            logger.info(f"✓ Metadatos: {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            return False
    
    def load_real_model(self) -> bool:
        """Carga un modelo temporal REAL pre-entrenado."""
        try:
            if not self.model_save_path.exists():
                logger.warning(f"Archivo de modelo no encontrado: {self.model_save_path}")
                return False
            
            # Cargar modelo de Keras
            self.siamese_model = keras.models.load_model(
                str(self.model_save_path),
                custom_objects={
                    'contrastive_loss_real': self._contrastive_loss_real,
                    'far_metric_real': self._far_metric_real,
                    'frr_metric_real': self._frr_metric_real
                }
            )
            
            # Cargar metadatos
            metadata_path = self.model_save_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.optimal_threshold = metadata.get('optimal_threshold', 0.5)
                self.is_trained = metadata.get('is_trained', True)
                
                logger.info(f"✓ Modelo temporal REAL cargado: {self.model_save_path}")
                logger.info(f"  - Umbral óptimo: {self.optimal_threshold}")
                logger.info(f"  - Muestras de entrenamiento: {metadata.get('training_samples', 'N/A')}")
                logger.info(f"  - Versión: {metadata.get('version', 'N/A')}")
            
            self.is_compiled = True
            
            return True
            
        except Exception as e:
            logger.error("Error cargando modelo temporal REAL", e)
            return False

        
    def get_real_model_summary(self) -> Dict[str, Any]:
        """Obtiene resumen completo del modelo temporal REAL."""
        try:
            total_params = self.siamese_model.count_params() if self.siamese_model else 0
            base_params = self.base_network.count_params() if self.base_network else 0
            
            summary = {
                "architecture": {
                    "model_type": "Real Siamese Dynamic Network",
                    "embedding_dim": self.embedding_dim,
                    "sequence_length": self.sequence_length,
                    "feature_dim": self.feature_dim,
                    "total_parameters": total_params,
                    "base_network_parameters": base_params,
                    "lstm_units": self.config['lstm_units'],
                    "sequence_processing": self.config['sequence_processing'],
                    "temporal_pooling": self.config['temporal_pooling'],
                    "distance_metric": self.config['distance_metric']
                },
                "training": {
                    "is_trained": self.is_trained,
                    "training_samples": len(self.real_training_samples),
                    "validation_samples": len(self.real_validation_samples),
                    "users_trained": self.users_trained_count,
                    "optimal_threshold": self.optimal_threshold,
                    "training_time": self.training_history.total_training_time
                },
                "config": self.config,
                "status": {
                    "ready_for_inference": self.is_trained and self.is_compiled,
                    "model_saved": self.model_save_path.exists(),
                    "version": "2.0"
                }
            }
            
            if self.current_metrics:
                summary["performance"] = {
                    "eer": self.current_metrics.eer,
                    "auc_score": self.current_metrics.auc_score,
                    "accuracy": self.current_metrics.accuracy,
                    "far": self.current_metrics.far,
                    "frr": self.current_metrics.frr,
                    "optimal_threshold": self.current_metrics.threshold,
                    "sequence_correlation": self.current_metrics.sequence_correlation,
                    "temporal_consistency": self.current_metrics.temporal_consistency,
                    "rhythm_similarity": self.current_metrics.rhythm_similarity
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen: {e}")
            return {}


# ===== INSTANCIA GLOBAL =====
_real_siamese_dynamic_instance = None

def get_real_siamese_dynamic_network(embedding_dim: int = 128, 
                                   sequence_length: int = 50,
                                   feature_dim: int = 320) -> RealSiameseDynamicNetwork:
    """
    Obtiene instancia global de la red siamesa dinámica REAL.
    Verifica si hay modelo entrenado guardado y lo carga automáticamente.
    """
    global _real_siamese_dynamic_instance
    
    if _real_siamese_dynamic_instance is None:
        _real_siamese_dynamic_instance = RealSiameseDynamicNetwork(embedding_dim, sequence_length, feature_dim)
    
    # Verificar modelo guardado
    if not _real_siamese_dynamic_instance.is_trained:
        try:
            models_dir = Path('biometric_data/models')
            model_path = models_dir / 'dynamic_model.h5'
            
            if model_path.exists():
                logger.info(f"Cargando modelo dinámico: {model_path}")
                try:
                    if _real_siamese_dynamic_instance.siamese_model is None:
                        _real_siamese_dynamic_instance.build_real_base_network()
                        _real_siamese_dynamic_instance.build_real_siamese_model()
                        _real_siamese_dynamic_instance.compile_real_model()
                    
                    _real_siamese_dynamic_instance.siamese_model.load_weights(str(model_path))
                    _real_siamese_dynamic_instance.is_trained = True
                    
                    logger.info(f"✅ Red dinámica cargada: {model_path}")
                    logger.info(f"✅ Estado: is_trained = {_real_siamese_dynamic_instance.is_trained}")
                    
                except Exception as load_error:
                    logger.warning(f"Error cargando modelo: {load_error}")
            else:
                logger.info(f"No se encontró modelo guardado: {model_path}")
        
        except Exception as e:
            logger.warning(f"Error verificando modelo: {e}")
    
    return _real_siamese_dynamic_instance


# Alias para compatibilidad
SiameseDynamicNetwork = RealSiameseDynamicNetwork
get_siamese_dynamic_network = get_real_siamese_dynamic_network