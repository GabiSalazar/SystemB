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
        return None
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
    """Muestra de secuencia dinámica temporal REAL de usuario."""
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
    """Par de secuencias temporales REALES para entrenamiento."""
    sample1: RealDynamicSample
    sample2: RealDynamicSample
    is_genuine: bool
    temporal_distance: Optional[float] = None
    confidence: float = 1.0


@dataclass
class RealTemporalMetrics:
    """Métricas específicas REALES para evaluación temporal."""
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
    """Historial de entrenamiento REAL para modelo temporal."""
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
    Red Siamesa REAL para autenticación biométrica basada en características dinámicas temporales.
    Implementa arquitectura twin network con BiLSTM para procesar secuencias.
    """
    
    def __init__(self, embedding_dim: int = 128, sequence_length: int = 50, feature_dim: int = 320):
        """Inicializa la red siamesa dinámica REAL."""
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow no disponible")
        
        # Configuración del modelo
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.config = self._load_real_dynamic_config()
        
        # Arquitectura del modelo
        self.base_network = None
        self.siamese_model = None
        self.is_compiled = False
        
        # Estado de entrenamiento REAL
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
        
        logger.info("RealSiameseDynamicNetwork inicializada - 100% SIN SIMULACIÓN")
    
    def _load_real_dynamic_config(self) -> Dict[str, Any]:
        """Carga configuración REAL de la red siamesa dinámica."""
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
        
        logger.info("Configuración REAL de red dinámica cargada")
        return real_config
    
    def _get_real_model_save_path(self) -> Path:
        """Obtiene ruta para guardar modelos REALES."""
        models_dir = Path(get_config('paths.models', 'biometric_data/models'))
        return models_dir / 'dynamic_model.h5'
    
    def build_real_base_network(self) -> Model:
        """Construye la red base temporal REAL con BiLSTM."""
        try:
            logger.info("Construyendo red base temporal REAL...")
            
            # Input layer para secuencias temporales
            input_layer = layers.Input(
                shape=(self.sequence_length, self.feature_dim), 
                name='real_dynamic_sequence'
            )
            
            x = input_layer
            
            # Masking para secuencias de longitud variable
            if self.config['use_masking']:
                x = layers.Masking(mask_value=0.0, name='real_sequence_masking')(x)
                logger.info("  - Masking aplicado")
            
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
            
            return base_model
            
        except Exception as e:
            logger.error(f"Error construyendo red base temporal: {e}")
            raise
    
    def _build_real_temporal_layers(self, x):
        """Construye las capas temporales REALES (BiLSTM)."""
        try:
            lstm_units = self.config['lstm_units']
            processing_type = self.config['sequence_processing']
            
            logger.info(f"=== CONSTRUYENDO CAPAS TEMPORALES ===")
            logger.info(f"Input shape: {x.shape}")
            logger.info(f"Processing: {processing_type}")
            logger.info(f"LSTM units: {lstm_units}")
            
            for i, units in enumerate(lstm_units):
                logger.info(f"--- Capa LSTM {i+1}/{len(lstm_units)} ---")
                
                # IMPORTANTE: Si usamos pooling personalizado, todas las capas retornan secuencias
                if self.config['temporal_pooling'] in ['attention', 'last']:
                    return_sequences = True
                    logger.info(f"Capa {i+1}: return_sequences=True (pooling personalizado)")
                else:
                    return_sequences = i < len(lstm_units) - 1
                    logger.info(f"Capa {i+1}: return_sequences={return_sequences}")
                
                logger.info(f"Antes de capa {i+1}: shape = {x.shape}")
                
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
                        logger.info(f"✓ BiLSTM {i+1} construido")
                        logger.info(f"Después de BiLSTM {i+1}: shape = {x.shape}")
                    except Exception as lstm_error:
                        logger.error(f"ERROR en BiLSTM {i+1}: {lstm_error}")
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
                        logger.info(f"✓ LSTM {i+1} construido")
                        logger.info(f"Después de LSTM {i+1}: shape = {x.shape}")
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
                        logger.info(f"✓ GRU {i+1} construido")
                        logger.info(f"Después de GRU {i+1}: shape = {x.shape}")
                    except Exception as gru_error:
                        logger.error(f"ERROR en GRU {i+1}: {gru_error}")
                        raise
                
                # Normalización entre capas
                if i < len(lstm_units) - 1:
                    logger.info(f"Aplicando LayerNormalization después de capa {i+1}...")
                    try:
                        x = layers.LayerNormalization(name=f'real_layer_norm_{i+1}')(x)
                        logger.info(f"✓ LayerNormalization {i+1} aplicada")
                        logger.info(f"Después de LayerNorm {i+1}: shape = {x.shape}")
                    except Exception as norm_error:
                        logger.error(f"ERROR en LayerNormalization {i+1}: {norm_error}")
                        raise
                else:
                    logger.info(f"Última capa: omitiendo LayerNormalization")
            
            logger.info(f"=== CAPAS TEMPORALES COMPLETADAS ===")
            logger.info(f"Shape final: {x.shape}")
            
            return x
            
        except Exception as e:
            logger.error(f"=== ERROR CONSTRUYENDO CAPAS TEMPORALES ===")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _build_real_temporal_pooling(self, x):
        """Construye el pooling temporal REAL con attention mechanism."""
        try:
            pooling_type = self.config['temporal_pooling']
            logger.info(f"=== CONSTRUYENDO POOLING TEMPORAL ===")
            logger.info(f"Input shape: {x.shape}")
            logger.info(f"Pooling type: {pooling_type}")
            
            if pooling_type == 'attention':
                logger.info("--- CONSTRUYENDO ATTENTION MECHANISM ---")
                
                try:
                    # 1. Crear contexto global
                    logger.info("Paso 1: Global context...")
                    global_context = layers.GlobalAveragePooling1D(name='real_global_context')(x)
                    logger.info(f"Global context shape: {global_context.shape}")
                    
                    # 2. Expandir contexto
                    logger.info("Paso 2: Expandiendo context...")
                    global_context_expanded = layers.RepeatVector(
                        self.sequence_length, 
                        name='real_context_expanded'
                    )(global_context)
                    logger.info(f"Context expanded shape: {global_context_expanded.shape}")
                    
                    # 3. Concatenar
                    logger.info("Paso 3: Concatenando...")
                    combined = layers.Concatenate(
                        axis=-1, 
                        name='real_combined_features'
                    )([x, global_context_expanded])
                    logger.info(f"Combined shape: {combined.shape}")
                    
                    # 4. Attention scores
                    logger.info("Paso 4: Attention scores...")
                    attention_scores = layers.Dense(
                        1, 
                        activation='tanh', 
                        name='real_attention_scores'
                    )(combined)
                    logger.info(f"Attention scores shape: {attention_scores.shape}")
                    
                    # 5. Normalizar con softmax
                    logger.info("Paso 5: Softmax...")
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
        """Construye el modelo siamés temporal REAL completo."""
        try:
            if self.base_network is None:
                self.build_real_base_network()
            
            logger.info("Construyendo modelo siamés temporal REAL...")
            
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
            logger.info(f"  - Métrica: {self.config['distance_metric']}")
            
            return siamese_model
            
        except Exception as e:
            logger.error(f"Error construyendo modelo siamés: {e}")
            raise
    
    def compile_real_model(self):
        """Compila el modelo siamés temporal REAL."""
        try:
            if self.siamese_model is None:
                self.build_real_siamese_model()
            
            logger.info("Compilando modelo siamés temporal REAL...")
            
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
            logger.info(f"  - Pérdida: {self.config['loss_function']}")
            
        except Exception as e:
            logger.error(f"Error compilando modelo: {e}")
            raise
    
    def _contrastive_loss_real(self, y_true, y_pred):
        """Función de pérdida contrastiva REAL."""
        epsilon = 1e-8
        margin = tf.constant(self.config.get('margin', 1.0), dtype=tf.float32)
        
        distance = tf.sqrt(tf.square(y_pred) + epsilon)
        
        square_pred = tf.square(distance)
        margin_square = tf.square(tf.maximum(margin - distance, 0.0))
        
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    
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
    
    def _pad_or_truncate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Ajusta la secuencia a la longitud fija requerida."""
        current_length = sequence.shape[0]
        
        if current_length >= self.sequence_length:
            return sequence[:self.sequence_length]
        else:
            padding = np.zeros((self.sequence_length - current_length, self.feature_dim))
            return np.vstack([sequence, padding])
    
    def _setup_real_training_callbacks(self) -> List[callbacks.Callback]:
        """Configura callbacks para entrenamiento REAL."""
        callbacks_list = []
        
        logger.info(f"=== CONFIGURANDO CALLBACKS ===")
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        logger.info(f"✓ Early stopping: patience={self.config['early_stopping_patience']}")
        
        # Reduce learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['reduce_lr_patience'],
            min_lr=self.config['min_lr'],
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        logger.info(f"✓ ReduceLR: patience={self.config['reduce_lr_patience']}")
        
        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(self.model_save_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks_list.append(checkpoint)
        logger.info(f"✓ Checkpoint: {self.model_save_path}")
        
        # Monitor de LR
        class LRMonitor(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                
                current_lr = float(self.model.optimizer.learning_rate)
                train_loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                
                logger.info(f"Época {epoch+1}:")
                logger.info(f"  ┣━ LR: {current_lr:.2e}")
                logger.info(f"  ┣━ Train Loss: {train_loss:.6f}")
                logger.info(f"  ┗━ Val Loss: {val_loss:.6f}")
        
        callbacks_list.append(LRMonitor())
        logger.info(f"✓ LR Monitor configurado")
        
        # Monitor anti-NaN
        class NaNStoppingCallback(callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                if logs:
                    current_loss = logs.get('loss', 0)
                    if tf.math.is_nan(current_loss) or tf.math.is_inf(current_loss):
                        logger.error(f"NaN/Inf en batch {batch}")
                        self.model.stop_training = True
            
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    val_loss = logs.get('val_loss', 0)
                    if tf.math.is_nan(val_loss) or tf.math.is_inf(val_loss):
                        logger.error(f"NaN/Inf en época {epoch}")
                        self.model.stop_training = True
        
        callbacks_list.append(NaNStoppingCallback())
        logger.info(f"✓ Anti-NaN configurado")
        
        logger.info(f"=== TOTAL CALLBACKS: {len(callbacks_list)} ===")
        
        return callbacks_list
    
    def load_real_temporal_data_from_database(self, database) -> bool:
        """
        Carga datos temporales REALES desde la base de datos biométrica.
        VERSIÓN FINAL - Maneja usuarios Bootstrap y Normales correctamente.
        """
        try:
            logger.info("=== CARGANDO DATOS TEMPORALES REALES ===")
            logger.info("🔄 Buscando templates con datos temporales...")
            
            # Obtener usuarios
            real_users = database.list_users()
            
            if len(real_users) < self.config.get('min_users_for_training', 2):
                logger.error(f"Insuficientes usuarios: {len(real_users)} < 2")
                return False
            
            logger.info(f"📊 Usuarios encontrados: {len(real_users)}")
            
            # Limpiar muestras
            self.real_training_samples.clear()
            
            users_with_sufficient_data = 0
            total_samples_loaded = 0
            
            for user in real_users:
                try:
                    logger.info(f"📂 Procesando: {user.username} ({user.user_id})")
                    
                    # Obtener templates
                    user_templates_list = []
                    for template_id, template in database.templates.items():
                        if template.user_id == user.user_id:
                            user_templates_list.append(template)
                    
                    if not user_templates_list:
                        logger.info(f"   ⚠️ Sin templates")
                        continue
                    
                    logger.info(f"   📊 Templates: {len(user_templates_list)}")
                    
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
                    
                    logger.info(f"   📊 Templates temporales: {len(temporal_templates)}")
                    
                    # Procesar templates temporales
                    user_temporal_samples = []
                    
                    for template in temporal_templates:
                        try:
                            temporal_sequence = template.metadata.get('temporal_sequence', None)
                            individual_sequences = template.metadata.get('individual_temporal_sequences', [])
                            
                            has_individual_data = individual_sequences and len(individual_sequences) > 1
                            has_temporal_sequence = temporal_sequence and len(temporal_sequence) >= 5
                            
                            if has_temporal_sequence or has_individual_data:
                                logger.info(f"   🔧 Procesando: {template.gesture_name}")
                                
                                # PROCESAR SECUENCIAS INDIVIDUALES (USUARIOS NORMALES)
                                if has_individual_data:
                                    logger.info(f"       🎯 {len(individual_sequences)} secuencias individuales")
                                    
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
                                                        'sequence_index': seq_idx,
                                                        'confidence': template.confidence,
                                                        'gesture_name': template.gesture_name,
                                                        'parent_template_id': template.template_id
                                                    }
                                                )
                                                user_temporal_samples.append(dynamic_sample)
                                                sequences_loaded += 1
                                    
                                    genuine_pairs = sequences_loaded * (sequences_loaded - 1) // 2 if sequences_loaded >= 2 else 0
                                    
                                    logger.info(f"       ✅ Secuencias: {sequences_loaded}")
                                    logger.info(f"       📊 Pares genuinos: {genuine_pairs}")
                                
                                # PROCESAR SECUENCIA TEMPORAL (BOOTSTRAP)
                                elif has_temporal_sequence:
                                    temporal_array = np.array(temporal_sequence, dtype=np.float32)
                                    
                                    if len(temporal_array.shape) == 2 and temporal_array.shape[1] == self.feature_dim:
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
                                        logger.warning(f"   ⚠️ Dimensiones incorrectas: {temporal_array.shape}")
                            else:
                                logger.warning(f"   ⚠️ Sin datos temporales válidos")
                        
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
                        logger.info(f"   📊 Secuencias: {len(user_temporal_samples)}")
                        logger.info(f"   🎯 Gestos: {len(gesture_counts)}")
                        for gesture, count in gesture_counts.items():
                            logger.info(f"      • {gesture}: {count}")
                    else:
                        logger.warning(f"   ⚠️ Pocas secuencias: {len(user_temporal_samples)}")
                    
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
                logger.error(f"Cargadas: {total_samples_loaded} < {min_total_samples}")
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
                logger.warning(f"División simple: {e}")
                split_idx = int(0.8 * len(self.real_training_samples))
                self.real_validation_samples = self.real_training_samples[split_idx:]
                self.real_training_samples = self.real_training_samples[:split_idx]
            
            # Actualizar contador
            all_samples = self.real_training_samples + self.real_validation_samples
            user_stats = {}
            for sample in all_samples:
                user_stats[sample.user_id] = user_stats.get(sample.user_id, 0) + 1
            
            self.users_trained_count = len(user_stats)
            
            # Reporte final
            logger.info("=" * 60)
            logger.info("✅ DATOS TEMPORALES CARGADOS")
            logger.info("=" * 60)
            logger.info(f"👥 Usuarios: {users_with_sufficient_data}")
            logger.info(f"🧬 Total secuencias: {total_samples_loaded}")
            logger.info(f"📊 Promedio/usuario: {total_samples_loaded/users_with_sufficient_data:.1f}")
            logger.info(f"📐 Dimensiones: {self.feature_dim}")
            logger.info(f"📈 DISTRIBUCIÓN:")
            for user_id, count in user_stats.items():
                user_name = next((u.username for u in real_users if u.user_id == user_id), user_id)
                logger.info(f"   • {user_name}: {count}")
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
        """Valida la calidad de los datos temporales REALES."""
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
                    logger.error(f"Dimensión incorrecta en muestra {i}")
                    return False
            
            # Validar longitudes
            sequence_lengths = [sample.temporal_features.shape[0] for sample in self.real_training_samples]
            min_length = min(sequence_lengths)
            max_length = max(sequence_lengths)
            avg_length = sum(sequence_lengths) / len(sequence_lengths)
            
            logger.info(f"Longitudes - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.1f}")
            
            if min_length < 5:
                logger.error(f"Secuencia muy corta: {min_length} < 5")
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
        """Crea pares de secuencias temporales REALES."""
        try:
            logger.info("Creando pares temporales REALES...")
            
            pairs_a = []
            pairs_b = []
            labels = []
            
            samples = self.real_training_samples
            
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
            
            logger.info(f"Pares genuinos: {genuine_pairs}")
            
            # Crear pares impostores
            impostor_pairs = 0
            target_impostor_pairs = genuine_pairs
            
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
            
            logger.info(f"Pares impostores: {impostor_pairs}")
            
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
    
    def train_with_real_data(self, database, validation_split: float = 0.2) -> RealTemporalTrainingHistory:
        """Entrena el modelo con datos temporales REALES."""
        try:
            start_time = time.time()
            logger.info("=== INICIANDO ENTRENAMIENTO TEMPORAL ===")
            
            # 1. Cargar datos
            if not self.load_real_temporal_data_from_database(database):
                raise ValueError("No se pudieron cargar datos temporales REALES")
            
            # 2. Validar calidad
            if not self.validate_real_temporal_data_quality():
                raise ValueError("Datos no cumplen criterios de calidad")
            
            # 3. Compilar modelo
            if not self.is_compiled:
                self.compile_real_model()
            
            # 4. Crear pares
            pairs_a, pairs_b, labels = self.create_real_temporal_pairs()
            
            # 5. Callbacks
            callbacks_list = self._setup_real_training_callbacks()
            
            # 6. Logs pre-entrenamiento
            logger.info(f"=== CONFIGURACIÓN PRE-ENTRENAMIENTO ===")
            logger.info(f"Learning rate: {self.siamese_model.optimizer.learning_rate.numpy()}")
            logger.info(f"Margen: {self.config.get('margin', 'NO DEFINIDO')}")
            logger.info(f"Batch size: {self.config['batch_size']}")
            logger.info(f"Total parámetros: {self.siamese_model.count_params()}")
            logger.info(f"Pares: {len(labels)}")
            logger.info(f"Shape: {pairs_a.shape}, {pairs_b.shape}")
            
            # 7. Entrenar
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
            
            # 8. Actualizar estado
            self.is_trained = True
            self.training_history.loss = history.history['loss']
            self.training_history.val_loss = history.history['val_loss']
            
            # 9. Evaluar
            metrics = self.evaluate_real_model(pairs_a, pairs_b, labels)
            self.current_metrics = metrics
            
            # 10. Guardar
            self.save_real_model()
            
            total_time = time.time() - start_time
            self.training_history.total_training_time = total_time
            
            logger.info("=== ENTRENAMIENTO COMPLETADO ===")
            logger.info(f"  - Tiempo: {total_time:.1f}s")
            logger.info(f"  - Épocas: {len(history.history['loss'])}")
            logger.info(f"  - Mejor pérdida: {min(history.history['val_loss']):.4f}")
            logger.info(f"  - EER: {metrics.eer:.3f}")
            logger.info(f"  - AUC: {metrics.auc_score:.3f}")
            logger.info("✓ Red dinámica marcada como entrenada")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            raise
        
    def evaluate_real_model(self, sequences_a: np.ndarray, sequences_b: np.ndarray, 
                           labels: np.ndarray) -> RealTemporalMetrics:
        """Evalúa el modelo temporal REAL con métricas específicas."""
        try:
            logger.info("Evaluando modelo temporal REAL...")
            
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
                
                # FAR
                false_accepts = np.sum((predictions == 1) & impostor_mask)
                total_impostors = np.sum(impostor_mask)
                far = false_accepts / total_impostors if total_impostors > 0 else 0
                
                # FRR
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
        """Predice similitud temporal REAL entre dos secuencias."""
        try:
            if not self.is_trained:
                raise ValueError("Modelo no entrenado")
            
            if self.siamese_model is None:
                raise ValueError("Modelo no inicializado")
            
            # Validar dimensiones
            if sequence1.shape[1] != self.feature_dim or sequence2.shape[1] != self.feature_dim:
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
        """Guarda el modelo temporal REAL entrenado."""
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
                'version': '2.0_real'
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
                logger.warning(f"Modelo no encontrado: {self.model_save_path}")
                return False
            
            # Construir arquitectura
            if not self.base_network:
                self.build_real_base_network()
            
            if not self.siamese_model:
                self.build_real_siamese_model()
            
            if not self.is_compiled:
                self.compile_real_model()
            
            # Cargar pesos
            self.siamese_model.load_weights(str(self.model_save_path))
            self.is_trained = True
            self.is_compiled = True
            
            logger.info(f"✓ Modelo cargado: {self.model_save_path}")
            logger.info(f"Parámetros: {self.siamese_model.count_params():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.is_trained = False
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
                    "version": "2.0_real"
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


# ===== TESTING =====
if __name__ == "__main__":
    print("=== TESTING MÓDULO 10: SIAMESE_DYNAMIC_NETWORK REAL ===")
    
    # Test 1: Inicialización
    network = RealSiameseDynamicNetwork(embedding_dim=128, sequence_length=50, feature_dim=320)
    print("✓ Red siamesa temporal inicializada")
    
    # Test 2: Construcción
    try:
        base_model = network.build_real_base_network()
        siamese_model = network.build_real_siamese_model()
        print(f"✓ Arquitectura construida: {siamese_model.count_params():,} parámetros")
    except Exception as e:
        print(f"✗ Error construyendo arquitectura: {e}")
    
    # Test 3: Compilación
    try:
        network.compile_real_model()
        print("✓ Modelo compilado")
    except Exception as e:
        print(f"✗ Error compilando: {e}")
    
    # Test 4: Resumen
    summary = network.get_real_model_summary()
    print(f"✓ Resumen: {summary['architecture']['total_parameters']:,} parámetros")
    print(f"  - Tipo: {summary['architecture']['model_type']}")
    print(f"  - Entrenado: {summary['training']['is_trained']}")
    print(f"  - Listo: {summary['status']['ready_for_inference']}")
    print(f"  - Arquitectura: {summary['architecture']['sequence_processing']}")
    print(f"  - LSTM units: {summary['architecture']['lstm_units']}")
    print(f"  - Pooling: {summary['architecture']['temporal_pooling']}")
    print(f"  - Versión: {summary['status']['version']}")
    
    # Test 5: Predicción
    try:
        seq1 = np.random.randn(25, 320)
        seq2 = np.random.randn(30, 320)
        similarity = network.predict_temporal_similarity_real(seq1, seq2)
        print(f"✓ Predicción: {similarity:.3f}")
    except Exception as e:
        print(f"✓ Error esperado (no entrenado): {str(e)[:50]}...")
    
    print("=== FIN TESTING MÓDULO 10 REAL ===")