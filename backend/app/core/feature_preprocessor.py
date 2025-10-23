"""
MÓDULO 11: FEATURE_PREPROCESSOR
Pipeline unificado de preprocesamiento para características anatómicas y dinámicas (100% REAL)
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from pathlib import Path
import time
import json
import pickle

# Scikit-learn imports
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn no disponible")

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
    """Función de conveniencia para warnings."""
    try:
        logger.warning(message)
    except:
        print(f"WARNING: {message}")


class NormalizationMethod(Enum):
    """Métodos de normalización para datos reales."""
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    QUANTILE = "quantile"
    NONE = "none"


class BalancingMethod(Enum):
    """Métodos de balanceo usando solo datos reales."""
    NONE = "none"
    UNDERSAMPLE = "undersample"
    BALANCED_SUBSAMPLE = "balanced_subsample"
    WEIGHTED = "weighted"
    STRATIFIED_SPLIT = "stratified_split"


class AugmentationStrategy(Enum):
    """Estrategias de augmentación usando solo variaciones reales."""
    NONE = "none"
    TEMPORAL_SHIFTS = "temporal_shifts"
    NOISE_INJECTION = "noise_injection"
    FEATURE_DROPOUT = "feature_dropout"
    MODERATE = "moderate"


@dataclass
class RealPreprocessingConfig:
    """Configuración de preprocesamiento."""
    anatomical_normalization: NormalizationMethod = NormalizationMethod.ROBUST
    dynamic_normalization: NormalizationMethod = NormalizationMethod.STANDARD
    normalize_per_user: bool = False
    
    balancing_method: BalancingMethod = BalancingMethod.BALANCED_SUBSAMPLE
    target_balance_ratio: float = 1.0
    
    augmentation_strategy: AugmentationStrategy = AugmentationStrategy.MODERATE
    augmentation_factor: float = 1.5
    
    cv_folds: int = 5
    cv_strategy: str = "stratified_user"
    test_size: float = 0.2
    validation_size: float = 0.2
    
    outlier_threshold: float = 3.0
    min_samples_per_user: int = 5
    feature_selection: bool = False
    variance_threshold: float = 0.01
    
    cache_transformations: bool = True
    parallel_processing: bool = False
    random_state: int = 42
    stratify_by_user: bool = True


@dataclass
class RealDataQualityMetrics:
    """Métricas de calidad para datos."""
    total_samples: int
    total_users: int
    samples_per_user: Dict[str, int]
    gesture_distribution: Dict[str, int]
    data_quality_score: float
    outlier_percentage: float
    missing_data_percentage: float
    feature_correlation_matrix: Optional[np.ndarray] = None
    recommendations: List[str] = field(default_factory=list)


@dataclass 
class RealBiometricSample:
    """Muestra biométrica."""
    user_id: str
    sample_id: str
    features: np.ndarray
    gesture_name: str
    confidence: float
    timestamp: float
    hand_side: str = "unknown"
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealDynamicSample:
    """Muestra temporal para entrenamiento de red dinámica."""
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
class RealProcessedDataset:
    """Dataset procesado con datos únicamente."""
    anatomical_features: np.ndarray
    anatomical_labels: np.ndarray
    anatomical_users: np.ndarray
    
    dynamic_sequences: np.ndarray
    dynamic_labels: np.ndarray
    dynamic_users: np.ndarray
    
    splits: Dict[str, Dict[str, Any]]
    
    anatomical_pipeline: Pipeline
    dynamic_pipeline: Pipeline
    
    quality_metrics: RealDataQualityMetrics
    preprocessing_stats: Dict[str, Any]
    
class RealFeaturePreprocessor:
    """
    Pipeline unificado de preprocesamiento para características anatómicas y dinámicas REALES.
    Prepara datos de usuarios reales para entrenamiento de redes siamesas biométricas.
    """
    
    def __init__(self):
        """Inicializa el preprocesador de características."""
        
        self.logger = get_logger()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn no disponible")
        
        # Configuración
        self.config = self._load_real_preprocessing_config()
        
        # Extractores de características
        self.anatomical_extractor = self._get_real_anatomical_extractor()
        self.dynamic_extractor = self._get_real_dynamic_extractor()
        
        # Pipelines de transformación
        self.anatomical_pipeline = None
        self.dynamic_pipeline = None
        
        # Estado del preprocesador REAL
        self.is_fitted = False
        self.user_encoders = {}
        self.class_encoders = {}
        
        # Dataset procesado REAL
        self.processed_dataset: Optional[RealProcessedDataset] = None
        
        # Estadísticas REALES
        self.preprocessing_stats = {}
        
        logger.info("RealFeaturePreprocessor inicializado")
    
    def _get_real_anatomical_extractor(self):
        """Obtiene extractor anatómico."""
        try:
            from app.core.anatomical_features_extractor import get_anatomical_features_extractor
            return get_anatomical_features_extractor()
        except ImportError:
            logger.warning("No se pudo importar extractor anatómico")
            return None
    
    def _get_real_dynamic_extractor(self):
        """Obtiene extractor dinámico."""
        try:
            from app.core.dynamic_features_extractor import get_dynamic_features_extractor
            return get_dynamic_features_extractor()
        except ImportError:
            logger.warning("No se pudo importar extractor dinámico")
            return None
    
    def _load_real_preprocessing_config(self) -> RealPreprocessingConfig:
        """Carga configuración de preprocesamiento."""
        try:
            default_config = {
                'anatomical_normalization': 'robust',
                'dynamic_normalization': 'standard', 
                'normalize_per_user': False,
                
                'balancing_method': 'balanced_subsample',
                'target_balance_ratio': 1.0,
                
                'augmentation_strategy': 'moderate',
                'augmentation_factor': 1.5,
                
                'cv_folds': 5,
                'cv_strategy': 'stratified_user',
                'test_size': 0.2,
                'validation_size': 0.2,
                
                'outlier_threshold': 3.0,
                'min_samples_per_user': 5,
                'feature_selection': False,
                'variance_threshold': 0.01,
                
                'cache_transformations': True,
                'parallel_processing': False,
                'random_state': 42,
                'stratify_by_user': True,
            }
            
            config_dict = get_config('biometric.feature_preprocessing', default_config)
            
            config = RealPreprocessingConfig(
                anatomical_normalization=NormalizationMethod(config_dict['anatomical_normalization']),
                dynamic_normalization=NormalizationMethod(config_dict['dynamic_normalization']),
                normalize_per_user=config_dict['normalize_per_user'],
                balancing_method=BalancingMethod(config_dict['balancing_method']),
                target_balance_ratio=config_dict['target_balance_ratio'],
                augmentation_strategy=AugmentationStrategy(config_dict['augmentation_strategy']),
                augmentation_factor=config_dict['augmentation_factor'],
                cv_folds=config_dict['cv_folds'],
                cv_strategy=config_dict['cv_strategy'],
                test_size=config_dict['test_size'],
                outlier_threshold=config_dict['outlier_threshold'],
                min_samples_per_user=config_dict['min_samples_per_user'],
                feature_selection=config_dict['feature_selection']
            )
            
            logger.info("Configuración de preprocesamiento cargada")
            return config
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return RealPreprocessingConfig()
    
    def create_real_anatomical_pipeline(self) -> Pipeline:
        """Crea pipeline de preprocesamiento para características anatómicas."""
        try:
            logger.info("Creando pipeline anatómico...")
            
            steps = []
            
            # 1. Removedor de outliers
            if self.config.outlier_threshold > 0:
                outlier_remover = RealOutlierRemover(threshold=self.config.outlier_threshold)
                steps.append(('outlier_removal', outlier_remover))
                logger.info(f"  - Outlier remover: threshold={self.config.outlier_threshold}")
            
            # 2. Selector de varianza
            if self.config.feature_selection:
                variance_selector = RealVarianceThresholdSelector(threshold=self.config.variance_threshold)
                steps.append(('variance_selection', variance_selector))
                logger.info(f"  - Variance selector: threshold={self.config.variance_threshold}")
            
            # 3. Normalizador
            if self.config.anatomical_normalization != NormalizationMethod.NONE:
                normalizer = self._create_real_normalizer(self.config.anatomical_normalization)
                steps.append(('normalization', normalizer))
                logger.info(f"  - Normalizer: {self.config.anatomical_normalization.value}")
            
            pipeline = Pipeline(steps, memory=None if not self.config.cache_transformations else 'cache')
            
            logger.info(f"Pipeline anatómico creado: {len(steps)} pasos")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error creando pipeline anatómico: {e}")
            raise
    
    def create_real_dynamic_pipeline(self) -> Pipeline:
        """Crea pipeline de preprocesamiento para secuencias dinámicas REALES."""
        try:
            logger.info("Creando pipeline dinámico...")
            
            steps = []
            
            # 1. Removedor de outliers temporales
            if self.config.outlier_threshold > 0:
                temporal_outlier_remover = RealTemporalOutlierRemover(threshold=self.config.outlier_threshold)
                steps.append(('temporal_outlier_removal', temporal_outlier_remover))
                logger.info(f"  - Temporal outlier remover añadido")
            
            # 2. Suavizador temporal
            temporal_smoother = RealTemporalSmoother(window_size=3)
            steps.append(('temporal_smoothing', temporal_smoother))
            logger.info(f"  - Temporal smoother añadido")
            
            # 3. Normalizador temporal
            if self.config.dynamic_normalization != NormalizationMethod.NONE:
                temporal_normalizer = self._create_real_temporal_normalizer(self.config.dynamic_normalization)
                steps.append(('temporal_normalization', temporal_normalizer))
                logger.info(f"  - Temporal normalizer: {self.config.dynamic_normalization.value}")
            
            pipeline = Pipeline(steps, memory=None if not self.config.cache_transformations else 'cache')
            
            logger.info(f"Pipeline dinámico creado: {len(steps)} pasos")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error creando pipeline dinámico: {e}")
            raise
    
    def _create_real_normalizer(self, method: NormalizationMethod):
        """Crea normalizador REAL para características anatómicas."""
        if method == NormalizationMethod.STANDARD:
            return StandardScaler()
        elif method == NormalizationMethod.ROBUST:
            return RobustScaler()
        elif method == NormalizationMethod.MINMAX:
            return MinMaxScaler()
        else:
            raise ValueError(f"Método no soportado: {method}")
    
    def _create_real_temporal_normalizer(self, method: NormalizationMethod):
        """Crea normalizador REAL para secuencias temporales."""
        if method == NormalizationMethod.STANDARD:
            return RealTemporalStandardScaler()
        elif method == NormalizationMethod.ROBUST:
            return RealTemporalRobustScaler()
        elif method == NormalizationMethod.MINMAX:
            return RealTemporalMinMaxScaler()
        else:
            raise ValueError(f"Método temporal no soportado: {method}")
    
    def create_real_biometric_samples_from_features(self, 
                                               anatomical_features: List,
                                               dynamic_features: List,
                                               user_ids: List[str],
                                               gesture_names: List[str],
                                               additional_metadata: Optional[List[Dict]] = None) -> Tuple[List[RealBiometricSample], List[RealDynamicSample]]:
        """
        Convierte vectores de características REALES en muestras biométricas.
        
        Args:
            anatomical_features: Lista de vectores anatómicos REALES
            dynamic_features: Lista de vectores dinámicos REALES  
            user_ids: IDs de usuarios REALES correspondientes
            gesture_names: Nombres de gestos REALES
            additional_metadata: Metadata adicional REAL (opcional)
            
        Returns:
            Tupla (muestras_anatómicas_reales, muestras_dinámicas_reales)
        """
        try:
            logger.info("Creando muestras biométricas REALES desde características...")
            
            if len(anatomical_features) != len(dynamic_features):
                raise ValueError("Número de características anatómicas y dinámicas debe coincidir")
            
            if len(user_ids) != len(anatomical_features):
                raise ValueError("Número de user_ids debe coincidir con características")
            
            anatomical_samples = []
            dynamic_samples = []
            
            for i, (anat_feat, dyn_feat, user_id, gesture) in enumerate(
                zip(anatomical_features, dynamic_features, user_ids, gesture_names)
            ):
                # Metadata adicional REAL
                metadata = additional_metadata[i] if additional_metadata else {}
                
                # Extraer características REALES
                if hasattr(anat_feat, 'complete_vector'):
                    anat_vector = anat_feat.complete_vector
                else:
                    anat_vector = np.array(anat_feat)
                
                if hasattr(dyn_feat, 'complete_vector'):
                    dyn_vector = dyn_feat.complete_vector
                else:
                    dyn_vector = np.array(dyn_feat)
                
                # Muestra anatómica REAL
                anat_sample = RealBiometricSample(
                    user_id=user_id,
                    sample_id=f"{user_id}_{gesture}_{i}_{int(time.time())}",
                    features=anat_vector,
                    gesture_name=gesture,
                    confidence=metadata.get('confidence', 1.0),
                    timestamp=time.time(),
                    hand_side=metadata.get('hand_side', 'unknown'),
                    quality_score=metadata.get('quality_score', 1.0),
                    metadata=metadata
                )
                anatomical_samples.append(anat_sample)
                
                # Muestra dinámica REAL
                # Si es secuencia temporal, mantener forma; si es vector, convertir a secuencia
                if dyn_vector.ndim == 1:
                    sequence = dyn_vector.reshape(1, -1)  # Convertir vector a secuencia mínima
                    seq_length = 1
                else:
                    sequence = dyn_vector
                    seq_length = dyn_vector.shape[0]
                
                dyn_sample = RealDynamicSample(
                    user_id=user_id,
                    sample_id=f"{user_id}_{gesture}_seq_{i}_{int(time.time())}",
                    sequence=sequence,
                    transition_type=metadata.get('transition_type', f"{gesture}->Next"),
                    start_gesture=gesture,
                    end_gesture=metadata.get('end_gesture', 'Unknown'),
                    sequence_length=seq_length,
                    duration=metadata.get('duration', 1.0),
                    quality_score=metadata.get('quality_score', 1.0),
                    metadata=metadata
                )
                dynamic_samples.append(dyn_sample)
            
            unique_users = len(set(user_ids))
            logger.info(f"Creadas {len(anatomical_samples)} muestras biométricas REALES de {unique_users} usuarios")
            logger.info(f"  - Anatomical samples: {len(anatomical_samples)}")
            logger.info(f"  - Dynamic samples: {len(dynamic_samples)}")
            
            return anatomical_samples, dynamic_samples
            
        except Exception as e:
            logger.error("Error creando muestras biométricas REALES", e)
            raise


    def analyze_real_data_quality(self, anatomical_samples: List[RealBiometricSample], 
                                 dynamic_samples: List[RealDynamicSample]) -> RealDataQualityMetrics:
        """Analiza la calidad de datos REALES."""
        try:
            logger.info("Analizando calidad de datos REALES...")
            
            # Contar samples y usuarios
            total_anat_samples = len(anatomical_samples)
            total_dyn_samples = len(dynamic_samples)
            
            anat_users = set(sample.user_id for sample in anatomical_samples)
            dyn_users = set(sample.user_id for sample in dynamic_samples)
            all_users = anat_users.union(dyn_users)
            
            # Samples por usuario
            samples_per_user = {}
            for user_id in all_users:
                anat_count = sum(1 for s in anatomical_samples if s.user_id == user_id)
                dyn_count = sum(1 for s in dynamic_samples if s.user_id == user_id)
                samples_per_user[user_id] = {
                    'anatomical': anat_count, 
                    'dynamic': dyn_count, 
                    'total': anat_count + dyn_count
                }
            
            # Distribución de gestos
            gesture_distribution = {}
            for sample in anatomical_samples:
                gesture_distribution[sample.gesture_name] = gesture_distribution.get(sample.gesture_name, 0) + 1
            
            # Detectar outliers
            if anatomical_samples:
                anatomical_features = np.array([sample.features for sample in anatomical_samples])
                quality_scores = np.array([sample.quality_score for sample in anatomical_samples])

                z_scores = np.abs((anatomical_features - np.mean(anatomical_features, axis=0)) / (np.std(anatomical_features, axis=0) + 1e-8))
                outlier_mask = np.any(z_scores > self.config.outlier_threshold, axis=1)
                outlier_percentage = np.mean(outlier_mask) * 100
                
                # Matriz de correlación
                try:
                    correlation_matrix = np.corrcoef(anatomical_features.T)
                except:
                    correlation_matrix = None
            else:
                outlier_percentage = 0.0
                correlation_matrix = None
            
            # Detectar datos faltantes
            missing_data_percentage = 0.0
            if anatomical_samples:
                total_features = len(anatomical_samples) * len(anatomical_samples[0].features)
                missing_features = sum(
                    np.sum((sample.features == 0) | np.isnan(sample.features)) 
                    for sample in anatomical_samples
                )
                missing_data_percentage = (missing_features / total_features) * 100
            
            # Calcular score de calidad
            quality_score = self._calculate_real_quality_score(
                samples_per_user, gesture_distribution, outlier_percentage, missing_data_percentage
            )
            
            # Generar recomendaciones
            recommendations = self._generate_real_recommendations(
                samples_per_user, outlier_percentage, missing_data_percentage
            )
            
            metrics = RealDataQualityMetrics(
                total_samples=total_anat_samples + total_dyn_samples,
                total_users=len(all_users),
                samples_per_user=samples_per_user,
                gesture_distribution=gesture_distribution,
                data_quality_score=quality_score,
                outlier_percentage=outlier_percentage,
                missing_data_percentage=missing_data_percentage,
                feature_correlation_matrix=correlation_matrix,
                recommendations=recommendations
            )
            
            logger.info(f"Análisis completado - Score: {quality_score:.1f}/100")
            logger.info(f"  - Total muestras: {metrics.total_samples}")
            logger.info(f"  - Total usuarios: {metrics.total_users}")
            logger.info(f"  - Outliers: {outlier_percentage:.1f}%")
            logger.info(f"  - Faltantes: {missing_data_percentage:.1f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analizando calidad: {e}")
            return RealDataQualityMetrics(
                total_samples=len(anatomical_samples) + len(dynamic_samples),
                total_users=len(set(s.user_id for s in anatomical_samples + dynamic_samples)),
                samples_per_user={},
                gesture_distribution={},
                data_quality_score=50.0,
                outlier_percentage=0.0,
                missing_data_percentage=0.0,
                recommendations=["Error en análisis"]
            )
    
    def _calculate_real_quality_score(self, samples_per_user: Dict, gesture_distribution: Dict, 
                                     outlier_percentage: float, missing_data_percentage: float) -> float:
        """Calcula score de calidad basado en datos REALES."""
        try:
            score = 100.0
            
            # Penalizar usuarios con pocas muestras
            user_samples = [info['total'] for info in samples_per_user.values()]
            if user_samples:
                avg_samples = np.mean(user_samples)
                if avg_samples < self.config.min_samples_per_user:
                    score -= 20.0
            
            # Penalizar outliers
            if outlier_percentage > 20:
                score -= 25.0
            elif outlier_percentage > 10:
                score -= 15.0
            
            # Penalizar datos faltantes
            if missing_data_percentage > 10:
                score -= 20.0
            elif missing_data_percentage > 5:
                score -= 10.0
            
            # Penalizar desbalance de gestos
            if gesture_distribution:
                gesture_counts = list(gesture_distribution.values())
                cv_gestures = np.std(gesture_counts) / (np.mean(gesture_counts) + 1e-8)
                if cv_gestures > 0.5:
                    score -= 15.0
            
            return max(0.0, score)
            
        except Exception:
            return 50.0
    
    def _generate_real_recommendations(self, samples_per_user: Dict, outlier_percentage: float, 
                                      missing_data_percentage: float) -> List[str]:
        """Genera recomendaciones basadas en datos REALES."""
        recommendations = []
        
        try:
            # Muestras por usuario
            low_sample_users = [
                user_id for user_id, info in samples_per_user.items() 
                if info['total'] < self.config.min_samples_per_user
            ]
            if low_sample_users:
                recommendations.append(f"Usuarios con <{self.config.min_samples_per_user} muestras")
            
            # Outliers
            if outlier_percentage > 20:
                recommendations.append(f"Alto % outliers: {outlier_percentage:.1f}%")
            
            # Datos faltantes
            if missing_data_percentage > 5:
                recommendations.append(f"Datos faltantes detectados: {missing_data_percentage:.1f}%")
            
        except Exception:
            recommendations.append("Error generando recomendaciones")
        
        return recommendations
    
    def fit_real_data(self, anatomical_samples: List[RealBiometricSample], 
                 dynamic_samples: List[RealDynamicSample]) -> bool:
        """
        Ajusta el preprocesador.
        
        Args:
            anatomical_samples: Muestras anatómicas 
            dynamic_samples: Muestras dinámicas 
            
        Returns:
            True si el ajuste fue exitoso
        """
        try:
            logger.info("=== AJUSTANDO PREPROCESADOR CON DATOS  ===")
            
            # 1. Análisis de calidad de datos 
            logger.info("Analizando calidad de datos...")
            quality_metrics = self.analyze_real_data_quality(anatomical_samples, dynamic_samples)
            
            # 2. Crear pipelines 
            logger.info("Creando pipelines de transformación...")
            self.anatomical_pipeline = self.create_real_anatomical_pipeline()
            self.dynamic_pipeline = self.create_real_dynamic_pipeline()
            
            # 3. Extraer características y metadatos 
            logger.info("Extrayendo características...")
            anatomical_features = np.array([sample.features for sample in anatomical_samples])
            anatomical_labels = np.array([sample.gesture_name for sample in anatomical_samples])
            anatomical_users = np.array([sample.user_id for sample in anatomical_samples])
            
            # Para secuencias dinámicas, aplanar temporalmente para pipeline
            dynamic_sequences = []
            dynamic_labels = []
            dynamic_users = []
            
            for sample in dynamic_samples:
                # Asegurar que la secuencia tenga forma correcta
                if sample.sequence.ndim == 1:
                    sequence = sample.sequence.reshape(1, -1)
                else:
                    sequence = sample.sequence
                
                dynamic_sequences.append(sequence)
                dynamic_labels.append(sample.transition_type)
                dynamic_users.append(sample.user_id)
            
            dynamic_sequences = np.array(dynamic_sequences)
            dynamic_labels = np.array(dynamic_labels)
            dynamic_users = np.array(dynamic_users)
            
            # 4. Ajustar pipelines con datos 
            logger.info("Ajustando pipelines de transformación...")
            anatomical_features_transformed = self.anatomical_pipeline.fit_transform(anatomical_features)
            
            # Procesar secuencias dinámicas
            original_shape = dynamic_sequences.shape
            dynamic_sequences_flat = dynamic_sequences.reshape(len(dynamic_sequences), -1)
            dynamic_sequences_transformed_flat = self.dynamic_pipeline.fit_transform(dynamic_sequences_flat)
            dynamic_sequences_transformed = dynamic_sequences_transformed_flat.reshape(original_shape)
            
            # 5. Crear encoders
            logger.info("Creando encoders...")
            self.user_encoders = {user: i for i, user in enumerate(set(anatomical_users))}
            self.class_encoders = {cls: i for i, cls in enumerate(set(anatomical_labels))}
            
            # 6. Balancear clases usando solo datos 
            if self.config.balancing_method != BalancingMethod.NONE:
                logger.info("Aplicando balanceo...")
                anatomical_features_balanced, anatomical_labels_balanced, anatomical_users_balanced = \
                    self.balance_real_classes(anatomical_features_transformed, anatomical_labels, anatomical_users)
            else:
                anatomical_features_balanced = anatomical_features_transformed
                anatomical_labels_balanced = anatomical_labels
                anatomical_users_balanced = anatomical_users
            
            # 7. Crear splits estratificados 
            logger.info("Creando splits estratificados por usuario...")
            splits = self.create_real_user_stratified_splits(anatomical_samples, dynamic_samples)
            
            # 8. Crear dataset procesado 
            self.processed_dataset = RealProcessedDataset(
                anatomical_features=anatomical_features_balanced,
                anatomical_labels=anatomical_labels_balanced,
                anatomical_users=anatomical_users_balanced,
                dynamic_sequences=dynamic_sequences_transformed,
                dynamic_labels=dynamic_labels,
                dynamic_users=dynamic_users,
                splits=splits,
                anatomical_pipeline=self.anatomical_pipeline,
                dynamic_pipeline=self.dynamic_pipeline,
                quality_metrics=quality_metrics,
                preprocessing_stats=self._calculate_real_preprocessing_stats()
            )
            
            # 9. Actualizar estado
            self.is_fitted = True
            
            logger.info("✓ Preprocesador ajustado exitosamente con datos REALES")
            logger.info(f"  - Muestras anatómicas procesadas: {len(anatomical_features_balanced)}")
            logger.info(f"  - Secuencias dinámicas procesadas: {len(dynamic_sequences_transformed)}")
            logger.info(f"  - Usuarios únicos: {len(self.user_encoders)}")
            logger.info(f"  - Gestos únicos: {len(self.class_encoders)}")
            
            return True
            
        except Exception as e:
            logger.error("Error ajustando preprocesador con datos REALES", e)
            return False

    def balance_real_classes(self, features: np.ndarray, labels: np.ndarray, 
                            users: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Balancea clases."""
        try:
            logger.info("Balanceando clases con datos...")
            
            if self.config.balancing_method == BalancingMethod.NONE:
                return features, labels, users
            
            elif self.config.balancing_method == BalancingMethod.UNDERSAMPLE:
                # Submuestreo de clase mayoritaria
                unique_labels, label_counts = np.unique(labels, return_counts=True)
                min_samples = np.min(label_counts)
                
                balanced_indices = []
                for label in unique_labels:
                    label_indices = np.where(labels == label)[0]
                    selected_indices = np.random.choice(label_indices, min_samples, replace=False)
                    balanced_indices.extend(selected_indices)
                
                balanced_indices = np.array(balanced_indices)
                return features[balanced_indices], labels[balanced_indices], users[balanced_indices]
            
            elif self.config.balancing_method == BalancingMethod.BALANCED_SUBSAMPLE:
                # Submuestreo balanceado
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2:
                    logger.warning("Menos de 2 clases, sin balanceo")
                    return features, labels, users
                
                label_counts = Counter(labels)
                target_size = int(np.mean(list(label_counts.values())) * self.config.target_balance_ratio)
                
                balanced_features = []
                balanced_labels = []
                balanced_users = []
                
                for label in unique_labels:
                    label_mask = labels == label
                    label_features = features[label_mask]
                    label_labels = labels[label_mask]
                    label_users = users[label_mask]
                    
                    if len(label_features) > target_size:
                        resampled_features, resampled_labels, resampled_users = resample(
                            label_features, label_labels, label_users,
                            n_samples=target_size,
                            random_state=self.config.random_state,
                            replace=False
                        )
                    else:
                        resampled_features = label_features
                        resampled_labels = label_labels
                        resampled_users = label_users
                    
                    balanced_features.append(resampled_features)
                    balanced_labels.append(resampled_labels)
                    balanced_users.append(resampled_users)
                
                return (np.vstack(balanced_features), 
                       np.concatenate(balanced_labels), 
                       np.concatenate(balanced_users))
            
            else:
                logger.warning(f"Método no implementado: {self.config.balancing_method}")
                return features, labels, users
                
        except Exception as e:
            logger.error(f"Error balanceando clases: {e}")
            return features, labels, users
    
    def create_real_user_stratified_splits(self, anatomical_samples: List[RealBiometricSample], 
                                          dynamic_samples: List[RealDynamicSample]) -> Dict[str, Dict[str, Any]]:
        """Crea splits estratificados por usuario."""
        try:
            logger.info("Creando splits estratificados...")
            
            all_users = list(set([s.user_id for s in anatomical_samples + dynamic_samples]))
            
            if len(all_users) < 3:
                logger.warning("Pocos usuarios para splits, división simple")
                train_users = all_users[:max(1, int(len(all_users) * 0.6))]
                val_users = all_users[len(train_users):max(len(train_users)+1, len(train_users) + int(len(all_users) * 0.2))]
                test_users = all_users[len(train_users)+len(val_users):]
            else:
                train_users, temp_users = train_test_split(
                    all_users, 
                    test_size=self.config.test_size + self.config.validation_size,
                    random_state=self.config.random_state
                )
                
                if len(temp_users) >= 2:
                    val_users, test_users = train_test_split(
                        temp_users,
                        test_size=self.config.test_size / (self.config.test_size + self.config.validation_size),
                        random_state=self.config.random_state
                    )
                else:
                    val_users = temp_users[:len(temp_users)//2] if temp_users else []
                    test_users = temp_users[len(temp_users)//2:] if temp_users else []
            
            splits = {
                'users': {
                    'train': train_users,
                    'validation': val_users,
                    'test': test_users
                },
                'samples': {
                    'train': [],
                    'validation': [],
                    'test': []
                }
            }
            
            for sample in anatomical_samples + dynamic_samples:
                if sample.user_id in train_users:
                    splits['samples']['train'].append(sample.sample_id if hasattr(sample, 'sample_id') else sample.sequence_id)
                elif sample.user_id in val_users:
                    splits['samples']['validation'].append(sample.sample_id if hasattr(sample, 'sample_id') else sample.sequence_id)
                elif sample.user_id in test_users:
                    splits['samples']['test'].append(sample.sample_id if hasattr(sample, 'sample_id') else sample.sequence_id)
            
            logger.info(f"Splits creados - Train: {len(train_users)} usuarios, Val: {len(val_users)} usuarios, Test: {len(test_users)} usuarios")
            
            return splits
            
        except Exception as e:
            logger.error(f"Error creando splits: {e}")
            all_users = list(set([s.user_id for s in anatomical_samples + dynamic_samples]))
            return {
                'users': {
                    'train': all_users[:max(1, len(all_users)//2)],
                    'validation': all_users[len(all_users)//2:],
                    'test': []
                },
                'samples': {'train': [], 'validation': [], 'test': []}
            }
        
    
    #NUEVO
    def fit_real_data(self, anatomical_samples: List[RealBiometricSample], 
                     dynamic_samples: List[RealDynamicSample]) -> bool:
        """Ajusta el preprocesador con datos REALES de usuarios."""
        try:
            logger.info("=== AJUSTANDO PREPROCESADOR CON DATOS REALES ===")
            
            # 1. Análisis de calidad
            logger.info("Analizando calidad...")
            quality_metrics = self.analyze_real_data_quality(anatomical_samples, dynamic_samples)
            
            # 2. Crear pipelines
            logger.info("Creando pipelines...")
            self.anatomical_pipeline = self.create_real_anatomical_pipeline()
            self.dynamic_pipeline = self.create_real_dynamic_pipeline()
            
            # 3. Extraer características
            logger.info("Extrayendo características...")
            anatomical_features = np.array([sample.features for sample in anatomical_samples])
            anatomical_labels = np.array([sample.gesture_name for sample in anatomical_samples])
            anatomical_users = np.array([sample.user_id for sample in anatomical_samples])
            
            # Para secuencias dinámicas
            dynamic_sequences = []
            dynamic_labels = []
            dynamic_users = []
            
            for sample in dynamic_samples:
                if sample.temporal_features.ndim == 1:
                    sequence = sample.temporal_features.reshape(1, -1)
                else:
                    sequence = sample.temporal_features
                
                dynamic_sequences.append(sequence)
                dynamic_labels.append(sample.transition_types[0] if sample.transition_types else 'unknown')
                dynamic_users.append(sample.user_id)
            
            dynamic_sequences = np.array(dynamic_sequences, dtype=object)
            dynamic_labels = np.array(dynamic_labels)
            dynamic_users = np.array(dynamic_users)
            
            # 4. Ajustar pipelines
            logger.info("Ajustando pipelines...")
            anatomical_features_transformed = self.anatomical_pipeline.fit_transform(anatomical_features)
            
            # Procesar secuencias dinámicas
            if len(dynamic_sequences) > 0 and dynamic_sequences[0] is not None:
                # Obtener forma común
                max_len = max(seq.shape[0] for seq in dynamic_sequences)
                feature_dim = dynamic_sequences[0].shape[1]
                
                # Padding
                dynamic_sequences_padded = []
                for seq in dynamic_sequences:
                    if seq.shape[0] < max_len:
                        padding = np.zeros((max_len - seq.shape[0], feature_dim))
                        seq_padded = np.vstack([seq, padding])
                    else:
                        seq_padded = seq[:max_len]
                    dynamic_sequences_padded.append(seq_padded)
                
                dynamic_sequences_array = np.array(dynamic_sequences_padded)
                original_shape = dynamic_sequences_array.shape
                
                # Aplanar para pipeline
                dynamic_sequences_flat = dynamic_sequences_array.reshape(len(dynamic_sequences_array), -1)
                dynamic_sequences_transformed_flat = self.dynamic_pipeline.fit_transform(dynamic_sequences_flat)
                dynamic_sequences_transformed = dynamic_sequences_transformed_flat.reshape(original_shape)
            else:
                dynamic_sequences_transformed = np.array([])
            
            # 5. Crear encoders
            logger.info("Creando encoders...")
            self.user_encoders = {user: i for i, user in enumerate(set(anatomical_users))}
            self.class_encoders = {cls: i for i, cls in enumerate(set(anatomical_labels))}
            
            # 6. Balancear clases
            if self.config.balancing_method != BalancingMethod.NONE:
                logger.info("Aplicando balanceo...")
                anatomical_features_balanced, anatomical_labels_balanced, anatomical_users_balanced = \
                    self.balance_real_classes(anatomical_features_transformed, anatomical_labels, anatomical_users)
            else:
                anatomical_features_balanced = anatomical_features_transformed
                anatomical_labels_balanced = anatomical_labels
                anatomical_users_balanced = anatomical_users
            
            # 7. Crear splits
            logger.info("Creando splits...")
            splits = self.create_real_user_stratified_splits(anatomical_samples, dynamic_samples)
            
            # 8. Crear dataset procesado
            self.processed_dataset = RealProcessedDataset(
                anatomical_features=anatomical_features_balanced,
                anatomical_labels=anatomical_labels_balanced,
                anatomical_users=anatomical_users_balanced,
                dynamic_sequences=dynamic_sequences_transformed,
                dynamic_labels=dynamic_labels,
                dynamic_users=dynamic_users,
                splits=splits,
                anatomical_pipeline=self.anatomical_pipeline,
                dynamic_pipeline=self.dynamic_pipeline,
                quality_metrics=quality_metrics,
                preprocessing_stats=self._calculate_real_preprocessing_stats()
            )
            
            # 9. Actualizar estado
            self.is_fitted = True
            
            logger.info("✓ Preprocesador ajustado exitosamente")
            logger.info(f"  - Muestras anatómicas: {len(anatomical_features_balanced)}")
            logger.info(f"  - Secuencias dinámicas: {len(dynamic_sequences_transformed) if len(dynamic_sequences_transformed) > 0 else 0}")
            logger.info(f"  - Usuarios únicos: {len(self.user_encoders)}")
            logger.info(f"  - Gestos únicos: {len(self.class_encoders)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ajustando preprocesador: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _calculate_real_preprocessing_stats(self) -> Dict[str, Any]:
        """Calcula estadísticas de preprocesamiento."""
        try:
            if not self.processed_dataset:
                return {}
            
            anat_features = self.processed_dataset.anatomical_features
            dyn_sequences = self.processed_dataset.dynamic_sequences
            
            stats = {
                'anatomical': {
                    'original_shape': anat_features.shape,
                    'transformed_shape': anat_features.shape,
                    'mean': float(np.mean(anat_features)),
                    'std': float(np.std(anat_features)),
                    'min': float(np.min(anat_features)),
                    'max': float(np.max(anat_features)),
                    'feature_count': int(anat_features.shape[1]),
                },
                'dynamic': {
                    'original_shape': dyn_sequences.shape if len(dyn_sequences) > 0 else (0,),
                    'transformed_shape': dyn_sequences.shape if len(dyn_sequences) > 0 else (0,),
                    'mean': float(np.mean(dyn_sequences)) if len(dyn_sequences) > 0 else 0.0,
                    'std': float(np.std(dyn_sequences)) if len(dyn_sequences) > 0 else 0.0,
                    'min': float(np.min(dyn_sequences)) if len(dyn_sequences) > 0 else 0.0,
                    'max': float(np.max(dyn_sequences)) if len(dyn_sequences) > 0 else 0.0,
                    'sequence_length': int(dyn_sequences.shape[1]) if len(dyn_sequences) > 0 and dyn_sequences.ndim > 1 else 0,
                },
                'general': {
                    'total_samples': int(len(anat_features)),
                    'total_users': int(len(set(self.processed_dataset.anatomical_users))),
                    'preprocessing_time': time.time(),
                    'is_real_data': True,
                    'no_synthetic_data': True
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculando estadísticas: {e}")
            return {'error': str(e), 'is_real_data': True}
    
    def get_real_preprocessing_summary(self) -> Dict[str, Any]:
        """Obtiene resumen completo del preprocesamiento REAL."""
        try:
            if not self.is_fitted or not self.processed_dataset:
                return {
                    'status': 'not_fitted',
                    'message': 'Preprocesador no ajustado',
                    'is_real': True
                }
            
            summary = {
                'status': 'fitted',
                'version': '2.0',
                'is_real_data': True,
                'no_synthetic_data': True,
                'config': {
                    'anatomical_normalization': self.config.anatomical_normalization.value,
                    'dynamic_normalization': self.config.dynamic_normalization.value,
                    'balancing_method': self.config.balancing_method.value,
                    'outlier_threshold': self.config.outlier_threshold,
                    'min_samples_per_user': self.config.min_samples_per_user
                },
                'data_quality': {
                    'quality_score': self.processed_dataset.quality_metrics.data_quality_score,
                    'total_samples': self.processed_dataset.quality_metrics.total_samples,
                    'total_users': self.processed_dataset.quality_metrics.total_users,
                    'outlier_percentage': self.processed_dataset.quality_metrics.outlier_percentage,
                    'recommendations_count': len(self.processed_dataset.quality_metrics.recommendations)
                },
                'splits': {
                    'train_users': len(self.processed_dataset.splits['users']['train']),
                    'validation_users': len(self.processed_dataset.splits['users']['validation']),
                    'test_users': len(self.processed_dataset.splits['users']['test'])
                },
                'features': {
                    'anatomical_dim': int(self.processed_dataset.anatomical_features.shape[1]),
                    'dynamic_sequence_shape': tuple(int(x) for x in self.processed_dataset.dynamic_sequences.shape[1:]) if len(self.processed_dataset.dynamic_sequences) > 0 else (0,)
                },
                'pipelines': {
                    'anatomical_steps': len(self.anatomical_pipeline.steps),
                    'dynamic_steps': len(self.dynamic_pipeline.steps)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'is_real_data': True
            }
    
    def save_real_preprocessor(self, filepath: Optional[str] = None) -> bool:
        """Guarda el preprocesador ajustado."""
        try:
            if not self.is_fitted:
                logger.error("Preprocesador no está ajustado")
                return False
            
            if filepath is None:
                models_dir = Path('biometric_data/models')
                models_dir.mkdir(exist_ok=True)
                filepath = models_dir / 'real_feature_preprocessor.pkl'
            
            save_data = {
                'anatomical_pipeline': self.anatomical_pipeline,
                'dynamic_pipeline': self.dynamic_pipeline,
                'user_encoders': self.user_encoders,
                'class_encoders': self.class_encoders,
                'config': self.config,
                'preprocessing_stats': self.preprocessing_stats,
                'is_fitted': self.is_fitted,
                'version': '2.0',
                'is_real_data': True,
                'no_synthetic_data': True
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Preprocesador guardado: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando: {e}")
            return False
    
    def load_real_preprocessor(self, filepath: str) -> bool:
        """Carga un preprocesador previamente ajustado."""
        try:
            if not Path(filepath).exists():
                logger.error(f"Archivo no encontrado: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            if not save_data.get('is_real_data', False):
                logger.error("No es un preprocesador REAL")
                return False
            
            self.anatomical_pipeline = save_data['anatomical_pipeline']
            self.dynamic_pipeline = save_data['dynamic_pipeline']
            self.user_encoders = save_data['user_encoders']
            self.class_encoders = save_data['class_encoders']
            self.config = save_data['config']
            self.preprocessing_stats = save_data['preprocessing_stats']
            self.is_fitted = save_data['is_fitted']
            
            logger.info(f"Preprocesador cargado: {filepath}")
            logger.info(f"  - Versión: {save_data.get('version', 'unknown')}")
            logger.info(f"  - Usuarios: {len(self.user_encoders)}")
            logger.info(f"  - Clases: {len(self.class_encoders)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando: {e}")
            return False
        
# ===== TRANSFORMADORES PERSONALIZADOS =====

class RealOutlierRemover(BaseEstimator, TransformerMixin):
    """Removedor de outliers para características anatómicas."""
    
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.bounds_ = None
    
    def fit(self, X, y=None):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.bounds_ = {
            'lower': mean - self.threshold * std,
            'upper': mean + self.threshold * std
        }
        logger.info(f"OutlierRemover ajustado: threshold={self.threshold}")
        return self
    
    def transform(self, X):
        if self.bounds_ is None:
            raise ValueError("Transformer no ajustado")
        
        X_clean = X.copy()
        X_clean = np.clip(X_clean, self.bounds_['lower'], self.bounds_['upper'])
        return X_clean


class RealVarianceThresholdSelector(BaseEstimator, TransformerMixin):
    """Selector de características por varianza."""
    
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.selected_features_ = None
    
    def fit(self, X, y=None):
        variances = np.var(X, axis=0)
        self.selected_features_ = variances > self.threshold
        selected_count = np.sum(self.selected_features_)
        logger.info(f"Variance selector: {selected_count}/{len(variances)} features")
        return self
    
    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("Selector no ajustado")
        return X[:, self.selected_features_]


class RealTemporalOutlierRemover(BaseEstimator, TransformerMixin):
    """Removedor de outliers para secuencias temporales."""
    
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.global_bounds_ = None
    
    def fit(self, X, y=None):
        mean = np.mean(X)
        std = np.std(X)
        self.global_bounds_ = {
            'lower': mean - self.threshold * std,
            'upper': mean + self.threshold * std
        }
        logger.info(f"TemporalOutlierRemover ajustado")
        return self
    
    def transform(self, X):
        if self.global_bounds_ is None:
            raise ValueError("Transformer no ajustado")
        
        return np.clip(X, self.global_bounds_['lower'], self.global_bounds_['upper'])


class RealTemporalStandardScaler(BaseEstimator, TransformerMixin):
    """Standard scaler para datos temporales."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X, y=None):
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        logger.info(f"TemporalStandardScaler: mean={self.mean_:.3f}, std={self.std_:.3f}")
        return self
    
    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("Scaler no ajustado")
        return (X - self.mean_) / (self.std_ + 1e-8)


class RealTemporalRobustScaler(BaseEstimator, TransformerMixin):
    """Robust scaler para datos temporales REALES."""
    
    def __init__(self):
        self.median_ = None
        self.mad_ = None
    
    def fit(self, X, y=None):
        self.median_ = np.median(X)
        self.mad_ = np.median(np.abs(X - self.median_))
        logger.info(f"TemporalRobustScaler: median={self.median_:.3f}, mad={self.mad_:.3f}")
        return self
    
    def transform(self, X):
        if self.median_ is None:
            raise ValueError("Scaler no ajustado")
        return (X - self.median_) / (self.mad_ + 1e-8)


class RealTemporalMinMaxScaler(BaseEstimator, TransformerMixin):
    """MinMax scaler para datos temporales."""
    
    def __init__(self):
        self.min_ = None
        self.max_ = None
    
    def fit(self, X, y=None):
        self.min_ = np.min(X)
        self.max_ = np.max(X)
        logger.info(f"TemporalMinMaxScaler: min={self.min_:.3f}, max={self.max_:.3f}")
        return self
    
    def transform(self, X):
        if self.min_ is None:
            raise ValueError("Scaler no ajustado")
        return (X - self.min_) / (self.max_ - self.min_ + 1e-8)


class RealTemporalSmoother(BaseEstimator, TransformerMixin):
    """Suavizador temporal para secuencias."""
    
    def __init__(self, window_size=3):
        self.window_size = window_size
    
    def fit(self, X, y=None):
        logger.info(f"TemporalSmoother: window={self.window_size}")
        return self
    
    def transform(self, X):
        if X.ndim == 1:
            return self._smooth_1d(X)
        else:
            return np.array([self._smooth_1d(row) for row in X])
    
    def _smooth_1d(self, x):
        """Suavizado 1D con ventana móvil."""
        if len(x) < self.window_size:
            return x
        
        smoothed = np.convolve(x, np.ones(self.window_size)/self.window_size, mode='same')
        return smoothed


# ===== INSTANCIA GLOBAL =====
_real_preprocessor_instance = None

def get_real_feature_preprocessor() -> RealFeaturePreprocessor:
    """Obtiene instancia global del preprocesador."""
    global _real_preprocessor_instance
    
    if _real_preprocessor_instance is None:
        _real_preprocessor_instance = RealFeaturePreprocessor()
    
    return _real_preprocessor_instance


# Alias para compatibilidad
FeaturePreprocessor = RealFeaturePreprocessor
get_feature_preprocessor = get_real_feature_preprocessor


