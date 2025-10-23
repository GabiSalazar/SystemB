"""
MÓDULO 12: SCORE_FUSION_SYSTEM
Sistema de fusión multimodal para autenticación biométrica (100% REAL)
"""

import numpy as np
import time
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Scikit-learn imports
try:
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve, confusion_matrix
    from sklearn.model_selection import train_test_split
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
warnings.filterwarnings("ignore", category=RuntimeWarning)


def log_warning(message: str):
    """Función de conveniencia para warnings."""
    try:
        logger.warning(message)
    except:
        print(f"WARNING: {message}")


class RealFusionStrategy(Enum):
    """Estrategias de fusión usando solo datos reales."""
    WEIGHTED_AVERAGE = "weighted_average"
    PRODUCT_RULE = "product_rule"
    MAX_RULE = "max_rule"
    MIN_RULE = "min_rule"
    SVM_FUSION = "svm_fusion"
    NEURAL_FUSION = "neural_fusion"
    LOGISTIC_FUSION = "logistic_fusion"
    ADAPTIVE_FUSION = "adaptive_fusion"
    ENSEMBLE_FUSION = "ensemble_fusion"


class RealScoreCalibration(Enum):
    """Métodos de calibración usando solo datos reales."""
    NONE = "none"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"


class RealWeightOptimization(Enum):
    """Métodos de optimización de pesos usando datos reales."""
    FIXED = "fixed"
    GRID_SEARCH = "grid_search"
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    CONFIDENCE_BASED = "confidence_based"


@dataclass
class RealIndividualScores:
    """Scores individuales de ambas modalidades."""
    anatomical_score: float
    dynamic_score: float
    anatomical_confidence: float
    dynamic_confidence: float
    user_id: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealFusedScore:
    """Score fusionado final con decisión."""
    fused_score: float
    decision: bool
    confidence: float
    fusion_strategy: RealFusionStrategy
    individual_scores: RealIndividualScores
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class RealFusionMetrics:
    """Métricas completas del sistema de fusión REAL."""
    far: float
    frr: float
    eer: float
    auc_score: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    fusion_improvement: float
    anatomical_weight: float
    dynamic_weight: float
    optimal_threshold: float
    
    anatomical_metrics: Dict[str, float]
    dynamic_metrics: Dict[str, float]
    
    calibration_quality: float
    fusion_consistency: float
    decision_confidence_avg: float


@dataclass
class RealFusionConfiguration:
    """Configuración del sistema de fusión REAL."""
    fusion_strategy: RealFusionStrategy
    calibration_method: RealScoreCalibration
    weight_optimization: RealWeightOptimization
    anatomical_weight: float
    dynamic_weight: float
    decision_threshold: float
    use_confidence_weighting: bool
    adaptive_threshold: bool
    cross_validation_folds: int
    optimization_metric: str

class RealScoreFusionSystem:
    """
    Sistema de fusión multimodal para autenticación biométrica.
    Combina scores anatómicos y dinámicos para decisión final.
    """
    
    def __init__(self):
        """Inicializa el sistema de fusión."""
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn no disponible")
        
        self.logger = get_logger()
        
        # Configuración
        self.config = self._load_real_fusion_config()
        
        # Redes siamesas
        self.anatomical_network = None
        self.dynamic_network = None
        self.preprocessor = None
        
        # Modelos de fusión entrenados
        self.real_fusion_models = {}
        self.real_score_calibrators = {}
        self.optimal_weights = {'anatomical': 0.5, 'dynamic': 0.5}
        self.optimal_threshold = 0.5
        
        # Estado del sistema REAL
        self.is_trained = False
        self.is_calibrated = False
        self.is_initialized = False
        self.training_history = []
        self.fusion_metrics: Optional[RealFusionMetrics] = None
        
        logger.info("RealScoreFusionSystem inicializado")
    
    def _load_real_fusion_config(self) -> RealFusionConfiguration:
        """Carga configuración del sistema de fusión."""
        try:
            default_config = {
                'fusion_strategy': 'weighted_average',
                'calibration_method': 'none',
                'weight_optimization': 'grid_search',
                'anatomical_weight': 0.6,
                'dynamic_weight': 0.4,
                'decision_threshold': 0.5,
                'use_confidence_weighting': True,
                'adaptive_threshold': False,
                'cross_validation_folds': 5,
                'optimization_metric': 'eer'
            }
            
            config_dict = get_config('biometric.score_fusion', default_config)
            
            config = RealFusionConfiguration(
                fusion_strategy=RealFusionStrategy(config_dict['fusion_strategy']),
                calibration_method=RealScoreCalibration(config_dict['calibration_method']),
                weight_optimization=RealWeightOptimization(config_dict['weight_optimization']),
                anatomical_weight=config_dict['anatomical_weight'],
                dynamic_weight=config_dict['dynamic_weight'],
                decision_threshold=config_dict['decision_threshold'],
                use_confidence_weighting=config_dict['use_confidence_weighting'],
                adaptive_threshold=config_dict['adaptive_threshold'],
                cross_validation_folds=config_dict['cross_validation_folds'],
                optimization_metric=config_dict['optimization_metric']
            )
            
            logger.info("Configuración de fusión cargada")
            return config
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return RealFusionConfiguration(
                fusion_strategy=RealFusionStrategy.WEIGHTED_AVERAGE,
                calibration_method=RealScoreCalibration.NONE,
                weight_optimization=RealWeightOptimization.GRID_SEARCH,
                anatomical_weight=0.6,
                dynamic_weight=0.4,
                decision_threshold=0.5,
                use_confidence_weighting=True,
                adaptive_threshold=False,
                cross_validation_folds=5,
                optimization_metric='eer'
            )
    
    def initialize_real_networks(self, anatomical_network, dynamic_network, preprocessor) -> bool:
        """Inicializa las redes siamesas y preprocesador."""
        try:
            logger.info("Inicializando redes en sistema de fusión...")
            
            # Verificar redes entrenadas
            if not getattr(anatomical_network, 'is_trained', False):
                logger.error("Red anatómica no entrenada")
                return False
            
            if not getattr(dynamic_network, 'is_trained', False):
                logger.error("Red dinámica no entrenada")
                return False
            
            # Validación flexible del preprocesador
            if preprocessor is None:
                logger.error("Preprocesador es None")
                return False
            
            if not hasattr(preprocessor, 'config'):
                logger.error("Preprocesador no tiene configuración válida")
                return False
            
            # No requerir is_fitted en la inicialización
            is_fitted = getattr(preprocessor, 'is_fitted', False)
            if is_fitted:
                logger.info("✓ Preprocesador ya ajustado")
            else:
                logger.info("ℹ Preprocesador será ajustado cuando se necesite")
            
            self.anatomical_network = anatomical_network
            self.dynamic_network = dynamic_network
            self.preprocessor = preprocessor
            
            logger.info("✓ Redes siamesas inicializadas")
            logger.info(f"  - Red anatómica: entrenada")
            logger.info(f"  - Red dinámica: entrenada")
            
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando redes: {e}")
            return False
    
    def initialize_networks(self, anatomical_network, dynamic_network, preprocessor) -> bool:
        """Método alias para compatibilidad con el flujo de entrenamiento."""
        try:
            logger.info("Inicializando redes mediante método de compatibilidad...")
            
            result = self.initialize_real_networks(anatomical_network, dynamic_network, preprocessor)
            
            if result:
                self.is_initialized = True
                logger.info("✓ Sistema de fusión marcado como inicializado")
            else:
                self.is_initialized = False
                logger.error("✗ Falló inicialización  de sistema de fusión")
                
            return result
            
        except Exception as e:
            logger.error(f"Error en initialize_networks: {e}")
            self.is_initialized = False
            return False
    
    def predict_real_individual_scores(self, anatomical_features: np.ndarray,
                                      dynamic_sequence: np.ndarray,
                                      reference_anatomical: List[np.ndarray],
                                      reference_dynamic: List[np.ndarray],
                                      user_id: str) -> RealIndividualScores:
        """Predice scores individuales de ambas modalidades."""
        try:
            logger.info(f"Prediciendo scores individuales para: {user_id}")
            
            # Validar redes
            if not self.anatomical_network or not self.dynamic_network:
                raise ValueError("Redes no inicializadas")
            
            # Validar datos
            if anatomical_features.size == 0 or dynamic_sequence.size == 0:
                raise ValueError("Características vacías")
            
            if len(reference_anatomical) == 0 or len(reference_dynamic) == 0:
                raise ValueError("Referencias vacías")
            
            # Predicción anatómica
            anatomical_similarities = []
            for ref_template in reference_anatomical:
                try:
                    similarity = self.anatomical_network.predict_similarity_real(
                        anatomical_features, ref_template
                    )
                    anatomical_similarities.append(similarity)
                except Exception as e:
                    logger.warning(f"Error en predicción anatómica: {e}")
                    continue
            
            if not anatomical_similarities:
                logger.warning("No se pudieron calcular similitudes anatómicas")
                anatomical_score = 0.0
                anatomical_confidence = 0.0
            else:
                # Usar voting mechanism
                anatomical_score = calculate_score_with_voting(
                    anatomical_similarities,
                    vote_threshold=0.85,
                    min_vote_ratio=0.5
                )
                anatomical_confidence = self._calculate_real_confidence(anatomical_similarities)
            
            # Predicción dinámica
            dynamic_similarities = []
            for ref_sequence in reference_dynamic:
                try:
                    similarity = self.dynamic_network.predict_temporal_similarity_real(
                        dynamic_sequence, ref_sequence
                    )
                    dynamic_similarities.append(similarity)
                except Exception as e:
                    logger.warning(f"Error en predicción dinámica: {e}")
                    continue
            
            if not dynamic_similarities:
                logger.warning("No se pudieron calcular similitudes dinámicas")
                dynamic_score = 0.0
                dynamic_confidence = 0.0
            else:
                # Usar voting mechanism
                dynamic_score = calculate_score_with_voting(
                    dynamic_similarities,
                    vote_threshold=0.85,
                    min_vote_ratio=0.5
                )
                dynamic_confidence = self._calculate_real_confidence(dynamic_similarities)
            
            # Crear scores individuales
            individual_scores = RealIndividualScores(
                anatomical_score=anatomical_score,
                dynamic_score=dynamic_score,
                anatomical_confidence=anatomical_confidence,
                dynamic_confidence=dynamic_confidence,
                user_id=user_id,
                timestamp=time.time(),
                metadata={
                    'anatomical_references': len(reference_anatomical),
                    'dynamic_references': len(reference_dynamic),
                    'anatomical_similarities': anatomical_similarities,
                    'dynamic_similarities': dynamic_similarities
                }
            )
            
            logger.info(f"✓ Scores individuales calculados:")
            logger.info(f"  - Anatómico: {anatomical_score:.3f} (conf: {anatomical_confidence:.3f})")
            logger.info(f"  - Dinámico: {dynamic_score:.3f} (conf: {dynamic_confidence:.3f})")
            
            return individual_scores
            
        except Exception as e:
            logger.error(f"Error prediciendo scores: {e}")
            return RealIndividualScores(
                anatomical_score=0.0,
                dynamic_score=0.0,
                anatomical_confidence=0.0,
                dynamic_confidence=0.0,
                user_id=user_id,
                timestamp=time.time(),
                metadata={'error': str(e)}
            )
    
    def _calculate_real_confidence(self, similarities: List[float]) -> float:
        """Calcula confianza basada en distribución de similitudes."""
        try:
            if not similarities:
                return 0.0
            
            similarities = np.array(similarities)
            
            max_similarity = np.max(similarities)
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            num_references = len(similarities)
            
            # Confianza por número de referencias
            ref_confidence = min(1.0, num_references / 5.0)
            
            # Confianza por consistencia
            consistency_confidence = max(0.0, 1.0 - std_similarity)
            
            # Confianza por score
            score_confidence = max_similarity
            
            # Combinar factores
            overall_confidence = (
                0.4 * ref_confidence +
                0.3 * consistency_confidence +
                0.3 * score_confidence
            )
            
            return float(np.clip(overall_confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return 0.5
        
    def fuse_real_scores(self, individual_scores: RealIndividualScores,
                        strategy: Optional[RealFusionStrategy] = None) -> RealFusedScore:
        """Fusiona scores individuales usando la estrategia especificada."""
        try:
            if strategy is None:
                strategy = self.config.fusion_strategy
            
            logger.info(f"Fusionando scores con estrategia: {strategy.value}")
            
            # Calibrar scores
            anat_score = self._calibrate_real_score(individual_scores.anatomical_score, 'anatomical')
            dyn_score = self._calibrate_real_score(individual_scores.dynamic_score, 'dynamic')
            
            # Obtener pesos
            weights = self._get_real_fusion_weights(individual_scores, strategy)
            
            # Aplicar estrategia de fusión
            if strategy == RealFusionStrategy.WEIGHTED_AVERAGE:
                fused_score = self._real_weighted_average_fusion(anat_score, dyn_score, weights)
                
            elif strategy == RealFusionStrategy.PRODUCT_RULE:
                fused_score = self._real_product_rule_fusion(anat_score, dyn_score, weights)
                
            elif strategy == RealFusionStrategy.MAX_RULE:
                fused_score = max(anat_score, dyn_score)
                
            elif strategy == RealFusionStrategy.MIN_RULE:
                fused_score = min(anat_score, dyn_score)
                
            elif strategy == RealFusionStrategy.SVM_FUSION:
                fused_score = self._real_svm_fusion(anat_score, dyn_score)
                
            elif strategy == RealFusionStrategy.NEURAL_FUSION:
                fused_score = self._real_neural_fusion(anat_score, dyn_score)
                
            elif strategy == RealFusionStrategy.LOGISTIC_FUSION:
                fused_score = self._real_logistic_fusion(anat_score, dyn_score)
                
            elif strategy == RealFusionStrategy.ADAPTIVE_FUSION:
                fused_score = self._real_adaptive_fusion(individual_scores)
                
            elif strategy == RealFusionStrategy.ENSEMBLE_FUSION:
                fused_score = self._real_ensemble_fusion(anat_score, dyn_score)
                
            else:
                logger.warning(f"Estrategia no reconocida: {strategy}, usando weighted_average")
                fused_score = self._real_weighted_average_fusion(anat_score, dyn_score, weights)
            
            # Asegurar rango válido
            fused_score = float(np.clip(fused_score, 0.0, 1.0))
            
            # Decisión final
            decision = fused_score >= self.optimal_threshold
            
            # Calcular confianza
            decision_confidence = self._calculate_real_decision_confidence(
                fused_score, individual_scores, weights
            )
            
            # Crear resultado
            fused_result = RealFusedScore(
                fused_score=fused_score,
                decision=decision,
                confidence=decision_confidence,
                fusion_strategy=strategy,
                individual_scores=individual_scores,
                details={
                    'weights_used': weights,
                    'threshold_used': self.optimal_threshold,
                    'calibrated_scores': {'anatomical': anat_score, 'dynamic': dyn_score},
                    'strategy_name': strategy.value,
                    'is_real_fusion': True
                }
            )
            
            logger.info(f"✓ Fusión completada:")
            logger.info(f"  - Score: {fused_score:.3f}")
            logger.info(f"  - Decisión: {'✓ Aceptado' if decision else '✗ Rechazado'}")
            logger.info(f"  - Confianza: {decision_confidence:.3f}")
            logger.info(f"  - Estrategia: {strategy.value}")
            
            return fused_result
            
        except Exception as e:
            logger.error(f"Error fusionando scores: {e}")
            return RealFusedScore(
                fused_score=0.0,
                decision=False,
                confidence=0.0,
                fusion_strategy=strategy or RealFusionStrategy.WEIGHTED_AVERAGE,
                individual_scores=individual_scores,
                details={'error': str(e), 'is_real_fusion': True}
            )
    
    def _calibrate_real_score(self, score: float, modality: str) -> float:
        """Calibra score de una modalidad específica."""
        try:
            if not self.is_calibrated or modality not in self.real_score_calibrators:
                return score
            
            calibrator = self.real_score_calibrators[modality]
            
            if calibrator.get('identity', False):
                return score
            elif 'scaler' in calibrator:
                return calibrator['scaler'].transform([[score]])[0][0]
            elif 'mean' in calibrator and 'std' in calibrator:
                return (score - calibrator['mean']) / (calibrator['std'] + 1e-8)
            elif 'sigmoid_params' in calibrator:
                a, b = calibrator['sigmoid_params']
                return 1.0 / (1.0 + np.exp(-(a * score + b)))
            else:
                return score
                
        except Exception as e:
            logger.error(f"Error calibrando score de {modality}: {e}")
            return score
    
    def _get_real_fusion_weights(self, individual_scores: RealIndividualScores, 
                                strategy: RealFusionStrategy) -> Dict[str, float]:
        """Obtiene pesos de fusión basados en la estrategia."""
        try:
            if strategy == RealFusionStrategy.ADAPTIVE_FUSION and self.config.use_confidence_weighting:
                total_conf = individual_scores.anatomical_confidence + individual_scores.dynamic_confidence
                if total_conf > 0:
                    anat_weight = individual_scores.anatomical_confidence / total_conf
                    dyn_weight = individual_scores.dynamic_confidence / total_conf
                else:
                    anat_weight = self.optimal_weights['anatomical']
                    dyn_weight = self.optimal_weights['dynamic']
            else:
                anat_weight = self.optimal_weights['anatomical']
                dyn_weight = self.optimal_weights['dynamic']
            
            # Normalizar pesos
            total_weight = anat_weight + dyn_weight
            if total_weight > 0:
                anat_weight /= total_weight
                dyn_weight /= total_weight
            else:
                anat_weight = 0.5
                dyn_weight = 0.5
            
            return {'anatomical': anat_weight, 'dynamic': dyn_weight}
            
        except Exception as e:
            logger.error(f"Error obteniendo pesos: {e}")
            return {'anatomical': 0.5, 'dynamic': 0.5}
    
    def _real_weighted_average_fusion(self, anat_score: float, dyn_score: float,
                                     weights: Dict[str, float]) -> float:
        """Fusión por promedio ponderado."""
        return weights['anatomical'] * anat_score + weights['dynamic'] * dyn_score
    
    def _real_product_rule_fusion(self, anat_score: float, dyn_score: float,
                                 weights: Dict[str, float]) -> float:
        """Fusión por regla del producto."""
        return (anat_score ** weights['anatomical']) * (dyn_score ** weights['dynamic'])
    
    def _real_svm_fusion(self, anat_score: float, dyn_score: float) -> float:
        """Fusión usando SVM entrenado."""
        try:
            if 'svm' in self.real_fusion_models:
                features = np.array([[anat_score, dyn_score]])
                decision_scores = self.real_fusion_models['svm'].decision_function(features)
                return 1.0 / (1.0 + np.exp(-decision_scores[0]))
            else:
                return 0.5 * anat_score + 0.5 * dyn_score
        except Exception as e:
            logger.error(f"Error en SVM fusion: {e}")
            return 0.5 * anat_score + 0.5 * dyn_score
    
    def _real_neural_fusion(self, anat_score: float, dyn_score: float) -> float:
        """Fusión usando red neuronal entrenada."""
        try:
            if 'neural' in self.real_fusion_models:
                features = np.array([[anat_score, dyn_score]])
                return self.real_fusion_models['neural'].predict_proba(features)[0, 1]
            else:
                return 0.5 * anat_score + 0.5 * dyn_score
        except Exception as e:
            logger.error(f"Error en neural fusion: {e}")
            return 0.5 * anat_score + 0.5 * dyn_score
    
    def _real_logistic_fusion(self, anat_score: float, dyn_score: float) -> float:
        """Fusión usando regresión logística entrenada."""
        try:
            if 'logistic' in self.real_fusion_models:
                features = np.array([[anat_score, dyn_score]])
                return self.real_fusion_models['logistic'].predict_proba(features)[0, 1]
            else:
                return 0.5 * anat_score + 0.5 * dyn_score
        except Exception as e:
            logger.error(f"Error en logistic fusion: {e}")
            return 0.5 * anat_score + 0.5 * dyn_score
    
    def _real_adaptive_fusion(self, individual_scores: RealIndividualScores) -> float:
        """Fusión adaptativa basada en confianza."""
        try:
            anat_score = individual_scores.anatomical_score
            dyn_score = individual_scores.dynamic_score
            anat_conf = individual_scores.anatomical_confidence
            dyn_conf = individual_scores.dynamic_confidence
            
            # Si una modalidad tiene baja confianza, dar más peso a la otra
            if anat_conf < 0.3 and dyn_conf > 0.7:
                return 0.2 * anat_score + 0.8 * dyn_score
            elif dyn_conf < 0.3 and anat_conf > 0.7:
                return 0.8 * anat_score + 0.2 * dyn_score
            else:
                weights = self._get_real_fusion_weights(individual_scores, RealFusionStrategy.ADAPTIVE_FUSION)
                return weights['anatomical'] * anat_score + weights['dynamic'] * dyn_score
                
        except Exception as e:
            logger.error(f"Error en adaptive fusion: {e}")
            return 0.5 * individual_scores.anatomical_score + 0.5 * individual_scores.dynamic_score
    
    def _real_ensemble_fusion(self, anat_score: float, dyn_score: float) -> float:
        """Fusión ensemble usando múltiples estrategias."""
        try:
            weights = {'anatomical': self.optimal_weights['anatomical'], 'dynamic': self.optimal_weights['dynamic']}
            
            fusion_results = []
            
            # Weighted average
            result1 = self._real_weighted_average_fusion(anat_score, dyn_score, weights)
            fusion_results.append(result1)
            
            # Product rule
            result2 = self._real_product_rule_fusion(anat_score, dyn_score, weights)
            fusion_results.append(result2)
            
            # SVM si disponible
            if 'svm' in self.real_fusion_models:
                result3 = self._real_svm_fusion(anat_score, dyn_score)
                fusion_results.append(result3)
            
            # Neural si disponible
            if 'neural' in self.real_fusion_models:
                result4 = self._real_neural_fusion(anat_score, dyn_score)
                fusion_results.append(result4)
            
            return float(np.mean(fusion_results))
            
        except Exception as e:
            logger.error(f"Error en ensemble fusion: {e}")
            return 0.5 * anat_score + 0.5 * dyn_score
    
    def _calculate_real_decision_confidence(self, fused_score: float, 
                                           individual_scores: RealIndividualScores,
                                           weights: Dict[str, float]) -> float:
        """Calcula confianza en la decisión fusionada."""
        try:
            # Confianza por score
            score_confidence = fused_score if fused_score >= self.optimal_threshold else (1 - fused_score)
            
            # Confianza promedio de modalidades
            modal_confidence = (
                weights['anatomical'] * individual_scores.anatomical_confidence +
                weights['dynamic'] * individual_scores.dynamic_confidence
            )
            
            # Consistencia entre modalidades
            score_diff = abs(individual_scores.anatomical_score - individual_scores.dynamic_score)
            consistency_confidence = max(0.0, 1.0 - score_diff)
            
            # Combinar factores
            overall_confidence = (
                0.4 * score_confidence +
                0.3 * modal_confidence +
                0.3 * consistency_confidence
            )
            
            return float(np.clip(overall_confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculando confianza de decisión: {e}")
            return 0.5
        
    def train_real_fusion_models(self, real_training_data: List[Tuple[RealIndividualScores, bool]]) -> bool:
        """Entrena modelos de fusión."""
        try:
            if len(real_training_data) < 10:
                logger.error(f"Datos insuficientes: {len(real_training_data)} < 10")
                return False
            
            logger.info(f"Entrenando modelos con {len(real_training_data)} muestras...")
            
            # Preparar datos
            X = []
            y = []
            
            for scores, label in real_training_data:
                X.append([scores.anatomical_score, scores.dynamic_score])
                y.append(1 if label else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Validar datos
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.error("Datos contienen valores inválidos")
                return False
            
            # Entrenar SVM
            try:
                svm_model = SVC(kernel='rbf', probability=True, random_state=42)
                svm_model.fit(X, y)
                self.real_fusion_models['svm'] = svm_model
                logger.info("✓ Modelo SVM entrenado")
            except Exception as e:
                logger.error(f"Error entrenando SVM: {e}")
            
            # Entrenar Red Neuronal
            try:
                neural_model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
                neural_model.fit(X, y)
                self.real_fusion_models['neural'] = neural_model
                logger.info("✓ Modelo Neural entrenado")
            except Exception as e:
                logger.error(f"Error entrenando red neuronal: {e}")
            
            # Entrenar Regresión Logística
            try:
                logistic_model = LogisticRegression(random_state=42)
                logistic_model.fit(X, y)
                self.real_fusion_models['logistic'] = logistic_model
                logger.info("✓ Modelo Logístico entrenado")
            except Exception as e:
                logger.error(f"Error entrenando regresión logística: {e}")
            
            # Entrenar Random Forest
            try:
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X, y)
                self.real_fusion_models['random_forest'] = rf_model
                logger.info("✓ Modelo Random Forest entrenado")
            except Exception as e:
                logger.error(f"Error entrenando random forest: {e}")
            
            self.is_trained = True
            logger.info(f"✓ Modelos de fusión entrenados: {list(self.real_fusion_models.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando modelos: {e}")
            return False
    
    def optimize_real_fusion_weights(self, real_validation_data: List[Tuple[RealIndividualScores, bool]]) -> Dict[str, float]:
        """Optimiza pesos de fusión usando solo datos REALES de validación."""
        try:
            if len(real_validation_data) < 5:
                logger.error(f"Datos insuficientes para optimización: {len(real_validation_data)} < 5")
                return self.optimal_weights
            
            logger.info(f"Optimizando pesos con {len(real_validation_data)} muestras REALES...")
            
            best_eer = float('inf')
            best_weights = self.optimal_weights.copy()
            
            # Búsqueda en grilla
            weight_resolution = 0.05
            
            for anat_weight in np.arange(0.1, 1.0, weight_resolution):
                dyn_weight = 1.0 - anat_weight
                
                predictions = []
                true_labels = []
                
                for scores, true_label in real_validation_data:
                    fused_score = (anat_weight * scores.anatomical_score + 
                                 dyn_weight * scores.dynamic_score)
                    predictions.append(fused_score)
                    true_labels.append(1 if true_label else 0)
                
                # Calcular EER
                try:
                    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
                    
                    fnr = 1 - tpr
                    
                    # Validación antes del cálculo
                    if len(np.unique(true_labels)) < 2:
                        eer_threshold = 0.5
                        eer = 0.5
                        logger.warning("Solo una clase, usando EER por defecto")
                    else:
                        try:
                            
                            eer_differences = np.absolute(fnr - fpr)
                            eer_idx = np.nanargmin(eer_differences)
                            eer_threshold = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.5
                            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
                        except (ValueError, IndexError):
                            eer_threshold = 0.5
                            eer = 0.5
                            logger.warning("Error calculando EER, usando valores por defecto")
                    
                    if eer < best_eer:
                        best_eer = eer
                        best_weights = {'anatomical': anat_weight, 'dynamic': dyn_weight}
                        self.optimal_threshold = eer_threshold
                        
                except Exception as e:
                    logger.warning(f"Error calculando EER para pesos {anat_weight:.2f}/{dyn_weight:.2f}: {e}")
                    continue
            
            # Actualizar pesos optimizados
            self.optimal_weights = best_weights
            
            logger.info(f"✓ Pesos optimizados:")
            logger.info(f"  - Anatómico: {best_weights['anatomical']:.3f}")
            logger.info(f"  - Dinámico: {best_weights['dynamic']:.3f}")
            logger.info(f"  - EER óptimo: {best_eer:.4f}")
            logger.info(f"  - Umbral óptimo: {self.optimal_threshold:.3f}")
            
            return best_weights
            
        except Exception as e:
            logger.error(f"Error optimizando pesos: {e}")
            return self.optimal_weights
    
    def calibrate_real_scores(self, real_calibration_data: List[Tuple[RealIndividualScores, bool]]) -> bool:
        """Calibra scores individuales usando solo datos REALES."""
        try:
            if len(real_calibration_data) < 10:
                logger.error("Datos insuficientes para calibración")
                return False
            
            logger.info(f"Calibrando scores con {len(real_calibration_data)} muestras REALES...")
            
            # Separar scores por modalidad
            anat_scores = []
            dyn_scores = []
            labels = []
            
            for scores, label in real_calibration_data:
                anat_scores.append(scores.anatomical_score)
                dyn_scores.append(scores.dynamic_score)
                labels.append(1 if label else 0)
            
            anat_scores = np.array(anat_scores)
            dyn_scores = np.array(dyn_scores)
            labels = np.array(labels)
            
            # Calibrar scores anatómicos
            self.real_score_calibrators['anatomical'] = self._fit_real_score_calibrator(anat_scores, labels)
            
            # Calibrar scores dinámicos
            self.real_score_calibrators['dynamic'] = self._fit_real_score_calibrator(dyn_scores, labels)
            
            self.is_calibrated = True
            logger.info("✓ Calibración de scores completada")
            
            return True
            
        except Exception as e:
            logger.error(f"Error calibrando scores: {e}")
            return False
    
    def _fit_real_score_calibrator(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Ajusta calibrador para una modalidad específica."""
        try:
            calibrator = {}
            
            if self.config.calibration_method == RealScoreCalibration.MIN_MAX:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                scaler.fit(scores.reshape(-1, 1))
                calibrator['scaler'] = scaler
                
            elif self.config.calibration_method == RealScoreCalibration.Z_SCORE:
                calibrator['mean'] = np.mean(scores)
                calibrator['std'] = np.std(scores)
                
            elif self.config.calibration_method == RealScoreCalibration.SIGMOID:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression()
                lr.fit(scores.reshape(-1, 1), labels)
                a = lr.coef_[0, 0]
                b = -lr.intercept_[0] / a if a != 0 else 0
                calibrator['sigmoid_params'] = (a, b)
                
            else:
                calibrator['identity'] = True
            
            return calibrator
            
        except Exception as e:
            logger.error(f"Error ajustando calibrador: {e}")
            return {'identity': True}
        
    def evaluate_real_fusion_system(self, real_test_data: List[Tuple[RealIndividualScores, bool]]) -> RealFusionMetrics:
        """Evalúa el sistema de fusión."""
        try:
            if len(real_test_data) < 5:
                logger.error("Datos insuficientes para evaluación")
                return None
            
            logger.info(f"Evaluando sistema con {len(real_test_data)} muestras...")
            
            # Predecir con fusión
            fused_predictions = []
            true_labels = []
            anat_predictions = []
            dyn_predictions = []
            confidence_scores = []
            
            for scores, true_label in real_test_data:
                # Fusión principal
                fused_result = self.fuse_real_scores(scores)
                fused_predictions.append(fused_result.fused_score)
                confidence_scores.append(fused_result.confidence)
                
                # Predicciones individuales
                anat_predictions.append(scores.anatomical_score)
                dyn_predictions.append(scores.dynamic_score)
                
                true_labels.append(1 if true_label else 0)
            
            # Convertir a arrays
            fused_pred = np.array(fused_predictions)
            true_labels = np.array(true_labels)
            anat_pred = np.array(anat_predictions)
            dyn_pred = np.array(dyn_predictions)
            confidence_scores = np.array(confidence_scores)
            
            # Calcular métricas principales
            fusion_metrics = self._calculate_real_comprehensive_metrics(fused_pred, true_labels)
            
            # Métricas individuales
            anat_metrics = self._calculate_real_comprehensive_metrics(anat_pred, true_labels)
            dyn_metrics = self._calculate_real_comprehensive_metrics(dyn_pred, true_labels)
            
            # Calcular mejora por fusión
            best_individual_eer = min(anat_metrics['eer'], dyn_metrics['eer'])
            fusion_improvement = max(0, best_individual_eer - fusion_metrics['eer'])
            
            # Métricas adicionales
            calibration_quality = self._evaluate_real_calibration_quality(fused_pred, confidence_scores, true_labels)
            fusion_consistency = self._calculate_real_fusion_consistency(anat_pred, dyn_pred)
            
            # Crear métricas finales
            final_metrics = RealFusionMetrics(
                far=fusion_metrics['far'],
                frr=fusion_metrics['frr'],
                eer=fusion_metrics['eer'],
                auc_score=fusion_metrics['auc'],
                accuracy=fusion_metrics['accuracy'],
                precision=fusion_metrics['precision'],
                recall=fusion_metrics['recall'],
                f1_score=fusion_metrics['f1'],
                fusion_improvement=fusion_improvement,
                anatomical_weight=self.optimal_weights['anatomical'],
                dynamic_weight=self.optimal_weights['dynamic'],
                optimal_threshold=self.optimal_threshold,
                anatomical_metrics=anat_metrics,
                dynamic_metrics=dyn_metrics,
                calibration_quality=calibration_quality,
                fusion_consistency=fusion_consistency,
                decision_confidence_avg=np.mean(confidence_scores)
            )
            
            self.fusion_metrics = final_metrics
            
            logger.info("✓ Evaluación completada:")
            logger.info(f"  - EER: {final_metrics.eer:.4f}")
            logger.info(f"  - AUC: {final_metrics.auc_score:.4f}")
            logger.info(f"  - Precisión: {final_metrics.accuracy:.4f}")
            logger.info(f"  - Mejora: {final_metrics.fusion_improvement:.4f}")
            logger.info(f"  - Confianza promedio: {final_metrics.decision_confidence_avg:.3f}")
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error evaluando sistema: {e}")
            return None
    
    def _calculate_real_comprehensive_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """Calcula métricas completas usando datos REALES."""
        try:
            # ROC curve
            fpr, tpr, thresholds = roc_curve(true_labels, predictions)
            auc_score = auc(fpr, tpr)
            
            # EER
            fnr = 1 - tpr
            eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
            eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            
            # Predicciones binarias
            binary_predictions = (predictions >= eer_threshold).astype(int)
            
            # Métricas adicionales
            accuracy = accuracy_score(true_labels, binary_predictions)
            
            # FAR y FRR
            tn, fp, fn, tp = confusion_matrix(true_labels, binary_predictions).ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'far': far,
                'frr': frr,
                'eer': eer,
                'auc': auc_score,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}")
            return {
                'far': 0.5, 'frr': 0.5, 'eer': 0.5, 'auc': 0.5,
                'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5
            }
    
    def _evaluate_real_calibration_quality(self, predictions: np.ndarray, 
                                          confidence_scores: np.ndarray, 
                                          true_labels: np.ndarray) -> float:
        """Evalúa calidad de calibración."""
        try:
            n_bins = min(10, len(predictions) // 2)
            if n_bins < 2:
                return 0.5
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    confidence_in_bin = confidence_scores[in_bin].mean()
                    accuracy_in_bin = true_labels[in_bin].mean()
                    calibration_error += np.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            quality_score = max(0, 1 - calibration_error)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error evaluando calibración: {e}")
            return 0.5
    
    def _calculate_real_fusion_consistency(self, anat_pred: np.ndarray, dyn_pred: np.ndarray) -> float:
        """Calcula consistencia entre modalidades usando datos REALES."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                correlation = np.corrcoef(anat_pred, dyn_pred)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            
            mean_abs_diff = np.mean(np.abs(anat_pred - dyn_pred))
            consistency_score = max(0, 1 - mean_abs_diff)
            
            overall_consistency = (abs(correlation) + consistency_score) / 2
            
            return overall_consistency
            
        except Exception as e:
            logger.error(f"Error calculando consistencia: {e}")
            return 0.5
    
    def get_real_fusion_summary(self) -> Dict[str, Any]:
        """Obtiene resumen completo del sistema de fusión REAL."""
        try:
            summary = {
                "status": "real_system",
                "version": "2.0",
                "is_real_data": True,
                "no_synthetic_data": True,
                "config": {
                    "fusion_strategy": self.config.fusion_strategy.value,
                    "calibration_method": self.config.calibration_method.value,
                    "anatomical_weight": self.optimal_weights['anatomical'],
                    "dynamic_weight": self.optimal_weights['dynamic'],
                    "decision_threshold": self.optimal_threshold
                },
                "training": {
                    "is_trained": self.is_trained,
                    "is_calibrated": self.is_calibrated,
                    "is_initialized": self.is_initialized,
                    "networks_initialized": self.anatomical_network is not None and self.dynamic_network is not None,
                    "available_fusion_models": list(self.real_fusion_models.keys()),
                    "calibrated_modalities": list(self.real_score_calibrators.keys())
                },
                "performance": {}
            }
            
            if self.fusion_metrics:
                summary["performance"] = {
                    "eer": self.fusion_metrics.eer,
                    "auc_score": self.fusion_metrics.auc_score,
                    "accuracy": self.fusion_metrics.accuracy,
                    "fusion_improvement": self.fusion_metrics.fusion_improvement,
                    "calibration_quality": self.fusion_metrics.calibration_quality,
                    "fusion_consistency": self.fusion_metrics.fusion_consistency,
                    "confidence_avg": self.fusion_metrics.decision_confidence_avg
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen: {e}")
            return {
                "status": "error",
                "error": str(e),
                "is_real_data": True
            }
    def save_real_fusion_system(self, filepath: Optional[str] = None) -> bool:
        """Guarda el sistema de fusión completo."""
        try:
            if filepath is None:
                models_dir = Path(get_config('paths.models', 'biometric_data/models'))
                models_dir.mkdir(exist_ok=True)
                filepath = models_dir / 'real_score_fusion_system.pkl'
            
            save_data = {
                'config': self.config,
                'optimal_weights': self.optimal_weights,
                'optimal_threshold': self.optimal_threshold,
                'real_fusion_models': self.real_fusion_models,
                'real_score_calibrators': self.real_score_calibrators,
                'is_trained': self.is_trained,
                'is_calibrated': self.is_calibrated,
                'is_initialized': self.is_initialized,
                'fusion_metrics': self.fusion_metrics,
                'training_history': self.training_history,
                'version': '2.0',
                'is_real_data': True,
                'no_synthetic_data': True
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"✓ Sistema de fusión guardado: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando sistema de fusión: {e}")
            return False
    
    def load_real_fusion_system(self, filepath: str) -> bool:
        """Carga un sistema de fusión previamente entrenado."""
        try:
            if not Path(filepath).exists():
                logger.error(f"Archivo no encontrado: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            if not save_data.get('is_real_data', False):
                logger.error("No es un sistema REAL")
                return False
            
            self.config = save_data['config']
            self.optimal_weights = save_data['optimal_weights']
            self.optimal_threshold = save_data['optimal_threshold']
            self.real_fusion_models = save_data['real_fusion_models']
            self.real_score_calibrators = save_data['real_score_calibrators']
            self.is_trained = save_data['is_trained']
            self.is_calibrated = save_data['is_calibrated']
            self.is_initialized = save_data.get('is_initialized', False)
            self.fusion_metrics = save_data['fusion_metrics']
            self.training_history = save_data['training_history']
            
            logger.info(f"✓ Sistema cargado: {filepath}")
            logger.info(f"  - Versión: {save_data.get('version', 'unknown')}")
            logger.info(f"  - Modelos: {len(self.real_fusion_models)}")
            logger.info(f"  - Entrenado: {self.is_trained}")
            logger.info(f"  - Calibrado: {self.is_calibrated}")
            logger.info(f"  - Inicializado: {self.is_initialized}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando sistema: {e}")
            return False


# ===== INSTANCIA GLOBAL =====
_real_fusion_system_instance = None

def get_real_score_fusion_system() -> RealScoreFusionSystem:
    """Obtiene instancia global del sistema de fusión REAL."""
    global _real_fusion_system_instance
    
    if _real_fusion_system_instance is None:
        _real_fusion_system_instance = RealScoreFusionSystem()
    
    return _real_fusion_system_instance


# Alias para compatibilidad
ScoreFusionSystem = RealScoreFusionSystem
get_score_fusion_system = get_real_score_fusion_system
IndividualScores = RealIndividualScores
FusedScore = RealFusedScore
FusionStrategy = RealFusionStrategy
ScoreCalibration = RealScoreCalibration
WeightOptimization = RealWeightOptimization

