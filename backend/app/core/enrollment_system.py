"""
MÓDULO 14: ENROLLMENT_SYSTEM
Sistema completo de registro/enrollment biométrico con modo Bootstrap
"""

import cv2
import numpy as np
import time
import json
import uuid
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import threading

# ====================================================================
# IMPORTS con 'app.core.'
# ====================================================================

try:
    from app.core.config_manager import get_config, get_logger, log_error, log_info
    from app.core.camera_manager import get_camera_manager, release_camera
    from app.core.mediapipe_processor import get_mediapipe_processor, ProcessingResult
    from app.core.quality_validator import get_quality_validator, QualityAssessment
    from app.core.reference_area_manager import get_reference_area_manager
    from app.core.anatomical_features_extractor import get_anatomical_features_extractor, AnatomicalFeatureVector
    from app.core.dynamic_features_extractor import get_real_dynamic_features_extractor, DynamicFeatureVector
    from app.core.sequence_manager import get_sequence_manager, SequenceState
    from app.core.siamese_anatomical_network import get_real_siamese_anatomical_network
    from app.core.siamese_dynamic_network import get_real_siamese_dynamic_network
    from app.core.feature_preprocessor import get_real_feature_preprocessor
    from app.core.score_fusion_system import get_real_score_fusion_system
    from app.core.biometric_database import (
        get_biometric_database, 
        BiometricTemplate, 
        UserProfile, 
        TemplateType
    )
    from app.core.roi_normalization import get_roi_normalization_system
    from app.core.visual_feedback import get_visual_feedback_manager
    
except ImportError as e:
    logging.error(f"Error importando módulos: {e}")
    # Fallback para testing standalone
    def get_config(key, default=None): return default
    def get_logger(): return logging.getLogger(__name__)
    def log_error(msg, exc=None): logging.error(f"ERROR: {msg}")
    def log_info(msg): logging.info(f"INFO: {msg}")

# Logger
logger = logging.getLogger(__name__)


# ====================================================================
# ENUMERACIONES
# ====================================================================

class EnrollmentPhase(Enum):
    """Fases del proceso de enrollment."""
    INITIALIZATION = "initialization"
    USER_SETUP = "user_setup"
    SEQUENCE_DEFINITION = "sequence_definition"
    SAMPLE_COLLECTION = "sample_collection"
    QUALITY_VALIDATION = "quality_validation"
    TEMPLATE_GENERATION = "template_generation"
    DATABASE_STORAGE = "database_storage"
    ENROLLMENT_COMPLETE = "enrollment_complete"


class EnrollmentStatus(Enum):
    """Estados del enrollment."""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    COLLECTING_SAMPLES = "collecting_samples"
    VALIDATING_QUALITY = "validating_quality"
    GENERATING_TEMPLATES = "generating_templates"
    STORING_DATA = "storing_data"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SampleType(Enum):
    """Tipos de muestras biométricas REALES."""
    ANATOMICAL = "anatomical"
    DYNAMIC = "dynamic"
    COMBINED = "combined"


# ====================================================================
# DATACLASSES
# ====================================================================

@dataclass
class RealEnrollmentSample:
    """Muestra de enrollment."""
    sample_id: str
    user_id: str
    sample_type: SampleType
    gesture_name: str
    
    anatomical_features: Optional[AnatomicalFeatureVector] = None
    dynamic_features: Optional[DynamicFeatureVector] = None
    
    timestamp: float = field(default_factory=time.time)
    capture_duration: float = 0.0
    frame_count: int = 0
    
    quality_assessment: Optional[QualityAssessment] = None
    confidence: float = 0.0
    
    anatomical_embedding: Optional[np.ndarray] = None
    dynamic_embedding: Optional[np.ndarray] = None
    
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    temporal_sequence: Optional[np.ndarray] = None
    sequence_length: int = 0
    has_temporal_data: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RealEnrollmentConfig:
    """Configuración para enrollment REAL."""
    samples_per_gesture: int = 10
    min_samples_per_gesture: int = 7
    max_samples_per_gesture: int = 15
    
    quality_threshold: float = 0.8
    min_confidence: float = 0.7
    min_stability_frames: int = 10
    
    sample_timeout: float = 120.0
    session_timeout: float = 3600.0
    capture_interval: float = 0.5
    
    require_all_gestures: bool = True
    enable_quality_check: bool = True
    enable_duplicate_check: bool = True
    duplicate_threshold: float = 0.95
    
    template_fusion_strategy: str = "average"
    enable_template_optimization: bool = True
    embedding_dimension_check: bool = True
    
    show_preview: bool = True
    show_quality_feedback: bool = True
    save_enrollment_video: bool = False


@dataclass 
class RealEnrollmentSession:
    """Sesión de enrollment."""
    session_id: str
    user_id: str
    username: str
    gesture_sequence: List[str]
    
    status: EnrollmentStatus = EnrollmentStatus.NOT_STARTED
    current_phase: EnrollmentPhase = EnrollmentPhase.INITIALIZATION
    current_gesture: str = ""
    current_gesture_index: int = 0
    
    total_samples_needed: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    
    samples: List[RealEnrollmentSample] = field(default_factory=list)
    
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    last_sample_time: float = field(default_factory=time.time)
    last_capture_time: float = 0.0 

    frames_processed: int = 0
    total_frames_captured: int = 0
    
    progress_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    
    # Fase 2: Captura dinámica de secuencia fluida
    dynamic_phase_completed: bool = False
    dynamic_sequence_template_id: Optional[str] = None
    fluid_sequence_captured: bool = False
    
    #is_bootstrap: bool = False
    
    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def progress_percentage(self) -> float:
        if self.total_samples_needed == 0:
            return 0.0
        return (self.successful_samples / self.total_samples_needed) * 100

    def add_sample(self, sample: 'RealEnrollmentSample') -> None:
        """Agrega una muestra a la sesión."""
        try:
            if sample and sample.is_valid:
                self.samples.append(sample)
                self.successful_samples += 1
                print(f"✅ Muestra agregada: {sample.sample_id}")
                print(f"   Progreso: {self.successful_samples}/{self.total_samples_needed}")
            else:
                self.failed_samples += 1
                print(f"❌ Muestra inválida rechazada")
        except Exception as e:
            print(f"Error agregando muestra: {e}")
            self.failed_samples += 1
    
    def is_current_gesture_complete(self, samples_per_gesture: int) -> bool:
        """Verifica si el gesto actual tiene suficientes muestras."""
        current_gesture_samples = [s for s in self.samples if s.gesture_name == self.current_gesture]
        return len(current_gesture_samples) >= samples_per_gesture
    
    
        
# ====================================================================
# CONTROLADOR DE CALIDAD
# ====================================================================

class RealQualityController:
    """Controlador de calidad para enrollment."""
    
    def __init__(self, config: RealEnrollmentConfig):
        """Inicializa controlador con validación."""
        self.config = config
        self.logger = get_logger()
        
        self.quality_validator = get_quality_validator()
        self.area_manager = get_reference_area_manager()
        
        print("RealQualityController inicializado")
    
    def validate_sample_quality(self, sample: RealEnrollmentSample, bootstrap_mode: bool = False) -> Tuple[bool, List[str]]:
        """
        Valida calidad de muestra REAL con soporte Bootstrap.
        
        Args:
            sample: Muestra REAL a validar
            bootstrap_mode: Si está en modo bootstrap (más permisivo)
            
        Returns:
            (es_válida, lista_errores)
        """
        try:
            mode_text = "BOOTSTRAP" if bootstrap_mode else "NORMAL"
            print(f"Validando calidad de muestra {sample.sample_id} (modo {mode_text})")
            
            errors = []
            
            if not sample:
                errors.append("Sample es None")
                return False, errors
            
            if not sample.quality_assessment:
                errors.append("Falta evaluación de calidad")
            else:
                quality_threshold = 50.0 if bootstrap_mode else self.config.quality_threshold
                if sample.quality_assessment.quality_score < quality_threshold:
                    errors.append(f"Calidad insuficiente: {sample.quality_assessment.quality_score:.3f} < {quality_threshold}")
            
            confidence_threshold = 0.4 if bootstrap_mode else self.config.min_confidence
            if sample.confidence < confidence_threshold:
                errors.append(f"Confianza insuficiente: {sample.confidence:.3f} < {confidence_threshold}")
            
            if sample.sample_type in [SampleType.ANATOMICAL, SampleType.COMBINED]:
                if sample.anatomical_features is None:
                    errors.append("Faltan características anatómicas")
                elif not self._validate_anatomical_features_real(sample.anatomical_features):
                    errors.append("Características anatómicas inválidas")
            
            if sample.sample_type in [SampleType.DYNAMIC, SampleType.COMBINED]:
                if sample.dynamic_features is None:
                    if bootstrap_mode:
                        print("Características dinámicas ausentes - OK en bootstrap")
                    else:
                        errors.append("Faltan características dinámicas")
                elif not self._validate_dynamic_features_real(sample.dynamic_features):
                    if bootstrap_mode:
                        print("Características dinámicas inválidas - tolerado en bootstrap")
                    else:
                        errors.append("Características dinámicas inválidas")
            
            if bootstrap_mode:
                print("🔧 Bootstrap: NO validando embeddings")
                
                if sample.anatomical_embedding is not None:
                    print("⚠️ Bootstrap tiene embedding anatómico")
                
                if sample.dynamic_embedding is not None:
                    print("⚠️ Bootstrap tiene embedding dinámico")
                    
            else:
                if sample.anatomical_embedding is not None:
                    if not self._validate_real_embedding(sample.anatomical_embedding, "anatomical"):
                        errors.append("Embedding anatómico inválido")
                else:
                    errors.append("Falta embedding anatómico en modo normal")
                
                if sample.dynamic_embedding is not None:
                    if not self._validate_real_embedding(sample.dynamic_embedding, "dynamic"):
                        errors.append("Embedding dinámico inválido")
                else:
                    print("Embedding dinámico ausente - OK")
            
            is_valid = len(errors) == 0
            sample.is_valid = is_valid
            sample.validation_errors = errors
            
            if is_valid:
                print(f"✅ Muestra {sample.sample_id} validada exitosamente (modo {mode_text})")
            else:
                print(f"❌ Muestra {sample.sample_id} falló validación {mode_text}: {errors}")
            
            return is_valid, errors
            
        except Exception as e:
            print(f"Error validando muestra: {e}")
            return False, [f"Error: {str(e)}"]
    
    def _validate_anatomical_features_real(self, features: AnatomicalFeatureVector) -> bool:
        """Valida características anatómicas REALES."""
        try:
            if features is None or features.complete_vector is None:
                return False
            
            vector = features.complete_vector
            
            if vector.shape[0] != 180:
                print(f"Dimensión anatómica incorrecta: {vector.shape[0]}")
                return False
            
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                print("Características anatómicas con NaN o infinitos")
                return False
            
            if np.allclose(vector, 0.0):
                print("Características anatómicas son todas cero")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validando caracteristicas anatómicas: {e}")
            return False
    
    def _validate_dynamic_features_real(self, features: DynamicFeatureVector) -> bool:
        """Valida características dinámicas REALES."""
        try:
            if features is None or features.complete_vector is None:
                return False
            
            vector = features.complete_vector
            
            if vector.shape[0] != 320:
                print(f"Dimensión dinámica incorrecta: {vector.shape[0]} != 320")
                return False
            
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                print("Características dinámicas con NaN")
                return False
            
            if np.allclose(vector, 0.0):
                print("Características dinámicas son todas cero")
                return False
            
            if not self._validate_temporal_components_real(features):
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validando dinámicas: {e}")
            return False
    
    def _validate_temporal_components_real(self, features: DynamicFeatureVector) -> bool:
        """Valida componentes temporales."""
        try:
            if hasattr(features, 'velocity_features') and features.velocity_features is not None:
                if np.var(features.velocity_features) < 1e-6:
                    print("Características de velocidad sin variación temporal")
                    return False
            
            if hasattr(features, 'acceleration_features') and features.acceleration_features is not None:
                if np.var(features.acceleration_features) < 1e-6:
                    print("Características de aceleración sin variación temporal")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error validando componentes temporales: {e}")
            return False
    
    def _validate_real_embedding(self, embedding: np.ndarray, embedding_type: str) -> bool:
        """Valida embedding generado por redes entrenadas."""
        try:
            if embedding is None:
                return False
            
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                print(f"Embedding {embedding_type} con NaN")
                return False
            
            if np.allclose(embedding, 0.0):
                print(f"Embedding {embedding_type} es vector cero")
                return False
            
            expected_dims = {"anatomical": 64, "dynamic": 128}
            
            if embedding_type in expected_dims:
                if embedding.shape[0] != expected_dims[embedding_type]:
                    print(f"Dimensión de embedding {embedding_type} incorrecta: {embedding.shape[0]} != {expected_dims[embedding_type]}")

                    return False
            
            magnitude = np.linalg.norm(embedding)
            if magnitude < 0.1 or magnitude > 100.0:
                print(f"Magnitud {embedding_type} fuera de rango: {magnitude}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validando embedding {embedding_type}: {e}")
            return False
    
    def get_quality_feedback_real(self, sample: RealEnrollmentSample) -> Dict[str, str]:
        """Obtiene feedback de calidad REAL."""
        try:
            feedback = {}
            
            if not sample.quality_assessment:
                feedback["status"] = "Sin evaluación de calidad"
                return feedback
            
            assessment = sample.quality_assessment
            
            if assessment.quality_score >= self.config.quality_threshold:
                feedback["quality"] = f"Calidad excelente: {assessment.quality_score:.2f}"
            else:
                feedback["quality"] = f"Mejorar calidad: {assessment.quality_score:.2f}"
            
            if hasattr(assessment, 'hand_size') and assessment.hand_size:
                if assessment.hand_size.distance_status == "muy_lejos":
                    feedback["distance"] = "Acerca más la mano"
                elif assessment.hand_size.distance_status == "muy_cerca":
                    feedback["distance"] = "Aleja la mano"
                else:
                    feedback["distance"] = "Distancia perfecta"
            
            if hasattr(assessment, 'movement') and assessment.movement:
                if assessment.movement.is_moving:
                    feedback["movement"] = "Mantén quieta"
                elif not assessment.movement.is_stable:
                    feedback["stability"] = f"Estabilizando: {assessment.movement.stable_frames}"
                else:
                    feedback["stability"] = "Perfectamente estable"
            
            if sample.confidence >= self.config.min_confidence:
                feedback["confidence"] = f"Deteccion confiable: {sample.confidence:.2f}"
            else:
                feedback["confidence"] = f"Mejorar gesto: {sample.confidence:.2f}"
            
            return feedback
            
        except Exception as e:
            print(f"Error generando feedback: {e}")
            return {"error": "Error generando feedback"}
        
# ====================================================================
# GENERADOR DE TEMPLATES
# ====================================================================

class RealTemplateGenerator:
    """Generador de templates biométricos."""
    
    def __init__(self, config: RealEnrollmentConfig):
        """Inicializa generador con redes entrenadas."""
        self.config = config
        self.logger = get_logger()
        
        self.anatomical_network = get_real_siamese_anatomical_network()
        self.dynamic_network = get_real_siamese_dynamic_network()
        
        self.preprocessor = get_real_feature_preprocessor()
        
        print("RealTemplateGenerator inicializado con redes entrenadas")
     
    def generate_real_templates(self, samples: List[RealEnrollmentSample], user_id: str, bootstrap_mode: bool = False) -> Dict[str, List[np.ndarray]]:
        """
        Genera templates biométricos REALES con soporte Bootstrap.
        
        Args:
            samples: Lista de muestras REALES validadas
            user_id: ID del usuario
            bootstrap_mode: Si está en modo bootstrap
            
        Returns:
            Diccionario con templates {tipo: [embeddings]}
        """
        try:
            mode_text = "BOOTSTRAP" if bootstrap_mode else "NORMAL"
            print(f"Generando templates para {user_id} con {len(samples)} muestras (modo {mode_text})")
            
            templates = {
                'anatomical': [],
                'dynamic': []
            }
            
            if bootstrap_mode:
                print("MODO BOOTSTRAP: Guardando SIN embeddings")
                print("   Embeddings se generarán después del entrenamiento de redes")
                
                valid_samples = [s for s in samples if s.is_valid]
                
                anatomical_count = 0
                dynamic_count = 0
                
                for sample in valid_samples:
                    if sample.anatomical_features and sample.sample_type in [SampleType.ANATOMICAL, SampleType.COMBINED]:
                        anatomical_count += 1
                        print(f"Caracteristicas anatómicas guardadas: {sample.sample_id}")
                    
                    if sample.dynamic_features and sample.sample_type in [SampleType.DYNAMIC, SampleType.COMBINED]:
                        dynamic_count += 1
                        print(f"Caracteristicas dinámicas guardadas: {sample.sample_id}")
                
                print(f"Bootstrap procesadas:")
                print(f"   Caracteristicas anatómicas: {anatomical_count}")
                print(f"   Caracteristicas dinámicas: {dynamic_count}")
                
                return templates
            
            # MODO NORMAL: Generar embeddings (redes entrenadas)
            if not self.anatomical_network.is_trained:
                print("Red anatómica no entrenada en modo normal")
                return templates
            
            if not self.dynamic_network.is_trained:
                print("Red dinámica no entrenada en modo normal - continuando solo anatomica")
            
            valid_samples = [s for s in samples if s.is_valid]
            print(f"Procesando {len(valid_samples)} muestras validas de {len(samples)} totales")

            
            anatomical_count = 0
            dynamic_count = 0
            
            for sample in valid_samples:
                if sample.anatomical_features and sample.sample_type in [SampleType.ANATOMICAL, SampleType.COMBINED]:
                    if sample.anatomical_embedding is not None:
                        templates['anatomical'].append(sample.anatomical_embedding)
                        anatomical_count += 1
                        print(f"Embedding anatómico existente: {sample.sample_id}")
                    else:
                        anatomical_embedding = self._generate_real_anatomical_embedding(
                            sample.anatomical_features, user_id, sample.sample_id
                        )
                        if anatomical_embedding is not None:
                            templates['anatomical'].append(anatomical_embedding)
                            sample.anatomical_embedding = anatomical_embedding
                            anatomical_count += 1
                            print(f"Embedding anatómico generado: {sample.sample_id}")
                        else:
                            print(f"Error generando anatómico: {sample.sample_id}")
                
                if (sample.dynamic_features and 
                    sample.sample_type in [SampleType.DYNAMIC, SampleType.COMBINED] and
                    self.dynamic_network.is_trained):
                    
                    if hasattr(sample, 'temporal_sequence') and sample.temporal_sequence is not None:
                        sample.dynamic_features.temporal_sequence = sample.temporal_sequence
                        print(f"Temporal sequence copiada: {len(sample.temporal_sequence)} frames")
                    elif hasattr(sample, 'metadata') and isinstance(sample.metadata, dict):
                        if 'temporal_sequence' in sample.metadata and sample.metadata['temporal_sequence']:
                            sample.dynamic_features.temporal_sequence = np.array(sample.metadata['temporal_sequence'], dtype=np.float32)
                            print(f"Temporal sequence desde metadata")
                    
                    if sample.dynamic_embedding is not None:
                        templates['dynamic'].append(sample.dynamic_embedding)
                        dynamic_count += 1
                        print(f"Embedding dinámico existente: {sample.sample_id}")
                    else:
                        dynamic_embedding = self._generate_real_dynamic_embedding(
                            sample.dynamic_features, user_id, sample.sample_id
                        )
                        if dynamic_embedding is not None:
                            templates['dynamic'].append(dynamic_embedding)
                            sample.dynamic_embedding = dynamic_embedding
                            dynamic_count += 1
                            print(f"Embedding dinámico generado: {sample.sample_id}")
                        else:
                            print(f"No se pudo generar dinámico: {sample.sample_id}")
            
  
            print(f"Templates generados exitosamente (modo {mode_text}):")
            print(f"   Embeddings anatomicos REALES: {anatomical_count}")
            print(f"   Embeddings dinamicos REALES: {dynamic_count}")
            print(f"   Total templates: {len(templates['anatomical']) + len(templates['dynamic'])}")
            
            return templates
            
        except Exception as e:
            print(f"Error generando templates: {e}")
            return {'anatomical': [], 'dynamic': []}
    
    def _generate_real_anatomical_embedding(self, features: AnatomicalFeatureVector, user_id: str, sample_id: str) -> Optional[np.ndarray]:
        """Genera embedding anatómico."""
        try:
            print(f"Generando embedding anatómico para {sample_id}")
            
            if self.anatomical_network.base_network:
                features_array = features.complete_vector.reshape(1, -1)
                
                expected_input_dim = self.anatomical_network.input_dim
                if features_array.shape[1] != expected_input_dim:
                    print(f"Dimensión incorrecta: {features_array.shape[1]} != {expected_input_dim}")
                    return None
                
                embedding = self.anatomical_network.base_network.predict(features_array, verbose=0)[0]
                
                if self._validate_generated_embedding(embedding, "anatomical"):
                    print(f"Embedding anatómico generado: dim={embedding.shape[0]}, norm={np.linalg.norm(embedding):.3f}")
                    return embedding
                else:
                    print("Embedding anatomico generado es inválido")
                    return None
            else:
                print("Red anatomica base no disponible")
                return None
                
        except Exception as e:
            print(f"Error generando anatómico: {e}")
            return None
    
    def _generate_real_dynamic_embedding(self, features: DynamicFeatureVector, user_id: str = None, sample_id: str = None) -> Optional[np.ndarray]:
        """Genera embedding dinámico."""
        try:
            log_msg = "Generando embedding dinámico"
            if sample_id:
                log_msg += f" para {sample_id}"
            print(log_msg)
            
            if not self.dynamic_network.base_network:
                print("Red dinámica no disponible")
                return None
            
            if not hasattr(features, 'temporal_sequence') or features.temporal_sequence is None:
                if sample_id:
                    print(f"No hay temporal_sequence para {sample_id}")
                else:
                    print("No hay temporal_sequence")
                print("No se puede generar embedding dinámico sin datos temporales")
                return None
            
            temporal_array = features.temporal_sequence
            expected_seq_length = self.dynamic_network.sequence_length
            expected_feature_dim = self.dynamic_network.feature_dim
            
            print(f"Temporal sequence shape: {temporal_array.shape}")
            
            if temporal_array.shape[0] > expected_seq_length:
                temporal_array = temporal_array[:expected_seq_length]
            elif temporal_array.shape[0] < expected_seq_length:
                padding = np.zeros((expected_seq_length - temporal_array.shape[0], temporal_array.shape[1]))
                temporal_array = np.vstack([temporal_array, padding])
            
            if temporal_array.shape[1] != expected_feature_dim:
                if temporal_array.shape[1] > expected_feature_dim:
                    temporal_array = temporal_array[:, :expected_feature_dim]
                else:
                    padding = np.zeros((temporal_array.shape[0], expected_feature_dim - temporal_array.shape[1]))
                    temporal_array = np.hstack([temporal_array, padding])
            
            sequence = temporal_array.reshape(1, expected_seq_length, expected_feature_dim)
            
            embedding = self.dynamic_network.base_network.predict(sequence, verbose=0)[0]
            
            if self._validate_generated_embedding(embedding, "dynamic"):
                print(f"Embedding dinámico generado: dim={embedding.shape[0]}, norm={np.linalg.norm(embedding):.3f}")
                return embedding
            else:
                print("Embedding dinámico inválido")
                return None
                
        except Exception as e:
            print(f"Error generando dinámico: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def _validate_generated_embedding(self, embedding: np.ndarray, embedding_type: str) -> bool:
        """Valida embedding generado."""
        try:
            if embedding is None:
                return False
            
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                print(f"Embedding {embedding_type} con NaN/infinitos")
                return False
            
            if np.allclose(embedding, 0.0, atol=1e-6):
                print(f"Embedding {embedding_type} es vector cero")
                return False
            
            magnitude = np.linalg.norm(embedding)
            if magnitude < 0.01 or magnitude > 1000.0:
                print(f"Magnitud {embedding_type} fuera de rango: {magnitude}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validando embedding {embedding_type}: {e}")
            return False
    
    def optimize_real_templates(self, templates: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        """Mantiene templates individuales SIN fusión."""
        try:
            print("✅ Manteniendo templates individuales (SIN promediado)")
            
            optimized = {}
            
            for modality, embeddings in templates.items():
                if not embeddings:
                    print(f"⚠️ No hay embeddings {modality}")
                    continue
                
                optimized[modality] = embeddings
                
                print(f"✅ {len(embeddings)} embeddings {modality} preservados")
                print(f"   Norma promedio: {np.mean([np.linalg.norm(e) for e in embeddings]):.3f}")
            
            print(f"✅ Optimización completada: {len(optimized)} modalidades")
            return optimized
            
        except Exception as e:
            print(f"❌ Error procesando templates: {e}")
            return {}
        
# ====================================================================
# WORKFLOW DE ENROLLMENT
# ====================================================================

class RealEnrollmentWorkflow:
    """Flujo de trabajo del proceso de enrollment REAL."""
    
    def __init__(self, config: RealEnrollmentConfig):
        """Inicializa workflow con componentes REALES."""
        self.config = config
        self.logger = get_logger()

        self.window_created = False
        self.window_name = "SISTEMA BIOMÉTRICO REAL"
        
        self.camera_manager = get_camera_manager()
        self.mediapipe_processor = get_mediapipe_processor()
        self.quality_validator = get_quality_validator()
        self.area_manager = get_reference_area_manager()
        self.sequence_manager = get_sequence_manager()
        self.anatomical_extractor = get_anatomical_features_extractor()
        self.dynamic_extractor = get_real_dynamic_features_extractor()
    
        self.quality_controller = RealQualityController(config)
        self.template_generator = RealTemplateGenerator(config)

        self.bootstrap_mode = False
        self.current_quality_assessment = None
        self.stats = {
            'frames_processed': 0,
            'samples_captured': 0,
            'quality_checks': 0,
            'bootstrap_mode_active': False,
            'bootstrap_enrollments': 0
        }
        
        self.database = get_biometric_database()
        
        self.current_session: Optional[RealEnrollmentSession] = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=30)
        
        print("RealEnrollmentWorkflow inicializado")
    
    def start_real_enrollment(self, user_id: str, username: str, 
                              gesture_sequence: List[str],
                              progress_callback: Optional[Callable] = None,
                              error_callback: Optional[Callable] = None) -> RealEnrollmentSession:
        """Inicia proceso de enrollment."""
        try:
            print(f"Iniciando enrollment para usuario {user_id}")
            print(f"  - Modo Bootstrap: {'SÍ' if self.bootstrap_mode else 'NO'}")
            
            session = RealEnrollmentSession(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
                username=username,
                gesture_sequence=gesture_sequence,
                progress_callback=progress_callback,
                error_callback=error_callback
            )

            session.is_bootstrap = self.bootstrap_mode
            
            session.total_samples_needed = len(gesture_sequence) * self.config.samples_per_gesture
            
            session.status = EnrollmentStatus.INITIALIZING
            session.current_phase = EnrollmentPhase.INITIALIZATION
            session.current_gesture = gesture_sequence[0] if gesture_sequence else ""

            if hasattr(self, 'workflow') and hasattr(self.workflow, 'set_bootstrap_mode'):
                self.workflow.set_bootstrap_mode(self.bootstrap_mode)
                
            if not self._initialize_real_components():
                session.status = EnrollmentStatus.FAILED
                error_msg = "Error inicializando componentes"
                if error_callback:
                    error_callback(error_msg)
                print(error_msg)
                return session
            
            session.status = EnrollmentStatus.COLLECTING_SAMPLES
            session.current_phase = EnrollmentPhase.SAMPLE_COLLECTION
            
            self.current_session = session
            self.is_running = True
            
            print(f"Enrollment iniciado: {session.session_id}")
            print(f"  - Gestos: {' → '.join(gesture_sequence)}")
            print(f"  - Muestras/gesto: {self.config.samples_per_gesture}")
            print(f"  - Total muestras: {session.total_samples_needed}")
            print(f"  - Bootstrap: {'SÍ' if self.bootstrap_mode else 'NO'}")

            if self.bootstrap_mode:
                self.stats['bootstrap_enrollments'] = self.stats.get('bootstrap_enrollments', 0) + 1
                            
            return session
            
        except Exception as e:
            print(f"Error iniciando enrollment: {e}")
            if error_callback:
                error_callback(str(e))
            raise
    

    #NUEVO
    def _finalize_enrollment_session(self, session: RealEnrollmentSession) -> Dict[str, Any]:
        """Finaliza enrollment y genera templates."""
        try:
            print("Generando templates biométricos...")
            
            # Generar templates usando el workflow
            templates = self.workflow.template_generator.generate_real_templates(
                session.samples,
                session.user_id,
                self.bootstrap_mode
            )
            
            print(f"✅ {len(templates)} templates generados")
            
            # Guardar en base de datos
            for template in templates:
                self.database.store_biometric_template(template)
            
            # Guardar perfil de usuario
            from app.core.biometric_database import UserProfile
            
            user_profile = UserProfile(
                user_id=session.user_id,
                username=session.username,
                gesture_sequence=session.gesture_sequence,
                total_enrollments=1,
                metadata={
                    'enrollment_date': time.time(),
                    'bootstrap_mode': self.bootstrap_mode
                }
            )
            
            self.database.store_user_profile(user_profile)
            
            print(f"✅ Usuario {session.user_id} registrado exitosamente")
            
            # Marcar sesión como completada
            session.status = EnrollmentStatus.COMPLETED
            
            return {
                'session_id': session.session_id,
                'status': 'completed',
                'progress': 100.0,
                'current_gesture': session.current_gesture,
                'current_gesture_index': session.current_gesture_index,
                'total_gestures': len(session.gesture_sequence),
                'samples_collected': len(session.samples),
                'samples_needed': session.total_samples_needed,
                'sample_captured': False,
                'session_completed': True,
                'templates_generated': len(templates),
                'message': '¡Enrollment completado exitosamente!'
            }
            
        except Exception as e:
            print(f"Error finalizando enrollment: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                'error': f'Error generando templates: {str(e)}',
                'session_id': session.session_id,
                'status': 'error',
                'session_completed': False
            }
        
    def _initialize_real_components(self) -> bool:
        """Inicializa componentes para captura REAL."""
        try:
            print("Inicializando componentes")
            
            if not self.camera_manager.is_initialized:
                if not self.camera_manager.initialize():
                    print("Error inicializando cámara")
                    return False
            
            if not self.mediapipe_processor.is_initialized:
                if not self.mediapipe_processor.initialize():
                    print("Error inicializando MediaPipe")
                    return False
            
            if not self.anatomical_extractor:
                print("Extractor anatómico no disponible")
                return False
            
            if not self.dynamic_extractor:
                print("Extractor dinámico no disponible")
                return False
            
            print("Componentes inicializados")
            return True
            
        except Exception as e:
            print(f"Error inicializando componentes: {e}")
            return False

    def set_bootstrap_mode(self, enabled: bool):
        """Configura el modo bootstrap."""
        self.bootstrap_mode = enabled
        print(f"Bootstrap mode: {'ENABLED' if enabled else 'DISABLED'}")
        
        if hasattr(self, 'quality_validator') and self.quality_validator:
            print("Quality validator configurado para bootstrap")

    def get_current_quality_assessment(self):
        """Obtiene el último quality assessment."""
        return getattr(self, 'current_quality_assessment', None)
    
    def process_real_frame(self):
        """
        Procesa un frame REAL para enrollment con ROI NORMALIZATION.        
        Returns:
            Muestra procesada o None si no es válida
        """
        try:
            if not self.current_session:
                return None
            
            session = self.current_session
            current_time = time.time()
            
            if session.last_capture_time > 0:
                time_since_last = current_time - session.last_capture_time
                if time_since_last < 1.5:
                    return None
            
            if (current_time - session.last_sample_time) > self.config.sample_timeout:
                print("Timeout de muestra")
                session.status = EnrollmentStatus.FAILED
                if session.error_callback:
                    session.error_callback("Timeout de captura")
                return None
            
            # ========================================================================
            # 🆕 PASO 1: CAPTURAR FRAME ORIGINAL
            # ========================================================================
            
            print("=" * 70)
            print("ENROLLMENT: PROCESAMIENTO CON ROI")
            print("=" * 70)
            
            ret, frame_original = self.camera_manager.capture_frame()
            if not ret or frame_original is None:
                print("Frame no capturado")
                return None
            
            session.frames_processed += 1
            print(f"Frame #{session.frames_processed} capturado - Shape: {frame_original.shape}")
            
            # ========================================================================
            # 🆕 PASO 2: DETECCIÓN INICIAL CON MEDIAPIPE (obtener landmarks)
            # ========================================================================
            
            print("Procesando frame original...")
            processing_result_initial = self.mediapipe_processor.process_frame(frame_original)
            
            if not processing_result_initial or not processing_result_initial.hand_result or not processing_result_initial.hand_result.is_valid:
                print("No se detectó mano válida")
                return None
            
            print("✅ Mano detectada")
            print(f"Confianza: {processing_result_initial.hand_result.confidence:.3f}")
            
            # ========================================================================
            # 🆕 PASO 3: EXTRAER Y VALIDAR ROI
            # ========================================================================
            
            roi_system = get_roi_normalization_system()
            
            print("=" * 70)
            print(f"EXTRAYENDO ROI - Gesto: {session.current_gesture}")
            print("=" * 70)
            
            roi_result = roi_system.extract_and_validate_roi(
                frame_original,
                processing_result_initial.hand_result.landmarks,
                session.current_gesture
            )

            self.last_roi_result = roi_result
            print(f"🔍 ROI GUARDADO:")
            print(f"   - is_valid: {roi_result.is_valid}")
            print(f"   - roi_bbox: {getattr(roi_result, 'roi_bbox', 'NO EXISTE')}")
            print(f"   - roi_width: {getattr(roi_result, 'roi_width', 'NO EXISTE')}")
            
            # ========================================================================
            # 🆕 PASO 4: VALIDAR DISTANCIA DEL ROI
            # ========================================================================
            
            if not roi_result.is_valid:
                print("=" * 70)
                print(f"❌ ROI NO VÁLIDO")
                print(f"Estado: {roi_result.distance_status.value}")
                print(f"Mensaje: {roi_result.feedback_message}")
                print(f"Tamaño ROI: {roi_result.roi_width}px")
                print("=" * 70)
                
                return None
            
            print("=" * 70)
            print("✅✅✅ ROI VÁLIDO - CAPTURANDO ✅✅✅")
            print(f"ROI dimensions: {roi_result.roi_width}x{roi_result.roi_height}px")
            print(f"Scaling factor: {roi_result.scaling_factor:.3f}x")
            print(f"Processing time: {roi_result.processing_time_ms:.2f}ms")
            print("=" * 70)
            
            print("✅ Usando landmarks del frame ORIGINAL")
            
            processing_result = processing_result_initial
            hand_result = processing_result.hand_result
            gesture_result = processing_result.gesture_result
            
            # ========================================================================
            # PASO 6: CONTINUAR CON LÓGICA NORMAL (usando datos del ROI normalizado)
            # ========================================================================
            
            reference_area_coords = self.area_manager.calculate_area_coordinates(
                session.current_gesture, frame_original.shape[:2]
            )
            reference_area = (reference_area_coords.x1, reference_area_coords.y1, 
                            reference_area_coords.x2, reference_area_coords.y2)
            
            print(f"🔍 PRE-VALIDACIÓN:")
            print(f"   - Gesto detectado: '{gesture_result.gesture_name if gesture_result else 'None'}'")
            print(f"   - Gesto esperado: '{session.current_gesture}'")
            print(f"   - Confianza gesto: {gesture_result.confidence if gesture_result else 0.0:.3f}")
            print(f"   - Confianza mano: {hand_result.confidence:.3f}")
            print(f"   - Frame: {session.frames_processed}")
            print(f"   - Modo Bootstrap: {self.bootstrap_mode}")
            print(f"   - ROI usado: {roi_result.roi_width}x{roi_result.roi_height}px")
            
            quality_assessment = self.quality_validator.validate_complete_quality(
                hand_landmarks=hand_result.landmarks,
                handedness=hand_result.handedness,
                detected_gesture=gesture_result.gesture_name if gesture_result else "None",
                gesture_confidence=gesture_result.confidence if gesture_result else 0.0,
                target_gesture=session.current_gesture,
                reference_area=reference_area,
                frame_shape=frame_original.shape[:2]
            )
            
            if quality_assessment:
                self.current_quality_assessment = quality_assessment
            
            if quality_assessment:
                print(f"🔍 QUALITY ASSESSMENT:")
                print(f"   - ready_for_capture: {quality_assessment.ready_for_capture}")
                print(f"   - overall_valid: {quality_assessment.overall_valid}")
                print(f"   - quality_score: {quality_assessment.quality_score:.3f}")
                print(f"   - bootstrap_mode: {self.bootstrap_mode}")

            
            if not quality_assessment or not quality_assessment.ready_for_capture:
                print(f"❌ NO LISTO - Esperando mejor calidad")
                return None
            
            current_gesture_samples = [s for s in session.samples if s.gesture_name == session.current_gesture]
            sample_number = len(current_gesture_samples) + 1
            
            print("=" * 70)
            print(f"🎯 READY_FOR_CAPTURE = TRUE! CAPTURANDO")
            print(f"   - Gesto: {session.current_gesture}")
            print(f"   - Muestra #{sample_number}")
            print(f"   - Calidad: {quality_assessment.quality_score:.3f}")
            print(f"   - Modo bootstrap: {self.bootstrap_mode}")
            print(f"   - Procesando con ROI: {roi_result.roi_width}x{roi_result.roi_height}px → 224x224px")
            print("=" * 70)
            
            anatomical_features = None
            if hand_result.landmarks:
                try:
                    anatomical_features = self.anatomical_extractor.extract_features(
                        hand_result.landmarks, 
                        hand_result.world_landmarks,
                        hand_result.handedness.classification[0].label if hand_result.handedness else 'unknown'
                    )
                    
                    if anatomical_features:
                        print(f"✅ Características anatómicas: {anatomical_features.complete_vector.shape}")
                    else:
                        print(f"❌ Error extrayendo anatómicas")
                        return None
                        
                except Exception as e:
                    print(f"❌ Excepción anatómicas: {e}")
                    return None
            else:
                print(f"❌ No hay landmarks")
                return None

            try:
                self.dynamic_extractor.add_frame_real(
                    landmarks=hand_result.landmarks,
                    gesture_name=gesture_result.gesture_name if gesture_result else "Unknown",
                    confidence=gesture_result.confidence if gesture_result else 0.8,
                    world_landmarks=hand_result.world_landmarks
                )
                
                print(f"✅ Frame agregado al extractor dinámico. Buffer: {len(self.dynamic_extractor.temporal_buffer)}/50")
                
            except Exception as e:
                print(f"❌ Error agregando frame: {e}")
            
            # =========================================================================
            # ✅ EXTRAER CARACTERÍSTICAS DINÁMICAS
            # =========================================================================
            
            dynamic_features = None
            temporal_sequence = None
            
            if len(self.dynamic_extractor.temporal_buffer) >= 10:
                try:
                    buffer_data = []
                    for frame in self.dynamic_extractor.temporal_buffer:
                        buffer_data.append({
                            'landmarks': frame.landmarks,
                            'gesture': frame.gesture_name,
                            'timestamp': frame.timestamp
                        })
                    
                    dynamic_features = self.dynamic_extractor.extract_features_from_sequence_real(
                        landmarks_sequence=[frame['landmarks'] for frame in buffer_data],
                        gesture_sequence=[frame['gesture'] for frame in buffer_data],
                        timestamps=[frame['timestamp'] for frame in buffer_data]
                    )
                    
                    if dynamic_features:
                        print(f"✅ Características dinámicas: {dynamic_features.complete_vector.shape}")
                    else:
                        print(f"⏳ Dinámicas: esperando más frames")
                    
                    temporal_sequence = self._extract_temporal_sequence_for_dynamic_network()
                    if temporal_sequence is not None:
                        print(f"✅ Secuencia temporal: {temporal_sequence.shape}")
                    else:
                        print("⚠️ No se pudo extraer secuencia temporal")
                            
                except Exception as e:
                    print(f"❌ Error dinámicas: {e}")
            else:
                print(f"⏳ Buffer: {len(self.dynamic_extractor.temporal_buffer)}/50")
            
            # =========================================================================
            # ✅ CREAR MUESTRA COMPLETA
            # =========================================================================
            
            sample_id = f"{session.session_id}_{session.current_gesture}_{sample_number}"
            
            sample = RealEnrollmentSample(
                sample_id=sample_id,
                user_id=session.user_id,
                sample_type=SampleType.COMBINED,
                gesture_name=session.current_gesture,
                anatomical_features=anatomical_features,
                dynamic_features=dynamic_features,
                quality_assessment=quality_assessment,
                confidence=gesture_result.confidence if gesture_result else 0.0,
                timestamp=current_time,
                capture_duration=current_time - session.start_time,
                frame_count=session.frames_processed
            )
            
            if not hasattr(sample, 'metadata'):
                sample.metadata = {}
            
            sample.metadata['roi_normalization'] = {
                'used': True,
                'roi_width': roi_result.roi_width,
                'roi_height': roi_result.roi_height,
                'scaling_factor': roi_result.scaling_factor,
                'distance_status': roi_result.distance_status.value,
                'processing_time_ms': roi_result.processing_time_ms
            }
            
            if temporal_sequence is not None:
                sample.temporal_sequence = temporal_sequence
                sample.sequence_length = len(temporal_sequence)
                sample.has_temporal_data = True
                print(f"✅ SECUENCIA TEMPORAL: {len(temporal_sequence)} frames")
                
                sample.metadata['temporal_sequence'] = temporal_sequence.tolist()
                sample.metadata['sequence_length'] = len(temporal_sequence)
                sample.metadata['has_temporal_data'] = True
                sample.metadata['data_source'] = 'real_dynamic_extractor_buffer'
            else:
                sample.temporal_sequence = None
                sample.sequence_length = 0
                sample.has_temporal_data = False
                print(f"⏳ Sin secuencia temporal")
            
            print(f"✅ Muestra creada: {sample_id}")
            
            if self.bootstrap_mode:
                print(f"🔧 BOOTSTRAP: Guardando SIN embeddings")
                
                sample.anatomical_embedding = None
                sample.dynamic_embedding = None
                sample.is_bootstrap_sample = True
                
            else:
                try:
                    print(f"🧠 NORMAL: Generando embeddings")
                    
                    if self.template_generator.anatomical_network.is_trained:
                        anatomical_embedding = self.template_generator._generate_real_anatomical_embedding(
                            anatomical_features, session.user_id, sample_id
                        )
                        sample.anatomical_embedding = anatomical_embedding
                        
                        if anatomical_embedding is not None:
                            print(f"✅ Embedding anatómico: {anatomical_embedding.shape}")
                        else:
                            print(f"❌ Error generando embedding anatómico")
                            return None
                    else:
                        print(f"❌ Red anatómica no entrenada en modo normal")
                        return None
                    
                    if dynamic_features and self.template_generator.dynamic_network.is_trained:
                        dynamic_embedding = self.template_generator._generate_real_dynamic_embedding(
                            dynamic_features, session.user_id, sample_id
                        )
                        sample.dynamic_embedding = dynamic_embedding
                        
                        if dynamic_embedding is not None:
                            print(f"✅ Embedding dinámico: {dynamic_embedding.shape}")
                        else:
                            print(f"⏳ Embedding dinámico pendiente")
                    elif not self.template_generator.dynamic_network.is_trained:
                        print(f"❌ Red dinámica no entrenada")
                    
                    sample.is_bootstrap_sample = False
                    
                except Exception as e:
                    print(f"❌ Error generando embeddings: {e}")
                    return None
            
            try:
                print(f"🔍 Validando muestra...")
                
                is_valid, validation_errors = self.quality_controller.validate_sample_quality(
                    sample, bootstrap_mode=self.bootstrap_mode
                )
                
                if not is_valid:
                    print(f"❌ Muestra inválida:")
                    for error in validation_errors:
                        print(f"   - {error}")
                    session.failed_samples += 1
                    return None
                
                sample.is_valid = True
                print(f"✅ Muestra validada")
                
            except Exception as e:
                print(f"❌ Error validando muestra: {e}")
                session.failed_samples += 1
                return None
            
            session.add_sample(sample)
            session.last_sample_time = current_time
            session.last_capture_time = current_time
            session.total_frames_captured += 1
            
            # ✅✅✅ BLOQUE BOOTSTRAP MEJORADO ✅✅✅
            if self.bootstrap_mode:
                print("="*70)
                print("💾 BOOTSTRAP: Guardando muestra durante captura")
                print(f"   Usuario: {session.user_id}")
                print(f"   Gesto: {sample.gesture_name}")
                print(f"   Sample ID: {sample.sample_id}")
                print("="*70)
                
                try:
                    # Verificar que hay características anatómicas
                    if sample.anatomical_features is None:
                        print("❌ BOOTSTRAP: No hay anatomical_features")
                    else:
                        print(f"✅ BOOTSTRAP: anatomical_features OK - dim={len(sample.anatomical_features.complete_vector)}")
                    
                    # Verificar database
                    if not hasattr(self, 'database') or self.database is None:
                        print("❌ BOOTSTRAP: self.database NO EXISTE")
                    else:
                        print(f"✅ BOOTSTRAP: database OK - tipo={type(self.database).__name__}")
                    
                    # Verificar método
                    if not hasattr(self.database, 'enroll_template_bootstrap'):
                        print("❌ BOOTSTRAP: método enroll_template_bootstrap NO EXISTE")
                        print(f"   Métodos disponibles: {[m for m in dir(self.database) if not m.startswith('_')]}")
                    else:
                        print("✅ BOOTSTRAP: método enroll_template_bootstrap existe")
                    
                    print("🔄 BOOTSTRAP: Llamando a enroll_template_bootstrap()...")
                    
                    template_id = self.database.enroll_template_bootstrap(
                        user_id=session.user_id,
                        anatomical_features=sample.anatomical_features.complete_vector if sample.anatomical_features else None,
                        gesture_name=sample.gesture_name,
                        quality_score=sample.quality_assessment.quality_score if sample.quality_assessment else 0.0,
                        confidence=sample.confidence,
                        sample_metadata={
                            'sample_id': sample.sample_id,
                            'capture_timestamp': current_time,
                            'gesture_sequence_position': session.current_gesture_index,
                            'session_id': session.session_id,
                            'bootstrap_mode': self.bootstrap_mode,
                            'sample_number': sample_number,
                            'session_username': session.username,
                            'has_temporal_data': sample.has_temporal_data,
                            'temporal_sequence': sample.temporal_sequence.tolist() if sample.temporal_sequence is not None else None,
                            'sequence_length': sample.sequence_length,
                            'bootstrap_features': sample.anatomical_features.complete_vector.tolist() if sample.anatomical_features is not None else None,
                            'feature_dimensions': len(sample.anatomical_features.complete_vector) if sample.anatomical_features is not None else 0,
                            'has_anatomical_raw': sample.anatomical_features is not None,
                            'data_source': 'real_enrollment_capture',
                            'roi_normalization_used': True,
                            'roi_width': roi_result.roi_width,
                            'roi_height': roi_result.roi_height,
                            'roi_scaling_factor': roi_result.scaling_factor
                        }
                    )
                    
                    print(f"📊 BOOTSTRAP: enroll_template_bootstrap() retornó: {template_id}")
                    
                    if template_id:
                        print(f"✅ BOOTSTRAP: Muestra guardada exitosamente")
                        print(f"   Template ID: {template_id}")
                        sample.template_id = template_id
                        
                        # Verificar inmediatamente
                        from time import sleep as time_sleep
                        time_sleep(0.1)
                        
                        templates_verificacion = self.database.list_user_templates(session.user_id)
                        print(f"📊 BOOTSTRAP: Verificación - {len(templates_verificacion)} templates totales para {session.user_id}")
                        
                    else:
                        print("❌ BOOTSTRAP: enroll_template_bootstrap retornó None/False")
                        print("   La muestra NO se guardó")
                        
                except AttributeError as attr_err:
                    print("="*70)
                    print(f"❌ BOOTSTRAP: AttributeError - {attr_err}")
                    print("="*70)
                    import traceback
                    print(traceback.format_exc())
                    
                except Exception as e:
                    print("="*70)
                    print(f"❌ BOOTSTRAP: Excepción guardando - {e}")
                    print("="*70)
                    import traceback
                    print(traceback.format_exc())

            print("=" * 70)
            print(f"🎉 MUESTRA AGREGADA CON ROI!")
            print(f"   📝 ID: {sample_id}")
            print(f"   🤚 Gesto: {session.current_gesture}")
            print(f"   📊 Progreso: {session.successful_samples}/{session.total_samples_needed}")
            print(f"   📈 Porcentaje: {session.progress_percentage:.1f}%")
            print(f"   🔧 Bootstrap: {self.bootstrap_mode}")
            print(f"   🧠 Embeddings: {'No (Bootstrap)' if self.bootstrap_mode else 'Sí (Normal)'}")
            print(f"   ⏱️ Datos temporales: {'Sí' if sample.has_temporal_data else 'No'}")
            print(f"   🎯 ROI: {roi_result.roi_width}x{roi_result.roi_height}px")
            print("=" * 70)
            
            if session.is_current_gesture_complete(self.config.samples_per_gesture):
                print(f"🎉 GESTO '{session.current_gesture}' COMPLETADO!")
                
                if session.advance_to_next_gesture():
                    print(f"➡️ Siguiente: {session.current_gesture}")
                else:
                    print(f"🏁 ENROLLMENT COMPLETADO!")
                    session.status = EnrollmentStatus.COMPLETED
                    
                    if self.bootstrap_mode:
                        log_info(f"🧠 Bootstrap completado - entrenamiento pendiente")
            
            if session.progress_callback:
                try:
                    progress_data = {
                        'progress_percentage': session.progress_percentage,
                        'current_gesture': session.current_gesture,
                        'current_gesture_index': session.current_gesture_index,
                        'total_gestures': len(session.gesture_sequence),
                        'samples_captured': session.successful_samples,
                        'samples_needed': session.total_samples_needed,
                        'failed_samples': session.failed_samples,
                        'sample_captured': True,
                        'sample_id': sample_id,
                        'sample_quality': quality_assessment.quality_score,
                        'sample_confidence': sample.confidence,
                        'anatomical_embedding_generated': sample.anatomical_embedding is not None,
                        'dynamic_embedding_generated': sample.dynamic_embedding is not None,
                        'is_real_processing': True,
                        'no_simulation': True,
                        'bootstrap_mode': self.bootstrap_mode,
                        'session_status': session.status.value,
                        'duration': session.duration,
                        'has_temporal_data': sample.has_temporal_data,
                        'roi_used': True,
                        'roi_width': roi_result.roi_width,
                        'roi_result': roi_result
                    }
                    
                    session.progress_callback(progress_data)
                    
                except Exception as e:
                    print(f"❌ Error callback: {e}")
            
            if self.config.show_preview:
                self._draw_real_feedback(frame_original, quality_assessment, processing_result)
            
            return sample
            
        except Exception as e:
            print(f"❌ Error crítico: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            
            if hasattr(self, 'current_session') and self.current_session and self.current_session.error_callback:
                self.current_session.error_callback(f"Error: {str(e)}")
            
            return None
    
    def _extract_temporal_sequence_for_dynamic_network(self) -> Optional[np.ndarray]:
        """Extrae secuencia temporal para red dinámica."""
        try:
            if len(self.dynamic_extractor.temporal_buffer) < 5:
                logger.warning("Buffer temporal insuficiente")
                return None
            
            temporal_frames = []
            for frame_data in self.dynamic_extractor.temporal_buffer:
                if hasattr(frame_data, 'landmarks') and frame_data.landmarks is not None:
                    landmarks = frame_data.landmarks
                    world_landmarks = getattr(frame_data, 'world_landmarks', None)
                    
                    frame_features = self._extract_single_frame_features(landmarks, world_landmarks)

                    if frame_features is not None:
                        temporal_frames.append(frame_features)
            
            if len(temporal_frames) < 5:
                logger.warning("Insuficientes frames válidos para secuencia")
                return None
            
            temporal_sequence = np.array(temporal_frames, dtype=np.float32)
            
            if len(temporal_sequence) > 50:
                temporal_sequence = temporal_sequence[-50:]
            
            print(f"Secuencia temporal extraída: {temporal_sequence.shape}")
            return temporal_sequence
            
        except Exception as e:
            print(f"Error extrayendo secuencia temporal: {e}")
            return None
    
    def _capture_fluid_dynamic_sequence(self, session: RealEnrollmentSession) -> Optional[np.ndarray]:
        """
        Captura UNA secuencia FLUIDA completa para características dinámicas.
        
        Returns:
            np.ndarray: Secuencia temporal (50, 320) para la red BiLSTM
        """
        try:
            print("\n" + "="*80)
            print("🎬 FASE 2: CAPTURA DE SECUENCIA DINÁMICA FLUIDA")
            print("="*80)
            print(f"Secuencia: {' → '.join(session.gesture_sequence)}")
            print("\n⚠️ IMPORTANTE:")
            print("  - Realiza los gestos de forma FLUIDA")
            print("  - NO te detengas entre gestos")
            print("  - Mantén ritmo constante")
            input("\nPresiona ENTER cuando estés listo...")
            
            # CORRECCIÓN: Usar reset_state() en lugar de reset_real_buffer()
            self.dynamic_extractor.reset_state()
            print("✅ Buffer limpiado")
            
            target_frames = 50
            frames_captured = 0
            start_time = time.time()
            max_duration = 20.0
            
            # Listas para almacenar datos temporales
            landmarks_sequence = []
            gesture_sequence = []
            timestamps = []
            temporal_features_sequence = []
            
            print(f"\n🎬 CAPTURANDO {target_frames} frames...\n")
            
            while frames_captured < target_frames:
                if time.time() - start_time > max_duration:
                    print(f"\n⚠️ Timeout ({max_duration}s)")
                    break
                
                ret, frame = self.camera_manager.capture_frame()
                if not ret or frame is None:
                    continue
                
                result = self.mediapipe_processor.process_frame(frame)
                
                if result.hand_detected and result.hand_landmarks:
                    # Agregar al buffer del extractor
                    success = self.dynamic_extractor.add_frame_real(
                        landmarks=result.hand_landmarks,
                        gesture_name=result.gesture_name,
                        confidence=result.gesture_confidence,
                        world_landmarks=result.world_landmarks
                    )
                    
                    if success:
                        # Guardar para reconstruir secuencia
                        landmarks_sequence.append(result.hand_landmarks)
                        gesture_sequence.append(result.gesture_name)
                        timestamps.append(time.time())
                        
                        # Extraer características del frame actual para la secuencia temporal
                        frame_features = self._extract_single_frame_features(
                            result.hand_landmarks,
                            result.world_landmarks
                        )
                        
                        if frame_features is not None and len(frame_features) == 320:
                            temporal_features_sequence.append(frame_features)
                            frames_captured += 1
                            
                            if frames_captured % 10 == 0:
                                print(f"📊 {frames_captured}/{target_frames} - {result.gesture_name}")
                            
                            # Mostrar feedback visual
                            display = frame.copy()
                            cv2.putText(display, f"Frames: {frames_captured}/{target_frames}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(display, f"Gesto: {result.gesture_name}", 
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.imshow("Captura Secuencia Fluida", display)
                            
                            if cv2.waitKey(1) & 0xFF == 27:
                                cv2.destroyAllWindows()
                                return None
            
            cv2.destroyAllWindows()
            
            duration = time.time() - start_time
            print(f"\n✅ {frames_captured} frames capturados en {duration:.1f}s")
            
            # Validar que tenemos suficientes frames
            if len(temporal_features_sequence) < target_frames:
                print(f"⚠️ Frames insuficientes: {len(temporal_features_sequence)} < {target_frames}")
                # Padding con zeros
                while len(temporal_features_sequence) < target_frames:
                    temporal_features_sequence.append(np.zeros(320, dtype=np.float32))
            elif len(temporal_features_sequence) > target_frames:
                # Truncar
                temporal_features_sequence = temporal_features_sequence[:target_frames]
            
            # Convertir a numpy array (50, 320)
            sequence_array = np.array(temporal_features_sequence, dtype=np.float32)
            
            print(f"✅ Secuencia temporal construida: {sequence_array.shape}")
            print(f"   - Frames: {sequence_array.shape[0]}")
            print(f"   - Features por frame: {sequence_array.shape[1]}")
            print(f"   - Dtype: {sequence_array.dtype}")
            
            # Validar shape final
            if sequence_array.shape != (50, 320):
                print(f"⚠️ Shape incorrecto: {sequence_array.shape}, esperado (50, 320)")
                return None
            
            return sequence_array
            
        except Exception as e:
            print(f"Error capturando secuencia fluida: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            cv2.destroyAllWindows()
            return None
    
    def _extract_single_frame_features(self, landmarks, world_landmarks=None) -> Optional[np.ndarray]:
        """Extrae características de un frame individual."""
        try:
            anatomical_features = self.anatomical_extractor.extract_features(landmarks, world_landmarks)
            
            if anatomical_features and anatomical_features.complete_vector is not None:
                frame_features = anatomical_features.complete_vector
                
                if len(frame_features) >= 180:
                    padded_features = np.zeros(320, dtype=np.float32)
                    padded_features[:180] = frame_features[:180]
                    
                    remaining_dims = 320 - 180
                    if len(frame_features) >= 140:
                        padded_features[180:] = frame_features[:140]
                    else:
                        feature_cycle = np.tile(frame_features, (remaining_dims // len(frame_features)) + 1)
                        padded_features[180:] = feature_cycle[:remaining_dims]
                    
                    return padded_features
            
            return None
            
        except Exception as e:
            print(f"Error extrayendo features de frame: {e}")
            return None
    
    def show_preview_with_feedback(self, frame, session_info):
        """Muestra preview con feedback visual."""
        try:
            if frame is None:
                return
            
            current_gesture = session_info.get('current_gesture', 'Unknown')
            feedback_messages = get_visual_feedback_manager().generate_real_time_feedback(
                self.current_quality_assessment, current_gesture, session_info
            )
            
            frame_with_feedback = get_visual_feedback_manager().draw_feedback_overlay(
                frame, feedback_messages, self.current_quality_assessment
            )
            
            cv2.imshow("ENROLLMENT - Sistema Biométrico", frame_with_feedback)
            
        except Exception as e:
            print(f"Error mostrando preview: {e}")
            cv2.imshow("ENROLLMENT - Sistema Biométrico", frame)
    
    def _extract_real_dynamic_features(self) -> Optional[DynamicFeatureVector]:
        """Extrae características dinámicas del buffer temporal."""
        try:
            if len(self.frame_buffer) < 5:
                return None
            
            landmarks_sequence = []
            gesture_sequence = []
            timestamps = []
            
            for frame_data in self.frame_buffer:
                landmarks_sequence.append(frame_data['landmarks'])
                gesture_sequence.append(frame_data.get('gesture', 'Unknown'))
                timestamps.append(frame_data['timestamp'])
            
            dynamic_features = self.dynamic_extractor.extract_features_from_sequence_real(
                landmarks_sequence=landmarks_sequence,
                gesture_sequence=gesture_sequence,
                timestamps=timestamps
            )
            
            if dynamic_features and self._validate_real_dynamic_features(dynamic_features):
                print(f"Características dinámicas extraídas: dim={dynamic_features.complete_vector.shape[0]}")
                return dynamic_features
            else:
                print("Error extrayendo características dinámicas")
                return None
                
        except Exception as e:
            print(f"Error extrayendo dinámicas: {e}")
            return None
    
    def _validate_real_dynamic_features(self, features: DynamicFeatureVector) -> bool:
        """Valida las características dinámicas."""
        try:
            if not features or features.complete_vector is None:
                return False
            
            vector = features.complete_vector
            
            if np.var(vector) < 1e-8:
                print("Características dinamicas sin variación")
                return False
            
            if len(vector) > 10:
                autocorr = np.correlate(vector, vector, mode='full')
                if np.max(autocorr[len(autocorr)//2+1:]) > 0.95 * np.max(autocorr):
                    print("Caracteristicas dinamicas con patrones artificiales detectados")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error validando dinámicas: {e}")
            return False
    
    
    def _advance_to_next_gesture(self, session: RealEnrollmentSession):
        """Avanza al siguiente gesto en la secuencia."""
        try:
            session.current_gesture_index += 1
            
            if session.current_gesture_index < len(session.gesture_sequence):
                # Siguiente gesto
                session.current_gesture = session.gesture_sequence[session.current_gesture_index]
                print(f"Avanzando al gesto: {session.current_gesture}")
            else:
                # Secuencia completada
                print("Secuencia de gestos completada - iniciando generación de templates")
                session.current_phase = EnrollmentPhase.TEMPLATE_GENERATION
                session.status = EnrollmentStatus.GENERATING_TEMPLATES
                
                # Procesar templates finales
                self._finalize_real_enrollment(session)
                
        except Exception as e:
            print(f"Error avanzando a siguiente gesto: {e}")
            session.status = EnrollmentStatus.FAILED
    
    def _finalize_real_enrollment(self, session: RealEnrollmentSession):
        """Finaliza el enrollment generando templates finales."""
        try:
            print(f"Finalizando enrollment para usuario {session.user_id}")
            
            # Filtrar muestras válidas
            valid_samples = [s for s in session.samples if s.is_valid]
            print(f"Muestras válidas para templates: {len(valid_samples)}/{len(session.samples)}")
            
            if len(valid_samples) < self.config.min_samples_per_gesture:
                session.status = EnrollmentStatus.FAILED
                error_msg = f"Insuficientes muestras válidas: {len(valid_samples)} < {self.config.min_samples_per_gesture}"
                print(error_msg)
                if session.error_callback:
                    session.error_callback(error_msg)
                return
            
            # DEBUG: Verificar estados antes de decidir modo
            session_is_bootstrap = getattr(session, 'is_bootstrap', False)
            system_bootstrap_mode = getattr(self, 'bootstrap_mode', False)
            
            print("DEBUG FINALIZE ENROLLMENT:")
            print(f"   - session.is_bootstrap: {session_is_bootstrap}")
            print(f"   - self.bootstrap_mode: {system_bootstrap_mode}")
            print(f"   - Condicion original: {session_is_bootstrap or system_bootstrap_mode}")
            
            # Verificar estado de redes siamesas
            try:
                anatomical_network = get_real_siamese_anatomical_network()
                dynamic_network = get_real_siamese_dynamic_network()
                
                anatomical_trained = getattr(anatomical_network, 'is_trained', False)
                dynamic_trained = getattr(dynamic_network, 'is_trained', False)
                
                print(f"   - Red anatomica entrenada: {anatomical_trained}")
                print(f"   - Red dinamica entrenada: {dynamic_trained}")
                
                # Si las redes están entrenadas, usar modo normal
                if anatomical_trained and dynamic_trained:
                    print("AMBAS REDES ENTRENADAS - FORZANDO MODO NORMAL")
                    use_bootstrap_mode = False
                elif anatomical_trained or dynamic_trained:
                    print("REDES PARCIALMENTE ENTRENADAS - FORZANDO MODO NORMAL")
                    use_bootstrap_mode = False
                else:
                    print("REDES NO ENTRENADAS - USANDO LOGICA ORIGINAL")
                    use_bootstrap_mode = session_is_bootstrap or system_bootstrap_mode
                    
            except Exception as e:
                print(f"Error verificando redes: {e}")
                # En caso de error, forzar modo normal
                use_bootstrap_mode = False
                print("ERROR VERIFICANDO REDES - FORZANDO MODO NORMAL")
            
            print(f"DECISION FINAL: {'BOOTSTRAP' if use_bootstrap_mode else 'NORMAL'}")
            
            # VERIFICAR MODO BOOTSTRAP ANTES DE GENERAR TEMPLATES
            if use_bootstrap_mode:
                # MODO BOOTSTRAP: Los datos ya se guardaron durante la captura
                print("MODO BOOTSTRAP: Datos ya guardados durante captura - Finalizando sesion")
                print("SALTANDO generacion de templates (redes no entrenadas en bootstrap)")
                
                session.status = EnrollmentStatus.COMPLETED
                session.current_phase = EnrollmentPhase.ENROLLMENT_COMPLETE
                session.end_time = time.time()
                
                print(f"Enrollment BOOTSTRAP completado exitosamente para usuario {session.user_id}")
                print(f"  - Duracion: {session.duration:.1f} segundos")
                print(f"  - Muestras capturadas: {len(session.samples)}")
                print(f"  - Muestras validas: {len(valid_samples)}")
                print(f"  - Modo: Bootstrap (sin embeddings)")
                
                if session.progress_callback:
                    session.progress_callback(100.0)
                
                return
            
            # MODO NORMAL: Procesar templates con embeddings
            print("MODO NORMAL: Generando templates con embeddings")
            
            # Generar templates finales
            session.current_phase = EnrollmentPhase.TEMPLATE_GENERATION
            
            # VERIFICAR SI EL TEMPLATE_GENERATOR EXISTE
            if not hasattr(self, 'template_generator'):
                print("template_generator no existe - creando uno basico")
                self.template_generator = self._create_basic_template_generator()
            
            templates = self.template_generator.generate_real_templates(valid_samples, session.user_id)
            
            # AGREGAR ESTAS LÍNEAS:
            print(f"DEBUG: Templates generados = {templates}")
            print(f"DEBUG: Anatomical templates = {len(templates.get('anatomical', []))}")
            print(f"DEBUG: Dynamic templates = {len(templates.get('dynamic', []))}")

            if not templates['anatomical'] and not templates['dynamic']:
                session.status = EnrollmentStatus.FAILED
                error_msg = "Error generando templates biometricos"
                print(error_msg)
                if session.error_callback:
                    session.error_callback(error_msg)
                return
            
            # Optimizar templates (ahora mantiene individuales sin promediar)
            optimized_templates = self.template_generator.optimize_real_templates(templates)
            
            # CAMBIO CRITICO: Logging actualizado para templates individuales
            print(f"Templates individuales generados exitosamente:")
            print(f"   - Anatomicos: {len(optimized_templates.get('anatomical', []))} templates individuales")
            print(f"   - Dinamicos: {len(optimized_templates.get('dynamic', []))} templates individuales")
            
            # Logging adicional de normas para verificacion
            if optimized_templates.get('anatomical'):
                avg_norm_anat = np.mean([np.linalg.norm(e) for e in optimized_templates['anatomical']])
                print(f"   - Norma promedio anatomica: {avg_norm_anat:.3f}")
            
            if optimized_templates.get('dynamic'):
                avg_norm_dyn = np.mean([np.linalg.norm(e) for e in optimized_templates['dynamic']])
                print(f"   - Norma promedio dinamica: {avg_norm_dyn:.3f}")
            
            # Almacenar en base de datos
            session.current_phase = EnrollmentPhase.DATABASE_STORAGE
            session.status = EnrollmentStatus.STORING_DATA
            
            print("Iniciando almacenamiento en base de datos...")
            
            # ========== FASE 2: CAPTURA DINÁMICA FLUIDA ==========
            print("\n" + "="*80)
            print("🎬 FASE 2: CAPTURA DE SECUENCIA DINÁMICA FLUIDA")
            print("="*80)
            
            fluid_sequence = self._capture_fluid_dynamic_sequence(session)
            
            if fluid_sequence is not None:
                print(f"✅ Secuencia fluida capturada: {fluid_sequence.shape}")
                session.dynamic_phase_completed = True
                session.fluid_sequence_captured = True
                session.dynamic_sequence_data = fluid_sequence
            else:
                print("⚠️ No se capturó secuencia fluida")
                session.dynamic_phase_completed = False
                session.fluid_sequence_captured = False
            
            print("="*80 + "\n")
            # =====================================================
                
            # Modo normal: usar almacenamiento estandar
            if self._store_real_user_data(session, optimized_templates):
                session.status = EnrollmentStatus.COMPLETED
                session.current_phase = EnrollmentPhase.ENROLLMENT_COMPLETE
                session.end_time = time.time()
                
                # CAMBIO CRITICO: Logging final actualizado
                total_templates = sum(len(v) for v in optimized_templates.values() if isinstance(v, list))
                
                print(f"Enrollment NORMAL completado exitosamente para usuario {session.user_id}")
                print(f"  - Duracion: {session.duration:.1f} segundos")
                print(f"  - Muestras capturadas: {len(session.samples)}")
                print(f"  - Templates individuales generados: {total_templates}")
                print(f"    * Anatomicos: {len(optimized_templates.get('anatomical', []))}")
                print(f"    * Dinamicos: {len(optimized_templates.get('dynamic', []))}")
                
                if session.progress_callback:
                    session.progress_callback(100.0)
            else:
                session.status = EnrollmentStatus.FAILED
                error_msg = "Error almacenando datos en base de datos"
                print(error_msg)
                if session.error_callback:
                    session.error_callback(error_msg)
        
        except Exception as e:
            print(f"Error finalizando enrollment: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            session.status = EnrollmentStatus.FAILED
            if session.error_callback:
                session.error_callback(str(e))

    def _create_basic_template_generator(self):
        """Crea un generador básico de templates si no existe."""
        class BasicTemplateGenerator:
            def generate_real_templates(self, valid_samples, user_id):
                """Genera templates básicos desde las muestras."""
                templates = {'anatomical': [], 'dynamic': []}
                
                for sample in valid_samples:
                    # CORRECCIÓN CRÍTICA: Copiar temporal_sequence antes de procesar embeddings dinámicos
                    if hasattr(sample, 'dynamic_features') and sample.dynamic_features is not None:
                        # Método 1: Copiar desde atributo directo del sample
                        if hasattr(sample, 'temporal_sequence') and sample.temporal_sequence is not None:
                            sample.dynamic_features.temporal_sequence = sample.temporal_sequence
                            print(f"Temporal sequence copiada para {sample.sample_id}: {len(sample.temporal_sequence)} frames")
                        # Método 2: Copiar desde metadata como fallback
                        elif hasattr(sample, 'metadata') and isinstance(sample.metadata, dict):
                            if 'temporal_sequence' in sample.metadata and sample.metadata['temporal_sequence']:
                                import numpy as np
                                sample.dynamic_features.temporal_sequence = np.array(sample.metadata['temporal_sequence'], dtype=np.float32)
                                print(f"Temporal sequence copiada desde metadata: {len(sample.metadata['temporal_sequence'])} frames")
                    
                    # Agregar embeddings anatómicos
                    if hasattr(sample, 'anatomical_embedding') and sample.anatomical_embedding is not None:
                        templates['anatomical'].append(sample.anatomical_embedding)
                    
                    # Agregar embeddings dinámicos
                    if hasattr(sample, 'dynamic_embedding') and sample.dynamic_embedding is not None:
                        templates['dynamic'].append(sample.dynamic_embedding)
                
                print(f"Templates basicos generados: {len(templates['anatomical'])} anatomicos, {len(templates['dynamic'])} dinamicos")
                return templates
            
            def optimize_real_templates(self, templates):
                """Mantiene templates individuales SIN promediado."""
                print("✅ Modo básico: preservando templates individuales")
                
                optimized = {}
                
                if templates['anatomical']:
                    # ✅ MANTENER lista de embeddings individuales
                    optimized['anatomical'] = templates['anatomical']
                    print(f"✅ {len(templates['anatomical'])} templates anatómicos individuales preservados")
                
                if templates['dynamic']:
                    # ✅ MANTENER lista de embeddings individuales
                    optimized['dynamic'] = templates['dynamic']
                    print(f"✅ {len(templates['dynamic'])} templates dinámicos individuales preservados")
                
                return optimized
        
        return BasicTemplateGenerator()

    
    def _store_real_user_data(self, session: RealEnrollmentSession, templates: Dict[str, List[np.ndarray]]) -> bool:
        """
        Almacena datos del usuario en la base de datos.
        ✅ REFACTORIZADO: Guarda múltiples templates individuales (sin promediado).
        
        Args:
            session: Sesión de enrollment con muestras capturadas
            templates: Dict con listas de embeddings individuales por modalidad
                    {'anatomical': [emb1, emb2, ...], 'dynamic': [emb1, emb2, ...]}
        
        Returns:
            bool: True si se almacenó exitosamente
        """
        try:
            print(f"Almacenando datos REALES del usuario {session.user_id}")
            
            # Crear perfil de usuario
            user_profile = UserProfile(
                user_id=session.user_id,
                username=session.username,
                gesture_sequence=session.gesture_sequence,
                metadata={
                    'enrollment_mode': 'normal',
                    'session_id': session.session_id,
                    'total_samples': len(session.samples),
                    'valid_samples': len([s for s in session.samples if s.is_valid]),
                    'enrollment_duration': session.duration,
                    'enrollment_date': session.start_time,
                    'quality_score': np.mean([s.quality_assessment.quality_score for s in session.samples 
                                            if s.quality_assessment and hasattr(s.quality_assessment, 'quality_score')]),
                    'created_with_system': 'real_enrollment_workflow',
                    'template_storage_mode': 'individual'
                }
            )
            
            user_profile.total_enrollments = 1
            user_profile.updated_at = time.time()
            
            # Crear múltiples templates individuales
            biometric_templates = []
            
            # Procesar templates ANATÓMICOS individuales
            if 'anatomical' in templates and templates['anatomical']:
                print(f"Procesando {len(templates['anatomical'])} templates anatomicos individuales")
                
                for i, anatomical_embedding in enumerate(templates['anatomical']):
                    sample = session.samples[i] if i < len(session.samples) else None
                    
                    template_id = f"{session.user_id}_anatomical_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
                    
                    gesture_index = i % len(session.gesture_sequence)
                    gesture_name = session.gesture_sequence[gesture_index]
                    
                    biometric_template = BiometricTemplate(
                        user_id=session.user_id,
                        template_id=template_id,
                        template_type=TemplateType.ANATOMICAL,
                        anatomical_embedding=anatomical_embedding,
                        dynamic_embedding=None,
                        gesture_name=gesture_name,
                        quality_score=sample.quality_assessment.quality_score if (sample and sample.quality_assessment 
                                    and hasattr(sample.quality_assessment, 'quality_score')) else 0.8,
                        confidence=sample.quality_assessment.quality_score * 0.9 if (sample and sample.quality_assessment 
                                and hasattr(sample.quality_assessment, 'quality_score')) else 0.8,
                        enrollment_session=session.session_id,
                        metadata={
                            'modality': 'anatomical',
                            'sample_index': i,
                            'total_samples': len(templates['anatomical']),
                            'gesture_sequence': session.gesture_sequence,
                            'is_real_data': True,
                            'no_synthetic_data': True,
                            'creation_date': time.time(),
                            'version': "2.0_real_individual",
                            'storage_mode': 'individual',
                            'data_source': 'real_enrollment_capture',
                            
                            # Guardar características raw para regeneración
                            'bootstrap_features': (sample.anatomical_features.complete_vector.tolist() 
                                                if sample and sample.anatomical_features else []),
                            'has_anatomical_raw': True,
                            'bootstrap_mode': False
                        }
                    )
                    
                    biometric_templates.append(biometric_template)
                    print(f"  Template anatomico {i+1}/{len(templates['anatomical'])} creado (norma: {np.linalg.norm(anatomical_embedding):.3f})")
            
            # Procesar templates DINÁMICOS individuales
            if 'dynamic' in templates and templates['dynamic']:
                print(f"Procesando {len(templates['dynamic'])} templates dinamicos individuales")
                
                for i, dynamic_embedding in enumerate(templates['dynamic']):
                    sample = session.samples[i] if i < len(session.samples) else None
                    
                    template_id = f"{session.user_id}_dynamic_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
                    
                    gesture_index = i % len(session.gesture_sequence)
                    gesture_name = session.gesture_sequence[gesture_index]
                    
                    # CORRECCIÓN CRÍTICA: Extraer temporal_sequence correctamente
                    temporal_seq = []
                    if sample:
                        # Método 1: Atributo directo de sample
                        if hasattr(sample, 'temporal_sequence') and sample.temporal_sequence is not None:
                            temporal_seq = sample.temporal_sequence.tolist() if hasattr(sample.temporal_sequence, 'tolist') else sample.temporal_sequence
                            print(f"  Temporal sequence extraida de sample.temporal_sequence: {len(temporal_seq)} frames")
                        
                        # Método 2: Metadata de sample
                        elif hasattr(sample, 'metadata') and isinstance(sample.metadata, dict):
                            if 'temporal_sequence' in sample.metadata and sample.metadata['temporal_sequence']:
                                temporal_seq = sample.metadata['temporal_sequence']
                                print(f"  Temporal sequence extraida de sample.metadata: {len(temporal_seq)} frames")
                        
                        # Método 3: dynamic_features.temporal_sequence
                        elif (sample.dynamic_features and 
                            hasattr(sample.dynamic_features, 'temporal_sequence') and 
                            sample.dynamic_features.temporal_sequence is not None):
                            temporal_seq = sample.dynamic_features.temporal_sequence.tolist() if hasattr(sample.dynamic_features.temporal_sequence, 'tolist') else sample.dynamic_features.temporal_sequence
                            print(f"  Temporal sequence extraida de dynamic_features: {len(temporal_seq)} frames")
                    
                    if not temporal_seq:
                        print(f"  Sin temporal_sequence disponible para muestra {i+1}")
                    
                    biometric_template = BiometricTemplate(
                        user_id=session.user_id,
                        template_id=template_id,
                        template_type=TemplateType.DYNAMIC,
                        anatomical_embedding=None,
                        dynamic_embedding=dynamic_embedding,
                        gesture_name=gesture_name,
                        quality_score=sample.quality_assessment.quality_score if (sample and sample.quality_assessment 
                                    and hasattr(sample.quality_assessment, 'quality_score')) else 0.8,
                        confidence=sample.quality_assessment.quality_score * 0.9 if (sample and sample.quality_assessment 
                                and hasattr(sample.quality_assessment, 'quality_score')) else 0.8,
                        enrollment_session=session.session_id,
                        metadata={
                            'modality': 'dynamic',
                            'sample_index': i,
                            'total_samples': len(templates['dynamic']),
                            'gesture_sequence': session.gesture_sequence,
                            'is_real_data': True,
                            'no_synthetic_data': True,
                            'creation_date': time.time(),
                            'version': "2.0_real_individual",
                            'storage_mode': 'individual',
                            'data_source': 'real_enrollment_capture',
                            
                            # Guardar secuencia temporal para regeneración
                            'temporal_sequence': temporal_seq,
                            'has_temporal_data': len(temporal_seq) > 0,
                            'bootstrap_mode': False
                        }
                    )
                    
                    biometric_templates.append(biometric_template)
                    print(f"  Template dinamico {i+1}/{len(templates['dynamic'])} creado (norma: {np.linalg.norm(dynamic_embedding):.3f})")
            
            # ========== NUEVO: Procesar template DINÁMICO de secuencia fluida ==========
            if session.dynamic_phase_completed and session.fluid_sequence_captured:
                print(f"\n🎬 Procesando template dinámico de SECUENCIA FLUIDA")
                
                # Verificar si tenemos la secuencia guardada
                if hasattr(session, 'dynamic_sequence_data') and session.dynamic_sequence_data is not None:
                    
                    sequence_array = session.dynamic_sequence_data  # Shape: (50, 320)
                    
                    print(f"📊 Secuencia fluida:")
                    print(f"   - Shape: {sequence_array.shape}")
                    print(f"   - Dtype: {sequence_array.dtype}")
                    
                    # Validar shape
                    if len(sequence_array.shape) != 2:
                        print(f"⚠️ Shape incorrecto: esperado (50, 320), obtenido {sequence_array.shape}")
                        sequence_array = None
                    elif sequence_array.shape[0] != 50 or sequence_array.shape[1] != 320:
                        print(f"⚠️ Dimensiones incorrectas: {sequence_array.shape}")
                        sequence_array = None
                    
                    if sequence_array is not None:
                        # Generar embedding
                        if self.dynamic_network and hasattr(self.dynamic_network, 'is_trained') and self.dynamic_network.is_trained:
                            try:
                                # Preprocesar para la red BiLSTM
                                # Input esperado: (1, 50, 320)
                                sequence_input = np.expand_dims(sequence_array, axis=0)
                                
                                # Predecir embedding usando la red base
                                dynamic_embedding_sequence = self.dynamic_network.base_network.predict(
                                    sequence_input,
                                    verbose=0
                                )
                                dynamic_embedding_sequence = dynamic_embedding_sequence.flatten()  # (128,)
                                
                                print(f"✅ Embedding con red entrenada: {dynamic_embedding_sequence.shape}")
                                
                            except Exception as e:
                                print(f"⚠️ Error con red entrenada: {e}")
                                print(f"Error generando embedding de secuencia: {e}")
                                # Fallback: usar promedio de frames
                                dynamic_embedding_sequence = np.mean(sequence_array, axis=0).flatten()[:128]
                        else:
                            # Bootstrap: usar promedio de características
                            print("⚠️ Modo bootstrap: usando promedio de frames")
                            dynamic_embedding_sequence = np.mean(sequence_array, axis=0).flatten()[:128]
                        
                        # Crear template de secuencia
                        template_id_seq = f"{session.user_id}_dynamic_sequence_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
                        
                        biometric_template_sequence = BiometricTemplate(
                            user_id=session.user_id,
                            template_id=template_id_seq,
                            template_type=TemplateType.DYNAMIC,
                            anatomical_embedding=None,
                            dynamic_embedding=dynamic_embedding_sequence,
                            gesture_name="FLUID_SEQUENCE",  # Identificador especial
                            quality_score=0.9,
                            confidence=0.9,
                            enrollment_session=session.session_id,
                            metadata={
                                'modality': 'dynamic',
                                'is_sequence': True,
                                'sequence_type': 'fluid_transition',
                                'gesture_sequence': session.gesture_sequence,
                                'temporal_sequence': sequence_array.tolist(),  # Guardar secuencia completa
                                'sequence_frames': len(sequence_array),
                                'sequence_shape': list(sequence_array.shape),
                                'is_real_data': True,
                                'no_synthetic_data': True,
                                'creation_date': time.time(),
                                'version': "2.1_fluid_sequence",
                                'storage_mode': 'sequence',
                                'data_source': 'fluid_sequence_capture'
                            }
                        )
                        
                        biometric_templates.append(biometric_template_sequence)
                        session.dynamic_sequence_template_id = template_id_seq
                        
                        print(f"✅ Template de secuencia fluida creado: {template_id_seq}")
                        print(f"   - Frames: {len(sequence_array)}")
                        print(f"   - Embedding shape: {dynamic_embedding_sequence.shape}")
                    else:
                        print("⚠️ Secuencia inválida - no se creó template")
                else:
                    print("⚠️ No se encontró dynamic_sequence_data en sesión")
            else:
                print("⚠️ Fase dinámica no completada - sin template de secuencia")
            # =========================================================================
                    
            # Almacenar perfil de usuario
            if self.database.store_user_profile(user_profile):
                print(f"Perfil de usuario {session.user_id} almacenado")
            else:
                print(f"Error almacenando perfil de usuario {session.user_id}")
                return False
            
            # Almacenar todos los templates individuales
            templates_stored = 0
            for template in biometric_templates:
                if self.database.store_biometric_template(template):
                    modality = template.metadata.get('modality', 'unknown')
                    templates_stored += 1
                else:
                    modality = template.metadata.get('modality', 'unknown')
                    print(f"Error almacenando template {modality} indice {template.metadata.get('sample_index')}")
                    return False
            
            print(f"Todos los datos almacenados exitosamente para usuario {session.user_id}")
            print(f"   Total templates guardados: {templates_stored}")
            print(f"   Anatomicos: {len(templates.get('anatomical', []))}")
            print(f"   Dinamicos: {len(templates.get('dynamic', []))}")
            return True
            
        except Exception as e:
            print(f"Error almacenando datos: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False

    def _draw_real_feedback(self, frame: np.ndarray, quality_assessment: Optional[QualityAssessment], 
                   processing_result: ProcessingResult, errors: Optional[List[str]] = None):
        """Dibuja feedback visual en el frame."""
        try:
            if not self.config.show_preview:
                return
            
            WINDOW_NAME = 'BIOMETRICO_FEEDBACK_REAL'
            
            if self.current_session:
                session = self.current_session
                cv2.putText(frame, f"Usuario: {session.user_id}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Gesto: {session.current_gesture}", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Progreso: {session.successful_samples}/{session.total_samples_needed}", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if quality_assessment:
                score = quality_assessment.quality_score
                
                if score >= 0.60:
                    quality_color = (0, 255, 0)
                elif score >= 0.50:
                    quality_color = (0, 200, 100)
                elif score >= 0.40:
                    quality_color = (0, 150, 200)
                elif score >= 0.30:
                    quality_color = (0, 100, 255)
                else:
                    quality_color = (0, 0, 255)
                
                cv2.putText(frame, f"Calidad: {score:.3f}", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
                
                ready_text = "✅ LISTO PARA CAPTURA" if quality_assessment.ready_for_capture else "⏳ Mejorando posición..."
                ready_color = (0, 255, 0) if quality_assessment.ready_for_capture else (0, 255, 255)
                cv2.putText(frame, ready_text, (20, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, ready_color, 2)
                
                if self.current_session and hasattr(self, 'quality_controller'):
                    try:
                        feedback_messages = [
                            f"Estabilidad: {'OK' if quality_assessment.gesture_stable else 'Moviendo'}",
                            f"Visibilidad: {'OK' if quality_assessment.hand_visible else 'Parcial'}",
                            f"Confianza: {quality_assessment.quality_score:.2f}"
                        ]
                        
                        y_offset = 180
                        for message in feedback_messages[:3]:
                            cv2.putText(frame, message, (20, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            y_offset += 25
                    except Exception as e:
                        pass
            
            if errors:
                y_offset = 300
                cv2.putText(frame, "Errores de validación:", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 25
                for error in errors[:3]:
                    cv2.putText(frame, f"- {error}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += 20
            
        except Exception as e:
            print(f"Error dibujando feedback: {e}")
    
    
    def cleanup(self):
        """Limpia recursos del workflow."""
        try:
            self.is_running = False
            self.current_session = None
            self.frame_buffer.clear()
            
            if hasattr(self, 'window_created') and self.window_created:
                cv2.destroyWindow(self.window_name)
                self.window_created = False
                print(f"Ventana {self.window_name} cerrada")
            
            try:
                release_camera()
                print("✅ Cámara global liberada")
            except Exception as e:
                print(f"Error liberando cámara: {e}")
    
            if self.mediapipe_processor:
                self.mediapipe_processor.close()
            
            try:
                release_camera()
                print("✅ Cámara global liberada")
            except Exception as e:
                print(f"Error liberando cámara: {e}")
                global _camera_instance
                _camera_instance = None
                print("🔧 Reset manual ejecutado")
            
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            cv2.waitKey(100)
            
            print("Recursos liberados")
            
        except Exception as e:
            print(f"Error liberando recursos: {e}")

    def _extract_existing_temporal_sequence(self, session):
        """Usa la función temporal existente."""
        try:
            temporal_sequence = self._extract_temporal_sequence_for_dynamic_network()
            
            if temporal_sequence is not None:
                return temporal_sequence.tolist()
            
            return None
            
        except Exception as e:
            print(f"Error usando función temporal: {e}")
            return None

    def _extract_individual_temporal_sequences_from_session(self, session):
        """Extrae secuencias temporales individuales."""
        try:
            individual_sequences = []
            
            for sample in session.samples:
                if (sample.is_valid and 
                    hasattr(sample, 'metadata') and 
                    sample.metadata and
                    'temporal_sequence' in sample.metadata and
                    sample.metadata['temporal_sequence'] is not None):
                    
                    sequence = sample.metadata['temporal_sequence']
                    if len(sequence) >= 3:
                        individual_sequences.append(sequence)
                        
            print(f"✅ Extraídas {len(individual_sequences)} secuencias individuales")
            return individual_sequences
            
        except Exception as e:
            print(f"Error extrayendo secuencias: {e}")
            return []
        
# ====================================================================
# SISTEMA DE ENROLLMENT PRINCIPAL
# ====================================================================

class RealEnrollmentSystem:
    """
    Sistema principal de enrollment REAL.
    Coordina todo el proceso de registro de usuarios.
    Incluye MODO BOOTSTRAP para resolver chicken-and-egg.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Inicializa el sistema de enrollment.
        
        Args:
            config_override: Configuración personalizada (opcional)
        """
        self.logger = get_logger()
        
        default_config = self._load_real_default_config()
        if config_override:
            default_config.update(config_override)
        
        self.config = RealEnrollmentConfig(**default_config)
        
        self.bootstrap_mode = self._check_bootstrap_needed()
        
        self.workflow = RealEnrollmentWorkflow(self.config)
        self.database = get_biometric_database()
        
        self.feedback_manager = get_visual_feedback_manager()
        
        self.active_sessions: Dict[str, RealEnrollmentSession] = {}
        self.session_history: List[RealEnrollmentSession] = []
        
        self.stats = {
            'total_enrollments': 0,
            'successful_enrollments': 0,
            'failed_enrollments': 0,
            'total_samples_captured': 0,
            'total_real_templates_generated': 0,
            'average_duration': 0.0,
            'average_samples_per_user': 0.0,
            'average_quality_score': 0.0,
            'bootstrap_enrollments': 0,
            'networks_trained': False
        }
        
        print("RealEnrollmentSystem inicializado")
        print(f"  - Config: {self.config.samples_per_gesture} muestras/gesto, umbral {self.config.quality_threshold}")
        print(f"  - Bootstrap: {'ACTIVADO' if self.bootstrap_mode else 'DESACTIVADO'}")

    def check_bootstrap_mode(self) -> bool:
        """
        Método público para verificar si el sistema está en modo bootstrap.
        
        Returns:
            bool: True si está en modo bootstrap, False si no
        """
        try:
            self.bootstrap_mode = self._check_bootstrap_needed()
            print(f"Bootstrap mode verificado: {'ACTIVO' if self.bootstrap_mode else 'DESACTIVADO'}")
            return self.bootstrap_mode
        except Exception as e:
            print(f"Error verificando bootstrap mode: {e}")
            return False
        
    def _check_bootstrap_needed(self) -> bool:
        """Verifica si necesitamos modo bootstrap."""
        try:
            try:
                from app.core.siamese_anatomical_network import get_real_siamese_anatomical_network
                from app.core.siamese_dynamic_network import get_real_siamese_dynamic_network
                
                anatomical_net = get_real_siamese_anatomical_network()
                dynamic_net = get_real_siamese_dynamic_network()
                
                if anatomical_net.is_trained and dynamic_net.is_trained:
                    print("🎯 Redes YA ENTRENADAS - Modo normal")
                    return False
                    
            except Exception as e:
                print(f"⚠️ No se pudieron cargar redes: {e}")
            
            if not hasattr(self, 'database') or self.database is None:
                print("🔧 Database no inicializada - Bootstrap")
                return True
            
            try:
                users = self.database.list_users()
                users_with_data = [u for u in users if u.total_templates > 0]
                
                sufficient_users = 0
                for user in users_with_data:
                    user_templates = self.database.list_user_templates(user.user_id)
                    if len(user_templates) >= 15:
                        sufficient_users += 1
                
                bootstrap_needed = sufficient_users < 2
                
                if bootstrap_needed:
                    print("🔧 BOOTSTRAP ACTIVADO:")
                    print(f"   - Usuarios suficientes: {sufficient_users}/2")
                    print(f"   - Redes se entrenarán después del 2º usuario")
                else:
                    print("🎯 MODO NORMAL: Suficientes datos")
                
                return bootstrap_needed
                
            except Exception as db_error:
                print(f"⚠️ Error accediendo DB: {db_error}")
                print("🔧 Activando bootstrap")
                return True
            
        except Exception as e:
            print(f"Error verificando bootstrap: {e}")
            print("🔧 Activando bootstrap")
            return True
    
    def _load_real_default_config(self) -> Dict[str, Any]:
        """Carga configuración por defecto."""
        return {
            'samples_per_gesture': get_config('biometric.enrollment.samples_per_gesture', 8),
            'min_samples_per_gesture': get_config('biometric.enrollment.min_samples_per_gesture', 5),
            'max_samples_per_gesture': get_config('biometric.enrollment.max_samples_per_gesture', 12),
            'quality_threshold': get_config('biometric.enrollment.quality_threshold', 0.60),
            'min_confidence': get_config('biometric.enrollment.min_confidence', 0.65),
            'min_stability_frames': get_config('biometric.enrollment.min_stability_frames', 8),
            'require_all_gestures': get_config('biometric.enrollment.require_all_gestures', True),
            'sample_timeout': get_config('biometric.enrollment.sample_timeout', 120.0),
            'session_timeout': get_config('biometric.enrollment.session_timeout', 3600.0),
            'capture_interval': get_config('biometric.enrollment.capture_interval', 0.8),
            'enable_quality_check': get_config('biometric.enrollment.enable_quality_check', True),
            'enable_duplicate_check': get_config('biometric.enrollment.enable_duplicate_check', True),
            'duplicate_threshold': get_config('biometric.enrollment.duplicate_threshold', 0.92),
            'template_fusion_strategy': get_config('biometric.enrollment.template_fusion_strategy', 'average'),
            'enable_template_optimization': get_config('biometric.enrollment.enable_template_optimization', True),
            'embedding_dimension_check': get_config('biometric.enrollment.embedding_dimension_check', True),
            'show_preview': get_config('biometric.enrollment.show_preview', True),
            'show_quality_feedback': get_config('biometric.enrollment.show_quality_feedback', False),
            'save_enrollment_video': get_config('biometric.enrollment.save_enrollment_video', False)
        }
    
    def start_real_enrollment(self, user_id: str, username: str, 
                              gesture_sequence: List[str],
                              progress_callback: Optional[Callable] = None,
                              error_callback: Optional[Callable] = None) -> str:
        """
        Inicia proceso de enrollment con soporte Bootstrap.
        
        Args:
            user_id: ID único del usuario
            username: Nombre del usuario  
            gesture_sequence: Secuencia de gestos
            progress_callback: Callback de progreso
            error_callback: Callback de errores
            
        Returns:
            ID de sesión de enrollment
        """
        try:
            self.bootstrap_mode = self._check_bootstrap_needed()
            
            print(f"Iniciando enrollment: {user_id}")
            print(f"  - Nombre: {username}")
            print(f"  - Gestos: {' → '.join(gesture_sequence)}")
            print(f"  - Muestras/gesto: {self.config.samples_per_gesture}")
            print(f"  - Bootstrap: {'SÍ' if self.bootstrap_mode else 'NO'}")
            
            if not user_id or not username or not gesture_sequence:
                raise ValueError("user_id, username y gesture_sequence requeridos")
            
            if self.config.enable_duplicate_check:
                existing_user = self.database.get_user(user_id)
                if existing_user:
                    print(f"Usuario {user_id} ya existe - se actualizará")
            
            self.workflow.set_bootstrap_mode(self.bootstrap_mode)
            
            session = self.workflow.start_real_enrollment(
                user_id=user_id,
                username=username,
                gesture_sequence=gesture_sequence,
                progress_callback=progress_callback,
                error_callback=error_callback
            )
            
            if session.status == EnrollmentStatus.FAILED:
                self.stats['failed_enrollments'] += 1
                raise RuntimeError("Error iniciando sesión")
            
            session.is_bootstrap = self.bootstrap_mode
            
            self.active_sessions[session.session_id] = session
            self.stats['total_enrollments'] += 1
            if self.bootstrap_mode:
                self.stats['bootstrap_enrollments'] += 1
            
            print(f"Sesión iniciada: {session.session_id}")
            print(f"  - Muestras necesarias: {session.total_samples_needed}")
            print(f"  - Estado: {session.status.value}")
            print(f"  - Bootstrap: {'SÍ' if self.bootstrap_mode else 'NO'}")
            
            return session.session_id
            
        except Exception as e:
            print(f"Error iniciando enrollment: {e}")
            self.stats['failed_enrollments'] += 1
            if error_callback:
                error_callback(str(e))
            raise
    


    
    
    # ESTA SE AGREGO NUEVA 

    
    def process_enrollment_frame_with_image(self, session_id: str, frame_image: np.ndarray) -> Dict[str, Any]:
        """
        Procesa un frame de enrollment recibido desde el frontend.
        ✅ CORREGIDO: Guardado durante captura en modo Bootstrap según notebook
        
        Args:
            session_id: ID de la sesión activa
            frame_image: Imagen OpenCV (numpy array BGR)
            
        Returns:
            Diccionario con el resultado del procesamiento
        """
        try:
            print(f"🎥 Procesando frame para sesión {session_id}")
            print(f"   Frame shape: {frame_image.shape}")
            
            if session_id not in self.active_sessions:
                print(f"❌ Sesión {session_id} no encontrada")
                return {
                    'error': 'Sesión no encontrada',
                    'session_id': session_id,
                    'status': 'error',
                    'phase': 'error',
                    'progress': 0,
                    'current_gesture': '',
                    'current_gesture_index': 0,
                    'total_gestures': 0,
                    'samples_collected': 0,
                    'samples_needed': 0,
                    'sample_captured': False,
                    'session_completed': False,
                    'is_real_processing': True,
                    'bootstrap_mode': self.bootstrap_mode
                }
            
            session = self.active_sessions[session_id]
            
            # ✅ Procesar frame directamente con MediaPipe
            processing_result = self.workflow.mediapipe_processor.process_frame(frame_image)

            # Verificar si se detectó mano
            if not processing_result or not processing_result.hand_result or not processing_result.hand_result.is_valid:
                return {
                    'session_id': session_id,
                    'status': session.status.value,
                    'phase': session.current_phase.value,
                    'progress': (len(session.samples) / session.total_samples_needed) * 100,
                    'current_gesture': session.current_gesture,
                    'current_gesture_index': session.current_gesture_index,
                    'total_gestures': len(session.gesture_sequence),
                    'samples_collected': len(session.samples),
                    'samples_needed': session.total_samples_needed,
                    'sample_captured': False,
                    'session_completed': False,
                    'message': 'No se detectó mano',
                    'is_real_processing': True,
                    'bootstrap_mode': self.bootstrap_mode
                }

            hand_result = processing_result.hand_result
            gesture_result = processing_result.gesture_result
            
            # Verificar confianza básica
            if hand_result.confidence < 0.8:
                return {
                    'session_id': session_id,
                    'status': session.status.value,
                    'phase': session.current_phase.value,
                    'progress': (len(session.samples) / session.total_samples_needed) * 100,
                    'current_gesture': session.current_gesture,
                    'current_gesture_index': session.current_gesture_index,
                    'total_gestures': len(session.gesture_sequence),
                    'samples_collected': len(session.samples),
                    'samples_needed': session.total_samples_needed,
                    'sample_captured': False,
                    'session_completed': False,
                    'message': f'Confianza baja: {hand_result.confidence:.2f}',
                    'is_real_processing': True,
                    'bootstrap_mode': self.bootstrap_mode
                }
            
            # Verificar gesto correcto
            detected_gesture = gesture_result.gesture_name if gesture_result else None
            if detected_gesture != session.current_gesture:
                return {
                    'session_id': session_id,
                    'status': session.status.value,
                    'phase': session.current_phase.value,
                    'progress': (len(session.samples) / session.total_samples_needed) * 100,
                    'current_gesture': session.current_gesture,
                    'current_gesture_index': session.current_gesture_index,
                    'total_gestures': len(session.gesture_sequence),
                    'samples_collected': len(session.samples),
                    'samples_needed': session.total_samples_needed,
                    'sample_captured': False,
                    'session_completed': False,
                    'message': f'Gesto incorrecto. Esperado: {session.current_gesture}, Detectado: {detected_gesture or "Ninguno"}',
                    'is_real_processing': True,
                    'bootstrap_mode': self.bootstrap_mode
                }
            
            # Control de tiempo entre capturas
            current_time = time.time()
            if session.last_capture_time > 0:
                time_since_last = current_time - session.last_capture_time
                if time_since_last < 1.5:
                    return {
                        'session_id': session_id,
                        'status': session.status.value,
                        'phase': session.current_phase.value,
                        'progress': (len(session.samples) / session.total_samples_needed) * 100,
                        'current_gesture': session.current_gesture,
                        'current_gesture_index': session.current_gesture_index,
                        'total_gestures': len(session.gesture_sequence),
                        'samples_collected': len(session.samples),
                        'samples_needed': session.total_samples_needed,
                        'sample_captured': False,
                        'session_completed': False,
                        'message': 'Espera un momento entre capturas',
                        'is_real_processing': True,
                        'bootstrap_mode': self.bootstrap_mode
                    }
            
            # ✅ CAPTURA VÁLIDA - Extraer características ANATÓMICAS
            try:
                anatomical_features = self.workflow.anatomical_extractor.extract_features(
                    hand_result.landmarks,
                    hand_result.world_landmarks,
                    hand_result.handedness.classification[0].label if hand_result.handedness else 'unknown'
                )
                
                if not anatomical_features:
                    print("Error extrayendo características anatómicas")
                    return {
                        'session_id': session_id,
                        'status': session.status.value,
                        'phase': session.current_phase.value,
                        'progress': (len(session.samples) / session.total_samples_needed) * 100,
                        'current_gesture': session.current_gesture,
                        'current_gesture_index': session.current_gesture_index,
                        'total_gestures': len(session.gesture_sequence),
                        'samples_collected': len(session.samples),
                        'samples_needed': session.total_samples_needed,
                        'sample_captured': False,
                        'session_completed': False,
                        'message': 'Error extrayendo características',
                        'is_real_processing': True,
                        'bootstrap_mode': self.bootstrap_mode
                    }
                
                print(f"✅ Características anatómicas extraídas: {anatomical_features.complete_vector.shape}")
                
            except Exception as e:
                print(f"Error extrayendo características anatómicas: {e}")
                import traceback
                print(traceback.format_exc())
                return {
                    'session_id': session_id,
                    'status': session.status.value,
                    'phase': session.current_phase.value,
                    'progress': (len(session.samples) / session.total_samples_needed) * 100,
                    'current_gesture': session.current_gesture,
                    'current_gesture_index': session.current_gesture_index,
                    'total_gestures': len(session.gesture_sequence),
                    'samples_collected': len(session.samples),
                    'samples_needed': session.total_samples_needed,
                    'sample_captured': False,
                    'session_completed': False,
                    'message': f'Error extrayendo características: {e}',
                    'is_real_processing': True,
                    'bootstrap_mode': self.bootstrap_mode
                }
            
            # ✅ Agregar frame al buffer temporal del extractor dinámico
            try:
                self.workflow.dynamic_extractor.add_frame_real(
                    hand_result.landmarks,
                    session.current_gesture,
                    hand_result.confidence,
                    hand_result.world_landmarks
                )
                print(f"Frame agregado. Buffer: {len(self.workflow.dynamic_extractor.temporal_buffer)}/50")
                # TAMBIÉN agregar a buffer de sesión para secuencia fluida
                if not hasattr(session, 'all_frames_buffer'):
                    session.all_frames_buffer = []
                
                session.all_frames_buffer.append({
                    'landmarks': hand_result.landmarks,
                    'world_landmarks': hand_result.world_landmarks,
                    'gesture': session.current_gesture,
                    'timestamp': time.time()
                })
                
                print(f"Buffer sesión: {len(session.all_frames_buffer)} frames totales")
            except Exception as e:
                print(f"❌ Error agregando frame: {e}")
            
            # ✅ EXTRAER CARACTERÍSTICAS DINÁMICAS
            dynamic_features = None
            temporal_sequence = None
            
            if len(self.workflow.dynamic_extractor.temporal_buffer) >= 10:
                try:
                    buffer_data = []
                    for frame in self.workflow.dynamic_extractor.temporal_buffer:
                        buffer_data.append({
                            'landmarks': frame.landmarks,
                            'gesture': frame.gesture_name,
                            'timestamp': frame.timestamp
                        })
                    
                    dynamic_features = self.workflow.dynamic_extractor.extract_features_from_sequence_real(
                        landmarks_sequence=[frame['landmarks'] for frame in buffer_data],
                        gesture_sequence=[frame['gesture'] for frame in buffer_data],
                        timestamps=[frame['timestamp'] for frame in buffer_data]
                    )
                    
                    if dynamic_features:
                        print(f"✅ Características dinámicas: {dynamic_features.complete_vector.shape}")
                    else:
                        print(f"⏳ Dinámicas: esperando más frames")
                    
                    temporal_sequence = self.workflow._extract_temporal_sequence_for_dynamic_network()
                    if temporal_sequence is not None:
                        print(f"✅ Secuencia temporal: {temporal_sequence.shape}")
                    else:
                        print("⚠️ No se pudo extraer secuencia temporal")
                            
                except Exception as e:
                    print(f"❌ Error dinámicas: {e}")
                    import traceback
                    print(traceback.format_exc())
            else:
                print(f"⏳ Buffer: {len(self.workflow.dynamic_extractor.temporal_buffer)}/50")
            
            # ✅ CREAR MUESTRA COMPLETA
            sample_id = f"{session.user_id}_{session.current_gesture}_{len(session.samples)}"
            
            sample = RealEnrollmentSample(
                sample_id=sample_id,
                user_id=session.user_id,
                sample_type=SampleType.COMBINED,
                gesture_name=session.current_gesture,
                anatomical_features=anatomical_features,
                dynamic_features=dynamic_features,
                confidence=hand_result.confidence,
                timestamp=current_time
            )
            
            sample.is_valid = True
            sample.frame_count = 1
            sample.capture_duration = 0.0
            
            # ✅ Agregar datos temporales DESPUÉS de crear la muestra
            if temporal_sequence is not None and len(temporal_sequence) >= 5:
                sample.temporal_sequence = temporal_sequence
                sample.sequence_length = len(temporal_sequence)
                sample.has_temporal_data = True
                print(f"📊 Datos temporales agregados: {sample.sequence_length} frames")
            else:
                sample.has_temporal_data = False
                sample.temporal_sequence = None
                sample.sequence_length = 0
            
            # Agregar a sesión
            session.samples.append(sample)
            session.last_capture_time = current_time
            
            print(f"✅ Muestra creada: {sample_id}")
            print(f"   Total muestras en sesión: {len(session.samples)}")
            
            # ✅✅✅ GUARDAR EN MODO BOOTSTRAP (CORREGIDO SEGÚN NOTEBOOK) ✅✅✅
            if self.bootstrap_mode:
                try:
                    print("="*70)
                    print("💾 GUARDANDO MUESTRA EN MODO BOOTSTRAP")
                    print("="*70)
                    
                    # ✅ Preparar metadata de muestra para el método enroll_template_bootstrap
                    sample_metadata = {
                        'sample_id': sample_id,
                        'capture_timestamp': current_time,
                        'gesture_sequence_position': session.current_gesture_index,
                        'session_id': session.session_id,
                        'bootstrap_mode': True,
                        'session_username': session.username,
                        'has_temporal_data': sample.has_temporal_data,
                        'temporal_sequence': temporal_sequence.tolist() if temporal_sequence is not None else None,
                        'sequence_length': sample.sequence_length if sample.has_temporal_data else 0,
                        'data_source': 'real_enrollment_capture',
                        'is_real_temporal': temporal_sequence is not None
                    }
                    
                    # ✅ LLAMAR AL MÉTODO enroll_template_bootstrap DEL NOTEBOOK
                    template_id = self.database.enroll_template_bootstrap(
                        user_id=session.user_id,
                        anatomical_features=anatomical_features.complete_vector,
                        gesture_name=session.current_gesture,
                        quality_score=0.85,  # Valor por defecto
                        confidence=float(hand_result.confidence),
                        sample_metadata=sample_metadata
                    )
                    
                    if template_id:
                        print(f"✅ Template bootstrap guardado exitosamente")
                        print(f"   Template ID: {template_id}")
                        print(f"   Usuario: {session.user_id}")
                        print(f"   Gesto: {session.current_gesture}")
                        
                        # Verificar que se guardó
                        templates = self.database.list_user_templates(session.user_id)
                        print(f"📊 Total templates para {session.user_id}: {len(templates)}")
                    else:
                        print("❌ enroll_template_bootstrap retornó None")
                        print("   La muestra NO se guardó en la base de datos")
                        
                except Exception as e:
                    print("="*70)
                    print(f"❌ ERROR GUARDANDO EN MODO BOOTSTRAP: {e}")
                    print("="*70)
                    import traceback
                    print(traceback.format_exc())
            
            # Preparar respuesta base
            samples_this_gesture = len([s for s in session.samples if s.gesture_name == session.current_gesture])
            
            response_base = {
                'session_id': session_id,
                'progress': (len(session.samples) / session.total_samples_needed) * 100,
                'current_gesture': session.current_gesture,
                'current_gesture_index': session.current_gesture_index,
                'total_gestures': len(session.gesture_sequence),
                'samples_collected': len(session.samples),
                'samples_needed': session.total_samples_needed,
                'samples_this_gesture': samples_this_gesture,
                'is_real_processing': True,
                'bootstrap_mode': self.bootstrap_mode
            }
            
            # ✅ Verificar si el gesto actual está completo
            if samples_this_gesture >= self.config.samples_per_gesture:
                print(f"🎉 ¡GESTO '{session.current_gesture}' COMPLETADO!")
                
                # Avanzar al siguiente gesto
                session.current_gesture_index += 1
                
                if session.current_gesture_index >= len(session.gesture_sequence):
                    # ✅✅✅ ENROLLMENT COMPLETADO
                    print("=" * 70)
                    print("🎉 ENROLLMENT REAL COMPLETADO!")
                    print("=" * 70)
                    
                    session.status = EnrollmentStatus.COMPLETED
                    session.current_phase = EnrollmentPhase.ENROLLMENT_COMPLETE
                    session.end_time = time.time()
                    
                    if self.bootstrap_mode:
                        print("=" * 70)
                        print("🔧 MODO BOOTSTRAP: DATOS YA GUARDADOS DURANTE CAPTURA")
                        print(f"   Total muestras guardadas: {len(session.samples)}")
                        print("=" * 70)
                        
                        from time import sleep as time_sleep
                        time_sleep(0.5)
                        
                        db_users = self.database.list_users()
                        print(f"📊 Usuarios en BD: {len(db_users)}")
                        
                        user_found = False
                        for user in db_users:
                            print(f"   - Usuario en DB: {user.user_id}")
                            if user.user_id == session.user_id:
                                user_found = True
                                templates = self.database.list_user_templates(user.user_id)
                                print(f"✅ Usuario {session.user_id} CONFIRMADO con {len(templates)} templates")
                        
                        if not user_found:
                            print("="*70)
                            print(f"❌ Usuario {session.user_id} NO ENCONTRADO en base de datos")
                            print("❌ El guardado FALLÓ durante captura")
                            print("="*70)
                            
                            session.status = EnrollmentStatus.FAILED
                            return {
                                **response_base,
                                'status': EnrollmentStatus.FAILED.value,
                                'phase': EnrollmentPhase.DATABASE_STORAGE.value,
                                'progress': 100.0,
                                'session_completed': True,
                                'sample_captured': False,
                                'message': 'Error: Datos no se guardaron durante captura',
                                'error': 'User not found in database after bootstrap enrollment'
                            }
                        
                        print("="*70)
                        print("🎉🎉🎉 ENROLLMENT BOOTSTRAP COMPLETADO EXITOSAMENTE 🎉🎉🎉")
                        print(f"   Usuario: {session.user_id} con {len(templates)} templates")
                        print("="*70)
                        
                        # ========== AGREGAR TEMPLATE DE SECUENCIA FLUIDA ==========
                        try:
                            print("🎬 GENERANDO TEMPLATE DE SECUENCIA FLUIDA")
                            
                            # Usar buffer de sesión en lugar del extractor
                            if hasattr(session, 'all_frames_buffer'):
                                buffer_size = len(session.all_frames_buffer)
                                print(f"📊 Buffer de sesión: {buffer_size} frames")
                                
                                if buffer_size >= 50:
                                    # Tomar los últimos 50 frames
                                    recent_frames = session.all_frames_buffer[-50:]
                                    temporal_sequence = []
                                    
                                    for frame_data in recent_frames:
                                        frame_features = self._extract_single_frame_features(
                                            frame_data['landmarks'],
                                            frame_data.get('world_landmarks')
                                        )
                                        if frame_features is not None and len(frame_features) == 320:
                                            temporal_sequence.append(frame_features)
                                    
                                    if len(temporal_sequence) >= 50:
                                        sequence_array = np.array(temporal_sequence[:50], dtype=np.float32)
                                        embedding = np.mean(sequence_array, axis=0).flatten()[:128]
                                        
                                        import uuid
                                        template_id = f"{session.user_id}_dynamic_sequence_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
                                        
                                        from app.core.biometric_database import BiometricTemplate, TemplateType
                                        
                                        template = BiometricTemplate(
                                            user_id=session.user_id,
                                            template_id=template_id,
                                            template_type=TemplateType.DYNAMIC,
                                            anatomical_embedding=None,
                                            dynamic_embedding=embedding,
                                            gesture_name="FLUID_SEQUENCE",
                                            quality_score=0.9,
                                            confidence=0.9,
                                            enrollment_session=session.session_id,
                                            metadata={
                                                'is_sequence': True,
                                                'temporal_sequence': sequence_array.tolist(),
                                                'sequence_frames': 50,
                                                'bootstrap_mode': True
                                            }
                                        )
                                        
                                        success = self.database.store_biometric_template(template)
                                        
                                        if success:
                                            print("="*70)
                                            print(f"✅ TEMPLATE SECUENCIA GUARDADO: {template_id}")
                                            print("="*70)
                                        else:
                                            logger.error("❌ Error guardando template")
                                    else:
                                        logger.warning(f"⚠️ Frames válidos insuficientes: {len(temporal_sequence)}/50")
                                else:
                                    logger.warning(f"⚠️ Buffer insuficiente: {buffer_size}/50")
                            else:
                                logger.error("❌ No existe buffer de sesión")
                                
                        except Exception as e:
                            logger.error(f"❌ ERROR: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                        # =========================================================
                        return {
                            **response_base,
                            'status': EnrollmentStatus.COMPLETED.value,
                            'phase': EnrollmentPhase.ENROLLMENT_COMPLETE.value,
                            'progress': 100.0,
                            'session_completed': True,
                            'sample_captured': True,
                            'message': '¡Enrollment completado! Todas las muestras guardadas.',
                            'user_saved': True
                        }
                    
                    else:
                        # MODO NORMAL
                        print("🔄 MODO NORMAL: Llamando a _finalize_real_enrollment")
                        # Aquí iría la lógica de finalización normal
                        pass
                
                else:
                    # Avanzar al siguiente gesto
                    session.current_gesture = session.gesture_sequence[session.current_gesture_index]
                    print(f"🔄 Cambiando al siguiente gesto: {session.current_gesture} ({session.current_gesture_index + 1}/{len(session.gesture_sequence)})")
            
            return {
                **response_base,
                'status': EnrollmentStatus.COLLECTING_SAMPLES.value,
                'phase': EnrollmentPhase.SAMPLE_COLLECTION.value,
                'sample_captured': True,
                'session_completed': False,
                'message': f'Muestra {samples_this_gesture}/{self.config.samples_per_gesture} capturada'
            }
            
        except Exception as e:
            print(f"❌ Error procesando frame: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                'error': str(e),
                'session_id': session_id,
                'status': EnrollmentStatus.FAILED.value,
                'phase': EnrollmentPhase.ENROLLMENT_COMPLETE.value,
                'progress': 0,
                'current_gesture': '',
                'current_gesture_index': 0,
                'total_gestures': 0,
                'samples_collected': 0,
                'samples_needed': 0,
                'sample_captured': False,
                'session_completed': False,
                'is_real_processing': True,
                'bootstrap_mode': self.bootstrap_mode
            }
    
    def process_enrollment_frame(self, session_id: str) -> Dict[str, Any]:
        """
        Procesa un frame para enrollment.
        Incluye FEEDBACK VISUAL en tiempo real.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Información del frame procesado
        """
        try:
            if session_id not in self.active_sessions:
                return {'error': 'Sesión no encontrada', 'is_real': True}
            
            session = self.active_sessions[session_id]
            
            if session.status not in [EnrollmentStatus.COLLECTING_SAMPLES, EnrollmentStatus.IN_PROGRESS]:
                return {
                    'error': f'Sesión no recolectando: {session.status.value}',
                    'is_real': True,
                    'status': session.status.value
                }
            
            sample, visual_feedback = self._process_frame_with_feedback(session)
            
            info = {
                'session_id': session_id,
                'status': session.status.value,
                'phase': session.current_phase.value,
                'progress': session.progress_percentage,
                'current_gesture': session.current_gesture,
                'current_gesture_index': session.current_gesture_index,
                'total_gestures': len(session.gesture_sequence),
                'samples_collected': session.successful_samples,
                'samples_needed': session.total_samples_needed,
                'failed_samples': session.failed_samples,
                'duration': session.duration,
                'sample_captured': sample is not None,
                'is_real_processing': True,
                'no_simulation': True,
                'bootstrap_mode': self.bootstrap_mode,
                'visual_feedback': visual_feedback
            }
            
            if sample:
                info.update({
                    'sample_id': sample.sample_id,
                    'sample_quality': sample.quality_assessment.quality_score if sample.quality_assessment else 0.0,
                    'sample_confidence': sample.confidence,
                    'sample_gesture': sample.gesture_name,
                    'anatomical_embedding_generated': sample.anatomical_embedding is not None,
                    'dynamic_embedding_generated': sample.dynamic_embedding is not None,
                    'sample_validation_errors': sample.validation_errors,
                    'is_bootstrap_sample': getattr(sample, 'is_bootstrap', self.bootstrap_mode)
                })
                
                self.stats['total_samples_captured'] += 1
                if sample.anatomical_embedding is not None:
                    self.stats['total_real_templates_generated'] += 1
                if sample.dynamic_embedding is not None:
                    self.stats['total_real_templates_generated'] += 1
            
            if session.status in [EnrollmentStatus.COMPLETED, EnrollmentStatus.FAILED, EnrollmentStatus.CANCELLED]:
                self._finalize_real_session(session)
                info['session_completed'] = True
                info['final_status'] = session.status.value
                
                if session.status == EnrollmentStatus.COMPLETED and self.bootstrap_mode:
                    training_attempted = self._attempt_bootstrap_training()
                    info['bootstrap_training_attempted'] = training_attempted
            
            return info
            
        except Exception as e:
            print(f"Error procesando frame: {e}")
            return {
                'error': str(e),
                'is_real': True,
                'no_simulation': True
            }

    def _process_frame_with_feedback(self, session: RealEnrollmentSession) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Procesa frame integrando feedback visual.
        
        Returns:
            Tuple de (muestra_capturada, información_feedback)
        """
        try:
            sample = self.workflow.process_real_frame()
            
            quality_assessment = self.workflow.get_current_quality_assessment()
            
            roi_result = getattr(self.workflow, 'last_roi_result', None)
            print(f"🔍 ROI EN FEEDBACK:")
            print(f"   - roi_result es None: {roi_result is None}")
            if roi_result:
                print(f"   - is_valid: {roi_result.is_valid}")
                print(f"   - tiene roi_bbox: {hasattr(roi_result, 'roi_bbox')}")
                print(f"   - roi_bbox value: {getattr(roi_result, 'roi_bbox', 'NO EXISTE')}")
            
            session_info = {
                'current_gesture': session.current_gesture,
                'samples_captured': len(session.samples),
                'samples_needed': self.config.samples_per_gesture,
                'bootstrap_mode': self.bootstrap_mode,
                'total_progress': session.progress_percentage
            }
            
            feedback_messages = self.feedback_manager.generate_real_time_feedback(
                quality_assessment, session.current_gesture, session_info
            )
            
            visual_feedback = {
                'messages': [
                    {
                        'text': msg.text,
                        'level': msg.level.value,
                        'priority': msg.priority,
                        'action': msg.action
                    }
                    for msg in feedback_messages
                ],
                'quality_score': quality_assessment.quality_score if quality_assessment else 0.0,
                'ready_for_capture': quality_assessment.ready_for_capture if quality_assessment else False,
                'overall_valid': quality_assessment.overall_valid if quality_assessment else False,
                'detected_gesture': quality_assessment.validation_details.get('detected_gesture', 'Ninguno') if quality_assessment and quality_assessment.validation_details else 'Ninguno',
                'gesture_confidence': quality_assessment.validation_details.get('gesture_confidence', 0.0) if quality_assessment and quality_assessment.validation_details else 0.0,
                'quality_assessment': quality_assessment,
                'roi_result': roi_result
            }
            
            return sample, visual_feedback
            
        except Exception as e:
            print(f"Error procesando frame con feedback: {e}")
            return None, {'error': str(e), 'messages': []}
        
    def _attempt_bootstrap_training(self) -> bool:
        """
        Intenta entrenar redes después de completar enrollment bootstrap.
        
        Returns:
            True si se inició entrenamiento
        """
        try:
            print("🧠 VERIFICANDO posibilidad de entrenamiento...")
            
            users = self.database.list_users()
            sufficient_users = 0
            total_samples = 0
            
            for user in users:
                user_templates = self.database.list_user_templates(user.user_id)
                if len(user_templates) >= 15:
                    sufficient_users += 1
                    total_samples += len(user_templates)
            
            print(f"📊 Estado: {sufficient_users} usuarios, {total_samples} muestras")
            
            if sufficient_users >= 2:
                print(f"🎉 DATOS SUFICIENTES!")
                print(f"   - {sufficient_users} usuarios con 15+ muestras")
                print(f"   - {total_samples} muestras totales")
                print("🧠 Iniciando entrenamiento automático...")
                
                try:
                    from app.core.siamese_anatomical_network import get_real_siamese_anatomical_network
                    anatomical_net = get_real_siamese_anatomical_network()
                    
                    if anatomical_net.train_with_real_data(self.database):
                        print("✅ Red anatómica entrenada")
                        anatomical_trained = True
                    else:
                        print("❌ Error entrenando anatómica")
                        anatomical_trained = False
                        
                except Exception as e:
                    print(f"❌ Error inicializando anatómica: {e}")
                    anatomical_trained = False
                
                try:
                    from app.core.siamese_dynamic_network import get_real_siamese_dynamic_network
                    dynamic_net = get_real_siamese_dynamic_network()
                    if dynamic_net.train_with_real_data(self.database):
                        print("✅ Red dinámica entrenada")
                        dynamic_trained = True
                    else:
                        print("❌ Error entrenando dinámica")
                        dynamic_trained = False
                        
                except Exception as e:
                    print(f"❌ Error inicializando dinámica: {e}")
                    dynamic_trained = False
                
                if anatomical_trained and dynamic_trained:
                    print("🎯 ENTRENAMIENTO COMPLETO! Desactivando bootstrap...")
                    self.bootstrap_mode = False
                    self.stats['networks_trained'] = True
                    print("✅ Sistema en MODO NORMAL")
                    return True
                else:
                    print("⚠️ Entrenamiento parcial - manteniendo bootstrap")
                    return False
                    
            else:
                print(f"📊 Faltan datos:")
                print(f"   - Usuarios: {sufficient_users}/2")
                print(f"   - Requiere 2 usuarios con 15+ muestras")
                print("🔧 Manteniendo bootstrap")
                return False
                
        except Exception as e:
            print(f"Error intentando entrenamiento: {e}")
            return False

    def get_enrollment_status(self, session_id: str) -> Dict[str, Any]:
        """Obtiene estado detallado de una sesión."""
        try:
            if session_id not in self.active_sessions:
                return {
                    'error': 'Sesión no encontrada',
                    'is_real': True
                }
            
            session = self.active_sessions[session_id]
            
            status_info = {
                'session_id': session_id,
                'user_id': session.user_id,
                'username': session.username,
                'status': session.status.value,
                'phase': session.current_phase.value,
                'progress_percentage': session.progress_percentage,
                'duration': session.duration,
                'is_real_session': True,
                'no_simulation': True,
                'bootstrap_mode': self.bootstrap_mode,
                'is_bootstrap_session': getattr(session, 'is_bootstrap', False)
            }
            
            status_info.update({
                'gesture_sequence': session.gesture_sequence,
                'current_gesture': session.current_gesture,
                'current_gesture_index': session.current_gesture_index,
                'samples_per_gesture': self.config.samples_per_gesture,
                'samples_collected': session.successful_samples,
                'samples_needed': session.total_samples_needed,
                'failed_samples': session.failed_samples
            })
            
            if session.samples:
                valid_samples = [s for s in session.samples if s.is_valid]
                quality_scores = [s.quality_assessment.quality_score for s in valid_samples if s.quality_assessment]
                confidence_scores = [s.confidence for s in valid_samples]
                
                status_info.update({
                    'total_samples': len(session.samples),
                    'valid_samples': len(valid_samples),
                    'average_quality': np.mean(quality_scores) if quality_scores else 0.0,
                    'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                    'samples_with_anatomical_embeddings': len([s for s in valid_samples if s.anatomical_embedding is not None]),
                    'samples_with_dynamic_embeddings': len([s for s in valid_samples if s.dynamic_embedding is not None])
                })
            
            status_info.update({
                'config': {
                    'quality_threshold': self.config.quality_threshold,
                    'min_confidence': self.config.min_confidence,
                    'template_fusion_strategy': self.config.template_fusion_strategy,
                    'enable_quality_check': self.config.enable_quality_check
                }
            })
            
            return status_info
            
        except Exception as e:
            print(f"Error obteniendo estado: {e}")
            return {
                'error': str(e),
                'is_real': True
            }
    
    def cancel_enrollment(self, session_id: str) -> bool:
        """Cancela una sesión de enrollment."""
        try:
            if session_id not in self.active_sessions:
                print(f"Sesión {session_id} no encontrada")
                return False
            
            session = self.active_sessions[session_id]
            session.status = EnrollmentStatus.CANCELLED
            session.end_time = time.time()
            
            self.workflow.is_running = False
            
            print(f"Sesión cancelada: {session_id}")
            print(f"  - Usuario: {session.user_id}")
            print(f"  - Duración: {session.duration:.1f}s")
            print(f"  - Muestras: {session.successful_samples}")
            print(f"  - Bootstrap: {'SÍ' if getattr(session, 'is_bootstrap', False) else 'NO'}")
            
            self._finalize_real_session(session)
            return True
            
        except Exception as e:
            print(f"Error cancelando enrollment: {e}")
            return False
    
    def _finalize_real_session(self, session: RealEnrollmentSession):
        """Finaliza una sesión de enrollment."""
        try:
            print(f"Finalizando sesión: {session.session_id} - Estado: {session.status.value}")
            
            if session.status == EnrollmentStatus.COMPLETED:
                print("🎯 Sesión completada - ejecutando finalización")
                try:
                    self.workflow._finalize_real_enrollment(session)
                    print("✅ Finalización ejecutada")
                except Exception as e:
                    print(f"❌ Error en finalización: {e}")
                    session.status = EnrollmentStatus.FAILED
            
            if session.status == EnrollmentStatus.COMPLETED:
                self.stats['successful_enrollments'] += 1
                
                if session.samples:
                    valid_samples = [s for s in session.samples if s.is_valid]
                    if valid_samples:
                        avg_quality = np.mean([s.quality_assessment.quality_score for s in valid_samples if s.quality_assessment])
                        self.stats['average_quality_score'] = (
                            (self.stats['average_quality_score'] * (self.stats['successful_enrollments'] - 1) + avg_quality) /
                            self.stats['successful_enrollments']
                        )
            elif session.status == EnrollmentStatus.FAILED:
                self.stats['failed_enrollments'] += 1
            
            if self.stats['total_enrollments'] > 0:
                self.stats['average_duration'] = (
                    (self.stats['average_duration'] * (self.stats['total_enrollments'] - 1) + session.duration) /
                    self.stats['total_enrollments']
                )
            
            if self.stats['successful_enrollments'] > 0:
                self.stats['average_samples_per_user'] = (
                    self.stats['total_samples_captured'] / self.stats['successful_enrollments']
                )
            
            self.session_history.append(session)
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
            
            if not self.active_sessions:
                self.workflow.cleanup()
            
            print(f"Sesión finalizada: {session.session_id}")
            
            if session.status == EnrollmentStatus.COMPLETED:
                print("🎯 VERIFICACIÓN FINAL:")
                print(f"   - Usuario: {session.user_id}")
                print(f"   - Muestras válidas: {len([s for s in session.samples if s.is_valid])}")
                print(f"   - Estado: {session.status.value}")
                print("   - Datos guardados: ✅")
            
        except Exception as e:
            print(f"Error finalizando sesión: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema."""
        return {
            'enrollment_stats': self.stats.copy(),
            'active_sessions': len(self.active_sessions),
            'total_users_in_db': len(self.database.list_users()),
            'database_stats': self.database.stats.__dict__,
            'config': {
                'samples_per_gesture': self.config.samples_per_gesture,
                'quality_threshold': self.config.quality_threshold,
                'min_confidence': self.config.min_confidence,
                'template_fusion_strategy': self.config.template_fusion_strategy
            },
            'system_status': {
                'is_real_system': True,
                'no_simulation': True,
                'version': '2.1_bootstrap',
                'components_real': True,
                'bootstrap_mode': self.bootstrap_mode,
                'networks_trained': self.stats['networks_trained']
            }
        }
    
    def force_bootstrap_training(self) -> Dict[str, Any]:
        """
        Fuerza el entrenamiento de redes (para testing/debugging).
        
        Returns:
            Resultado del entrenamiento
        """
        try:
            print("🔧 FORZANDO entrenamiento...")
            
            training_result = {
                'attempted': True,
                'anatomical_success': False,
                'dynamic_success': False,
                'bootstrap_disabled': False,
                'error': None
            }
            
            users = self.database.list_users()
            if len(users) < 2:
                training_result['error'] = f"Insuficientes usuarios: {len(users)}/2"
                return training_result
            
            success = self._attempt_bootstrap_training()
            training_result['bootstrap_disabled'] = not self.bootstrap_mode
            training_result['overall_success'] = success
            
            return training_result
            
        except Exception as e:
            print(f"Error forzando entrenamiento: {e}")
            return {
                'attempted': True,
                'overall_success': False,
                'error': str(e)
            }
    
    def cleanup(self):
        """Limpia recursos del sistema."""
        try:
            print("Limpiando sistema de enrollment")
            
            for session_id in list(self.active_sessions.keys()):
                self.cancel_enrollment(session_id)
            
            self.workflow.cleanup()
            
            release_camera()
            print("✅ Verificación: Cámara liberada")
            
            cv2.destroyAllWindows()
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            cv2.waitKey(50)
            
            print("Sistema de enrollment limpiado")
            
        except Exception as e:
            print(f"Error limpiando sistema: {e}")
            try:
                release_camera()
                cv2.destroyAllWindows()
            except:
                pass
            
# ====================================================================
# FUNCIONES DE CONVENIENCIA PARA INSTANCIA GLOBAL
# ====================================================================

_real_enrollment_system_instance = None

def get_real_enrollment_system(config_override: Optional[Dict[str, Any]] = None) -> RealEnrollmentSystem:
    """
    Obtiene una instancia global del sistema de enrollment.
    
    Args:
        config_override: Configuración personalizada (opcional)
        
    Returns:
        Instancia de RealEnrollmentSystem
    """
    global _real_enrollment_system_instance
    
    if _real_enrollment_system_instance is None:
        _real_enrollment_system_instance = RealEnrollmentSystem(config_override)
    
    return _real_enrollment_system_instance


# Alias para compatibilidad
EnrollmentSystem = RealEnrollmentSystem
get_enrollment_system = get_real_enrollment_system

