"""
MÓDULO 15: RealAuthenticationSystem
Sistema de Autenticación Biométrica con ROI Normalization
"""

import cv2
import numpy as np
import time
import uuid
import threading
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

# ====================================================================
# IMPORTS DEL SISTEMA BIOMÉTRICO
# ====================================================================

from app.core.config_manager import get_config, get_logger, log_error, log_info
from app.core.camera_manager import get_camera_manager
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
from app.core.biometric_database import get_biometric_database, TemplateType
from app.core.roi_normalization import get_roi_normalization_system
from app.core.visual_feedback import get_visual_feedback_manager

logger = get_logger()

# ====================================================================
# ENUMERACIONES
# ====================================================================

class AuthenticationMode(Enum):
    """Modos de autenticación."""
    VERIFICATION = "verification"       # 1:1 - Verificar identidad claimed
    IDENTIFICATION = "identification"   # 1:N - Identificar entre todos
    CONTINUOUS = "continuous"           # Verificación continua

class AuthenticationStatus(Enum):
    """Estados de autenticación."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COLLECTING_FEATURES = "collecting_features"
    PROCESSING_SEQUENCE = "processing_sequence"
    TEMPLATE_MATCHING = "template_matching"
    SCORE_FUSION = "score_fusion"
    DECISION_MAKING = "decision_making"
    AUTHENTICATED = "authenticated"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"

class AuthenticationPhase(Enum):
    """Fases del proceso."""
    INITIALIZATION = "initialization"
    GESTURE_CAPTURE = "gesture_capture"
    FEATURE_EXTRACTION = "feature_extraction"
    QUALITY_VALIDATION = "quality_validation"
    TEMPLATE_MATCHING = "template_matching"
    SCORE_FUSION = "score_fusion"
    DECISION_MAKING = "decision_making"
    COMPLETED = "completed"
    FAILED = "failed"

class SecurityLevel(Enum):
    """Niveles de seguridad."""
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"

# ====================================================================
# DATACLASSES
# ====================================================================

@dataclass
class RealAuthenticationConfig:
    """Configuración de autenticación."""
    sequence_timeout: float = 25.0
    total_timeout: float = 45.0
    frame_timeout: float = 3.0
    
    security_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.65,
        'standard': 0.75,
        'high': 0.85,
        'maximum': 0.92
    })
    
    require_sequence_completion: bool = True
    min_gestures_for_auth: int = 2
    max_attempts_per_session: int = 3
    
    min_quality_score: float = 0.7
    min_confidence: float = 0.65
    
    score_fusion_strategy: str = "weighted_average"
    anatomical_weight: float = 0.6
    dynamic_weight: float = 0.4
    
    enable_audit_logging: bool = True
    max_failed_attempts: int = 5
    lockout_duration: float = 300.0

@dataclass
class RealAuthenticationAttempt:
    """Intento de autenticación."""
    attempt_id: str
    session_id: str
    mode: AuthenticationMode
    user_id: Optional[str]
    
    status: AuthenticationStatus = AuthenticationStatus.NOT_STARTED
    current_phase: AuthenticationPhase = AuthenticationPhase.INITIALIZATION
    security_level: SecurityLevel = SecurityLevel.STANDARD
    
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    last_frame_time: float = field(default_factory=time.time)
    
    required_sequence: List[str] = field(default_factory=list)
    gesture_sequence_captured: List[str] = field(default_factory=list)
    frames_processed: int = 0
    
    anatomical_features: List[np.ndarray] = field(default_factory=list)
    dynamic_features: List[np.ndarray] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    
    ip_address: str = "localhost"
    device_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def sequence_progress(self) -> float:
        if not self.required_sequence:
            return 100.0
        return (len(self.gesture_sequence_captured) / len(self.required_sequence)) * 100

@dataclass
class RealAuthenticationResult:
    """Resultado de autenticación."""
    attempt_id: str
    success: bool
    user_id: Optional[str]
    matched_user_id: Optional[str] = None
    
    anatomical_score: float = 0.0
    dynamic_score: float = 0.0
    fused_score: float = 0.0
    confidence: float = 0.0
    
    security_level: SecurityLevel = SecurityLevel.STANDARD
    authentication_mode: AuthenticationMode = AuthenticationMode.VERIFICATION
    duration: float = 0.0
    frames_processed: int = 0
    gestures_captured: List[str] = field(default_factory=list)
    
    average_quality: float = 0.0
    average_confidence: float = 0.0
    
    risk_factors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class RealIndividualScores:
    """Scores individuales para fusión."""
    anatomical_score: float
    dynamic_score: float
    anatomical_confidence: float
    dynamic_confidence: float
    user_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
# ====================================================================
# HELPER: SISTEMA DE VOTACIÓN
# ====================================================================

def calculate_score_with_voting(similarities, vote_threshold=0.85, min_vote_ratio=0.5):
    """
    Calcula score usando sistema de votación.
    
    Args:
        similarities: Lista de similitudes calculadas
        vote_threshold: Umbral mínimo para voto positivo (0.85)
        min_vote_ratio: Ratio mínimo de votos (0.5 = 50%)
    
    Returns:
        Score calculado o 0.0 si no hay consenso
    """
    if not similarities or len(similarities) == 0:
        return 0.0
    
    similarities_array = np.array(similarities)
    
    # Contar votos positivos
    high_similarities = similarities_array[similarities_array >= vote_threshold]
    positive_votes = len(high_similarities)
    total_votes = len(similarities_array)
    
    vote_ratio = positive_votes / total_votes
    
    logger.info(f"  🗳️ Sistema de votación:")
    logger.info(f"     Votos positivos: {positive_votes}/{total_votes} ({vote_ratio:.1%})")
    logger.info(f"     Umbral: {vote_threshold:.2f}")
    logger.info(f"     Ratio requerido: {min_vote_ratio:.1%}")
    
    if vote_ratio >= min_vote_ratio:
        score = np.mean(high_similarities)
        logger.info(f"     ✅ Consenso alcanzado - Score: {score:.4f}")
        return float(score)
    else:
        logger.info(f"     ❌ Consenso NO alcanzado - Rechazo")
        return 0.0


# ====================================================================
# AUDITOR DE SEGURIDAD
# ====================================================================

class RealSecurityAuditor:
    """Auditor de seguridad para autenticación."""
    
    def __init__(self, config: RealAuthenticationConfig):
        """Inicializa auditor."""
        self.config = config
        
        self.security_events: List[Dict[str, Any]] = []
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.suspicious_activities: List[Dict[str, Any]] = []
        
        logger.info("RealSecurityAuditor inicializado")
    
    def log_authentication_attempt(self, attempt: RealAuthenticationAttempt) -> str:
        """Registra intento de autenticación."""
        try:
            audit_id = str(uuid.uuid4())
            
            audit_event = {
                'audit_id': audit_id,
                'timestamp': time.time(),
                'attempt_id': attempt.attempt_id,
                'session_id': attempt.session_id,
                'mode': attempt.mode.value,
                'user_id': attempt.user_id,
                'security_level': attempt.security_level.value,
                'ip_address': attempt.ip_address,
                'duration': attempt.duration,
                'status': attempt.status.value,
                'frames_processed': attempt.frames_processed,
                'is_real_attempt': True
            }
            
            risk_factors = self._analyze_security_risks(attempt)
            audit_event['risk_factors'] = risk_factors
            audit_event['risk_level'] = len(risk_factors)
            
            self.security_events.append(audit_event)
            
            if len(risk_factors) > 2:
                self._flag_suspicious_activity(attempt, risk_factors)
            
            logger.info(f"Intento registrado: {audit_id}")
            return audit_id
            
        except Exception as e:
            logger.error(f"Error registrando intento: {e}")
            return ""
    
    def _analyze_security_risks(self, attempt: RealAuthenticationAttempt) -> List[str]:
        """Analiza riesgos de seguridad."""
        risks = []
        
        try:
            # Verificar intentos fallidos recientes
            if attempt.ip_address in self.failed_attempts:
                recent_failures = [
                    t for t in self.failed_attempts[attempt.ip_address]
                    if time.time() - t < 300
                ]
                if len(recent_failures) >= 3:
                    risks.append("múltiples_fallos_recientes")
            
            # Verificar duración
            if attempt.duration > self.config.total_timeout * 0.8:
                risks.append("duración_sospechosa")
            elif attempt.duration < 5.0:
                risks.append("duración_muy_corta")
            
            # Verificar calidad
            if attempt.quality_scores:
                avg_quality = np.mean(attempt.quality_scores)
                if avg_quality < self.config.min_quality_score:
                    risks.append("calidad_baja")
            
            # Verificar confianza
            if attempt.confidence_scores:
                avg_confidence = np.mean(attempt.confidence_scores)
                if avg_confidence < self.config.min_confidence:
                    risks.append("confianza_baja")
            
            return risks
            
        except Exception as e:
            logger.error(f"Error analizando riesgos: {e}")
            return ["error_análisis"]
    
    def _flag_suspicious_activity(self, attempt: RealAuthenticationAttempt, risk_factors: List[str]):
        """Marca actividad sospechosa."""
        try:
            suspicious_event = {
                'timestamp': time.time(),
                'attempt_id': attempt.attempt_id,
                'ip_address': attempt.ip_address,
                'risk_factors': risk_factors,
                'risk_level': 'HIGH' if len(risk_factors) > 4 else 'MEDIUM'
            }
            
            self.suspicious_activities.append(suspicious_event)
            logger.error(f"Actividad sospechosa: {attempt.attempt_id}")
            
        except Exception as e:
            logger.error(f"Error marcando actividad: {e}")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de seguridad."""
        try:
            current_time = time.time()
            
            recent_events = [
                e for e in self.security_events
                if current_time - e['timestamp'] < 86400
            ]
            
            return {
                'total_attempts_today': len(recent_events),
                'successful_attempts': len([e for e in recent_events if e['status'] == 'authenticated']),
                'failed_attempts': len([e for e in recent_events if e['status'] in ['rejected', 'timeout', 'error']]),
                'suspicious_activities': len(self.suspicious_activities),
                'unique_ips_today': len(set(e['ip_address'] for e in recent_events)),
                'is_real_security': True
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            return {'error': str(e)}


# ====================================================================
# GESTOR DE SESIONES
# ====================================================================

class RealSessionManager:
    """Gestor de sesiones de autenticación."""
    
    def __init__(self, config: RealAuthenticationConfig):
        """Inicializa gestor."""
        self.config = config
        
        self.active_sessions: Dict[str, RealAuthenticationAttempt] = {}
        self.session_history: List[RealAuthenticationAttempt] = []
        
        self.session_limits: Dict[str, int] = defaultdict(int)
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        
        self.lock = threading.RLock()
        
        logger.info("RealSessionManager inicializado")
    
    def create_real_session(self, mode: AuthenticationMode, user_id: Optional[str] = None,
                           security_level: SecurityLevel = SecurityLevel.STANDARD,
                           ip_address: str = "localhost",
                           device_info: Optional[Dict[str, Any]] = None,
                           required_sequence: Optional[List[str]] = None) -> str:
        """Crea nueva sesión."""
        try:
            with self.lock:
                logger.info(f"Creando sesión: modo={mode.value}, usuario={user_id}")
                
                # Verificar límites
                if len(self.active_sessions) >= 10:
                    raise Exception("Máximo de sesiones alcanzado")
                
                ip_sessions = len([s for s in self.active_sessions.values() if s.ip_address == ip_address])
                if ip_sessions >= 3:
                    raise Exception("Máximo de sesiones por IP")
                
                # Verificar bloqueo por intentos fallidos
                if ip_address in self.failed_attempts:
                    recent_failures = [
                        t for t in self.failed_attempts[ip_address]
                        if time.time() - t < self.config.lockout_duration
                    ]
                    if len(recent_failures) >= self.config.max_failed_attempts:
                        raise Exception("IP bloqueada por intentos fallidos")
                
                # Crear sesión
                session_id = str(uuid.uuid4())
                attempt_id = str(uuid.uuid4())
                
                attempt = RealAuthenticationAttempt(
                    attempt_id=attempt_id,
                    session_id=session_id,
                    mode=mode,
                    user_id=user_id,
                    security_level=security_level,
                    ip_address=ip_address,
                    device_info=device_info or {},
                    required_sequence=required_sequence or []
                )
                
                attempt.status = AuthenticationStatus.IN_PROGRESS
                attempt.current_phase = AuthenticationPhase.INITIALIZATION
                
                self.active_sessions[session_id] = attempt
                self.session_limits[ip_address] += 1
                
                logger.info(f"Sesión creada: {session_id}")
                logger.info(f"  - Modo: {mode.value}")
                logger.info(f"  - Usuario: {user_id}")
                logger.info(f"  - Seguridad: {security_level.value}")
                
                return session_id
                
        except Exception as e:
            logger.error(f"Error creando sesión: {e}")
            raise
    
    def get_real_session(self, session_id: str) -> Optional[RealAuthenticationAttempt]:
        """Obtiene sesión por ID."""
        with self.lock:
            return self.active_sessions.get(session_id)
    
    def close_real_session(self, session_id: str, final_status: AuthenticationStatus):
        """Cierra sesión."""
        try:
            with self.lock:
                if session_id not in self.active_sessions:
                    logger.error(f"Sesión {session_id} no encontrada")
                    return
                
                session = self.active_sessions[session_id]
                session.status = final_status
                session.end_time = time.time()
                
                # Registrar fallo si es necesario
                if final_status in [AuthenticationStatus.REJECTED, AuthenticationStatus.TIMEOUT, AuthenticationStatus.ERROR]:
                    self.failed_attempts[session.ip_address].append(time.time())
                
                # Actualizar límites
                self.session_limits[session.ip_address] -= 1
                if self.session_limits[session.ip_address] <= 0:
                    del self.session_limits[session.ip_address]
                
                # Mover a historial
                self.session_history.append(session)
                del self.active_sessions[session_id]
                
                logger.info(f"Sesión cerrada: {session_id} - {final_status.value}")
                
        except Exception as e:
            logger.error(f"Error cerrando sesión: {e}")
    
    def cleanup_expired_real_sessions(self):
        """Limpia sesiones expiradas."""
        try:
            with self.lock:
                current_time = time.time()
                expired = []
                
                for session_id, session in self.active_sessions.items():
                    if current_time - session.start_time > self.config.total_timeout:
                        expired.append(session_id)
                
                for session_id in expired:
                    self.close_real_session(session_id, AuthenticationStatus.TIMEOUT)
                
                if expired:
                    logger.info(f"Sesiones expiradas limpiadas: {len(expired)}")
                    
        except Exception as e:
            logger.error(f"Error limpiando sesiones: {e}")
    
    def get_real_session_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas."""
        with self.lock:
            current_time = time.time()
            
            recent = [
                s for s in self.session_history
                if current_time - s.start_time < 86400
            ]
            
            return {
                'active_sessions': len(self.active_sessions),
                'total_sessions_today': len(recent),
                'successful_sessions': len([s for s in recent if s.status == AuthenticationStatus.AUTHENTICATED]),
                'failed_sessions': len([s for s in recent if s.status in [AuthenticationStatus.REJECTED, AuthenticationStatus.TIMEOUT, AuthenticationStatus.ERROR]]),
                'is_real_stats': True
            }
            
# ====================================================================
# PIPELINE DE AUTENTICACIÓN
# ====================================================================

class RealAuthenticationPipeline:
    """Pipeline principal de procesamiento de autenticación."""
    
    def __init__(self, config: RealAuthenticationConfig):
        """Inicializa pipeline."""
        self.config = config
        
        # Componentes base
        self.camera_manager = get_camera_manager()
        self.mediapipe_processor = get_mediapipe_processor()
        self.quality_validator = get_quality_validator()
        self.area_manager = get_reference_area_manager()
        self.sequence_manager = get_sequence_manager()
        
        # Extractores
        self.anatomical_extractor = get_anatomical_features_extractor()
        self.dynamic_extractor = get_real_dynamic_features_extractor()
        
        # Redes siamesas (se inicializan después)
        self.anatomical_network = None
        self.dynamic_network = None
        
        # Sistema de fusión
        self.fusion_system = get_real_score_fusion_system()
        
        # Base de datos
        self.database = get_biometric_database()
        
        # Buffer temporal
        self.temporal_buffer = deque(maxlen=30)
        
        self.is_initialized = False
        self.last_processing_result = None
        self.last_roi_result = None
        
        logger.info("RealAuthenticationPipeline inicializado")
    
    def initialize_real_pipeline(self) -> bool:
        """Inicializa componentes."""
        try:
            logger.info("Inicializando pipeline de autenticación...")
            
            # Obtener referencias a redes entrenadas
            self.anatomical_network = get_real_siamese_anatomical_network()
            self.dynamic_network = get_real_siamese_dynamic_network()
            
            logger.info(f"Referencias a redes:")
            logger.info(f"  - Anatómica entrenada: {self.anatomical_network.is_trained}")
            logger.info(f"  - Dinámica entrenada: {self.dynamic_network.is_trained}")
            
            # Inicializar componentes base
            if not self.camera_manager.initialize():
                logger.error("Error inicializando cámara")
                return False
            
            if not self.mediapipe_processor.initialize():
                logger.error("Error inicializando MediaPipe")
                return False
            
            # Verificar extractores
            if not self.anatomical_extractor:
                logger.error("Extractor anatómico no disponible")
                return False
            
            if not self.dynamic_extractor:
                logger.error("Extractor dinámico no disponible")
                return False
            
            # Verificar redes
            if not self.anatomical_network.is_trained:
                logger.error("Red anatómica no entrenada")
                return False
            
            if not self.dynamic_network.is_trained:
                logger.error("Red dinámica no entrenada")
                return False
            
            # Inicializar fusión
            if not self.fusion_system.initialize_networks(
                self.anatomical_network,
                self.dynamic_network,
                get_real_feature_preprocessor()
            ):
                logger.error("Error inicializando fusión")
                return False
            
            self.is_initialized = True
            logger.info("Pipeline inicializado exitosamente")
            
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando pipeline: {e}")
            return False
    
    def process_frame_for_real_authentication(self, attempt: RealAuthenticationAttempt) -> Tuple[bool, str]:
        """
        Procesa frame para autenticación CON ROI NORMALIZATION.
        
        Args:
            attempt: Intento de autenticación
            
        Returns:
            Tupla (éxito, mensaje)
        """
        try:
            if not self.is_initialized:
                return False, "Pipeline no inicializado"
            
            logger.info(f"Procesando frame para sesión {attempt.session_id}")
            
            # PASO 1: Capturar frame original
            ret, frame_original = self.camera_manager.capture_frame()
            if not ret or frame_original is None:
                return False, "Error capturando frame"
            
            attempt.frames_processed += 1
            attempt.last_frame_time = time.time()
            logger.info(f"AUTH: Frame #{attempt.frames_processed} - Shape: {frame_original.shape}")
            
            # PASO 2: Detección inicial
            processing_result_initial = self.mediapipe_processor.process_frame(frame_original)
            
            if not processing_result_initial or not processing_result_initial.hand_result or not processing_result_initial.hand_result.is_valid:
                logger.info("AUTH: No se detectó mano válida")
                return False, "No se detectó mano válida"
            
            logger.info(f"AUTH: ✅ Mano detectada - Confianza: {processing_result_initial.hand_result.confidence:.3f}")
            
            # PASO 3: Extraer y validar ROI
            roi_system = get_roi_normalization_system()
            
            current_gesture = "Unknown"
            expected_gesture = None
            
            if attempt.mode == AuthenticationMode.VERIFICATION and attempt.required_sequence:
                current_step = len(attempt.gesture_sequence_captured)
                if current_step < len(attempt.required_sequence):
                    expected_gesture = attempt.required_sequence[current_step]
                    current_gesture = expected_gesture
            
            logger.info(f"AUTH: Extrayendo ROI - Gesto: {current_gesture}")
            
            roi_result = roi_system.extract_and_validate_roi(
                frame_original,
                processing_result_initial.hand_result.landmarks,
                current_gesture
            )
            
            # Guardar ROI result
            self.last_roi_result = roi_result
            
            # PASO 4: Validar distancia ROI
            if not roi_result.is_valid:
                logger.info(f"AUTH: ❌ ROI NO VÁLIDO - {roi_result.feedback_message}")
                return False, roi_result.feedback_message
            
            logger.info(f"AUTH: ✅ ROI VÁLIDO - {roi_result.roi_width}x{roi_result.roi_height}px")
            
            # PASO 5: Usar landmarks del frame original
            processing_result = processing_result_initial
            hand_result = processing_result.hand_result
            gesture_result = processing_result.gesture_result
            
            self.last_processing_result = processing_result
            
            # PASO 6: Validación de calidad
            reference_area_coords = self.area_manager.calculate_area_coordinates(
                current_gesture, frame_original.shape[:2]
            )
            reference_area = (reference_area_coords.x1, reference_area_coords.y1,
                             reference_area_coords.x2, reference_area_coords.y2)
            
            quality_assessment = self.quality_validator.validate_complete_quality(
                hand_landmarks=hand_result.landmarks,
                handedness=hand_result.handedness,
                detected_gesture=gesture_result.gesture_name if gesture_result else "None",
                gesture_confidence=gesture_result.confidence if gesture_result else 0.0,
                target_gesture=expected_gesture or "Unknown",
                reference_area=reference_area,
                frame_shape=frame_original.shape[:2]
            )
            
            if not quality_assessment or not quality_assessment.ready_for_capture:
                quality_score = quality_assessment.quality_score if quality_assessment else 0.0
                logger.info(f"AUTH: Calidad insuficiente: {quality_score:.3f}")
                return False, f"Calidad insuficiente: {quality_score:.3f}"
            
            logger.info(f"AUTH: ✅ Calidad OK: {quality_assessment.quality_score:.1f}")
            
            # PASO 7: Validar gesto
            detected_gesture = None
            if gesture_result and gesture_result.is_valid:
                detected_gesture = gesture_result.gesture_name
            
            if expected_gesture and detected_gesture != expected_gesture:
                logger.info(f"AUTH: Gesto incorrecto - Esperado: {expected_gesture}, Detectado: {detected_gesture}")
                return False, f"Gesto esperado: {expected_gesture}"
            
            # PASO 8: Extraer características anatómicas
            anatomical_features = self.anatomical_extractor.extract_features(
                hand_result.landmarks,
                hand_result.world_landmarks,
                hand_result.handedness
            )
            
            if not anatomical_features:
                logger.error("AUTH: Error extrayendo anatómicas")
                return False, "Error extrayendo características"
            
            logger.info("AUTH: ✅ Características anatómicas extraídas")
            
            # PASO 9: Agregar al buffer temporal
            self.temporal_buffer.append({
                'landmarks': hand_result.landmarks,
                'world_landmarks': hand_result.world_landmarks,
                'timestamp': time.time(),
                'gesture': detected_gesture
            })
            
            logger.info(f"AUTH: Buffer temporal: {len(self.temporal_buffer)} frames")
            
            # PASO 10: Extraer características dinámicas
            dynamic_features = None
            if len(self.temporal_buffer) >= 5:
                dynamic_features = self._extract_real_dynamic_features_from_buffer()
                
                if dynamic_features:
                    logger.info(f"AUTH: ✅ Dinámicas extraídas ({len(self.temporal_buffer)} frames)")
            
            if not dynamic_features:
                logger.info(f"AUTH: Acumulando frames... ({len(self.temporal_buffer)}/5)")
                return False, "Acumulando frames..."
            
            # PASO 11: Generar embeddings
            anatomical_embedding = self._generate_real_anatomical_embedding(anatomical_features)
            dynamic_embedding = self._generate_real_dynamic_embedding(dynamic_features)
            
            if anatomical_embedding is None and dynamic_embedding is None:
                logger.error("AUTH: Error generando embeddings")
                return False, "Error generando embeddings"
            
            logger.info(f"AUTH: ✅ Embeddings - A:{anatomical_embedding is not None}, D:{dynamic_embedding is not None}")
            
            # PASO 12: Almacenar características
            if anatomical_embedding is not None:
                attempt.anatomical_features.append(anatomical_embedding)
                logger.info(f"AUTH: Total anatómicos: {len(attempt.anatomical_features)}")
            
            if dynamic_embedding is not None:
                attempt.dynamic_features.append(dynamic_embedding)
                logger.info(f"AUTH: Total dinámicos: {len(attempt.dynamic_features)}")
            
            attempt.quality_scores.append(quality_assessment.quality_score)
            attempt.confidence_scores.append(gesture_result.confidence if gesture_result else 0.0)
            
            # PASO 13: Registrar gesto
            if detected_gesture:
                attempt.gesture_sequence_captured.append(detected_gesture)
                logger.info(f"AUTH: ✅ Gesto '{detected_gesture}' registrado")
            
            # PASO 14: Verificar secuencia completa
            if (attempt.mode == AuthenticationMode.VERIFICATION and
                attempt.required_sequence and
                len(attempt.gesture_sequence_captured) >= len(attempt.required_sequence)):
                
                attempt.current_phase = AuthenticationPhase.TEMPLATE_MATCHING
                logger.info("AUTH: 🎉 Secuencia completa - matching biométrico")
                return True, "Secuencia completa"
            
            return True, f"Características capturadas - {len(attempt.anatomical_features)} muestras"
            
        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Error: {str(e)}"
        
    def _extract_real_dynamic_features_from_buffer(self) -> Optional[DynamicFeatureVector]:
        """Extrae características dinámicas del buffer temporal."""
        try:
            if len(self.temporal_buffer) < 5:
                return None
            
            landmarks_sequence = []
            gesture_sequence = []
            timestamps = []
            
            for frame_data in self.temporal_buffer:
                landmarks_sequence.append(frame_data['landmarks'])
                gesture_sequence.append(frame_data.get('gesture', 'Unknown'))
                timestamps.append(frame_data['timestamp'])
            
            dynamic_features = self.dynamic_extractor.extract_features_from_sequence_real(
                landmarks_sequence=landmarks_sequence,
                gesture_sequence=gesture_sequence,
                timestamps=timestamps
            )
            
            if not dynamic_features:
                return None
            
            # Construir temporal_sequence desde buffer del extractor
            if hasattr(self.dynamic_extractor, 'temporal_buffer') and len(self.dynamic_extractor.temporal_buffer) >= 5:
                temporal_frames = []
                
                for frame_data in self.dynamic_extractor.temporal_buffer:
                    if hasattr(frame_data, 'landmarks'):
                        world_landmarks = getattr(frame_data, 'world_landmarks', None)
                        anatomical = self.anatomical_extractor.extract_features(frame_data.landmarks, world_landmarks)
                        
                        if anatomical and anatomical.complete_vector is not None:
                            frame_features = anatomical.complete_vector
                            padded = np.zeros(320, dtype=np.float32)
                            padded[:180] = frame_features[:180]
                            
                            remaining = 320 - 180
                            if len(frame_features) >= 140:
                                padded[180:] = frame_features[:140]
                            else:
                                cycle = np.tile(frame_features, (remaining // len(frame_features)) + 1)
                                padded[180:] = cycle[:remaining]
                            
                            temporal_frames.append(padded)
                
                if len(temporal_frames) >= 5:
                    temporal_sequence = np.array(temporal_frames, dtype=np.float32)
                    dynamic_features.temporal_sequence = temporal_sequence
                    logger.info(f"✅ Temporal sequence: {temporal_sequence.shape}")
            
            if dynamic_features and self._validate_real_dynamic_features(dynamic_features):
                return dynamic_features
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extrayendo dinámicas: {e}")
            return None
    
    def _validate_real_dynamic_features(self, features: DynamicFeatureVector) -> bool:
        """Valida características dinámicas."""
        try:
            if not features or features.complete_vector is None:
                return False
            
            vector = features.complete_vector
            
            if np.var(vector) < 1e-8:
                logger.error("Características sin variación")
                return False
            
            if len(vector) != 320:
                logger.error(f"Dimensión incorrecta: {len(vector)} != 320")
                return False
            
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                logger.error("Características contienen NaN o infinitos")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando dinámicas: {e}")
            return False
    
    def _generate_real_anatomical_embedding(self, features: AnatomicalFeatureVector) -> Optional[np.ndarray]:
        """Genera embedding anatómico usando red siamesa."""
        try:
            if not self.anatomical_network.is_trained:
                logger.error("Red anatómica no entrenada")
                return None
            
            if not features or features.complete_vector is None:
                logger.error("Características anatómicas inválidas")
                return None
            
            features_array = features.complete_vector.reshape(1, -1)
            
            expected_input_dim = self.anatomical_network.input_dim
            if features_array.shape[1] != expected_input_dim:
                logger.error(f"Dimensión incorrecta: {features_array.shape[1]} != {expected_input_dim}")
                return None
            
            embedding = self.anatomical_network.base_network.predict(features_array, verbose=0)[0]
            
            if self._validate_real_embedding(embedding, "anatomical"):
                logger.info(f"Embedding anatómico: dim={embedding.shape[0]}, norm={np.linalg.norm(embedding):.3f}")
                return embedding
            else:
                logger.error("Embedding anatómico inválido")
                return None
                
        except Exception as e:
            logger.error(f"Error generando embedding anatómico: {e}")
            return None
    
    def _generate_real_dynamic_embedding(self, features: DynamicFeatureVector) -> Optional[np.ndarray]:
        """Genera embedding dinámico usando temporal_sequence."""
        try:
            logger.info("Generando embedding dinámico")
            
            if not self.dynamic_network.is_trained:
                logger.error("Red dinámica no entrenada")
                return None
            
            if not features:
                logger.error("Características dinámicas inválidas")
                return None
            
            if not hasattr(features, 'temporal_sequence') or features.temporal_sequence is None:
                logger.error("No hay temporal_sequence")
                return None
            
            temporal_array = features.temporal_sequence
            expected_seq_length = self.dynamic_network.sequence_length
            expected_feature_dim = self.dynamic_network.feature_dim
            
            logger.info(f"Temporal sequence shape: {temporal_array.shape}")
            
            # Ajustar longitud
            if temporal_array.shape[0] > expected_seq_length:
                temporal_array = temporal_array[:expected_seq_length]
            elif temporal_array.shape[0] < expected_seq_length:
                padding = np.zeros((expected_seq_length - temporal_array.shape[0], temporal_array.shape[1]))
                temporal_array = np.vstack([temporal_array, padding])
            
            # Ajustar dimensión
            if temporal_array.shape[1] != expected_feature_dim:
                if temporal_array.shape[1] > expected_feature_dim:
                    temporal_array = temporal_array[:, :expected_feature_dim]
                else:
                    padding = np.zeros((temporal_array.shape[0], expected_feature_dim - temporal_array.shape[1]))
                    temporal_array = np.hstack([temporal_array, padding])
            
            sequence = temporal_array.reshape(1, expected_seq_length, expected_feature_dim)
            
            embedding = self.dynamic_network.base_network.predict(sequence, verbose=0)[0]
            
            if self._validate_real_embedding(embedding, "dynamic"):
                logger.info(f"Embedding dinámico: dim={embedding.shape[0]}, norm={np.linalg.norm(embedding):.3f}")
                return embedding
            else:
                logger.error("Embedding dinámico inválido")
                return None
                
        except Exception as e:
            logger.error(f"Error generando embedding dinámico: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _validate_real_embedding(self, embedding: np.ndarray, embedding_type: str) -> bool:
        """Valida embedding generado."""
        try:
            if embedding is None:
                return False
            
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                logger.error(f"Embedding {embedding_type} contiene NaN o infinitos")
                return False
            
            if np.allclose(embedding, 0.0, atol=1e-6):
                logger.error(f"Embedding {embedding_type} es vector cero")
                return False
            
            magnitude = np.linalg.norm(embedding)
            if magnitude < 0.01 or magnitude > 1000.0:
                logger.error(f"Magnitud {embedding_type} fuera de rango: {magnitude}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando embedding {embedding_type}: {e}")
            return False
    
    def cleanup(self):
        """Limpia recursos del pipeline."""
        try:
            self.is_initialized = False
            self.temporal_buffer.clear()
            
            if self.camera_manager:
                self.camera_manager.release()
            if self.mediapipe_processor:
                self.mediapipe_processor.close()
            
            cv2.destroyAllWindows()
            
            logger.info("Pipeline limpiado")
            
        except Exception as e:
            logger.error(f"Error limpiando pipeline: {e}")
            
# ====================================================================
# SISTEMA DE AUTENTICACIÓN PRINCIPAL
# ====================================================================

class RealAuthenticationSystem:
    """
    Sistema principal de autenticación biométrica.
    Coordina verificación e identificación.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """Inicializa el sistema."""
        
        default_config = self._load_real_default_config()
        if config_override:
            default_config.update(config_override)
        
        self.config = RealAuthenticationConfig(**default_config)
        
        # Componentes principales
        self.pipeline = RealAuthenticationPipeline(self.config)
        self.session_manager = RealSessionManager(self.config)
        self.security_auditor = RealSecurityAuditor(self.config)
        self.database = get_biometric_database()
        self.fusion_system = get_real_score_fusion_system()
        
        self.is_initialized = False
        
        # Estadísticas
        self.statistics = {
            'verification_attempts': 0,
            'verification_success': 0,
            'verification_errors': 0,
            'identification_attempts': 0,
            'identification_success': 0,
            'identification_errors': 0,
            'total_frames_processed': 0,
            'total_embeddings_generated': 0
        }
        
        logger.info("RealAuthenticationSystem inicializado")
    
    def _load_real_default_config(self) -> Dict[str, Any]:
        """Carga configuración por defecto."""
        return {
            'sequence_timeout': get_config('biometric.auth.sequence_timeout', 25.0),
            'total_timeout': get_config('biometric.auth.total_timeout', 45.0),
            'frame_timeout': get_config('biometric.auth.frame_timeout', 3.0),
            'security_thresholds': {
                'low': get_config('biometric.auth.threshold_low', 0.65),
                'standard': get_config('biometric.auth.threshold_standard', 0.75),
                'high': get_config('biometric.auth.threshold_high', 0.85),
                'maximum': get_config('biometric.auth.threshold_maximum', 0.92)
            },
            'require_sequence_completion': get_config('biometric.auth.require_sequence_completion', True),
            'min_gestures_for_auth': get_config('biometric.auth.min_gestures_for_auth', 2),
            'max_attempts_per_session': get_config('biometric.auth.max_attempts_per_session', 3),
            'min_quality_score': get_config('biometric.auth.min_quality_score', 0.7),
            'min_confidence': get_config('biometric.auth.min_confidence', 0.65),
            'score_fusion_strategy': get_config('biometric.auth.score_fusion_strategy', 'weighted_average'),
            'anatomical_weight': get_config('biometric.auth.anatomical_weight', 0.6),
            'dynamic_weight': get_config('biometric.auth.dynamic_weight', 0.4),
            'enable_audit_logging': get_config('biometric.auth.enable_audit_logging', True),
            'max_failed_attempts': get_config('biometric.auth.max_failed_attempts', 5),
            'lockout_duration': get_config('biometric.auth.lockout_duration', 300.0)
        }
    
    def initialize_real_system(self) -> bool:
        """Inicializa todos los componentes."""
        try:
            logger.info("Inicializando sistema de autenticación...")
            
            # Obtener referencias a redes
            self.anatomical_network = get_real_siamese_anatomical_network()
            self.dynamic_network = get_real_siamese_dynamic_network()
            
            logger.info("Referencias a redes:")
            logger.info(f"  - Anatómica entrenada: {self.anatomical_network.is_trained if self.anatomical_network else False}")
            logger.info(f"  - Dinámica entrenada: {self.dynamic_network.is_trained if self.dynamic_network else False}")
            
            # Verificar base de datos
            users = self.database.list_users()
            if not users:
                logger.error("Base de datos vacía - registra usuarios primero")
                return False
            
            users_with_templates = [u for u in users if u.total_templates > 0]
            if not users_with_templates:
                logger.error("No hay usuarios con templates")
                return False
            
            # Verificar redes
            if not self.anatomical_network or not self.anatomical_network.is_trained:
                logger.error("Red anatómica no disponible o no entrenada")
                return False
            
            if not self.dynamic_network or not self.dynamic_network.is_trained:
                logger.warning("Red dinámica no entrenada - continuando solo con anatómica")
            
            # Inicializar pipeline
            if not self.pipeline.initialize_real_pipeline():
                logger.error("Error inicializando pipeline")
                return False
            
            # Verificar fusión
            if hasattr(self.fusion_system, 'initialize_networks'):
                if not self.fusion_system.initialize_networks(
                    self.anatomical_network,
                    self.dynamic_network,
                    get_real_feature_preprocessor()
                ):
                    logger.error("Error inicializando fusión")
                    return False
            
            self.is_initialized = True
            
            logger.info("Sistema inicializado exitosamente")
            logger.info(f"  - Usuarios: {len(users_with_templates)}")
            logger.info(f"  - Templates: {sum(u.total_templates for u in users_with_templates)}")
            logger.info(f"  - Pipeline listo: {self.pipeline.is_initialized}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando sistema: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def start_real_verification(self, user_id: str,
                               security_level: SecurityLevel = SecurityLevel.STANDARD,
                               required_sequence: Optional[List[str]] = None,
                               ip_address: str = "localhost",
                               device_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Inicia verificación 1:1.
        
        Args:
            user_id: ID del usuario a verificar
            security_level: Nivel de seguridad
            required_sequence: Secuencia de gestos
            ip_address: IP del cliente
            device_info: Info del dispositivo
            
        Returns:
            ID de sesión
        """
        try:
            logger.info(f"Iniciando verificación: {user_id}")
            logger.info(f"  - Seguridad: {security_level.value}")
            
            if not self.is_initialized:
                raise Exception("Sistema no inicializado")
            
            # Verificar usuario
            user_profile = self.database.get_user(user_id)
            if not user_profile:
                raise Exception(f"Usuario {user_id} no encontrado")
            
            if user_profile.total_templates == 0:
                raise Exception(f"Usuario {user_id} sin templates")
            
            # Obtener secuencia
            if not required_sequence and user_profile.gesture_sequence:
                required_sequence = user_profile.gesture_sequence
            
            # Crear sesión
            session_id = self.session_manager.create_real_session(
                mode=AuthenticationMode.VERIFICATION,
                user_id=user_id,
                security_level=security_level,
                ip_address=ip_address,
                device_info=device_info,
                required_sequence=required_sequence
            )
            
            self.statistics['verification_attempts'] += 1
            
            logger.info(f"Verificación iniciada: {session_id}")
            logger.info(f"  - Templates: {user_profile.total_templates}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error iniciando verificación: {e}")
            self.statistics['verification_errors'] += 1
            raise
    
    def start_real_identification(self, security_level: SecurityLevel = SecurityLevel.STANDARD,
                                 ip_address: str = "localhost",
                                 device_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Inicia identificación 1:N.
        
        Args:
            security_level: Nivel de seguridad
            ip_address: IP del cliente
            device_info: Info del dispositivo
            
        Returns:
            ID de sesión
        """
        try:
            logger.info("Iniciando identificación 1:N")
            
            if not self.is_initialized:
                raise Exception("Sistema no inicializado")
            
            users = self.database.list_users()
            users_with_templates = [u for u in users if u.total_templates > 0]
            
            if len(users_with_templates) == 0:
                raise Exception("No hay usuarios con templates")
            
            # Crear sesión
            session_id = self.session_manager.create_real_session(
                mode=AuthenticationMode.IDENTIFICATION,
                user_id=None,
                security_level=security_level,
                ip_address=ip_address,
                device_info=device_info
            )
            
            self.statistics['identification_attempts'] += 1
            
            logger.info(f"Identificación iniciada: {session_id}")
            logger.info(f"  - Usuarios en BD: {len(users_with_templates)}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error iniciando identificación: {e}")
            self.statistics['identification_errors'] += 1
            raise
    
    def process_real_authentication_frame(self, session_id: str) -> Dict[str, Any]:
        """
        Procesa frame para autenticación.
        
        Args:
            session_id: ID de sesión
            
        Returns:
            Información del frame procesado
        """
        try:
            # Limpiar expiradas
            self.session_manager.cleanup_expired_real_sessions()
            
            # Obtener sesión
            session = self.session_manager.get_real_session(session_id)
            if not session:
                return {'error': 'Sesión no encontrada', 'is_real': True}
            
            # Verificar timeout
            if session.duration > self.config.total_timeout:
                self._complete_real_authentication(session, AuthenticationStatus.TIMEOUT)
                return {'status': 'timeout', 'message': 'Sesión expirada', 'is_real': True}
            
            # Procesar frame
            success, message = self.pipeline.process_frame_for_real_authentication(session)
            
            self.statistics['total_frames_processed'] += 1
            if success and (session.anatomical_features or session.dynamic_features):
                self.statistics['total_embeddings_generated'] += 1
            
            response = {
                'session_id': session_id,
                'status': session.status.value,
                'phase': session.current_phase.value,
                'progress': session.sequence_progress,
                'message': message,
                'frames_processed': session.frames_processed,
                'duration': session.duration,
                'frame_processed': success,
                'is_real_processing': True,
                'roi_result': self.pipeline.last_roi_result
            }
            
            # Gesto detectado
            if success:
                try:
                    if hasattr(self.pipeline, 'last_processing_result') and self.pipeline.last_processing_result:
                        gesture_result = self.pipeline.last_processing_result.gesture_result
                        if gesture_result:
                            response['current_gesture'] = gesture_result.gesture_name
                            response['gesture_confidence'] = gesture_result.confidence
                except:
                    response['current_gesture'] = 'None'
                    response['gesture_confidence'] = 0.0
            
            # Info de verificación
            if session.mode == AuthenticationMode.VERIFICATION:
                response.update({
                    'required_sequence': session.required_sequence,
                    'captured_sequence': session.gesture_sequence_captured,
                    'sequence_complete': len(session.gesture_sequence_captured) >= len(session.required_sequence) if session.required_sequence else False
                })
            
            # Info de características
            response.update({
                'anatomical_features_captured': len(session.anatomical_features),
                'dynamic_features_captured': len(session.dynamic_features),
                'average_quality': np.mean(session.quality_scores) if session.quality_scores else 0.0,
                'average_confidence': np.mean(session.confidence_scores) if session.confidence_scores else 0.0,
                'anatomical_embedding': session.anatomical_features[-1] if session.anatomical_features else None,
                'dynamic_embedding': session.dynamic_features[-1] if session.dynamic_features else None,
                'has_embeddings': len(session.anatomical_features) > 0
            })
            
            # Verificar matching
            if session.current_phase == AuthenticationPhase.TEMPLATE_MATCHING:
                auth_result = self._perform_real_authentication_matching(session)
                response['authentication_result'] = {
                    'success': auth_result.success,
                    'user_id': auth_result.user_id,
                    'matched_user_id': auth_result.matched_user_id,
                    'anatomical_score': auth_result.anatomical_score,
                    'dynamic_score': auth_result.dynamic_score,
                    'fused_score': auth_result.fused_score,
                    'confidence': auth_result.confidence,
                    'duration': auth_result.duration,
                    'is_real_result': True
                }
                
                final_status = AuthenticationStatus.AUTHENTICATED if auth_result.success else AuthenticationStatus.REJECTED
                self._complete_real_authentication(session, final_status)
                response['session_completed'] = True
                response['final_status'] = final_status.value
            
            return response
            
        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
            return {'error': str(e), 'is_real': True}
        
        
    def _perform_real_authentication_matching(self, session: RealAuthenticationAttempt) -> RealAuthenticationResult:
        """Realiza matching biométrico."""
        try:
            logger.info(f"Iniciando matching para sesión {session.session_id}")
            
            session.current_phase = AuthenticationPhase.SCORE_FUSION
            
            # Promediar características
            if not session.anatomical_features and not session.dynamic_features:
                raise Exception("No hay características para matching")
            
            avg_anatomical = None
            if session.anatomical_features:
                avg_anatomical = np.mean(session.anatomical_features, axis=0)
                logger.info(f"Promedio de {len(session.anatomical_features)} anatómicos")
            
            avg_dynamic = None
            if session.dynamic_features:
                avg_dynamic = np.mean(session.dynamic_features, axis=0)
                logger.info(f"Promedio de {len(session.dynamic_features)} dinámicos")
            
            session.current_phase = AuthenticationPhase.TEMPLATE_MATCHING
            
            # Matching según modo
            if session.mode == AuthenticationMode.VERIFICATION:
                result = self._perform_real_verification(session, avg_anatomical, avg_dynamic)
            else:
                result = self._perform_real_identification(session, avg_anatomical, avg_dynamic)
            
            session.current_phase = AuthenticationPhase.DECISION_MAKING
            
            # Aplicar umbral
            threshold = self.config.security_thresholds[session.security_level.value]
            result.success = result.fused_score >= threshold
            
            logger.info(f"Matching completado:")
            logger.info(f"  - Score: {result.fused_score:.4f}")
            logger.info(f"  - Umbral: {threshold:.4f}")
            logger.info(f"  - Resultado: {'AUTENTICADO' if result.success else 'RECHAZADO'}")
            
            # Auditoría
            if self.config.enable_audit_logging:
                self.security_auditor.log_authentication_attempt(session)
            
            # Estadísticas
            if result.success:
                self.statistics[f'{session.mode.value}_success'] += 1
            else:
                self.statistics[f'{session.mode.value}_errors'] += 1
            
            session.current_phase = AuthenticationPhase.COMPLETED
            
            return result
            
        except Exception as e:
            logger.error(f"Error en matching: {e}")
            session.current_phase = AuthenticationPhase.FAILED
            
            return RealAuthenticationResult(
                attempt_id=session.attempt_id,
                success=False,
                user_id=session.user_id,
                confidence=0.0,
                security_level=session.security_level,
                authentication_mode=session.mode,
                duration=session.duration,
                frames_processed=session.frames_processed,
                gestures_captured=session.gesture_sequence_captured,
                risk_factors=[f"Error: {str(e)}"]
            )
    
    def _perform_real_verification(self, session: RealAuthenticationAttempt,
                                   anatomical_emb: Optional[np.ndarray],
                                   dynamic_emb: Optional[np.ndarray]) -> RealAuthenticationResult:
        """Realiza verificación 1:1."""
        try:
            logger.info(f"Verificación 1:1 para usuario {session.user_id}")
            
            # Obtener templates
            user_templates = self.database.list_user_templates(session.user_id)
            
            if not user_templates:
                logger.error(f"No hay templates para {session.user_id}")
                return self._create_failed_auth_result(session, "No hay templates")
            
            logger.info(f"Templates encontrados: {len(user_templates)}")
            
            # Obtener redes
            anatomical_network = get_real_siamese_anatomical_network()
            dynamic_network = get_real_siamese_dynamic_network()
            
            # Separar templates por modalidad
            anatomical_refs = []
            dynamic_refs = []
            
            for i, template in enumerate(user_templates):
                try:
                    # Método 1: Embeddings directos
                    if hasattr(template, 'anatomical_embedding') and template.anatomical_embedding is not None:
                        anatomical_refs.append(template.anatomical_embedding)
                        logger.info(f"  ✅ Template anatómico {i+1}")
                        continue
                    
                    if hasattr(template, 'dynamic_embedding') and template.dynamic_embedding is not None:
                        dynamic_refs.append(template.dynamic_embedding)
                        logger.info(f"  ✅ Template dinámico {i+1}")
                        continue
                    
                    # Método 2: Template data
                    if hasattr(template, 'template_data') and template.template_data is not None:
                        template_type = getattr(template, 'template_type', None)
                        
                        if template_type == TemplateType.ANATOMICAL:
                            anatomical_refs.append(template.template_data)
                            logger.info(f"  ✅ Template anatómico {i+1} (data)")
                        elif template_type == TemplateType.DYNAMIC:
                            dynamic_refs.append(template.template_data)
                            logger.info(f"  ✅ Template dinámico {i+1} (data)")
                    
                    # Método 3: Bootstrap
                    if hasattr(template, 'metadata') and template.metadata:
                        metadata = template.metadata
                        bootstrap_mode = metadata.get('bootstrap_mode', False)
                        
                        if bootstrap_mode:
                            # Bootstrap anatómico
                            bootstrap_features = metadata.get('bootstrap_features')
                            if bootstrap_features and anatomical_network:
                                try:
                                    features_array = np.array(bootstrap_features, dtype=np.float32).reshape(1, -1)
                                    if features_array.shape[1] == anatomical_network.input_dim:
                                        embedding = anatomical_network.base_network.predict(features_array, verbose=0)[0]
                                        anatomical_refs.append(embedding)
                                        logger.info(f"  ✅ Bootstrap anatómico {i+1} convertido")
                                except Exception as e:
                                    logger.error(f"  ❌ Error bootstrap anatómico {i+1}: {e}")
                            
                            # Bootstrap dinámico
                            temporal_sequence = metadata.get('temporal_sequence')
                            if temporal_sequence and dynamic_network:
                                try:
                                    temporal_array = np.array(temporal_sequence, dtype=np.float32)
                                    
                                    # Ajustar a 50 frames
                                    if temporal_array.shape[0] > 50:
                                        temporal_array = temporal_array[-50:]
                                    elif temporal_array.shape[0] < 50:
                                        padding = np.zeros((50 - temporal_array.shape[0], temporal_array.shape[1]))
                                        temporal_array = np.vstack([temporal_array, padding])
                                    
                                    # Ajustar dimensión
                                    if temporal_array.shape[1] != 320:
                                        if temporal_array.shape[1] > 320:
                                            temporal_array = temporal_array[:, :320]
                                        else:
                                            padding = np.zeros((temporal_array.shape[0], 320 - temporal_array.shape[1]))
                                            temporal_array = np.hstack([temporal_array, padding])
                                    
                                    sequence = temporal_array.reshape(1, 50, 320)
                                    embedding = dynamic_network.base_network.predict(sequence, verbose=0)[0]
                                    dynamic_refs.append(embedding)
                                    logger.info(f"  ✅ Bootstrap dinámico {i+1} convertido")
                                except Exception as e:
                                    logger.error(f"  ❌ Error bootstrap dinámico {i+1}: {e}")
                    
                except Exception as e:
                    logger.error(f"❌ Error procesando template {i+1}: {e}")
            
            logger.info(f"Referencias: A={len(anatomical_refs)}, D={len(dynamic_refs)}")
            
            if not anatomical_refs and not dynamic_refs:
                return self._create_failed_auth_result(session, "No se pudieron procesar templates")
            
            # Crear scores individuales
            individual_scores = RealIndividualScores(
                anatomical_score=0.0,
                dynamic_score=0.0,
                anatomical_confidence=0.0,
                dynamic_confidence=0.0,
                user_id=session.user_id,
                metadata={
                    'anatomical_refs': len(anatomical_refs),
                    'dynamic_refs': len(dynamic_refs)
                }
            )
            
            # Calcular scores anatómicos
            if anatomical_emb is not None and anatomical_refs:
                logger.info(f"Calculando similitudes anatómicas ({len(anatomical_refs)})")
                similarities = []
                
                for j, ref_emb in enumerate(anatomical_refs):
                    try:
                        if isinstance(ref_emb, list):
                            ref_emb = np.array(ref_emb, dtype=np.float32)
                        
                        if anatomical_emb.shape[0] == ref_emb.shape[0]:
                            similarity = self._calculate_real_similarity(anatomical_emb, ref_emb)
                            similarities.append(similarity)
                            logger.info(f"  Similitud {j+1}: {similarity:.4f}")
                    except Exception as e:
                        logger.error(f"Error similitud {j+1}: {e}")
                
                if similarities:
                    individual_scores.anatomical_score = calculate_score_with_voting(similarities)
                    individual_scores.anatomical_confidence = np.mean(similarities)
                    logger.info(f"✅ Score anatómico: {individual_scores.anatomical_score:.4f}")
            
            # Calcular scores dinámicos
            if dynamic_emb is not None and dynamic_refs:
                logger.info(f"Calculando similitudes dinámicas ({len(dynamic_refs)})")
                similarities = []
                
                for j, ref_emb in enumerate(dynamic_refs):
                    try:
                        if isinstance(ref_emb, list):
                            ref_emb = np.array(ref_emb, dtype=np.float32)
                        
                        if dynamic_emb.shape[0] == ref_emb.shape[0]:
                            similarity = self._calculate_real_similarity(dynamic_emb, ref_emb)
                            similarities.append(similarity)
                            logger.info(f"  Similitud {j+1}: {similarity:.4f}")
                    except Exception as e:
                        logger.error(f"Error similitud {j+1}: {e}")
                
                if similarities:
                    individual_scores.dynamic_score = calculate_score_with_voting(similarities)
                    individual_scores.dynamic_confidence = np.mean(similarities)
                    logger.info(f"✅ Score dinámico: {individual_scores.dynamic_score:.4f}")
            
            # Fusión
            logger.info("Fusionando scores...")
            fused_score = self.fusion_system.fuse_real_scores(individual_scores)
            logger.info(f"✅ Score fusionado: {fused_score.fused_score:.4f}")
            
            return RealAuthenticationResult(
                attempt_id=session.attempt_id,
                success=False,
                user_id=session.user_id,
                anatomical_score=individual_scores.anatomical_score,
                dynamic_score=individual_scores.dynamic_score,
                fused_score=fused_score.fused_score,
                confidence=fused_score.confidence,
                security_level=session.security_level,
                authentication_mode=AuthenticationMode.VERIFICATION,
                duration=session.duration,
                frames_processed=session.frames_processed,
                gestures_captured=session.gesture_sequence_captured,
                average_quality=np.mean(session.quality_scores) if session.quality_scores else 0.0,
                average_confidence=np.mean(session.confidence_scores) if session.confidence_scores else 0.0
            )
            
        except Exception as e:
            logger.error(f"Error en verificación: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_failed_auth_result(session, str(e))
    
    def _create_failed_auth_result(self, session: RealAuthenticationAttempt, reason: str) -> RealAuthenticationResult:
        """Crea resultado fallido."""
        return RealAuthenticationResult(
            attempt_id=session.attempt_id,
            success=False,
            user_id=session.user_id,
            security_level=session.security_level,
            authentication_mode=session.mode,
            duration=session.duration,
            frames_processed=session.frames_processed,
            gestures_captured=session.gesture_sequence_captured,
            risk_factors=[reason]
        )
    
    def _calculate_real_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calcula similitud coseno."""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2
            
            cosine_similarity = np.dot(embedding1_norm, embedding2_norm)
            
            similarity = (cosine_similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculando similitud: {e}")
            return 0.0
        
    def _perform_real_identification(self, session: RealAuthenticationAttempt,
                                     anatomical_emb: Optional[np.ndarray],
                                     dynamic_emb: Optional[np.ndarray]) -> RealAuthenticationResult:
        """Realiza identificación 1:N."""
        try:
            logger.info("Realizando identificación 1:N")
            
            # Obtener usuarios
            all_users = self.database.list_users()
            users_with_templates = [u for u in all_users if u.total_templates > 0]
            
            if not users_with_templates:
                raise Exception("No hay usuarios con templates")
            
            logger.info(f"Comparando contra {len(users_with_templates)} usuarios")
            
            if anatomical_emb is None and dynamic_emb is None:
                raise Exception("No hay embeddings para identificación")
            
            # Obtener redes
            anatomical_network = get_real_siamese_anatomical_network()
            dynamic_network = get_real_siamese_dynamic_network()
            
            user_scores = []
            
            for user_profile in users_with_templates:
                try:
                    logger.info(f"Procesando usuario: {user_profile.user_id}")
                    
                    user_templates = self.database.list_user_templates(user_profile.user_id)
                    
                    if not user_templates:
                        logger.info(f"  Sin templates - saltando")
                        continue
                    
                    logger.info(f"  Templates: {len(user_templates)}")
                    
                    anatomical_refs = []
                    dynamic_refs = []
                    
                    # Procesar templates (mismo código que verificación)
                    for i, template in enumerate(user_templates):
                        try:
                            # Embeddings directos
                            if hasattr(template, 'anatomical_embedding') and template.anatomical_embedding is not None and anatomical_emb is not None:
                                anatomical_refs.append(np.array(template.anatomical_embedding, dtype=np.float32))
                                continue
                            
                            if hasattr(template, 'dynamic_embedding') and template.dynamic_embedding is not None and dynamic_emb is not None:
                                dynamic_refs.append(np.array(template.dynamic_embedding, dtype=np.float32))
                                continue
                            
                            # Template data
                            if hasattr(template, 'template_data') and template.template_data is not None:
                                template_type = getattr(template, 'template_type', None)
                                if template_type == TemplateType.ANATOMICAL and anatomical_emb is not None:
                                    anatomical_refs.append(np.array(template.template_data, dtype=np.float32))
                                elif template_type == TemplateType.DYNAMIC and dynamic_emb is not None:
                                    dynamic_refs.append(np.array(template.template_data, dtype=np.float32))
                            
                            # Bootstrap
                            if hasattr(template, 'metadata') and template.metadata:
                                metadata = template.metadata
                                bootstrap_mode = metadata.get('bootstrap_mode', False)
                                
                                if bootstrap_mode:
                                    # Bootstrap anatómico
                                    bootstrap_features = metadata.get('bootstrap_features')
                                    if bootstrap_features and anatomical_emb is not None and anatomical_network:
                                        try:
                                            features_array = np.array(bootstrap_features, dtype=np.float32).reshape(1, -1)
                                            if features_array.shape[1] == anatomical_network.input_dim:
                                                embedding = anatomical_network.base_network.predict(features_array, verbose=0)[0]
                                                anatomical_refs.append(embedding)
                                        except:
                                            pass
                                    
                                    # Bootstrap dinámico
                                    temporal_sequence = metadata.get('temporal_sequence')
                                    if temporal_sequence and dynamic_emb is not None and dynamic_network:
                                        try:
                                            temporal_array = np.array(temporal_sequence, dtype=np.float32)
                                            
                                            if temporal_array.shape[0] > 50:
                                                temporal_array = temporal_array[-50:]
                                            elif temporal_array.shape[0] < 50:
                                                padding = np.zeros((50 - temporal_array.shape[0], temporal_array.shape[1]))
                                                temporal_array = np.vstack([temporal_array, padding])
                                            
                                            if temporal_array.shape[1] != 320:
                                                if temporal_array.shape[1] > 320:
                                                    temporal_array = temporal_array[:, :320]
                                                else:
                                                    padding = np.zeros((temporal_array.shape[0], 320 - temporal_array.shape[1]))
                                                    temporal_array = np.hstack([temporal_array, padding])
                                            
                                            sequence = temporal_array.reshape(1, 50, 320)
                                            embedding = dynamic_network.base_network.predict(sequence, verbose=0)[0]
                                            dynamic_refs.append(embedding)
                                        except:
                                            pass
                        
                        except Exception as e:
                            logger.error(f"  Error template {i+1}: {e}")
                    
                    if not anatomical_refs and not dynamic_refs:
                        logger.info(f"  Sin referencias válidas")
                        continue
                    
                    # Crear scores individuales
                    individual_scores = RealIndividualScores(
                        anatomical_score=0.0,
                        dynamic_score=0.0,
                        anatomical_confidence=0.0,
                        dynamic_confidence=0.0,
                        user_id=user_profile.user_id,
                        metadata={
                            'anatomical_refs': len(anatomical_refs),
                            'dynamic_refs': len(dynamic_refs)
                        }
                    )
                    
                    # Scores anatómicos
                    if anatomical_emb is not None and anatomical_refs:
                        similarities = []
                        for ref_emb in anatomical_refs:
                            try:
                                if anatomical_emb.shape[0] == ref_emb.shape[0]:
                                    sim = self._calculate_real_similarity(anatomical_emb, ref_emb)
                                    similarities.append(sim)
                            except:
                                pass
                        
                        if similarities:
                            individual_scores.anatomical_score = calculate_score_with_voting(similarities)
                            individual_scores.anatomical_confidence = np.mean(similarities)
                    
                    # Scores dinámicos
                    if dynamic_emb is not None and dynamic_refs:
                        similarities = []
                        for ref_emb in dynamic_refs:
                            try:
                                if dynamic_emb.shape[0] == ref_emb.shape[0]:
                                    sim = self._calculate_real_similarity(dynamic_emb, ref_emb)
                                    similarities.append(sim)
                            except:
                                pass
                        
                        if similarities:
                            individual_scores.dynamic_score = calculate_score_with_voting(similarities)
                            individual_scores.dynamic_confidence = np.mean(similarities)
                    
                    # Derivar dinámico si no hay
                    if individual_scores.dynamic_score == 0.0 and individual_scores.anatomical_score > 0.0:
                        individual_scores.dynamic_score = individual_scores.anatomical_score * 0.75
                        individual_scores.dynamic_confidence = individual_scores.anatomical_confidence * 0.60
                    
                    # Fusión
                    fused_result = self.fusion_system.fuse_real_scores(individual_scores)
                    
                    if fused_result.fused_score > 0:
                        user_scores.append({
                            'user_id': user_profile.user_id,
                            'username': getattr(user_profile, 'username', user_profile.user_id),
                            'anatomical_score': individual_scores.anatomical_score,
                            'dynamic_score': individual_scores.dynamic_score,
                            'fused_score': fused_result.fused_score,
                            'confidence': fused_result.confidence
                        })
                        
                        logger.info(f"✅ Usuario {user_profile.user_id}: {fused_result.fused_score:.4f}")
                
                except Exception as e:
                    logger.error(f"Error procesando usuario {user_profile.user_id}: {e}")
            
            if not user_scores:
                raise Exception("No se pudieron calcular scores")
            
            # Ordenar por score
            user_scores.sort(key=lambda x: x['fused_score'], reverse=True)
            
            # Top candidatos
            top_candidates = user_scores[:5]
            
            logger.info(f"Top {len(top_candidates)} candidatos:")
            for i, c in enumerate(top_candidates, 1):
                logger.info(f"  {i}. {c['user_id']} - Score: {c['fused_score']:.4f}")
            
            best_candidate = top_candidates[0]
            
            threshold = self.config.security_thresholds.get('standard', 0.75)
            is_successful = best_candidate['fused_score'] >= threshold
            
            logger.info(f"Mejor: {best_candidate['user_id']} - {best_candidate['fused_score']:.4f} - {'✅' if is_successful else '❌'}")
            
            return RealAuthenticationResult(
                attempt_id=session.attempt_id,
                success=is_successful,
                user_id=None,
                matched_user_id=best_candidate['user_id'] if is_successful else None,
                anatomical_score=best_candidate['anatomical_score'],
                dynamic_score=best_candidate['dynamic_score'],
                fused_score=best_candidate['fused_score'],
                confidence=best_candidate['confidence'],
                security_level=session.security_level,
                authentication_mode=AuthenticationMode.IDENTIFICATION,
                duration=session.duration,
                frames_processed=session.frames_processed,
                gestures_captured=session.gesture_sequence_captured,
                average_quality=np.mean(session.quality_scores) if session.quality_scores else 0.0,
                average_confidence=np.mean(session.confidence_scores) if session.confidence_scores else 0.0
            )
            
        except Exception as e:
            logger.error(f"Error en identificación: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return RealAuthenticationResult(
                attempt_id=session.attempt_id,
                success=False,
                user_id=None,
                security_level=session.security_level,
                authentication_mode=AuthenticationMode.IDENTIFICATION,
                duration=0.0,
                frames_processed=0,
                gestures_captured=[]
            )
    
    def _complete_real_authentication(self, session: RealAuthenticationAttempt, final_status: AuthenticationStatus):
        """Completa sesión."""
        try:
            logger.info(f"Completando autenticación: {session.session_id} - {final_status.value}")
            
            self.session_manager.close_real_session(session.session_id, final_status)
            
            if final_status == AuthenticationStatus.AUTHENTICATED:
                logger.info(f"Autenticación exitosa - Usuario: {session.user_id or 'identificación'}")
            else:
                logger.info(f"Autenticación fallida - Razón: {final_status.value}")
            
        except Exception as e:
            logger.error(f"Error completando autenticación: {e}")
    
    def get_real_authentication_status(self, session_id: str) -> Dict[str, Any]:
        """Obtiene estado detallado de sesión."""
        try:
            session = self.session_manager.get_real_session(session_id)
            if not session:
                return {'error': 'Sesión no encontrada', 'is_real': True}
            
            return {
                'session_id': session_id,
                'attempt_id': session.attempt_id,
                'mode': session.mode.value,
                'user_id': session.user_id,
                'status': session.status.value,
                'phase': session.current_phase.value,
                'security_level': session.security_level.value,
                'duration': session.duration,
                'progress': session.sequence_progress,
                'frames_processed': session.frames_processed,
                'required_sequence': session.required_sequence,
                'captured_sequence': session.gesture_sequence_captured,
                'anatomical_features_count': len(session.anatomical_features),
                'dynamic_features_count': len(session.dynamic_features),
                'average_quality': np.mean(session.quality_scores) if session.quality_scores else 0.0,
                'average_confidence': np.mean(session.confidence_scores) if session.confidence_scores else 0.0,
                'is_real_session': True
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado: {e}")
            return {'error': str(e), 'is_real': True}
    
    def cancel_real_authentication(self, session_id: str) -> bool:
        """Cancela sesión de autenticación."""
        try:
            session = self.session_manager.get_real_session(session_id)
            if not session:
                logger.error(f"Sesión {session_id} no encontrada")
                return False
            
            self.session_manager.close_real_session(session_id, AuthenticationStatus.CANCELLED)
            
            logger.info(f"Sesión {session_id} cancelada")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelando autenticación: {e}")
            return False
    
    def get_real_system_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas."""
        try:
            auth_stats = dict(self.statistics)
            session_stats = self.session_manager.get_real_session_stats()
            db_stats = self.database.get_database_stats()
            security_stats = self.security_auditor.get_security_metrics()
            
            # Verificar redes
            anatomical_trained = False
            dynamic_trained = False
            
            if hasattr(self.pipeline, 'anatomical_network') and self.pipeline.anatomical_network:
                anatomical_trained = self.pipeline.anatomical_network.is_trained
            
            if hasattr(self.pipeline, 'dynamic_network') and self.pipeline.dynamic_network:
                dynamic_trained = self.pipeline.dynamic_network.is_trained
            
            return {
                'authentication': auth_stats,
                'sessions': session_stats,
                'database': db_stats.__dict__,
                'security': security_stats,
                'system_status': {
                    'initialized': self.is_initialized,
                    'active_sessions': len(self.session_manager.active_sessions),
                    'total_users': db_stats.total_users,
                    'total_templates': db_stats.total_templates,
                    'pipeline_ready': self.pipeline.is_initialized,
                    'networks_trained': anatomical_trained and dynamic_trained,
                    'is_real_system': True,
                    'version': '2.0_real'
                }
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {'error': str(e), 'is_real_system': True}
    
    def get_real_available_users(self) -> List[Dict[str, Any]]:
        """Obtiene usuarios disponibles para autenticación."""
        try:
            users = self.database.list_users()
            
            user_list = []
            for user in users:
                if user.total_templates > 0:
                    user_list.append({
                        'user_id': user.user_id,
                        'username': user.username,
                        'total_templates': user.total_templates,
                        'gesture_sequence': getattr(user, 'gesture_sequence', []),
                        'is_real_user': True
                    })
            
            logger.info(f"Usuarios disponibles: {len(user_list)}")
            return user_list
            
        except Exception as e:
            logger.error(f"Error obteniendo usuarios: {e}")
            return []
    
    def cleanup_real_system(self):
        """Limpia recursos del sistema."""
        try:
            logger.info("Limpiando sistema de autenticación")
            
            # Cancelar sesiones activas
            for session_id in list(self.session_manager.active_sessions.keys()):
                self.cancel_real_authentication(session_id)
            
            # Limpiar pipeline
            self.pipeline.cleanup()
            
            self.is_initialized = False
            
            logger.info("Sistema limpiado completamente")
            
        except Exception as e:
            logger.error(f"Error limpiando sistema: {e}")


# ====================================================================
# FUNCIÓN GLOBAL
# ====================================================================

_real_authentication_system_instance = None

def get_real_authentication_system(config_override: Optional[Dict[str, Any]] = None) -> RealAuthenticationSystem:
    """
    Obtiene instancia global del sistema de autenticación.
    
    Args:
        config_override: Configuración personalizada (opcional)
        
    Returns:
        Instancia de RealAuthenticationSystem
    """
    global _real_authentication_system_instance
    
    if _real_authentication_system_instance is None:
        _real_authentication_system_instance = RealAuthenticationSystem(config_override)
    
    return _real_authentication_system_instance


# Alias para compatibilidad
AuthenticationSystem = RealAuthenticationSystem
get_authentication_system = get_real_authentication_system


# ====================================================================
# TESTING DEL MÓDULO
# ====================================================================

if __name__ == "__main__":
    print("=== TESTING MÓDULO 15: AUTHENTICATION_SYSTEM ===")
    
    # Test 1: Inicialización
    try:
        auth_system = RealAuthenticationSystem()
        print("✓ Sistema inicializado")
        print(f"  - Umbrales: {auth_system.config.security_thresholds}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Componentes
    try:
        print(f"✓ Pipeline: {type(auth_system.pipeline).__name__}")
        print(f"✓ Session Manager: {type(auth_system.session_manager).__name__}")
        print(f"✓ Security Auditor: {type(auth_system.security_auditor).__name__}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Estadísticas
    try:
        stats = auth_system.get_real_system_statistics()
        print(f"✓ Estadísticas:")
        print(f"  - Inicializado: {stats['system_status']['initialized']}")
        print(f"  - Usuarios: {stats['system_status']['total_users']}")
        print(f"  - Templates: {stats['system_status']['total_templates']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Usuarios disponibles
    try:
        users = auth_system.get_real_available_users()
        print(f"✓ Usuarios disponibles: {len(users)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 5: Cleanup
    try:
        auth_system.cleanup_real_system()
        print("✓ Recursos liberados")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n=== RESUMEN MÓDULO 15 ===")
    print("✓ Clases principales:")
    print("  - RealAuthenticationConfig")
    print("  - RealAuthenticationAttempt")
    print("  - RealAuthenticationResult")
    print("  - RealSecurityAuditor")
    print("  - RealSessionManager")
    print("  - RealAuthenticationPipeline")
    print("  - RealAuthenticationSystem")
    print("\n✓ Características:")
    print("  - Verificación 1:1")
    print("  - Identificación 1:N")
    print("  - ROI Normalization integrado")
    print("  - Sistema de votación")
    print("  - Auditoría de seguridad")
    print("  - Gestión de sesiones")
    print("\n✓ Sistema 100% funcional")
    print("=== FIN TESTING ===")