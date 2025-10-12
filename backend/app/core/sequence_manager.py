# =============================================================================
# MÓDULO 8: SEQUENCE_MANAGER
# Gestor de secuencias personalizadas de gestos por usuario
# =============================================================================

import json
import time
import uuid
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

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


class SequenceState(Enum):
    """Estados de la secuencia de gestos."""
    IDLE = "idle"
    WAITING_START = "waiting_start"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"


class GestureValidation(Enum):
    """Tipos de validación de gestos."""
    STRICT = "strict"
    MODERATE = "moderate"
    FLEXIBLE = "flexible"


class SequenceEventType(Enum):
    """Eventos que pueden ocurrir durante la secuencia."""
    SEQUENCE_STARTED = "sequence_started"
    GESTURE_DETECTED = "gesture_detected"
    GESTURE_VALIDATED = "gesture_validated"
    GESTURE_REJECTED = "gesture_rejected"
    SEQUENCE_ADVANCED = "sequence_advanced"
    SEQUENCE_COMPLETED = "sequence_completed"
    SEQUENCE_FAILED = "sequence_failed"
    SEQUENCE_TIMEOUT = "sequence_timeout"
    SEQUENCE_RESET = "sequence_reset"


@dataclass
class GestureStep:
    """Paso individual en la secuencia de gestos."""
    gesture_name: str
    min_confidence: float = 0.7
    min_stable_frames: int = 3
    max_duration: float = 10.0
    allow_interruption: bool = True
    validation_mode: GestureValidation = GestureValidation.MODERATE


@dataclass
class UserSequence:
    """Secuencia personalizada de un usuario."""
    user_id: str
    sequence_name: str
    gesture_steps: List[GestureStep]
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    total_attempts: int = 0
    successful_completions: int = 0
    
    @property
    def success_rate(self) -> float:
        """Tasa de éxito de la secuencia."""
        return (self.successful_completions / self.total_attempts * 100) if self.total_attempts > 0 else 0.0
    
    @property
    def gesture_names(self) -> List[str]:
        """Lista de nombres de gestos en la secuencia."""
        return [step.gesture_name for step in self.gesture_steps]


@dataclass
class SequenceAttempt:
    """Intento de ejecución de secuencia."""
    attempt_id: str
    user_sequence: UserSequence
    start_time: float
    end_time: Optional[float] = None
    current_step: int = 0
    completed_steps: List[Dict[str, Any]] = field(default_factory=list)
    final_state: Optional[SequenceState] = None
    failure_reason: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Duración del intento."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def is_active(self) -> bool:
        """Si el intento está activo."""
        return self.final_state is None
    
    @property
    def progress_percentage(self) -> float:
        """Porcentaje de progreso."""
        return (self.current_step / len(self.user_sequence.gesture_steps)) * 100


@dataclass
class SequenceEventLog:
    """Evento ocurrido durante la secuencia."""
    event_type: SequenceEventType
    timestamp: float
    attempt_id: str
    step_index: int
    gesture_detected: str
    confidence: float
    additional_data: Dict[str, Any] = field(default_factory=dict)


class SequenceManager:
    """
    Gestor de secuencias personalizadas de gestos para autenticación biométrica.
    Maneja secuencias de 3 gestos definidas por cada usuario.
    """
    
    def __init__(self):
        """Inicializa el gestor de secuencias."""
        
        # Configuración
        self.config = self._load_sequence_config()
        
        # Estado actual
        self.current_attempt: Optional[SequenceAttempt] = None
        self.current_user_sequence: Optional[UserSequence] = None
        
        # Almacenamiento
        self.user_sequences: Dict[str, UserSequence] = {}
        self.sequence_history: List[SequenceAttempt] = []
        self.event_log: List[SequenceEventLog] = []
        
        # Callbacks para eventos
        self.event_callbacks: Dict[SequenceEventType, List[Callable]] = {}
        
        # Estado de validación actual
        self.current_gesture_frames = 0
        self.last_gesture_time = 0
        self.gesture_stable_count = 0
        
        # Estadísticas
        self.total_attempts = 0
        self.successful_sequences = 0
        
        # Cargar secuencias guardadas
        self._load_user_sequences()
        
        logger.info("SequenceManager inicializado")
    
    def _load_sequence_config(self) -> Dict[str, Any]:
        """Carga configuración del gestor de secuencias."""
        default_config = {
            'sequence_length': 3,
            'max_sequence_duration': 60.0,
            'max_gesture_duration': 15.0,
            'min_gesture_confidence': 0.7,
            'min_stable_frames': 3,
            'gesture_timeout': 10.0,
            'sequence_timeout': 45.0,
            'max_failed_attempts': 3,
            'cooldown_period': 5.0,
            'auto_save_sequences': True,
            'validation_mode': 'moderate',
            'allow_gesture_skipping': False,
            'require_exact_order': True,
            'enable_adaptive_timeouts': True
        }
        
        return get_config('biometric.sequence_management', default_config)

    def create_user_sequence(self, user_id: str, gesture_names: List[str], 
                           sequence_name: Optional[str] = None) -> UserSequence:
        """
        Crea una nueva secuencia personalizada para un usuario.
        
        Args:
            user_id: ID único del usuario
            gesture_names: Lista de 3 nombres de gestos en orden
            sequence_name: Nombre personalizado (opcional)
            
        Returns:
            Secuencia de usuario creada
        """
        try:
            if len(gesture_names) != self.config['sequence_length']:
                raise ValueError(f"La secuencia debe tener exactamente {self.config['sequence_length']} gestos")
            
            available_gestures = get_config('available_gestures', [
                "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", 
                "Thumb_Up", "Victory", "ILoveYou"
            ])
            
            for gesture in gesture_names:
                if gesture not in available_gestures:
                    raise ValueError(f"Gesto '{gesture}' no está disponible")
            
            if len(set(gesture_names)) != len(gesture_names):
                raise ValueError("No se permiten gestos duplicados en la secuencia")
            
            gesture_steps = []
            for i, gesture_name in enumerate(gesture_names):
                step = GestureStep(
                    gesture_name=gesture_name,
                    min_confidence=self.config['min_gesture_confidence'],
                    min_stable_frames=self.config['min_stable_frames'],
                    max_duration=self.config['max_gesture_duration'],
                    validation_mode=GestureValidation(self.config['validation_mode'])
                )
                gesture_steps.append(step)
            
            if sequence_name is None:
                sequence_name = f"Secuencia_{user_id}_{int(time.time())}"
            
            user_sequence = UserSequence(
                user_id=user_id,
                sequence_name=sequence_name,
                gesture_steps=gesture_steps
            )
            
            self.user_sequences[user_id] = user_sequence
            
            if self.config['auto_save_sequences']:
                self._save_user_sequences()
            
            logger.info(f"Secuencia creada para usuario {user_id}: {' → '.join(gesture_names)}")
            return user_sequence
            
        except Exception as e:
            logger.error(f"Error creando secuencia para usuario {user_id}: {e}")
            raise
    
    def start_sequence(self, user_id: str) -> bool:
        """
        Inicia una nueva secuencia para un usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            True si se inició correctamente
        """
        try:
            if user_id not in self.user_sequences:
                logger.error(f"No existe secuencia para usuario {user_id}")
                return False
            
            if self.current_attempt and self.current_attempt.is_active:
                logger.error("Ya hay una secuencia activa")
                return False
            
            user_sequence = self.user_sequences[user_id]
            
            attempt_id = str(uuid.uuid4())
            self.current_attempt = SequenceAttempt(
                attempt_id=attempt_id,
                user_sequence=user_sequence,
                start_time=time.time()
            )
            
            self.current_user_sequence = user_sequence
            
            self.current_gesture_frames = 0
            self.last_gesture_time = time.time()
            self.gesture_stable_count = 0
            
            self.total_attempts += 1
            user_sequence.total_attempts += 1
            user_sequence.last_used = time.time()
            
            self._log_event(SequenceEventType.SEQUENCE_STARTED, attempt_id, 0, "None", 0.0)
            
            self._execute_callbacks(SequenceEventType.SEQUENCE_STARTED)
            
            logger.info(f"Secuencia iniciada para usuario {user_id} (ID: {attempt_id})")
            logger.info(f"Secuencia objetivo: {' → '.join(user_sequence.gesture_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando secuencia para usuario {user_id}: {e}")
            return False
    
    def process_gesture(self, gesture_name: str, confidence: float, 
                       additional_data: Optional[Dict[str, Any]] = None) -> SequenceState:
        """
        Procesa un gesto detectado en el contexto de la secuencia actual.
        
        Args:
            gesture_name: Nombre del gesto detectado
            confidence: Confianza de la detección
            additional_data: Datos adicionales (opcional)
            
        Returns:
            Estado actual de la secuencia
        """
        try:
            if not self.current_attempt or not self.current_attempt.is_active:
                return SequenceState.IDLE
            
            current_time = time.time()
            
            if self._check_sequence_timeout():
                return self._handle_sequence_timeout()
            
            current_step_index = self.current_attempt.current_step
            if current_step_index >= len(self.current_user_sequence.gesture_steps):
                return self._complete_sequence()
            
            current_step = self.current_user_sequence.gesture_steps[current_step_index]
            expected_gesture = current_step.gesture_name
            
            self._log_event(SequenceEventType.GESTURE_DETECTED, 
                          self.current_attempt.attempt_id, current_step_index, 
                          gesture_name, confidence, additional_data)
            
            is_valid_gesture = self._validate_gesture(gesture_name, expected_gesture, 
                                                    confidence, current_step)
            
            if is_valid_gesture:
                self.gesture_stable_count += 1
                self.current_gesture_frames += 1
                
                if self.gesture_stable_count >= current_step.min_stable_frames:
                    return self._advance_sequence(gesture_name, confidence, additional_data)
                else:
                    self._log_event(SequenceEventType.GESTURE_VALIDATED, 
                                  self.current_attempt.attempt_id, current_step_index,
                                  gesture_name, confidence)
                    return SequenceState.IN_PROGRESS
            else:
                self._log_event(SequenceEventType.GESTURE_REJECTED,
                              self.current_attempt.attempt_id, current_step_index,
                              gesture_name, confidence, 
                              {"expected": expected_gesture, "reason": "invalid_gesture"})
                
                self.gesture_stable_count = 0
                
                if not current_step.allow_interruption:
                    return self._fail_sequence("Gesto incorrecto detectado")
                
                return SequenceState.IN_PROGRESS
            
        except Exception as e:
            logger.error(f"Error procesando gesto en secuencia: {e}")
            return self._fail_sequence(f"Error interno: {str(e)}")
    
    def _validate_gesture(self, detected_gesture: str, expected_gesture: str,
                         confidence: float, step: GestureStep) -> bool:
        """Valida si un gesto detectado es aceptable para el paso actual."""
        try:
            if confidence < step.min_confidence:
                return False
            
            if step.validation_mode == GestureValidation.STRICT:
                return detected_gesture == expected_gesture
            
            elif step.validation_mode == GestureValidation.MODERATE:
                return detected_gesture == expected_gesture
            
            elif step.validation_mode == GestureValidation.FLEXIBLE:
                similar_gestures = self._get_similar_gestures(expected_gesture)
                return detected_gesture in similar_gestures
            
            return False
            
        except Exception as e:
            logger.error(f"Error validando gesto: {e}")
            return False
    
    def _get_similar_gestures(self, gesture: str) -> List[str]:
        """Obtiene lista de gestos similares para validación flexible."""
        gesture_groups = {
            "Thumb_Up": ["Thumb_Up"],
            "Thumb_Down": ["Thumb_Down"],
            "Victory": ["Victory", "Pointing_Up"],
            "Open_Palm": ["Open_Palm"],
            "Closed_Fist": ["Closed_Fist"],
            "Pointing_Up": ["Pointing_Up", "Victory"],
            "ILoveYou": ["ILoveYou"]
        }
        
        return gesture_groups.get(gesture, [gesture])
    
    def _advance_sequence(self, gesture_name: str, confidence: float, 
                         additional_data: Optional[Dict[str, Any]]) -> SequenceState:
        """Avanza la secuencia al siguiente paso."""
        try:
            current_step_index = self.current_attempt.current_step
            current_time = time.time()
            
            completed_step = {
                "step_index": current_step_index,
                "gesture_name": gesture_name,
                "confidence": confidence,
                "completion_time": current_time,
                "frames_stable": self.gesture_stable_count,
                "step_duration": current_time - self.last_gesture_time,
                "additional_data": additional_data or {}
            }
            
            self.current_attempt.completed_steps.append(completed_step)
            
            self.current_attempt.current_step += 1
            
            self.gesture_stable_count = 0
            self.current_gesture_frames = 0
            self.last_gesture_time = current_time
            
            self._log_event(SequenceEventType.SEQUENCE_ADVANCED,
                          self.current_attempt.attempt_id, current_step_index,
                          gesture_name, confidence, completed_step)
            
            if self.current_attempt.current_step >= len(self.current_user_sequence.gesture_steps):
                return self._complete_sequence()
            
            self._execute_callbacks(SequenceEventType.SEQUENCE_ADVANCED)
            
            logger.info(f"Paso {current_step_index + 1} completado: {gesture_name} (confianza: {confidence:.2f})")
            logger.info(f"Progreso: {self.current_attempt.progress_percentage:.1f}% - Siguiente: {self.current_user_sequence.gesture_steps[self.current_attempt.current_step].gesture_name}")
            
            return SequenceState.IN_PROGRESS
            
        except Exception as e:
            logger.error(f"Error avanzando secuencia: {e}")
            return self._fail_sequence(f"Error avanzando secuencia: {str(e)}")
    
    def _complete_sequence(self) -> SequenceState:
        """Completa la secuencia exitosamente."""
        try:
            if not self.current_attempt:
                return SequenceState.IDLE
            
            self.current_attempt.end_time = time.time()
            self.current_attempt.final_state = SequenceState.COMPLETED
            
            self.successful_sequences += 1
            self.current_user_sequence.successful_completions += 1
            
            self._log_event(SequenceEventType.SEQUENCE_COMPLETED,
                          self.current_attempt.attempt_id, 
                          len(self.current_user_sequence.gesture_steps),
                          "SEQUENCE_COMPLETE", 1.0)
            
            self.sequence_history.append(self.current_attempt)
            
            self._execute_callbacks(SequenceEventType.SEQUENCE_COMPLETED)
            
            duration = self.current_attempt.duration
            logger.info(f"¡SECUENCIA COMPLETADA! Duración: {duration:.2f}s")
            logger.info(f"Usuario: {self.current_user_sequence.user_id} - Tasa de éxito: {self.current_user_sequence.success_rate:.1f}%")
            
            self.current_attempt = None
            self.current_user_sequence = None
            
            return SequenceState.COMPLETED
            
        except Exception as e:
            logger.error(f"Error completando secuencia: {e}")
            return SequenceState.FAILED
    
    def _fail_sequence(self, reason: str) -> SequenceState:
        """Marca la secuencia como fallida."""
        try:
            if not self.current_attempt:
                return SequenceState.IDLE
            
            self.current_attempt.end_time = time.time()
            self.current_attempt.final_state = SequenceState.FAILED
            self.current_attempt.failure_reason = reason
            
            self._log_event(SequenceEventType.SEQUENCE_FAILED,
                          self.current_attempt.attempt_id, 
                          self.current_attempt.current_step,
                          "SEQUENCE_FAILED", 0.0, {"reason": reason})
            
            self.sequence_history.append(self.current_attempt)
            
            self._execute_callbacks(SequenceEventType.SEQUENCE_FAILED)
            
            logger.info(f"Secuencia falló: {reason}")
            
            self.current_attempt = None
            self.current_user_sequence = None
            
            return SequenceState.FAILED
            
        except Exception as e:
            logger.error(f"Error manejando fallo de secuencia: {e}")
            return SequenceState.FAILED
    
    def _check_sequence_timeout(self) -> bool:
        """Verifica si la secuencia ha excedido el timeout."""
        if not self.current_attempt:
            return False
        
        elapsed_time = time.time() - self.current_attempt.start_time
        return elapsed_time > self.config['sequence_timeout']
    
    def _handle_sequence_timeout(self) -> SequenceState:
        """Maneja timeout de secuencia."""
        if not self.current_attempt:
            return SequenceState.IDLE
        
        self.current_attempt.end_time = time.time()
        self.current_attempt.final_state = SequenceState.TIMEOUT
        self.current_attempt.failure_reason = "Secuencia expiró"
        
        self._log_event(SequenceEventType.SEQUENCE_TIMEOUT,
                      self.current_attempt.attempt_id, 
                      self.current_attempt.current_step,
                      "TIMEOUT", 0.0)
        
        self.sequence_history.append(self.current_attempt)
        
        logger.info(f"Secuencia expiró después de {self.current_attempt.duration:.2f}s")
        
        self.current_attempt = None
        self.current_user_sequence = None
        
        return SequenceState.TIMEOUT
    
    def reset_sequence(self) -> bool:
        """Reinicia la secuencia actual."""
        try:
            if self.current_attempt and self.current_attempt.is_active:
                self.current_attempt.end_time = time.time()
                self.current_attempt.final_state = SequenceState.INTERRUPTED
                self.current_attempt.failure_reason = "Secuencia reiniciada manualmente"
                
                self._log_event(SequenceEventType.SEQUENCE_RESET,
                              self.current_attempt.attempt_id, 
                              self.current_attempt.current_step,
                              "RESET", 0.0)
                
                self.sequence_history.append(self.current_attempt)
                
                logger.info("Secuencia reiniciada manualmente")
            
            self.current_attempt = None
            self.current_user_sequence = None
            self.gesture_stable_count = 0
            self.current_gesture_frames = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error reiniciando secuencia: {e}")
            return False
    
    def get_current_state(self) -> Dict[str, Any]:
        """Obtiene el estado actual de la secuencia."""
        try:
            if not self.current_attempt or not self.current_user_sequence:
                return {
                    "state": SequenceState.IDLE.value,
                    "active": False,
                    "user_id": None,
                    "progress": 0.0,
                    "current_step": 0,
                    "total_steps": 0,
                    "expected_gesture": None,
                    "elapsed_time": 0.0
                }
            
            current_step_index = self.current_attempt.current_step
            expected_gesture = None
            
            if current_step_index < len(self.current_user_sequence.gesture_steps):
                expected_gesture = self.current_user_sequence.gesture_steps[current_step_index].gesture_name
            
            return {
                "state": SequenceState.IN_PROGRESS.value,
                "active": True,
                "user_id": self.current_user_sequence.user_id,
                "sequence_name": self.current_user_sequence.sequence_name,
                "progress": self.current_attempt.progress_percentage,
                "current_step": current_step_index + 1,
                "total_steps": len(self.current_user_sequence.gesture_steps),
                "expected_gesture": expected_gesture,
                "gesture_sequence": self.current_user_sequence.gesture_names,
                "elapsed_time": self.current_attempt.duration,
                "remaining_time": max(0, self.config['sequence_timeout'] - self.current_attempt.duration),
                "stable_frames": self.gesture_stable_count,
                "required_stable_frames": self.current_user_sequence.gesture_steps[current_step_index].min_stable_frames if current_step_index < len(self.current_user_sequence.gesture_steps) else 0,
                "completed_steps": len(self.current_attempt.completed_steps)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado actual: {e}")
            return {"state": SequenceState.FAILED.value, "error": str(e)}
    
    def get_user_sequences(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene todas las secuencias de usuarios."""
        result = {}
        
        for user_id, sequence in self.user_sequences.items():
            result[user_id] = {
                "sequence_name": sequence.sequence_name,
                "gesture_names": sequence.gesture_names,
                "created_at": sequence.created_at,
                "last_used": sequence.last_used,
                "total_attempts": sequence.total_attempts,
                "successful_completions": sequence.successful_completions,
                "success_rate": sequence.success_rate
            }
        
        return result
    
    def get_user_sequence(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene la secuencia de un usuario específico."""
        if user_id not in self.user_sequences:
            return None
        
        sequence = self.user_sequences[user_id]
        return {
            "user_id": sequence.user_id,
            "sequence_name": sequence.sequence_name,
            "gesture_names": sequence.gesture_names,
            "created_at": sequence.created_at,
            "last_used": sequence.last_used,
            "total_attempts": sequence.total_attempts,
            "successful_completions": sequence.successful_completions,
            "success_rate": sequence.success_rate
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor de secuencias."""
        success_rate = (self.successful_sequences / self.total_attempts * 100) if self.total_attempts > 0 else 0
        
        return {
            "total_attempts": self.total_attempts,
            "successful_sequences": self.successful_sequences,
            "success_rate_percent": round(success_rate, 2),
            "registered_users": len(self.user_sequences),
            "active_sequence": self.current_attempt is not None,
            "sequence_history_size": len(self.sequence_history),
            "event_log_size": len(self.event_log),
            "config": self.config.copy()
        }
    
    def get_sequence_history(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene historial de secuencias."""
        history = self.sequence_history
        
        if user_id:
            history = [attempt for attempt in history if attempt.user_sequence.user_id == user_id]
        
        history = sorted(history, key=lambda x: x.start_time, reverse=True)[:limit]
        
        return [
            {
                "attempt_id": attempt.attempt_id,
                "user_id": attempt.user_sequence.user_id,
                "sequence_name": attempt.user_sequence.sequence_name,
                "start_time": attempt.start_time,
                "end_time": attempt.end_time,
                "duration": attempt.duration,
                "final_state": attempt.final_state.value if attempt.final_state else None,
                "failure_reason": attempt.failure_reason,
                "progress_percentage": attempt.progress_percentage,
                "completed_steps": len(attempt.completed_steps)
            }
            for attempt in history
        ]
    
    def add_event_callback(self, event_type: SequenceEventType, callback: Callable):
        """Añade callback para eventos de secuencia."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        
        self.event_callbacks[event_type].append(callback)
        logger.info(f"Callback añadido para evento {event_type.value}")
    
    def _execute_callbacks(self, event_type: SequenceEventType):
        """Ejecuta callbacks para un tipo de evento."""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(self.get_current_state())
                except Exception as e:
                    logger.error(f"Error ejecutando callback para {event_type.value}: {e}")
    
    def _log_event(self, event_type: SequenceEventType, attempt_id: str, 
                   step_index: int, gesture_detected: str, confidence: float,
                   additional_data: Optional[Dict[str, Any]] = None):
        """Registra un evento en el log."""
        event = SequenceEventLog(
            event_type=event_type,
            timestamp=time.time(),
            attempt_id=attempt_id,
            step_index=step_index,
            gesture_detected=gesture_detected,
            confidence=confidence,
            additional_data=additional_data or {}
        )
        
        self.event_log.append(event)
        
        max_log_size = 1000
        if len(self.event_log) > max_log_size:
            self.event_log = self.event_log[-max_log_size:]
    
    def _save_user_sequences(self):
        """Guarda las secuencias de usuarios en disco."""
        try:
            sequences_dir = Path(get_config('paths.user_profiles', 'biometric_data/user_profiles'))
            sequences_dir.mkdir(parents=True, exist_ok=True)
            
            sequences_file = sequences_dir / "user_sequences.json"
            
            sequences_data = {}
            for user_id, sequence in self.user_sequences.items():
                sequences_data[user_id] = {
                    "user_id": sequence.user_id,
                    "sequence_name": sequence.sequence_name,
                    "gesture_steps": [
                        {
                            "gesture_name": step.gesture_name,
                            "min_confidence": step.min_confidence,
                            "min_stable_frames": step.min_stable_frames,
                            "max_duration": step.max_duration,
                            "allow_interruption": step.allow_interruption,
                            "validation_mode": step.validation_mode.value
                        }
                        for step in sequence.gesture_steps
                    ],
                    "created_at": sequence.created_at,
                    "last_used": sequence.last_used,
                    "total_attempts": sequence.total_attempts,
                    "successful_completions": sequence.successful_completions
                }
            
            with open(sequences_file, 'w') as f:
                json.dump(sequences_data, f, indent=2)
            
            logger.info(f"Secuencias guardadas: {len(sequences_data)} usuarios")
            
        except Exception as e:
            logger.error(f"Error guardando secuencias de usuarios: {e}")
    
    def _load_user_sequences(self):
        """Carga las secuencias de usuarios desde disco."""
        try:
            sequences_dir = Path(get_config('paths.user_profiles', 'biometric_data/user_profiles'))
            sequences_file = sequences_dir / "user_sequences.json"
            
            if not sequences_file.exists():
                logger.info("No se encontraron secuencias guardadas")
                return
            
            with open(sequences_file, 'r') as f:
                sequences_data = json.load(f)
            
            for user_id, data in sequences_data.items():
                gesture_steps = []
                for step_data in data["gesture_steps"]:
                    step = GestureStep(
                        gesture_name=step_data["gesture_name"],
                        min_confidence=step_data["min_confidence"],
                        min_stable_frames=step_data["min_stable_frames"],
                        max_duration=step_data["max_duration"],
                        allow_interruption=step_data["allow_interruption"],
                        validation_mode=GestureValidation(step_data["validation_mode"])
                    )
                    gesture_steps.append(step)
                
                sequence = UserSequence(
                    user_id=data["user_id"],
                    sequence_name=data["sequence_name"],
                    gesture_steps=gesture_steps,
                    created_at=data["created_at"],
                    last_used=data["last_used"],
                    total_attempts=data["total_attempts"],
                    successful_completions=data["successful_completions"]
                )
                
                self.user_sequences[user_id] = sequence
            
            logger.info(f"Secuencias cargadas: {len(self.user_sequences)} usuarios")
            
        except Exception as e:
            logger.error(f"Error cargando secuencias de usuarios: {e}")
    
    def delete_user_sequence(self, user_id: str) -> bool:
        """Elimina la secuencia de un usuario."""
        try:
            if user_id in self.user_sequences:
                if (self.current_user_sequence and 
                    self.current_user_sequence.user_id == user_id):
                    self.reset_sequence()
                
                del self.user_sequences[user_id]
                
                if self.config['auto_save_sequences']:
                    self._save_user_sequences()
                
                logger.info(f"Secuencia eliminada para usuario {user_id}")
                return True
            else:
                logger.error(f"No existe secuencia para usuario {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error eliminando secuencia para usuario {user_id}: {e}")
            return False
    
    def update_user_sequence(self, user_id: str, gesture_names: List[str]) -> bool:
        """Actualiza la secuencia de un usuario existente."""
        try:
            if user_id not in self.user_sequences:
                logger.error(f"No existe secuencia para usuario {user_id}")
                return False
            
            old_sequence = self.user_sequences[user_id]
            sequence_name = old_sequence.sequence_name
            
            new_sequence = self.create_user_sequence(user_id, gesture_names, sequence_name)
            
            new_sequence.created_at = old_sequence.created_at
            new_sequence.total_attempts = old_sequence.total_attempts
            new_sequence.successful_completions = old_sequence.successful_completions
            
            logger.info(f"Secuencia actualizada para usuario {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando secuencia para usuario {user_id}: {e}")
            return False
    
    def list_available_gestures(self) -> List[str]:
        """Lista los gestos disponibles para crear secuencias."""
        return get_config('available_gestures', [
            "Closed_Fist", "Open_Palm", "Pointing_Up", 
            "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"
        ])
    
    def validate_gesture_sequence(self, gesture_names: List[str]) -> Tuple[bool, Optional[str]]:
        """Valida si una secuencia de gestos es válida."""
        try:
            if len(gesture_names) != self.config['sequence_length']:
                return False, f"La secuencia debe tener {self.config['sequence_length']} gestos"
            
            available = self.list_available_gestures()
            for gesture in gesture_names:
                if gesture not in available:
                    return False, f"Gesto '{gesture}' no está disponible"
            
            if len(set(gesture_names)) != len(gesture_names):
                return False, "No se permiten gestos duplicados"
            
            return True, None
            
        except Exception as e:
            return False, f"Error validando secuencia: {str(e)}"


# ===== INSTANCIA GLOBAL =====
_sequence_manager_instance = None

def get_sequence_manager() -> SequenceManager:
    """Obtiene una instancia global del gestor de secuencias."""
    global _sequence_manager_instance
    
    if _sequence_manager_instance is None:
        _sequence_manager_instance = SequenceManager()
    
    return _sequence_manager_instance