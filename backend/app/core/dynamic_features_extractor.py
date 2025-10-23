# =============================================================================
# MÓDULO 7: DYNAMIC_FEATURES_EXTRACTOR
# Extractor de características dinámicas REAL (100% SIN SIMULACIÓN)
# =============================================================================

import numpy as np
import cv2
import time
import logging
from collections import deque
from typing import List, Dict, Tuple, Optional, Any, Deque, Callable
from dataclasses import dataclass, field
from enum import Enum

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


class TransitionPhase(Enum):
    """Fases de transición entre gestos."""
    STABLE = "stable"
    PREPARING = "preparing"
    TRANSITIONING = "transitioning"
    COMPLETING = "completing"
    STABILIZING = "stabilizing"


class MotionType(Enum):
    """Tipos de movimiento característicos."""
    SMOOTH = "smooth"
    ABRUPT = "abrupt"
    CURVED = "curved"
    LINEAR = "linear"
    OSCILLATORY = "oscillatory"


@dataclass
class TemporalFrame:
    """Frame temporal con landmarks y metadata."""
    frame_id: int
    timestamp: float
    landmarks: Any
    world_landmarks: Optional[Any] = None
    gesture_name: str = "None"
    confidence: float = 0.0
    
    velocity_vectors: Optional[np.ndarray] = None
    acceleration_vectors: Optional[np.ndarray] = None
    position_3d: Optional[np.ndarray] = None
    
    frame_quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionEvent:
    """Evento de transición detectado entre gestos."""
    start_frame: int
    end_frame: int
    start_gesture: str
    end_gesture: str
    transition_type: str
    duration_ms: float
    transition_frames: List[TemporalFrame] = field(default_factory=list)
    motion_type: MotionType = MotionType.SMOOTH
    confidence: float = 0.0


@dataclass
class VelocityProfile:
    """Perfil completo de velocidad durante una transición."""
    landmark_velocities: np.ndarray
    peak_velocities: np.ndarray
    avg_velocities: np.ndarray
    velocity_patterns: np.ndarray
    timing_features: np.ndarray


@dataclass
class AccelerationProfile:
    """Perfil de aceleración durante una transición."""
    landmark_accelerations: np.ndarray
    peak_accelerations: np.ndarray
    avg_accelerations: np.ndarray
    jerk_patterns: np.ndarray
    smoothness_metrics: np.ndarray


@dataclass
class TrajectoryProfile:
    """Perfil de trayectoria durante una transición."""
    landmark_trajectories: np.ndarray
    trajectory_lengths: np.ndarray
    curvature_profiles: np.ndarray
    direction_changes: np.ndarray
    spatial_efficiency: np.ndarray


@dataclass
class DynamicFeatureVector:
    """Vector completo de características dinámicas REALES."""
    velocity_features: np.ndarray
    acceleration_features: np.ndarray
    trajectory_features: np.ndarray
    timing_features: np.ndarray
    rhythm_features: np.ndarray
    transition_features: np.ndarray
    
    @property
    def complete_vector(self) -> np.ndarray:
        """Vector completo concatenado (320 dimensiones)."""
        return np.concatenate([
            self.velocity_features,
            self.acceleration_features,
            self.trajectory_features,
            self.timing_features,
            self.rhythm_features,
            self.transition_features
        ])
    
    @property
    def dimension(self) -> int:
        """Dimensión total del vector."""
        return len(self.complete_vector)

class RealDynamicFeaturesExtractor:
    """
    Extractor de características dinámicas para biometría temporal.
    Captura y analiza secuencias temporales REALES.
    """
    
    def __init__(self, sequence_length: int = 50):
        """Inicializa el extractor de características dinámicas REAL."""
        
        self.logger = get_logger()
        
        # Configuración
        self.sequence_length = sequence_length
        self.dynamic_config = self._load_dynamic_config()
        
        # Buffer temporal para frames REALES
        self.temporal_buffer: Deque[TemporalFrame] = deque(maxlen=sequence_length)
        self.previous_frame: Optional[TemporalFrame] = None
        
        # Estado de seguimiento de transiciones REAL
        self.current_gesture = "None"
        self.gesture_stable_count = 0
        self.transition_active = False
        self.transition_start_frame = 0
        self.frame_counter = 0
        
        # Historial de transiciones detectadas REALES
        self.detected_transitions: List[TransitionEvent] = []
        
        # Estadísticas REALES
        self.frames_processed = 0
        self.transitions_detected = 0
        self.successful_extractions = 0
        
        logger.info("RealDynamicFeaturesExtractor inicializado")
    
    def _load_dynamic_config(self) -> Dict[str, Any]:
        """Carga configuración para extracción dinámica REAL."""
        default_config = {
            'min_transition_frames': 8,
            'max_transition_frames': 40,
            'gesture_stability_threshold': 5,
            'velocity_smoothing_window': 3,
            'min_movement_threshold': 0.005,
            'transition_detection_sensitivity': 0.85,
            'temporal_downsampling': 1,
            'normalize_temporal_features': True,
            'use_3d_trajectories': True,
            'velocity_threshold_percentile': 75,
            'acceleration_smoothing': True,
            'jerk_threshold': 0.1,
            'minimum_sequence_duration_ms': 200
        }
        
        return get_config('biometric.dynamic_features', default_config)
    
    def add_frame_real(self, landmarks, gesture_name: str, confidence: float, 
                      world_landmarks: Optional[Any] = None) -> bool:
        """Añade un frame REAL al buffer temporal y calcula características en tiempo real."""
        try:
            current_time = time.time()
            
            # Extraer posiciones 3D REALES
            if world_landmarks is not None:
                position_3d = self._extract_3d_positions_real(world_landmarks)
            else:
                position_3d = self._extract_2d_positions_real(landmarks)
            
            # Crear frame temporal REAL
            temporal_frame = TemporalFrame(
                frame_id=self.frame_counter,
                timestamp=current_time,
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                gesture_name=gesture_name,
                confidence=confidence,
                position_3d=position_3d,
                frame_quality=confidence
            )
            
            # Calcular velocidades y aceleraciones REALES
            if self.previous_frame is not None:
                temporal_frame.velocity_vectors = self._calculate_real_velocities(
                    self.previous_frame.position_3d, 
                    position_3d, 
                    current_time - self.previous_frame.timestamp
                )
                
                if len(self.temporal_buffer) > 0:
                    prev_velocities = self.temporal_buffer[-1].velocity_vectors
                    if prev_velocities is not None and temporal_frame.velocity_vectors is not None:
                        temporal_frame.acceleration_vectors = self._calculate_real_accelerations(
                            prev_velocities,
                            temporal_frame.velocity_vectors,
                            current_time - self.temporal_buffer[-1].timestamp
                        )
            
            # Añadir al buffer
            self.temporal_buffer.append(temporal_frame)
            self.previous_frame = temporal_frame
            self.frame_counter += 1
            self.frames_processed += 1
            
            # Detectar transiciones
            transition_detected = self._detect_real_transition(gesture_name, confidence)
            
            if transition_detected:
                logger.info(f"Transición REAL detectada: {self.current_gesture} → {gesture_name}")
                self.transitions_detected += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error añadiendo frame REAL: {e}")
            return False
    
    def _extract_3d_positions_real(self, world_landmarks) -> np.ndarray:
        """Extrae posiciones 3D REALES de world landmarks."""
        try:
            positions = np.zeros((21, 3), dtype=np.float32)
            
            for i, landmark in enumerate(world_landmarks.landmark):
                if i >= 21:
                    break
                positions[i] = [landmark.x, landmark.y, landmark.z]
            
            return positions
            
        except Exception as e:
            logger.error(f"Error extrayendo posiciones 3D reales: {e}")
            return np.zeros((21, 3), dtype=np.float32)
    
    def _extract_2d_positions_real(self, landmarks) -> np.ndarray:
        """Extrae posiciones 2D y estima Z."""
        try:
            positions = np.zeros((21, 3), dtype=np.float32)
            
            for i, landmark in enumerate(landmarks.landmark):
                if i >= 21:
                    break
                positions[i] = [landmark.x, landmark.y, landmark.z]
            
            return positions
            
        except Exception as e:
            logger.error(f"Error extrayendo posiciones 2D reales: {e}")
            return np.zeros((21, 3), dtype=np.float32)
    
    def _calculate_real_velocities(self, pos_prev: np.ndarray, pos_curr: np.ndarray, 
                                  delta_time: float) -> np.ndarray:
        """Calcula velocidades REALES entre frames consecutivos."""
        try:
            if delta_time <= 0:
                return np.zeros((21, 3), dtype=np.float32)
            
            velocities = (pos_curr - pos_prev) / delta_time
            
            if self.dynamic_config['velocity_smoothing_window'] > 1:
                velocities = self._apply_velocity_smoothing(velocities)
            
            return velocities.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error calculando velocidades reales: {e}")
            return np.zeros((21, 3), dtype=np.float32)
    
    def _calculate_real_accelerations(self, vel_prev: np.ndarray, vel_curr: np.ndarray,
                                     delta_time: float) -> np.ndarray:
        """Calcula aceleraciones REALES entre frames consecutivos."""
        try:
            if delta_time <= 0:
                return np.zeros((21, 3), dtype=np.float32)
            
            accelerations = (vel_curr - vel_prev) / delta_time
            
            if self.dynamic_config['acceleration_smoothing']:
                accelerations = self._apply_acceleration_smoothing(accelerations)
            
            return accelerations.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error calculando aceleraciones reales: {e}")
            return np.zeros((21, 3), dtype=np.float32)
    
    def _apply_velocity_smoothing(self, velocities: np.ndarray) -> np.ndarray:
        """Aplica suavizado a las velocidades REALES."""
        try:
            if len(self.temporal_buffer) >= 3:
                prev_velocities = []
                for frame in list(self.temporal_buffer)[-3:]:
                    if frame.velocity_vectors is not None:
                        prev_velocities.append(frame.velocity_vectors)
                
                if len(prev_velocities) >= 2:
                    weights = np.array([0.2, 0.3, 0.5])
                    all_velocities = np.array(prev_velocities[-2:] + [velocities])
                    return np.average(all_velocities, axis=0, weights=weights[-len(all_velocities):])
            
            return velocities
            
        except Exception as e:
            logger.error(f"Error en suavizado de velocidades: {e}")
            return velocities
    
    def _apply_acceleration_smoothing(self, accelerations: np.ndarray) -> np.ndarray:
        """Aplica suavizado a las aceleraciones REALES."""
        try:
            noise_threshold = 0.01
            accelerations[np.abs(accelerations) < noise_threshold] = 0
            
            return accelerations
            
        except Exception as e:
            logger.error(f"Error en suavizado de aceleraciones: {e}")
            return accelerations
    
    def _detect_real_transition(self, current_gesture: str, confidence: float) -> bool:
        """Detecta transiciones REALES entre gestos."""
        try:
            gesture_changed = current_gesture != self.current_gesture
            confidence_threshold = self.dynamic_config['transition_detection_sensitivity']
            stability_threshold = self.dynamic_config['gesture_stability_threshold']
            
            if not gesture_changed:
                self.gesture_stable_count += 1
                
                if self.transition_active and self.gesture_stable_count >= stability_threshold:
                    self._finalize_real_transition(current_gesture)
                    return True
                
                return False
            
            else:
                if confidence >= confidence_threshold:
                    if not self.transition_active:
                        self._start_real_transition(self.current_gesture, current_gesture)
                    
                    self.current_gesture = current_gesture
                    self.gesture_stable_count = 1
                    
                    return False
                else:
                    return False
            
        except Exception as e:
            logger.error(f"Error detectando transición real: {e}")
            return False
    
    def _start_real_transition(self, start_gesture: str, end_gesture: str):
        """Inicia una nueva transición REAL."""
        try:
            self.transition_active = True
            self.transition_start_frame = self.frame_counter
            
            logger.info(f"Iniciando transición REAL: {start_gesture} → {end_gesture}")
            
        except Exception as e:
            logger.error(f"Error iniciando transición real: {e}")
    
    def _finalize_real_transition(self, end_gesture: str):
        """Finaliza una transición y extrae características."""
        try:
            if not self.transition_active:
                return
            
            transition_frames = []
            frames_in_transition = self.frame_counter - self.transition_start_frame
            
            if frames_in_transition <= len(self.temporal_buffer):
                transition_frames = list(self.temporal_buffer)[-frames_in_transition:]
            
            if len(transition_frames) < self.dynamic_config['min_transition_frames']:
                logger.info("Transición muy corta, ignorando")
                self.transition_active = False
                return
            
            if len(transition_frames) >= 2:
                duration_ms = (transition_frames[-1].timestamp - transition_frames[0].timestamp) * 1000
                
                if duration_ms < self.dynamic_config['minimum_sequence_duration_ms']:
                    logger.info(f"Transición muy rápida ({duration_ms:.1f}ms), ignorando")
                    self.transition_active = False
                    return
            
            transition_event = TransitionEvent(
                start_frame=self.transition_start_frame,
                end_frame=self.frame_counter,
                start_gesture=self.current_gesture,
                end_gesture=end_gesture,
                transition_type=f"{self.current_gesture}_to_{end_gesture}",
                duration_ms=duration_ms if len(transition_frames) >= 2 else 0,
                transition_frames=transition_frames.copy(),
                motion_type=self._classify_motion_type_real(transition_frames),
                confidence=np.mean([f.confidence for f in transition_frames])
            )
            
            self.detected_transitions.append(transition_event)
            self.transition_active = False
            
            logger.info(f"Transición REAL finalizada: {len(transition_frames)} frames, {duration_ms:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error finalizando transición real: {e}")
            self.transition_active = False
    
    def _classify_motion_type_real(self, frames: List[TemporalFrame]) -> MotionType:
        """Clasifica el tipo de movimiento REAL basado en las características."""
        try:
            if len(frames) < 3:
                return MotionType.LINEAR
            
            velocities = []
            for frame in frames:
                if frame.velocity_vectors is not None:
                    vel_magnitude = np.mean(np.linalg.norm(frame.velocity_vectors, axis=1))
                    velocities.append(vel_magnitude)
            
            if len(velocities) < 2:
                return MotionType.LINEAR
            
            velocities = np.array(velocities)
            
            velocity_std = np.std(velocities)
            velocity_mean = np.mean(velocities)
            
            if velocity_mean == 0:
                return MotionType.SMOOTH
            
            coefficient_variation = velocity_std / velocity_mean
            
            if coefficient_variation < 0.2:
                return MotionType.SMOOTH
            elif coefficient_variation > 0.8:
                return MotionType.ABRUPT
            elif self._has_oscillations_real(velocities):
                return MotionType.OSCILLATORY
            elif self._is_curved_motion_real(frames):
                return MotionType.CURVED
            else:
                return MotionType.LINEAR
                
        except Exception as e:
            logger.error(f"Error clasificando tipo de movimiento: {e}")
            return MotionType.SMOOTH
    
    def _has_oscillations_real(self, velocities: np.ndarray) -> bool:
        """Detecta oscilaciones REALES en la velocidad."""
        try:
            velocity_diff = np.diff(velocities)
            sign_changes = np.sum(np.diff(np.sign(velocity_diff)) != 0)
            
            oscillation_ratio = sign_changes / len(velocities)
            return oscillation_ratio > 0.3
            
        except Exception as e:
            logger.error(f"Error detectando oscilaciones: {e}")
            return False
    
    def _is_curved_motion_real(self, frames: List[TemporalFrame]) -> bool:
        """Detecta movimiento curvo REAL basado en trayectorias."""
        try:
            if len(frames) < 5:
                return False
            
            key_landmarks = [4, 8, 12, 16, 20]
            
            for landmark_idx in key_landmarks:
                positions = []
                for frame in frames:
                    if frame.position_3d is not None:
                        positions.append(frame.position_3d[landmark_idx])
                
                if len(positions) >= 5:
                    positions = np.array(positions)
                    
                    start_pos = positions[0]
                    end_pos = positions[-1]
                    
                    line_vector = end_pos - start_pos
                    line_length = np.linalg.norm(line_vector)
                    
                    if line_length > 0:
                        max_deviation = 0
                        for i, pos in enumerate(positions):
                            t = i / (len(positions) - 1)
                            expected_pos = start_pos + t * line_vector
                            deviation = np.linalg.norm(pos - expected_pos)
                            max_deviation = max(max_deviation, deviation)
                        
                        curvature_ratio = max_deviation / line_length
                        if curvature_ratio > 0.15:
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detectando movimiento curvo: {e}")
            return False
    
    def extract_transition_features_real(self, transition_event: TransitionEvent) -> Optional[DynamicFeatureVector]:
        """Extrae características dinámicas REALES de un evento de transición."""
        try:
            transition_frames = transition_event.transition_frames
            
            if len(transition_frames) < self.dynamic_config['min_transition_frames']:
                logger.error("Insuficientes frames para extracción REAL")
                return None
            
            velocity_profile = self._extract_real_velocity_profile(transition_frames)
            acceleration_profile = self._extract_real_acceleration_profile(transition_frames)
            trajectory_profile = self._extract_real_trajectory_profile(transition_frames)
            
            velocity_features = self._extract_real_velocity_features(velocity_profile)
            acceleration_features = self._extract_real_acceleration_features(acceleration_profile)
            trajectory_features = self._extract_real_trajectory_features(trajectory_profile)
            timing_features = self._extract_real_timing_features(transition_frames)
            rhythm_features = self._extract_real_rhythm_features(transition_frames)
            transition_features = self._extract_real_transition_characteristics(transition_frames)
            
            feature_vector = DynamicFeatureVector(
                velocity_features=velocity_features,
                acceleration_features=acceleration_features,
                trajectory_features=trajectory_features,
                timing_features=timing_features,
                rhythm_features=rhythm_features,
                transition_features=transition_features
            )
            
            if self.dynamic_config['normalize_temporal_features']:
                feature_vector = self._normalize_real_features(feature_vector)
            
            if self._validate_real_feature_quality(feature_vector):
                self.successful_extractions += 1
                logger.info(f"Características dinámicas extraídas: {feature_vector.dimension} dim")
                return feature_vector
            else:
                logger.error("Vector dinámico no cumple criterios de calidad")
                return None
                
        except Exception as e:
            logger.error(f"Error extrayendo características de transición: {e}")
            return None
    
    def _extract_real_velocity_profile(self, frames: List[TemporalFrame]) -> VelocityProfile:
        """Extrae perfil de velocidad REAL de la secuencia."""
        try:
            velocities_sequence = []
            valid_frames = []
            
            for frame in frames:
                if frame.velocity_vectors is not None:
                    velocities_sequence.append(frame.velocity_vectors)
                    valid_frames.append(frame)
            
            if not velocities_sequence:
                return VelocityProfile(
                    landmark_velocities=np.zeros((21, len(frames), 3)),
                    peak_velocities=np.zeros(21),
                    avg_velocities=np.zeros(21),
                    velocity_patterns=np.zeros(50),
                    timing_features=np.zeros(20)
                )
            
            velocities_array = np.array(velocities_sequence)
            landmark_velocities = np.transpose(velocities_array, (1, 0, 2))
            
            velocity_magnitudes = np.linalg.norm(landmark_velocities, axis=2)
            peak_velocities = np.max(velocity_magnitudes, axis=1)
            avg_velocities = np.mean(velocity_magnitudes, axis=1)
            
            velocity_patterns = self._calculate_real_velocity_patterns(velocity_magnitudes)
            timing_features = self._calculate_real_velocity_timing(velocity_magnitudes)
            
            return VelocityProfile(
                landmark_velocities=landmark_velocities,
                peak_velocities=peak_velocities,
                avg_velocities=avg_velocities,
                velocity_patterns=velocity_patterns,
                timing_features=timing_features
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo perfil de velocidad REAL: {e}")
            return VelocityProfile(
                landmark_velocities=np.zeros((21, 1, 3)),
                peak_velocities=np.zeros(21),
                avg_velocities=np.zeros(21),
                velocity_patterns=np.zeros(50),
                timing_features=np.zeros(20)
            )
    
    def _calculate_real_velocity_patterns(self, velocity_magnitudes: np.ndarray) -> np.ndarray:
        """Calcula patrones de velocidad REALES (50 dim)."""
        try:
            features = []
            
            if velocity_magnitudes.shape[1] < 2:
                return np.zeros(50, dtype=np.float32)
            
            for landmark_idx in range(min(velocity_magnitudes.shape[0], 21)):
                landmark_vels = velocity_magnitudes[landmark_idx]
                
                features.extend([
                    np.max(landmark_vels),
                    np.mean(landmark_vels),
                ])
            
            all_velocities = velocity_magnitudes.flatten()
            features.extend([
                np.percentile(all_velocities, 25),
                np.percentile(all_velocities, 75),
                np.std(all_velocities),
                np.var(all_velocities),
                np.median(all_velocities),
                np.sum(all_velocities > np.mean(all_velocities)) / len(all_velocities),
                np.max(all_velocities) - np.min(all_velocities),
                len(all_velocities[all_velocities > 0]) / len(all_velocities)
            ])
            
            while len(features) < 50:
                features.append(0.0)
            
            return np.array(features[:50], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error calculando patrones de velocidad REALES: {e}")
            return np.zeros(50, dtype=np.float32)
    
    def _calculate_real_velocity_timing(self, velocity_magnitudes: np.ndarray) -> np.ndarray:
        """Calcula características temporales de velocidad (20 dim)."""
        try:
            features = []
            
            if velocity_magnitudes.shape[1] < 2:
                return np.zeros(20, dtype=np.float32)
            
            timing_stats = []
            for landmark_idx in range(min(velocity_magnitudes.shape[0], 21)):
                landmark_vels = velocity_magnitudes[landmark_idx]
                
                if len(landmark_vels) > 1:
                    max_vel_time = np.argmax(landmark_vels) / len(landmark_vels)
                    timing_stats.append(max_vel_time)
            
            if timing_stats:
                timing_array = np.array(timing_stats)
                features.extend([
                    np.mean(timing_array),
                    np.std(timing_array),
                    np.min(timing_array),
                    np.max(timing_array),
                    np.median(timing_array),
                ])
            else:
                features.extend([0.0] * 5)
            
            for landmark_idx in range(min(velocity_magnitudes.shape[0], 5)):
                landmark_vels = velocity_magnitudes[landmark_idx]
                
                if len(landmark_vels) > 2:
                    peak_idx = np.argmax(landmark_vels)
                    accel_phase = landmark_vels[:peak_idx+1] if peak_idx > 0 else [landmark_vels[0]]
                    decel_phase = landmark_vels[peak_idx:] if peak_idx < len(landmark_vels)-1 else [landmark_vels[-1]]
                    
                    features.extend([
                        len(accel_phase) / len(landmark_vels),
                        len(decel_phase) / len(landmark_vels),
                        np.mean(accel_phase) if len(accel_phase) > 0 else 0,
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
            
            while len(features) < 20:
                features.append(0.0)
            
            return np.array(features[:20], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error calculando timing de velocidad REAL: {e}")
            return np.zeros(20, dtype=np.float32)
    
    def _extract_real_acceleration_profile(self, frames: List[TemporalFrame]) -> AccelerationProfile:
        """Extrae perfil de aceleración REAL de la secuencia."""
        try:
            accelerations_sequence = []
            
            for frame in frames:
                if frame.acceleration_vectors is not None:
                    accelerations_sequence.append(frame.acceleration_vectors)
            
            if not accelerations_sequence:
                return AccelerationProfile(
                    landmark_accelerations=np.zeros((21, len(frames), 3)),
                    peak_accelerations=np.zeros(21),
                    avg_accelerations=np.zeros(21),
                    jerk_patterns=np.zeros(30),
                    smoothness_metrics=np.zeros(15)
                )
            
            accelerations_array = np.array(accelerations_sequence)
            landmark_accelerations = np.transpose(accelerations_array, (1, 0, 2))
            
            acceleration_magnitudes = np.linalg.norm(landmark_accelerations, axis=2)
            peak_accelerations = np.max(acceleration_magnitudes, axis=1)
            avg_accelerations = np.mean(acceleration_magnitudes, axis=1)
            
            jerk_patterns = self._calculate_real_jerk_patterns(landmark_accelerations)
            smoothness_metrics = self._calculate_real_smoothness_metrics(acceleration_magnitudes)
            
            return AccelerationProfile(
                landmark_accelerations=landmark_accelerations,
                peak_accelerations=peak_accelerations,
                avg_accelerations=avg_accelerations,
                jerk_patterns=jerk_patterns,
                smoothness_metrics=smoothness_metrics
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo perfil de aceleración: {e}")
            return AccelerationProfile(
                landmark_accelerations=np.zeros((21, 1, 3)),
                peak_accelerations=np.zeros(21),
                avg_accelerations=np.zeros(21),
                jerk_patterns=np.zeros(30),
                smoothness_metrics=np.zeros(15)
            )
    
    def _calculate_real_jerk_patterns(self, landmark_accelerations: np.ndarray) -> np.ndarray:
        """Calcula patrones de jerk (30 dim)."""
        try:
            features = []
            
            if landmark_accelerations.shape[1] < 2:
                return np.zeros(30, dtype=np.float32)
            
            key_landmarks = [4, 8, 12, 16, 20]
            
            for landmark_idx in key_landmarks:
                if landmark_idx < landmark_accelerations.shape[0]:
                    landmark_accel = landmark_accelerations[landmark_idx]
                    
                    if landmark_accel.shape[0] > 1:
                        jerk_vectors = np.diff(landmark_accel, axis=0)
                        jerk_magnitudes = np.linalg.norm(jerk_vectors, axis=1)
                        
                        features.extend([
                            np.max(jerk_magnitudes) if len(jerk_magnitudes) > 0 else 0,
                            np.mean(jerk_magnitudes) if len(jerk_magnitudes) > 0 else 0,
                            np.std(jerk_magnitudes) if len(jerk_magnitudes) > 0 else 0,
                            np.sum(jerk_magnitudes > np.mean(jerk_magnitudes)) / len(jerk_magnitudes) if len(jerk_magnitudes) > 0 else 0,
                            np.percentile(jerk_magnitudes, 90) if len(jerk_magnitudes) > 0 else 0,
                            len(jerk_magnitudes[jerk_magnitudes > 0]) / len(jerk_magnitudes) if len(jerk_magnitudes) > 0 else 0
                        ])
                    else:
                        features.extend([0.0] * 6)
                else:
                    features.extend([0.0] * 6)
            
            return np.array(features[:30], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error calculando patrones de jerk: {e}")
            return np.zeros(30, dtype=np.float32)
    
    def _calculate_real_smoothness_metrics(self, acceleration_magnitudes: np.ndarray) -> np.ndarray:
        """Calcula métricas de suavidad (15 dim)."""
        try:
            features = []
            
            if acceleration_magnitudes.shape[1] < 2:
                return np.zeros(15, dtype=np.float32)
            
            all_accelerations = acceleration_magnitudes.flatten()
            
            features.extend([
                np.std(all_accelerations),
                np.mean(np.abs(np.diff(all_accelerations))),
                np.max(all_accelerations) - np.min(all_accelerations),
                np.var(all_accelerations),
                np.percentile(all_accelerations, 95),
            ])
            
            smoothness_per_landmark = []
            for landmark_idx in range(min(acceleration_magnitudes.shape[0], 21)):
                landmark_accels = acceleration_magnitudes[landmark_idx]
                
                if len(landmark_accels) > 1:
                    changes = np.abs(np.diff(landmark_accels))
                    smoothness = 1.0 / (1.0 + np.mean(changes))
                    smoothness_per_landmark.append(smoothness)
            
            if smoothness_per_landmark:
                smoothness_array = np.array(smoothness_per_landmark)
                features.extend([
                    np.mean(smoothness_array),
                    np.std(smoothness_array),
                    np.min(smoothness_array),
                    np.max(smoothness_array),
                    np.median(smoothness_array),
                    np.percentile(smoothness_array, 25),
                    np.percentile(smoothness_array, 75),
                    np.var(smoothness_array),
                    len(smoothness_array[smoothness_array > np.mean(smoothness_array)]) / len(smoothness_array),
                    np.sum(smoothness_array),
                ])
            else:
                features.extend([0.0] * 10)
            
            return np.array(features[:15], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error calculando métricas de suavidad: {e}")
            return np.zeros(15, dtype=np.float32)
        
    
    def _extract_real_trajectory_profile(self, frames: List[TemporalFrame]) -> TrajectoryProfile:
        """Extrae perfil de trayectoria de la secuencia."""
        try:
            positions_sequence = []
            
            for frame in frames:
                if frame.position_3d is not None:
                    positions_sequence.append(frame.position_3d)
            
            if not positions_sequence:
                return TrajectoryProfile(
                    landmark_trajectories=np.zeros((21, len(frames), 3)),
                    trajectory_lengths=np.zeros(21),
                    curvature_profiles=np.zeros(21),
                    direction_changes=np.zeros(21),
                    spatial_efficiency=np.zeros(21)
                )
            
            positions_array = np.array(positions_sequence)
            landmark_trajectories = np.transpose(positions_array, (1, 0, 2))
            
            trajectory_lengths = self._calculate_real_trajectory_lengths(landmark_trajectories)
            curvature_profiles = self._calculate_real_curvature_profiles(landmark_trajectories)
            direction_changes = self._calculate_real_direction_changes(landmark_trajectories)
            spatial_efficiency = self._calculate_real_spatial_efficiency(landmark_trajectories)
            
            return TrajectoryProfile(
                landmark_trajectories=landmark_trajectories,
                trajectory_lengths=trajectory_lengths,
                curvature_profiles=curvature_profiles,
                direction_changes=direction_changes,
                spatial_efficiency=spatial_efficiency
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo perfil de trayectoria REAL: {e}")
            return TrajectoryProfile(
                landmark_trajectories=np.zeros((21, 1, 3)),
                trajectory_lengths=np.zeros(21),
                curvature_profiles=np.zeros(21),
                direction_changes=np.zeros(21),
                spatial_efficiency=np.zeros(21)
            )
    
    def _calculate_real_trajectory_lengths(self, landmark_trajectories: np.ndarray) -> np.ndarray:
        """Calcula longitudes de trayectoria."""
        try:
            lengths = np.zeros(landmark_trajectories.shape[0])
            
            for landmark_idx in range(landmark_trajectories.shape[0]):
                trajectory = landmark_trajectories[landmark_idx]
                
                if trajectory.shape[0] > 1:
                    distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
                    lengths[landmark_idx] = np.sum(distances)
            
            return lengths
            
        except Exception as e:
            logger.error(f"Error calculando longitudes de trayectoria: {e}")
            return np.zeros(21)
    
    def _calculate_real_curvature_profiles(self, landmark_trajectories: np.ndarray) -> np.ndarray:
        """Calcula perfiles de curvatura."""
        try:
            curvatures = np.zeros(landmark_trajectories.shape[0])
            
            for landmark_idx in range(landmark_trajectories.shape[0]):
                trajectory = landmark_trajectories[landmark_idx]
                
                if trajectory.shape[0] > 2:
                    curvature_values = []
                    
                    for i in range(1, trajectory.shape[0] - 1):
                        p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
                        
                        v1 = p2 - p1
                        v2 = p3 - p2
                        
                        cross_product = np.cross(v1, v2)
                        
                        v1_mag = np.linalg.norm(v1)
                        v2_mag = np.linalg.norm(v2)
                        
                        if v1_mag > 0 and v2_mag > 0:
                            if v1.ndim == 1 and v2.ndim == 1:
                                curvature = np.linalg.norm(cross_product) / (v1_mag * v2_mag)
                            else:
                                curvature = np.abs(cross_product) / (v1_mag * v2_mag)
                            curvature_values.append(curvature)
                    
                    if curvature_values:
                        curvatures[landmark_idx] = np.mean(curvature_values)
            
            return curvatures
            
        except Exception as e:
            logger.error(f"Error calculando curvatura: {e}")
            return np.zeros(21)
    
    def _calculate_real_direction_changes(self, landmark_trajectories: np.ndarray) -> np.ndarray:
        """Calcula cambios de dirección."""
        try:
            direction_changes = np.zeros(landmark_trajectories.shape[0])
            
            for landmark_idx in range(landmark_trajectories.shape[0]):
                trajectory = landmark_trajectories[landmark_idx]
                
                if trajectory.shape[0] > 2:
                    directions = np.diff(trajectory, axis=0)
                    
                    direction_norms = np.linalg.norm(directions, axis=1)
                    valid_directions = direction_norms > 1e-6
                    
                    if np.sum(valid_directions) > 1:
                        normalized_directions = directions[valid_directions]
                        normalized_directions = normalized_directions / direction_norms[valid_directions, np.newaxis]
                        
                        angle_changes = []
                        for i in range(len(normalized_directions) - 1):
                            dot_product = np.dot(normalized_directions[i], normalized_directions[i + 1])
                            dot_product = np.clip(dot_product, -1.0, 1.0)
                            angle_change = np.arccos(dot_product)
                            angle_changes.append(angle_change)
                        
                        if angle_changes:
                            direction_changes[landmark_idx] = np.sum(angle_changes)
            
            return direction_changes
            
        except Exception as e:
            logger.error(f"Error calculando cambios de dirección: {e}")
            return np.zeros(21)
    
    def _calculate_real_spatial_efficiency(self, landmark_trajectories: np.ndarray) -> np.ndarray:
        """Calcula eficiencia espacial."""
        try:
            efficiency = np.zeros(landmark_trajectories.shape[0])
            
            for landmark_idx in range(landmark_trajectories.shape[0]):
                trajectory = landmark_trajectories[landmark_idx]
                
                if trajectory.shape[0] > 1:
                    direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
                    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
                    
                    if total_distance > 0:
                        efficiency[landmark_idx] = direct_distance / total_distance
                    else:
                        efficiency[landmark_idx] = 1.0
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Error calculando eficiencia espacial: {e}")
            return np.zeros(21)
    
    def _extract_real_velocity_features(self, velocity_profile: VelocityProfile) -> np.ndarray:
        """Extrae características de velocidad (70 dim)."""
        try:
            features = []
            
            features.extend(velocity_profile.peak_velocities.tolist())
            features.extend(velocity_profile.avg_velocities.tolist())
            
            features.extend(velocity_profile.velocity_patterns[:20].tolist())
            features.extend(velocity_profile.timing_features[:7].tolist())
            
            while len(features) < 70:
                features.append(0.0)
            
            return np.array(features[:70], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características de velocidad: {e}")
            return np.zeros(70, dtype=np.float32)
    
    def _extract_real_acceleration_features(self, acceleration_profile: AccelerationProfile) -> np.ndarray:
        """Extrae características de aceleración (65 dim)."""
        try:
            features = []
            
            features.extend(acceleration_profile.peak_accelerations.tolist())
            features.extend(acceleration_profile.avg_accelerations.tolist())
            
            features.extend(acceleration_profile.jerk_patterns[:15].tolist())
            features.extend(acceleration_profile.smoothness_metrics[:7].tolist())
            
            while len(features) < 65:
                features.append(0.0)
            
            return np.array(features[:65], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características de aceleración: {e}")
            return np.zeros(65, dtype=np.float32)
    
    def _extract_real_trajectory_features(self, trajectory_profile: TrajectoryProfile) -> np.ndarray:
        """Extrae características de trayectoria (85 dim)."""
        try:
            features = []
            
            features.extend(trajectory_profile.trajectory_lengths.tolist())
            features.extend(trajectory_profile.curvature_profiles.tolist())
            features.extend(trajectory_profile.direction_changes.tolist())
            features.extend(trajectory_profile.spatial_efficiency.tolist())
            
            global_efficiency = np.mean(trajectory_profile.spatial_efficiency)
            features.append(global_efficiency)
            
            return np.array(features[:85], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características de trayectoria: {e}")
            return np.zeros(85, dtype=np.float32)
    
    def _extract_real_timing_features(self, frames: List[TemporalFrame]) -> np.ndarray:
        """Extrae características temporales (40 dim)."""
        try:
            features = []
            
            if len(frames) < 2:
                return np.zeros(40, dtype=np.float32)
            
            total_duration = frames[-1].timestamp - frames[0].timestamp
            avg_frame_interval = total_duration / (len(frames) - 1) if len(frames) > 1 else 0
            
            features.extend([
                total_duration,
                avg_frame_interval,
                len(frames),
                1.0 / avg_frame_interval if avg_frame_interval > 0 else 0,
            ])
            
            if len(frames) > 2:
                intervals = []
                for i in range(1, len(frames)):
                    interval = frames[i].timestamp - frames[i-1].timestamp
                    intervals.append(interval)
                
                intervals = np.array(intervals)
                features.extend([
                    np.std(intervals),
                    np.min(intervals),
                    np.max(intervals),
                    np.median(intervals),
                ])
            else:
                features.extend([0.0] * 4)
            
            confidences = [frame.confidence for frame in frames]
            features.extend([
                np.mean(confidences),
                np.std(confidences),
                np.min(confidences),
                np.max(confidences),
            ])
            
            qualities = [frame.frame_quality for frame in frames]
            features.extend([
                np.mean(qualities),
                np.std(qualities),
                np.min(qualities),
                np.max(qualities),
            ])
            
            gesture_changes = 0
            for i in range(1, len(frames)):
                if frames[i].gesture_name != frames[i-1].gesture_name:
                    gesture_changes += 1
            
            features.extend([
                gesture_changes,
                gesture_changes / len(frames),
                len(set(frame.gesture_name for frame in frames)),
            ])
            
            third = len(frames) // 3
            if third > 0:
                phase1_frames = frames[:third]
                phase2_frames = frames[third:2*third]
                phase3_frames = frames[2*third:]
                
                phase1_conf = np.mean([f.confidence for f in phase1_frames])
                phase2_conf = np.mean([f.confidence for f in phase2_frames])
                phase3_conf = np.mean([f.confidence for f in phase3_frames])
                
                features.extend([
                    phase1_conf,
                    phase2_conf,
                    phase3_conf,
                    phase3_conf - phase1_conf,
                    np.std([phase1_conf, phase2_conf, phase3_conf]),
                ])
            else:
                features.extend([0.0] * 5)
            
            while len(features) < 40:
                features.append(0.0)
            
            return np.array(features[:40], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características temporales: {e}")
            return np.zeros(40, dtype=np.float32)
    
    def _extract_real_rhythm_features(self, frames: List[TemporalFrame]) -> np.ndarray:
        """Extrae características de ritmo (35 dim)."""
        try:
            features = []
            
            if len(frames) < 3:
                return np.zeros(35, dtype=np.float32)
            
            velocity_rhythms = []
            for frame in frames:
                if frame.velocity_vectors is not None:
                    frame_velocity = np.mean(np.linalg.norm(frame.velocity_vectors, axis=1))
                    velocity_rhythms.append(frame_velocity)
            
            if len(velocity_rhythms) > 2:
                velocity_rhythms = np.array(velocity_rhythms)
                
                features.extend([
                    np.mean(velocity_rhythms),
                    np.std(velocity_rhythms),
                    np.max(velocity_rhythms),
                    np.min(velocity_rhythms),
                    np.median(velocity_rhythms),
                ])
                
                if len(velocity_rhythms) > 4:
                    autocorr_values = []
                    for lag in range(1, min(5, len(velocity_rhythms)//2)):
                        autocorr = np.corrcoef(velocity_rhythms[:-lag], velocity_rhythms[lag:])[0, 1]
                        autocorr_values.append(autocorr if not np.isnan(autocorr) else 0)
                    
                    features.extend(autocorr_values[:4])
                    features.append(np.max(autocorr_values) if autocorr_values else 0)
                else:
                    features.extend([0.0] * 5)
                
                rhythm_changes = np.diff(velocity_rhythms)
                features.extend([
                    np.mean(np.abs(rhythm_changes)),
                    np.std(rhythm_changes),
                    np.sum(rhythm_changes > 0) / len(rhythm_changes),
                    np.sum(rhythm_changes < 0) / len(rhythm_changes),
                ])
                
                percentiles = [25, 50, 75, 90]
                rhythm_percentiles = [np.percentile(velocity_rhythms, p) for p in percentiles]
                features.extend(rhythm_percentiles)
                
                features.extend([
                    np.var(velocity_rhythms),
                    len(velocity_rhythms[velocity_rhythms > np.mean(velocity_rhythms)]) / len(velocity_rhythms),
                ])
                
            else:
                features.extend([0.0] * 20)
            
            gesture_transitions = []
            for i in range(1, len(frames)):
                if frames[i].gesture_name != frames[i-1].gesture_name:
                    transition_time = frames[i].timestamp - frames[i-1].timestamp
                    gesture_transitions.append(transition_time)
            
            if gesture_transitions:
                gesture_transitions = np.array(gesture_transitions)
                features.extend([
                    np.mean(gesture_transitions),
                    np.std(gesture_transitions),
                    np.min(gesture_transitions),
                    np.max(gesture_transitions),
                    len(gesture_transitions),
                ])
            else:
                features.extend([0.0] * 5)
            
            timestamps = [frame.timestamp for frame in frames]
            if len(timestamps) > 2:
                intervals = np.diff(timestamps)
                features.extend([
                    np.mean(intervals),
                    np.std(intervals),
                    np.max(intervals) - np.min(intervals),
                ])
            else:
                features.extend([0.0] * 3)
            
            confidences = [frame.confidence for frame in frames]
            if len(confidences) > 2:
                conf_changes = np.diff(confidences)
                features.extend([
                    np.mean(np.abs(conf_changes)),
                    np.std(confidences),
                ])
            else:
                features.extend([0.0] * 2)
            
            while len(features) < 35:
                features.append(0.0)
            
            return np.array(features[:35], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características de ritmo: {e}")
            return np.zeros(35, dtype=np.float32)
        
    def _extract_real_transition_characteristics(self, frames: List[TemporalFrame]) -> np.ndarray:
        """Extrae características de transición (25 dim)."""
        try:
            features = []
            
            if len(frames) < 2:
                return np.zeros(25, dtype=np.float32)
            
            start_gesture = frames[0].gesture_name
            end_gesture = frames[-1].gesture_name
            
            features.extend([
                1.0 if start_gesture != end_gesture else 0.0,
                len(set(frame.gesture_name for frame in frames)),
            ])
            
            transition_confidence = np.mean([frame.confidence for frame in frames])
            confidence_stability = np.std([frame.confidence for frame in frames])
            
            features.extend([
                transition_confidence,
                confidence_stability,
            ])
            
            total_duration = frames[-1].timestamp - frames[0].timestamp
            features.extend([
                total_duration,
                len(frames),
                len(frames) / total_duration if total_duration > 0 else 0,
            ])
            
            if len(frames) > 2:
                velocity_smoothness = []
                for frame in frames:
                    if frame.velocity_vectors is not None:
                        frame_vel_magnitude = np.mean(np.linalg.norm(frame.velocity_vectors, axis=1))
                        velocity_smoothness.append(frame_vel_magnitude)
                
                if len(velocity_smoothness) > 2:
                    velocity_changes = np.abs(np.diff(velocity_smoothness))
                    features.extend([
                        np.mean(velocity_changes),
                        np.std(velocity_changes),
                        np.max(velocity_changes),
                        np.sum(velocity_changes > np.mean(velocity_changes)) / len(velocity_changes),
                    ])
                else:
                    features.extend([0.0] * 4)
            else:
                features.extend([0.0] * 4)
            
            if len(frames) >= 3:
                third = len(frames) // 3
                
                inicio_frames = frames[:third] if third > 0 else [frames[0]]
                medio_frames = frames[third:2*third] if third > 0 else [frames[len(frames)//2]]
                final_frames = frames[2*third:] if third > 0 else [frames[-1]]
                
                inicio_conf = np.mean([f.confidence for f in inicio_frames])
                medio_conf = np.mean([f.confidence for f in medio_frames])
                final_conf = np.mean([f.confidence for f in final_frames])
                
                features.extend([
                    inicio_conf,
                    medio_conf,
                    final_conf,
                    final_conf - inicio_conf,
                    abs(medio_conf - (inicio_conf + final_conf) / 2),
                ])
            else:
                features.extend([0.0] * 5)
            
            total_movement = 0
            max_landmark_movement = 0
            
            if len(frames) > 1:
                start_positions = frames[0].position_3d
                end_positions = frames[-1].position_3d
                
                if start_positions is not None and end_positions is not None:
                    landmark_movements = np.linalg.norm(end_positions - start_positions, axis=1)
                    total_movement = np.sum(landmark_movements)
                    max_landmark_movement = np.max(landmark_movements)
            
            features.extend([
                total_movement,
                max_landmark_movement,
                total_movement / len(frames) if len(frames) > 0 else 0,
            ])
            
            if total_duration > 0:
                movement_efficiency = total_movement / total_duration
                features.append(movement_efficiency)
            else:
                features.append(0.0)
            
            gesture_changes = sum(1 for i in range(1, len(frames)) if frames[i].gesture_name != frames[i-1].gesture_name)
            complexity = gesture_changes / len(frames) if len(frames) > 0 else 0
            features.append(complexity)
            
            while len(features) < 25:
                features.append(0.0)
            
            return np.array(features[:25], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características de transición: {e}")
            return np.zeros(25, dtype=np.float32)
    
    def _normalize_real_features(self, feature_vector: DynamicFeatureVector) -> DynamicFeatureVector:
        """Normaliza el vector de características dinámicas."""
        try:
            def robust_normalize_real(arr):
                """Normalización robusta REAL usando mediana y MAD."""
                if len(arr) == 0:
                    return arr
                
                median = np.median(arr)
                mad = np.median(np.abs(arr - median))
                
                if mad == 0:
                    return arr - median
                
                return (arr - median) / mad
            
            normalized_velocity = robust_normalize_real(feature_vector.velocity_features)
            normalized_acceleration = robust_normalize_real(feature_vector.acceleration_features)
            normalized_trajectory = robust_normalize_real(feature_vector.trajectory_features)
            normalized_timing = robust_normalize_real(feature_vector.timing_features)
            normalized_rhythm = robust_normalize_real(feature_vector.rhythm_features)
            normalized_transition = robust_normalize_real(feature_vector.transition_features)
            
            return DynamicFeatureVector(
                velocity_features=normalized_velocity,
                acceleration_features=normalized_acceleration,
                trajectory_features=normalized_trajectory,
                timing_features=normalized_timing,
                rhythm_features=normalized_rhythm,
                transition_features=normalized_transition
            )
            
        except Exception as e:
            logger.error(f"Error normalizando características dinámicas: {e}")
            return feature_vector
    
    def _validate_real_feature_quality(self, feature_vector: DynamicFeatureVector) -> bool:
        """Valida calidad del vector de características dinámicas."""
        try:
            complete_vector = feature_vector.complete_vector
            
            if not np.all(np.isfinite(complete_vector)):
                logger.error("Vector contiene NaN o infinitos")
                return False
            
            if np.all(complete_vector == 0):
                logger.error("Vector completamente vacío")
                return False
            
            if np.std(complete_vector) < 1e-6:
                logger.error("Vector sin variabilidad suficiente")
                return False
            
            if len(complete_vector) != 320:
                logger.error(f"Dimensión incorrecta: {len(complete_vector)} != 320")
                return False
            
            velocity_range = np.max(feature_vector.velocity_features) - np.min(feature_vector.velocity_features)
            acceleration_range = np.max(feature_vector.acceleration_features) - np.min(feature_vector.acceleration_features)
            
            if velocity_range < 1e-8 or acceleration_range < 1e-8:
                logger.error("Rangos de características demasiado pequeños para ser reales")
                return False
            
            logger.info("Vector de características dinámicas validado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error validando calidad de características dinámicas: {e}")
            return False
    
    def get_last_transition_real(self) -> Optional[TransitionEvent]:
        """Obtiene la última transición detectada."""
        if self.detected_transitions:
            return self.detected_transitions[-1]
        return None
    
    def extract_features_from_sequence_real(self, landmarks_sequence: List[Any], 
                                          gesture_sequence: List[str],
                                          timestamps: List[float]) -> Optional[DynamicFeatureVector]:
        """Extrae características dinámicas REALES de una secuencia completa."""
        try:
            if len(landmarks_sequence) != len(gesture_sequence) or len(landmarks_sequence) != len(timestamps):
                logger.error("Longitudes de secuencias no coinciden")
                return None
            
            if len(landmarks_sequence) < self.dynamic_config['min_transition_frames']:
                logger.error("Secuencia demasiado corta para extracción REAL")
                return None
            
            self.reset_state()
            
            for i, (landmarks, gesture, timestamp) in enumerate(zip(landmarks_sequence, gesture_sequence, timestamps)):
                if timestamp == 0:
                    timestamp = i * 0.033
                
                success = self.add_frame_real(
                    landmarks=landmarks,
                    gesture_name=gesture,
                    confidence=1.0,
                    world_landmarks=None
                )
                
                if not success:
                    logger.error(f"Error procesando frame {i}")
                    continue
            
            if self.transition_active:
                self._finalize_real_transition(gesture_sequence[-1])
            
            last_transition = self.get_last_transition_real()
            
            if last_transition:
                return self.extract_transition_features_real(last_transition)
            else:
                artificial_transition = TransitionEvent(
                    start_frame=0,
                    end_frame=len(landmarks_sequence),
                    start_gesture=gesture_sequence[0],
                    end_gesture=gesture_sequence[-1],
                    transition_type=f"{gesture_sequence[0]}_to_{gesture_sequence[-1]}",
                    duration_ms=(timestamps[-1] - timestamps[0]) * 1000 if len(timestamps) > 1 else 0,
                    transition_frames=list(self.temporal_buffer),
                    motion_type=MotionType.SMOOTH,
                    confidence=1.0
                )
                
                return self.extract_transition_features_real(artificial_transition)
                
        except Exception as e:
            logger.error(f"Error extrayendo características de secuencia: {e}")
            return None
    
    def get_extraction_stats_real(self) -> Dict[str, Any]:
        """Obtiene estadísticas de extracción REALES."""
        success_rate = (self.successful_extractions / self.transitions_detected * 100) if self.transitions_detected > 0 else 0
        
        return {
            'frames_processed': self.frames_processed,
            'transitions_detected': self.transitions_detected,
            'successful_extractions': self.successful_extractions,
            'success_rate_percent': round(success_rate, 2),
            'feature_dimension': 320,
            'sequence_length': self.sequence_length,
            'current_gesture': self.current_gesture,
            'transition_active': self.transition_active,
            'buffer_size': len(self.temporal_buffer),
            'detected_transitions_count': len(self.detected_transitions),
            'extractor_type': 'REAL - Sin simulación',
            'version': '2.0'
        }
    
    def reset_state(self):
        """Reinicia el estado del extractor."""
        self.temporal_buffer.clear()
        self.previous_frame = None
        self.current_gesture = "None"
        self.gesture_stable_count = 0
        self.transition_active = False
        self.transition_start_frame = 0
        self.frame_counter = 0
        self.detected_transitions.clear()
        logger.info("Estado del extractor dinámico reiniciado")
    
    def reset_stats(self):
        """Reinicia estadísticas."""
        self.frames_processed = 0
        self.transitions_detected = 0
        self.successful_extractions = 0
        logger.info("Estadísticas de extracción dinámica reiniciadas")


# ===== INSTANCIA GLOBAL =====
_real_dynamic_extractor_instance = None

def get_real_dynamic_features_extractor(sequence_length: int = 50) -> RealDynamicFeaturesExtractor:
    """Obtiene una instancia global del extractor de características dinámicas REAL."""
    global _real_dynamic_extractor_instance
    
    if _real_dynamic_extractor_instance is None:
        _real_dynamic_extractor_instance = RealDynamicFeaturesExtractor(sequence_length)
    
    return _real_dynamic_extractor_instance


# Alias para compatibilidad
DynamicFeaturesExtractor = RealDynamicFeaturesExtractor
get_dynamic_features_extractor = get_real_dynamic_features_extractor