# =============================================================================
# MÓDULO 4: QUALITY_VALIDATOR
# Sistema de validación de calidad para capturas de gestos
# =============================================================================

import numpy as np
import time
import logging
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

# Importar módulos anteriores
try:
    from app.core.config_manager import get_config, get_logger, log_error, log_info
    from app.core.mediapipe_processor import HandDetectionResult, ProcessingResult
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


class ValidationStatus(Enum):
    """Estados de validación."""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"


class DistanceStatus(Enum):
    """Estados de distancia de la mano."""
    TOO_FAR = "muy_lejos"
    TOO_CLOSE = "muy_cerca"
    CORRECT = "correcta"


@dataclass
class HandSizeMetrics:
    """Métricas de tamaño de la mano."""
    hand_size: float = 0.0
    main_length: float = 0.0
    hand_width: float = 0.0
    distance_status: DistanceStatus = DistanceStatus.TOO_FAR
    is_valid: bool = False


@dataclass
class MovementAnalysis:
    """Análisis de movimiento de la mano."""
    is_moving: bool = True
    movement_amount: float = 0.0
    max_movement: float = 0.0
    stable_frames: int = 0
    is_stable: bool = False
    stability_required: int = 1


@dataclass
class VisibilityAnalysis:
    """Análisis de visibilidad de puntos."""
    all_points_visible: bool = False
    points_outside_frame: int = 0
    total_points: int = 21
    visibility_percentage: float = 0.0
    margin_violations: List[int] = field(default_factory=list)


@dataclass
class AreaValidation:
    """Validación del área de referencia."""
    hand_in_area: bool = False
    points_inside: int = 0
    total_points_checked: int = 0
    core_points_inside: int = 0
    center_in_area: bool = False
    coverage_percentage: float = 0.0


@dataclass
class QualityAssessment:
    """Evaluación completa de calidad."""
    hand_size: HandSizeMetrics
    movement: MovementAnalysis
    visibility: VisibilityAnalysis
    area: AreaValidation
    overall_valid: bool = False
    confidence_valid: bool = False
    gesture_valid: bool = False
    ready_for_capture: bool = False
    quality_score: float = 0.0
    hand_confidence: float = 0.0
    gesture_confidence: float = 0.0
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
class QualityValidator:
    """
    Validador de calidad para capturas de gestos de manos.
    Implementa todas las validaciones del sistema original.
    """
    
    def __init__(self):
        """Inicializa el validador de calidad."""
        # Cargar configuraciones
        self.thresholds = self._load_thresholds()
        self.visibility_config = self._load_visibility_config()
        self.area_config = self._load_area_config()
        
        # Flag para normalización ROI
        self.use_roi_normalization = get_config('roi_normalization.enabled', True)
        
        # Control de movimiento y estabilidad
        self.landmark_history = deque(maxlen=5)
        self.stable_frame_count = 0
        
        # Estadísticas
        self.validations_performed = 0
        self.valid_captures = 0
        
        logger.info(f"QualityValidator inicializado (ROI normalization: {self.use_roi_normalization})")
    
    def _load_thresholds(self) -> Dict[str, float]:
        """Carga los umbrales de validación."""
        default_tolerance = 0.12 if get_config('roi_normalization.enabled', True) else 0.06
        
        return {
            'hand_confidence': get_config('thresholds.hand_confidence', 0.90),
            'gesture_confidence': get_config('thresholds.gesture_confidence', 0.60),
            'movement_threshold': get_config('thresholds.movement_threshold', 0.015),
            'target_hand_size': get_config('thresholds.target_hand_size', 0.22),
            'size_tolerance': get_config('thresholds.size_tolerance', default_tolerance),
            'required_stable_frames': get_config('capture.required_stable_frames', 1)
        }
    
    def _load_visibility_config(self) -> Dict[str, float]:
        """Carga configuración de visibilidad."""
        return {
            'margin': get_config('thresholds.visibility_margin', 0.05)
        }
    
    def _load_area_config(self) -> Dict[str, Any]:
        """Carga configuración de áreas de referencia."""
        return get_config('reference_area', {})
    
    def calculate_hand_size(self, hand_landmarks) -> HandSizeMetrics:
        """Calcula el tamaño de la mano basado en distancias entre puntos clave."""
        try:
            wrist = hand_landmarks.landmark[0]
            middle_tip = hand_landmarks.landmark[12]
            thumb_tip = hand_landmarks.landmark[4]
            pinky_tip = hand_landmarks.landmark[20]
            
            main_length = np.sqrt((wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2)
            hand_width = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)
            hand_size = (main_length * 0.7 + hand_width * 0.3)
            
            distance_status = self._check_hand_distance(hand_size)
            
            return HandSizeMetrics(
                hand_size=hand_size,
                main_length=main_length,
                hand_width=hand_width,
                distance_status=distance_status,
                is_valid=(distance_status == DistanceStatus.CORRECT)
            )
            
        except Exception as e:
            logger.error(f"Error calculando tamaño de mano: {e}")
            return HandSizeMetrics()
    
    def _check_hand_distance(self, hand_size: float) -> DistanceStatus:
        """Verifica si la mano está a la distancia correcta."""
        if self.use_roi_normalization:
            logger.debug(f"QUALITY: ROI normalization activo - distancia automáticamente CORRECT (hand_size={hand_size:.4f})")
            return DistanceStatus.CORRECT
        
        target_size = self.thresholds['target_hand_size']
        tolerance = self.thresholds['size_tolerance']
        
        min_size = target_size - tolerance
        max_size = target_size + tolerance
        
        if hand_size < min_size:
            logger.debug(f"QUALITY: Hand_size {hand_size:.4f} < min {min_size:.4f} - TOO_FAR")
            return DistanceStatus.TOO_FAR
        elif hand_size > max_size:
            logger.debug(f"QUALITY: Hand_size {hand_size:.4f} > max {max_size:.4f} - TOO_CLOSE")
            return DistanceStatus.TOO_CLOSE
        else:
            logger.debug(f"QUALITY: Hand_size {hand_size:.4f} en rango [{min_size:.4f}, {max_size:.4f}] - CORRECT")
            return DistanceStatus.CORRECT
    
    def detect_hand_movement(self, current_landmarks, 
                           previous_landmarks: Optional[Any] = None) -> MovementAnalysis:
        """Detecta si la mano está en movimiento."""
        try:
            if previous_landmarks is None:
                previous_landmarks = self.landmark_history[-1] if self.landmark_history else None
            
            if previous_landmarks is None:
                self.landmark_history.append(current_landmarks)
                return MovementAnalysis(
                    is_moving=True,
                    movement_amount=0.0,
                    max_movement=0.0,
                    stable_frames=0,
                    is_stable=False,
                    stability_required=int(self.thresholds['required_stable_frames'])
                )
            
            total_movement = 0
            max_movement = 0
            
            for i, (curr, prev) in enumerate(zip(current_landmarks.landmark, previous_landmarks.landmark)):
                movement = np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
                total_movement += movement
                max_movement = max(max_movement, movement)
            
            avg_movement = total_movement / len(current_landmarks.landmark)
            is_moving = avg_movement > self.thresholds['movement_threshold']
            
            if not is_moving:
                self.stable_frame_count += 1
            else:
                self.stable_frame_count = 0
            
            is_stable = self.stable_frame_count >= self.thresholds['required_stable_frames']
            
            self.landmark_history.append(current_landmarks)
            
            return MovementAnalysis(
                is_moving=is_moving,
                movement_amount=avg_movement,
                max_movement=max_movement,
                stable_frames=self.stable_frame_count,
                is_stable=is_stable,
                stability_required=int(self.thresholds['required_stable_frames'])
            )
            
        except Exception as e:
            logger.error(f"Error detectando movimiento: {e}")
            return MovementAnalysis()
    
    def check_visibility(self, hand_landmarks, frame_shape: Tuple[int, int]) -> VisibilityAnalysis:
        """Verifica que todos los puntos de la mano estén visibles en el frame."""
        try:
            margin = self.visibility_config['margin']
            points_outside = 0
            margin_violations = []
            total_points = len(hand_landmarks.landmark)
            
            for i, landmark in enumerate(hand_landmarks.landmark):
                if (landmark.x < margin or landmark.x > (1.0 - margin) or 
                    landmark.y < margin or landmark.y > (1.0 - margin)):
                    points_outside += 1
                    margin_violations.append(i)
            
            all_visible = points_outside == 0
            visibility_percentage = ((total_points - points_outside) / total_points) * 100
            
            return VisibilityAnalysis(
                all_points_visible=all_visible,
                points_outside_frame=points_outside,
                total_points=total_points,
                visibility_percentage=visibility_percentage,
                margin_violations=margin_violations
            )
            
        except Exception as e:
            logger.error(f"Error verificando visibilidad: {e}")
            return VisibilityAnalysis()
    
    def check_hand_in_reference_area(self, hand_landmarks, reference_area: Tuple[int, int, int, int],
                                   frame_shape: Tuple[int, int], current_gesture: str = "Open_Palm") -> AreaValidation:
        """Verifica si la mano está dentro del área de referencia según el tipo de gesto."""
        try:
            height, width = frame_shape[:2]
            x1, y1, x2, y2 = reference_area
            
            hand_points = []
            for landmark in hand_landmarks.landmark:
                x_pixel = int(landmark.x * width)
                y_pixel = int(landmark.y * height)
                hand_points.append((x_pixel, y_pixel))
            
            important_points, core_points, tolerance = self._get_gesture_validation_points(current_gesture)
            
            points_inside = 0
            for point_idx in important_points:
                x_pixel, y_pixel = hand_points[point_idx]
                if x1 <= x_pixel <= x2 and y1 <= y_pixel <= y2:
                    points_inside += 1
            
            core_points_inside = 0
            for point_idx in core_points:
                x_pixel, y_pixel = hand_points[point_idx]
                if x1 <= x_pixel <= x2 and y1 <= y_pixel <= y2:
                    core_points_inside += 1
            
            center_x = np.mean([hand_points[i][0] for i in core_points])
            center_y = np.mean([hand_points[i][1] for i in core_points])
            
            center_in_area = x1 <= center_x <= x2 and y1 <= center_y <= y2
            
            core_ok = core_points_inside >= len(core_points)
            important_ok = points_inside >= len(important_points) * tolerance
            
            hand_in_area = core_ok and important_ok and center_in_area
            coverage_percentage = (points_inside / len(important_points)) * 100
            
            return AreaValidation(
                hand_in_area=hand_in_area,
                points_inside=points_inside,
                total_points_checked=len(important_points),
                core_points_inside=core_points_inside,
                center_in_area=center_in_area,
                coverage_percentage=coverage_percentage
            )
            
        except Exception as e:
            logger.error(f"Error verificando área de referencia: {e}")
            return AreaValidation()
    
    def _get_gesture_validation_points(self, current_gesture: str) -> Tuple[List[int], List[int], float]:
        """Obtiene los puntos de validación según el gesto."""
        if current_gesture == "Pointing_Up":
            important_points = [0, 1, 5, 9, 13, 17]
            core_points = [0, 1, 5, 9]
            tolerance = 1.0
        elif current_gesture == "Victory":
            important_points = [0, 1, 5, 13, 17]
            core_points = [0, 1, 5, 13]
            tolerance = 1.0
        elif current_gesture in ["Thumb_Up", "Thumb_Down"]:
            important_points = [0, 5, 9, 13, 17]
            core_points = [0, 5, 9, 13]
            tolerance = 1.0
        elif current_gesture == "ILoveYou":
            important_points = [0, 9, 13]
            core_points = [0, 9, 13]
            tolerance = 1.0
        else:
            important_points = [0, 4, 8, 12, 16, 20]
            core_points = [0, 1, 5, 9, 13, 17]
            tolerance = 0.8
        
        return important_points, core_points, tolerance
    
    def check_hand_extension(self, hand_landmarks, gesture_name: str) -> bool:
        """Verifica si la mano está suficientemente extendida para el gesto."""
        try:
            requires_extension = gesture_name in ["Open_Palm"]
            
            if not requires_extension:
                return True
            
            wrist = hand_landmarks.landmark[0]
            middle_finger_tip = hand_landmarks.landmark[12]
            
            distance = np.sqrt(
                (wrist.x - middle_finger_tip.x)**2 + 
                (wrist.y - middle_finger_tip.y)**2
            )
            
            extension_threshold = 0.2
            
            return distance > extension_threshold
            
        except Exception as e:
            logger.error(f"Error verificando extensión de mano: {e}")
            return False
        
    def validate_complete_quality(self, hand_landmarks, handedness, 
                            detected_gesture: str, gesture_confidence: float,
                            target_gesture: str, reference_area: Tuple[int, int, int, int],
                            frame_shape: Tuple[int, int]) -> QualityAssessment:
        """Realiza una validación completa de calidad."""
        try:
            self.validations_performed += 1
            
            # 1. Calcular métricas de tamaño
            hand_size = self.calculate_hand_size(hand_landmarks)
            
            # 2. Analizar movimiento
            movement = self.detect_hand_movement(hand_landmarks)
            
            # 3. Verificar visibilidad
            visibility = self.check_visibility(hand_landmarks, frame_shape)
            
            # 4. Validar área de referencia
            area = self.check_hand_in_reference_area(hand_landmarks, reference_area, frame_shape, target_gesture)
            
            # 5. Verificar confianza de mano
            hand_confidence = handedness.classification[0].score
            confidence_valid = hand_confidence >= self.thresholds['hand_confidence']
            
            # 6. Verificar gesto
            if target_gesture == "Unknown":
                gesture_valid = (detected_gesture not in ["None", "Unknown", None] and 
                               gesture_confidence >= self.thresholds['gesture_confidence'])
            else:
                gesture_valid = (detected_gesture == target_gesture and 
                               gesture_confidence >= self.thresholds['gesture_confidence'])
            
            logger.debug(f"🎯 GESTO DEBUG: Detectado='{detected_gesture}', Esperado='{target_gesture}', Confianza={gesture_confidence:.3f}, Válido={gesture_valid}")
            
            # 7. Verificar extensión
            extension_valid = self.check_hand_extension(hand_landmarks, target_gesture)
            
            # 8. Evaluación global
            all_conditions = [
                confidence_valid,
                gesture_valid,
                visibility.all_points_visible,
                area.hand_in_area,
                hand_size.is_valid,
                not movement.is_moving,
                movement.is_stable,
                extension_valid
            ]
            
            overall_valid = all(all_conditions)
            ready_for_capture = overall_valid
            
            # 9. Calcular score de calidad
            quality_score = self._calculate_quality_score(
                hand_confidence, gesture_confidence, visibility, area, 
                hand_size, movement, extension_valid
            )
            
            # 10. Detalles de validación
            validation_details = {
                'hand_confidence': hand_confidence,
                'gesture_confidence': gesture_confidence,
                'detected_gesture': detected_gesture,
                'target_gesture': target_gesture,
                'extension_valid': extension_valid,
                'conditions_met': sum(all_conditions),
                'total_conditions': len(all_conditions),
                'thresholds': self.thresholds.copy()
            }
            
            if ready_for_capture:
                self.valid_captures += 1
            
            return QualityAssessment(
                hand_size=hand_size,
                movement=movement,
                visibility=visibility,
                area=area,
                overall_valid=overall_valid,
                confidence_valid=confidence_valid,
                gesture_valid=gesture_valid,
                ready_for_capture=ready_for_capture,
                quality_score=quality_score,
                hand_confidence=hand_confidence,
                gesture_confidence=gesture_confidence,
                validation_details=validation_details
            )
            
        except Exception as e:
            logger.error(f"Error en validación completa de calidad: {e}", exc_info=True)
            return QualityAssessment(
                hand_size=HandSizeMetrics(),
                movement=MovementAnalysis(),
                visibility=VisibilityAnalysis(),
                area=AreaValidation()
            )
    
    def _calculate_quality_score(self, hand_confidence: float, gesture_confidence: float,
                           visibility: VisibilityAnalysis, area: AreaValidation,
                           hand_size: HandSizeMetrics, movement: MovementAnalysis,
                           extension_valid: bool) -> float:
        """Calcula un score de calidad general (0-100)."""
        try:
            components = {
                'hand_confidence': hand_confidence * 25,
                'gesture_confidence': gesture_confidence * 20,
                'visibility': (visibility.visibility_percentage / 100) * 15,
                'area_coverage': (area.coverage_percentage / 100) * 15,
                'size_quality': (1.0 if hand_size.is_valid else 0.0) * 10,
                'stability': (1.0 if movement.is_stable else 0.0) * 10,
                'extension': (1.0 if extension_valid else 0.0) * 5
            }
            
            total_score = sum(components.values())
            
            logger.debug("=" * 70)
            logger.debug("QUALITY SCORE BREAKDOWN")
            logger.debug(f"hand_confidence:    {components['hand_confidence']:6.2f}/25")
            logger.debug(f"gesture_confidence: {components['gesture_confidence']:6.2f}/20")
            logger.debug(f"visibility:         {components['visibility']:6.2f}/15")
            logger.debug(f"area_coverage:      {components['area_coverage']:6.2f}/15")
            logger.debug(f"size_quality:       {components['size_quality']:6.2f}/10")
            logger.debug(f"stability:          {components['stability']:6.2f}/10")
            logger.debug(f"extension:          {components['extension']:6.2f}/5")
            logger.debug(f"TOTAL SCORE:        {total_score:6.2f}/100")
            logger.debug("=" * 70)
            
            return min(100.0, max(0.0, total_score))
            
        except Exception as e:
            logger.error(f"Error calculando score de calidad: {e}")
            return 0.0
    
    def get_validation_feedback(self, assessment: QualityAssessment) -> Dict[str, str]:
        """Genera feedback detallado sobre la validación."""
        feedback = {}
        
        if assessment.hand_size.distance_status == DistanceStatus.TOO_FAR:
            feedback['distance'] = "ACERCA LA MANO"
        elif assessment.hand_size.distance_status == DistanceStatus.TOO_CLOSE:
            feedback['distance'] = "ALEJA LA MANO"
        else:
            feedback['distance'] = "DISTANCIA CORRECTA"
        
        if assessment.movement.is_moving:
            feedback['movement'] = f"Mano en movimiento - Mantén quieta"
        elif not assessment.movement.is_stable:
            feedback['stability'] = f"Mano estable: {assessment.movement.stable_frames}/{assessment.movement.stability_required}"
        else:
            feedback['stability'] = "MANO ESTABLE"
        
        if not assessment.visibility.all_points_visible:
            feedback['visibility'] = f"Puntos fuera: {assessment.visibility.points_outside_frame} - Centra la mano"
        else:
            feedback['visibility'] = "TODOS LOS PUNTOS VISIBLES"
        
        if not assessment.area.hand_in_area:
            feedback['area'] = f"En área: {assessment.area.points_inside}/{assessment.area.total_points_checked} puntos"
        else:
            feedback['area'] = "POSICIÓN CORRECTA"
        
        if not assessment.confidence_valid:
            feedback['confidence'] = "Confianza de mano insuficiente"
        else:
            feedback['confidence'] = "Confianza de mano adecuada"
        
        if not assessment.gesture_valid:
            feedback['gesture'] = "Gesto no válido o confianza baja"
        else:
            feedback['gesture'] = "GESTO VÁLIDO"
        
        return feedback
    
    def reset_stability_counter(self):
        """Reinicia el contador de estabilidad."""
        self.stable_frame_count = 0
        self.landmark_history.clear()
        logger.info("Contador de estabilidad reiniciado")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de validación."""
        success_rate = (self.valid_captures / self.validations_performed * 100) if self.validations_performed > 0 else 0
        
        return {
            'validations_performed': self.validations_performed,
            'valid_captures': self.valid_captures,
            'success_rate_percent': round(success_rate, 2),
            'current_stable_frames': self.stable_frame_count,
            'landmark_history_size': len(self.landmark_history),
            'thresholds': self.thresholds.copy()
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Actualiza los umbrales de validación."""
        self.thresholds.update(new_thresholds)
        logger.info(f"Umbrales actualizados: {new_thresholds}")
    
    def reset_stats(self):
        """Reinicia todas las estadísticas."""
        self.validations_performed = 0
        self.valid_captures = 0
        self.stable_frame_count = 0
        self.landmark_history.clear()
        logger.info("Estadísticas de validación reiniciadas")


# ===== INSTANCIA GLOBAL =====
_validator_instance = None

def get_quality_validator() -> QualityValidator:
    """Obtiene una instancia global del validador de calidad."""
    global _validator_instance
    
    if _validator_instance is None:
        _validator_instance = QualityValidator()
    
    return _validator_instance