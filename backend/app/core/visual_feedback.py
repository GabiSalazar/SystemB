# =============================================================================
# MÓDULO 0.5: VISUAL FEEDBACK MANAGER
# Gestiona el feedback visual en tiempo real para el sistema biométrico
# =============================================================================

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
import time
import logging

# Logger
logger = logging.getLogger(__name__)


class FeedbackLevel(Enum):
    """Niveles de feedback visual con colores específicos."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"
    BOOTSTRAP = "bootstrap"


@dataclass
class FeedbackMessage:
    """Mensaje de feedback visual con toda la información necesaria."""
    text: str
    level: FeedbackLevel
    priority: int
    icon: str = ""
    action: str = ""
    details: str = ""
    progress: float = 0.0


class VisualFeedbackManager:
    """Gestor de feedback visual en tiempo real para enrollment biométrico."""
    
    def __init__(self):
        # Configuración de colores BGR para OpenCV
        self.colors = {
            FeedbackLevel.SUCCESS: (0, 255, 0),      # Verde brillante
            FeedbackLevel.WARNING: (0, 255, 255),    # Amarillo
            FeedbackLevel.ERROR: (0, 0, 255),        # Rojo
            FeedbackLevel.INFO: (255, 200, 0),       # Azul claro
            FeedbackLevel.BOOTSTRAP: (255, 0, 255)   # Magenta (modo bootstrap)
        }
        
        # Iconos y símbolos
        self.icons = {
            "distance_far": "↔", "distance_close": "↔", "movement": "⚡",
            "stability": "⏱", "gesture": "✋", "confidence": "📊",
            "area": "📍", "success": "✅", "warning": "⚠", "error": "❌",
            "info": "ℹ", "bootstrap": "🔧", "progress": "📈"
        }
        
        logger.info("=" * 70)
        logger.info("VISUAL FEEDBACK MANAGER INICIALIZADO")
        logger.info("  ✓ 5 niveles de feedback configurados")
        logger.info("  ✓ Colores BGR definidos")
        logger.info("  ✓ Sistema de priorización activo")
        logger.info("=" * 70)
    
    def generate_real_time_feedback(self, quality_assessment=None, target_gesture: str = "", 
                              session_info: Dict = None, frame=None, session_state="",
                              current_gesture="", roi_result=None, processing_result=None,
                              progress_percentage=0.0, samples_captured=0, samples_needed=0,
                              bootstrap_mode=False) -> List[FeedbackMessage]:
        """
        Genera feedback en tiempo real basado en la evaluación de calidad.
        VERSIÓN ACTUALIZADA: Compatible con llamadas del enrollment con ROI.
        """
        try:
            messages = []
            
            # Normalizar parámetros (compatibilidad con diferentes firmas)
            if session_info is None:
                session_info = {
                    'bootstrap_mode': bootstrap_mode,
                    'samples_captured': samples_captured,
                    'samples_needed': samples_needed if samples_needed > 0 else 8
                }
            
            target_gesture = target_gesture or current_gesture
            
            # Obtener información de sesión
            bootstrap_mode = session_info.get('bootstrap_mode', bootstrap_mode)
            samples_captured = session_info.get('samples_captured', samples_captured)
            samples_needed = session_info.get('samples_needed', samples_needed if samples_needed > 0 else 8)
            
            logger.debug(f"Generando feedback - Gesto: {target_gesture}, Capturas: {samples_captured}/{samples_needed}")
            
            # 1. MENSAJE DE MODO BOOTSTRAP (si aplica)
            if bootstrap_mode:
                messages.append(FeedbackMessage(
                    "MODO BOOTSTRAP - Registro inicial",
                    FeedbackLevel.BOOTSTRAP, 0, "🔧", 
                    "Primeros usuarios", "Las redes se entrenarán después"
                ))
                logger.info("Feedback: Modo BOOTSTRAP activo")
            
            # 2. VERIFICAR SI HAY ASSESSMENT
            if not quality_assessment:
                messages.append(FeedbackMessage(
                    "Coloca tu mano frente a la cámara",
                    FeedbackLevel.INFO, 1, "✋", "Mostrar mano",
                    f"Gesto objetivo: {target_gesture}"
                ))
                logger.debug("Feedback: Sin assessment - esperando mano")
                return self._filter_and_sort_messages(messages)
            
            # 3. VALIDACIÓN CRÍTICA: Verificar confianza de mano
            if hasattr(quality_assessment, 'hand_confidence'):
                if quality_assessment.hand_confidence < 0.7:
                    messages.append(FeedbackMessage(
                        "No se detecta mano válida",
                        FeedbackLevel.WARNING, 1, "⚠️", "Mostrar mano claramente"
                    ))
                    logger.warning(f"Feedback: Confianza baja ({quality_assessment.hand_confidence:.2f})")
                    return self._filter_and_sort_messages(messages)
            
            # 4. FEEDBACK DE ÉXITO (máxima prioridad)
            if quality_assessment.ready_for_capture:
                messages.append(FeedbackMessage(
                    "¡PERFECTO! Capturando muestra...",
                    FeedbackLevel.SUCCESS, 0, "✅", "Mantener posición",
                    f"Calidad: {quality_assessment.quality_score:.0f}%"
                ))
                logger.info(f"Feedback: LISTO para captura - Calidad: {quality_assessment.quality_score:.0f}%")
                return self._filter_and_sort_messages(messages)
            
            # 5-9: Otros feedbacks...
            if roi_result:
                roi_msg = self._get_roi_feedback(roi_result)
                if roi_msg:
                    messages.append(roi_msg)
            
            distance_msg = self._get_distance_feedback(quality_assessment)
            if distance_msg:
                messages.append(distance_msg)
            
            gesture_msg = self._get_gesture_feedback(quality_assessment, target_gesture)
            if gesture_msg:
                messages.append(gesture_msg)
            
            movement_msg = self._get_movement_feedback(quality_assessment)
            if movement_msg:
                messages.append(movement_msg)
            
            progress_msg = self._get_progress_feedback(session_info, quality_assessment)
            if progress_msg:
                messages.append(progress_msg)
            
            return self._filter_and_sort_messages(messages)
            
        except Exception as e:
            logger.error(f"ERROR generando feedback: {e}", exc_info=True)
            return [FeedbackMessage("Error en feedback visual", FeedbackLevel.ERROR, 1, "❌")]
    
    # ... (resto de los métodos igual - _get_roi_feedback, _get_distance_feedback, etc.)
    # Mantén todo lo demás igual, solo cambiamos el logger
    
    def _get_roi_feedback(self, roi_result) -> Optional[FeedbackMessage]:
        """Genera feedback específico del ROI normalization."""
        if not hasattr(roi_result, 'is_valid'):
            return None
        
        if roi_result.is_valid:
            return FeedbackMessage(
                f"ROI válido: {roi_result.roi_width}x{roi_result.roi_height}px",
                FeedbackLevel.SUCCESS, 5, "✅", "Distancia correcta"
            )
        elif hasattr(roi_result, 'distance_status'):
            if roi_result.distance_status.value == "too_far":
                return FeedbackMessage(
                    "Acércate a la cámara", FeedbackLevel.WARNING, 1, "↔", "Acercar mano"
                )
            elif roi_result.distance_status.value == "too_close":
                return FeedbackMessage(
                    "Aléjate de la cámara", FeedbackLevel.WARNING, 1, "↔", "Alejar mano"
                )
        return None
    
    def _get_distance_feedback(self, assessment) -> Optional[FeedbackMessage]:
        if not hasattr(assessment, 'hand_size') or not assessment.hand_size:
            return None
        
        hand_size = assessment.hand_size
        
        if hand_size.distance_status == "muy_lejos":
            return FeedbackMessage(
                "Acerca más la mano a la cámara", FeedbackLevel.WARNING, 1, "↔", "Acercar mano"
            )
        elif hand_size.distance_status == "muy_cerca":
            return FeedbackMessage(
                "Aleja un poco la mano de la cámara", FeedbackLevel.WARNING, 1, "↔", "Alejar mano"
            )
        elif hand_size.distance_status == "correcta":
            return FeedbackMessage(
                "Distancia perfecta", FeedbackLevel.SUCCESS, 4, "✅", "Mantener distancia"
            )
        return None
    
    def _get_gesture_feedback(self, assessment, target_gesture: str) -> Optional[FeedbackMessage]:
        if not hasattr(assessment, 'gesture_valid'):
            return None
        
        if assessment.gesture_valid:
            return FeedbackMessage(
                f"Gesto {target_gesture} detectado", FeedbackLevel.SUCCESS, 2, "✋", "Mantener gesto"
            )
        else:
            return FeedbackMessage(
                f"Haz el gesto: {target_gesture}", FeedbackLevel.ERROR, 1, "✋", f"Hacer {target_gesture}"
            )
    
    def _get_movement_feedback(self, assessment) -> Optional[FeedbackMessage]:
        if not hasattr(assessment, 'movement') or not assessment.movement:
            return None
        
        movement = assessment.movement
        
        if movement.is_moving:
            return FeedbackMessage(
                "Mantén la mano quieta", FeedbackLevel.WARNING, 2, "⚡", "No mover"
            )
        elif not movement.is_stable:
            frames_needed = movement.stability_required - movement.stable_frames
            return FeedbackMessage(
                f"Estabilizando... {frames_needed} frames más", FeedbackLevel.INFO, 3, "⏱", "Mantener quieta"
            )
        else:
            return FeedbackMessage(
                "Mano estable", FeedbackLevel.SUCCESS, 5, "✅", "Continuar"
            )
    
    def _get_progress_feedback(self, session_info: Dict, assessment) -> Optional[FeedbackMessage]:
        try:
            samples_captured = session_info.get('samples_captured', 0)
            samples_needed = session_info.get('samples_needed', 8)
            bootstrap_mode = session_info.get('bootstrap_mode', False)
            
            if samples_needed <= 0:
                samples_needed = 8
            
            if samples_captured > 0:
                progress = (samples_captured / max(samples_needed, 1)) * 100
                bootstrap_text = " (Bootstrap)" if bootstrap_mode else ""
                return FeedbackMessage(
                    f"Progreso: {samples_captured}/{samples_needed} ({progress:.0f}%){bootstrap_text}",
                    FeedbackLevel.INFO, 6, "📈", "Continuar", "", progress
                )
            
            mode_text = "Bootstrap - " if bootstrap_mode else ""
            return FeedbackMessage(
                f"{mode_text}Iniciando captura", FeedbackLevel.INFO, 6, "📝", "Preparar gesto"
            )
            
        except Exception as e:
            logger.error(f"Error generando feedback de progreso: {e}")
            return FeedbackMessage(
                "Error en progreso", FeedbackLevel.ERROR, 6, "❌", "Reintentar"
            )
        
    def _filter_and_sort_messages(self, messages: List[FeedbackMessage]) -> List[FeedbackMessage]:
        filtered = sorted(messages, key=lambda m: m.priority)[:4]
        logger.debug(f"Mensajes filtrados: {len(filtered)}/{len(messages)}")
        return filtered
    
    def draw_feedback_overlay(self, frame: np.ndarray, messages: List[FeedbackMessage] = None, 
                            quality_assessment=None) -> np.ndarray:
        try:
            if frame is None:
                return frame
            
            h, w = frame.shape[:2]
            overlay_frame = frame.copy()
            
            if messages:
                self._draw_feedback_panel(overlay_frame, messages, h, w)
            
            if quality_assessment:
                self._draw_quality_indicator(overlay_frame, quality_assessment, h, w)
            
            if quality_assessment:
                self._draw_hand_indicators(overlay_frame, quality_assessment, h, w)
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"ERROR dibujando feedback overlay: {e}", exc_info=True)
            return frame
    
    def _draw_feedback_panel(self, frame: np.ndarray, messages: List[FeedbackMessage], h: int, w: int):
        if not messages:
            return
        
        panel_height = min(200, len(messages) * 40 + 50)
        panel_width = min(w - 40, 500)
        panel_x = 20
        panel_y = 20
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        border_color = self.colors[messages[0].level] if messages else (100, 100, 100)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), border_color, 3)
        
        cv2.putText(frame, "FEEDBACK EN TIEMPO REAL", (panel_x + 15, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset = panel_y + 50
        for message in messages:
            if y_offset + 30 > panel_y + panel_height:
                break
            
            color = self.colors[message.level]
            cv2.circle(frame, (panel_x + 20, y_offset + 10), 6, color, -1)
            cv2.putText(frame, message.text, (panel_x + 35, y_offset + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if message.action:
                cv2.putText(frame, f"-> {message.action}", (panel_x + 35, y_offset + 28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            y_offset += 35
    
    def _draw_quality_indicator(self, frame: np.ndarray, assessment, h: int, w: int):
        indicator_w = 120
        indicator_h = 80
        indicator_x = w - indicator_w - 20
        indicator_y = 20
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (indicator_x, indicator_y), 
                     (indicator_x + indicator_w, indicator_y + indicator_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        if assessment.ready_for_capture:
            border_color = self.colors[FeedbackLevel.SUCCESS]
            status_text = "LISTO"
        elif assessment.quality_score > 60:
            border_color = self.colors[FeedbackLevel.WARNING]
            status_text = "AJUSTAR"
        else:
            border_color = self.colors[FeedbackLevel.ERROR]
            status_text = "MEJORAR"
        
        cv2.rectangle(frame, (indicator_x, indicator_y), 
                     (indicator_x + indicator_w, indicator_y + indicator_h), border_color, 3)
        
        cv2.putText(frame, "CALIDAD", (indicator_x + 10, indicator_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"{assessment.quality_score:.0f}%", (indicator_x + 20, indicator_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)
        cv2.putText(frame, status_text, (indicator_x + 10, indicator_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)
    
    def _draw_hand_indicators(self, frame: np.ndarray, assessment, h: int, w: int):
        try:
            if not assessment:
                return
            
            if not hasattr(assessment, 'hand_confidence'):
                return
            
            if assessment.hand_confidence < 0.8:
                return
            
            if not hasattr(assessment, 'hand_size') or not assessment.hand_size:
                return
            
            center_x, center_y = w // 2, h // 2
            hand_size = assessment.hand_size
            
            if hand_size.distance_status == "muy_cerca":
                cv2.arrowedLine(frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 0, 255), 5)
                cv2.putText(frame, "ALEJAR", (center_x - 40, center_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            elif hand_size.distance_status == "muy_lejos":
                cv2.arrowedLine(frame, (center_x, center_y + 30), (center_x, center_y - 30), (0, 255, 255), 5)
                cv2.putText(frame, "ACERCAR", (center_x - 50, center_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            elif hand_size.distance_status == "correcta":
                if hasattr(assessment, 'ready_for_capture') and assessment.ready_for_capture:
                    cv2.circle(frame, (center_x, center_y), 25, (0, 255, 0), 3)
                    cv2.putText(frame, "LISTO", (center_x - 40, center_y + 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        except Exception as e:
            pass


# ===== INSTANCIA GLOBAL =====
_visual_feedback_instance = None

def get_visual_feedback_manager() -> VisualFeedbackManager:
    """Obtiene o crea la instancia global del Visual Feedback Manager."""
    global _visual_feedback_instance
    if _visual_feedback_instance is None:
        _visual_feedback_instance = VisualFeedbackManager()
    return _visual_feedback_instance