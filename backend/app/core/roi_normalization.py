# =============================================================================
# MÓDULO 0: ROI NORMALIZATION SYSTEM
# Sistema de normalización por ROI para eliminar dependencia de distancia del usuario
# Detecta, extrae, valida y normaliza la región de la mano automáticamente
# =============================================================================

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

# Importar módulo de configuración desde el mismo paquete
try:
    from app.core.config_manager import get_config, get_logger, log_error, log_info
except ImportError:
    # Fallback si no está disponible
    def get_config(key, default=None): 
        return default
    def get_logger(): 
        return None
    def log_error(msg, exc=None): 
        print(f"ERROR: {msg}")
    def log_info(msg): 
        print(f"INFO: {msg}")

# Importar módulo de configuración centralizado
#from app.core.config_manager import get_config, get_logger, log_error, log_info

class ROIDistanceStatus(Enum):
    """Estados de distancia basados en resolución del ROI."""
    TOO_FAR = "too_far"           # ROI < 150px - Usuario muy lejos
    TOO_CLOSE = "too_close"       # ROI > 600px - Usuario muy cerca
    ACCEPTABLE = "acceptable"     # 150-600px - Distancia correcta
    UNKNOWN = "unknown"           # No detectado


@dataclass
class ROIExtractionResult:
    """Resultado completo de la extracción y validación de ROI."""
    # Frame procesado
    roi_frame: Optional[np.ndarray] = None          # Frame normalizado 224x224
    original_roi: Optional[np.ndarray] = None       # ROI original sin escalar
    
    # Bounding box
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2) en frame original
    
    # Métricas de tamaño
    roi_width: int = 0                              # Ancho del ROI en píxeles
    roi_height: int = 0                             # Alto del ROI en píxeles
    aspect_ratio: float = 0.0                       # Relación de aspecto
    
    # Estado de validación
    is_valid: bool = False                          # ¿Es válido para procesamiento?
    distance_status: ROIDistanceStatus = ROIDistanceStatus.UNKNOWN
    
    # Feedback para usuario
    feedback_message: str = ""                      # Mensaje descriptivo
    feedback_color: Tuple[int, int, int] = (0, 255, 255)  # Color BGR para UI
    
    # Metadata
    scaling_factor: float = 1.0                     # Factor de escala aplicado
    processing_time_ms: float = 0.0                 # Tiempo de procesamiento


class ROINormalizationSystem:
    """
    Sistema de normalización por ROI que garantiza:
    1. Detección automática de la mano en el frame
    2. Extracción de región de interés (ROI) con padding
    3. Validación de resolución mínima/máxima
    4. Escalado a tamaño estándar (224x224)
    5. Feedback visual en tiempo real
    
    BENEFICIO: El usuario mantiene distancia fija (~60cm) para TODOS los gestos
    """
    
    def __init__(self):
        """Inicializa el sistema de normalización ROI."""
        self.logger = get_logger()
        
        # ===== PARÁMETROS DE VALIDACIÓN =====
        # Resolución mínima del ROI (muy lejos)
        self.min_roi_width = get_config('roi_normalization.min_roi_width', 150)
        
        # Resolución máxima del ROI (muy cerca)
        self.max_roi_width = get_config('roi_normalization.max_roi_width', 600)
        
        # Tamaño normalizado de salida
        target_size_config = get_config('roi_normalization.target_size', [224, 224])
        self.target_size = tuple(target_size_config) if isinstance(target_size_config, list) else (224, 224)
        
        # Padding alrededor de landmarks (15% extra)
        self.roi_padding = get_config('roi_normalization.roi_padding', 0.15)
        
        # ===== PARÁMETROS DE MEJORA DE IMAGEN =====
        # Si ROI < 200px, aplicar sharpening
        self.sharpening_threshold = 200
        self.apply_sharpening = get_config('roi_normalization.apply_sharpening', True)
        
        # Ajuste de contraste automático
        self.apply_contrast_enhancement = get_config('roi_normalization.apply_contrast', False)
        
        # ===== ESTADÍSTICAS =====
        self.total_extractions = 0
        self.valid_extractions = 0
        self.too_far_count = 0
        self.too_close_count = 0
        self.total_processing_time = 0.0
        
        # ===== COLORES PARA FEEDBACK VISUAL =====
        self.colors = {
            ROIDistanceStatus.ACCEPTABLE: (0, 255, 0),     # Verde
            ROIDistanceStatus.TOO_FAR: (0, 255, 255),      # Amarillo
            ROIDistanceStatus.TOO_CLOSE: (0, 165, 255),    # Naranja
            ROIDistanceStatus.UNKNOWN: (128, 128, 128)     # Gris
        }
        
        log_info("=" * 70)
        log_info("ROI NORMALIZATION SYSTEM INICIALIZADO")
        log_info(f"  ✓ Resolución mínima: {self.min_roi_width}px")
        log_info(f"  ✓ Resolución máxima: {self.max_roi_width}px")
        log_info(f"  ✓ Tamaño normalizado: {self.target_size}")
        log_info(f"  ✓ Padding: {self.roi_padding * 100:.0f}%")
        log_info(f"  ✓ Sharpening: {'ACTIVADO' if self.apply_sharpening else 'DESACTIVADO'}")
        log_info("=" * 70)
    
    def extract_and_validate_roi(self, frame: np.ndarray, 
                                  hand_landmarks,
                                  gesture_name: str = "Unknown") -> ROIExtractionResult:
        """
        Extrae ROI de la mano y valida su calidad.
        
        Args:
            frame: Frame completo de cámara (1280x720)
            hand_landmarks: Landmarks de MediaPipe (21 puntos)
            gesture_name: Nombre del gesto actual (para logging)
            
        Returns:
            ROIExtractionResult con ROI normalizado o información de error
        """
        start_time = time.time()
        self.total_extractions += 1
        
        try:
            # ===== VALIDACIÓN INICIAL =====
            if frame is None or frame.size == 0:
                log_error("ROI: Frame inválido (None o vacío)")
                return self._create_error_result("Frame inválido", ROIDistanceStatus.UNKNOWN)
            
            if hand_landmarks is None:
                log_info("ROI: No hay landmarks (mano no detectada)")
                return self._create_error_result("No se detectó mano", ROIDistanceStatus.UNKNOWN)
            
            h, w = frame.shape[:2]
            log_info(f"ROI: Procesando frame {w}x{h} - Gesto: {gesture_name}")
            
            # ===== PASO 1: CALCULAR BOUNDING BOX =====
            log_info("ROI: PASO 1 - Calculando bounding box desde landmarks")
            
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            log_info(f"ROI: Bbox crudo - x:[{x_min}, {x_max}], y:[{y_min}, {y_max}]")
            
            # ===== PASO 2: AGREGAR PADDING =====
            log_info(f"ROI: PASO 2 - Agregando padding del {self.roi_padding * 100:.0f}%")
            
            roi_width_raw = x_max - x_min
            roi_height_raw = y_max - y_min
            
            padding_x = int(roi_width_raw * self.roi_padding)
            padding_y = int(roi_height_raw * self.roi_padding)
            
            x1 = max(0, x_min - padding_x)
            y1 = max(0, y_min - padding_y)
            x2 = min(w, x_max + padding_x)
            y2 = min(h, y_max + padding_y)
            
            roi_width = x2 - x1
            roi_height = y2 - y1
            aspect_ratio = roi_width / max(roi_height, 1)
            
            log_info(f"ROI: Bbox con padding - x:[{x1}, {x2}], y:[{y1}, {y2}]")
            log_info(f"ROI: Dimensiones finales - {roi_width}x{roi_height} (ratio: {aspect_ratio:.2f})")
            
            # ===== PASO 3: VALIDAR TAMAÑO DE ROI =====
            log_info(f"ROI: PASO 3 - Validando tamaño (rango: {self.min_roi_width}-{self.max_roi_width}px)")
            
            distance_status, feedback_msg, feedback_color = self._validate_roi_size(roi_width)
            
            if distance_status == ROIDistanceStatus.TOO_FAR:
                self.too_far_count += 1
                log_info(f"ROI: ❌ RECHAZO - Usuario muy lejos ({roi_width}px < {self.min_roi_width}px)")
                return self._create_result(
                    None, None, (x1, y1, x2, y2), roi_width, roi_height,
                    aspect_ratio, False, distance_status, feedback_msg,
                    feedback_color, 1.0, (time.time() - start_time) * 1000
                )
            
            if distance_status == ROIDistanceStatus.TOO_CLOSE:
                self.too_close_count += 1
                log_info(f"ROI: ❌ RECHAZO - Usuario muy cerca ({roi_width}px > {self.max_roi_width}px)")
                return self._create_result(
                    None, None, (x1, y1, x2, y2), roi_width, roi_height,
                    aspect_ratio, False, distance_status, feedback_msg,
                    feedback_color, 1.0, (time.time() - start_time) * 1000
                )
            
            log_info(f"ROI: ✅ Tamaño ACEPTABLE - {roi_width}px en rango válido")
            
            # ===== PASO 4: EXTRAER ROI =====
            log_info("ROI: PASO 4 - Extrayendo región de interés")
            
            roi_original = frame[y1:y2, x1:x2].copy()
            
            if roi_original.size == 0:
                log_error("ROI: ROI extraído está vacío")
                return self._create_error_result("ROI vacío", distance_status)
            
            log_info(f"ROI: Extraído exitosamente - Shape: {roi_original.shape}")
            
            # ===== PASO 5: ESCALAR A TAMAÑO ESTÁNDAR =====
            log_info(f"ROI: PASO 5 - Escalando a {self.target_size}")
            
            scaling_factor = self.target_size[0] / roi_width
            roi_normalized = cv2.resize(roi_original, self.target_size, interpolation=cv2.INTER_CUBIC)
            
            log_info(f"ROI: Escalado aplicado - Factor: {scaling_factor:.3f}x")
            
            # ===== PASO 6: MEJORAS DE IMAGEN (OPCIONAL) =====
            if self.apply_sharpening and roi_width < self.sharpening_threshold:
                log_info(f"ROI: PASO 6 - Aplicando sharpening (ROI pequeño: {roi_width}px)")
                roi_normalized = self._apply_sharpening(roi_normalized)
            
            if self.apply_contrast_enhancement:
                log_info("ROI: Aplicando mejora de contraste")
                roi_normalized = self._apply_contrast_enhancement(roi_normalized)
            
            # ===== RESULTADO EXITOSO =====
            self.valid_extractions += 1
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            
            log_info("=" * 70)
            log_info("ROI: ✅✅✅ EXTRACCIÓN EXITOSA ✅✅✅")
            log_info(f"ROI: Tiempo de procesamiento: {processing_time:.2f}ms")
            log_info(f"ROI: Estadísticas - Total: {self.total_extractions}, Válidos: {self.valid_extractions}")
            log_info(f"ROI: Success rate: {(self.valid_extractions/self.total_extractions)*100:.1f}%")
            log_info("=" * 70)
            
            return self._create_result(
                roi_normalized, roi_original, (x1, y1, x2, y2),
                roi_width, roi_height, aspect_ratio, True,
                distance_status, feedback_msg, feedback_color,
                scaling_factor, processing_time
            )
            
        except Exception as e:
            log_error(f"ROI: Error crítico en extracción: {e}", e)
            processing_time = (time.time() - start_time) * 1000
            return self._create_result(
                None, None, (0, 0, 0, 0), 0, 0, 0.0, False,
                ROIDistanceStatus.UNKNOWN, f"Error: {str(e)}",
                (0, 0, 255), 1.0, processing_time
            )
    
    def _validate_roi_size(self, roi_width: int) -> Tuple[ROIDistanceStatus, str, Tuple[int, int, int]]:
        """
        Valida el tamaño del ROI y genera feedback.
        
        Args:
            roi_width: Ancho del ROI en píxeles
            
        Returns:
            (estado, mensaje, color_bgr)
        """
        if roi_width < self.min_roi_width:
            deficit = self.min_roi_width - roi_width
            return (
                ROIDistanceStatus.TOO_FAR,
                f"⚠️ Acérquese {deficit}px más",
                self.colors[ROIDistanceStatus.TOO_FAR]
            )
        
        if roi_width > self.max_roi_width:
            excess = roi_width - self.max_roi_width
            return (
                ROIDistanceStatus.TOO_CLOSE,
                f"⚠️ Aléjese {excess}px",
                self.colors[ROIDistanceStatus.TOO_CLOSE]
            )
        
        return (
            ROIDistanceStatus.ACCEPTABLE,
            f"✅ Distancia perfecta ({roi_width}px)",
            self.colors[ROIDistanceStatus.ACCEPTABLE]
        )
    
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Aplica filtro de sharpening para mejorar detalles."""
        try:
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) / 9
            sharpened = cv2.filter2D(image, -1, kernel)
            log_info("ROI: Sharpening aplicado exitosamente")
            return sharpened
        except Exception as e:
            log_error(f"ROI: Error aplicando sharpening: {e}")
            return image
    
    def _apply_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Aplica mejora de contraste usando CLAHE."""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            log_info("ROI: Mejora de contraste aplicada")
            return enhanced
        except Exception as e:
            log_error(f"ROI: Error aplicando contraste: {e}")
            return image
    
    def draw_roi_indicator(self, frame: np.ndarray, roi_result: ROIExtractionResult):
        """
        Dibuja indicador visual del ROI en el frame original.
        
        Args:
            frame: Frame original donde dibujar
            roi_result: Resultado de extracción de ROI
        """
        try:
            if roi_result.bbox == (0, 0, 0, 0):
                return
            
            x1, y1, x2, y2 = roi_result.bbox
            color = roi_result.feedback_color
            
            # ===== RECTÁNGULO PRINCIPAL =====
            thickness = 3 if roi_result.is_valid else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # ===== ESQUINAS DECORATIVAS =====
            corner_size = 25
            corner_thickness = 4
            
            # Esquina superior izquierda
            cv2.line(frame, (x1, y1), (x1 + corner_size, y1), color, corner_thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_size), color, corner_thickness)
            
            # Esquina superior derecha
            cv2.line(frame, (x2, y1), (x2 - corner_size, y1), color, corner_thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_size), color, corner_thickness)
            
            # Esquina inferior izquierda
            cv2.line(frame, (x1, y2), (x1 + corner_size, y2), color, corner_thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_size), color, corner_thickness)
            
            # Esquina inferior derecha
            cv2.line(frame, (x2, y2), (x2 - corner_size, y2), color, corner_thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_size), color, corner_thickness)
            
            # ===== TEXTO INFORMATIVO =====
            # Fondo del texto
            text_y = y1 - 10 if y1 > 40 else y2 + 30
            cv2.rectangle(frame, (x1, text_y - 25), (x1 + 350, text_y + 5), (0, 0, 0), -1)
            
            # Mensaje de feedback
            cv2.putText(frame, roi_result.feedback_message, (x1 + 5, text_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Información técnica (abajo)
            info_text = f"ROI: {roi_result.roi_width}x{roi_result.roi_height}px | Escala: {roi_result.scaling_factor:.2f}x"
            cv2.rectangle(frame, (x1, y2 + 5), (x1 + 400, y2 + 35), (0, 0, 0), -1)
            cv2.putText(frame, info_text, (x1 + 5, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            log_error(f"ROI: Error dibujando indicador: {e}")
    
    def draw_statistics_panel(self, frame: np.ndarray):
        """Dibuja panel con estadísticas del sistema ROI."""
        try:
            h, w = frame.shape[:2]
            
            # Panel en esquina inferior derecha
            panel_w, panel_h = 280, 120
            panel_x, panel_y = w - panel_w - 10, h - panel_h - 10
            
            # Fondo
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                         (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Borde
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                         (0, 255, 255), 2)
            
            # Título
            cv2.putText(frame, "ROI STATS", (panel_x + 10, panel_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Estadísticas
            stats_y = panel_y + 40
            success_rate = (self.valid_extractions / max(self.total_extractions, 1)) * 100
            avg_time = self.total_processing_time / max(self.total_extractions, 1)
            
            cv2.putText(frame, f"Total: {self.total_extractions}", (panel_x + 10, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Validos: {self.valid_extractions} ({success_rate:.1f}%)", 
                       (panel_x + 10, stats_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, f"Lejos: {self.too_far_count} | Cerca: {self.too_close_count}", 
                       (panel_x + 10, stats_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f"Tiempo avg: {avg_time:.1f}ms", (panel_x + 10, stats_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        except Exception as e:
            log_error(f"ROI: Error dibujando panel de estadísticas: {e}")
    
    def _create_result(self, roi_frame, original_roi, bbox, roi_width, roi_height,
                       aspect_ratio, is_valid, distance_status, feedback_msg,
                       feedback_color, scaling_factor, processing_time) -> ROIExtractionResult:
        """Helper para crear resultado completo."""
        return ROIExtractionResult(
            roi_frame=roi_frame,
            original_roi=original_roi,
            bbox=bbox,
            roi_width=roi_width,
            roi_height=roi_height,
            aspect_ratio=aspect_ratio,
            is_valid=is_valid,
            distance_status=distance_status,
            feedback_message=feedback_msg,
            feedback_color=feedback_color,
            scaling_factor=scaling_factor,
            processing_time_ms=processing_time
        )
    
    def _create_error_result(self, message: str, status: ROIDistanceStatus) -> ROIExtractionResult:
        """Helper para crear resultado de error."""
        return ROIExtractionResult(
            is_valid=False,
            distance_status=status,
            feedback_message=message,
            feedback_color=self.colors[status]
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema."""
        success_rate = (self.valid_extractions / max(self.total_extractions, 1)) * 100
        avg_time = self.total_processing_time / max(self.total_extractions, 1)
        
        return {
            'total_extractions': self.total_extractions,
            'valid_extractions': self.valid_extractions,
            'success_rate_percent': success_rate,
            'too_far_count': self.too_far_count,
            'too_close_count': self.too_close_count,
            'avg_processing_time_ms': avg_time,
            'total_processing_time_ms': self.total_processing_time,
            'config': {
                'min_roi_width': self.min_roi_width,
                'max_roi_width': self.max_roi_width,
                'target_size': self.target_size,
                'roi_padding': self.roi_padding
            }
        }
    
    def reset_statistics(self):
        """Resetea las estadísticas del sistema."""
        self.total_extractions = 0
        self.valid_extractions = 0
        self.too_far_count = 0
        self.too_close_count = 0
        self.total_processing_time = 0.0
        log_info("ROI: Estadísticas reseteadas")


# ===== INSTANCIA GLOBAL =====
_roi_system_instance = None

def get_roi_normalization_system() -> ROINormalizationSystem:
    """
    Obtiene o crea la instancia global del sistema ROI.
    
    Returns:
        Instancia única de ROINormalizationSystem
    """
    global _roi_system_instance
    if _roi_system_instance is None:
        _roi_system_instance = ROINormalizationSystem()
    return _roi_system_instance

def reset_roi_system():
    """Resetea la instancia global del sistema ROI."""
    global _roi_system_instance
    if _roi_system_instance:
        _roi_system_instance.reset_statistics()