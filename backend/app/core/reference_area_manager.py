# =============================================================================
# MÓDULO 5: REFERENCE_AREA_MANAGER
# Sistema de áreas de referencia adaptativas y feedback visual
# =============================================================================

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Importar módulos anteriores
try:
    from app.core.config_manager import get_config, get_logger, log_error, log_info
    from app.core.quality_validator import DistanceStatus, HandSizeMetrics
except ImportError:
    def get_config(key, default=None): 
        return default
    def get_logger(): 
        return print
    def log_error(msg, exc=None): 
        logging.error(f"ERROR: {msg}")
    def log_info(msg): 
        logging.info(f"INFO: {msg}")
    
    class DistanceStatus(Enum):
        TOO_FAR = "muy_lejos"
        TOO_CLOSE = "muy_cerca"
        CORRECT = "correcta"

# Logger
logger = logging.getLogger(__name__)


class AreaType(Enum):
    """Tipos de área de referencia."""
    NARROW_HIGH = "narrow_high"
    WIDE_HIGH = "wide_high"
    MEDIUM_HIGH = "medium_high"
    STANDARD = "standard"


@dataclass
class AreaDimensions:
    """Dimensiones del área de referencia."""
    width_ratio: float
    height_ratio: float
    center_y_offset: float
    area_type: AreaType


@dataclass
class AreaCoordinates:
    """Coordenadas del área de referencia."""
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    width: int
    height: int


@dataclass
class VisualFeedback:
    """Configuración de feedback visual."""
    instruction_text: str
    text_offset: int
    area_color: Tuple[int, int, int]
    requirements_text: str

class ReferenceAreaManager:
    """
    Gestor de áreas de referencia adaptativas para diferentes gestos.
    Maneja dibujo, validación y feedback visual del área donde debe colocarse la mano.
    """

    def __init__(self):
        """Inicializa el gestor de áreas de referencia."""
        self.logger = get_logger()
        # Cargar configuraciones
        self.area_config = self._load_area_config()
        self.color_config = self._load_color_config()
        self.text_config = self._load_text_config()
        
        # Flag para usar área única (ROI normalization)
        self.use_single_area = get_config('roi_normalization.use_single_area', True)
        
        # Configuración de área única fija para TODOS los gestos
        self.single_area_config = {
            "width_ratio": 0.45,
            "height_ratio": 0.65,
            "center_y_offset": 0.5
        }
        
        # Configuraciones de dibujo
        self.corner_size = self.area_config.get('corner_size', 20)
        self.line_thickness = self.area_config.get('line_thickness', 3)
        
        # Cache para dimensiones calculadas
        self._dimensions_cache = {}
        
        logger.info(f"ReferenceAreaManager inicializado (Área única: {self.use_single_area})")
        if self.use_single_area:
            logger.info(f"  ✓ Área única: {self.single_area_config['width_ratio']*100:.0f}% × {self.single_area_config['height_ratio']*100:.0f}%")
    
    def _load_area_config(self) -> Dict[str, Any]:
        """Carga configuración de áreas de referencia."""
        default_areas = {
            "Pointing_Up": {"width_ratio": 0.4, "height_ratio": 0.8, "center_y_offset": 0.55},
            "Victory": {"width_ratio": 0.45, "height_ratio": 0.75, "center_y_offset": 0.52},
            "Thumb_Up": {"width_ratio": 0.4, "height_ratio": 0.7, "center_y_offset": 0.5},
            "Thumb_Down": {"width_ratio": 0.4, "height_ratio": 0.7, "center_y_offset": 0.5},
            "ILoveYou": {"width_ratio": 0.5, "height_ratio": 0.75, "center_y_offset": 0.5},
            "default": {"width_ratio": 0.45, "height_ratio": 0.6, "center_y_offset": 0.5}
        }
        
        return get_config('reference_area.gesture_areas', default_areas)
    
    def _load_color_config(self) -> Dict[str, Tuple[int, int, int]]:
        """Carga configuración de colores."""
        default_colors = {
            "area_outline": (0, 255, 255),
            "valid": (0, 255, 0),
            "invalid": (0, 0, 255),
            "text": (0, 255, 255),
            "feedback_correct": (0, 255, 0),
            "feedback_error": (0, 0, 255)
        }
        
        config_colors = get_config('reference_area.colors', {})
        
        for key, value in config_colors.items():
            if isinstance(value, list):
                config_colors[key] = tuple(value)
        
        default_colors.update(config_colors)
        return default_colors
    
    def _load_text_config(self) -> Dict[str, Any]:
        """Carga configuración de texto."""
        return {
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'thickness': 2,
            'scale': 0.7,
            'debug_scale': 0.5,
            'info_scale': 0.6
        }
    
    def get_gesture_requirements(self, gesture_name: str) -> str:
        """Devuelve los requisitos de área para cada gesto."""
        requirements = {
            "Pointing_Up": "Solo la base de la mano debe estar en el área",
            "Victory": "Solo la base de la mano debe estar en el área",
            "Thumb_Up": "Base de la mano (sin pulgar) en el área",
            "Thumb_Down": "Base de la mano (sin pulgar) en el área",
            "ILoveYou": "Centro de la mano en el área",
            "Open_Palm": "Toda la mano debe estar en el área",
            "Closed_Fist": "Toda la mano debe estar en el área"
        }
        return requirements.get(gesture_name, "Toda la mano debe estar en el área")
    
    def get_area_dimensions(self, gesture_name: str, frame_shape: Tuple[int, int]) -> AreaDimensions:
        """Calcula las dimensiones del área para un gesto específico."""
        height, width = frame_shape[:2]
        
        # Si se usa área única, ignorar el gesto y usar config fija
        if self.use_single_area:
            logger.debug(f"AREA: Usando área única fija (ignorando gesto: {gesture_name})")
            gesture_config = self.single_area_config
            area_type = AreaType.STANDARD
            
            dimensions = AreaDimensions(
                width_ratio=gesture_config["width_ratio"],
                height_ratio=gesture_config["height_ratio"],
                center_y_offset=gesture_config["center_y_offset"],
                area_type=area_type
            )
            
            area_width = int(width * gesture_config["width_ratio"])
            area_height = int(height * gesture_config["height_ratio"])
            logger.debug(f"AREA: Dimensiones únicas - {area_width}x{area_height}px")
            
            return dimensions
        
        # Lógica original (si NO se usa área única)
        cache_key = f"{gesture_name}_{width}_{height}"
        
        if cache_key in self._dimensions_cache:
            logger.debug(f"AREA: Usando cache para {gesture_name}")
            return self._dimensions_cache[cache_key]
        
        gesture_config = self.area_config.get(gesture_name, self.area_config["default"])
        
        if gesture_name == "Pointing_Up":
            area_type = AreaType.NARROW_HIGH
        elif gesture_name in ["Victory", "ILoveYou"]:
            area_type = AreaType.WIDE_HIGH
        elif gesture_name in ["Thumb_Up", "Thumb_Down"]:
            area_type = AreaType.MEDIUM_HIGH
        else:
            area_type = AreaType.STANDARD
        
        logger.debug(f"AREA: Configuración específica para {gesture_name} - tipo: {area_type.value}")
        
        dimensions = AreaDimensions(
            width_ratio=gesture_config["width_ratio"],
            height_ratio=gesture_config["height_ratio"],
            center_y_offset=gesture_config["center_y_offset"],
            area_type=area_type
        )
        
        self._dimensions_cache[cache_key] = dimensions
        
        return dimensions
    
    def calculate_area_coordinates(self, gesture_name: str, frame_shape: Tuple[int, int]) -> AreaCoordinates:
        """Calcula las coordenadas exactas del área de referencia."""
        height, width = frame_shape[:2]
        dimensions = self.get_area_dimensions(gesture_name, frame_shape)
        
        ref_width = int(width * dimensions.width_ratio)
        ref_height = int(height * dimensions.height_ratio)
        
        center_x = width // 2
        center_y = int(height * dimensions.center_y_offset)
        
        x1 = center_x - ref_width // 2
        y1 = center_y - ref_height // 2
        x2 = center_x + ref_width // 2
        y2 = center_y + ref_height // 2
        
        return AreaCoordinates(
            x1=x1, y1=y1, x2=x2, y2=y2,
            center_x=center_x, center_y=center_y,
            width=ref_width, height=ref_height
        )
    
    def draw_reference_area(self, frame: np.ndarray, current_gesture: str = "Open_Palm") -> Tuple[int, int, int, int]:
        """Dibuja el área de referencia donde debe colocarse la mano según el gesto."""
        try:
            coords = self.calculate_area_coordinates(current_gesture, frame.shape)
            color = self.color_config["area_outline"]
            
            cv2.rectangle(frame, (coords.x1, coords.y1), (coords.x2, coords.y2), color, 2)
            
            self._draw_area_corners(frame, coords, color)
            self._draw_instruction_text(frame, current_gesture, coords)
            
            return (coords.x1, coords.y1, coords.x2, coords.y2)
            
        except Exception as e:
            logger.error(f"Error dibujando área de referencia: {e}", exc_info=True)
            return (0, 0, 0, 0)
    
    def _draw_area_corners(self, frame: np.ndarray, coords: AreaCoordinates, color: Tuple[int, int, int]):
        """Dibuja las esquinas del área de referencia."""
        corner_size = self.corner_size
        thickness = self.line_thickness
        
        # Esquina superior izquierda
        cv2.line(frame, (coords.x1, coords.y1), (coords.x1 + corner_size, coords.y1), color, thickness)
        cv2.line(frame, (coords.x1, coords.y1), (coords.x1, coords.y1 + corner_size), color, thickness)
        
        # Esquina superior derecha
        cv2.line(frame, (coords.x2, coords.y1), (coords.x2 - corner_size, coords.y1), color, thickness)
        cv2.line(frame, (coords.x2, coords.y1), (coords.x2, coords.y1 + corner_size), color, thickness)
        
        # Esquina inferior izquierda
        cv2.line(frame, (coords.x1, coords.y2), (coords.x1 + corner_size, coords.y2), color, thickness)
        cv2.line(frame, (coords.x1, coords.y2), (coords.x1, coords.y2 - corner_size), color, thickness)
        
        # Esquina inferior derecha
        cv2.line(frame, (coords.x2, coords.y2), (coords.x2 - corner_size, coords.y2), color, thickness)
        cv2.line(frame, (coords.x2, coords.y2), (coords.x2, coords.y2 - corner_size), color, thickness)
    
    def _draw_instruction_text(self, frame: np.ndarray, gesture_name: str, coords: AreaCoordinates):
        """Dibuja el texto instructivo específico por gesto."""
        if gesture_name == "Pointing_Up":
            instruction = "COLOCA LA BASE - DEDO PUEDE SALIR"
            text_offset = 200
        elif gesture_name == "Victory":
            instruction = "COLOCA LA BASE - DEDOS PUEDEN SALIR"
            text_offset = 210
        elif gesture_name in ["Thumb_Up", "Thumb_Down"]:
            instruction = "COLOCA LA BASE - PULGAR PUEDE SALIR"
            text_offset = 210
        else:
            instruction = "COLOCA LA BASE DE TU MANO AQUI"
            text_offset = 180
        
        text_x = coords.center_x - text_offset
        text_y = coords.y1 - 15
        
        self.add_styled_text(frame, instruction, (text_x, text_y), 
                           self.text_config['scale'], self.color_config["text"])
    
    def draw_distance_feedback(self, frame: np.ndarray, distance_status, 
                             hand_size: float, target_size: float):
        """Dibuja feedback visual sobre la distancia de la mano."""
        try:
            height, width = frame.shape[:2]
            feedback_y = height - 150
            
            # Convertir enum a string si es necesario
            if hasattr(distance_status, 'value'):
                distance_str = distance_status.value
            else:
                distance_str = str(distance_status)
            
            if distance_str == "muy_lejos":
                color = self.color_config["feedback_error"]
                message = "ACERCA LA MANO"
                arrow = "↑"
            elif distance_str == "muy_cerca":
                color = self.color_config["feedback_error"]
                message = "ALEJA LA MANO"
                arrow = "↓"
            else:
                color = self.color_config["feedback_correct"]
                message = "DISTANCIA CORRECTA"
                arrow = "✓"
            
            self.add_styled_text(frame, f"{arrow} {message}", (20, feedback_y), 0.8, color)
            self._draw_distance_bar(frame, hand_size, target_size, feedback_y)
            
        except Exception as e:
            logger.error(f"Error dibujando feedback de distancia: {e}")
    
    def _draw_distance_bar(self, frame: np.ndarray, hand_size: float, 
                          target_size: float, feedback_y: int):
        """Dibuja la barra medidora de distancia."""
        bar_width = 200
        bar_height = 20
        bar_x = 20
        bar_y = feedback_y + 30
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        target_start = int(bar_width * 0.4)
        target_end = int(bar_width * 0.6)
        cv2.rectangle(frame, (bar_x + target_start, bar_y), 
                      (bar_x + target_end, bar_y + bar_height), self.color_config["valid"], -1)
        
        current_pos = min(max(int((hand_size / (target_size * 2)) * bar_width), 0), bar_width)
        
        if current_pos < target_start or current_pos > target_end:
            indicator_color = self.color_config["feedback_error"]
        else:
            indicator_color = self.color_config["feedback_correct"]
        
        cv2.circle(frame, (bar_x + current_pos, bar_y + bar_height // 2), 8, indicator_color, -1)
        
        self.add_styled_text(frame, "Lejos", (bar_x, bar_y + bar_height + 20), 
                           self.text_config['info_scale'], (255, 255, 255))
        self.add_styled_text(frame, "Cerca", (bar_x + bar_width - 30, bar_y + bar_height + 20), 
                           self.text_config['info_scale'], (255, 255, 255))
    
    def add_styled_text(self, img: np.ndarray, text: str, position: Tuple[int, int], 
                       size: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255)):
        """Añade texto estilizado al frame."""
        try:
            font = self.text_config['font']
            thickness = self.text_config['thickness']
            cv2.putText(img, text, position, font, size, color, thickness)
        except Exception as e:
            logger.error(f"Error añadiendo texto estilizado: {e}")
    
    def draw_area_validation_info(self, frame: np.ndarray, gesture_name: str, 
                                points_inside: int, total_points: int, 
                                hand_in_area: bool, position: Tuple[int, int]):
        """Dibuja información sobre la validación del área."""
        try:
            area_color = self.color_config["valid"] if hand_in_area else self.color_config["invalid"]
            
            status_text = f"En área: {points_inside}/{total_points} puntos"
            status_text += " ✓" if hand_in_area else " ✗"
            
            self.add_styled_text(frame, status_text, position, 0.8, area_color)
            
            requirements = self.get_gesture_requirements(gesture_name)
            req_position = (position[0], position[1] + 120)
            self.add_styled_text(frame, requirements, req_position, 
                               self.text_config['info_scale'], (200, 200, 200))
            
        except Exception as e:
            logger.error(f"Error dibujando información de validación: {e}")
    
    def draw_debug_info(self, frame: np.ndarray, gesture_name: str, 
                       hand_size: float, position: Tuple[int, int]):
        """Dibuja información de debug."""
        try:
            if gesture_name == "Pointing_Up":
                debug_msg = "Verificando: muñeca + base de dedos"
            elif gesture_name == "Victory":
                debug_msg = "Verificando: muñeca + base (sin índice/medio)"
            else:
                debug_msg = f"Modo: {gesture_name}"
            
            debug_position = (position[0], position[1] + 50)
            self.add_styled_text(frame, debug_msg, debug_position, 
                               self.text_config['debug_scale'], (150, 150, 150))
            
            size_position = (position[0], position[1] + 70)
            self.add_styled_text(frame, f"Tamaño mano: {hand_size:.3f}", size_position, 
                               self.text_config['info_scale'], (255, 255, 255))
            
        except Exception as e:
            logger.error(f"Error dibujando información de debug: {e}")
    
    def get_visual_feedback(self, gesture_name: str) -> VisualFeedback:
        """Obtiene la configuración de feedback visual para un gesto."""
        if self.use_single_area:
            logger.debug(f"AREA: Feedback genérico para área única (gesto: {gesture_name})")
            instruction = "COLOCA TU MANO DENTRO DEL AREA"
            text_offset = 180
            requirements_text = "Mano completa visible y centrada"
            
            return VisualFeedback(
                instruction_text=instruction,
                text_offset=text_offset,
                area_color=self.color_config["area_outline"],
                requirements_text=requirements_text
            )
        
        if gesture_name == "Pointing_Up":
            instruction = "COLOCA LA BASE - DEDO PUEDE SALIR"
            text_offset = 200
        elif gesture_name == "Victory":
            instruction = "COLOCA LA BASE - DEDOS PUEDEN SALIR"
            text_offset = 210
        elif gesture_name in ["Thumb_Up", "Thumb_Down"]:
            instruction = "COLOCA LA BASE - PULGAR PUEDE SALIR"
            text_offset = 210
        else:
            instruction = "COLOCA LA BASE DE TU MANO AQUI"
            text_offset = 180
        
        logger.debug(f"AREA: Feedback específico para {gesture_name}")
        
        return VisualFeedback(
            instruction_text=instruction,
            text_offset=text_offset,
            area_color=self.color_config["area_outline"],
            requirements_text=self.get_gesture_requirements(gesture_name)
        )
    
    def clear_cache(self):
        """Limpia el cache de dimensiones."""
        self._dimensions_cache.clear()
        logger.info("Cache de dimensiones limpiado")
    
    def get_area_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor de áreas."""
        return {
            'cached_dimensions': len(self._dimensions_cache),
            'available_gestures': len(self.area_config) - 3,  # -3 por 'default', 'corner_size', 'line_thickness'
            'corner_size': self.corner_size,
            'line_thickness': self.line_thickness,
            'color_configs': len(self.color_config),
            'text_configs': len(self.text_config),
            'use_single_area': self.use_single_area
        }


# ===== INSTANCIA GLOBAL =====
_area_manager_instance = None

def get_reference_area_manager() -> ReferenceAreaManager:
    """Obtiene una instancia global del gestor de áreas de referencia."""
    global _area_manager_instance
    
    if _area_manager_instance is None:
        _area_manager_instance = ReferenceAreaManager()
    
    return _area_manager_instance