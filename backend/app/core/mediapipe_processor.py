# =============================================================================
# MÓDULO 3: MEDIAPIPE_PROCESSOR
# Wrapper para MediaPipe Hands y GestureRecognizer
# =============================================================================

import cv2
import mediapipe as mp
import numpy as np
import os
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Importar las clases necesarias para el reconocedor de gestos
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Importar config_manager
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

# Configurar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Logger
logger = logging.getLogger(__name__)


class HandSide(Enum):
    """Enumeración para lateralidad de la mano."""
    LEFT = "Izquierda"
    RIGHT = "Derecha"
    UNKNOWN = "Desconocida"


@dataclass
class HandDetectionResult:
    """Resultado de detección de mano."""
    landmarks: Optional[Any] = None
    world_landmarks: Optional[Any] = None
    handedness: Optional[Any] = None
    hand_side: HandSide = HandSide.UNKNOWN
    confidence: float = 0.0
    is_valid: bool = False


@dataclass
class GestureRecognitionResult:
    """Resultado de reconocimiento de gesto."""
    gesture_name: str = "None"
    confidence: float = 0.0
    is_valid: bool = False
    all_gestures: List[Dict] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Resultado completo del procesamiento."""
    hand_result: HandDetectionResult
    gesture_result: GestureRecognitionResult
    frame_processed: Optional[np.ndarray] = None
    processing_time: float = 0.0
    timestamp: float = 0.0
    
class MediaPipeProcessor:
    """
    Procesador MediaPipe para detección de manos y reconocimiento de gestos.
    Wrapper que encapsula MediaPipe Hands y GestureRecognizer.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el procesador MediaPipe.
        
        Args:
            model_path: Ruta al modelo gesture_recognizer.task
        """
        # Configuración desde config_manager
        self.hands_config = self._load_hands_config()
        self.gesture_config = self._load_gesture_config()
        
        # Estados del procesador
        self.hands = None
        self.gesture_recognizer = None
        self.is_initialized = False
        
        # Modelo path
        self.model_path = model_path or self._get_model_path()
        
        # Gestos disponibles
        self.available_gestures = get_config('available_gestures', [
            "None", "Closed_Fist", "Open_Palm", "Pointing_Up",
            "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"
        ])
        
        # Contadores y estadísticas
        self.frames_processed = 0
        self.hands_detected = 0
        self.gestures_recognized = 0
        
        logger.info("MediaPipeProcessor inicializado")
    
    def _load_hands_config(self) -> Dict[str, Any]:
        """Carga configuración para MediaPipe Hands."""
        return {
            'static_image_mode': get_config('mediapipe.hands.static_image_mode', False),
            'max_num_hands': get_config('mediapipe.hands.max_num_hands', 1),
            'model_complexity': get_config('mediapipe.hands.model_complexity', 1),
            'min_detection_confidence': get_config('mediapipe.hands.min_detection_confidence', 0.8),
            'min_tracking_confidence': get_config('mediapipe.hands.min_tracking_confidence', 0.8)
        }
    
    def _load_gesture_config(self) -> Dict[str, Any]:
        """Carga configuración para GestureRecognizer."""
        return {
            'num_hands': get_config('mediapipe.gesture_recognizer.num_hands', 1),
            'min_hand_detection_confidence': get_config('mediapipe.gesture_recognizer.min_hand_detection_confidence', 0.8),
            'min_hand_presence_confidence': get_config('mediapipe.gesture_recognizer.min_hand_presence_confidence', 0.8),
            'min_tracking_confidence': get_config('mediapipe.gesture_recognizer.min_tracking_confidence', 0.8)
        }
    
    def _get_model_path(self) -> str:
        """Obtiene la ruta del modelo desde config_manager."""
        try:
            from app.core.config_manager import get_config_manager
            config_mgr = get_config_manager()
            return config_mgr.get_model_path()
        except:
            models_dir = get_config('paths.models', 'biometric_data/models')
            model_file = get_config('paths.model_file', 'gesture_recognizer.task')
            return os.path.join(models_dir, model_file)
    
    def initialize(self) -> bool:
        """
        Inicializa MediaPipe Hands y GestureRecognizer.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            logger.info("Inicializando MediaPipe Hands y GestureRecognizer...")
            
            # Verificar que el modelo existe
            if not os.path.exists(self.model_path):
                logger.error(f"Modelo no encontrado: {self.model_path}")
                logger.error("Descarga el modelo desde: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task")
                return False
            
            # Inicializar MediaPipe Hands
            success_hands = self._initialize_hands()
            if not success_hands:
                logger.error("Error inicializando MediaPipe Hands")
                return False
            
            # Inicializar GestureRecognizer
            success_gesture = self._initialize_gesture_recognizer()
            if not success_gesture:
                logger.error("Error inicializando GestureRecognizer")
                return False
            
            self.is_initialized = True
            self._log_initialization_info()
            logger.info("MediaPipe inicializado correctamente")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en inicialización de MediaPipe: {e}", exc_info=True)
            return False
    
    def _initialize_hands(self) -> bool:
        """Inicializa MediaPipe Hands."""
        try:
            self.hands = mp_hands.Hands(
                static_image_mode=self.hands_config['static_image_mode'],
                max_num_hands=self.hands_config['max_num_hands'],
                model_complexity=self.hands_config['model_complexity'],
                min_detection_confidence=self.hands_config['min_detection_confidence'],
                min_tracking_confidence=self.hands_config['min_tracking_confidence']
            )
            
            logger.info("MediaPipe Hands inicializado")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando MediaPipe Hands: {e}", exc_info=True)
            return False
    
    def _initialize_gesture_recognizer(self) -> bool:
        """Inicializa MediaPipe GestureRecognizer."""
        try:
            # Leer el modelo
            with open(self.model_path, "rb") as f:
                model_content = f.read()
            
            # Configurar opciones base
            base_options = python.BaseOptions(model_asset_buffer=model_content)
            
            # Configurar opciones del reconocedor
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.gesture_config['num_hands'],
                min_hand_detection_confidence=self.gesture_config['min_hand_detection_confidence'],
                min_hand_presence_confidence=self.gesture_config['min_hand_presence_confidence'],
                min_tracking_confidence=self.gesture_config['min_tracking_confidence']
            )
            
            # Crear el reconocedor
            self.gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
            
            logger.info("GestureRecognizer inicializado")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando GestureRecognizer: {e}", exc_info=True)
            return False
    
    def _log_initialization_info(self):
        """Registra información de inicialización."""
        logger.info("=" * 70)
        logger.info("MEDIAPIPE PROCESSOR - CONFIGURACIÓN")
        logger.info(f"  ✓ Modelo: {self.model_path}")
        logger.info(f"  ✓ Hands - Confianza detección: {self.hands_config['min_detection_confidence']}")
        logger.info(f"  ✓ Hands - Confianza tracking: {self.hands_config['min_tracking_confidence']}")
        logger.info(f"  ✓ Gesture - Confianza: {self.gesture_config['min_hand_detection_confidence']}")
        logger.info(f"  ✓ Gestos disponibles: {len(self.available_gestures)}")
        logger.info("=" * 70)
    
    def process_frame(self, frame: np.ndarray, 
                     draw_landmarks: bool = False) -> ProcessingResult:
        """
        Procesa un frame completo con detección de manos y reconocimiento de gestos.
        
        Args:
            frame: Frame a procesar
            draw_landmarks: Si dibujar landmarks en el frame
            
        Returns:
            Resultado completo del procesamiento
        """
        import time
        start_time = time.time()
        
        if not self.is_initialized:
            logger.error("MediaPipe no inicializado")
            return ProcessingResult(
                hand_result=HandDetectionResult(),
                gesture_result=GestureRecognitionResult()
            )
        
        try:
            # Procesar con MediaPipe Hands
            hand_result = self._process_hand_detection(frame)
            
            # Procesar con GestureRecognizer
            gesture_result = self._process_gesture_recognition(frame)
            
            # Dibujar landmarks si se solicita
            processed_frame = frame.copy() if draw_landmarks else None
            if draw_landmarks and hand_result.is_valid:
                processed_frame = self._draw_landmarks(processed_frame, hand_result.landmarks)
            
            # Estadísticas
            self.frames_processed += 1
            if hand_result.is_valid:
                self.hands_detected += 1
            if gesture_result.is_valid:
                self.gestures_recognized += 1
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                hand_result=hand_result,
                gesture_result=gesture_result,
                frame_processed=processed_frame,
                processing_time=processing_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error procesando frame: {e}", exc_info=True)
            return ProcessingResult(
                hand_result=HandDetectionResult(),
                gesture_result=GestureRecognitionResult(),
                processing_time=time.time() - start_time
            )
    
    def _process_hand_detection(self, frame: np.ndarray) -> HandDetectionResult:
        """Procesa detección de manos con MediaPipe Hands."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0]
                
                detected_side = handedness.classification[0].label
                hand_side = self._correct_hand_side(detected_side)
                confidence = handedness.classification[0].score
                
                world_landmarks = None
                if hasattr(results, 'multi_hand_world_landmarks') and results.multi_hand_world_landmarks:
                    world_landmarks = results.multi_hand_world_landmarks[0]
                
                return HandDetectionResult(
                    landmarks=hand_landmarks,
                    world_landmarks=world_landmarks,
                    handedness=handedness,
                    hand_side=hand_side,
                    confidence=confidence,
                    is_valid=True
                )
            
            return HandDetectionResult()
            
        except Exception as e:
            logger.error(f"Error en detección de manos: {e}")
            return HandDetectionResult()
    
    def _process_gesture_recognition(self, frame: np.ndarray) -> GestureRecognitionResult:
        """Procesa reconocimiento de gestos con GestureRecognizer."""
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            results = self.gesture_recognizer.recognize(mp_image)
            
            if results.gestures and len(results.gestures) > 0:
                all_gestures = []
                
                for hand_gestures in results.gestures:
                    if hand_gestures and len(hand_gestures) > 0:
                        for gesture in hand_gestures:
                            all_gestures.append({
                                'name': gesture.category_name,
                                'confidence': gesture.score
                            })
                
                if all_gestures:
                    best_gesture = max(all_gestures, key=lambda x: x['confidence'])
                    
                    return GestureRecognitionResult(
                        gesture_name=best_gesture['name'],
                        confidence=best_gesture['confidence'],
                        is_valid=True,
                        all_gestures=all_gestures
                    )
            
            return GestureRecognitionResult()
            
        except Exception as e:
            logger.error(f"Error en reconocimiento de gestos: {e}")
            return GestureRecognitionResult()
    
    def _correct_hand_side(self, detected_side: str) -> HandSide:
        """Corrige la lateralidad de la mano (la cámara actúa como espejo)."""
        if detected_side == "Right":
            return HandSide.LEFT
        elif detected_side == "Left":
            return HandSide.RIGHT
        else:
            return HandSide.UNKNOWN
    
    def _draw_landmarks(self, frame: np.ndarray, hand_landmarks) -> np.ndarray:
        """Dibuja landmarks en el frame."""
        try:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            return frame
            
        except Exception as e:
            logger.error(f"Error dibujando landmarks: {e}")
            return frame
    
    def validate_gesture_match(self, detected_gesture: str, target_gesture: str,
                             confidence_threshold: Optional[float] = None) -> bool:
        """Valida si un gesto detectado coincide con el objetivo."""
        if confidence_threshold is None:
            confidence_threshold = get_config('thresholds.gesture_confidence', 0.60)
        
        return detected_gesture == target_gesture
    
    def validate_hand_confidence(self, confidence: float,
                               confidence_threshold: Optional[float] = None) -> bool:
        """Valida si la confianza de detección de mano es suficiente."""
        if confidence_threshold is None:
            confidence_threshold = get_config('thresholds.hand_confidence', 0.90)
        
        return confidence >= confidence_threshold
    
    def get_gesture_info(self, gesture_name: str) -> Dict[str, Any]:
        """Obtiene información detallada de un gesto."""
        try:
            from app.core.config_manager import get_config_manager
            config_mgr = get_config_manager()
            requirements = config_mgr.get_gesture_requirements(gesture_name)
        except:
            requirements = "Información no disponible"
        
        return {
            'name': gesture_name,
            'is_available': gesture_name in self.available_gestures,
            'requirements': requirements,
            'index': self.available_gestures.index(gesture_name) if gesture_name in self.available_gestures else -1
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de procesamiento."""
        hand_detection_rate = (self.hands_detected / self.frames_processed * 100) if self.frames_processed > 0 else 0
        gesture_recognition_rate = (self.gestures_recognized / self.frames_processed * 100) if self.frames_processed > 0 else 0
        
        return {
            'frames_processed': self.frames_processed,
            'hands_detected': self.hands_detected,
            'gestures_recognized': self.gestures_recognized,
            'hand_detection_rate_percent': round(hand_detection_rate, 2),
            'gesture_recognition_rate_percent': round(gesture_recognition_rate, 2),
            'is_initialized': self.is_initialized,
            'available_gestures_count': len(self.available_gestures),
            'model_path': self.model_path
        }
    
    def reset_stats(self):
        """Reinicia las estadísticas de procesamiento."""
        self.frames_processed = 0
        self.hands_detected = 0
        self.gestures_recognized = 0
        logger.info("Estadísticas de procesamiento reiniciadas")
    
    def close(self):
        """Cierra y libera recursos de MediaPipe."""
        try:
            if self.hands is not None:
                self.hands.close()
                logger.info("MediaPipe Hands cerrado")
            
            if self.gesture_recognizer is not None:
                self.gesture_recognizer.close()
                logger.info("GestureRecognizer cerrado")
            
            self.is_initialized = False
            
            stats = self.get_processing_stats()
            logger.info(f"Estadísticas finales - Frames: {stats['frames_processed']}, "
                       f"Manos: {stats['hands_detected']}, Gestos: {stats['gestures_recognized']}")
            
        except Exception as e:
            logger.error(f"Error cerrando MediaPipe: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor para asegurar liberación de recursos."""
        self.close()


# ===== INSTANCIA GLOBAL =====
_processor_instance = None

def get_mediapipe_processor(model_path: Optional[str] = None) -> Optional[MediaPipeProcessor]:
    """Obtiene una instancia global del procesador MediaPipe."""
    global _processor_instance
    
    if _processor_instance is None:
        _processor_instance = MediaPipeProcessor(model_path)
        if not _processor_instance.initialize():
            logger.error("ERROR: No se pudo inicializar MediaPipe")
            _processor_instance = None
            return None
    elif not _processor_instance.is_initialized:
        logger.info("Reinicializando MediaPipe existente...")
        if not _processor_instance.initialize():
            logger.error("ERROR: No se pudo reinicializar MediaPipe")
            _processor_instance = None
            return None
    
    return _processor_instance


def release_mediapipe():
    """Libera la instancia global del procesador."""
    global _processor_instance
    
    if _processor_instance is not None:
        _processor_instance.close()
        _processor_instance = None