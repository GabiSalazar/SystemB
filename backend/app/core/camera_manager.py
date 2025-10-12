# =============================================================================
# MÓDULO 2: CAMERA_MANAGER
# Gestión de cámara, captura de frames y mejora de imagen
# =============================================================================

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

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

# Logger
logger = logging.getLogger(__name__)


class CameraManager:
    """
    Gestor de cámara para captura de alta calidad de gestos de manos.
    Maneja inicialización, configuración, captura y mejora de imagen.
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Inicializa el gestor de cámara.
        
        Args:
            camera_index: Índice de la cámara (0 para cámara por defecto)
        """
        self.camera_index = camera_index
        self.camera = None
        self.is_initialized = False
        self.frame_count = 0
        self.last_frame_time = 0
        
        # Obtener configuraciones desde config_manager
        self.config = self._load_camera_config()
        
        logger.info("CameraManager inicializado")
    
    def _load_camera_config(self) -> Dict[str, Any]:
        """Carga la configuración de cámara desde config_manager."""
        return {
            'width': get_config('camera.width', 1280),
            'height': get_config('camera.height', 720),
            'autofocus': get_config('camera.autofocus', True),
            'brightness': get_config('camera.brightness', 200),
            'contrast': get_config('camera.contrast', 200),
            'jpeg_quality': get_config('camera.jpeg_quality', 95),
            'fps_target': get_config('camera.fps_target', 30)
        }
    
    def initialize(self) -> bool:
        """
        Inicializa la cámara con configuraciones optimizadas.
        
        Returns:
            True si la inicialización fue exitosa, False en caso contrario
        """
        try:
            logger.info(f"Inicializando cámara {self.camera_index}...")
            
            # Crear objeto de captura
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                logger.error("ERROR: No se pudo abrir la cámara")
                return False
            
            # Configurar propiedades de la cámara
            success = self._configure_camera()
            
            if success:
                self.is_initialized = True
                self._log_camera_info()
                logger.info("Cámara inicializada correctamente")
                
                # Periodo de calentamiento
                self._warmup_camera()
                
            return success
            
        except Exception as e:
            logger.error(f"Error al inicializar cámara: {e}", exc_info=True)
            return False
    
    def _configure_camera(self) -> bool:
        """
        Configura todos los parámetros de la cámara.
        
        Returns:
            True si la configuración fue exitosa
        """
        try:
            # Configurar resolución
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            
            # Configurar autofocus
            if self.config['autofocus']:
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # Configurar brillo y contraste
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.config['brightness'])
            self.camera.set(cv2.CAP_PROP_CONTRAST, self.config['contrast'])
            
            # Auto Exposure
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.camera.set(cv2.CAP_PROP_GAIN, 250)
            self.camera.set(cv2.CAP_PROP_GAMMA, 300)
            self.camera.set(cv2.CAP_PROP_SATURATION, 150)
            
            # Configurar buffer para reducir latencia
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verificar que las configuraciones se aplicaron
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width != self.config['width'] or actual_height != self.config['height']:
                logger.warning(f"Resolución configurada {self.config['width']}x{self.config['height']}, "
                             f"actual {actual_width}x{actual_height}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error configurando cámara: {e}", exc_info=True)
            return False
    
    def _warmup_camera(self, warmup_frames: int = 30):
        """
        Periodo de calentamiento para estabilizar la cámara.
        
        Args:
            warmup_frames: Número de frames de calentamiento
        """
        logger.info(f"Calentando cámara ({warmup_frames} frames)...")
        
        for i in range(warmup_frames):
            ret, _ = self.camera.read()
            if not ret:
                logger.error("Error durante calentamiento de cámara")
                break
            time.sleep(0.033)  # ~30 FPS
        
        logger.info("Calentamiento de cámara completado")
    
    def _log_camera_info(self):
        """Registra información detallada de la cámara."""
        if not self.camera:
            return
        
        try:
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            brightness = self.camera.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast = self.camera.get(cv2.CAP_PROP_CONTRAST)
            
            logger.info("=" * 70)
            logger.info("CAMERA MANAGER - INFORMACIÓN DE CÁMARA")
            logger.info(f"  ✓ Resolución: {width}x{height}")
            logger.info(f"  ✓ FPS: {fps}")
            logger.info(f"  ✓ Brillo: {brightness}")
            logger.info(f"  ✓ Contraste: {contrast}")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Error obteniendo información de cámara: {e}")
    
    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Captura un frame de la cámara.
        
        Returns:
            Tupla (éxito, frame) donde éxito indica si la captura fue exitosa
        """
        if not self.is_initialized or not self.camera:
            logger.error("Cámara no inicializada")
            return False, None
        
        try:
            ret, frame = self.camera.read()
            
            if ret:
                self.frame_count += 1
                self.last_frame_time = time.time()
                
                # Log cada 100 frames para monitoreo
                if self.frame_count % 100 == 0:
                    logger.debug(f"Frame #{self.frame_count} capturado")
                
                return ret, frame
            else:
                logger.error("Error capturando frame de cámara - intentando recovery...")
                
                # Verificar si necesitamos recovery
                if not self.check_camera_health():
                    logger.info("Cámara corrupta detectada - ejecutando reset...")
                    
                    # Intentar recovery
                    if self.reset_camera():
                        logger.info("Recovery exitoso - reintentando captura...")
                        ret_retry, frame_retry = self.camera.read()
                        if ret_retry:
                            self.frame_count += 1
                            self.last_frame_time = time.time()
                            logger.info("✅ Captura exitosa después de recovery")
                            return ret_retry, frame_retry
                        else:
                            logger.error("❌ Recovery falló - captura sigue fallando")
                    else:
                        logger.error("❌ Reset de cámara falló")
                        self.is_initialized = False
                
                return False, None
                
        except Exception as e:
            logger.error(f"Excepción durante captura de frame: {e}", exc_info=True)
            
            # Recovery en excepciones
            logger.info("Intentando recovery por excepción...")
            if self.reset_camera():
                logger.info("Recovery post-excepción exitoso")
                try:
                    ret_retry, frame_retry = self.camera.read()
                    if ret_retry:
                        self.frame_count += 1
                        self.last_frame_time = time.time()
                        return ret_retry, frame_retry
                except:
                    pass
            
            return False, None
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Mejora la nitidez y calidad de la imagen.
        
        Args:
            image: Imagen de entrada
            
        Returns:
            Imagen mejorada
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar mejora de nitidez
            blur = cv2.GaussianBlur(gray, (0, 0), 3)
            sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
            
            # Convertir de vuelta a color
            sharp_color = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
            
            # Mezclar con la imagen original
            enhanced = cv2.addWeighted(image, 0.7, sharp_color, 0.3, 0)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error en mejora de imagen: {e}")
            return image
    
    def capture_high_quality_frame(self, stabilization_delay: float = 0.5) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Captura un frame de alta calidad con estabilización.
        
        Args:
            stabilization_delay: Tiempo de espera para estabilización
            
        Returns:
            Tupla (éxito, frame_mejorado)
        """
        logger.info("Capturando frame de alta calidad...")
        
        # Pausa para estabilización
        time.sleep(stabilization_delay)
        
        # Capturar múltiples frames y seleccionar el mejor
        frames = []
        scores = []
        
        for i in range(3):
            ret, frame = self.capture_frame()
            if ret and frame is not None:
                # Calcular score de calidad (varianza de Laplaciano)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                frames.append(frame)
                scores.append(score)
            
            time.sleep(0.1)
        
        if not frames:
            logger.error("No se pudieron capturar frames de alta calidad")
            return False, None
        
        # Seleccionar el frame con mejor score
        best_idx = np.argmax(scores)
        best_frame = frames[best_idx]
        
        logger.info(f"Frame de alta calidad seleccionado (score: {scores[best_idx]:.2f})")
        
        # Aplicar mejora
        enhanced_frame = self.enhance_image(best_frame)
        
        return True, enhanced_frame
    
    def save_frame(self, frame: np.ndarray, filepath: str, 
                   quality: Optional[int] = None) -> bool:
        """
        Guarda un frame en disco.
        
        Args:
            frame: Frame a guardar
            filepath: Ruta del archivo
            quality: Calidad JPEG (1-100)
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            if quality is None:
                quality = self.config['jpeg_quality']
            
            compression_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            
            success = cv2.imwrite(filepath, frame, compression_params)
            
            if success:
                logger.info(f"Frame guardado: {filepath} (calidad: {quality})")
            else:
                logger.error(f"Error guardando frame: {filepath}")
            
            return success
            
        except Exception as e:
            logger.error(f"Excepción guardando frame: {filepath} - {e}")
            return False
    
    def get_camera_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la cámara.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.camera:
            return {}
        
        try:
            return {
                'frame_count': self.frame_count,
                'last_frame_time': self.last_frame_time,
                'is_initialized': self.is_initialized,
                'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.camera.get(cv2.CAP_PROP_FPS),
                'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {'error': str(e)}
    
    def check_camera_health(self) -> bool:
        """Verifica el estado de salud de la cámara."""
        if not self.is_initialized or not self.camera:
            return False
        
        try:
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                logger.error("Health check falló: no se pudo capturar frame")
                return False
            
            if frame.size == 0:
                logger.error("Health check falló: frame vacío")
                return False
            
            h, w = frame.shape[:2]
            expected_w, expected_h = self.config['width'], self.config['height']
            
            if abs(w - expected_w) > 50 or abs(h - expected_h) > 50:
                logger.warning(f"Health check: dimensiones inesperadas {w}x{h}")
                return False
            
            logger.debug("Health check de cámara: ✓ OK")
            return True
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return False
    
    def reset_camera(self) -> bool:
        """Reinicia la cámara en caso de problemas."""
        logger.info("Reiniciando cámara...")
        
        try:
            self.release()
            time.sleep(1)
            return self.initialize()
            
        except Exception as e:
            logger.error(f"Error reiniciando cámara: {e}")
            return False
    
    def release(self):
        """Libera los recursos de la cámara."""
        try:
            if self.camera is not None:
                self.camera.release()
                logger.info(f"Cámara liberada. Total frames capturados: {self.frame_count}")
            
            self.camera = None
            self.is_initialized = False
            self.frame_count = 0
            
        except Exception as e:
            logger.error(f"Error liberando cámara: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor para asegurar liberación de recursos."""
        self.release()


# ===== INSTANCIA GLOBAL =====
_camera_instance = None
_retry_count = 0

def get_camera_manager(camera_index: int = 0) -> Optional[CameraManager]:
    """Obtiene o crea la instancia global de CameraManager."""
    global _camera_instance, _retry_count
    
    if _camera_instance is None:
        _camera_instance = CameraManager(camera_index)
        if not _camera_instance.initialize():
            logger.error("ERROR: No se pudo inicializar cámara")
            _camera_instance = None
            return None
    elif not _camera_instance.is_initialized:
        logger.info("Reinicializando cámara existente...")
        if not _camera_instance.initialize():
            logger.error("ERROR: No se pudo reinicializar cámara")
            _camera_instance = None
            
            # Evitar recursión infinita
            _retry_count += 1
            if _retry_count < 3:
                logger.info(f"Reintento {_retry_count}/3...")
                return get_camera_manager(camera_index)
            else:
                logger.error("FATAL: Máximo de reintentos alcanzado")
                _retry_count = 0
                return None
    
    _retry_count = 0
    return _camera_instance


def release_camera():
    """Libera la instancia global de cámara."""
    global _camera_instance
    
    if _camera_instance is not None:
        _camera_instance.release()
        _camera_instance = None


def reset_camera_for_new_operation():
    """Reset simple para nueva operación."""
    global _camera_instance
    
    if _camera_instance is not None:
        _camera_instance.release()
        _camera_instance = None
    
    cv2.destroyAllWindows()
    cv2.waitKey(50)