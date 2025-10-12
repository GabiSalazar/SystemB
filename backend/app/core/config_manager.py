# =============================================================================
# MÓDULO 1: CONFIG_MANAGER
# Gestión centralizada de configuración, logging y constantes del sistema
# =============================================================================

import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class ConfigManager:
    """
    Gestor centralizado de configuración para el sistema biométrico de gestos.
    Maneja logging, rutas, constantes y configuraciones del sistema.
    """
    
    def __init__(self, config_file: str = "biometric_config.json"):
        """
        Inicializa el gestor de configuración.
        
        Args:
            config_file: Archivo de configuración JSON
        """
        self.config_file = config_file
        self.logger = None
        
        # Cargar configuración por defecto
        self._config = self._load_default_config()
        
        # Intentar cargar archivo de configuración
        self.load_config(self.config_file)
        
        # Configurar sistema
        self._setup_directories()
        self._setup_logging()
        
        # Crear archivo si no existe
        self._ensure_config_file_exists()
        
        # Validar después de configurar
        if not self.validate_config():
            raise ValueError("Configuración inválida")
            
        self._log_system_info()
    
    def _load_default_config(self) -> Dict:
        """Carga la configuración por defecto del sistema."""
        return {
            # === CONFIGURACIÓN DE CAPTURA ===
            "capture": {
                "samples_per_gesture": 7,
                "gestures_per_user": 3,
                "total_captures_formula": "samples_per_gesture * gestures_per_user",
                "required_stable_frames": 3,
                "capture_delay_seconds": 2,
                "enhancement_enabled": True
            },
            
            # === UMBRALES DE CALIDAD ===
            "thresholds": {
                "hand_confidence": 0.90,
                "gesture_confidence": 0.60,
                "movement_threshold": 0.008,
                "target_hand_size": 0.22,
                "size_tolerance": 0.06,
                "visibility_margin": 0.05
            },
            
            # === CONFIGURACIÓN DE CÁMARA ===
            "camera": {
                "width": 1280,
                "height": 720,
                "fps_target": 30,
                "autofocus": True,
                "brightness": 300,
                "contrast": 300,
                "jpeg_quality": 95,
                "warmup_frames": 30
            },
            
            # === CONFIGURACIÓN DE MEDIAPIPE ===
            "mediapipe": {
                "hands": {
                    "static_image_mode": False,
                    "max_num_hands": 1,
                    "model_complexity": 1,
                    "min_detection_confidence": 0.8,
                    "min_tracking_confidence": 0.8
                },
                "gesture_recognizer": {
                    "num_hands": 1,
                    "min_hand_detection_confidence": 0.8,
                    "min_hand_presence_confidence": 0.8,
                    "min_tracking_confidence": 0.8
                }
            },
            
            # === CONFIGURACIÓN ROI NORMALIZATION ===
            "roi_normalization": {
                "min_roi_width": 150,
                "max_roi_width": 600,
                "target_size": [224, 224],
                "roi_padding": 0.15,
                "apply_sharpening": True,
                "apply_contrast": False
            },
            
            # === RUTAS DEL SISTEMA ===
            "paths": {
                "data_root": "biometric_data",
                "logs": "biometric_data/logs",
                "base_captures": "biometric_data/capturas", 
                "models": "biometric_data/models",
                "biometric_db": "biometric_data",
                "templates": "biometric_data/templates",
                "user_profiles": "biometric_data/user_profiles",
                "training_data": "biometric_data/training_data", 
                "backups": "biometric_data/backups",
                "cache": "biometric_data/cache",
                "model_file": "gesture_recognizer.task"
            },
                        
            # === CONFIGURACIÓN BIOMÉTRICA ===
            "biometric": {
                "enrollment": {
                    "min_samples_per_gesture": 10,
                    "max_samples_per_gesture": 20,
                    "quality_threshold": 0.60,
                    "timeout_seconds": 300,
                    "auto_save": True
                },
                "authentication": {
                    "max_attempts": 3,
                    "timeout_seconds": 30,
                    "similarity_threshold": 0.75,
                    "enable_1_to_n": True,
                    "enable_1_to_1": True
                },
                "siamese_networks": {
                    "anatomical": {
                        "embedding_dim": 128,
                        "input_dim": 180,
                        "learning_rate": 0.001,
                        "batch_size": 32,
                        "epochs": 100,
                        "patience": 15,
                        "validation_split": 0.2
                    },
                    "dynamic": {
                        "sequence_length": 50,
                        "feature_dim": 320,
                        "embedding_dim": 128,
                        "learning_rate": 0.0005,
                        "batch_size": 16,
                        "epochs": 150,
                        "patience": 20,
                        "validation_split": 0.2
                    }
                },
                "feature_extraction": {
                    "anatomical_features": [
                        "finger_lengths", "palm_dimensions", "joint_angles",
                        "finger_spreads", "palm_curvature", "hand_proportions",
                        "landmark_distances", "geometric_ratios"
                    ],
                    "dynamic_features": [
                        "transition_velocities", "acceleration_patterns",
                        "gesture_timing", "movement_trajectories",
                        "pressure_patterns", "rhythm_analysis"
                    ]
                },
                "database": {
                    "encryption_enabled": True,
                    "search_strategy": "lsh",
                    "cache_size": 1000,
                    "auto_backup": True,
                    "max_templates_per_user": 50
                }
            },
            
            # === GESTOS DISPONIBLES ===
            "available_gestures": [
                "None", "Closed_Fist", "Open_Palm", "Pointing_Up",
                "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"
            ],
            
            # === CONFIGURACIÓN DE ÁREA DE REFERENCIA ===
            "reference_area": {
                "gesture_areas": {
                    "Pointing_Up": {"width_ratio": 0.4, "height_ratio": 0.8, "center_y_offset": 0.55},
                    "Victory": {"width_ratio": 0.45, "height_ratio": 0.75, "center_y_offset": 0.52},
                    "Thumb_Up": {"width_ratio": 0.4, "height_ratio": 0.7, "center_y_offset": 0.5},
                    "Thumb_Down": {"width_ratio": 0.4, "height_ratio": 0.7, "center_y_offset": 0.5},
                    "ILoveYou": {"width_ratio": 0.5, "height_ratio": 0.75, "center_y_offset": 0.5},
                    "Open_Palm": {"width_ratio": 0.45, "height_ratio": 0.6, "center_y_offset": 0.5},
                    "Closed_Fist": {"width_ratio": 0.45, "height_ratio": 0.6, "center_y_offset": 0.5},
                    "default": {"width_ratio": 0.45, "height_ratio": 0.6, "center_y_offset": 0.5}
                },
                "corner_size": 20,
                "line_thickness": 3,
                "colors": {
                    "area_outline": [0, 255, 255],
                    "valid": [0, 255, 0],
                    "invalid": [0, 0, 255],
                    "warning": [0, 165, 255],
                    "info": [255, 255, 0],
                    "text": [255, 255, 255]
                }
            },
            
            # === CONFIGURACIÓN DE SISTEMA ===
            "system": {
                "debug_mode": False,
                "performance_monitoring": True,
                "auto_cleanup": True,
                "max_log_files": 10,
                "log_retention_days": 30,
                "enable_metrics": True
            }
        }
    
    def _setup_directories(self):
        """Crea SOLO la estructura mínima esencial para el sistema biométrico."""
        base_dir = Path("biometric_data")
        
        try:
            base_dir.mkdir(exist_ok=True)
            
            essential_subdirs = ["models", "templates", "users"]
            
            created_count = 1
            for subdir in essential_subdirs:
                (base_dir / subdir).mkdir(exist_ok=True)
                created_count += 1
            
            self._config["paths"]["models"] = "biometric_data/models"
            self._config["paths"]["biometric_db"] = "biometric_data"
            self._config["paths"]["templates"] = "biometric_data/templates"
            self._config["paths"]["logs"] = "biometric_data/logs"
            self._config["paths"]["backups"] = "biometric_data/backups"
            
            logging.info(f"Estructura mínima creada - {created_count} directorios esenciales")
            logging.info(f"Las demás carpetas se crearán automáticamente cuando se necesiten")
            
        except Exception as e:
            logging.error(f"No se pudo crear estructura mínima: {e}")
            base_dir.mkdir(exist_ok=True)
            self._config["paths"]["biometric_db"] = "biometric_data"
            logging.info(f"1 directorio creado (fallback)")
        
    def _setup_logging(self):
        """Configura el sistema de logging evitando duplicados."""
        if self.logger and self.logger.handlers:
            return
        
        root_logger = logging.getLogger()
        root_logger.handlers = []
        
        self.logger = logging.getLogger('biometric_gesture_system')
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        try:
            log_filename = f"biometric_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            logs_dir = Path(self._config["paths"]["logs"])
            logs_dir.mkdir(exist_ok=True)
            
            log_filepath = logs_dir / log_filename
            
            file_handler = logging.FileHandler(str(log_filepath), encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            self.logger.propagate = False
            
        except Exception as e:
            logging.error(f"ERROR configurando logging: {e}")
            self.logger = None
    
    def _ensure_config_file_exists(self):
        """Crea el archivo de configuración si no existe."""
        if not os.path.exists(self.config_file):
            try:
                self.save_config(self.config_file)
                if self.logger:
                    self.logger.info(f"Archivo de configuración creado: {self.config_file}")
                else:
                    logging.info(f"Archivo de configuración creado: {self.config_file}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error creando archivo de configuración: {e}")
                else:
                    logging.error(f"Error creando archivo de configuración: {e}")
    
    def _log_system_info(self):
        """Registra información del sistema al inicio."""
        if not self.logger:
            return
            
        self.logger.info("="*80)
        self.logger.info("SISTEMA BIOMÉTRICO DE GESTOS DE MANOS - CONFIG MANAGER INICIALIZADO")
        self.logger.info("="*80)
        self.logger.info(f"Configuración cargada desde: {self.config_file}")
        self.logger.info(f"Muestras por gesto: {self.get('capture.samples_per_gesture')}")
        self.logger.info(f"Gestos por usuario: {self.get('capture.gestures_per_user')}")
        self.logger.info(f"Umbral confianza mano: {self.get('thresholds.hand_confidence')}")
        self.logger.info(f"Umbral confianza gesto: {self.get('thresholds.gesture_confidence')}")
        self.logger.info(f"Resolución cámara: {self.get('camera.width')}x{self.get('camera.height')}")
        self.logger.info(f"Directorios creados: {len(self._config['paths'])} rutas configuradas")
        self.logger.info("="*80)
    
    def get(self, key: str, default=None):
        """
        Obtiene un valor de configuración usando notación de punto.
        
        Args:
            key: Clave en formato 'seccion.subseccion.valor'
            default: Valor por defecto si no se encuentra la clave
            
        Returns:
            Valor de configuración o default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if self.logger:
                self.logger.warning(f"Clave de configuración no encontrada: {key}, usando default: {default}")
            return default
    
    def set(self, key: str, value):
        """
        Establece un valor de configuración usando notación de punto.
        
        Args:
            key: Clave en formato 'seccion.subseccion.valor'
            value: Nuevo valor
        """
        keys = key.split('.')
        config_section = self._config
        
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]
        
        old_value = config_section.get(keys[-1], "No definido")
        config_section[keys[-1]] = value
        
        if self.logger:
            self.logger.info(f"Configuración actualizada - {key}: {old_value} → {value}")
    
    def load_config(self, filepath: str):
        """Carga configuración desde un archivo JSON."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                self._deep_merge(self._config, loaded_config)
                if self.logger:
                    self.logger.info(f"Configuración cargada desde: {filepath}")
            else:
                if self.logger:
                    self.logger.warning(f"Archivo de configuración no encontrado: {filepath}")
                    self.logger.info("Usando configuración por defecto")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error al cargar configuración: {e}")
                self.logger.info("Usando configuración por defecto")
    
    def save_config(self, filepath: Optional[str] = None):
        """Guarda la configuración actual en un archivo JSON."""
        if filepath is None:
            filepath = self.config_file
            
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            if self.logger:
                self.logger.info(f"Configuración guardada en: {filepath}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error al guardar configuración: {e}")
    
    def backup_config(self):
        """Crea backup de la configuración actual."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = Path(self.get('paths.backups', 'backups'))
            backup_dir.mkdir(exist_ok=True)
            backup_file = backup_dir / f"config_backup_{timestamp}.json"
            
            self.save_config(str(backup_file))
            if self.logger:
                self.logger.info(f"Backup de configuración creado: {backup_file}")
            return str(backup_file)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creando backup: {e}")
            return None
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """Fusiona diccionarios de forma recursiva."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _validate_paths(self) -> bool:
        """Valida que solo las rutas esenciales sean accesibles."""
        essential_paths = ["models", "templates", "biometric_db"]
        
        paths = self.get('paths', {})
        for path_name, path_value in paths.items():
            if path_name == 'model_file':
                continue
                
            if path_name in essential_paths:
                try:
                    Path(path_value).mkdir(parents=True, exist_ok=True)
                    if self.logger:
                        self.logger.debug(f"Ruta esencial validada: {path_name} -> {path_value}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error validando ruta esencial {path_name} ({path_value}): {e}")
                    return False
            else:
                if not path_value:
                    if self.logger:
                        self.logger.warning(f"Ruta opcional {path_name} no definida")
                else:
                    if self.logger:
                        self.logger.debug(f"Ruta opcional configurada: {path_name} -> {path_value}")
        
        return True
    
    def validate_config(self) -> bool:
        """Valida que la configuración actual sea correcta."""
        required_keys = [
            'capture.samples_per_gesture',
            'capture.gestures_per_user',
            'thresholds.hand_confidence',
            'thresholds.gesture_confidence',
            'camera.width',
            'camera.height',
            'paths.models',
            'paths.biometric_db'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                if self.logger:
                    self.logger.error(f"Configuración inválida: falta la clave requerida '{key}'")
                return False
        
        if not (0.0 <= self.get('thresholds.hand_confidence') <= 1.0):
            if self.logger:
                self.logger.error("thresholds.hand_confidence debe estar entre 0.0 y 1.0")
            return False
        
        if not (0.0 <= self.get('thresholds.gesture_confidence') <= 1.0):
            if self.logger:
                self.logger.error("thresholds.gesture_confidence debe estar entre 0.0 y 1.0")
            return False
        
        width = self.get('camera.width')
        height = self.get('camera.height')
        if not (isinstance(width, int) and width > 0):
            if self.logger:
                self.logger.error("camera.width debe ser un entero positivo")
            return False
        if not (isinstance(height, int) and height > 0):
            if self.logger:
                self.logger.error("camera.height debe ser un entero positivo")
            return False
        
        if not self._validate_paths():
            return False
        
        if self.logger:
            self.logger.info("Validación de configuración: ✓ EXITOSA")
        return True
    
    # === MÉTODOS DE CONVENIENCIA ===
    
    def get_gesture_requirements(self, gesture_name: str) -> str:
        """Obtiene los requisitos de área para un gesto específico."""
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
    
    def get_model_path(self) -> str:
        """Obtiene la ruta completa del modelo MediaPipe."""
        model_file = self.get('paths.model_file')
        models_dir = self.get('paths.models')
        return os.path.join(models_dir, model_file)
    
    def get_user_profile_path(self, user_id: str) -> str:
        """Obtiene la ruta del perfil de un usuario específico."""
        profiles_dir = self.get('paths.user_profiles')
        return os.path.join(profiles_dir, f"user_{user_id}.json")
    
    def get_total_captures(self) -> int:
        """Calcula el total de capturas requeridas."""
        samples_per_gesture = self.get('capture.samples_per_gesture', 7)
        gestures_per_user = self.get('capture.gestures_per_user', 3)
        return samples_per_gesture * gestures_per_user
    
    def log_capture_info(self, gesture_name: str, capture_num: int, 
                        hand_confidence: float, gesture_confidence: float, 
                        hand_size: float, user_id: Optional[str] = None):
        """Registra información detallada de una captura."""
        if not self.logger:
            return
            
        user_info = f"Usuario: {user_id} - " if user_id else ""
        self.logger.info(
            f"CAPTURA - {user_info}Gesto: {gesture_name} #{capture_num} - "
            f"Conf.Mano: {hand_confidence:.3f} - Conf.Gesto: {gesture_confidence:.3f} - "
            f"Tamaño: {hand_size:.3f}"
        )
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Registra errores con información detallada."""
        if not self.logger:
            logging.error(f"ERROR: {message}")
            return
            
        if exception:
            self.logger.error(f"{message} - Excepción: {str(exception)}", exc_info=True)
        else:
            self.logger.error(message)
    
    def get_system_info(self) -> Dict:
        """Obtiene información completa del sistema."""
        return {
            "config_file": self.config_file,
            "total_captures_required": self.get_total_captures(),
            "gestures_available": len(self.get('available_gestures', [])),
            "paths_configured": len(self.get('paths', {})),
            "camera_resolution": f"{self.get('camera.width')}x{self.get('camera.height')}",
            "logging_enabled": self.logger is not None,
            "config_valid": self.validate_config()
        }


# ===== INSTANCIA GLOBAL =====
_config_manager_instance = None

def get_config_manager() -> ConfigManager:
    """Obtiene o crea la instancia global de ConfigManager."""
    global _config_manager_instance
    if _config_manager_instance is None:
        try:
            _config_manager_instance = ConfigManager()
        except Exception as e:
            logging.error(f"ERROR CRÍTICO inicializando ConfigManager: {e}")
            raise
    return _config_manager_instance


# ===== FUNCIONES DE CONVENIENCIA =====
def get_config(key: str, default=None):
    """Función de conveniencia para obtener configuración."""
    try:
        config_mgr = get_config_manager()
        return config_mgr.get(key, default)
    except:
        return default

def get_logger():
    """Función de conveniencia para obtener el logger."""
    try:
        config_mgr = get_config_manager()
        return config_mgr.logger
    except:
        return None

def log_info(message: str):
    """Función de conveniencia para logging de información."""
    try:
        config_mgr = get_config_manager()
        if config_mgr and config_mgr.logger:
            config_mgr.logger.info(message)
        else:
            logging.info(message)
    except:
        logging.info(message)

def log_error(message: str, exception: Optional[Exception] = None):
    """Función de conveniencia para logging de errores."""
    try:
        config_mgr = get_config_manager()
        if config_mgr:
            config_mgr.log_error(message, exception)
        else:
            logging.error(message)
    except:
        logging.error(message)