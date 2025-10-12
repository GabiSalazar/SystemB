"""
System Manager - Gestor principal del sistema biométrico para FastAPI
Adaptado del Main.py del notebook
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.core.config_manager import get_logger
from app.core.biometric_database import get_biometric_database
from app.core.enrollment_system import get_real_enrollment_system
from app.core.authentication_system import get_real_authentication_system
from app.core.siamese_anatomical_network import get_real_siamese_anatomical_network
from app.core.siamese_dynamic_network import get_real_siamese_dynamic_network

logger = get_logger()


# ====================================================================
# ENUMERACIONES Y ESTADO
# ====================================================================

class SystemMode(Enum):
    """Modos principales del sistema."""
    BASIC_SETUP = "basic_setup"
    ENROLLMENT_READY = "enrollment_ready"
    TRAINING_READY = "training_ready"
    FULL_SYSTEM = "full_system"
    ERROR = "error"


class InitializationLevel(Enum):
    """Niveles de inicialización."""
    NONE = 0
    BASIC_COMPONENTS = 1
    FEATURE_EXTRACTION = 2
    NEURAL_NETWORKS = 3
    FULL_PIPELINE = 4


@dataclass
class SystemState:
    """Estado actual del sistema."""
    initialization_level: InitializationLevel = InitializationLevel.NONE
    users_count: int = 0
    networks_trained: bool = False
    database_ready: bool = False
    enrollment_active: bool = False
    authentication_active: bool = False
    error_message: Optional[str] = None


# ====================================================================
# GESTOR DEL SISTEMA
# ====================================================================

class BiometricSystemManager:
    """
    Gestor principal del sistema biométrico para FastAPI.
    Coordina todos los componentes del sistema.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa el gestor (solo una vez)."""
        if self._initialized:
            return
        
        self.state = SystemState()
        self.start_time = time.time()
        
        # Componentes principales
        self.database = None
        self.enrollment_system = None
        self.authentication_system = None
        self.anatomical_network = None
        self.dynamic_network = None
        
        logger.info("BiometricSystemManager creado")
        self._initialized = True
    
    def initialize_system(self) -> bool:
        """
        Inicializa el sistema completo de forma progresiva.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            logger.info("=== INICIANDO INICIALIZACIÓN DEL SISTEMA ===")
            
            # Nivel 1: Base de datos
            if not self._initialize_database():
                self.state.error_message = "Error inicializando base de datos"
                return False
            
            self.state.initialization_level = InitializationLevel.BASIC_COMPONENTS
            logger.info("✅ Nivel 1: Base de datos inicializada")
            
            # Verificar usuarios
            users = self.database.list_users()
            self.state.users_count = len(users)
            self.state.database_ready = True
            
            logger.info(f"📊 Usuarios en base de datos: {self.state.users_count}")
            
            # Nivel 2: Sistema de enrollment
            if not self._initialize_enrollment():
                self.state.error_message = "Error inicializando enrollment"
                return False
            
            self.state.enrollment_active = True
            self.state.initialization_level = InitializationLevel.FEATURE_EXTRACTION
            logger.info("✅ Nivel 2: Sistema de enrollment inicializado")
            
            # Nivel 3: Verificar redes entrenadas
            networks_trained = self._check_networks_trained()
            self.state.networks_trained = networks_trained
            
            if networks_trained:
                logger.info("✅ Redes siamesas ya están entrenadas")
                
                # Nivel 4: Sistema de autenticación
                if self._initialize_authentication():
                    self.state.authentication_active = True
                    self.state.initialization_level = InitializationLevel.FULL_PIPELINE
                    logger.info("✅ Nivel 4: Sistema de autenticación inicializado")
                else:
                    logger.warning("⚠️ Error inicializando autenticación")
            else:
                logger.info("📝 Redes necesitan entrenamiento")
            
            logger.info("=== SISTEMA INICIALIZADO CORRECTAMENTE ===")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error crítico en inicialización: {e}")
            self.state.error_message = str(e)
            return False
    
    def _initialize_database(self) -> bool:
        """Inicializa la base de datos."""
        try:
            self.database = get_biometric_database()
            logger.info("✅ Base de datos inicializada")
            return True
        except Exception as e:
            logger.error(f"❌ Error inicializando base de datos: {e}")
            return False
    
    def _initialize_enrollment(self) -> bool:
        """Inicializa el sistema de enrollment."""
        try:
            self.enrollment_system = get_real_enrollment_system()
            logger.info("✅ Sistema de enrollment inicializado")
            return True
        except Exception as e:
            logger.error(f"❌ Error inicializando enrollment: {e}")
            return False
    
    def _check_networks_trained(self) -> bool:
        """Verifica si las redes están entrenadas."""
        try:
            # Obtener redes globales
            self.anatomical_network = get_real_siamese_anatomical_network()
            self.dynamic_network = get_real_siamese_dynamic_network()
            
            # Verificar archivos de modelo
            models_dir = Path('biometric_data/models')
            anat_file = models_dir / 'anatomical_model.h5'
            dyn_file = models_dir / 'dynamic_model.h5'
            
            anat_trained = self.anatomical_network.is_trained or anat_file.exists()
            dyn_trained = self.dynamic_network.is_trained or dyn_file.exists()
            
            # Marcar como entrenadas si existen archivos
            if anat_file.exists() and not self.anatomical_network.is_trained:
                self.anatomical_network.is_trained = True
            
            if dyn_file.exists() and not self.dynamic_network.is_trained:
                self.dynamic_network.is_trained = True
            
            both_trained = anat_trained and dyn_trained
            
            logger.info(f"📊 Estado de redes:")
            logger.info(f"  - Anatómica: {'✅' if anat_trained else '❌'}")
            logger.info(f"  - Dinámica: {'✅' if dyn_trained else '❌'}")
            
            return both_trained
            
        except Exception as e:
            logger.error(f"❌ Error verificando redes: {e}")
            return False
    
    def _initialize_authentication(self) -> bool:
        """Inicializa el sistema de autenticación."""
        try:
            self.authentication_system = get_real_authentication_system()
            
            if self.authentication_system.initialize_real_system():
                logger.info("✅ Sistema de autenticación inicializado")
                return True
            else:
                logger.error("❌ Error inicializando autenticación")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error en autenticación: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema.
        
        Returns:
            Dict con el estado completo del sistema
        """
        uptime = time.time() - self.start_time
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        
        return {
            'initialization_level': self.state.initialization_level.name,
            'users_count': self.state.users_count,
            'networks_trained': self.state.networks_trained,
            'database_ready': self.state.database_ready,
            'enrollment_active': self.state.enrollment_active,
            'authentication_active': self.state.authentication_active,
            'error_message': self.state.error_message,
            'uptime': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            'version': '2.0.0',
            'status': 'operational' if not self.state.error_message else 'error'
        }
    
    def train_networks(self) -> bool:
        """
        Entrena las redes neuronales con datos disponibles.
        
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        try:
            if self.state.users_count < 2:
                logger.error("Se necesitan al menos 2 usuarios para entrenar")
                return False
            
            logger.info("🧠 Iniciando entrenamiento de redes...")
            
            # Entrenar red anatómica
            logger.info("Entrenando red anatómica...")
            if not self.anatomical_network.train_with_real_data(self.database):
                logger.error("Error entrenando red anatómica")
                return False
            
            logger.info("✅ Red anatómica entrenada")
            
            # Entrenar red dinámica
            logger.info("Entrenando red dinámica...")
            if not self.dynamic_network.train_with_real_data(self.database):
                logger.error("Error entrenando red dinámica")
                return False
            
            logger.info("✅ Red dinámica entrenada")
            
            # Actualizar estado
            self.state.networks_trained = True
            
            # Inicializar autenticación
            if self._initialize_authentication():
                self.state.authentication_active = True
                self.state.initialization_level = InitializationLevel.FULL_PIPELINE
            
            logger.info("✅ Entrenamiento completado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            return False


# ====================================================================
# FUNCIÓN GLOBAL
# ====================================================================

_system_manager_instance = None

def get_system_manager() -> BiometricSystemManager:
    """
    Obtiene la instancia global del gestor del sistema.
    
    Returns:
        BiometricSystemManager: Instancia única del gestor
    """
    global _system_manager_instance
    
    if _system_manager_instance is None:
        _system_manager_instance = BiometricSystemManager()
    
    return _system_manager_instance