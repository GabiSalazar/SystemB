"""
System Manager - Gestor principal del sistema biométrico para FastAPI
Arquitectura: 15 módulos en 4 capas funcionales
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np

from app.core.config_manager import get_logger, get_config
from app.core.biometric_database import get_biometric_database
from app.core.enrollment_system import get_real_enrollment_system
from app.core.authentication_system import get_real_authentication_system
from app.core.siamese_anatomical_network import get_real_siamese_anatomical_network
from app.core.siamese_dynamic_network import get_real_siamese_dynamic_network
from app.core.camera_manager import get_camera_manager, release_camera
from app.core.mediapipe_processor import get_mediapipe_processor, release_mediapipe

logger = get_logger()


# ====================================================================
# ENUMERACIONES Y ESTADO DEL SISTEMA
# ====================================================================

class SystemMode(Enum):
    """Modos principales del sistema."""
    BASIC_SETUP = "basic_setup"
    ENROLLMENT_READY = "enrollment_ready"
    TRAINING_READY = "training_ready"
    FULL_SYSTEM = "full_system"
    ERROR = "error"


class InitializationLevel(Enum):
    """Niveles de inicialización progresiva."""
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
    bootstrap_mode: bool = False  # Modo para primeros usuarios
    error_message: Optional[str] = None
    
    # Estadísticas de componentes
    modules_loaded: Dict[str, bool] = field(default_factory=dict)
    last_training_time: Optional[str] = None
    total_enrollments: int = 0
    total_authentications: int = 0
    total_verifications: int = 0
    total_identifications: int = 0


# ====================================================================
# GESTOR PRINCIPAL DEL SISTEMA
# ====================================================================

class BiometricSystemManager:
    """
    Gestor principal del sistema biométrico para FastAPI.
    
    Coordina todos los 15 módulos del sistema organizados en 4 capas:
    - CAPA 1: Componentes básicos (config, cámara, mediapipe, validación, áreas)
    - CAPA 2: Extracción de características (anatómicas, dinámicas, secuencias)
    - CAPA 3: Redes neuronales (siamesas anatómica/dinámica, preprocesador, fusión)
    - CAPA 4: Sistema completo (base de datos, enrollment, autenticación)
    
    Basado en BiometricGestureSystemReal del notebook MAIN.py
    """
    
    _instance = None
    _initialized = False
    
    # Módulos requeridos del sistema (15 módulos)
    REQUIRED_MODULES = {
        'config_manager': 'ConfigManager',
        'camera_manager': 'CameraManager',
        'mediapipe_processor': 'MediaPipeProcessor',
        'quality_validator': 'QualityValidator',
        'reference_area_manager': 'ReferenceAreaManager',
        'anatomical_features_extractor': 'AnatomicalFeaturesExtractor',
        'dynamic_features_extractor': 'RealDynamicFeaturesExtractor',
        'sequence_manager': 'SequenceManager',
        'siamese_anatomical_network': 'RealSiameseAnatomicalNetwork',
        'siamese_dynamic_network': 'RealSiameseDynamicNetwork',
        'feature_preprocessor': 'RealFeaturePreprocessor',
        'score_fusion_system': 'RealScoreFusionSystem',
        'biometric_database': 'BiometricDatabase',
        'enrollment_system': 'RealEnrollmentSystem',
        'authentication_system': 'RealAuthenticationSystem'
    }
    
    def __new__(cls):
        """Singleton pattern - Una sola instancia del sistema."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa el gestor (solo una vez)."""
        if self._initialized:
            return
        
        self.state = SystemState()
        self.start_time = time.time()
        
        # Componentes principales (Módulos 13-15)
        self.database = None
        self.enrollment_system = None
        self.authentication_system = None
        
        # Redes neuronales (Módulos 9-10)
        self.anatomical_network = None
        self.dynamic_network = None
        
        # Componentes auxiliares
        self.camera_manager = None
        self.mediapipe_processor = None
        
        logger.info("=" * 80)
        logger.info("🚀 BiometricSystemManager v2.0.0 Real Edition Iniciado")
        logger.info("🏗️  Arquitectura: 15 Módulos + Main | 4 Capas Funcionales")
        logger.info("🎯 Características: Redes siamesas, fusión multimodal, templates biométricos")
        logger.info("=" * 80)
        
        self._initialized = True
    
    def verify_modules(self) -> Tuple[bool, List[str]]:
        """
        Verifica que todos los módulos requeridos estén disponibles.
        Equivalente a verify_notebook_modules() del MAIN.
        
        Returns:
            Tuple[bool, List[str]]: (todos_disponibles, módulos_faltantes)
        """
        missing = []
        
        logger.info("VERIFICANDO MÓDULOS DEL SISTEMA...")
        logger.info("=" * 80)
        
        for module_name, class_name in self.REQUIRED_MODULES.items():
            try:
                # Intentar importar el módulo
                module_path = f"app.core.{module_name}"
                __import__(module_path)
                
                self.state.modules_loaded[class_name] = True
                logger.info(f"✅ {class_name}")
                
            except ImportError as e:
                self.state.modules_loaded[class_name] = False
                missing.append(class_name)
                logger.error(f"❌ {class_name} NO disponible: {e}")
        
        all_available = len(missing) == 0
        
        if all_available:
            logger.info(f"\n✅ TODOS LOS {len(self.REQUIRED_MODULES)} MÓDULOS DISPONIBLES")
        else:
            logger.error(f"\n❌ FALTAN {len(missing)} MÓDULOS:")
            for module in missing:
                logger.error(f"   - {module}")
        
        logger.info("=" * 80)
        return all_available, missing
    
    def initialize_system(self) -> bool:
        """
        Inicializa el sistema completo de forma progresiva.
        Equivalente a initialize_real_progressive() del MAIN.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("🔧 INICIANDO INICIALIZACIÓN PROGRESIVA DEL SISTEMA")
            logger.info("=" * 80)
            
            # Paso 0: Verificar módulos
            modules_ok, missing = self.verify_modules()
            if not modules_ok:
                self.state.error_message = f"Módulos faltantes: {', '.join(missing)}"
                logger.error("🚨 ERROR: Módulos no disponibles")
                return False
            
            # ================================================================
            # NIVEL 1: COMPONENTES BÁSICOS
            # ================================================================
            logger.info("\n🔧 NIVEL 1: Inicializando Componentes Básicos")
            logger.info("-" * 80)
            
            if not self._initialize_real_basic_components():
                self.state.error_message = "Error en Nivel 1 (Componentes Básicos)"
                return False
            
            self.state.initialization_level = InitializationLevel.BASIC_COMPONENTS
            logger.info("✅ NIVEL 1 COMPLETADO: Componentes básicos listos\n")
            
            # ================================================================
            # NIVEL 2: EXTRACTORES DE CARACTERÍSTICAS
            # ================================================================
            logger.info("📊 NIVEL 2: Inicializando Extractores de Características")
            logger.info("-" * 80)
            
            if not self._initialize_real_feature_extractors():
                self.state.error_message = "Error en Nivel 2 (Extractores)"
                return False
            
            self.state.initialization_level = InitializationLevel.FEATURE_EXTRACTION
            self.state.enrollment_active = True  # Enrollment siempre disponible
            logger.info("✅ NIVEL 2 COMPLETADO: Extractores listos\n")
            
            # ================================================================
            # NIVEL 3: REDES NEURONALES
            # ================================================================
            logger.info("🧠 NIVEL 3: Verificando Redes Neuronales")
            logger.info("-" * 80)
            
            networks_trained = self._check_real_networks_trained()
            self.state.networks_trained = networks_trained
            self.state.initialization_level = InitializationLevel.NEURAL_NETWORKS
            
            if networks_trained:
                logger.info("✅ NIVEL 3 COMPLETADO: Redes entrenadas y cargadas\n")
            else:
                logger.warning("⚠️ NIVEL 3 PARCIAL: Redes necesitan entrenamiento")
                logger.info(f"📝 Usuarios actuales: {self.state.users_count}")
                logger.info(f"📝 Mínimo requerido: 2 usuarios para entrenar")
                
                # Activar modo bootstrap
                if self.state.users_count < 2:
                    self.state.bootstrap_mode = True
                    logger.info("🚀 MODO BOOTSTRAP ACTIVADO: Permitir enrollment sin redes\n")
            
            # ================================================================
            # NIVEL 4: PIPELINE COMPLETO
            # ================================================================
            logger.info("🎯 NIVEL 4: Inicializando Pipeline Completo")
            logger.info("-" * 80)
            
            if self._initialize_real_authentication_system():
                self.state.authentication_active = True
                self.state.initialization_level = InitializationLevel.FULL_PIPELINE
                logger.info("✅ NIVEL 4 COMPLETADO: Sistema 100% funcional\n")
            else:
                logger.info("⚠️ NIVEL 4 PARCIAL: Enrollment disponible, autenticación pendiente\n")
            
            # Resumen final
            self._print_initialization_summary()
            
            logger.info("=" * 80)
            logger.info("✅ SISTEMA INICIALIZADO CORRECTAMENTE")
            logger.info("=" * 80)
            return True
            
        except Exception as e:
            logger.error(f"🚨 ERROR CRÍTICO EN INICIALIZACIÓN: {e}", exc_info=True)
            self.state.error_message = str(e)
            return False
    
    def _initialize_real_basic_components(self) -> bool:
        """
        NIVEL 1: Inicializa componentes básicos.
        Equivalente a _initialize_real_basic_components() del MAIN.
        """
        try:
            logger.info("Inicializando Base de Datos...")
            
            # Módulo 13: BiometricDatabase
            self.database = get_biometric_database()
            
            # Verificar usuarios existentes
            users = self.database.list_users()
            self.state.users_count = len(users)
            self.state.database_ready = True
            
            logger.info(f"✅ Base de datos lista: {self.state.users_count} usuarios registrados")
            
            # Obtener estadísticas
            try:
                db_stats = self.database.get_database_stats()
                total_templates = db_stats.get('total_templates', 0)
                logger.info(f"📊 Templates totales: {total_templates}")
            except:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en componentes básicos: {e}")
            return False
    
    def _initialize_real_feature_extractors(self) -> bool:
        """
        NIVEL 2: Inicializa extractores de características.
        Equivalente a _initialize_real_feature_extractors() del MAIN.
        """
        try:
            logger.info("Inicializando Sistema de Enrollment...")
            
            # Módulo 14: RealEnrollmentSystem
            self.enrollment_system = get_real_enrollment_system()
            
            # Verificar modo bootstrap
            bootstrap = self.enrollment_system.check_bootstrap_mode()
            if bootstrap:
                logger.info("🚀 Modo Bootstrap detectado - Primeros usuarios")
                self.state.bootstrap_mode = True
            
            logger.info("✅ Sistema de enrollment listo")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en extractores: {e}")
            return False
    
    def _check_real_networks_trained(self) -> bool:
        """
        NIVEL 3: Verifica si las redes están entrenadas.
        Equivalente a _check_real_networks_trained() del MAIN.
        """
        try:
            # Módulo 9: RealSiameseAnatomicalNetwork
            logger.info("Verificando Red Anatómica...")
            self.anatomical_network = get_real_siamese_anatomical_network()
            
            # Módulo 10: RealSiameseDynamicNetwork
            logger.info("Verificando Red Dinámica...")
            self.dynamic_network = get_real_siamese_dynamic_network()
            
            # Verificar archivos de modelo
            models_dir = Path('biometric_data/models')
            anat_file = models_dir / 'anatomical_model.h5'
            dyn_file = models_dir / 'dynamic_model.h5'
            
            anat_trained = anat_file.exists()
            dyn_trained = dyn_file.exists()
            
            logger.info(f"📊 Estado de redes:")
            logger.info(f"   - Anatómica: {'✅ Entrenada' if anat_trained else '❌ Sin entrenar'}")
            logger.info(f"   - Dinámica: {'✅ Entrenada' if dyn_trained else '❌ Sin entrenar'}")
            
            # Cargar modelos si existen
            if anat_trained:
                try:
                    success = self.anatomical_network.load_real_trained_model(str(anat_file))
                    if success:
                        logger.info("   ✅ Modelo anatómico cargado correctamente")
                except Exception as e:
                    logger.warning(f"   ⚠️ Error cargando modelo anatómico: {e}")
            
            if dyn_trained:
                try:
                    success = self.dynamic_network.load_real_trained_model(str(dyn_file))
                    if success:
                        logger.info("   ✅ Modelo dinámico cargado correctamente")
                except Exception as e:
                    logger.warning(f"   ⚠️ Error cargando modelo dinámico: {e}")
            
            both_trained = anat_trained and dyn_trained
            
            return both_trained
            
        except Exception as e:
            logger.error(f"❌ Error verificando redes: {e}")
            return False
    
    def _initialize_real_authentication_system(self) -> bool:
        """
        NIVEL 4: Inicializa el sistema de autenticación.
        Equivalente a _initialize_real_authentication_system() del MAIN.
        """
        try:
            if not self.state.networks_trained:
                logger.info("⏭️  Omitiendo autenticación (redes no entrenadas)")
                return False
            
            logger.info("Inicializando Sistema de Autenticación...")
            
            # Módulo 15: RealAuthenticationSystem
            self.authentication_system = get_real_authentication_system()
            
            # Inicializar pipeline completo
            if self.authentication_system.initialize_real_system():
                logger.info("✅ Sistema de autenticación listo")
                return True
            else:
                logger.error("❌ Error inicializando autenticación")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error en autenticación: {e}")
            return False
    
    def _print_initialization_summary(self):
        """
        Imprime resumen de inicialización.
        Equivalente a show_status() del MAIN.
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 RESUMEN DE INICIALIZACIÓN")
        logger.info("=" * 80)
        logger.info(f"  Nivel de Inicialización: {self.state.initialization_level.name}")
        logger.info(f"  Usuarios Registrados:    {self.state.users_count}")
        logger.info(f"  Redes Entrenadas:        {'✅ Sí' if self.state.networks_trained else '❌ No'}")
        logger.info(f"  Base de Datos:           {'✅ Lista' if self.state.database_ready else '❌ Error'}")
        logger.info(f"  Enrollment:              {'✅ Activo' if self.state.enrollment_active else '❌ Inactivo'}")
        logger.info(f"  Autenticación:           {'✅ Activa' if self.state.authentication_active else '❌ Inactiva'}")
        logger.info(f"  Modo Bootstrap:          {'🚀 Sí' if self.state.bootstrap_mode else 'No'}")
        
        if self.state.error_message:
            logger.error(f"  ⚠️ Error:                {self.state.error_message}")
        
        logger.info("=" * 80)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema.
        Equivalente a show_status() pero retorna dict para API.
        
        Returns:
            Dict con el estado completo del sistema
        """
        uptime = time.time() - self.start_time
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        
        # Determinar modo del sistema
        if self.state.error_message:
            mode = SystemMode.ERROR
        elif self.state.authentication_active:
            mode = SystemMode.FULL_SYSTEM
        elif self.state.networks_trained:
            mode = SystemMode.TRAINING_READY
        elif self.state.enrollment_active:
            mode = SystemMode.ENROLLMENT_READY
        else:
            mode = SystemMode.BASIC_SETUP
        
        return {
            'version': '2.0.0',
            'status': 'operational' if not self.state.error_message else 'error',
            'mode': mode.value,
            
            # Estado de inicialización
            'initialization': {
                'level': self.state.initialization_level.name,
                'level_value': self.state.initialization_level.value,
                'bootstrap_mode': self.state.bootstrap_mode
            },
            
            # Componentes
            'components': {
                'database_ready': self.state.database_ready,
                'enrollment_active': self.state.enrollment_active,
                'authentication_active': self.state.authentication_active,
                'networks_trained': self.state.networks_trained
            },
            
            # Estadísticas
            'statistics': {
                'users_count': self.state.users_count,
                'total_enrollments': self.state.total_enrollments,
                'total_authentications': self.state.total_authentications,
                'total_verifications': self.state.total_verifications,
                'total_identifications': self.state.total_identifications,
                'modules_loaded': len([v for v in self.state.modules_loaded.values() if v]),
                'total_modules': len(self.REQUIRED_MODULES)
            },
            
            # Sistema
            'system': {
                'uptime': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                'uptime_seconds': int(uptime),
                'last_training': self.state.last_training_time,
                'error': self.state.error_message
            },
            
            # Módulos cargados
            'modules': self.state.modules_loaded
        }
    
    def list_users(self) -> List[Dict[str, Any]]:
        """
        Lista todos los usuarios registrados.
        Equivalente a list_users() del MAIN.
        
        Returns:
            Lista de diccionarios con información de usuarios
        """
        if not self.database:
            return []
        
        users_list = []
        
        try:
            all_users = self.database.list_users()
            
            for user_data in all_users:
                # user_data puede ser UserProfile o dict
                if hasattr(user_data, 'user_id'):
                    # Es un UserProfile
                    users_list.append({
                        'user_id': user_data.user_id,
                        'username': user_data.username,
                        'gesture_sequence': user_data.gesture_sequence,
                        'total_enrollments': getattr(user_data, 'total_enrollments', 0),
                        'created_at': user_data.metadata.get('created_at', 'N/A') if hasattr(user_data, 'metadata') else 'N/A'
                    })
                else:
                    # Es un dict
                    users_list.append({
                        'user_id': user_data.get('user_id', 'unknown'),
                        'username': user_data.get('username', 'unknown'),
                        'gesture_sequence': user_data.get('gesture_sequence', []),
                        'total_enrollments': user_data.get('total_enrollments', 0),
                        'created_at': user_data.get('created_at', 'N/A')
                    })
        except Exception as e:
            logger.error(f"Error listando usuarios: {e}")
        
        return users_list
    
    def train_networks(self, force: bool = False) -> Dict[str, Any]:
        """
        Entrena las redes neuronales con datos disponibles.
        Equivalente a train_real_networks() del MAIN.
        
        Args:
            force: Forzar reentrenamiento incluso si ya están entrenadas
        
        Returns:
            Dict con resultado del entrenamiento
        """
        result = {
            'success': False,
            'message': '',
            'details': {}
        }
        
        try:
            # Validar usuarios
            if self.state.users_count < 2:
                result['message'] = f"Se necesitan al menos 2 usuarios (actual: {self.state.users_count})"
                logger.error(result['message'])
                return result
            
            # Verificar si ya están entrenadas
            if self.state.networks_trained and not force:
                result['message'] = "Las redes ya están entrenadas. Use force=True para reentrenar."
                logger.warning(result['message'])
                return result
            
            logger.info("=" * 80)
            logger.info("🧠 INICIANDO ENTRENAMIENTO DE REDES NEURONALES")
            logger.info("=" * 80)
            
            training_start = time.time()
            
            # ================================================================
            # ENTRENAR RED ANATÓMICA
            # ================================================================
            logger.info("\n🧠 [1/2] Entrenando Red Siamesa Anatómica...")
            logger.info("-" * 80)
            
            anat_success = self.anatomical_network.train_with_real_data(self.database)
            
            if not anat_success:
                result['message'] = "Error entrenando red anatómica"
                logger.error(result['message'])
                return result
            
            logger.info("✅ Red Anatómica entrenada correctamente")
            
            # ================================================================
            # ENTRENAR RED DINÁMICA
            # ================================================================
            logger.info("\n🧠 [2/2] Entrenando Red Siamesa Dinámica...")
            logger.info("-" * 80)
            
            dyn_success = self.dynamic_network.train_with_real_data(self.database)
            
            if not dyn_success:
                result['message'] = "Error entrenando red dinámica"
                logger.error(result['message'])
                return result
            
            logger.info("✅ Red Dinámica entrenada correctamente")
            
            training_time = time.time() - training_start
            
            # ================================================================
            # ACTUALIZAR ESTADO Y ACTIVAR AUTENTICACIÓN
            # ================================================================
            self.state.networks_trained = True
            self.state.bootstrap_mode = False
            self.state.last_training_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Inicializar autenticación si no estaba activa
            if not self.state.authentication_active:
                logger.info("\n🎯 Inicializando sistema de autenticación...")
                if self._initialize_real_authentication_system():
                    self.state.authentication_active = True
                    self.state.initialization_level = InitializationLevel.FULL_PIPELINE
                    logger.info("✅ Sistema de autenticación activado")
            
            result['success'] = True
            result['message'] = "Entrenamiento completado exitosamente"
            result['details'] = {
                'training_time_seconds': round(training_time, 2),
                'users_used': self.state.users_count,
                'timestamp': self.state.last_training_time,
                'anatomical_trained': True,
                'dynamic_trained': True
            }
            
            logger.info("=" * 80)
            logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            logger.info(f"⏱️  Tiempo total: {training_time:.2f}s")
            logger.info(f"👥 Usuarios utilizados: {self.state.users_count}")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            result['message'] = f"Error durante entrenamiento: {str(e)}"
            logger.error(result['message'], exc_info=True)
            return result
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información detallada de un usuario.
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Dict con información del usuario o None si no existe
        """
        if not self.database:
            return None
        
        try:
            profile = self.database.get_user_profile(user_id)
            if not profile:
                return None
            
            # Obtener templates del usuario
            templates = self.database.get_user_templates(user_id)
            
            return {
                'user_id': profile.user_id,
                'username': profile.username,
                'gesture_sequence': profile.gesture_sequence,
                'total_enrollments': profile.total_enrollments,
                'templates_count': len(templates) if templates else 0,
                'metadata': profile.metadata,
                'created_at': profile.metadata.get('created_at', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error obteniendo info de usuario {user_id}: {e}")
            return None
    
    def delete_user(self, user_id: str) -> Dict[str, Any]:
        """
        Elimina un usuario del sistema.
        
        Args:
            user_id: ID del usuario a eliminar
        
        Returns:
            Dict con resultado de la operación
        """
        result = {
            'success': False,
            'message': ''
        }
        
        try:
            if not self.database:
                result['message'] = "Base de datos no inicializada"
                return result
            
            # Verificar que el usuario existe
            profile = self.database.get_user_profile(user_id)
            if not profile:
                result['message'] = f"Usuario {user_id} no encontrado"
                return result
            
            # Eliminar usuario
            success = self.database.delete_user(user_id)
            
            if success:
                # Actualizar contador
                self.state.users_count -= 1
                
                # Si caemos debajo de 2 usuarios, desactivar autenticación
                if self.state.users_count < 2:
                    self.state.networks_trained = False
                    self.state.authentication_active = False
                    self.state.bootstrap_mode = True
                    logger.info("⚠️ Menos de 2 usuarios - Sistema en modo bootstrap")
                
                result['success'] = True
                result['message'] = f"Usuario {user_id} eliminado correctamente"
                logger.info(result['message'])
            else:
                result['message'] = "Error eliminando usuario"
                logger.error(result['message'])
            
            return result
            
        except Exception as e:
            result['message'] = f"Error: {str(e)}"
            logger.error(f"Error eliminando usuario {user_id}: {e}")
            return result
    
    def cleanup_resources(self):
        """
        Libera recursos del sistema (cámara, mediapipe).
        """
        try:
            logger.info("🧹 Liberando recursos del sistema...")
            
            # Liberar cámara
            try:
                release_camera()
                logger.info("  ✅ Cámara liberada")
            except Exception as e:
                logger.warning(f"  ⚠️ Error liberando cámara: {e}")
            
            # Liberar MediaPipe
            try:
                release_mediapipe()
                logger.info("  ✅ MediaPipe liberado")
            except Exception as e:
                logger.warning(f"  ⚠️ Error liberando MediaPipe: {e}")
            
            logger.info("✅ Recursos liberados")
            
        except Exception as e:
            logger.error(f"Error en cleanup: {e}")
    
    def regenerate_template(self, user_id: str, template_type: str = 'anatomical') -> Dict[str, Any]:
        """
        Regenera un template específico de un usuario.
        Útil después de reentrenar redes.
        
        Args:
            user_id: ID del usuario
            template_type: Tipo de template ('anatomical' o 'dynamic')
        
        Returns:
            Dict con resultado de la operación
        """
        result = {
            'success': False,
            'message': ''
        }
        
        try:
            if not self.database or not self.state.networks_trained:
                result['message'] = "Sistema no listo para regenerar templates"
                return result
            
            # Obtener templates del usuario
            templates = self.database.get_user_templates(user_id, template_type)
            
            if not templates:
                result['message'] = f"No se encontraron templates {template_type} para usuario {user_id}"
                return result
            
            regenerated_count = 0
            
            for template in templates:
                metadata = template.metadata or {}
                
                # REGENERACIÓN ANATÓMICA
                if template_type == 'anatomical':
                    raw_features_list = metadata.get('raw_anatomical_features', [])
                    
                    if raw_features_list and self.anatomical_network.is_trained:
                        try:
                            # Promediar características
                            avg_features = np.mean(raw_features_list, axis=0)
                            
                            if len(avg_features) == self.anatomical_network.input_dim:
                                # Regenerar embedding
                                new_embedding = self.anatomical_network.base_network.predict(
                                    avg_features.reshape(1, -1)
                                )[0]
                                
                                # Actualizar template
                                template.anatomical_embedding = new_embedding
                                self.database._save_template(template)
                                
                                regenerated_count += 1
                                logger.info(f"✅ Template anatómico regenerado para {user_id}")
                        except Exception as e:
                            logger.error(f"Error regenerando template anatómico: {e}")
                
                # REGENERACIÓN DINÁMICA
                elif template_type == 'dynamic':
                    temporal_sequence = metadata.get('temporal_sequence', [])
                    
                    if temporal_sequence and self.dynamic_network.is_trained:
                        try:
                            sequence_array = np.array(temporal_sequence, dtype=np.float32)
                            
                            # Ajustar dimensiones
                            expected_seq_length = self.dynamic_network.sequence_length
                            expected_feature_dim = self.dynamic_network.feature_dim
                            
                            if len(sequence_array.shape) == 2:
                                seq_length, feature_dim = sequence_array.shape
                                
                                # Ajustar longitud
                                if seq_length > expected_seq_length:
                                    sequence_array = sequence_array[:expected_seq_length]
                                elif seq_length < expected_seq_length:
                                    padding = np.zeros((expected_seq_length - seq_length, feature_dim))
                                    sequence_array = np.vstack([sequence_array, padding])
                                
                                # Ajustar features
                                if feature_dim > expected_feature_dim:
                                    sequence_array = sequence_array[:, :expected_feature_dim]
                                elif feature_dim < expected_feature_dim:
                                    padding = np.zeros((sequence_array.shape[0], expected_feature_dim - feature_dim))
                                    sequence_array = np.hstack([sequence_array, padding])
                                
                                # Regenerar embedding
                                sequence_input = sequence_array.reshape(1, expected_seq_length, expected_feature_dim)
                                new_embedding = self.dynamic_network.base_network.predict(sequence_input)[0]
                                
                                # Actualizar template
                                template.dynamic_embedding = new_embedding
                                self.database._save_template(template)
                                
                                regenerated_count += 1
                                logger.info(f"✅ Template dinámico regenerado para {user_id}")
                        except Exception as e:
                            logger.error(f"Error regenerando template dinámico: {e}")
            
            if regenerated_count > 0:
                result['success'] = True
                result['message'] = f"{regenerated_count} template(s) regenerado(s)"
            else:
                result['message'] = "No se pudieron regenerar templates"
            
            return result
            
        except Exception as e:
            result['message'] = f"Error: {str(e)}"
            logger.error(f"Error regenerando templates: {e}")
            return result
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas completas de la base de datos.
        
        Returns:
            Dict con estadísticas
        """
        if not self.database:
            return {}
        
        try:
            stats = self.database.get_database_stats()
            return stats
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}


# ====================================================================
# FUNCIONES GLOBALES
# ====================================================================

_system_manager_instance = None

def get_system_manager() -> BiometricSystemManager:
    """
    Obtiene la instancia global del gestor del sistema.
    
    Returns:
        BiometricSystemManager: Instancia única del gestor (Singleton)
    """
    global _system_manager_instance
    
    if _system_manager_instance is None:
        _system_manager_instance = BiometricSystemManager()
    
    return _system_manager_instance


def initialize_system_on_startup() -> bool:
    """
    Función helper para inicializar el sistema al arrancar FastAPI.
    Llama a initialize_system() del manager global.
    
    Returns:
        bool: True si la inicialización fue exitosa
    """
    manager = get_system_manager()
    return manager.initialize_system()


def get_system_status() -> Dict[str, Any]:
    """
    Obtiene el estado actual del sistema (función helper).
    
    Returns:
        Dict con el estado del sistema
    """
    manager = get_system_manager()
    return manager.get_system_status()


def cleanup_system_resources():
    """
    Limpia recursos del sistema (función helper para shutdown).
    """
    manager = get_system_manager()
    manager.cleanup_resources()