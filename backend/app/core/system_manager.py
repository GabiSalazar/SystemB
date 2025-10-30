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
from app.core.anatomical_features_extractor import get_anatomical_features_extractor
from app.core.dynamic_features_extractor import get_real_dynamic_features_extractor
from app.core.sequence_manager import get_sequence_manager
from app.core.quality_validator import get_quality_validator
from app.core.reference_area_manager import get_reference_area_manager

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
        
        # Componentes auxiliares (Nivel 1)
        self.camera_manager = None
        self.mediapipe_processor = None
        self.quality_validator = None
        self.reference_area_manager = None
        
        # Extractores (Nivel 2)
        self.anatomical_extractor = None
        self.dynamic_extractor = None
        self.sequence_manager = None
        
        print("=" * 80)
        print("🚀 BiometricSystemManager v2.0.0 Real Edition Iniciado")
        print("🏗️  Arquitectura: 15 Módulos + Main | 4 Capas Funcionales")
        print("🎯 Características: Redes siamesas, fusión multimodal, templates biométricos")
        print("=" * 80)
        
        self._initialized = True
    
    def verify_modules(self) -> Tuple[bool, List[str]]:
        """
        Verifica que todos los módulos requeridos estén disponibles.
        Equivalente a verify_notebook_modules() del MAIN.
        
        Returns:
            Tuple[bool, List[str]]: (todos_disponibles, módulos_faltantes)
        """
        missing = []
        
        print("VERIFICANDO MÓDULOS DEL SISTEMA...")
        print("=" * 80)
        
        for module_name, class_name in self.REQUIRED_MODULES.items():
            try:
                # Intentar importar el módulo
                module_path = f"app.core.{module_name}"
                __import__(module_path)
                
                self.state.modules_loaded[class_name] = True
                print(f"✅ {class_name}")
                
            except ImportError as e:
                self.state.modules_loaded[class_name] = False
                missing.append(class_name)
                logger.error(f"❌ {class_name} NO disponible: {e}")
        
        all_available = len(missing) == 0
        
        if all_available:
            print(f"\n✅ TODOS LOS {len(self.REQUIRED_MODULES)} MÓDULOS DISPONIBLES")
        else:
            logger.error(f"\n❌ FALTAN {len(missing)} MÓDULOS:")
            for module in missing:
                logger.error(f"   - {module}")
        
        print("=" * 80)
        return all_available, missing
    
    def initialize_system(self) -> bool:
        """
        Inicializa el sistema completo de forma progresiva.
        Equivalente a initialize_real_progressive() del MAIN.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            print("\n" + "=" * 80)
            print("🔧 INICIANDO INICIALIZACIÓN PROGRESIVA DEL SISTEMA")
            print("=" * 80)
            
            # Paso 0: Verificar módulos
            modules_ok, missing = self.verify_modules()
            if not modules_ok:
                self.state.error_message = f"Módulos faltantes: {', '.join(missing)}"
                logger.error("❌ Error inicializando sistema biométrico")
                logger.error(f"🔍 Detalle del error: {self.state.error_message}")
                return False
            
            # ================================================================
            # NIVEL 1: COMPONENTES BÁSICOS
            # ================================================================
            print("\n🔧 NIVEL 1: Inicializando Componentes Básicos")
            print("-" * 80)
            
            if not self._initialize_real_basic_components():
                self.state.error_message = "Error en Nivel 1 (Componentes Básicos)"
                logger.error("❌ Error inicializando sistema biométrico")
                logger.error(f"🔍 Detalle del error: {self.state.error_message}")
                return False
            
            self.state.initialization_level = InitializationLevel.BASIC_COMPONENTS
            print("✅ NIVEL 1 COMPLETADO: Componentes básicos listos\n")
            
            # ================================================================
            # NIVEL 2: EXTRACTORES DE CARACTERÍSTICAS
            # ================================================================
            print("📊 NIVEL 2: Inicializando Extractores de Características")
            print("-" * 80)
            
            if not self._initialize_real_feature_extractors():
                self.state.error_message = "Error en Nivel 2 (Extractores)"
                logger.error("❌ Error inicializando sistema biométrico")
                logger.error(f"🔍 Detalle del error: {self.state.error_message}")
                return False
            
            self.state.initialization_level = InitializationLevel.FEATURE_EXTRACTION
            self.state.enrollment_active = True  # Enrollment siempre disponible
            print("✅ NIVEL 2 COMPLETADO: Extractores listos\n")
            
            # ================================================================
            # NIVEL 3: REDES NEURONALES
            # ================================================================
            print("🧠 NIVEL 3: Verificando Redes Neuronales")
            print("-" * 80)
            
            networks_trained = self._check_real_networks_trained()
            self.state.networks_trained = networks_trained
            self.state.initialization_level = InitializationLevel.NEURAL_NETWORKS
            
            if networks_trained:
                print("✅ NIVEL 3 COMPLETADO: Redes entrenadas y cargadas\n")
            else:
                logger.warning("⚠️ NIVEL 3 PARCIAL: Redes necesitan entrenamiento")
                print(f"📝 Usuarios actuales: {self.state.users_count}")
                print(f"📝 Mínimo requerido: 2 usuarios para entrenar")
                
                # Activar modo bootstrap
                if self.state.users_count < 2:
                    self.state.bootstrap_mode = True
                    print("🚀 MODO BOOTSTRAP ACTIVADO: Permitir enrollment sin redes\n")
            
            # ================================================================
            # NIVEL 4: PIPELINE COMPLETO
            # ================================================================
            print("🎯 NIVEL 4: Inicializando Pipeline Completo")
            print("-" * 80)
            
            if self._initialize_real_authentication_system():
                self.state.authentication_active = True
                self.state.initialization_level = InitializationLevel.FULL_PIPELINE
                print("✅ NIVEL 4 COMPLETADO: Sistema 100% funcional\n")
            else:
                print("⚠️ NIVEL 4 PARCIAL: Enrollment disponible, autenticación pendiente\n")
            
            # Resumen final
            self._print_initialization_summary()
            
            print("=" * 80)
            print("✅ SISTEMA INICIALIZADO CORRECTAMENTE")
            print("=" * 80)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema biométrico", exc_info=True)
            logger.error(f"🔍 Detalle del error: {str(e)}")
            self.state.error_message = str(e)
            return False
    
    def _initialize_real_basic_components(self) -> bool:
        """
        NIVEL 1: Inicializa componentes básicos.
        Equivalente a _initialize_real_basic_components() del MAIN notebook.
        
        ORDEN CRÍTICO (igual al notebook):
        1. Base de datos
        2. Cámara
        3. MediaPipe
        4. Validadores
        """
        try:
            # ============================================================
            # 1. BASE DE DATOS
            # ============================================================
            print("Inicializando Base de Datos...")
            
            self.database = get_biometric_database()
            
            # Verificar usuarios existentes
            users = self.database.list_users()
            self.state.users_count = len(users)
            self.state.database_ready = True
            
            print(f"✅ Base de datos lista: {self.state.users_count} usuarios registrados")
            
            # Obtener estadísticas
            try:
                db_stats = self.database.get_database_stats()
                total_templates = db_stats.get('total_templates', 0)
                print(f"📊 Templates totales: {total_templates}")
            except:
                pass
            
            # ============================================================
            # 2. CÁMARA (CRÍTICO: Antes de extractores dinámicos)
            # ============================================================
            print("Inicializando Cámara...")
            self.camera_manager = get_camera_manager()
            print("✅ Cámara (instancia global)")
            
            # ============================================================
            # 3. MEDIAPIPE (CRÍTICO: Antes de extractores dinámicos)
            # ============================================================
            print("Inicializando MediaPipe...")
            self.mediapipe_processor = get_mediapipe_processor()
            
            if hasattr(self.mediapipe_processor, 'initialize'):
                if not self.mediapipe_processor.initialize():
                    logger.error("✗ ERROR: No se pudo inicializar MediaPipe")
                    return False
            
            print("✅ MediaPipe")
            
            # ============================================================
            # 4. VALIDADORES (Opcional pero recomendado)
            # ============================================================
            try:
                self.quality_validator = get_quality_validator()
                self.reference_area_manager = get_reference_area_manager()
                print("✅ Validadores de calidad")
            except Exception as e:
                logger.warning(f"⚠️ Validadores no inicializados: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en componentes básicos: {e}", exc_info=True)
            return False
    
    def _initialize_real_feature_extractors(self) -> bool:
        """
        NIVEL 2: Inicializa extractores de características.
        Equivalente a _initialize_real_feature_extractors() del MAIN notebook.
        
        IMPORTANTE: MediaPipe y Camera YA deben estar inicializados del Nivel 1
        
        ORDEN (igual al notebook):
        1. AnatomicalFeaturesExtractor
        2. RealDynamicFeaturesExtractor (requiere MediaPipe)
        3. SequenceManager
        4. EnrollmentSystem
        """
        try:
            # ============================================================
            # 1. ANATOMICAL FEATURES EXTRACTOR
            # ============================================================
            print("  Inicializando AnatomicalFeaturesExtractor...")
            self.anatomical_extractor = get_anatomical_features_extractor()
            print("  ✅ AnatomicalFeaturesExtractor inicializado")
            
            # ============================================================
            # 2. DYNAMIC FEATURES EXTRACTOR (requiere MediaPipe del Nivel 1)
            # ============================================================
            print("  Inicializando RealDynamicFeaturesExtractor...")
            
            # CRÍTICO: Verificar que MediaPipe esté disponible
            if self.mediapipe_processor is None:
                logger.error("ERROR: MediaPipeProcessor no está inicializado antes que DynamicFeaturesExtractor.")
                return False
            
            self.dynamic_extractor = get_real_dynamic_features_extractor()
            print("  ✅ RealDynamicFeaturesExtractor inicializado")
            
            # ============================================================
            # 3. SEQUENCE MANAGER
            # ============================================================
            print("  Inicializando SequenceManager...")
            self.sequence_manager = get_sequence_manager()
            print("  ✅ SequenceManager inicializado")
            
            # ============================================================
            # 4. ENROLLMENT SYSTEM
            # ============================================================
            print("Inicializando Sistema de Enrollment...")
            self.enrollment_system = get_real_enrollment_system()
            
            # Verificar modo bootstrap
            bootstrap = self.enrollment_system.check_bootstrap_mode()
            if bootstrap:
                print("🚀 Modo Bootstrap detectado - Primeros usuarios")
                self.state.bootstrap_mode = True
            
            print("✅ Sistema de enrollment listo")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en extractores: {e}", exc_info=True)
            return False
    
    def _check_real_networks_trained(self) -> bool:
        """
        NIVEL 3: Verifica si las redes están entrenadas.
        Equivalente a _check_real_networks_trained() del MAIN.
        """
        try:
            print("Verificando estado de redes neuronales...")
            
            # Obtener instancias de las redes
            self.anatomical_network = get_real_siamese_anatomical_network()
            self.dynamic_network = get_real_siamese_dynamic_network()
            
            # Verificar si están entrenadas
            anatomical_trained = self.anatomical_network.is_trained
            dynamic_trained = self.dynamic_network.is_trained
            
            print(f"  Red anatómica: {'✅ Entrenada' if anatomical_trained else '⚠️ No entrenada'}")
            print(f"  Red dinámica: {'✅ Entrenada' if dynamic_trained else '⚠️ No entrenada'}")
            
            both_trained = anatomical_trained and dynamic_trained
            
            if both_trained:
                print("✅ Ambas redes están entrenadas y listas")
            else:
                logger.warning("⚠️ Las redes necesitan entrenamiento")
                print(f"📝 Se requieren al menos 2 usuarios para entrenar")
            
            return both_trained
            
        except Exception as e:
            logger.error(f"❌ Error verificando redes: {e}", exc_info=True)
            return False
    
    def _initialize_real_authentication_system(self) -> bool:
        """
        NIVEL 4: Inicializa sistema de autenticación.
        Equivalente a _initialize_real_authentication_system() del MAIN.
        """
        try:
            if not self.state.networks_trained:
                logger.warning("⚠️ Redes no entrenadas - Autenticación no disponible aún")
                return False
            
            print("Inicializando Sistema de Autenticación...")
            self.authentication_system = get_real_authentication_system()
            
            print("✅ Sistema de autenticación listo")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en sistema de autenticación: {e}", exc_info=True)
            return False
    
    def _print_initialization_summary(self):
        """Imprime resumen de la inicialización."""
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE INICIALIZACIÓN")
        print("=" * 80)
        print(f"  🎯 Nivel alcanzado: {self.state.initialization_level.name}")
        print(f"  👥 Usuarios registrados: {self.state.users_count}")
        print(f"  🧠 Redes entrenadas: {'✅ Sí' if self.state.networks_trained else '⚠️ No'}")
        print(f"  📝 Enrollment: {'✅ Activo' if self.state.enrollment_active else '❌ Inactivo'}")
        print(f"  🔐 Autenticación: {'✅ Activa' if self.state.authentication_active else '❌ Inactiva'}")
        print(f"  🚀 Bootstrap: {'✅ Activo' if self.state.bootstrap_mode else '❌ Inactivo'}")
        print("=" * 80 + "\n")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema.
        
        Returns:
            Dict con información completa del estado
        """
        uptime = time.time() - self.start_time
        
        return {
            'status': 'operational' if self.state.initialization_level == InitializationLevel.FULL_PIPELINE else 'partial',
            'initialization_level': self.state.initialization_level.name,
            'initialization_level_value': self.state.initialization_level.value,
            'users_count': self.state.users_count,
            'networks_trained': self.state.networks_trained,
            'database_ready': self.state.database_ready,
            'enrollment_active': self.state.enrollment_active,
            'authentication_active': self.state.authentication_active,
            'bootstrap_mode': self.state.bootstrap_mode,
            'error_message': self.state.error_message,
            'uptime_seconds': uptime,
            'modules_loaded': self.state.modules_loaded,
            'statistics': {
                'total_enrollments': self.state.total_enrollments,
                'total_authentications': self.state.total_authentications,
                'total_verifications': self.state.total_verifications,
                'total_identifications': self.state.total_identifications
            }
        }
    
    def train_networks(self, force: bool = False) -> Dict[str, Any]:
        """
        Entrena o reentrena las redes neuronales.
        CORREGIDO: Manejo de RealTrainingHistory objects
        
        Args:
            force: Si True, fuerza reentrenamiento incluso si ya están entrenadas
        
        Returns:
            Dict con resultado del entrenamiento
        """
        result = {
            'success': False,
            'message': '',
            'anatomical_trained': False,
            'dynamic_trained': False
        }
        
        try:
            # Verificar que haya suficientes usuarios
            if self.state.users_count < 2:
                result['message'] = f"Se requieren al menos 2 usuarios. Actualmente: {self.state.users_count}"
                logger.warning(result['message'])
                return result
            
            print("=" * 80)
            print("INICIANDO ENTRENAMIENTO DE REDES NEURONALES")
            print("=" * 80)
            
            # Verificar si ya están entrenadas
            if self.state.networks_trained and not force:
                print("Las redes ya están entrenadas")
                print("Usa force=True para reentrenar")
                result['success'] = True
                result['message'] = "Redes ya entrenadas (usa force=True para reentrenar)"
                result['anatomical_trained'] = self.anatomical_network.is_trained
                result['dynamic_trained'] = self.dynamic_network.is_trained
                return result
            
            # ============================================================
            # ENTRENAR RED ANATÓMICA
            # ============================================================
            print("\nEntrenando Red Siamesa Anatómica...")
            print("-" * 80)
            
            anatomical_result = self.anatomical_network.train_with_real_data(self.database)
            
            # CORREGIDO: Verificar tipo de resultado
            if hasattr(anatomical_result, '__dict__'):
                # Es un objeto RealTrainingHistory
                anatomical_success = getattr(anatomical_result, 'success', True)
                anatomical_message = getattr(anatomical_result, 'message', 'Training completed')
            elif isinstance(anatomical_result, dict):
                # Ya es un diccionario
                anatomical_success = anatomical_result.get('success', False)
                anatomical_message = anatomical_result.get('message', 'Unknown result')
            else:
                # Resultado desconocido, asumimos éxito si no hay excepción
                anatomical_success = True
                anatomical_message = 'Training completed'
            
            if anatomical_success:
                print("Red anatómica entrenada exitosamente")
                result['anatomical_trained'] = True
            else:
                logger.error(f"Error entrenando red anatómica: {anatomical_message}")
                result['message'] = f"Error en red anatómica: {anatomical_message}"
                return result
            
            # ============================================================
            # ENTRENAR RED DINÁMICA
            # ============================================================
            print("\nEntrenando Red Siamesa Dinámica...")
            print("-" * 80)
            
            dynamic_result = self.dynamic_network.train_with_real_data(self.database)
            
            # CORREGIDO: Verificar tipo de resultado
            if hasattr(dynamic_result, '__dict__'):
                # Es un objeto RealTrainingHistory
                dynamic_success = getattr(dynamic_result, 'success', True)
                dynamic_message = getattr(dynamic_result, 'message', 'Training completed')
            elif isinstance(dynamic_result, dict):
                # Ya es un diccionario
                dynamic_success = dynamic_result.get('success', False)
                dynamic_message = dynamic_result.get('message', 'Unknown result')
            else:
                # Resultado desconocido, asumimos éxito si no hay excepción
                dynamic_success = True
                dynamic_message = 'Training completed'
            
            if dynamic_success:
                print("Red dinámica entrenada exitosamente")
                result['dynamic_trained'] = True
            else:
                logger.error(f"Error entrenando red dinámica: {dynamic_message}")
                result['message'] = f"Error en red dinámica: {dynamic_message}"
                return result
            
            # ============================================================
            # RESULTADO FINAL
            # ============================================================
            both_trained = result['anatomical_trained'] and result['dynamic_trained']
            
            if both_trained:
                self.state.networks_trained = True
                self.state.last_training_time = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Inicializar autenticación si no estaba activa
                if not self.state.authentication_active:
                    try:
                        if self._initialize_real_authentication_system():
                            self.state.authentication_active = True
                            self.state.initialization_level = InitializationLevel.FULL_PIPELINE
                            print("Sistema de autenticación activado")
                    except Exception as auth_error:
                        logger.warning(f"No se pudo activar autenticación: {auth_error}")
                        # No es crítico, las redes están entrenadas
                
                result['success'] = True
                result['message'] = "Ambas redes entrenadas exitosamente"
                print("\nENTRENAMIENTO COMPLETADO EXITOSAMENTE")
                print(f"   - Red Anatómica: Entrenada")
                print(f"   - Red Dinámica: Entrenada")
                print(f"   - Usuarios entrenados: {self.state.users_count}")
                print(f"   - Autenticación: {'Activa' if self.state.authentication_active else 'Inactiva'}")
            else:
                result['message'] = "Entrenamiento parcial o fallido"
                logger.warning("\nENTRENAMIENTO INCOMPLETO")
            
            print("=" * 80)
            
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            result['message'] = f"Error durante entrenamiento: {str(e)}"
            logger.error(f"Error en entrenamiento: {e}", exc_info=True)
            print(f"\nERROR EN ENTRENAMIENTO:")
            print(error_trace)
            return result
    
    def cleanup_resources(self):
        """
        Limpia recursos del sistema (cámara, MediaPipe, etc).
        """
        try:
            print("\n🧹 Limpiando recursos del sistema...")
            
            # Liberar cámara
            try:
                release_camera()
                print("  ✅ Cámara liberada")
            except Exception as e:
                logger.warning(f"  ⚠️ Error liberando cámara: {e}")
            
            # Liberar MediaPipe
            try:
                release_mediapipe()
                print("  ✅ MediaPipe liberado")
            except Exception as e:
                logger.warning(f"  ⚠️ Error liberando MediaPipe: {e}")
            
            print("✅ Recursos liberados")
            
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
                                print(f"✅ Template anatómico regenerado para {user_id}")
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
                                print(f"✅ Template dinámico regenerado para {user_id}")
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
    # MÉTODOS DE ENROLLMENT (WRAPPERS PARA LA API)
    # ====================================================================

    def start_enrollment_session(self, user_id: str, username: str, 
                                gesture_sequence: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Inicia sesión de enrollment (wrapper para la API).
        
        Args:
            user_id: ID del usuario
            username: Nombre del usuario
            gesture_sequence: Secuencia de gestos (opcional)
        
        Returns:
            Dict con información de la sesión
        """
        try:
            if not self.enrollment_system:
                return {
                    'success': False,
                    'message': 'Sistema de enrollment no disponible'
                }
            
            # Usar secuencia por defecto si no se proporciona
            if not gesture_sequence:
                gesture_sequence = ["thumbs_up", "peace", "ok"]
            
            # Iniciar enrollment
            session_id = self.enrollment_system.start_real_enrollment(
                user_id=user_id,
                username=username,
                gesture_sequence=gesture_sequence,
                progress_callback=None,
                error_callback=None
            )
            
            # Obtener información de la sesión
            session = self.enrollment_system.active_sessions.get(session_id)
            
            if not session:
                return {
                    'success': False,
                    'message': 'Error obteniendo sesión'
                }
            
            return {
                'success': True,
                'message': 'Sesión iniciada correctamente',
                'session': {
                    'session_id': session_id,
                    'user_id': user_id,
                    'username': username,
                    'gesture_sequence': gesture_sequence,
                    'total_gestures': len(gesture_sequence),
                    'samples_per_gesture': self.enrollment_system.config.samples_per_gesture,
                    'total_samples_needed': len(gesture_sequence) * self.enrollment_system.config.samples_per_gesture,
                    'bootstrap_mode': self.enrollment_system.bootstrap_mode
                }
            }
            
        except Exception as e:
            print(f"Error iniciando enrollment session: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }

    def process_enrollment_frame(self, session_id: str, frame: np.ndarray, 
                                current_gesture_index: int) -> Dict[str, Any]:
        """
        Procesa un frame de enrollment (wrapper para la API).
        
        Args:
            session_id: ID de la sesión
            frame: Frame de la cámara (numpy array BGR)
            current_gesture_index: Índice del gesto actual
        
        Returns:
            Dict con resultado del procesamiento
        """
        try:
            if not self.enrollment_system:
                return {
                    'success': False,
                    'message': 'Sistema de enrollment no disponible'
                }
            
            # Procesar frame usando el método correcto
            result = self.enrollment_system.process_enrollment_frame_with_image(
                session_id=session_id,
                frame_image=frame
            )
            
            # Adaptar respuesta al formato esperado por la API
            session = self.enrollment_system.active_sessions.get(session_id)
            
            if not session:
                return {
                    'success': False,
                    'message': 'Sesión no encontrada',
                    'error': 'Session not found'
                }
            
            samples_this_gesture = len([
                s for s in session.samples 
                if s.gesture_name == session.current_gesture
            ])
            
            gesture_completed = samples_this_gesture >= self.enrollment_system.config.samples_per_gesture
            all_completed = session.status.value == 'completed'
            
            return {
                'success': result.get('sample_captured', False) or True,
                'message': result.get('message', 'Frame procesado'),
                'current_gesture': session.current_gesture,
                'current_gesture_index': session.current_gesture_index,
                'samples_captured': samples_this_gesture,
                'samples_needed': self.enrollment_system.config.samples_per_gesture,
                'gesture_completed': gesture_completed,
                'all_gestures_completed': all_completed,
                'quality_score': result.get('quality_score'),
                'feedback': result.get('message'),
                'error': result.get('error')
            }
            
        except Exception as e:
            print(f"Error procesando frame: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'error': str(e)
            }

    def get_enrollment_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Obtiene estado de una sesión de enrollment.
        
        Args:
            session_id: ID de la sesión
        
        Returns:
            Dict con estado de la sesión
        """
        try:
            if not self.enrollment_system:
                return {
                    'success': False,
                    'message': 'Sistema de enrollment no disponible'
                }
            
            session = self.enrollment_system.active_sessions.get(session_id)
            
            if not session:
                return {
                    'success': False,
                    'message': 'Sesión no encontrada'
                }
            
            samples_this_gesture = len([
                s for s in session.samples 
                if s.gesture_name == session.current_gesture
            ])
            
            return {
                'success': True,
                'message': 'Estado obtenido',
                'session': {
                    'active': session.status.value in ['in_progress', 'collecting_samples'],
                    'user_id': session.user_id,
                    'username': session.username,
                    'current_gesture': session.current_gesture,
                    'current_gesture_index': session.current_gesture_index,
                    'total_gestures': len(session.gesture_sequence),
                    'samples_captured': samples_this_gesture,
                    'samples_needed': self.enrollment_system.config.samples_per_gesture,
                    'progress_percentage': session.progress_percentage
                }
            }
            
        except Exception as e:
            print(f"Error obteniendo estado: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }

    def complete_enrollment_session(self, session_id: str) -> Dict[str, Any]:
        """
        Completa una sesión de enrollment.
        
        Args:
            session_id: ID de la sesión
        
        Returns:
            Dict con resultado
        """
        try:
            if not self.enrollment_system:
                return {
                    'success': False,
                    'message': 'Sistema de enrollment no disponible'
                }
            
            session = self.enrollment_system.active_sessions.get(session_id)
            
            if not session:
                return {
                    'success': False,
                    'message': 'Sesión no encontrada'
                }
            
            # Finalizar enrollment
            self.enrollment_system.workflow._finalize_real_enrollment(session)
            
            return {
                'success': True,
                'message': 'Enrollment completado',
                'user_id': session.user_id,
                'username': session.username,
                'templates_created': len(session.samples),
                'enrollment_time': session.duration
            }
            
        except Exception as e:
            print(f"Error completando enrollment: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }

    def cancel_enrollment_session(self, session_id: str) -> Dict[str, Any]:
        """
        Cancela una sesión de enrollment.
        
        Args:
            session_id: ID de la sesión
        
        Returns:
            Dict con resultado
        """
        try:
            if not self.enrollment_system:
                return {
                    'success': False,
                    'message': 'Sistema de enrollment no disponible'
                }
            
            success = self.enrollment_system.cancel_enrollment(session_id)
            
            return {
                'success': success,
                'message': 'Sesión cancelada' if success else 'Error cancelando sesión'
            }
            
        except Exception as e:
            print(f"Error cancelando enrollment: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }

    def list_enrollment_sessions(self) -> List[Dict[str, Any]]:
        """Lista todas las sesiones activas."""
        try:
            if not self.enrollment_system:
                return []
            
            sessions = []
            for session_id, session in self.enrollment_system.active_sessions.items():
                sessions.append({
                    'session_id': session_id,
                    'user_id': session.user_id,
                    'username': session.username,
                    'status': session.status.value,
                    'progress': session.progress_percentage
                })
            
            return sessions
            
        except Exception as e:
            print(f"Error listando sesiones: {e}")
            return []

    def get_available_gestures(self) -> List[str]:
        """Obtiene lista de gestos disponibles."""
        return ["thumbs_up", "peace", "ok", "fist", "palm"]

    def get_enrollment_config(self) -> Dict[str, Any]:
        """Obtiene configuración de enrollment."""
        try:
            if not self.enrollment_system:
                return {}
            
            return {
                'samples_per_gesture': self.enrollment_system.config.samples_per_gesture,
                'quality_threshold': self.enrollment_system.config.quality_threshold,
                'min_confidence': self.enrollment_system.config.min_confidence,
                'bootstrap_mode': self.enrollment_system.bootstrap_mode
            }
            
        except Exception as e:
            print(f"Error obteniendo config: {e}")
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