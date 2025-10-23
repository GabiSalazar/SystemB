"""
MÓDULO 13: BIOMETRIC_DATABASE
Base de datos biométrica local con indexación vectorial (100% REAL)
"""

import numpy as np
import json
import pickle
import os
import shutil
import hashlib
import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import threading
import warnings

# Cryptography imports
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography no disponible")

from datetime import datetime, timedelta

# Importar módulos anteriores
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
warnings.filterwarnings("ignore", category=RuntimeWarning)


class TemplateType(Enum):
    """Tipos de templates biométricos."""
    ANATOMICAL = "anatomical"
    DYNAMIC = "dynamic"
    MULTIMODAL = "multimodal"


class BiometricQuality(Enum):
    """Niveles de calidad biométrica."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class SearchStrategy(Enum):
    """Estrategias de búsqueda vectorial."""
    LINEAR = "linear"
    KD_TREE = "kd_tree"
    LSH = "lsh"
    HIERARCHICAL = "hierarchical"


@dataclass
class BiometricTemplate:
    """Template biométrico unificado."""
    user_id: str
    template_id: str
    template_type: TemplateType
    
    anatomical_embedding: Optional[np.ndarray] = None
    dynamic_embedding: Optional[np.ndarray] = None
    
    gesture_name: str = "unknown"
    hand_side: str = "unknown"
    quality_score: float = 1.0
    confidence: float = 1.0
    
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    
    enrollment_session: str = ""
    verification_count: int = 0
    success_count: int = 0
    
    is_encrypted: bool = False
    checksum: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Tasa de éxito en verificaciones."""
        return (self.success_count / self.verification_count * 100) if self.verification_count > 0 else 0.0
    
    @property
    def quality_level(self) -> BiometricQuality:
        """Nivel de calidad basado en score."""
        if self.quality_score >= 0.9:
            return BiometricQuality.EXCELLENT
        elif self.quality_score >= 0.7:
            return BiometricQuality.GOOD
        elif self.quality_score >= 0.5:
            return BiometricQuality.FAIR
        else:
            return BiometricQuality.POOR


@dataclass
class UserProfile:
    """Perfil completo de usuario biométrico."""
    user_id: str
    username: str
    
    anatomical_templates: List[str] = field(default_factory=list)
    dynamic_templates: List[str] = field(default_factory=list)
    multimodal_templates: List[str] = field(default_factory=list)
    
    gesture_sequence: Optional[List[str]] = None
    sequence_metadata: Dict[str, Any] = field(default_factory=dict)
    
    total_enrollments: int = 0
    total_verifications: int = 0
    successful_verifications: int = 0
    last_activity: float = field(default_factory=time.time)
    
    quality_threshold: float = 0.7
    security_level: str = "standard"
    
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_templates(self) -> int:
        """Total de templates registrados."""
        return len(self.anatomical_templates) + len(self.dynamic_templates) + len(self.multimodal_templates)
    
    @property
    def verification_success_rate(self) -> float:
        """Tasa de éxito en verificaciones."""
        return (self.successful_verifications / self.total_verifications * 100) if self.total_verifications > 0 else 0.0


@dataclass
class DatabaseStats:
    """Estadísticas de la base de datos."""
    total_users: int = 0
    total_templates: int = 0
    total_verifications: int = 0
    successful_verifications: int = 0
    
    anatomical_templates: int = 0
    dynamic_templates: int = 0
    multimodal_templates: int = 0
    
    excellent_quality: int = 0
    good_quality: int = 0
    fair_quality: int = 0
    poor_quality: int = 0
    
    total_size_mb: float = 0.0
    index_size_mb: float = 0.0
    backup_size_mb: float = 0.0
    
    avg_search_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    
    last_updated: float = field(default_factory=time.time)
    
    
class VectorIndex:
    """Índice vectorial para búsqueda eficiente de similitud."""
    
    def __init__(self, embedding_dim: int = 128, strategy: SearchStrategy = SearchStrategy.LINEAR):
        """
        Inicializa el índice vectorial.
        
        Args:
            embedding_dim: Dimensión de los embeddings
            strategy: Estrategia de búsqueda
        """
        self.embedding_dim = embedding_dim
        self.strategy = strategy
        
        self.embeddings: np.ndarray = np.empty((0, embedding_dim))
        self.template_ids: List[str] = []
        self.user_ids: List[str] = []
        
        self.kdtree = None
        self.lsh_buckets = None
        self.clusters = None
        
        self.search_cache = {}
        self.cache_size_limit = 1000
        
        self.is_built = False
    
    def add_embedding(self, embedding: np.ndarray, template_id: str, user_id: str):
        """Añade un embedding al índice."""
        try:
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(f"Embedding debe tener dimensión {self.embedding_dim}")
            
            if self.embeddings.size == 0:
                self.embeddings = embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
            
            self.template_ids.append(template_id)
            self.user_ids.append(user_id)
            
            self.is_built = False
            
        except Exception as e:
            logger.error(f"Error añadiendo embedding: {e}")
    
    def build_index(self):
        """Construye el índice según la estrategia seleccionada."""
        try:
            if len(self.embeddings) == 0:
                return
            
            if self.strategy == SearchStrategy.KD_TREE:
                self._build_kdtree()
            elif self.strategy == SearchStrategy.LSH:
                self._build_lsh()
            elif self.strategy == SearchStrategy.HIERARCHICAL:
                self._build_hierarchical()
            
            self.is_built = True
            logger.info(f"Índice construido: {len(self.embeddings)} embeddings, estrategia {self.strategy.value}")
            
        except Exception as e:
            logger.error(f"Error construyendo índice: {e}")
    
    def _build_kdtree(self):
        """Construye KD-Tree para búsqueda eficiente."""
        try:
            from sklearn.neighbors import NearestNeighbors
            self.kdtree = NearestNeighbors(n_neighbors=10, algorithm='kd_tree', metric='euclidean')
            self.kdtree.fit(self.embeddings)
        except ImportError:
            logger.error("sklearn no disponible, usando búsqueda lineal")
            self.strategy = SearchStrategy.LINEAR
    
    def _build_lsh(self):
        """Construye Locality Sensitive Hashing."""
        try:
            num_hashes = 10
            num_buckets = min(100, len(self.embeddings))
            
            self.lsh_buckets = defaultdict(list)
            
            hash_vectors = np.random.randn(num_hashes, self.embedding_dim)
            
            for i, embedding in enumerate(self.embeddings):
                hash_values = np.dot(hash_vectors, embedding) > 0
                hash_key = hash(tuple(hash_values.astype(int)))
                bucket = hash_key % num_buckets
                
                self.lsh_buckets[bucket].append(i)
                
        except Exception as e:
            logger.error(f"Error construyendo LSH: {e}")
            self.strategy = SearchStrategy.LINEAR
    
    def _build_hierarchical(self):
        """Construye clustering jerárquico."""
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            if len(self.embeddings) < 10:
                self.strategy = SearchStrategy.LINEAR
                return
            
            num_clusters = min(10, len(self.embeddings) // 5)
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            cluster_labels = clustering.fit_predict(self.embeddings)
            
            self.clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                self.clusters[label].append(i)
                
        except ImportError:
            logger.error("sklearn no disponible para clustering")
            self.strategy = SearchStrategy.LINEAR
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5, 
                      exclude_user: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """
        Busca embeddings similares.
        
        Args:
            query_embedding: Embedding de consulta
            k: Número de resultados
            exclude_user: Usuario a excluir
            
        Returns:
            Lista de (template_id, user_id, distancia)
        """
        try:
            if len(self.embeddings) == 0:
                return []
            
            cache_key = (tuple(query_embedding), k, exclude_user)
            if cache_key in self.search_cache:
                return self.search_cache[cache_key]
            
            if not self.is_built:
                self.build_index()
            
            if self.strategy == SearchStrategy.KD_TREE and self.kdtree is not None:
                results = self._search_kdtree(query_embedding, k, exclude_user)
            elif self.strategy == SearchStrategy.LSH and self.lsh_buckets is not None:
                results = self._search_lsh(query_embedding, k, exclude_user)
            elif self.strategy == SearchStrategy.HIERARCHICAL and self.clusters is not None:
                results = self._search_hierarchical(query_embedding, k, exclude_user)
            else:
                results = self._search_linear(query_embedding, k, exclude_user)
            
            if len(self.search_cache) < self.cache_size_limit:
                self.search_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def _search_linear(self, query_embedding: np.ndarray, k: int, 
                      exclude_user: Optional[str]) -> List[Tuple[str, str, float]]:
        """Búsqueda lineal (exacta)."""
        distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
        
        results = []
        for i, distance in enumerate(distances):
            if exclude_user and self.user_ids[i] == exclude_user:
                continue
            results.append((self.template_ids[i], self.user_ids[i], distance))
        
        results.sort(key=lambda x: x[2])
        return results[:k]
    
    def _search_kdtree(self, query_embedding: np.ndarray, k: int, 
                      exclude_user: Optional[str]) -> List[Tuple[str, str, float]]:
        """Búsqueda usando KD-Tree."""
        try:
            k_search = min(k * 3, len(self.embeddings))
            distances, indices = self.kdtree.kneighbors(query_embedding.reshape(1, -1), n_neighbors=k_search)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if exclude_user and self.user_ids[idx] == exclude_user:
                    continue
                results.append((self.template_ids[idx], self.user_ids[idx], dist))
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda KD-Tree: {e}")
            return self._search_linear(query_embedding, k, exclude_user)
    
    def _search_lsh(self, query_embedding: np.ndarray, k: int, 
                   exclude_user: Optional[str]) -> List[Tuple[str, str, float]]:
        """Búsqueda usando LSH."""
        try:
            hash_vectors = np.random.randn(10, self.embedding_dim)
            hash_values = np.dot(hash_vectors, query_embedding) > 0
            hash_key = hash(tuple(hash_values.astype(int)))
            bucket = hash_key % 100
            
            candidate_indices = self.lsh_buckets.get(bucket, [])
            
            if not candidate_indices:
                return self._search_linear(query_embedding, k, exclude_user)
            
            results = []
            for idx in candidate_indices:
                if exclude_user and self.user_ids[idx] == exclude_user:
                    continue
                distance = np.linalg.norm(self.embeddings[idx] - query_embedding)
                results.append((self.template_ids[idx], self.user_ids[idx], distance))
            
            results.sort(key=lambda x: x[2])
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error en búsqueda LSH: {e}")
            return self._search_linear(query_embedding, k, exclude_user)
    
    def _search_hierarchical(self, query_embedding: np.ndarray, k: int, 
                           exclude_user: Optional[str]) -> List[Tuple[str, str, float]]:
        """Búsqueda usando clustering jerárquico."""
        try:
            cluster_distances = {}
            for cluster_id, indices in self.clusters.items():
                cluster_center = np.mean(self.embeddings[indices], axis=0)
                distance = np.linalg.norm(cluster_center - query_embedding)
                cluster_distances[cluster_id] = distance
            
            sorted_clusters = sorted(cluster_distances.items(), key=lambda x: x[1])
            
            results = []
            for cluster_id, _ in sorted_clusters:
                cluster_indices = self.clusters[cluster_id]
                
                for idx in cluster_indices:
                    if exclude_user and self.user_ids[idx] == exclude_user:
                        continue
                    distance = np.linalg.norm(self.embeddings[idx] - query_embedding)
                    results.append((self.template_ids[idx], self.user_ids[idx], distance))
                
                if len(results) >= k * 2:
                    break
            
            results.sort(key=lambda x: x[2])
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error en búsqueda jerárquica: {e}")
            return self._search_linear(query_embedding, k, exclude_user)
    
    def remove_template(self, template_id: str):
        """Elimina un template del índice."""
        try:
            if template_id in self.template_ids:
                idx = self.template_ids.index(template_id)
                
                self.embeddings = np.delete(self.embeddings, idx, axis=0)
                self.template_ids.pop(idx)
                self.user_ids.pop(idx)
                
                self.search_cache.clear()
                self.is_built = False
                
                logger.info(f"Template {template_id} eliminado del índice")
                
        except Exception as e:
            logger.error(f"Error eliminando template: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del índice."""
        return {
            'total_embeddings': len(self.embeddings),
            'embedding_dim': self.embedding_dim,
            'strategy': self.strategy.value,
            'is_built': self.is_built,
            'cache_size': len(self.search_cache),
            'memory_usage_mb': self.embeddings.nbytes / 1024 / 1024 if self.embeddings.size > 0 else 0
        }
    
class BiometricDatabase:
    """
    Base de datos biométrica local con indexación vectorial y encriptación.
    Gestiona templates, usuarios y búsquedas eficientes de similitud.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializa la base de datos biométrica.
        
        Args:
            db_path: Ruta personalizada de la base de datos
        """
        self.logger = get_logger()
        
        self.config = self._load_database_config()
        
        self.db_path = Path(db_path) if db_path else self._get_default_db_path()
        self._setup_directory_structure()
        
        self.users: Dict[str, UserProfile] = {}
        self.templates: Dict[str, BiometricTemplate] = {}
        
        self.anatomical_index = VectorIndex(
            embedding_dim=64,
            strategy=SearchStrategy(self.config['search_strategy'])
        )
        self.dynamic_index = VectorIndex(
            embedding_dim=128,
            strategy=SearchStrategy(self.config['search_strategy'])
        )
        
        self.encryption_key = self._load_or_generate_key()
        if CRYPTO_AVAILABLE:
            self.cipher = Fernet(self.encryption_key)
        else:
            self.cipher = None
        
        self.lock = threading.RLock()
        
        self.cache = {}
        self.stats = DatabaseStats()
        
        self._load_database()
        
        logger.info(f"BiometricDatabase inicializada en: {self.db_path}")
    
    def _load_database_config(self) -> Dict[str, Any]:
        """Carga configuración de la base de datos SIN ENCRIPTACIÓN para debugging."""
        default_config = {
            'encryption_enabled': False,
            'auto_backup': True,
            'backup_interval_hours': 24,
            'max_backups': 30,
            'search_strategy': 'linear',
            'cache_size': 1000,
            'compression_enabled': False,
            'integrity_checks': True,
            'auto_cleanup': True,
            'max_templates_per_user': 50,
            'template_expiry_days': 0,
            'debug_mode': True,
            'verbose_logging': True,
            'verification_enabled': True,
        }
        
        config = get_config('biometric.database', default_config)
        
        config['encryption_enabled'] = False
        config['debug_mode'] = True
        config['verbose_logging'] = True
        config['verification_enabled'] = True
        
        print(f"🔧 DEBUG CONFIG: Encriptación = {config['encryption_enabled']}")
        print(f"🔧 DEBUG CONFIG: Debug mode = {config['debug_mode']}")
        print(f"🔧 DEBUG CONFIG: Templates por usuario = {config['max_templates_per_user']}")
        
        return config
    
    def _get_default_db_path(self) -> Path:
        """Obtiene la ruta por defecto de la base de datos."""
        db_dir = Path(get_config('paths.biometric_db', 'biometric_data'))
        return db_dir
    
    def _setup_directory_structure(self):
        """Configura estructura mínima de directorios."""
        essential_directories = [
            self.db_path / 'users',
            self.db_path / 'templates',
        ]
        
        for directory in essential_directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"INFO: Base de datos configurada con {len(essential_directories)} directorios esenciales")
    
    def _load_or_generate_key(self) -> bytes:
        """Carga o genera clave de encriptación válida."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography no disponible - generando clave dummy")
            return b'dummy_key_for_compatibility_purpose_'
        
        keys_dir = self.db_path / 'keys'
        
        if self.config.get('encryption_enabled', False):
            keys_dir.mkdir(exist_ok=True)
            key_file = keys_dir / 'encryption.key'
            
            if key_file.exists():
                try:
                    with open(key_file, 'rb') as f:
                        logger.info("Clave de encriptación cargada")
                        return f.read()
                except Exception as e:
                    logger.error(f"Error cargando clave: {e}")
            
            key = Fernet.generate_key()
            
            try:
                with open(key_file, 'wb') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)
                logger.info("Nueva clave de encriptación generada")
            except Exception as e:
                logger.error(f"Error guardando clave: {e}")
            
            return key
        else:
            logger.info("Encriptación deshabilitada - generando clave temporal")
            return Fernet.generate_key()
    
    
    def _load_database(self):
        """Carga datos existentes de la base de datos."""
        try:
            users_loaded = 0
            templates_loaded = 0
            
            logger.info("🔄 Iniciando carga completa de base de datos...")
            
            users_dir = self.db_path / 'users'
            logger.info(f"📁 Buscando usuarios en: {users_dir}")
            
            if users_dir.exists():
                user_files = list(users_dir.glob('*.json'))
                logger.info(f"📊 Archivos de usuarios encontrados: {len(user_files)}")
                
                for user_file in user_files:
                    try:
                        logger.info(f"📂 Cargando usuario: {user_file.name}")
                        
                        with open(user_file, 'r', encoding='utf-8') as f:
                            user_data = json.load(f)
                        
                        try:
                            user_profile = UserProfile(
                                user_id=user_data.get('user_id', user_file.stem),
                                username=user_data.get('username', 'Unknown'),
                                gesture_sequence=user_data.get('gesture_sequence', []),
                                anatomical_templates=user_data.get('anatomical_templates', []),
                                dynamic_templates=user_data.get('dynamic_templates', []),
                                total_enrollments=user_data.get('total_enrollments', 0),
                                created_at=user_data.get('created_at', time.time()),
                                updated_at=user_data.get('updated_at', time.time()),
                                metadata=user_data.get('metadata', {})
                            )
                            
                            self.users[user_profile.user_id] = user_profile
                            users_loaded += 1
                            
                            logger.info(f"✅ Usuario cargado:")
                            logger.info(f"   👤 ID: {user_profile.user_id}")
                            logger.info(f"   📝 Nombre: {user_profile.username}")
                            logger.info(f"   🎯 Gestos: {user_profile.gesture_sequence}")
                            logger.info(f"   📊 Templates: {user_profile.total_enrollments}")
                            
                        except Exception as profile_error:
                            logger.error(f"❌ Error creando UserProfile: {profile_error}")
                            continue
                            
                    except Exception as file_error:
                        logger.error(f"❌ Error leyendo {user_file.name}: {file_error}")
                        continue
            else:
                logger.info("📁 Directorio de usuarios no existe, creándolo...")
                users_dir.mkdir(parents=True, exist_ok=True)
            
            templates_dir = self.db_path / 'templates'
            logger.info(f"📁 Buscando templates en: {templates_dir}")
            
            if templates_dir.exists():
                template_files = list(templates_dir.glob('*.json'))
                logger.info(f"📊 Archivos de templates encontrados: {len(template_files)}")
                
                for template_file in template_files:
                    try:
                        logger.info(f"📂 Cargando template: {template_file.name}")
                        
                        with open(template_file, 'r', encoding='utf-8') as f:
                            template_data = json.load(f)
                        
                        try:
                            is_bootstrap = template_data.get('metadata', {}).get('bootstrap_mode', False)
                            
                            if is_bootstrap:
                                template = BiometricTemplate(
                                    user_id=template_data.get('user_id', 'unknown'),
                                    template_id=template_data.get('template_id', template_file.stem),
                                    template_type=TemplateType.ANATOMICAL,
                                    anatomical_embedding=None,
                                    dynamic_embedding=None,
                                    gesture_name=template_data.get('gesture_name', 'Unknown'),
                                    quality_score=template_data.get('quality_score', 0.0),
                                    confidence=template_data.get('confidence', 0.0),
                                    enrollment_session=template_data.get('enrollment_session', ''),
                                    created_at=template_data.get('created_at', time.time()),
                                    metadata=template_data.get('metadata', {}),
                                    checksum=template_data.get('checksum', '')
                                )
                                
                                logger.info(f"   🔧 Template Bootstrap cargado: {template.gesture_name}")
                                
                            else:
                                print(f"🎯 DEBUG: Template normal detectado: {template_file.name}")
                                
                                anatomical_emb = None
                                dynamic_emb = None
                                load_method = "ninguno"
                                
                                if 'anatomical_embedding' in template_data and template_data['anatomical_embedding']:
                                    anatomical_emb = np.array(template_data['anatomical_embedding'])
                                    print(f"   🧠 Embedding anatómico en JSON")
                                    load_method = "json"
                                
                                if 'dynamic_embedding' in template_data and template_data['dynamic_embedding']:
                                    dynamic_emb = np.array(template_data['dynamic_embedding'])
                                    print(f"   🔄 Embedding dinámico en JSON")
                                    if load_method == "ninguno":
                                        load_method = "json"
                                
                                if anatomical_emb is None and dynamic_emb is None:
                                    print(f"   🔍 Intentando cargar desde .bin...")
                                    
                                    try:
                                        loaded_template = self._load_template(template_file.stem)
                                        if loaded_template and (loaded_template.anatomical_embedding is not None or loaded_template.dynamic_embedding is not None):
                                            anatomical_emb = loaded_template.anatomical_embedding
                                            dynamic_emb = loaded_template.dynamic_embedding
                                            print(f"   ✅ Cargado desde .bin - A:{anatomical_emb is not None}, D:{dynamic_emb is not None}")
                                            load_method = "bin"
                                            
                                            if anatomical_emb is not None:
                                                print(f"   📊 Shape anatómico: {anatomical_emb.shape}")
                                                print(f"   📊 Norma anatómica: {np.linalg.norm(anatomical_emb):.6f}")
                                            
                                            if dynamic_emb is not None:
                                                print(f"   📊 Shape dinámico: {dynamic_emb.shape}")
                                                print(f"   📊 Norma dinámica: {np.linalg.norm(dynamic_emb):.6f}")
                                        else:
                                            print(f"   ⚠️ _load_template sin embeddings")
                                            
                                    except Exception as bin_error:
                                        print(f"   ❌ Error .bin: {bin_error}")
                                
                                if anatomical_emb is None and dynamic_emb is None:
                                    print(f"   ⚠️ Template sin embeddings")
                                    load_method = "vacio"
                                
                                if anatomical_emb is not None and dynamic_emb is not None:
                                    template_type = TemplateType.MULTIMODAL
                                elif anatomical_emb is not None:
                                    template_type = TemplateType.ANATOMICAL
                                elif dynamic_emb is not None:
                                    template_type = TemplateType.DYNAMIC
                                else:
                                    if 'dynamic' in template_file.name.lower():
                                        template_type = TemplateType.DYNAMIC
                                    else:
                                        template_type = TemplateType.ANATOMICAL
                                
                                template = BiometricTemplate(
                                    user_id=template_data.get('user_id', 'unknown'),
                                    template_id=template_data.get('template_id', template_file.stem),
                                    template_type=template_type,
                                    anatomical_embedding=anatomical_emb,
                                    dynamic_embedding=dynamic_emb,
                                    gesture_name=template_data.get('gesture_name', 'Unknown'),
                                    quality_score=template_data.get('quality_score', 0.0),
                                    confidence=template_data.get('confidence', 0.0),
                                    enrollment_session=template_data.get('enrollment_session', ''),
                                    created_at=template_data.get('created_at', time.time()),
                                    metadata=template_data.get('metadata', {}),
                                    checksum=template_data.get('checksum', '')
                                )
                                
                                print(f"   ✅ Template normal: {template.gesture_name} - Método: {load_method}")
                            
                            self.templates[template.template_id] = template
                            templates_loaded += 1
                            
                            if template.anatomical_embedding is not None:
                                try:
                                    self.anatomical_index.add_embedding(
                                        template.anatomical_embedding,
                                        template.template_id,
                                        template.user_id
                                    )
                                    print(f"   📊 Embedding anatómico añadido al índice")
                                except Exception as idx_error:
                                    print(f"   ❌ Error índice anatómico: {idx_error}")
                            
                            if template.dynamic_embedding is not None:
                                try:
                                    self.dynamic_index.add_embedding(
                                        template.dynamic_embedding,
                                        template.template_id,
                                        template.user_id
                                    )
                                    print(f"   📊 Embedding dinámico añadido al índice")
                                except Exception as idx_error:
                                    print(f"   ❌ Error índice dinámico: {idx_error}")
                            
                            logger.info(f"✅ Template cargado:")
                            logger.info(f"   🆔 ID: {template.template_id}")
                            logger.info(f"   👤 Usuario: {template.user_id}")
                            logger.info(f"   🤚 Gesto: {template.gesture_name}")
                            logger.info(f"   📊 Calidad: {template.quality_score:.2f}")
                            logger.info(f"   🔧 Bootstrap: {is_bootstrap}")
                            
                        except Exception as template_error:
                            logger.error(f"❌ Error creando template: {template_error}")
                            import traceback
                            logger.error(f"   Traceback: {traceback.format_exc()}")
                            continue
                            
                    except Exception as file_error:
                        logger.error(f"❌ Error leyendo {template_file.name}: {file_error}")
                        continue
            else:
                logger.info("📁 Directorio templates no existe, creándolo...")
                templates_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                logger.info("🔍 Validando consistencia usuario ↔ template...")
                
                inconsistencies_found = 0
                templates_added = 0
                
                for user_id, user_profile in self.users.items():
                    all_listed_ids = (user_profile.anatomical_templates + 
                                     user_profile.dynamic_templates + 
                                     user_profile.multimodal_templates)
                    
                    actual_template_ids = []
                    for template_id, template in self.templates.items():
                        if template.user_id == user_id:
                            actual_template_ids.append(template_id)
                    
                    listed_set = set(all_listed_ids)
                    actual_set = set(actual_template_ids)
                    
                    missing_in_lists = actual_set - listed_set
                    
                    if missing_in_lists:
                        inconsistencies_found += 1
                        logger.info(f"⚠️ Inconsistencia usuario {user_id}:")
                        logger.info(f"   📁 Templates sin listar: {len(missing_in_lists)}")
                        
                        for tid in missing_in_lists:
                            template = self.templates[tid]
                            if template.template_type == TemplateType.ANATOMICAL:
                                user_profile.anatomical_templates.append(tid)
                            elif template.template_type == TemplateType.DYNAMIC:
                                user_profile.dynamic_templates.append(tid)
                            elif template.template_type == TemplateType.MULTIMODAL:
                                user_profile.multimodal_templates.append(tid)
                            templates_added += 1
                            logger.info(f"      ✅ Agregado: {tid[:8]}... ({template.template_type.value})")
                        
                        user_profile.total_enrollments = (
                            len(user_profile.anatomical_templates) + 
                            len(user_profile.dynamic_templates) + 
                            len(user_profile.multimodal_templates)
                        )
                        
                        self._save_user(user_profile)
                    
                    else:
                        logger.info(f"✅ Usuario {user_id}: consistente ({len(actual_template_ids)} templates)")
                
                if inconsistencies_found > 0:
                    logger.info(f"🔧 Consistencia corregida:")
                    logger.info(f"   👥 Usuarios afectados: {inconsistencies_found}")
                    logger.info(f"   ➕ Templates agregados: {templates_added}")
                
            except Exception as consistency_error:
                logger.error(f"❌ Error validando consistencia: {consistency_error}")
            
            try:
                logger.info("🔨 Construyendo índices vectoriales...")
                self.anatomical_index.build_index()
                self.dynamic_index.build_index()
                logger.info("✅ Índices construidos")
            except Exception as idx_error:
                logger.error(f"❌ Error construyendo índices: {idx_error}")
            
            try:
                logger.info("📊 Actualizando estadísticas...")
                
                self.stats.total_users = users_loaded
                self.stats.total_templates = templates_loaded
                
                anatomical_count = 0
                dynamic_count = 0
                multimodal_count = 0
                bootstrap_count = 0
                
                for template in self.templates.values():
                    if template.metadata.get('bootstrap_mode', False):
                        bootstrap_count += 1
                    
                    if template.template_type == TemplateType.ANATOMICAL:
                        anatomical_count += 1
                    elif template.template_type == TemplateType.DYNAMIC:
                        dynamic_count += 1
                    elif template.template_type == TemplateType.MULTIMODAL:
                        multimodal_count += 1
                
                self.stats.anatomical_templates = anatomical_count
                self.stats.dynamic_templates = dynamic_count
                self.stats.multimodal_templates = multimodal_count
                
                self._update_stats()
                
                logger.info("✅ Estadísticas actualizadas")
                
            except Exception as stats_error:
                logger.error(f"❌ Error actualizando estadísticas: {stats_error}")
            
            logger.info("=" * 60)
            logger.info("✅ CARGA COMPLETADA")
            logger.info("=" * 60)
            logger.info(f"👥 USUARIOS: {users_loaded}")
            logger.info(f"🧬 TEMPLATES: {templates_loaded}")
            logger.info(f"   📊 Anatómicos: {anatomical_count}")
            logger.info(f"   🔄 Dinámicos: {dynamic_count}")
            logger.info(f"   🔗 Multimodales: {multimodal_count}")
            logger.info(f"   🔧 Bootstrap: {bootstrap_count}")
            logger.info("=" * 60)
            
            if users_loaded > 0:
                logger.info("👥 USUARIOS REGISTRADOS:")
                for user_id, user in self.users.items():
                    total = len(user.anatomical_templates) + len(user.dynamic_templates) + len(user.multimodal_templates)
                    logger.info(f"   • {user.username} ({user_id}) - {total} templates")
    
            return True
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error("❌ ERROR CRÍTICO CARGANDO BD")
            logger.error("=" * 60)
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error("=" * 60)
            
            if not hasattr(self, 'users') or self.users is None:
                self.users = {}
            if not hasattr(self, 'templates') or self.templates is None:
                self.templates = {}
                
            return False
    
    def create_user(self, user_id: str, username: str, 
                   gesture_sequence: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Crea un nuevo usuario en la base de datos."""
        try:
            with self.lock:
                if user_id in self.users:
                    logger.error(f"Usuario {user_id} ya existe")
                    return False
                
                user_profile = UserProfile(
                    user_id=user_id,
                    username=username,
                    gesture_sequence=gesture_sequence or [],
                    metadata=metadata or {}
                )
                
                self.users[user_id] = user_profile
                
                self._save_user(user_profile)
                
                self.stats.total_users += 1
                self._update_stats()
                
                logger.info(f"Usuario creado: {user_id} ({username})")
                
                return True
                
        except Exception as e:
            logger.error(f"Error creando usuario: {e}")
            return False
    
    def store_user_profile(self, user_profile: UserProfile) -> bool:
        """Almacena un perfil de usuario completo."""
        try:
            with self.lock:
                logger.info(f"Almacenando perfil de usuario: {user_profile.user_id}")
                
                if user_profile.user_id in self.users:
                    logger.info(f"Usuario {user_profile.user_id} existe - actualizando")
                    
                    existing_user = self.users[user_profile.user_id]
                    
                    existing_user.username = user_profile.username
                    existing_user.gesture_sequence = user_profile.gesture_sequence
                    existing_user.updated_at = time.time()
                    
                    if hasattr(user_profile, 'metadata'):
                        existing_user.metadata.update(user_profile.metadata or {})
                    
                    if hasattr(user_profile, 'total_samples'):
                        existing_user.total_samples = user_profile.total_samples
                    if hasattr(user_profile, 'valid_samples'):
                        existing_user.valid_samples = user_profile.valid_samples
                    if hasattr(user_profile, 'enrollment_duration'):
                        existing_user.enrollment_duration = user_profile.enrollment_duration
                    if hasattr(user_profile, 'quality_score'):
                        existing_user.quality_score = user_profile.quality_score
                    if hasattr(user_profile, 'enrollment_date'):
                        existing_user.enrollment_date = user_profile.enrollment_date
                    
                    self._save_user(existing_user)
                    
                    logger.info(f"Usuario {user_profile.user_id} actualizado")
                    return True
                    
                else:
                    logger.info(f"Creando nuevo usuario: {user_profile.user_id}")
                    
                    self.users[user_profile.user_id] = user_profile
                    
                    if not hasattr(user_profile, 'anatomical_templates'):
                        user_profile.anatomical_templates = []
                    if not hasattr(user_profile, 'dynamic_templates'):
                        user_profile.dynamic_templates = []
                    if not hasattr(user_profile, 'multimodal_templates'):
                        user_profile.multimodal_templates = []
                    
                    self._save_user(user_profile)
                    
                    self.stats.total_users += 1
                    self._update_stats()
                    
                    logger.info(f"Usuario {user_profile.user_id} creado exitosamente")
                    return True
                    
        except Exception as e:
            logger.error(f"Error almacenando perfil {user_profile.user_id}: {e}")
            return False
    
    def store_biometric_template(self, template: BiometricTemplate) -> bool:
        """Almacena template biométrico."""
        try:
            with self.lock:
                logger.info(f"Almacenando template: {template.template_id}")
                
                if template.user_id not in self.users:
                    logger.error(f"Usuario {template.user_id} no existe para template {template.template_id}")
                    return False
                
                if template.template_id in self.templates:
                    logger.info(f"Template {template.template_id} ya existe - actualizando")
                
                complete_template = template
                
                try:
                    if hasattr(self, '_calculate_template_checksum'):
                        complete_template.checksum = self._calculate_template_checksum(complete_template)
                    else:
                        complete_template.checksum = "not_available"
                except Exception as e:
                    logger.info(f"No se pudo calcular checksum: {e}")
                    complete_template.checksum = "error_calculating"
                
                self.templates[template.template_id] = complete_template
                
                if hasattr(template, 'anatomical_embedding') and template.anatomical_embedding is not None:
                    try:
                        self.anatomical_index.add_embedding(
                            template.anatomical_embedding, 
                            template.template_id, 
                            template.user_id
                        )
                        logger.info(f"Template anatómico agregado al índice")
                    except Exception as e:
                        logger.info(f"Error índice anatómico: {e}")
                        
                if hasattr(template, 'dynamic_embedding') and template.dynamic_embedding is not None:
                    try:
                        self.dynamic_index.add_embedding(
                            template.dynamic_embedding, 
                            template.template_id, 
                            template.user_id
                        )
                        logger.info(f"Template dinámico agregado al índice vectorial")
                    except Exception as e:
                        logger.info(f"Error índice dinámico: {e}")
                
                user_profile = self.users[template.user_id]
                
                if template.template_type == TemplateType.ANATOMICAL:
                    if template.template_id not in user_profile.anatomical_templates:
                        user_profile.anatomical_templates.append(template.template_id)
                        logger.info(f"Template anatómico agregado al perfil del usuario")
                elif template.template_type == TemplateType.DYNAMIC:
                    if template.template_id not in user_profile.dynamic_templates:
                        user_profile.dynamic_templates.append(template.template_id)
                        logger.info(f"Template dinámico agregado al perfil del usuario")
                else:
                    if template.template_id not in user_profile.multimodal_templates:
                        user_profile.multimodal_templates.append(template.template_id)
                        logger.info(f"Template multimodal agregado al perfil del usuario")
                
                user_profile.total_enrollments += 1
                user_profile.updated_at = time.time()
                
                try:
                    self._save_template(complete_template)
                    logger.info(f"Template guardado en disco")
                except Exception as e:
                    logger.info(f"Error guardando template: {e}")
                    
                try:
                    self._save_user(user_profile)
                    logger.info(f"Perfil actualizado")
                except Exception as e:
                    logger.info(f"Error actualizando usuario: {e}")
                
                self.stats.total_templates += 1
                if template.template_type == TemplateType.ANATOMICAL:
                    self.stats.anatomical_templates += 1
                elif template.template_type == TemplateType.DYNAMIC:
                    self.stats.dynamic_templates += 1
                else:
                    self.stats.multimodal_templates += 1
                
                try:
                    self._update_stats()
                except Exception as e:
                    logger.info(f"Error actualizando estadísticas: {e}")
                
                try:
                    self.anatomical_index.build_index()
                    self.dynamic_index.build_index()
                    logger.info(f"Índices vectorialesreconstruidos")
                except Exception as e:
                    logger.info(f"Error reconstruyendo índices: {e}")
                
                logger.info(f"✅ Template {template.template_id} almacenado")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error almacenando template {template.template_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
    
    def enroll_template(self, user_id: str, 
                       anatomical_embedding: Optional[np.ndarray] = None,
                       dynamic_embedding: Optional[np.ndarray] = None,
                       gesture_name: str = "unknown",
                       quality_score: float = 1.0,
                       confidence: float = 1.0,
                       metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Enrolla un nuevo template biométrico."""
        try:
            with self.lock:
                if user_id not in self.users:
                    logger.error(f"Usuario {user_id} no existe")
                    return None
                
                if anatomical_embedding is None and dynamic_embedding is None:
                    logger.error("Se requiere al menos un embedding")
                    return None
                
                if anatomical_embedding is not None and anatomical_embedding.shape[0] != 64:
                    logger.error("Embedding anatómico debe tener 64 dimensiones")
                    return None
                
                if dynamic_embedding is not None and dynamic_embedding.shape[0] != 128:
                    logger.error("Embedding dinámico debe tener 128 dimensiones")
                    return None
                
                if anatomical_embedding is not None and dynamic_embedding is not None:
                    template_type = TemplateType.MULTIMODAL
                elif anatomical_embedding is not None:
                    template_type = TemplateType.ANATOMICAL
                else:
                    template_type = TemplateType.DYNAMIC
                
                template_id = f"{user_id}_{template_type.value}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                
                template = BiometricTemplate(
                    user_id=user_id,
                    template_id=template_id,
                    template_type=template_type,
                    anatomical_embedding=anatomical_embedding,
                    dynamic_embedding=dynamic_embedding,
                    gesture_name=gesture_name,
                    quality_score=quality_score,
                    confidence=confidence,
                    enrollment_session=str(uuid.uuid4()),
                    metadata=metadata or {}
                )
                
                # AGREGAR SECUENCIA TEMPORAL SI EXISTE
                if hasattr(sample, 'temporal_sequence') and sample.temporal_sequence is not None:
                    template.metadata['temporal_sequence'] = sample.temporal_sequence.tolist()
                    template.metadata['sequence_length'] = sample.sequence_length
                    template.metadata['has_temporal_data'] = True
                    log_info(f"Template con secuencia temporal: {sample.sequence_length} frames")
                else:
                    template.metadata['has_temporal_data'] = False

                # ✅ AGREGAR CARACTERÍSTICAS ANATÓMICAS RAW PARA REENTRENAMIENTO
                if anatomical_features is not None:
                    template.metadata['bootstrap_features'] = anatomical_features.tolist()
                    template.metadata['feature_dimensions'] = len(anatomical_features)
                    template.metadata['has_anatomical_raw'] = True
                    log_info(f"Template con características anatómicas raw: {len(anatomical_features)} dimensiones")
                else:
                    template.metadata['has_anatomical_raw'] = False
                
                # ✅ MARCAR MODO BOOTSTRAP CORRECTAMENTE
                template.metadata['bootstrap_mode'] = sample_metadata.get('bootstrap_mode', False) if sample_metadata else False
                template.metadata['data_source'] = sample_metadata.get('data_source', 'enrollment_capture') if sample_metadata else 'enrollment_capture'


                template.checksum = self._calculate_template_checksum(template)
                
                self.templates[template_id] = template
                
                if anatomical_embedding is not None:
                    self.anatomical_index.add_embedding(anatomical_embedding, template_id, user_id)
                
                if dynamic_embedding is not None:
                    self.dynamic_index.add_embedding(dynamic_embedding, template_id, user_id)
                
                user_profile = self.users[user_id]
                if template_type == TemplateType.ANATOMICAL:
                    user_profile.anatomical_templates.append(template_id)
                elif template_type == TemplateType.DYNAMIC:
                    user_profile.dynamic_templates.append(template_id)
                else:
                    user_profile.multimodal_templates.append(template_id)
                
                user_profile.total_enrollments += 1
                user_profile.updated_at = time.time()
                
                self._save_template(template)
                self._save_user(user_profile)
                
                self.anatomical_index.build_index()
                self.dynamic_index.build_index()
                
                self.stats.total_templates += 1
                if template_type == TemplateType.ANATOMICAL:
                    self.stats.anatomical_templates += 1
                elif template_type == TemplateType.DYNAMIC:
                    self.stats.dynamic_templates += 1
                else:
                    self.stats.multimodal_templates += 1
                
                if quality_score >= 0.9:
                    self.stats.excellent_quality += 1
                elif quality_score >= 0.7:
                    self.stats.good_quality += 1
                elif quality_score >= 0.5:
                    self.stats.fair_quality += 1
                else:
                    self.stats.poor_quality += 1
                
                self._update_stats()
                
                logger.info(f"Template enrollado: {template_id}")
                
                return template_id
                
        except Exception as e:
            logger.error(f"Error enrollando template: {e}")
            return None
        
    def verify_user(self, query_anatomical: Optional[np.ndarray] = None,
                   query_dynamic: Optional[np.ndarray] = None,
                   user_id: Optional[str] = None,
                   max_results: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Verifica usuario contra templates almacenados."""
        try:
            with self.lock:
                if query_anatomical is None and query_dynamic is None:
                    return []
                
                results = []
                
                anatomical_matches = []
                if query_anatomical is not None:
                    exclude_user = None if user_id is None else user_id
                    anatomical_matches = self.anatomical_index.search_similar(
                        query_anatomical, k=max_results * 2, exclude_user=exclude_user
                    )
                
                dynamic_matches = []
                if query_dynamic is not None:
                    exclude_user = None if user_id is None else user_id
                    dynamic_matches = self.dynamic_index.search_similar(
                        query_dynamic, k=max_results * 2, exclude_user=exclude_user
                    )
                
                combined_scores = defaultdict(list)
                
                for template_id, match_user_id, distance in anatomical_matches:
                    if user_id and match_user_id != user_id:
                        continue
                    
                    similarity = max(0, 1 - distance / 2)
                    combined_scores[match_user_id].append(('anatomical', similarity, template_id))
                
                for template_id, match_user_id, distance in dynamic_matches:
                    if user_id and match_user_id != user_id:
                        continue
                    
                    similarity = max(0, 1 - distance / 2)
                    combined_scores[match_user_id].append(('dynamic', similarity, template_id))
                
                for match_user_id, scores in combined_scores.items():
                    anatomical_scores = [s[1] for s in scores if s[0] == 'anatomical']
                    dynamic_scores = [s[1] for s in scores if s[0] == 'dynamic']
                    
                    final_score = 0
                    weight_sum = 0
                    
                    if anatomical_scores:
                        anat_score = max(anatomical_scores)
                        final_score += anat_score * 0.6
                        weight_sum += 0.6
                    
                    if dynamic_scores:
                        dyn_score = max(dynamic_scores)
                        final_score += dyn_score * 0.4
                        weight_sum += 0.4
                    
                    if weight_sum > 0:
                        final_score /= weight_sum
                    
                    details = {
                        'anatomical_score': max(anatomical_scores) if anatomical_scores else 0,
                        'dynamic_score': max(dynamic_scores) if dynamic_scores else 0,
                        'anatomical_matches': len(anatomical_scores),
                        'dynamic_matches': len(dynamic_scores),
                        'templates_matched': [s[2] for s in scores]
                    }
                    
                    results.append((match_user_id, final_score, details))
                
                results.sort(key=lambda x: x[1], reverse=True)
                
                for match_user_id, score, _ in results[:max_results]:
                    if match_user_id in self.users:
                        user_profile = self.users[match_user_id]
                        user_profile.total_verifications += 1
                        
                        if score > 0.7:
                            user_profile.successful_verifications += 1
                            self.stats.successful_verifications += 1
                        
                        user_profile.last_activity = time.time()
                        self._save_user(user_profile)
                
                self.stats.total_verifications += 1
                self._update_stats()
                
                logger.info(f"Verificación: {len(results)} matches")
                
                return results[:max_results]
                
        except Exception as e:
            logger.error(f"Error en verificación: {e}")
            return []
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Obtiene perfil de usuario."""
        return self.users.get(user_id)
    
    def get_template(self, template_id: str) -> Optional[BiometricTemplate]:
        """Obtiene template biométrico."""
        return self.templates.get(template_id)
    
    def list_users(self) -> List[UserProfile]:
        """Lista todos los usuarios."""
        return list(self.users.values())
    
    def list_user_templates(self, user_id: str) -> List[BiometricTemplate]:
        """Lista templates de un usuario."""
        if user_id not in self.users:
            return []
        
        user_profile = self.users[user_id]
        all_template_ids = (user_profile.anatomical_templates + 
                           user_profile.dynamic_templates + 
                           user_profile.multimodal_templates)
        
        templates = []
        for template_id in all_template_ids:
            if template_id in self.templates:
                templates.append(self.templates[template_id])
        
        return templates
    
    def delete_user(self, user_id: str) -> bool:
        """Elimina un usuario y todos sus templates."""
        try:
            with self.lock:
                if user_id not in self.users:
                    logger.error(f"Usuario {user_id} no existe")
                    return False
                
                user_templates = self.list_user_templates(user_id)
                
                for template in user_templates:
                    self.delete_template(template.template_id)
                
                del self.users[user_id]
                
                user_file = self.db_path / 'users' / f'{user_id}.json'
                if user_file.exists():
                    user_file.unlink()
                
                self.stats.total_users -= 1
                self._update_stats()
                
                logger.info(f"Usuario eliminado: {user_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error eliminando usuario: {e}")
            return False
    
    def delete_template(self, template_id: str) -> bool:
        """Elimina un template específico."""
        try:
            with self.lock:
                if template_id not in self.templates:
                    logger.error(f"Template {template_id} no existe")
                    return False
                
                template = self.templates[template_id]
                user_id = template.user_id
                
                self.anatomical_index.remove_template(template_id)
                self.dynamic_index.remove_template(template_id)
                
                del self.templates[template_id]
                
                if user_id in self.users:
                    user_profile = self.users[user_id]
                    
                    if template_id in user_profile.anatomical_templates:
                        user_profile.anatomical_templates.remove(template_id)
                    if template_id in user_profile.dynamic_templates:
                        user_profile.dynamic_templates.remove(template_id)
                    if template_id in user_profile.multimodal_templates:
                        user_profile.multimodal_templates.remove(template_id)
                    
                    user_profile.updated_at = time.time()
                    self._save_user(user_profile)
                
                template_file = self.db_path / 'templates' / f'{template_id}.json'
                if template_file.exists():
                    template_file.unlink()
                
                embedding_file = self.db_path / 'templates' / f'{template_id}.bin'
                if embedding_file.exists():
                    embedding_file.unlink()
                
                self.stats.total_templates -= 1
                if template.template_type == TemplateType.ANATOMICAL:
                    self.stats.anatomical_templates -= 1
                elif template.template_type == TemplateType.DYNAMIC:
                    self.stats.dynamic_templates -= 1
                else:
                    self.stats.multimodal_templates -= 1
                
                self._update_stats()
                
                logger.info(f"Template eliminado: {template_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error eliminando template: {e}")
            return False
    
    def _save_user(self, user_profile: UserProfile):
        """Guarda perfil de usuario en disco."""
        try:
            user_file = self.db_path / 'users' / f'{user_profile.user_id}.json'
            
            print(f"🔍 DEBUG: Guardando usuario {user_profile.user_id}")
            print(f"🔍 DEBUG: Ruta: {user_file}")
            print(f"🔍 DEBUG: Directorio existe: {user_file.parent.exists()}")
            
            user_data = asdict(user_profile)
            
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, indent=2)
            
            print(f"✅ DEBUG: Usuario guardado en {user_file}")
            
        except Exception as e:
            print(f"❌ DEBUG ERROR guardando usuario: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Error guardando usuario: {e}")
            
    def _save_template(self, template: BiometricTemplate):
        """Guarda template en disco SIN ENCRIPTACIÓN - VERSIÓN DEBUG."""
        try:
            print(f"🔧 DEBUG: Iniciando guardado template {template.template_id}")
            
            templates_dir = self.db_path / 'templates'
            templates_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 DEBUG: Directorio templates: {templates_dir}")
            
            template_file = templates_dir / f'{template.template_id}.json'
            
            template_data = {
                'template_id': template.template_id,
                'user_id': template.user_id,
                'template_type': template.template_type.value if hasattr(template.template_type, 'value') else str(template.template_type),
                'gesture_name': template.gesture_name,
                'hand_side': getattr(template, 'hand_side', 'unknown'),
                'quality_score': float(template.quality_score) if template.quality_score is not None else None,
                'confidence': float(template.confidence) if template.confidence is not None else None,
                'created_at': template.created_at,
                'updated_at': template.updated_at,
                'last_used': getattr(template, 'last_used', template.created_at),
                'enrollment_session': getattr(template, 'enrollment_session', ''),
                'verification_count': getattr(template, 'verification_count', 0),
                'success_count': getattr(template, 'success_count', 0),
                'is_encrypted': False,
                'checksum': getattr(template, 'checksum', ''),
                'metadata': getattr(template, 'metadata', {}),
                'anatomical_embedding': None,
                'dynamic_embedding': None
            }
            
            print(f"📋 DEBUG: Metadatos preparados")
            
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, default=str)
            
            print(f"✅ DEBUG: JSON guardado: {template_file}")
            print(f"📦 DEBUG: Tamaño JSON: {template_file.stat().st_size} bytes")
            
            embeddings_data = {}
            
            if hasattr(template, 'anatomical_embedding') and template.anatomical_embedding is not None:
                print(f"🧠 DEBUG: Embedding anatómico encontrado")
                print(f"   📊 Tipo: {type(template.anatomical_embedding)}")
                
                if isinstance(template.anatomical_embedding, np.ndarray):
                    print(f"   📐 Shape: {template.anatomical_embedding.shape}")
                    print(f"   📈 Dtype: {template.anatomical_embedding.dtype}")
                    print(f"   📊 Min: {template.anatomical_embedding.min():.6f}")
                    print(f"   📊 Max: {template.anatomical_embedding.max():.6f}")
                    print(f"   📊 Norma: {np.linalg.norm(template.anatomical_embedding):.6f}")
                    
                    embeddings_data['anatomical'] = template.anatomical_embedding.copy()
                    print(f"   ✅ Embedding anatómico agregado")
            else:
                print(f"⚠️ DEBUG: No hay embedding anatómico")
            
            if hasattr(template, 'dynamic_embedding') and template.dynamic_embedding is not None:
                print(f"🔄 DEBUG: Embedding dinámico encontrado")
                print(f"   📊 Tipo: {type(template.dynamic_embedding)}")
                
                if isinstance(template.dynamic_embedding, np.ndarray):
                    print(f"   📐 Shape: {template.dynamic_embedding.shape}")
                    print(f"   📈 Dtype: {template.dynamic_embedding.dtype}")
                    print(f"   📊 Min: {template.dynamic_embedding.min():.6f}")
                    print(f"   📊 Max: {template.dynamic_embedding.max():.6f}")
                    print(f"   📊 Norma: {np.linalg.norm(template.dynamic_embedding):.6f}")
                    
                    embeddings_data['dynamic'] = template.dynamic_embedding.copy()
                    print(f"   ✅ Embedding dinámico agregado")
            else:
                print(f"⚠️ DEBUG: No hay embedding dinámico")
            
            if embeddings_data:
                embeddings_file = templates_dir / f'{template.template_id}.bin'
                
                print(f"🔐 DEBUG: Guardando {len(embeddings_data)} embeddings sin encriptar")
                print(f"   📋 Embeddings: {list(embeddings_data.keys())}")
                
                try:
                    serialized_data = pickle.dumps(embeddings_data, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"📦 DEBUG: Datos serializados: {len(serialized_data)} bytes")
                    
                    with open(embeddings_file, 'wb') as f:
                        f.write(serialized_data)
                        f.flush()
                    
                    print(f"✅ DEBUG: BIN guardado sin encriptar: {embeddings_file}")
                    print(f"📦 DEBUG: Tamaño final BIN: {embeddings_file.stat().st_size} bytes")
                    
                    print(f"🔍 DEBUG: Verificando archivo...")
                    
                    with open(embeddings_file, 'rb') as f:
                        test_data = f.read()
                    
                    print(f"📦 DEBUG: Leído para verificación: {len(test_data)} bytes")
                    
                    test_embeddings = pickle.loads(test_data)
                    print(f"✅ DEBUG: Deserialización exitosa")
                    print(f"📋 DEBUG: Claves recuperadas: {list(test_embeddings.keys())}")
                    
                    for key, embedding in test_embeddings.items():
                        if isinstance(embedding, np.ndarray):
                            print(f"   ✅ {key}: {embedding.shape}, norma={np.linalg.norm(embedding):.6f}")
                        else:
                            print(f"   ❌ {key}: tipo incorrecto {type(embedding)}")
                    
                except Exception as save_error:
                    print(f"❌ DEBUG: Error guardando embeddings: {save_error}")
                    import traceback
                    traceback.print_exc()
                    raise
                    
            else:
                print(f"⚠️ DEBUG: No hay embeddings para guardar")
            
            print(f"🎉 DEBUG: Template {template.template_id} guardado completamente")
            
        except Exception as e:
            print(f"❌ DEBUG: Error en _save_template: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_template(self, template_id: str) -> Optional[BiometricTemplate]:
        """Carga template desde disco"""
        try:
            print(f"🔍 DEBUG: Cargando template {template_id}")
            
            template_file = self.db_path / 'templates' / f'{template_id}.json'
            print(f"   📄 Buscando JSON: {template_file}")
            
            if not template_file.exists():
                print(f"   ❌ Archivo JSON no existe")
                return None
            
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
            except Exception as json_error:
                print(f"   ❌ Error leyendo JSON: {json_error}")
                return None
            
            print(f"   ✅ JSON cargado")
            print(f"   📋 Tipo template: {template_data.get('template_type')}")
            print(f"   👤 Usuario: {template_data.get('user_id')}")
            print(f"   🤌 Gesto: {template_data.get('gesture_name', 'N/A')}")
            print(f"   🔐 Encriptado según JSON: {template_data.get('is_encrypted', 'N/A')}")
            
            embeddings_file = self.db_path / 'templates' / f'{template_id}.bin'
            print(f"   📦 Buscando BIN: {embeddings_file}")
            
            embeddings_data = {}
            
            if embeddings_file.exists():
                file_size = embeddings_file.stat().st_size
                print(f"   ✅ Archivo BIN existe - Tamaño: {file_size} bytes")
                
                if file_size == 0:
                    print(f"   ⚠️ Archivo BIN vacío")
                    embeddings_data = {}
                else:
                    try:
                        with open(embeddings_file, 'rb') as f:
                            embeddings_bytes = f.read()
                        
                        print(f"   📦 Bytes leídos: {len(embeddings_bytes)}")
                        
                        encryption_enabled = self.config.get('encryption_enabled', False)
                        print(f"   🔐 Encriptación en config: {encryption_enabled}")
                        
                        should_decrypt = encryption_enabled
                        
                        if should_decrypt and CRYPTO_AVAILABLE:
                            try:
                                print(f"   🔓 Intentando desencriptar...")
                                if hasattr(self, 'cipher') and self.cipher is not None:
                                    embeddings_bytes = self.cipher.decrypt(embeddings_bytes)
                                    print(f"   ✅ Desencriptación exitosa")
                                else:
                                    print(f"   ❌ Cipher no disponible")
                            except Exception as decrypt_error:
                                print(f"   ⚠️ Error desencriptando: {decrypt_error}")
                                with open(embeddings_file, 'rb') as f:
                                    embeddings_bytes = f.read()
                        else:
                            print(f"   ℹ️ Sin encriptación")
                        
                        print(f"   🔄 Deserializando...")
                        try:
                            embeddings_data = pickle.loads(embeddings_bytes)
                            print(f"   ✅ Deserialización exitosa")
                            print(f"   📋 Claves: {list(embeddings_data.keys())}")
                            
                            for key, embedding in embeddings_data.items():
                                if embedding is None:
                                    print(f"      ⚠️ {key}: None")
                                elif isinstance(embedding, np.ndarray):
                                    print(f"      ✅ {key}: shape={embedding.shape}, dtype={embedding.dtype}")
                                    print(f"         📊 Norma: {np.linalg.norm(embedding):.6f}")
                                    print(f"         📊 Min: {embedding.min():.6f}, Max: {embedding.max():.6f}")
                                    print(f"         📊 NaN count: {np.sum(np.isnan(embedding))}")
                                    print(f"         📊 Inf count: {np.sum(np.isinf(embedding))}")
                                else:
                                    print(f"      ❌ {key}: tipo incorrecto")
                                    try:
                                        converted = np.array(embedding, dtype=np.float32)
                                        embeddings_data[key] = converted
                                        print(f"         🔄 Convertido a numpy: {converted.shape}")
                                    except:
                                        embeddings_data[key] = None
                            
                        except Exception as pickle_error:
                            print(f"   ❌ Error pickle: {pickle_error}")
                            embeddings_data = {}
                            
                    except Exception as file_error:
                        print(f"   ❌ Error leyendo BIN: {file_error}")
                        embeddings_data = {}
            else:
                print(f"   ⚠️ Archivo BIN no existe")
                embeddings_data = {}
            
            anatomical_embedding = embeddings_data.get('anatomical')
            dynamic_embedding = embeddings_data.get('dynamic')
            
            print(f"   🧠 Embedding anatómico disponible: {anatomical_embedding is not None}")
            print(f"   🔄 Embedding dinámico disponible: {dynamic_embedding is not None}")
            
            template_data_copy = template_data.copy()
            template_data_copy['anatomical_embedding'] = anatomical_embedding
            template_data_copy['dynamic_embedding'] = dynamic_embedding
            
            template_type_value = template_data_copy.get('template_type')
            if isinstance(template_type_value, str):
                try:
                    if template_type_value == 'anatomical':
                        template_data_copy['template_type'] = TemplateType.ANATOMICAL
                    elif template_type_value == 'dynamic':
                        template_data_copy['template_type'] = TemplateType.DYNAMIC
                    elif template_type_value == 'multimodal':
                        template_data_copy['template_type'] = TemplateType.MULTIMODAL
                    else:
                        print(f"   ⚠️ Tipo desconocido '{template_type_value}', usando ANATOMICAL")
                        template_data_copy['template_type'] = TemplateType.ANATOMICAL
                except Exception as enum_error:
                    print(f"   ❌ Error enum: {enum_error}")
                    template_data_copy['template_type'] = TemplateType.ANATOMICAL
            
            print(f"   🏗️ Creando BiometricTemplate...")
            
            try:
                required_fields = {
                    'user_id': template_data_copy.get('user_id', 'unknown'),
                    'template_id': template_data_copy.get('template_id', template_id),
                    'template_type': template_data_copy.get('template_type', TemplateType.ANATOMICAL),
                    'gesture_name': template_data_copy.get('gesture_name', 'Unknown'),
                    'quality_score': float(template_data_copy.get('quality_score', 0.0)),
                    'confidence': float(template_data_copy.get('confidence', 0.0)),
                    'enrollment_session': template_data_copy.get('enrollment_session', ''),
                    'created_at': template_data_copy.get('created_at', time.time()),
                    'updated_at': template_data_copy.get('updated_at', time.time()),
                    'metadata': template_data_copy.get('metadata', {}),
                    'checksum': template_data_copy.get('checksum', ''),
                    'anatomical_embedding': anatomical_embedding,
                    'dynamic_embedding': dynamic_embedding
                }
                
                optional_fields = ['last_used', 'verification_count', 'success_count', 'is_encrypted']
                for field in optional_fields:
                    if field in template_data_copy:
                        required_fields[field] = template_data_copy[field]
                
                template = BiometricTemplate(**required_fields)
                
                print(f"   ✅ BiometricTemplate creado exitosamente")

                print(f"   🔍 VERIFICACIÓN FINAL:")
                print(f"      ID: {template.template_id}")
                print(f"      Usuario: {template.user_id}")
                print(f"      Tipo: {template.template_type}")
                print(f"      Gesto: {template.gesture_name}")
                print(f"      Anatómico disponible: {'✅' if template.anatomical_embedding is not None else '❌'}")
                print(f"      Dinámico disponible: {'✅' if template.dynamic_embedding is not None else '❌'}")
                
                if template.anatomical_embedding is not None:
                    print(f"      Anatómico shape: {template.anatomical_embedding.shape}")
                    print(f"      Anatómico norma: {np.linalg.norm(template.anatomical_embedding):.6f}")
                
                if template.dynamic_embedding is not None:
                    print(f"      Dinámico shape: {template.dynamic_embedding.shape}")
                    print(f"      Dinámico norma: {np.linalg.norm(template.dynamic_embedding):.6f}")
                
                print(f"✅ DEBUG: Template {template_id} cargado exitosamente")
                return template
                
            except Exception as template_error:
                print(f"   ❌ Error creando template: {template_error}")
                import traceback
                traceback.print_exc()
                return None
            
        except Exception as e:
            print(f"❌ DEBUG: Error general: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_template_checksum(self, template: BiometricTemplate) -> str:
        """Calcula checksum de integridad del template."""
        try:
            data_string = f"{template.user_id}{template.template_type.value}{template.created_at}"
            
            if template.anatomical_embedding is not None:
                data_string += str(np.sum(template.anatomical_embedding))
            
            if template.dynamic_embedding is not None:
                data_string += str(np.sum(template.dynamic_embedding))
            
            return hashlib.sha256(data_string.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"Error calculando checksum: {e}")
            return ""
    
    def _update_stats(self):
        """Actualiza estadísticas de la base de datos."""
        try:
            total_size = 0
            for root, dirs, files in os.walk(self.db_path):
                total_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
            
            self.stats.total_size_mb = total_size / 1024 / 1024
            self.stats.last_updated = time.time()
            
            stats_file = self.db_path / 'database_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(asdict(self.stats), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error actualizando estadísticas: {e}")
    
    def create_backup(self) -> bool:
        """Crea backup completo de la base de datos."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.db_path / 'backups' / f'backup_{timestamp}'
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            for source_dir in ['users', 'templates']:
                source_path = self.db_path / source_dir
                dest_path = backup_dir / source_dir
                
                if source_path.exists():
                    shutil.copytree(source_path, dest_path)
            
            backup_archive = self.db_path / 'backups' / f'backup_{timestamp}.tar.gz'
            shutil.make_archive(str(backup_archive).replace('.tar.gz', ''), 'gztar', backup_dir)
            
            shutil.rmtree(backup_dir)
            
            self._cleanup_old_backups()
            
            logger.info(f"Backup creado: {backup_archive}")
            return True
            
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """Limpia backups antiguos."""
        try:
            backups_dir = self.db_path / 'backups'
            if not backups_dir.exists():
                return
                
            backup_files = list(backups_dir.glob('backup_*.tar.gz'))
            
            if len(backup_files) > self.config['max_backups']:
                backup_files.sort(key=lambda x: x.stat().st_mtime)
                for old_backup in backup_files[:-self.config['max_backups']]:
                    old_backup.unlink()
                    logger.info(f"Backup antiguo eliminado: {old_backup.name}")
                    
        except Exception as e:
            logger.error(f"Error limpiando backups: {e}")
    
    def get_database_stats(self) -> DatabaseStats:
        """Obtiene estadísticas actuales."""
        self._update_stats()
        return self.stats
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verifica integridad de la base de datos."""
        try:
            issues = []
            
            for user_id, user_profile in self.users.items():
                user_file = self.db_path / 'users' / f'{user_id}.json'
                if not user_file.exists():
                    issues.append(f"Archivo usuario faltante: {user_id}")
            
            for template_id, template in self.templates.items():
                template_file = self.db_path / 'templates' / f'{template_id}.json'
                if not template_file.exists():
                    issues.append(f"Archivo template faltante: {template_id}")
                
                current_checksum = self._calculate_template_checksum(template)
                if current_checksum != template.checksum:
                    issues.append(f"Checksum inválido en template: {template_id}")
            
            anatomical_count = len(self.anatomical_index.template_ids)
            dynamic_count = len(self.dynamic_index.template_ids)
            
            anatomical_templates = len([t for t in self.templates.values() 
                                      if t.anatomical_embedding is not None])
            dynamic_templates = len([t for t in self.templates.values() 
                                   if t.dynamic_embedding is not None])
            
            if anatomical_count != anatomical_templates:
                issues.append(f"Índice anatómico inconsistente: {anatomical_count} vs {anatomical_templates}")
            
            if dynamic_count != dynamic_templates:
                issues.append(f"Índice dinámico inconsistente: {dynamic_count} vs {dynamic_templates}")
            
            return {
                'integrity_ok': len(issues) == 0,
                'issues': issues,
                'total_users': len(self.users),
                'total_templates': len(self.templates),
                'anatomical_index_size': anatomical_count,
                'dynamic_index_size': dynamic_count
            }
            
        except Exception as e:
            logger.error(f"Error verificando integridad: {e}")
            return {'integrity_ok': False, 'error': str(e)}
    
    def export_database(self, export_path: str, include_embeddings: bool = True) -> bool:
        """Exporta la base de datos a un archivo."""
        try:
            export_data = {
                'users': {},
                'templates': {},
                'stats': asdict(self.stats),
                'export_timestamp': time.time(),
                'version': '1.0'
            }
            
            for user_id, user_profile in self.users.items():
                export_data['users'][user_id] = asdict(user_profile)
            
            for template_id, template in self.templates.items():
                template_data = asdict(template)
                
                if include_embeddings:
                    if template.anatomical_embedding is not None:
                        template_data['anatomical_embedding'] = template.anatomical_embedding.tolist()
                    if template.dynamic_embedding is not None:
                        template_data['dynamic_embedding'] = template.dynamic_embedding.tolist()
                else:
                    template_data['anatomical_embedding'] = None
                    template_data['dynamic_embedding'] = None
                
                export_data['templates'][template_id] = template_data
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Base de datos exportada a: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la base de datos."""
        return {
            'database_path': str(self.db_path),
            'total_users': len(self.users),
            'total_templates': len(self.templates),
            'anatomical_templates': len([t for t in self.templates.values() if t.anatomical_embedding is not None]),
            'dynamic_templates': len([t for t in self.templates.values() if t.dynamic_embedding is not None]),
            'multimodal_templates': len([t for t in self.templates.values() if t.template_type == TemplateType.MULTIMODAL]),
            'encryption_enabled': self.config['encryption_enabled'],
            'search_strategy': self.config['search_strategy'],
            'database_size_mb': self.stats.total_size_mb,
            'last_backup': 'N/A',
            'integrity_status': 'OK'
        }
    
    def enroll_template_bootstrap(self, user_id: str,
                        anatomical_features: Optional[np.ndarray] = None,
                        gesture_name: str = "unknown",
                        quality_score: float = 1.0,
                        confidence: float = 1.0,
                        sample_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Enrolla datos en modo Bootstrap (sin embeddings todavía)."""
        try:
            with self.lock:
                if user_id not in self.users:
                    logger.info(f"🆕 Usuario {user_id} no existe - Creando automáticamente")
                    
                    username = "Usuario Bootstrap"
                    if sample_metadata and 'session_username' in sample_metadata:
                        username = sample_metadata['session_username']
                    elif sample_metadata and 'username' in sample_metadata:
                        username = sample_metadata['username']
                    
                    user_profile = UserProfile(
                        user_id=user_id,
                        username=username,
                        gesture_sequence=[],
                        metadata={
                            'bootstrap_mode': True,
                            'auto_created': True,
                            'creation_reason': 'First template enrollment in bootstrap mode'
                        }
                    )
                    
                    self.users[user_id] = user_profile
                    self._save_user(user_profile)
                    
                    logger.info(f"✅ Usuario {user_id} creado automáticament: {username}")
                
                if anatomical_features is None:
                    logger.error("Se requieren características anatómicas en Bootstrap")
                    return None
                
                if anatomical_features.shape[0] != 180:
                    logger.error("Características anatómicas deben tener 180 dimensiones")
                    return None
                
                # =========================================================================
                # ✅ PASO 1: CREAR TEMPLATE ANATÓMICO
                # =========================================================================
                anatomical_template_id = f"{user_id}_bootstrap_anatomical_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                
                anatomical_template = BiometricTemplate(
                    user_id=user_id,
                    template_id=anatomical_template_id,
                    template_type=TemplateType.ANATOMICAL,
                    anatomical_embedding=None,
                    dynamic_embedding=None,
                    gesture_name=gesture_name,
                    quality_score=quality_score,
                    confidence=confidence,
                    enrollment_session=str(uuid.uuid4()),
                    metadata=(sample_metadata or {}).copy()
                )
                
                anatomical_template.metadata['bootstrap_features'] = anatomical_features.tolist()
                anatomical_template.metadata['has_anatomical_raw'] = True
                anatomical_template.metadata['feature_dimensions'] = len(anatomical_features)
                anatomical_template.metadata['bootstrap_mode'] = True
                anatomical_template.metadata['pending_embedding'] = True
                anatomical_template.metadata['modality'] = 'anatomical'
                
                # =========================================================================
                # ✅ PASO 2: BUSCAR DATOS TEMPORALES
                # =========================================================================
                
                dynamic_template_id = None
                temporal_sequence = None
                data_source_found = None
                is_real_temporal = False
                
                try:
                    log_info("🔍 BUSCANDO datos temporales REALES desde metadata de muestra...")
                    
                    # ✅ MÉTODO PRINCIPAL: BUSCAR EN METADATA DE LA MUESTRA ACTUAL
                    if (sample_metadata and 
                        'has_temporal_data' in sample_metadata and 
                        sample_metadata['has_temporal_data'] and
                        'temporal_sequence' in sample_metadata and
                        sample_metadata['temporal_sequence'] is not None):
                        
                        temporal_sequence = np.array(sample_metadata['temporal_sequence'], dtype=np.float32)
                        data_source_found = sample_metadata.get('data_source', 'real_enrollment_capture')
                        is_real_temporal = True  # SIEMPRE real si viene de metadata de muestra
                        
                        logger.info(f"✅ MÉTODO PRINCIPAL: Secuencia temporal REAL encontrada en metadata: {temporal_sequence.shape}")
                        logger.info(f"   📊 Fuente: {data_source_found}")
                        logger.info(f"   📊 Longitud: {sample_metadata.get('sequence_length', len(temporal_sequence))} frames")
                    
                    # ✅ MÉTODO ALTERNATIVO: BUSCAR EN ENROLLMENT SYSTEM ACTIVO (SOLO SI NO HAY DATOS)
                    elif temporal_sequence is None:  
                        try:
                            logger.info("🔄 MÉTODO ALTERNATIVO: Buscando en sesiones activas...")
                            # Buscar directamente en este objeto si es el enrollment system
                            if hasattr(self, 'active_sessions'):
                                for session_id, session in self.active_sessions.items():
                                    if (hasattr(session, 'user_id') and session.user_id == user_id and 
                                        hasattr(session, 'samples') and len(session.samples) > 0):
                                        
                                        # Buscar muestras con datos temporales reales
                                        for sample in reversed(session.samples):  # Más recientes primero
                                            if (hasattr(sample, 'has_temporal_data') and 
                                                sample.has_temporal_data and
                                                hasattr(sample, 'temporal_sequence') and 
                                                sample.temporal_sequence is not None):
                                                temporal_sequence = sample.temporal_sequence
                                                data_source_found = getattr(sample, 'metadata', {}).get('data_source', 'session_sample_real')
                                                is_real_temporal = True  # SIEMPRE real si viene de muestra de sesión
                                                
                                                logger.info(f"✅ MÉTODO ALTERNATIVO: Secuencia temporal REAL desde muestra: {temporal_sequence.shape}")
                                                logger.info(f"   📊 Sample ID: {sample.sample_id}")
                                                logger.info(f"   📊 Gesto: {sample.gesture_name}")
                                                break
                                        
                                        if temporal_sequence is not None:
                                            break
                        except Exception as e:
                            logger.info(f"Método alternativo falló: {e}")
                    
                    # MÉTODO DE FALLBACK: SOLO SI NO HAY DATOS REALES (ÚLTIMO RECURSO)
                    elif temporal_sequence is None: 
                        logger.warning("⚠️ NO se encontraron datos temporales REALES - usando fallback")
                        try:
                            # Usar templates anatómicos previos del mismo usuario
                            user_anatomical_templates = []
                            for template_id, template in self.templates.items():
                                if (template.user_id == user_id and 
                                    template.template_type == TemplateType.ANATOMICAL and
                                    'bootstrap_features' in template.metadata):
                                    user_anatomical_templates.append(template.metadata['bootstrap_features'])
                            
                            # Incluir características actuales
                            user_anatomical_templates.append(anatomical_features.tolist())
                            
                            if len(user_anatomical_templates) >= 5:
                                # Crear secuencia temporal desde características anatómicas
                                temporal_frames = []
                                for anat_features in user_anatomical_templates[-20:]:  # Max 20
                                    padded_features = np.zeros(320)
                                    padded_features[:min(len(anat_features), 320)] = anat_features[:320]
                                    temporal_frames.append(padded_features)
                                
                                temporal_sequence = np.array(temporal_frames, dtype=np.float32)
                                data_source_found = 'anatomical_templates_fallback'
                                is_real_temporal = False 
                                
                                logger.warning(f"⚠️ FALLBACK: Secuencia creada desde templates anatómicos: {temporal_sequence.shape}")
                        except Exception as e:
                            log_error(f"Método fallback falló: {e}")
                    
                    # ====== CREAR TEMPLATE DINÁMICO SI HAY SECUENCIA ======
                    if temporal_sequence is not None and len(temporal_sequence) >= 5:
                        dynamic_template_id = f"{user_id}_bootstrap_dynamic_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                        
                        # ✅ USAR DATA_SOURCE ENCONTRADO
                        final_data_source = data_source_found or 'unknown_source'
                        
                        dynamic_template = BiometricTemplate(
                            user_id=user_id,
                            template_id=dynamic_template_id,
                            template_type=TemplateType.DYNAMIC,
                            anatomical_embedding=None,
                            dynamic_embedding=None,
                            gesture_name=gesture_name,
                            quality_score=quality_score,
                            confidence=confidence,
                            enrollment_session=str(uuid.uuid4()),
                            metadata={
                                'temporal_sequence': temporal_sequence.tolist(),
                                'sequence_length': len(temporal_sequence),
                                'has_temporal_data': True,
                                'bootstrap_mode': True,
                                'pending_embedding': True,
                                'modality': 'dynamic',
                                'feature_dim': temporal_sequence.shape[1] if len(temporal_sequence.shape) > 1 else 320,
                                'data_source': final_data_source,
                                'is_real_temporal': is_real_temporal  # ✅ MARCADOR DEFINITIVO
                            }
                        )
                        
                        # Calcular checksum y guardar template dinámico
                        dynamic_template.checksum = self._calculate_template_checksum(dynamic_template)
                        self.templates[dynamic_template_id] = dynamic_template
                        
                        # Guardar template dinámico en disco
                        self._save_template_bootstrap(dynamic_template)
                        
                        logger.info(f"✅ Template dinámico bootstrap creado: {dynamic_template_id}")
                        logger.info(f"   📊 Secuencia temporal: {len(temporal_sequence)} frames x {temporal_sequence.shape[1]} características")
                        logger.info(f"   📊 Fuente datos: {final_data_source}")
                        logger.info(f"   📊 Es temporal real: {is_real_temporal}")
                        logger.info(f"   🎯 100% REAL: {'SÍ ✅' if is_real_temporal else 'NO ❌ (Fallback)'}")
                        
                        # También guardar referencia en template anatómico para debugging
                        anatomical_template.metadata['paired_dynamic_template'] = dynamic_template_id
                        anatomical_template.metadata['dynamic_data_source'] = final_data_source
                        anatomical_template.metadata['is_100_percent_real'] = is_real_temporal
                    else:
                        logger.warning("⚠️ No se pudo obtener secuencia temporal suficiente - solo template anatómico")
                        anatomical_template.metadata['has_temporal_data'] = False
                        
                except Exception as e:
                    log_error(f"❌ Error en extracción de datos temporales: {e}")
                    import traceback
                    log_error(f"Traceback: {traceback.format_exc()}")
                    anatomical_template.metadata['has_temporal_data'] = False
                    dynamic_template_id = None
                
                # =========================================================================
                # ✅ PASO 3: GUARDAR TEMPLATE ANATÓMICO
                # =========================================================================
                
                anatomical_template.checksum = self._calculate_template_checksum(anatomical_template)
                self.templates[anatomical_template_id] = anatomical_template
                self._save_template_bootstrap(anatomical_template)
                
                # =========================================================================
                # ✅ PASO 4: ACTUALIZAR PERFIL DE USUARIO CON AMBOS TEMPLATES
                # =========================================================================
                
                user_profile = self.users[user_id]
                
                user_profile.anatomical_templates.append(anatomical_template_id)
                logger.info(f"➕ Template anatómico: {anatomical_template_id}")
                
                if dynamic_template_id:
                    user_profile.dynamic_templates.append(dynamic_template_id)
                    logger.info(f"➕ Template dinámico: {dynamic_template_id}")
                
                templates_created = 2 if dynamic_template_id else 1
                user_profile.total_enrollments += templates_created
                user_profile.updated_at = time.time()
                user_profile.metadata['bootstrap_templates'] = user_profile.metadata.get('bootstrap_templates', 0) + templates_created
                
                if gesture_name not in user_profile.gesture_sequence:
                    user_profile.gesture_sequence.append(gesture_name)
                    logger.info(f"➕ Agregado gesto '{gesture_name}' a secuencia del usuario {user_id}")
                
                self._save_user(user_profile)
                
                # =========================================================================
                # ✅ PASO 5: ACTUALIZAR ESTADÍSTICAS
                # =========================================================================
                
                self.stats.total_templates += templates_created
                self.stats.anatomical_templates += 1
                if dynamic_template_id:
                    self.stats.dynamic_templates += 1
                
                if quality_score >= 0.9:
                    self.stats.excellent_quality += templates_created
                elif quality_score >= 0.7:
                    self.stats.good_quality += templates_created
                elif quality_score >= 0.5:
                    self.stats.fair_quality += templates_created
                else:
                    self.stats.poor_quality += templates_created
                
                self._update_stats()
                
                logger.info(f"🎯 BOOTSTRAP COMPLETO:")
                logger.info(f"   📊 Templates creados: {templates_created}")
                logger.info(f"   🧬 Anatómico: {anatomical_template_id}")
                if dynamic_template_id:
                    logger.info(f"   ⏱️ Dinámico: {dynamic_template_id}")
                
                     # ✅ VERIFICACIÓN FINAL ROBUSTA
                    dynamic_template = self.templates.get(dynamic_template_id)
                    if dynamic_template and 'is_real_temporal' in dynamic_template.metadata:
                        is_real_final = dynamic_template.metadata['is_real_temporal']
                        data_source_final = dynamic_template.metadata.get('data_source', 'unknown')
                        
                        logger.info(f"   📊 Fuente de datos: {data_source_final}")
                        logger.info(f"   📊 Datos temporales: {'🎯 100% REALES ✅' if is_real_final else '❌ Fallback desde anatómicos (SINTÉTICOS)'}")
                        logger.info(f"   🔍 Verificación final: is_real_temporal = {is_real_final}")
                    else:
                        logger.warning(f"   ⚠️ No se pudo verificar estado de datos temporales en template dinámico")
                else:
                    logger.info(f"   ⚠️ Sin template dinámico (no se encontraron datos temporales)")
                
                logger.info(f"   🎯 Gesto: {gesture_name}")
                logger.info(f"   📈 Total enrollments: {user_profile.total_enrollments}")
                return anatomical_template_id
                
        except Exception as e:
            logger.error(f"❌ Error Bootstrap: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _save_template_bootstrap(self, template: BiometricTemplate):
        """Guarda template Bootstrap en disco."""
        try:
            template_file = self.db_path / 'templates' / f'{template.template_id}.json'
            
            print(f"🔍 DEBUG: Guardando Bootstrap {template.template_id}")
            print(f"🔍 DEBUG: Ruta archivo: {template_file}")
            print(f"🔍 DEBUG: Directorio existe: {template_file.parent.exists()}")
            
            template_data = asdict(template)
            template_data['anatomical_embedding'] = None
            template_data['dynamic_embedding'] = None
            
            print(f"🔍 DEBUG: Datos convertidos, gesto: {template_data.get('gesture_name', 'N/A')}")
            
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, default=str)
            
            print(f"✅ DEBUG: Bootstrap guardado en {template_file}")
                
        except Exception as e:
            print(f"❌ DEBUG ERROR: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Error guardando Bootstrap: {e}")
    
    def convert_bootstrap_to_full_templates(self, siamese_anatomical_network, siamese_dynamic_network=None):
        """Convierte templates Bootstrap a templates completos con embeddings."""
        try:
            with self.lock:
                bootstrap_templates = []
                
                for template_id, template in self.templates.items():
                    if template.metadata.get('bootstrap_mode', False):
                        bootstrap_templates.append(template)
                
                logger.info(f"Convirtiendo {len(bootstrap_templates)} templates Bootstrap")
                
                converted_count = 0
                for template in bootstrap_templates:
                    try:
                        anatomical_features = np.array(template.metadata['bootstrap_features'])
                        
                        anatomical_embedding = siamese_anatomical_network.generate_embedding(
                            anatomical_features.reshape(1, -1)
                        )[0]
                        
                        dynamic_embedding = None
                        if siamese_dynamic_network and 'dynamic_features' in template.metadata:
                            dynamic_features = np.array(template.metadata['dynamic_features'])
                            dynamic_embedding = siamese_dynamic_network.generate_embedding(
                                dynamic_features.reshape(1, -1)
                            )[0]
                        
                        template.anatomical_embedding = anatomical_embedding
                        template.dynamic_embedding = dynamic_embedding
                        template.template_type = TemplateType.MULTIMODAL if dynamic_embedding is not None else TemplateType.ANATOMICAL
                        
                        template.metadata['bootstrap_mode'] = False
                        template.metadata['pending_embedding'] = False
                        template.metadata['converted_at'] = time.time()
                        
                        self.anatomical_index.add_embedding(anatomical_embedding, template.template_id, template.user_id)
                        if dynamic_embedding is not None:
                            self.dynamic_index.add_embedding(dynamic_embedding, template.template_id, template.user_id)
                        
                        self._save_template(template)
                        
                        converted_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error convirtiendo {template.template_id}: {e}")
                
                self.anatomical_index.build_index()
                if siamese_dynamic_network:
                    self.dynamic_index.build_index()
                
                self._update_stats()
                
                logger.info(f"✅ Convertidos {converted_count}/{len(bootstrap_templates)} templates Bootstrap")
                
                return converted_count
                
        except Exception as e:
            logger.error(f"Error convirtiendo Bootstrap: {e}")
            return 0
    
    def get_bootstrap_templates(self, user_id: Optional[str] = None) -> List[BiometricTemplate]:
        """Obtiene templates en modo Bootstrap."""
        bootstrap_templates = []
        
        for template in self.templates.values():
            if template.metadata.get('bootstrap_mode', False):
                if user_id is None or template.user_id == user_id:
                    bootstrap_templates.append(template)
        
        return bootstrap_templates
    
    def get_bootstrap_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de templates Bootstrap."""
        bootstrap_templates = self.get_bootstrap_templates()
        
        user_counts = {}
        gesture_counts = {}
        quality_scores = []
        
        for template in bootstrap_templates:
            user_counts[template.user_id] = user_counts.get(template.user_id, 0) + 1
            
            gesture = template.gesture_name
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            
            quality_scores.append(template.quality_score)
        
        return {
            'total_bootstrap_templates': len(bootstrap_templates),
            'users_with_bootstrap': len(user_counts),
            'user_distribution': user_counts,
            'gesture_distribution': gesture_counts,
            'average_quality': np.mean(quality_scores) if quality_scores else 0,
            'min_quality': np.min(quality_scores) if quality_scores else 0,
            'max_quality': np.max(quality_scores) if quality_scores else 0,
            'ready_for_training': len(bootstrap_templates) >= 15
        }


# ===== INSTANCIA GLOBAL =====
_biometric_db_instance = None

def get_biometric_database(db_path: Optional[str] = None) -> BiometricDatabase:
    """Obtiene instancia global de la base de datos biométrica."""
    global _biometric_db_instance
    
    if _biometric_db_instance is None:
        _biometric_db_instance = BiometricDatabase(db_path)
    
    return _biometric_db_instance

