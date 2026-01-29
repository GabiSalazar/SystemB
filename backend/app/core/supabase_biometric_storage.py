"""
MÓDULO: SUPABASE_BIOMETRIC_STORAGE
Base de datos biométrica en Supabase con indexación vectorial
Reemplazo 1:1 de biometric_database.py con backend Supabase
"""

import numpy as np
import json
import time
import uuid
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import warnings
import base64
from pathlib import Path

# Cifrado AES-256
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("WARNING: cryptography no disponible - metadata sin cifrar")

# Supabase client
from app.core.supabase_client import get_supabase_client

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


# ============================================================================
# ENUMS Y DATACLASSES (IDÉNTICOS AL ORIGINAL)
# ============================================================================

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
class AuthenticationAttempt:
    """Registro de un intento de autenticación."""
    attempt_id: str
    user_id: str
    timestamp: float
    auth_type: str
    result: str
    confidence: float
    anatomical_score: float
    dynamic_score: float
    fused_score: float
    ip_address: Optional[str] = None
    device_info: Optional[str] = None
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """Perfil completo de usuario biométrico."""
    user_id: str
    username: str
    
    email: str
    phone_number: str
    age: int
    gender: str
    
    anatomical_templates: List[str] = field(default_factory=list)
    dynamic_templates: List[str] = field(default_factory=list)
    
    gesture_sequence: Optional[List[str]] = None
        
    total_enrollments: int = 0
    total_verifications: int = 0
    successful_verifications: int = 0
    last_activity: float = field(default_factory=time.time)
        
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    is_active: bool = True 
    
    failed_attempts: int = 0
    last_failed_timestamp: Optional[float] = None
    lockout_until: Optional[float] = None
    lockout_history: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def total_templates(self) -> int:
        """Total de templates registrados."""
        return len(self.anatomical_templates) + len(self.dynamic_templates)
    
    @property
    def verification_success_rate(self) -> float:
        """Tasa de éxito en verificaciones."""
        return (self.successful_verifications / self.total_verifications * 100) if self.total_verifications > 0 else 0.0


@dataclass
class PersonalityProfile:
    """Perfil de personalidad basado en cuestionario Big Five simplificado."""
    user_id: str
    
    extraversion_1: int
    agreeableness_1: int
    conscientiousness_1: int
    neuroticism_1: int
    openness_1: int
    extraversion_2: int
    agreeableness_2: int
    conscientiousness_2: int
    neuroticism_2: int
    openness_2: int
    
    raw_responses: str
    
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'user_id': self.user_id,
            'extraversion_1': self.extraversion_1,
            'agreeableness_1': self.agreeableness_1,
            'conscientiousness_1': self.conscientiousness_1,
            'neuroticism_1': self.neuroticism_1,
            'openness_1': self.openness_1,
            'extraversion_2': self.extraversion_2,
            'agreeableness_2': self.agreeableness_2,
            'conscientiousness_2': self.conscientiousness_2,
            'neuroticism_2': self.neuroticism_2,
            'openness_2': self.openness_2,
            'raw_responses': self.raw_responses,
            'completed_at': self.completed_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityProfile':
        """Crea desde diccionario."""
        return cls(
            user_id=data['user_id'],
            extraversion_1=data['extraversion_1'],
            agreeableness_1=data['agreeableness_1'],
            conscientiousness_1=data['conscientiousness_1'],
            neuroticism_1=data['neuroticism_1'],
            openness_1=data['openness_1'],
            extraversion_2=data['extraversion_2'],
            agreeableness_2=data['agreeableness_2'],
            conscientiousness_2=data['conscientiousness_2'],
            neuroticism_2=data['neuroticism_2'],
            openness_2=data['openness_2'],
            raw_responses=data['raw_responses'],
            completed_at=data.get('completed_at', datetime.now().isoformat()),
        )
    
    @classmethod
    def from_responses(cls, user_id: str, responses: List[int]) -> 'PersonalityProfile':
        """Crea desde lista de respuestas."""
        if len(responses) != 10:
            raise ValueError("Se requieren exactamente 10 respuestas")
        
        return cls(
            user_id=user_id,
            extraversion_1=responses[0],
            agreeableness_1=responses[1],
            conscientiousness_1=responses[2],
            neuroticism_1=responses[3],
            openness_1=responses[4],
            extraversion_2=responses[5],
            agreeableness_2=responses[6],
            conscientiousness_2=responses[7],
            neuroticism_2=responses[8],
            openness_2=responses[9],
            raw_responses=','.join(map(str, responses))
        )


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


# ============================================================================
# VECTORINDEX (IDÉNTICO AL ORIGINAL - SE MANTIENE EN MEMORIA)
# ============================================================================

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
            print(f"Índice construido: {len(self.embeddings)} embeddings, estrategia {self.strategy.value}")
            
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
                
                print(f"Template {template_id} eliminado del índice")
                
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


# ============================================================================
# CLASE PRINCIPAL: BiometricDatabase CON SUPABASE
# ============================================================================

class BiometricDatabase:
    """
    Base de datos biométrica en Supabase con indexación vectorial.
    Reemplazo 1:1 de la versión filesystem.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializa la base de datos biométrica con Supabase.
        
        Args:
            db_path: Ignorado (compatibilidad con versión anterior)
        """
        self.logger = get_logger()
        
        # CONFIGURACIÓN
        self.config = self._load_database_config()
        
        # CLIENTE SUPABASE
        self.supabase = get_supabase_client()
        print(f"Cliente Supabase conectado")
        
        # DICCIONARIOS EN MEMORIA
        self.users: Dict[str, UserProfile] = {}
        self.templates: Dict[str, BiometricTemplate] = {}
        
        # ÍNDICES VECTORIALES (EN MEMORIA)
        self.anatomical_index = VectorIndex(
            embedding_dim=64,
            strategy=SearchStrategy(self.config['search_strategy'])
        )
        self.dynamic_index = VectorIndex(
            embedding_dim=128,
            strategy=SearchStrategy(self.config['search_strategy'])
        )
        
        # LOCK Y CACHE
        # self.lock = threading.RLock()
        # self.cache = {}
        # self.stats = DatabaseStats()
        
        # # CARGAR DATOS DESDE SUPABASE
        # self._load_database()

        # LOCK Y CACHE
        self.lock = threading.RLock()
        self.cache = {}
        self.stats = DatabaseStats()
        
        # SISTEMA DE CIFRADO
        self.encryption_enabled = self.config.get('encryption_enabled', True)
        self.cipher_key = None
        self.cipher = None
        
        if self.encryption_enabled and CRYPTO_AVAILABLE:
            self._initialize_encryption_keys()
        else:
            if self.encryption_enabled and not CRYPTO_AVAILABLE:
                print("WARNING: Cifrado solicitado pero cryptography no disponible")
                print("WARNING: Metadata se guardara SIN CIFRAR")
                self.encryption_enabled = False
        
        # CARGAR DATOS DESDE SUPABASE
        self._load_database()
        
        print(f"BiometricDatabase inicializada con Supabase")
    
    # ========================================================================
    # MÉTODOS DE CIFRADO
    # ========================================================================
    
    def _initialize_encryption_keys(self):
        """Inicializa o carga claves de cifrado AES-256."""
        try:
            keys_dir = Path('biometric_data') / 'keys'
            keys_dir.mkdir(parents=True, exist_ok=True)
            
            key_file = keys_dir / 'encryption_key.key'
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.cipher_key = f.read()
                print(f"Clave de cifrado cargada desde {key_file}")
            else:
                self.cipher_key = Fernet.generate_key()
                
                with open(key_file, 'wb') as f:
                    f.write(self.cipher_key)
                
                print("=" * 80)
                print("NUEVA CLAVE DE CIFRADO GENERADA")
                print(f"Ubicacion: {key_file.absolute()}")
                print("CRITICO: Respalda este archivo inmediatamente")
                print("Sin esta clave NO podras descifrar los templates")
                print("=" * 80)
            
            self.cipher = Fernet(self.cipher_key)
            print("Sistema de cifrado inicializado correctamente")
            
        except Exception as e:
            print(f"Error inicializando cifrado: {e}")
            self.encryption_enabled = False
            self.cipher = None

    def _encrypt_metadata(self, metadata: dict):
        """
        Cifra metadata SOLO si cifrado esta habilitado.
        Retorna string cifrado o dict sin cambios.
        """
        if not self.encryption_enabled or not self.cipher:
            return metadata
        
        try:
            json_str = json.dumps(metadata, default=str)
            json_bytes = json_str.encode('utf-8')
            encrypted_bytes = self.cipher.encrypt(json_bytes)
            encrypted_b64 = base64.b64encode(encrypted_bytes).decode('utf-8')
            
            print(f"  Metadata cifrada: {len(metadata)} keys -> {len(encrypted_b64)} chars")
            return encrypted_b64
            
        except Exception as e:
            print(f"  Error cifrando metadata: {e}")
            print(f"  Guardando metadata SIN CIFRAR como fallback")
            return metadata
    
    def _decrypt_metadata(self, metadata_field):
        """
        Descifra metadata CON BACKWARD COMPATIBILITY automatica.
        Retorna dict con metadata descifrada o sin modificar.
        """
        # CASO 1: Ya es dict (formato viejo, sin cifrar)
        if isinstance(metadata_field, dict):
            return metadata_field
        
        # CASO 2: Es None o vacio
        if not metadata_field:
            return {}
        
        # CASO 3: Es string (formato nuevo, cifrado)
        if isinstance(metadata_field, str):
            if not self.encryption_enabled or not self.cipher:
                print(f"  WARNING: Metadata cifrada pero cifrado deshabilitado")
                return {}
            
            try:
                encrypted_bytes = base64.b64decode(metadata_field)
                decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
                json_str = decrypted_bytes.decode('utf-8')
                metadata_dict = json.loads(json_str)
                
                print(f"  Metadata descifrada: {len(metadata_dict)} keys")
                return metadata_dict
                
            except Exception as e:
                print(f"  Error descifrando metadata: {e}")
                print(f"  Retornando metadata vacia por seguridad")
                return {}
        
        # CASO 4: Tipo desconocido
        print(f"  WARNING: Metadata con tipo desconocido: {type(metadata_field)}")
        return {}


    # ========================================================================
    # MÉTODOS DE CONFIGURACIÓN
    # ========================================================================
    
    def _load_database_config(self) -> Dict[str, Any]:
        """Carga configuración de la base de datos."""
        default_config = {
            'encryption_enabled': False,
            'auto_backup': False,  # Supabase tiene su propio backup
            'search_strategy': 'linear',
            'cache_size': 1000,
            'debug_mode': True,
            'verbose_logging': True,
            'max_templates_per_user': 50,
        }
        
        config = get_config('biometric.database', default_config)
        config['encryption_enabled'] = True
        config['debug_mode'] = True
        
        print(f"CONFIG: Encriptación = {config['encryption_enabled']}")
        print(f"CONFIG: Debug mode = {config['debug_mode']}")
        
        return config
    
    # ========================================================================
    # MÉTODO DE CARGA DESDE SUPABASE
    # ========================================================================
    
    def _load_database(self):
        """Carga datos existentes desde Supabase."""
        try:
            users_loaded = 0
            templates_loaded = 0
            
            print("Iniciando carga desde Supabase...")
            
            # CARGAR USUARIOS
            print("Cargando usuarios...")
            try:
                users_response = self.supabase.table('users').select('*').execute()
                
                for user_data in users_response.data:
                    try:
                        # Convertir timestamps si son strings
                        created_at = user_data.get('created_at')
                        if isinstance(created_at, str):
                            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp()
                        
                        updated_at = user_data.get('updated_at')
                        if isinstance(updated_at, str):
                            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00')).timestamp()
                        
                        last_activity = user_data.get('last_activity')
                        if isinstance(last_activity, str):
                            last_activity = datetime.fromisoformat(last_activity.replace('Z', '+00:00')).timestamp()
                        
                        last_failed_timestamp = user_data.get('last_failed_timestamp')
                        if isinstance(last_failed_timestamp, str):
                            last_failed_timestamp = datetime.fromisoformat(last_failed_timestamp.replace('Z', '+00:00')).timestamp()
                        
                        lockout_until = user_data.get('lockout_until')
                        if isinstance(lockout_until, str):
                            lockout_until = datetime.fromisoformat(lockout_until.replace('Z', '+00:00')).timestamp()
                        
                        user_profile = UserProfile(
                            user_id=user_data['user_id'],
                            username=user_data['username'],
                            email=user_data['email'],
                            phone_number=user_data['phone_number'],
                            age=user_data['age'],
                            gender=user_data['gender'],
                            anatomical_templates=user_data.get('anatomical_templates', []),
                            dynamic_templates=user_data.get('dynamic_templates', []),
                            gesture_sequence=user_data.get('gesture_sequence', []),
                            total_enrollments=user_data.get('total_enrollments', 0),
                            total_verifications=user_data.get('total_verifications', 0),
                            successful_verifications=user_data.get('successful_verifications', 0),
                            last_activity=last_activity or time.time(),
                            is_active=user_data.get('is_active', True),
                            failed_attempts=user_data.get('failed_attempts', 0),
                            last_failed_timestamp=last_failed_timestamp,
                            lockout_until=lockout_until,
                            lockout_history=user_data.get('lockout_history', []),
                            created_at=created_at or time.time(),
                            updated_at=updated_at or time.time(),
                            metadata=user_data.get('metadata', {})
                        )
                        
                        self.users[user_profile.user_id] = user_profile
                        users_loaded += 1
                        
                        print(f"Usuario cargado: {user_profile.username} ({user_profile.user_id})")
                        
                    except Exception as user_error:
                        logger.error(f"Error cargando usuario: {user_error}")
                        continue
                        
            except Exception as users_error:
                logger.error(f"Error cargando usuarios: {users_error}")
            
            # CARGAR TEMPLATES
            # print("Cargando templates...")
            # try:
            #     templates_response = self.supabase.table('biometric_templates').select('*').execute()
                
            #     for template_data in templates_response.data:
            
            # # CARGAR TEMPLATES CON PAGINACIÓN EXPLÍCITA
            # print("Cargando templates...")
            # try:
            #     page_size = 100
            #     offset = 0
            #     all_templates = []
                
            #     while True:
            #         print(f"Cargando desde offset {offset}...")
                    
            #         templates_response = self.supabase.table('biometric_templates')\
            #             .select('*')\
            #             .limit(page_size)\
            #             .offset(offset)\
            #             .execute()
                    
            #         if not templates_response.data or len(templates_response.data) == 0:
            #             print(f"  ✓ Sin más datos")
            #             break
                    
            #         num_recibidos = len(templates_response.data)
            #         all_templates.extend(templates_response.data)
            #         print(f"  ✓ {num_recibidos} templates (total: {len(all_templates)})")
                    
            #         # Si recibimos menos que page_size, es la última página
            #         if num_recibidos < page_size:
            #             print(f"  ✓ Última página ({num_recibidos} < {page_size})")
            #             break
                    
            #         # Siguiente página
            #         offset += page_size
                
            #     print(f"Total templates cargados: {len(all_templates)}")
                
            #     for template_data in all_templates:
            #         try:
            #             # Deserializar embeddings
            #             anatomical_emb = None
            #             if template_data.get('anatomical_embedding'):
            #                 anatomical_emb = np.array(template_data['anatomical_embedding'], dtype=np.float32)
                        
            #             dynamic_emb = None
            #             if template_data.get('dynamic_embedding'):
            #                 dynamic_emb = np.array(template_data['dynamic_embedding'], dtype=np.float32)
                        
            #             # Convertir timestamps
            #             created_at = template_data.get('created_at')
            #             if isinstance(created_at, str):
            #                 created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp()
                        
            #             updated_at = template_data.get('updated_at')
            #             if isinstance(updated_at, str):
            #                 updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00')).timestamp()
                        
            #             last_used = template_data.get('last_used')
            #             if isinstance(last_used, str):
            #                 last_used = datetime.fromisoformat(last_used.replace('Z', '+00:00')).timestamp()
                        
            #             # Determinar tipo
            #             template_type_str = template_data.get('template_type', 'anatomical')
            #             if template_type_str == 'anatomical':
            #                 template_type = TemplateType.ANATOMICAL
            #             elif template_type_str == 'dynamic':
            #                 template_type = TemplateType.DYNAMIC
            #             else:
            #                 template_type = TemplateType.MULTIMODAL
                        
            #             template = BiometricTemplate(
            #                 user_id=template_data['user_id'],
            #                 template_id=template_data['template_id'],
            #                 template_type=template_type,
            #                 anatomical_embedding=anatomical_emb,
            #                 dynamic_embedding=dynamic_emb,
            #                 gesture_name=template_data.get('gesture_name', 'unknown'),
            #                 hand_side=template_data.get('hand_side', 'unknown'),
            #                 quality_score=float(template_data.get('quality_score', 1.0)),
            #                 confidence=float(template_data.get('confidence', 1.0)),
            #                 created_at=created_at or time.time(),
            #                 updated_at=updated_at or time.time(),
            #                 last_used=last_used or time.time(),
            #                 enrollment_session=template_data.get('enrollment_session', ''),
            #                 verification_count=template_data.get('verification_count', 0),
            #                 success_count=template_data.get('success_count', 0),
            #                 is_encrypted=False,
            #                 checksum=template_data.get('checksum', ''),
            #                 metadata=template_data.get('metadata', {})
            #             )
                        
            #             self.templates[template.template_id] = template
            #             templates_loaded += 1
                        
            #             # Añadir a índices vectoriales
            #             if anatomical_emb is not None:
            #                 self.anatomical_index.add_embedding(
            #                     anatomical_emb,
            #                     template.template_id,
            #                     template.user_id
            #                 )
                        
            #             if dynamic_emb is not None:
            #                 self.dynamic_index.add_embedding(
            #                     dynamic_emb,
            #                     template.template_id,
            #                     template.user_id
            #                 )
                        
            #             print(f"Template cargado: {template.template_id} ({template.gesture_name})")
                        
            #         except Exception as template_error:
            #             logger.error(f"Error cargando template: {template_error}")
            #             continue
                        
            # except Exception as templates_error:
            #     logger.error(f"Error cargando templates: {templates_error}")
            # ========================================================================
            # CARGAR TEMPLATES CON PAGINACIÓN ROBUSTA Y REINTENTOS
            # ========================================================================
            print("Cargando templates con reintentos automáticos...")

            page_size = 100
            offset = 0
            all_templates = []
            max_retries = 5  # 5 intentos por página
            retry_delay = 3  # 3 segundos entre reintentos

            total_pages = 0
            failed_pages = []

            while True:
                total_pages += 1
                print(f"\n[Página {total_pages}] Offset {offset}...")
                
                retry_count = 0
                page_success = False
                current_page_data = None
                
                # REINTENTAR ESTA PÁGINA HASTA max_retries VECES
                while retry_count < max_retries:
                    try:
                        templates_response = self.supabase.table('biometric_templates')\
                            .select('*')\
                            .limit(page_size)\
                            .offset(offset)\
                            .execute()
                        
                        # Página cargada exitosamente
                        current_page_data = templates_response.data
                        page_success = True
                        break
                        
                    except Exception as page_error:
                        retry_count += 1
                        print(f"   Intento {retry_count}/{max_retries} falló: {str(page_error)[:100]}")
                        
                        if retry_count < max_retries:
                            print(f"   Esperando {retry_delay}s antes de reintentar...")
                            import time as time_module
                            time_module.sleep(retry_delay)
                        else:
                            print(f"  ✗ Página FALLÓ después de {max_retries} intentos")
                            failed_pages.append(offset)
                
                # Evaluar resultado de la página
                if not page_success or not current_page_data:
                    print(f"   No se pudo cargar offset {offset} - ABORTANDO carga")
                    break
                
                # Página vacía = fin de datos
                if len(current_page_data) == 0:
                    print(f"  ✓ Fin de datos (página vacía)")
                    break
                
                # Página cargada exitosamente
                num_recibidos = len(current_page_data)
                all_templates.extend(current_page_data)
                print(f"  ✓ Cargados {num_recibidos} templates (TOTAL: {len(all_templates)})")
                
                # Última página (recibió menos de page_size)
                if num_recibidos < page_size:
                    print(f"  ✓ ÚLTIMA PÁGINA detectada ({num_recibidos} < {page_size})")
                    break
                
                # Siguiente página
                offset += page_size

            # ========================================================================
            # RESUMEN DE CARGA
            # ========================================================================
            print("\n" + "="*70)
            print("RESUMEN DE CARGA DE TEMPLATES")
            print("="*70)
            print(f"Total páginas procesadas: {total_pages}")
            print(f"Templates descargados: {len(all_templates)}")
            print(f"Páginas fallidas: {len(failed_pages)}")
            if failed_pages:
                print(f"Offsets fallidos: {failed_pages}")
            print("="*70 + "\n")

            # ========================================================================
            # PROCESAR TEMPLATES CARGADOS
            # ========================================================================
            print(f"Procesando {len(all_templates)} templates en memoria...")

            templates_procesados_ok = 0
            templates_procesados_error = 0

            for idx, template_data in enumerate(all_templates, 1):
                try:
                    # Mostrar progreso cada 100 templates
                    if idx % 100 == 0:
                        print(f"  Procesando template {idx}/{len(all_templates)}...")
                    
                    # Deserializar embeddings
                    anatomical_emb = None
                    if template_data.get('anatomical_embedding'):
                        anatomical_emb = np.array(template_data['anatomical_embedding'], dtype=np.float32)
                    
                    dynamic_emb = None
                    if template_data.get('dynamic_embedding'):
                        dynamic_emb = np.array(template_data['dynamic_embedding'], dtype=np.float32)
                    
                    # Convertir timestamps
                    created_at = template_data.get('created_at')
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp()
                    
                    updated_at = template_data.get('updated_at')
                    if isinstance(updated_at, str):
                        updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00')).timestamp()
                    
                    last_used = template_data.get('last_used')
                    if isinstance(last_used, str):
                        last_used = datetime.fromisoformat(last_used.replace('Z', '+00:00')).timestamp()
                    
                    # Determinar tipo
                    template_type_str = template_data.get('template_type', 'anatomical')
                    if template_type_str == 'anatomical':
                        template_type = TemplateType.ANATOMICAL
                    elif template_type_str == 'dynamic':
                        template_type = TemplateType.DYNAMIC
                    else:
                        template_type = TemplateType.MULTIMODAL
                    
                    # Crear objeto BiometricTemplate
                    template = BiometricTemplate(
                        user_id=template_data['user_id'],
                        template_id=template_data['template_id'],
                        template_type=template_type,
                        anatomical_embedding=anatomical_emb,
                        dynamic_embedding=dynamic_emb,
                        gesture_name=template_data.get('gesture_name', 'unknown'),
                        hand_side=template_data.get('hand_side', 'unknown'),
                        quality_score=float(template_data.get('quality_score', 1.0)),
                        confidence=float(template_data.get('confidence', 1.0)),
                        created_at=created_at or time.time(),
                        updated_at=updated_at or time.time(),
                        last_used=last_used or time.time(),
                        enrollment_session=template_data.get('enrollment_session', ''),
                        verification_count=template_data.get('verification_count', 0),
                        success_count=template_data.get('success_count', 0),
                        is_encrypted=False,
                        checksum=template_data.get('checksum', ''),
                        metadata=self._decrypt_metadata(template_data.get('metadata', {}))
                    )
                    
                    # Guardar en memoria
                    self.templates[template.template_id] = template
                    templates_loaded += 1
                    templates_procesados_ok += 1
                    
                    # Añadir a índices vectoriales (solo si tiene embeddings)
                    if anatomical_emb is not None:
                        self.anatomical_index.add_embedding(
                            anatomical_emb,
                            template.template_id,
                            template.user_id
                        )
                    
                    if dynamic_emb is not None:
                        self.dynamic_index.add_embedding(
                            dynamic_emb,
                            template.template_id,
                            template.user_id
                        )
                    
                except Exception as template_error:
                    templates_procesados_error += 1
                    logger.error(f"Error procesando template {idx}: {template_error}")
                    continue

            print(f"\n✓ Procesamiento completado:")
            print(f"  - Templates en memoria: {len(self.templates)}")
            print(f"  - Procesados OK: {templates_procesados_ok}")
            print(f"  - Errores: {templates_procesados_error}")

            # ========================================================================
            # VALIDACIÓN FINAL
            # ========================================================================
            print("\n" + "="*70)
            print("VALIDACIÓN FINAL")
            print("="*70)

            # Contar en Supabase (query rápida)
            try:
                count_response = self.supabase.table('biometric_templates')\
                    .select('template_id', count='exact')\
                    .execute()
                
                total_en_supabase = count_response.count if hasattr(count_response, 'count') else len(all_templates)
                
                print(f"Templates en Supabase: {total_en_supabase}")
                print(f"Templates en memoria:  {len(self.templates)}")
                
                if len(self.templates) == total_en_supabase:
                    print("✓✓✓ TODOS LOS TEMPLATES CARGADOS CORRECTAMENTE ✓✓✓")
                else:
                    diferencia = total_en_supabase - len(self.templates)
                    print(f" FALTAN {diferencia} TEMPLATES ")
                    
            except Exception as count_error:
                print(f"No se pudo validar count: {count_error}")

            print("="*70 + "\n")


            # CONSTRUIR ÍNDICES
            print(" Construyendo índices vectoriales...")
            self.anatomical_index.build_index()
            self.dynamic_index.build_index()
            
            # ACTUALIZAR ESTADÍSTICAS
            self.stats.total_users = users_loaded
            self.stats.total_templates = templates_loaded
            self._update_stats()
            
            print("=" * 60)
            print("CARGA COMPLETADA")
            print("=" * 60)
            print(f"USUARIOS: {users_loaded}")
            print(f"TEMPLATES: {templates_loaded}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR CRÍTICO CARGANDO BD: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    # ========================================================================
    # MÉTODOS DE VALIDACIÓN
    # ========================================================================
    
    # def is_email_unique(self, email: str, exclude_user_id: Optional[str] = None) -> bool:
    #     """Verifica si el email es único."""
    #     try:
    #         with self.lock:
    #             query = self.supabase.table('users').select('user_id').eq('email', email)
                
    #             if exclude_user_id:
    #                 query = query.neq('user_id', exclude_user_id)
                
    #             response = query.execute()
                
    #             is_unique = len(response.data) == 0
                
    #             if not is_unique:
    #                 print(f"Email {email} ya registrado")
                
    #             return is_unique
                
    #     except Exception as e:
    #         logger.error(f"Error verificando email único: {e}")
    #         return False
    
    def is_email_unique(self, email: str, exclude_user_id: Optional[str] = None) -> bool:
        """Verifica si el email es único entre usuarios activos."""
        try:
            with self.lock:
                query = self.supabase.table('users').select('user_id').eq('email', email).eq('is_active', True)
                
                if exclude_user_id:
                    query = query.neq('user_id', exclude_user_id)
                
                response = query.execute()
                
                is_unique = len(response.data) == 0
                
                if not is_unique:
                    print(f"Email {email} ya registrado para usuario activo")
                
                return is_unique
                
        except Exception as e:
            logger.error(f"Error verificando email único: {e}")
            return False
    
    # def is_phone_unique(self, phone_number: str, exclude_user_id: Optional[str] = None) -> bool:
    #     """Verifica si el teléfono es único."""
    #     try:
    #         with self.lock:
    #             query = self.supabase.table('users').select('user_id').eq('phone_number', phone_number)
                
    #             if exclude_user_id:
    #                 query = query.neq('user_id', exclude_user_id)
                
    #             response = query.execute()
                
    #             is_unique = len(response.data) == 0
                
    #             if not is_unique:
    #                 print(f"Teléfono {phone_number} ya registrado")
                
    #             return is_unique
                
    #     except Exception as e:
    #         logger.error(f"Error verificando teléfono único: {e}")
    #         return False
    
    def is_phone_unique(self, phone_number: str, exclude_user_id: Optional[str] = None) -> bool:
        """Verifica si el teléfono es único entre usuarios activos."""
        try:
            with self.lock:
                query = self.supabase.table('users').select('user_id').eq('phone_number', phone_number).eq('is_active', True)
                
                if exclude_user_id:
                    query = query.neq('user_id', exclude_user_id)
                
                response = query.execute()
                
                is_unique = len(response.data) == 0
                
                if not is_unique:
                    print(f"Teléfono {phone_number} ya registrado para usuario activo")
                
                return is_unique
                
        except Exception as e:
            logger.error(f"Error verificando teléfono único: {e}")
            return False
    
    def generate_unique_user_id(self, username: str) -> str:
        """Genera un ID único para un nuevo usuario."""
        import uuid
        import time
        
        timestamp = int(time.time() * 1000)
        unique_suffix = uuid.uuid4().hex[:8]
        clean_name = ''.join(c for c in username.lower() if c.isalnum())[:8]
        
        user_id = f"user_{clean_name}_{timestamp}_{unique_suffix}"
        
        while user_id in self.users:
            unique_suffix = uuid.uuid4().hex[:8]
            user_id = f"user_{clean_name}_{timestamp}_{unique_suffix}"
        
        print(f"ID generado: {user_id}")
        return user_id
    
    def get_user_by_email(self, email: str, active_only: bool = True) -> Optional[UserProfile]:
        """
        Obtiene usuario por email.
        
        Args:
            email: Email del usuario
            active_only: Si True, solo busca usuarios activos
        
        Returns:
            UserProfile o None
        """
        try:
            with self.lock:
                query = self.supabase.table('users').select('*').eq('email', email.lower().strip())
                
                if active_only:
                    query = query.eq('is_active', True)
                
                response = query.execute()
                
                if not response.data:
                    print(f"No se encontró usuario con email: {email}")
                    return None
                
                user_data = response.data[0]
                
                user = UserProfile(
                    user_id=user_data['user_id'],
                    username=user_data['username'],
                    email=user_data['email'],
                    phone_number=user_data['phone_number'],
                    age=user_data['age'],
                    gender=user_data['gender'],
                    gesture_sequence=user_data.get('gesture_sequence', []),
                    is_active=user_data.get('is_active', True),
                    created_at=user_data.get('created_at'),
                    updated_at=user_data.get('updated_at'),
                    metadata=user_data.get('metadata', {})
                )
                
                # Cargar templates si ya están cargados
                if user.user_id in self.users:
                    existing_user = self.users[user.user_id]
                    user.anatomical_templates = existing_user.anatomical_templates
                    user.dynamic_templates = existing_user.dynamic_templates
                
                print(f"Usuario encontrado: {user.user_id}")
                return user
                
        except Exception as e:
            logger.error(f"Error obteniendo usuario por email: {e}")
            return None
        
    # def deactivate_user_and_rename(self, user_id: str, reason: str = "forgot_sequence") -> dict:
    #     from datetime import datetime
    #     import traceback
        
    #     try:
    #         with self.lock:
    #             user = self.get_user(user_id)
    #             if not user:
    #                 raise ValueError(f"Usuario {user_id} no encontrado")
                
    #             timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    #             new_inactive_id = f"{user_id}_inactive_{timestamp}"
                
    #             print(f"Desactivando usuario: {user_id} -> {new_inactive_id}")
                
    #             new_metadata = {
    #                 **user.metadata,
    #                 'original_user_id': user_id,
    #                 'deactivation_reason': reason,
    #                 'deactivated_at': datetime.now().isoformat()
    #             }
                
    #             # Solo actualiza users, CASCADE hace el resto automáticamente
    #             self.supabase.table('users').update({
    #                 'user_id': new_inactive_id,
    #                 'is_active': False,
    #                 'metadata': new_metadata
    #             }).eq('user_id', user_id).execute()
                
    #             print(f"Usuario desactivado: {user_id} -> {new_inactive_id}")
                
    #             # Actualizar cache local
    #             if user_id in self.users:
    #                 del self.users[user_id]
                
    #             personality_profile = None
    #             try:
    #                 personality_profile = self.get_personality_profile(new_inactive_id)
    #             except Exception as e:
    #                 logger.warning(f"No se pudo obtener perfil de personalidad: {e}")
                
    #             return {
    #                 'success': True,
    #                 'original_user_id': user_id,
    #                 'new_inactive_id': new_inactive_id,
    #                 'user_data': {
    #                     'email': user.email,
    #                     'phone_number': user.phone_number,
    #                     'age': user.age,
    #                     'gender': user.gender,
    #                     'username': user.username,
    #                     'gesture_sequence': user.gesture_sequence
    #                 },
    #                 'personality_profile': personality_profile
    #             }
                
    #     except Exception as e:
    #         logger.error(f"Error desactivando usuario {user_id}: {e}")
    #         logger.error(f"Traceback: {traceback.format_exc()}")
    #         raise
    
    def deactivate_user_and_rename(self, user_id: str, reason: str = "forgot_sequence") -> dict:
        from datetime import datetime
        import traceback
        
        try:
            with self.lock:
                print("=" * 80)
                print(f"INICIANDO DEACTIVATE_USER_AND_RENAME")
                print(f"   User ID original: {user_id}")
                print("=" * 80)
                
                user = self.get_user(user_id)
                if not user:
                    raise ValueError(f"Usuario {user_id} no encontrado")
                
                print(f"Usuario encontrado en memoria: {user.username}")
                
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                new_inactive_id = f"{user_id}_inactive_{timestamp}"
                
                print(f"Nuevo ID inactivo: {new_inactive_id}")
                
                new_metadata = {
                    **user.metadata,
                    'original_user_id': user_id,
                    'deactivation_reason': reason,
                    'deactivated_at': datetime.now().isoformat()
                }
                
                print(f"Metadata actualizado: {new_metadata}")
                
                # VERIFICAR QUE EL USUARIO EXISTE EN SUPABASE ANTES DEL UPDATE
                print(f"\nVerificando usuario en Supabase...")
                check = self.supabase.table('users').select('id, user_id, is_active').eq('user_id', user_id).execute()
                print(f"   Resultado verificación: {check.data}")
                
                if not check.data:
                    raise ValueError(f"Usuario {user_id} no existe en Supabase")
                
                print(f"Usuario confirmado en Supabase")
                print(f"   ID interno Supabase: {check.data[0]['id']}")
                
                # EJECUTAR UPDATE
                print(f"\nEJECUTANDO UPDATE...")
                print(f"   Cambiando user_id: {user_id} -> {new_inactive_id}")
                print(f"   Cambiando is_active: True -> False")

                # LIMPIAR EMAIL Y TELÉFONO para liberar constraints
                fake_email = f"{new_inactive_id}@inactive.deleted"
                fake_phone = f"INACTIVE_{int(time.time())}"  # Timestamp único

                print(f"   Cambiando email a: {fake_email}")
                print(f"   Cambiando teléfono a: {fake_phone}")

                try:
                    result = self.supabase.table('users').update({
                        'user_id': new_inactive_id,
                        'email': fake_email,
                        'phone_number': fake_phone,
                        'is_active': False,
                        'metadata': new_metadata,
                        'updated_at': datetime.now().isoformat()
                    }).eq('user_id', user_id).execute()
                    
                    print(f"\nRESULTADO DEL UPDATE:")
                    print(f"   Status code: {getattr(result, 'status_code', 'N/A')}")
                    print(f"   Data: {result.data}")
                    print(f"   Count: {getattr(result, 'count', 'N/A')}")
                    
                    if result.data:
                        print(f"UPDATE EXITOSO - {len(result.data)} registro(s) afectado(s)")
                        print(f"   Nuevo user_id en BD: {result.data[0].get('user_id')}")
                    else:
                        print(f"UPDATE FALLÓ - No se afectaron registros")
                        raise Exception("UPDATE no afectó ningún registro")
                    
                except Exception as update_error:
                    print(f"\nEXCEPCIÓN EN UPDATE:")
                    print(f"   Error: {update_error}")
                    print(f"   Tipo: {type(update_error)}")
                    print(f"   Traceback:")
                    traceback.print_exc()
                    raise
                
                # VERIFICAR QUE EL CAMBIO SE APLICÓ
                print(f"\nVerificando cambio en Supabase...")
                verify = self.supabase.table('users').select('user_id, is_active').eq('user_id', new_inactive_id).execute()
                print(f"   Búsqueda por nuevo ID: {verify.data}")
                
                verify_old = self.supabase.table('users').select('user_id, is_active').eq('user_id', user_id).execute()
                print(f"   Búsqueda por ID original: {verify_old.data}")
                
                print(f"Usuario renombrado: {user_id} -> {new_inactive_id}")
                
                # Actualizar cache local
                if user_id in self.users:
                    del self.users[user_id]
                
                personality_profile = None
                try:
                    personality_profile = self.get_personality_profile(new_inactive_id)
                except Exception as e:
                    logger.warning(f"No se pudo obtener perfil de personalidad: {e}")
                
                print("=" * 80)
                print(f"DEACTIVATE_USER_AND_RENAME COMPLETADO")
                print("=" * 80)
                
                return {
                    'success': True,
                    'original_user_id': user_id,
                    'new_inactive_id': new_inactive_id,
                    'user_data': {
                        'email': user.email,
                        'phone_number': user.phone_number,
                        'age': user.age,
                        'gender': user.gender,
                        'username': user.username,
                        'gesture_sequence': user.gesture_sequence
                    },
                    'personality_profile': personality_profile
                }
                
        except Exception as e:
            print("=" * 80)
            print(f"ERROR EN DEACTIVATE_USER_AND_RENAME")
            print(f"   Error: {e}")
            print(f"   Traceback:")
            traceback.print_exc()
            print("=" * 80)
            logger.error(f"Error desactivando usuario {user_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
    def reactivate_user(self, original_user_id: str) -> bool:
        """
        Reactiva un usuario inactivo renombrándolo a su ID original.
        
        Args:
            original_user_id: ID original del usuario (sin sufijo _inactive_)
        
        Returns:
            bool: True si se reactivó exitosamente
        """
        from datetime import datetime
        import traceback
        
        try:
            with self.lock:
                # Buscar usuario inactivo con este ID original en metadata
                response = self.supabase.table('users')\
                    .select('*')\
                    .eq('is_active', False)\
                    .execute()
                
                if not response.data:
                    logger.warning(f"No hay usuarios inactivos")
                    return False
                
                # Buscar el que tiene original_user_id en metadata
                inactive_user = None
                for user in response.data:
                    metadata = user.get('metadata', {})
                    if metadata.get('original_user_id') == original_user_id:
                        inactive_user = user
                        break
                
                if not inactive_user:
                    logger.warning(f"No se encontró usuario inactivo con original_user_id: {original_user_id}")
                    return False
                
                old_user_id = inactive_user['user_id']
                
                print(f"Reactivando usuario: {old_user_id} → {original_user_id}")
                
                # Actualizar metadata
                new_metadata = inactive_user.get('metadata', {})
                new_metadata['reactivated_at'] = datetime.now().isoformat()
                new_metadata['reactivation_count'] = new_metadata.get('reactivation_count', 0) + 1
                
                # UPDATE en Supabase: renombrar y reactivar
                self.supabase.table('users').update({
                    'user_id': original_user_id,
                    'is_active': True,
                    'failed_attempts': 0,
                    'last_failed_timestamp': None,
                    'lockout_until': None,
                    'updated_at': datetime.now().isoformat(),
                    'metadata': new_metadata
                }).eq('user_id', old_user_id).execute()
                
                print(f"Usuario reactivado exitosamente: {original_user_id}")
                
                # Actualizar cache local si existe
                if old_user_id in self.users:
                    user_profile = self.users[old_user_id]
                    user_profile.user_id = original_user_id
                    user_profile.is_active = True
                    user_profile.failed_attempts = 0
                    user_profile.last_failed_timestamp = None
                    user_profile.lockout_until = None
                    user_profile.metadata = new_metadata
                    
                    # Mover en cache
                    self.users[original_user_id] = user_profile
                    del self.users[old_user_id]
                
                return True
                
        except Exception as e:
            logger.error(f"Error reactivando usuario {original_user_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    # ========================================================================
    # MÉTODOS DE USUARIO
    # ========================================================================
    
    def create_user(self, user_id: str, username: str,
                    email: str, phone_number: str, age: int, gender: str,
                    gesture_sequence: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Crea un nuevo usuario."""
        try:
            with self.lock:
                if user_id in self.users:
                    logger.error(f"Usuario {user_id} ya existe")
                    return False
                
                if not self.is_email_unique(email, exclude_user_id=user_id):
                    logger.error(f"El email {email} ya está registrado.")
                    return False
                
                if not self.is_phone_unique(phone_number, exclude_user_id=user_id):
                    logger.error(f"El teléfono {phone_number} ya está registrado.")
                    return False
                
                if not email or not phone_number:
                    logger.error("Email y teléfono son requeridos")
                    return False
                
                try:
                    age = int(age)
                except ValueError:
                    logger.error(f"Edad inválida: {age}")
                    return False
                
                if age < 1 or age > 120:
                    logger.error(f"Edad inválida: {age}")
                    return False
                
                if gender not in ["Femenino", "Masculino"]:
                    logger.error(f"Género inválido: {gender}")
                    return False
                
                user_profile = UserProfile(
                    user_id=user_id,
                    username=username,
                    email=email,
                    phone_number=phone_number,
                    age=age,
                    gender=gender,
                    gesture_sequence=gesture_sequence or [],
                    metadata=metadata or {}
                )
                
                self.users[user_id] = user_profile
                self._save_user(user_profile)
                
                self.stats.total_users += 1
                self._update_stats()
                
                print(f"Usuario creado: {user_id} ({username})")
                
                return True
                
        except Exception as e:
            logger.error(f"Error creando usuario: {e}")
            return False
    
    def store_user_profile(self, user_profile: UserProfile) -> bool:
        """Almacena un perfil de usuario completo."""
        try:
            with self.lock:
                print(f"Almacenando perfil de usuario: {user_profile.user_id}")
                
                if user_profile.user_id in self.users:
                    print(f"Usuario {user_profile.user_id} existe - actualizando")
                    
                    existing_user = self.users[user_profile.user_id]
                    
                    existing_user.username = user_profile.username
                    existing_user.gesture_sequence = user_profile.gesture_sequence
                    existing_user.updated_at = time.time()
                    
                    if hasattr(user_profile, 'metadata'):
                        existing_user.metadata.update(user_profile.metadata or {})
                    
                    # Copiar campos adicionales si existen
                    for attr in ['total_samples', 'valid_samples', 'enrollment_duration', 'quality_score', 'enrollment_date']:
                        if hasattr(user_profile, attr):
                            setattr(existing_user, attr, getattr(user_profile, attr))
                    
                    self._save_user(existing_user)
                    
                    print(f"Usuario {user_profile.user_id} actualizado")
                    return True
                    
                else:
                    print(f"Creando nuevo usuario: {user_profile.user_id}")
                    
                    self.users[user_profile.user_id] = user_profile
                    
                    # Inicializar listas si no existen
                    if not hasattr(user_profile, 'anatomical_templates'):
                        user_profile.anatomical_templates = []
                    if not hasattr(user_profile, 'dynamic_templates'):
                        user_profile.dynamic_templates = []
                    
                    self._save_user(user_profile)
                    
                    self.stats.total_users += 1
                    self._update_stats()
                    
                    print(f"Usuario {user_profile.user_id} creado exitosamente")
                    return True
                    
        except Exception as e:
            logger.error(f"Error almacenando perfil {user_profile.user_id}: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Obtiene perfil de usuario."""
        return self.users.get(user_id)
    
    # def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
    #     """Actualiza información de un usuario."""
    #     try:
    #         with self.lock:
    #             if user_id not in self.users:
    #                 logger.error(f"Usuario {user_id} no existe")
    #                 return False
                
    #             user = self.users[user_id]
                
    #             # Validar email único
    #             if 'email' in updates and updates['email']:
    #                 if not self.is_email_unique(updates['email'], exclude_user_id=user_id):
    #                     logger.error(f"Email {updates['email']} ya está registrado")
    #                     return False
    #                 user.email = updates['email']
                
    #             # Validar teléfono único
    #             if 'phone_number' in updates and updates['phone_number']:
    #                 if not self.is_phone_unique(updates['phone_number'], exclude_user_id=user_id):
    #                     logger.error(f"Teléfono {updates['phone_number']} ya está registrado")
    #                     return False
    #                 user.phone_number = updates['phone_number']
                
    #             # Actualizar otros campos
    #             if 'username' in updates:
    #                 user.username = updates['username']
                
    #             if 'age' in updates:
    #                 age = int(updates['age'])
    #                 if age < 1 or age > 120:
    #                     logger.error("Edad inválida")
    #                     return False
    #                 user.age = age
                
    #             if 'gender' in updates:
    #                 if updates['gender'] not in ["Femenino", "Masculino"]:
    #                     logger.error("Género inválido")
    #                     return False
    #                 user.gender = updates['gender']
                
    #             if 'gesture_sequence' in updates:
    #                 user.gesture_sequence = updates['gesture_sequence']
                
    #             user.updated_at = time.time()
    #             self._save_user(user)
                
    #             print(f"Usuario {user_id} actualizado exitosamente")
    #             return True
                
    #     except Exception as e:
    #         logger.error(f"Error actualizando usuario: {e}")
    #         return False
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Actualiza información de un usuario."""
        try:
            from datetime import datetime
            
            with self.lock:
                if user_id not in self.users:
                    logger.error(f"Usuario {user_id} no existe")
                    return False
                
                user = self.users[user_id]
                
                # Validar email único
                if 'email' in updates and updates['email']:
                    if not self.is_email_unique(updates['email'], exclude_user_id=user_id):
                        logger.error(f"Email {updates['email']} ya está registrado")
                        return False
                    user.email = updates['email']
                
                # Validar teléfono único
                if 'phone_number' in updates and updates['phone_number']:
                    if not self.is_phone_unique(updates['phone_number'], exclude_user_id=user_id):
                        logger.error(f"Teléfono {updates['phone_number']} ya está registrado")
                        return False
                    user.phone_number = updates['phone_number']
                
                # Actualizar username
                if 'username' in updates:
                    user.username = updates['username']
                
                # Actualizar age
                if 'age' in updates:
                    age = int(updates['age'])
                    if age < 1 or age > 120:
                        logger.error("Edad inválida")
                        return False
                    user.age = age
                
                # Actualizar gender
                if 'gender' in updates:
                    if updates['gender'] not in ["Femenino", "Masculino"]:
                        logger.error("Género inválido")
                        return False
                    user.gender = updates['gender']
                
                # Actualizar gesture_sequence
                if 'gesture_sequence' in updates:
                    user.gesture_sequence = updates['gesture_sequence']
                
                # ACTUALIZAR is_active
                if 'is_active' in updates:
                    is_active = updates['is_active']
                    print(f"\n{'='*80}")
                    print(f"ACTUALIZANDO is_active PARA {user_id}")
                    print(f"  Valor anterior: {getattr(user, 'is_active', True)}")
                    print(f"  Valor nuevo: {is_active}")
                    print(f"{'='*80}\n")
                    
                    # ASEGURAR que el atributo existe y se actualiza
                    user.is_active = is_active
                    
                    # Verificar que se actualizó
                    print(f"✓ user.is_active después de asignar: {user.is_active}")
                    
                    # Agregar metadata de desactivación
                    if not is_active:
                        if not hasattr(user, 'metadata') or user.metadata is None:
                            user.metadata = {}
                        user.metadata['deactivated_at'] = datetime.now().isoformat()
                        user.metadata['deactivated_by'] = 'admin'
                
                # Actualizar timestamp
                user.updated_at = time.time()
                
                # GUARDAR con _save_user (que debe incluir is_active)
                print(f"\nLlamando a _save_user() con is_active={user.is_active}")
                self._save_user(user)
                
                print(f"Usuario {user_id} actualizado exitosamente")
                return True
                
        except Exception as e:
            logger.error(f"Error actualizando usuario: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # def list_users(self) -> List[UserProfile]:
    #     """Lista todos los usuarios."""
    #     return list(self.users.values())
    
    def list_users(self, active_only: bool = True) -> List[UserProfile]:
        """
        Lista usuarios del sistema.
        
        Args:
            active_only: Si True, solo devuelve usuarios activos
        
        Returns:
            Lista de perfiles de usuario
        """
        if active_only:
            return [user for user in self.users.values() if user.is_active]
        return list(self.users.values())
    
    def delete_user(self, user_id: str) -> bool:
        """Elimina un usuario y todos sus templates (CASCADE)."""
        try:
            with self.lock:
                if user_id not in self.users:
                    logger.error(f"Usuario {user_id} no existe")
                    return False
                
                # ELIMINAR EN SUPABASE (CASCADE automático)
                self.supabase.table('users').delete().eq('user_id', user_id).execute()
                
                # Eliminar templates del índice en memoria
                user_templates = self.list_user_templates(user_id)
                for template in user_templates:
                    self.anatomical_index.remove_template(template.template_id)
                    self.dynamic_index.remove_template(template.template_id)
                    if template.template_id in self.templates:
                        del self.templates[template.template_id]
                
                # Eliminar de memoria
                del self.users[user_id]
                
                self.stats.total_users -= 1
                self._update_stats()
                
                print(f"Usuario eliminado: {user_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error eliminando usuario: {e}")
            return False
    
    # ========================================================================
    # MÉTODOS DE LOCKOUT
    # ========================================================================
    
    def check_if_locked(self, user_id: str) -> Tuple[bool, int]:
        """Verifica si un usuario está bloqueado."""
        try:
            with self.lock:
                if user_id not in self.users:
                    logger.warning(f"Usuario {user_id} no existe")
                    return False, 0
                
                user = self.users[user_id]
                
                if not hasattr(user, 'lockout_until') or user.lockout_until is None:
                    return False, 0
                
                current_time = time.time()
                
                if current_time < user.lockout_until:
                    remaining_seconds = user.lockout_until - current_time
                    remaining_minutes = int(remaining_seconds / 60) + 1
                    print(f"Usuario {user_id} bloqueado. Tiempo restante: {remaining_minutes} minutos")
                    return True, remaining_minutes
                else:
                    user.lockout_until = None
                    user.failed_attempts = 0
                    self._save_user(user)
                    print(f"Bloqueo de usuario {user_id} expirado")
                    return False, 0
                    
        except Exception as e:
            logger.error(f"Error verificando bloqueo: {e}")
            return False, 0
    
    def record_failed_attempt(self, user_id: str) -> int:
        """Registra un intento fallido de autenticación."""
        try:
            with self.lock:
                if user_id not in self.users:
                    logger.error(f"Usuario {user_id} no existe")
                    return 0
                
                user = self.users[user_id]
                
                if not hasattr(user, 'failed_attempts'):
                    user.failed_attempts = 0
                
                user.failed_attempts += 1
                user.last_failed_timestamp = time.time()
                
                self._save_user(user)
                
                logger.warning(f"Intento fallido registrado para {user_id}. Total: {user.failed_attempts}")
                
                return user.failed_attempts
                
        except Exception as e:
            logger.error(f"Error registrando intento fallido: {e}")
            return 0
    
    def lock_account(self, user_id: str, duration_minutes: int) -> float:
        """Bloquea una cuenta por un período de tiempo."""
        try:
            with self.lock:
                if user_id not in self.users:
                    logger.error(f"Usuario {user_id} no existe")
                    return 0.0
                
                user = self.users[user_id]
                
                current_time = time.time()
                lockout_until = current_time + (duration_minutes * 60)
                
                user.lockout_until = lockout_until
                
                if not hasattr(user, 'lockout_history'):
                    user.lockout_history = []
                
                lockout_record = {
                    'locked_at': current_time,
                    'lockout_until': lockout_until,
                    'duration_minutes': duration_minutes,
                    'failed_attempts': user.failed_attempts,
                    'reason': 'multiple_failed_attempts'
                }
                user.lockout_history.append(lockout_record)
                
                self._save_user(user)
                
                logger.warning(f"Cuenta {user_id} bloqueada por {duration_minutes} minutos")
                
                return lockout_until
                
        except Exception as e:
            logger.error(f"Error bloqueando cuenta: {e}")
            return 0.0
    
    def reset_failed_attempts(self, user_id: str) -> None:
        """Resetea el contador de intentos fallidos."""
        try:
            with self.lock:
                if user_id not in self.users:
                    logger.warning(f"Usuario {user_id} no existe")
                    return
                
                user = self.users[user_id]
                
                if hasattr(user, 'failed_attempts') and user.failed_attempts > 0:
                    previous_attempts = user.failed_attempts
                    user.failed_attempts = 0
                    user.last_failed_timestamp = None
                    
                    self._save_user(user)
                    
                    print(f"Intentos fallidos reseteados para {user_id} (tenía {previous_attempts})")
                    
        except Exception as e:
            logger.error(f"Error reseteando intentos fallidos: {e}")
    
    def get_lockout_info(self, user_id: str) -> Dict[str, Any]:
        """Obtiene información completa del estado de bloqueo."""
        try:
            with self.lock:
                if user_id not in self.users:
                    return {
                        'exists': False,
                        'is_locked': False,
                        'failed_attempts': 0,
                        'lockout_history': []
                    }
                
                user = self.users[user_id]
                
                is_locked, remaining_minutes = self.check_if_locked(user_id)
                
                return {
                    'exists': True,
                    'is_locked': is_locked,
                    'remaining_minutes': remaining_minutes,
                    'failed_attempts': getattr(user, 'failed_attempts', 0),
                    'last_failed_timestamp': getattr(user, 'last_failed_timestamp', None),
                    'lockout_until': getattr(user, 'lockout_until', None),
                    'lockout_history': getattr(user, 'lockout_history', [])
                }
                
        except Exception as e:
            logger.error(f"Error obteniendo info de bloqueo: {e}")
            return {
                'exists': False,
                'is_locked': False,
                'error': str(e)
            }
    
    # ========================================================================
    # MÉTODOS DE TEMPLATES
    # ========================================================================
    
    def store_biometric_template(self, template: BiometricTemplate) -> bool:
        """Almacena template biométrico en Supabase."""
        try:
            with self.lock:
                print(f"Almacenando template: {template.template_id}")
                
                if template.user_id not in self.users:
                    logger.error(f"Usuario {template.user_id} no existe")
                    return False
                
                # Calcular checksum
                template.checksum = self._calculate_template_checksum(template)
                
                # Guardar en memoria
                self.templates[template.template_id] = template
                
                # Añadir a índices
                if hasattr(template, 'anatomical_embedding') and template.anatomical_embedding is not None:
                    try:
                        self.anatomical_index.add_embedding(
                            template.anatomical_embedding, 
                            template.template_id, 
                            template.user_id
                        )
                        print(f"Template anatómico agregado al índice")
                    except Exception as e:
                        print(f"Error índice anatómico: {e}")
                        
                if hasattr(template, 'dynamic_embedding') and template.dynamic_embedding is not None:
                    try:
                        self.dynamic_index.add_embedding(
                            template.dynamic_embedding, 
                            template.template_id, 
                            template.user_id
                        )
                        print(f"Template dinámico agregado al índice")
                    except Exception as e:
                        print(f"Error índice dinámico: {e}")
                
                # Actualizar perfil de usuario
                user_profile = self.users[template.user_id]
                
                if template.template_type == TemplateType.ANATOMICAL:
                    if template.template_id not in user_profile.anatomical_templates:
                        user_profile.anatomical_templates.append(template.template_id)
                elif template.template_type == TemplateType.DYNAMIC:
                    if template.template_id not in user_profile.dynamic_templates:
                        user_profile.dynamic_templates.append(template.template_id)
                
                user_profile.total_enrollments += 1
                user_profile.updated_at = time.time()
                
                # Guardar en Supabase
                try:
                    self._save_template(template)
                    print(f"Template guardado en Supabase")
                except Exception as e:
                    print(f"ERROR guardando template: {e}")
                    return False
                                    
                try:
                    self._save_user(user_profile)
                    print(f"Usuario actualizado en Supabase")
                except Exception as e:
                    print(f"ERROR guardando usuario: {e}")
                    return False
                
                # Actualizar estadísticas
                self.stats.total_templates += 1
                if template.template_type == TemplateType.ANATOMICAL:
                    self.stats.anatomical_templates += 1
                elif template.template_type == TemplateType.DYNAMIC:
                    self.stats.dynamic_templates += 1
                
                try:
                    self._update_stats()
                except Exception as e:
                    print(f"Error actualizando estadísticas: {e}")
                
                try:
                    self.anatomical_index.build_index()
                    self.dynamic_index.build_index()
                    print(f"Índices reconstruidos")
                except Exception as e:
                    print(f"Error reconstruyendo índices: {e}")
                
                print(f"Template {template.template_id} almacenado")
                return True
                
        except Exception as e:
            logger.error(f"Error almacenando template: {e}")
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
                
                # Almacenar usando store_biometric_template
                success = self.store_biometric_template(template)
                
                if success:
                    print(f"Template enrollado: {template_id}")
                    return template_id
                else:
                    return None
                
        except Exception as e:
            logger.error(f"Error enrollando template: {e}")
            return None
    
    def get_template(self, template_id: str) -> Optional[BiometricTemplate]:
        """Obtiene template biométrico."""
        return self.templates.get(template_id)
    
    # def list_user_templates(self, user_id: str) -> List[BiometricTemplate]:
    #     """Lista templates de un usuario."""
    #     if user_id not in self.users:
    #         return []
        
    #     user_profile = self.users[user_id]
    #     all_template_ids = (user_profile.anatomical_templates + user_profile.dynamic_templates)
        
    #     templates = []
    #     for template_id in all_template_ids:
    #         if template_id in self.templates:
    #             templates.append(self.templates[template_id])
        
    #     return templates
    
    def list_user_templates(self, user_id: str) -> List[BiometricTemplate]:
        """Lista templates de un usuario - CONSULTA DIRECTA A SUPABASE."""
        try:
            # CONSULTAR DIRECTAMENTE EN SUPABASE
            response = self.supabase.table('biometric_templates').select('*').eq('user_id', user_id).execute()
            
            templates = []
            
            for template_data in response.data:
                try:
                    # Deserializar embeddings
                    anatomical_emb = None
                    if template_data.get('anatomical_embedding'):
                        anatomical_emb = np.array(template_data['anatomical_embedding'], dtype=np.float32)
                    
                    dynamic_emb = None
                    if template_data.get('dynamic_embedding'):
                        dynamic_emb = np.array(template_data['dynamic_embedding'], dtype=np.float32)
                    
                    # Convertir timestamps
                    created_at = template_data.get('created_at')
                    if isinstance(created_at, str):
                        from datetime import datetime
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp()
                    
                    updated_at = template_data.get('updated_at')
                    if isinstance(updated_at, str):
                        from datetime import datetime
                        updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00')).timestamp()
                    
                    last_used = template_data.get('last_used')
                    if isinstance(last_used, str):
                        from datetime import datetime
                        last_used = datetime.fromisoformat(last_used.replace('Z', '+00:00')).timestamp()
                    
                    # Determinar tipo
                    template_type_str = template_data.get('template_type', 'anatomical')
                    if template_type_str == 'anatomical':
                        template_type = TemplateType.ANATOMICAL
                    elif template_type_str == 'dynamic':
                        template_type = TemplateType.DYNAMIC
                    else:
                        template_type = TemplateType.MULTIMODAL
                    
                    # Crear objeto BiometricTemplate
                    template = BiometricTemplate(
                        template_id=template_data['template_id'],
                        user_id=template_data['user_id'],
                        template_type=template_type,
                        anatomical_embedding=anatomical_emb,
                        dynamic_embedding=dynamic_emb,
                        gesture_name=template_data.get('gesture_name', 'Unknown'),
                        hand_side=template_data.get('hand_side', 'unknown'),
                        quality_score=float(template_data.get('quality_score', 0.0)),
                        confidence=float(template_data.get('confidence', 0.0)),
                        enrollment_session=template_data.get('enrollment_session', ''),
                        created_at=created_at or time.time(),
                        updated_at=updated_at or time.time(),
                        last_used=last_used or time.time(),
                        verification_count=template_data.get('verification_count', 0),
                        success_count=template_data.get('success_count', 0),
                        is_encrypted=False,
                        checksum=template_data.get('checksum', ''),
                        metadata=self._decrypt_metadata(template_data.get('metadata', {}))
                    )
                    
                    templates.append(template)
                    
                except Exception as e:
                    print(f"Error procesando template {template_data.get('template_id')}: {e}")
                    continue
            
            return templates
            
        except Exception as e:
            print(f"Error consultando templates de Supabase para {user_id}: {e}")
            return []
    
    def delete_template(self, template_id: str) -> bool:
        """Elimina un template específico."""
        try:
            with self.lock:
                if template_id not in self.templates:
                    logger.error(f"Template {template_id} no existe")
                    return False
                
                template = self.templates[template_id]
                user_id = template.user_id
                
                # Eliminar de índices
                self.anatomical_index.remove_template(template_id)
                self.dynamic_index.remove_template(template_id)
                
                # Eliminar de memoria
                del self.templates[template_id]
                
                # Actualizar perfil de usuario
                if user_id in self.users:
                    user_profile = self.users[user_id]
                    
                    if template_id in user_profile.anatomical_templates:
                        user_profile.anatomical_templates.remove(template_id)
                    if template_id in user_profile.dynamic_templates:
                        user_profile.dynamic_templates.remove(template_id)
                    
                    user_profile.updated_at = time.time()
                    self._save_user(user_profile)
                
                # ELIMINAR DE SUPABASE
                self.supabase.table('biometric_templates').delete().eq('template_id', template_id).execute()
                
                self.stats.total_templates -= 1
                if template.template_type == TemplateType.ANATOMICAL:
                    self.stats.anatomical_templates -= 1
                elif template.template_type == TemplateType.DYNAMIC:
                    self.stats.dynamic_templates -= 1
                
                self._update_stats()
                
                print(f"Template eliminado: {template_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error eliminando template: {e}")
            return False
    
    # ========================================================================
    # MÉTODOS DE VERIFICACIÓN
    # ========================================================================
    
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
                    
                    # FILTRO: Solo considerar usuarios activos
                    if match_user_id in self.users and not self.users[match_user_id].is_active:
                        continue
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
                
                # Actualizar estadísticas de usuario
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
                
                print(f"Verificación: {len(results)} matches")
                
                return results[:max_results]
                
        except Exception as e:
            logger.error(f"Error en verificación: {e}")
            return []
    
    # ========================================================================
    # MÉTODOS DE AUTHENTICATION ATTEMPTS
    # ========================================================================
    
    def store_authentication_attempt(self, attempt: AuthenticationAttempt) -> bool:
        """Almacena un intento de autenticación en Supabase."""
        try:
            with self.lock:
                # Convertir a diccionario para Supabase
                attempt_data = {
                    'attempt_id': attempt.attempt_id,
                    'user_id': attempt.user_id,
                    'timestamp': datetime.fromtimestamp(attempt.timestamp).isoformat(),
                    'auth_type': attempt.auth_type,
                    'result': attempt.result,
                    'confidence': attempt.confidence,
                    'anatomical_score': attempt.anatomical_score,
                    'dynamic_score': attempt.dynamic_score,
                    'fused_score': attempt.fused_score,
                    'ip_address': attempt.ip_address,
                    'device_info': attempt.device_info,
                    'failure_reason': attempt.failure_reason,
                    'metadata': attempt.metadata
                }
                
                # INSERTAR EN SUPABASE
                self.supabase.table('authentication_attempts').insert(attempt_data).execute()
                
                print(f"Intento de autenticación guardado: {attempt.attempt_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error guardando intento: {e}")
            return False
    
    # def get_user_auth_attempts(self, user_id: str, limit: Optional[int] = None) -> List[AuthenticationAttempt]:
    #     """Obtiene intentos de autenticación de un usuario desde Supabase."""
    #     try:
    #         # QUERY SUPABASE
    #         query = self.supabase.table('authentication_attempts').select('*').eq('user_id', user_id).order('timestamp', desc=True)
            
    #         if limit:
    #             query = query.limit(limit)
            
    #         response = query.execute()
            
    #         attempts = []
    #         for data in response.data:
    #             # Convertir timestamp ISO a float
    #             timestamp = data.get('timestamp')
    #             if isinstance(timestamp, str):
    #                 timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                
    #             attempts.append(AuthenticationAttempt(
    #                 attempt_id=data['attempt_id'],
    #                 user_id=data['user_id'],
    #                 timestamp=timestamp,
    #                 auth_type=data['auth_type'],
    #                 result=data['result'],
    #                 confidence=data['confidence'],
    #                 anatomical_score=data['anatomical_score'],
    #                 dynamic_score=data['dynamic_score'],
    #                 fused_score=data['fused_score'],
    #                 ip_address=data.get('ip_address'),
    #                 device_info=data.get('device_info'),
    #                 failure_reason=data.get('failure_reason'),
    #                 metadata=data.get('metadata', {})
    #             ))
            
    #         return attempts
            
    #     except Exception as e:
    #         logger.error(f"Error obteniendo intentos: {e}")
    #         return []
    
    def get_user_auth_attempts(self, user_id: str, limit: Optional[int] = None) -> List[AuthenticationAttempt]:
        """Obtiene intentos de autenticación de un usuario - AMBAS TABLAS CON CAMPOS REALES."""
        try:
            import uuid
            from datetime import datetime
            attempts = []
            
            # ========================================
            # 1. OBTENER DE authentication_attempts (VERIFICACIONES)
            # ========================================
            try:
                response = self.supabase.table('authentication_attempts')\
                    .select('*')\
                    .eq('user_id', user_id)\
                    .order('timestamp', desc=True)\
                    .execute()
                
                for data in response.data:
                    timestamp = data.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                    
                    # Mapear system_decision a result
                    system_decision = data.get('system_decision', 'rejected')
                    result = 'success' if system_decision == 'authenticated' else 'failed'
                    
                    attempts.append(AuthenticationAttempt(
                        attempt_id=str(data.get('id', uuid.uuid4())),
                        user_id=data['user_id'],
                        timestamp=timestamp,
                        auth_type=data.get('mode', 'verification'),  # Usar 'mode' en lugar de 'auth_type'
                        result=result,  # Convertir system_decision a result
                        confidence=float(data.get('confidence', 0.0)),
                        anatomical_score=float(data.get('anatomical_score', 0.0)),
                        dynamic_score=float(data.get('dynamic_score', 0.0)),
                        fused_score=float(data.get('fused_score', 0.0)),
                        ip_address=data.get('ip_address'),
                        device_info=None,
                        failure_reason=None if result == 'success' else 'Autenticación rechazada',
                        metadata={
                            'session_id': data.get('session_id'),
                            'username': data.get('username'),
                            'duration': data.get('duration'),
                            'gestures_captured': data.get('gestures_captured', [])
                        }
                    ))
            except Exception as e:
                print(f"Error consultando authentication_attempts: {e}")
                import traceback
                traceback.print_exc()
            
            # ========================================
            # 2. OBTENER DE identification_attempts (IDENTIFICACIONES)
            # ========================================
            try:
                response = self.supabase.table('identification_attempts')\
                    .select('*')\
                    .eq('identified_user_id', user_id)\
                    .order('timestamp', desc=True)\
                    .execute()
                
                for data in response.data:
                    timestamp = data.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                    
                    # Mapear system_decision a result
                    system_decision = data.get('system_decision', 'rejected')
                    result = 'success' if system_decision == 'authenticated' else 'failed'
                    
                    attempts.append(AuthenticationAttempt(
                        attempt_id=str(data.get('id', uuid.uuid4())),
                        user_id=user_id,
                        timestamp=timestamp,
                        auth_type='identification',
                        result=result,
                        confidence=float(data.get('confidence', 0.0)),
                        anatomical_score=float(data.get('anatomical_score', 0.0)),
                        dynamic_score=float(data.get('dynamic_score', 0.0)),
                        fused_score=float(data.get('fused_score', 0.0)),
                        ip_address=data.get('ip_address'),
                        device_info=None,
                        failure_reason=None if result == 'success' else 'Usuario no identificado',
                        metadata={
                            'session_id': data.get('session_id'),
                            'username': data.get('username'),
                            'user_email': data.get('user_email'),
                            'duration': data.get('duration'),
                            'all_candidates': data.get('all_candidates', []),
                            'top_match_score': data.get('top_match_score'),
                            'gestures_captured': data.get('gestures_captured', [])
                        }
                    ))
            except Exception as e:
                print(f"Error consultando identification_attempts: {e}")
                import traceback
                traceback.print_exc()
            
            # ========================================
            # 3. ORDENAR POR TIMESTAMP Y LIMITAR
            # ========================================
            attempts.sort(key=lambda x: x.timestamp, reverse=True)
            
            if limit:
                attempts = attempts[:limit]
            
            print(f"Total intentos recuperados para {user_id}: {len(attempts)}")
            
            return attempts
            
        except Exception as e:
            logger.error(f"Error obteniendo intentos del usuario {user_id}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_all_auth_attempts(self, limit: Optional[int] = None) -> List[AuthenticationAttempt]:
        """Obtiene TODOS los intentos de autenticación del sistema (estructura REAL)."""
        try:
            print("\n" + "=" * 80)
            print("DEBUG: get_all_auth_attempts() INICIADO")
            print("=" * 80)
            
            # QUERY DIRECTA A SUPABASE CON ESTRUCTURA REAL
            print(f"Creando query a tabla 'authentication_attempts'...")
            print(f"   Limit: {limit}")
            
            query = self.supabase.table('authentication_attempts')\
                .select('*')\
                .order('timestamp', desc=True)
            
            if limit:
                query = query.limit(limit)
            
            print(f"Ejecutando query en Supabase...")
            response = query.execute()
            
            print(f"Query ejecutada exitosamente")
            print(f"Total de registros obtenidos: {len(response.data)}")
            
            if len(response.data) == 0:
                print("NO SE ENCONTRARON REGISTROS EN LA TABLA")
                print("=" * 80 + "\n")
                return []
            
            # Mostrar primer registro
            print(f"\nPRIMER REGISTRO (ejemplo):")
            first = response.data[0]
            print(f"   id: {first.get('id')}")
            print(f"   user_id: {first.get('user_id')}")
            print(f"   mode: {first.get('mode')}")
            print(f"   system_decision: {first.get('system_decision')}")
            print(f"   confidence: {first.get('confidence')}")
            print(f"   timestamp: {first.get('timestamp')}")
            print(f"   username: {first.get('username')}")
            print(f"   duration: {first.get('duration')}")
            
            attempts = []
            
            for idx, data in enumerate(response.data):
                try:
                    # Convertir timestamp ISO a float
                    timestamp = data.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                    elif timestamp is None:
                        timestamp = time.time()
                    
                    # MAPEAR ESTRUCTURA REAL DE SUPABASE
                    attempt_id = str(data.get('id', f"attempt_{idx}_{int(time.time())}"))
                    auth_type = data.get('mode', 'unknown')
                    result = 'success' if data.get('system_decision') == 'authenticated' else 'failed'
                    
                    # EXTRAER SCORES DIRECTAMENTE DE LAS COLUMNAS
                    anatomical_score = float(data.get('anatomical_score', 0.0))
                    dynamic_score = float(data.get('dynamic_score', 0.0))
                    fused_score = float(data.get('fused_score', 0.0))
                    confidence = float(data.get('confidence', fused_score))
                    
                    # EXTRAER GESTOS
                    gestures_captured = data.get('gestures_captured', [])
                    
                    attempts.append(AuthenticationAttempt(
                        attempt_id=attempt_id,
                        user_id=data.get('user_id', 'unknown'),
                        timestamp=timestamp,
                        auth_type=auth_type,
                        result=result,
                        confidence=confidence,
                        anatomical_score=anatomical_score,
                        dynamic_score=dynamic_score,
                        fused_score=fused_score,
                        ip_address=data.get('ip_address'),
                        device_info=None,
                        failure_reason=None,
                        metadata={
                            'session_id': data.get('session_id'),
                            'username': data.get('username'),
                            'duration': data.get('duration'),
                            'feedback_token': data.get('feedback_token'),
                            'user_feedback': data.get('user_feedback'),
                            'gestures_captured': gestures_captured
                        }
                    ))
                    
                except Exception as item_error:
                    print(f"Error procesando registro {idx}: {item_error}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\nTOTAL PROCESADO: {len(attempts)} intentos")
            print("=" * 80 + "\n")
            
            print(f"Total de intentos procesados correctamente: {len(attempts)}")
            return attempts
            
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"ERROR EN get_all_auth_attempts()")
            print(f"   Error: {e}")
            print("=" * 80 + "\n")
            logger.error(f"Error obteniendo todos los intentos: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _parse_timestamp(self, timestamp_value):
        """
        Parsea timestamp desde diferentes formatos a float Unix timestamp.
        
        Args:
            timestamp_value: Timestamp en formato ISO string, float, o None
            
        Returns:
            float: Unix timestamp
        """
        try:
            if timestamp_value is None:
                return time.time()
            
            # Si ya es float/int (timestamp Unix)
            if isinstance(timestamp_value, (int, float)):
                return float(timestamp_value)
            
            # Si es string ISO format (de Supabase)
            if isinstance(timestamp_value, str):
                dt = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                return dt.timestamp()
            
            # Fallback
            return time.time()
            
        except Exception as e:
            logger.error(f"Error parseando timestamp: {e}")
            return time.time()
    
    def get_all_identification_attempts(self, limit: int = 500) -> List[AuthenticationAttempt]:
        """
        Obtiene todos los intentos de IDENTIFICACIÓN (1:N) desde Supabase.
        
        Args:
            limit: Número máximo de intentos a recuperar (default: 500)
            
        Returns:
            Lista de AuthenticationAttempt con intentos de identificación
        """
        try:
            print(f"Recuperando intentos de IDENTIFICACIÓN desde Supabase (limit={limit})")
            
            # Query a tabla identification_attempts
            response = self.supabase.table('identification_attempts')\
                .select('*')\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            
            if not response.data:
                print("No se encontraron intentos de identificación")
                return []
            
            print(f"Se recuperaron {len(response.data)} intentos de identificación")
            
            # Convertir a AuthenticationAttempt
            attempts = []
            for data in response.data:
                try:
                    # Extraer campos
                    identified_user_id = data.get('identified_user_id')
                    username = data.get('username', 'Unknown')
                    
                    # Scores
                    anatomical_score = float(data.get('anatomical_score', 0.0))
                    dynamic_score = float(data.get('dynamic_score', 0.0))
                    fused_score = float(data.get('fused_score', 0.0))
                    confidence = float(data.get('confidence', fused_score))
                    
                    # Crear AuthenticationAttempt
                    attempt = AuthenticationAttempt(
                        attempt_id=data.get('id', str(uuid.uuid4())),
                        user_id=identified_user_id or 'unknown',
                        timestamp=self._parse_timestamp(data.get('timestamp')),
                        auth_type='identification',
                        result='success' if data.get('system_decision') == 'authenticated' else 'failed',
                        confidence=confidence,
                        anatomical_score=anatomical_score,
                        dynamic_score=dynamic_score,
                        fused_score=fused_score,
                        ip_address=data.get('ip_address', 'unknown'),
                        device_info=data.get('device_info', ''),
                        failure_reason=None if data.get('system_decision') == 'authenticated' else 'No identificado',
                        metadata={
                            'session_id': data.get('session_id'),
                            'username': username,
                            'user_email': data.get('user_email'),
                            'duration': data.get('duration'),
                            'gestures_captured': data.get('gestures_captured', []),
                            'all_candidates': data.get('all_candidates', []),
                            'top_match_score': data.get('top_match_score')
                        }
                    )
                    
                    attempts.append(attempt)
                    
                except Exception as e:
                    logger.error(f"Error procesando intento de identificación: {e}")
                    continue
            
            print(f"Procesados {len(attempts)} intentos de identificación")
            return attempts
            
        except Exception as e:
            logger.error(f"Error recuperando intentos de identificación: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    # ========================================================================
    # MÉTODOS DE PERSONALITY PROFILES
    # ========================================================================
    
    def store_personality_profile(self, profile: PersonalityProfile) -> bool:
        """Almacena el perfil de personalidad en Supabase."""
        try:
            profile_data = profile.to_dict()
            
            # UPSERT EN SUPABASE
            self.supabase.table('personality_profiles').upsert(profile_data, on_conflict='user_id').execute()
            
            print(f"Perfil de personalidad guardado: {profile.user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error guardando perfil de personalidad: {e}")
            return False
    
    def get_personality_profile(self, user_id: str) -> Optional[PersonalityProfile]:
        """Obtiene el perfil de personalidad desde Supabase."""
        try:
            response = self.supabase.table('personality_profiles').select('*').eq('user_id', user_id).execute()
            
            if not response.data:
                return None
            
            return PersonalityProfile.from_dict(response.data[0])
            
        except Exception as e:
            logger.error(f"Error cargando perfil de personalidad: {e}")
            return None
    
    def has_personality_profile(self, user_id: str) -> bool:
        """Verifica si un usuario tiene perfil de personalidad."""
        try:
            response = self.supabase.table('personality_profiles').select('user_id').eq('user_id', user_id).execute()
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Error verificando perfil de personalidad: {e}")
            return False
    
    # ========================================================================
    # MÉTODOS BOOTSTRAP
    # ========================================================================
    
    def enroll_template_bootstrap(self, user_id: str,
                        anatomical_features: Optional[np.ndarray] = None,
                        dynamic_features: Optional[np.ndarray] = None,
                        temporal_sequence: Optional[np.ndarray] = None,
                        gesture_name: str = "unknown",
                        quality_score: float = 1.0,
                        confidence: float = 1.0,
                        sample_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Enrolla datos en modo Bootstrap (sin embeddings todavía)."""
        try:
            with self.lock:
                if user_id not in self.users:
                    print(f"Usuario {user_id} no existe - Creando automáticamente")
                    
                    username = "Usuario Bootstrap"
                    if sample_metadata and 'session_username' in sample_metadata:
                        username = sample_metadata['session_username']
                    elif sample_metadata and 'username' in sample_metadata:
                        username = sample_metadata['username']
                    
                    email = sample_metadata.get('email')
                    phone_number = sample_metadata.get('phone_number')
                    age = sample_metadata.get('age')
                    gender = sample_metadata.get('gender')
    
                    if not all([email, phone_number, age, gender]):
                        error_msg = f"ERROR CRÍTICO: Usuario {user_id} sin datos completos"
                        print(error_msg)
                        raise ValueError("Datos obligatorios faltantes en enrollment bootstrap")
                    
                    user_profile = UserProfile(
                        user_id=user_id,
                        username=username,
                        email=email,
                        phone_number=phone_number,
                        age=age,
                        gender=gender,
                        gesture_sequence=[],
                        metadata={
                            'bootstrap_mode': True,
                            'auto_created': True,
                            'creation_reason': 'First template enrollment in bootstrap mode'
                        }
                    )
                    
                    self.users[user_id] = user_profile
                    self._save_user(user_profile)
                    
                    print(f"Usuario {user_id} creado automáticamente")
                
                if anatomical_features is None:
                    logger.error("Se requieren características anatómicas en Bootstrap")
                    return None
                
                if anatomical_features.shape[0] != 180:
                    logger.error("Características anatómicas deben tener 180 dimensiones")
                    return None
                
                # CREAR TEMPLATE ANATÓMICO
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
                
                print("=" * 80)
                print("DIAGNOSTICO ANATOMICO - ANTES DE MODIFICAR METADATA")
                print(f"sample_metadata keys: {list((sample_metadata or {}).keys())}")
                print(f"template.metadata keys iniciales: {list(anatomical_template.metadata.keys())}")
                print("=" * 80)
                
                anatomical_template.metadata['bootstrap_features'] = anatomical_features.tolist()
                anatomical_template.metadata['has_anatomical_raw'] = True
                anatomical_template.metadata['feature_dimensions'] = len(anatomical_features)
                anatomical_template.metadata['bootstrap_mode'] = True
                anatomical_template.metadata['pending_embedding'] = True
                anatomical_template.metadata['modality'] = 'anatomical'
                
                print("=" * 80)
                print("DIAGNOSTICO ANATOMICO - DESPUES DE MODIFICAR METADATA")
                print(f"template.metadata keys finales: {list(anatomical_template.metadata.keys())}")
                print(f"bootstrap_features presente: {'bootstrap_features' in anatomical_template.metadata}")
                if 'bootstrap_features' in anatomical_template.metadata:
                    print(f"bootstrap_features dimension: {len(anatomical_template.metadata['bootstrap_features'])}")
                # print(f"temporal_sequence presente: {'temporal_sequence' in anatomical_template.metadata}")
                # if 'temporal_sequence' in anatomical_template.metadata:
                #     print(f"temporal_sequence dimension: {len(anatomical_template.metadata['temporal_sequence'])} frames")
                # print("=" * 80)
                
                print(f"temporal_sequence presente: {'temporal_sequence' in anatomical_template.metadata}")
                if 'temporal_sequence' in anatomical_template.metadata and anatomical_template.metadata['temporal_sequence'] is not None:
                    print(f"temporal_sequence dimension: {len(anatomical_template.metadata['temporal_sequence'])} frames")
                print("=" * 80)
                # ============================================================================
                # DEBUG: Verificar que bootstrap_features se agregó correctamente
                # ============================================================================
                print("="*80)
                print("DEBUG BOOTSTRAP_FEATURES EN ENROLL_TEMPLATE_BOOTSTRAP")
                print("="*80)
                print(f"Template ID: {anatomical_template_id}")
                print(f"Metadata keys: {list(anatomical_template.metadata.keys())}")

                if 'bootstrap_features' in anatomical_template.metadata:
                    bf = anatomical_template.metadata['bootstrap_features']
                    print(f"✓ bootstrap_features PRESENTE")
                    print(f"  Tipo: {type(bf)}")
                    print(f"  Longitud: {len(bf)}")
                    print(f"  Primeros 5 valores: {bf[:5]}")
                else:
                    print(f"✗ bootstrap_features NO PRESENTE")

                # if 'temporal_sequence' in anatomical_template.metadata:
                #     ts = anatomical_template.metadata['temporal_sequence']
                #     print(f"✓ temporal_sequence PRESENTE")
                #     print(f"  Frames: {len(ts)}")
                # else:
                #     print(f"✗ temporal_sequence NO PRESENTE")
                # print("="*80)
                
                if 'temporal_sequence' in anatomical_template.metadata and anatomical_template.metadata['temporal_sequence'] is not None:
                    ts = anatomical_template.metadata['temporal_sequence']
                    print(f"✓ temporal_sequence PRESENTE")
                    print(f"  Frames: {len(ts)}")
                else:
                    print(f"✗ temporal_sequence NO PRESENTE o es None")
    
                # BUSCAR DATOS TEMPORALES
                dynamic_template_id = None
                temporal_sequence = None
                data_source_found = None
                is_real_temporal = False
                
                try:
                    # Buscar en metadata de muestra
                    if (sample_metadata and 
                        'has_temporal_data' in sample_metadata and 
                        sample_metadata['has_temporal_data'] and
                        'temporal_sequence' in sample_metadata and
                        sample_metadata['temporal_sequence'] is not None):
                        
                        temporal_sequence = np.array(sample_metadata['temporal_sequence'], dtype=np.float32)
                        data_source_found = sample_metadata.get('data_source', 'real_enrollment_capture')
                        is_real_temporal = True
                        
                        print(f"Secuencia temporal REAL encontrada: {temporal_sequence.shape}")
                    
                    # CREAR TEMPLATE DINÁMICO SI HAY SECUENCIA
                    if temporal_sequence is not None and len(temporal_sequence) >= 5:
                        dynamic_template_id = f"{user_id}_bootstrap_dynamic_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                        
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
                                'data_source': data_source_found or 'unknown_source',
                                'is_real_temporal': is_real_temporal
                            }
                        )
                        
                        dynamic_template.checksum = self._calculate_template_checksum(dynamic_template)
                        self.templates[dynamic_template_id] = dynamic_template
                        
                        self._save_template_bootstrap(dynamic_template)
                        
                        print(f"Template dinámico bootstrap creado: {dynamic_template_id}")
                        
                        anatomical_template.metadata['paired_dynamic_template'] = dynamic_template_id
                        
                except Exception as e:
                    logger.error(f"Error en extracción de datos temporales: {e}")
                    dynamic_template_id = None
                
                # GUARDAR TEMPLATE ANATÓMICO
                anatomical_template.checksum = self._calculate_template_checksum(anatomical_template)
                self.templates[anatomical_template_id] = anatomical_template
                self._save_template_bootstrap(anatomical_template)
                
                # ACTUALIZAR PERFIL DE USUARIO
                user_profile = self.users[user_id]
                
                user_profile.anatomical_templates.append(anatomical_template_id)
                if dynamic_template_id:
                    user_profile.dynamic_templates.append(dynamic_template_id)
                
                templates_created = 2 if dynamic_template_id else 1
                user_profile.total_enrollments += templates_created
                user_profile.updated_at = time.time()
                
                if gesture_name not in user_profile.gesture_sequence:
                    user_profile.gesture_sequence.append(gesture_name)
                
                self._save_user(user_profile)
                
                # ACTUALIZAR ESTADÍSTICAS
                self.stats.total_templates += templates_created
                self.stats.anatomical_templates += 1
                if dynamic_template_id:
                    self.stats.dynamic_templates += 1
                
                self._update_stats()
                
                print(f"BOOTSTRAP COMPLETO: {templates_created} templates")
                
                return anatomical_template_id
                
        except Exception as e:
            logger.error(f"Error Bootstrap: {e}")
            return None
    
    def convert_bootstrap_to_full_templates(self, siamese_anatomical_network, siamese_dynamic_network=None):
        """Convierte templates Bootstrap a templates completos con embeddings."""
        try:
            with self.lock:
                bootstrap_templates = []
                
                for template_id, template in self.templates.items():
                    if template.metadata.get('bootstrap_mode', False):
                        bootstrap_templates.append(template)
                
                print(f"Convirtiendo {len(bootstrap_templates)} templates Bootstrap")
                
                converted_count = 0
                for template in bootstrap_templates:
                    try:
                        anatomical_features = np.array(template.metadata['bootstrap_features'])
                        
                        anatomical_embedding = siamese_anatomical_network.generate_embedding(
                            anatomical_features.reshape(1, -1)
                        )[0]
                        
                        dynamic_embedding = None
                        if siamese_dynamic_network and 'temporal_sequence' in template.metadata:
                            temporal_sequence = np.array(template.metadata['temporal_sequence'])
                            dynamic_embedding = siamese_dynamic_network.generate_embedding(
                                temporal_sequence.reshape(1, -1)
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
                
                print(f"Convertidos {converted_count}/{len(bootstrap_templates)} templates Bootstrap")
                
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
    
    # ========================================================================
    # MÉTODOS INTERNOS CRÍTICOS (GUARDAR EN SUPABASE)
    # ========================================================================
    
    # def _save_user(self, user_profile: UserProfile):
    #     """Guarda perfil de usuario en Supabase."""
    #     try:
    #         print(f"Guardando usuario en Supabase: {user_profile.user_id}")
            
    #         # Convertir timestamps a ISO format
    #         created_at = datetime.fromtimestamp(user_profile.created_at).isoformat()
    #         updated_at = datetime.fromtimestamp(user_profile.updated_at).isoformat()
            
    #         last_activity = None
    #         if user_profile.last_activity:
    #             last_activity = datetime.fromtimestamp(user_profile.last_activity).isoformat()
            
    #         last_failed_timestamp = None
    #         if user_profile.last_failed_timestamp:
    #             last_failed_timestamp = datetime.fromtimestamp(user_profile.last_failed_timestamp).isoformat()
            
    #         lockout_until = None
    #         if user_profile.lockout_until:
    #             lockout_until = datetime.fromtimestamp(user_profile.lockout_until).isoformat()
            
    #         user_data = {
    #             'user_id': user_profile.user_id,
    #             'username': user_profile.username,
    #             'email': user_profile.email,
    #             'phone_number': user_profile.phone_number,
    #             'age': user_profile.age,
    #             'gender': user_profile.gender,
    #             'anatomical_templates': user_profile.anatomical_templates,
    #             'dynamic_templates': user_profile.dynamic_templates,
    #             'gesture_sequence': user_profile.gesture_sequence or [],
    #             'total_enrollments': user_profile.total_enrollments,
    #             'total_verifications': user_profile.total_verifications,
    #             'successful_verifications': user_profile.successful_verifications,
    #             'last_activity': last_activity,
    #             'is_active': user_profile.is_active,
    #             'failed_attempts': user_profile.failed_attempts,
    #             'last_failed_timestamp': last_failed_timestamp,
    #             'lockout_until': lockout_until,
    #             'lockout_history': user_profile.lockout_history,
    #             'created_at': created_at,
    #             'updated_at': updated_at,
    #             'metadata': user_profile.metadata
    #         }
            
    #         # UPSERT EN SUPABASE
    #         result = self.supabase.table('users').upsert(user_data, on_conflict='user_id').execute()
            
    #         print(f"Usuario guardado en Supabase: {user_profile.user_id}")
            
    #     except Exception as e:
    #         print(f"ERROR guardando usuario en Supabase: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         logger.error(f"Error guardando usuario: {e}")
    
    def _save_user(self, user_profile: UserProfile):
        """Guarda perfil de usuario en Supabase - CORREGIDO para re-enrollment."""
        try:
            from datetime import timezone
            print(f"Guardando usuario en Supabase: {user_profile.user_id}")
            
            # Convertir timestamps a ISO format con UTC explícito para consistencia
            created_at = datetime.fromtimestamp(user_profile.created_at, tz=timezone.utc).isoformat()
            updated_at = datetime.fromtimestamp(user_profile.updated_at, tz=timezone.utc).isoformat()
            
            last_activity = None
            if user_profile.last_activity:
                last_activity = datetime.fromtimestamp(user_profile.last_activity, tz=timezone.utc).isoformat()
            
            last_failed_timestamp = None
            if user_profile.last_failed_timestamp:
                last_failed_timestamp = datetime.fromtimestamp(user_profile.last_failed_timestamp, tz=timezone.utc).isoformat()
            
            lockout_until = None
            if user_profile.lockout_until:
                lockout_until = datetime.fromtimestamp(user_profile.lockout_until, tz=timezone.utc).isoformat()
            
            user_data = {
                'user_id': user_profile.user_id,
                'username': user_profile.username,
                'email': user_profile.email,
                'phone_number': user_profile.phone_number,
                'age': user_profile.age,
                'gender': user_profile.gender,
                'anatomical_templates': user_profile.anatomical_templates,
                'dynamic_templates': user_profile.dynamic_templates,
                'gesture_sequence': user_profile.gesture_sequence or [],
                'total_enrollments': user_profile.total_enrollments,
                'total_verifications': user_profile.total_verifications,
                'successful_verifications': user_profile.successful_verifications,
                'last_activity': last_activity,
                'is_active': user_profile.is_active,
                'failed_attempts': user_profile.failed_attempts,
                'last_failed_timestamp': last_failed_timestamp,
                'lockout_until': lockout_until,
                'lockout_history': user_profile.lockout_history,
                'created_at': created_at,
                'updated_at': updated_at,
                'metadata': user_profile.metadata
            }
            
            # VERIFICAR SI USUARIO EXISTE (evita sobrescritura en re-enrollment)
            existing = self.supabase.table('users')\
                .select('id')\
                .eq('user_id', user_profile.user_id)\
                .execute()
            
            if existing.data:
                # Usuario existe → UPDATE
                print(f"   Usuario existe, actualizando...")
                self.supabase.table('users')\
                    .update(user_data)\
                    .eq('user_id', user_profile.user_id)\
                    .execute()
                print(f"Usuario actualizado en Supabase: {user_profile.user_id}")
            else:
                # Usuario NO existe → INSERT (crea nuevo registro)
                print(f"   Usuario nuevo, insertando...")
                self.supabase.table('users')\
                    .insert(user_data)\
                    .execute()
                print(f"Usuario insertado en Supabase: {user_profile.user_id}")
            
        except Exception as e:
            print(f"ERROR guardando usuario en Supabase: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Error guardando usuario: {e}")
    def _save_template(self, template: BiometricTemplate):
        """Guarda template en Supabase."""
        try:
            print(f"Guardando template en Supabase: {template.template_id}")
            
            # DIAGNOSTICO: Verificar metadata en el objeto template
            print("=" * 80)
            print("DIAGNOSTICO _save_template - OBJETO TEMPLATE")
            print(f"Template ID: {template.template_id}")
            print(f"Template Type: {template.template_type}")
            print(f"Metadata type: {type(template.metadata)}")
            print(f"Metadata keys: {list(template.metadata.keys()) if hasattr(template.metadata, 'keys') else 'NO DICT'}")
            print(f"bootstrap_features en template.metadata: {'bootstrap_features' in template.metadata if isinstance(template.metadata, dict) else False}")
            if isinstance(template.metadata, dict) and 'bootstrap_features' in template.metadata:
                print(f"bootstrap_features dimension: {len(template.metadata['bootstrap_features'])}")
            print(f"temporal_sequence en template.metadata: {'temporal_sequence' in template.metadata if isinstance(template.metadata, dict) else False}")
            if isinstance(template.metadata, dict) and 'temporal_sequence' in template.metadata:
                temporal_seq = template.metadata['temporal_sequence']
                if temporal_seq:
                    print(f"temporal_sequence dimension: {len(temporal_seq)} frames")
            print("=" * 80)
        
            # Convertir timestamps
            created_at = datetime.fromtimestamp(template.created_at).isoformat()
            updated_at = datetime.fromtimestamp(template.updated_at).isoformat()
            last_used = datetime.fromtimestamp(template.last_used).isoformat()
            
            # ============================================================================
            # DEBUG: Verificar metadata ANTES de serializar
            # ============================================================================
            print("="*80)
            print("DEBUG EN _save_template() - ANTES DE CREAR template_data")
            print("="*80)
            print(f"Template ID: {template.template_id}")
            print(f"Template metadata keys: {list(template.metadata.keys())}")

            if 'bootstrap_features' in template.metadata:
                bf = template.metadata['bootstrap_features']
                print(f"✓ bootstrap_features en template.metadata")
                print(f"  Longitud: {len(bf)}")
            else:
                print(f"✗ bootstrap_features NO en template.metadata")

            if 'temporal_sequence' in template.metadata:
                print(f"✓ temporal_sequence en template.metadata")
            else:
                print(f"✗ temporal_sequence NO en template.metadata")
            print("="*80)

            template_data = {
                'template_id': template.template_id,
                'user_id': template.user_id,
                'template_type': template.template_type.value if hasattr(template.template_type, 'value') else str(template.template_type),
                'gesture_name': template.gesture_name,
                'hand_side': getattr(template, 'hand_side', 'unknown'),
                'quality_score': float(template.quality_score) if template.quality_score is not None else None,
                'confidence': float(template.confidence) if template.confidence is not None else None,
                'anatomical_embedding': template.anatomical_embedding.tolist() if template.anatomical_embedding is not None else None,
                'dynamic_embedding': template.dynamic_embedding.tolist() if template.dynamic_embedding is not None else None,
                'enrollment_session': getattr(template, 'enrollment_session', ''),
                'created_at': created_at,
                'updated_at': updated_at,
                'last_used': last_used,
                'verification_count': getattr(template, 'verification_count', 0),
                'success_count': getattr(template, 'success_count', 0),
                'is_encrypted': self.encryption_enabled,
                'checksum': getattr(template, 'checksum', ''),
                'metadata': self._encrypt_metadata(getattr(template, 'metadata', {}))
            }
            
            # DIAGNOSTICO: Verificar template_data antes de enviar
            print("=" * 80)
            print("DIAGNOSTICO _save_template - TEMPLATE_DATA PREPARADO")
            metadata_to_send = template_data.get('metadata', {})
            print(f"Metadata type en template_data: {type(metadata_to_send)}")
            print(f"Metadata keys en template_data: {list(metadata_to_send.keys()) if isinstance(metadata_to_send, dict) else 'NO DICT'}")
            print(f"bootstrap_features en template_data: {'bootstrap_features' in metadata_to_send if isinstance(metadata_to_send, dict) else False}")
            if isinstance(metadata_to_send, dict) and 'bootstrap_features' in metadata_to_send:
                print(f"bootstrap_features dimension en template_data: {len(metadata_to_send['bootstrap_features'])}")
            print(f"temporal_sequence en template_data: {'temporal_sequence' in metadata_to_send if isinstance(metadata_to_send, dict) else False}")
            if isinstance(metadata_to_send, dict) and 'temporal_sequence' in metadata_to_send:
                temporal_seq = metadata_to_send['temporal_sequence']
                if temporal_seq:
                    print(f"temporal_sequence dimension en template_data: {len(temporal_seq)} frames")
            print("=" * 80)
            
            # ============================================================================
            # DEBUG: Verificar metadata en template_data ANTES de upsert
            # ============================================================================
            print("="*80)
            print("DEBUG EN _save_template() - ANTES DE UPSERT")
            print("="*80)
            
            # Metadata puede ser string (cifrado) o dict (sin cifrar)
            metadata_value = template_data['metadata']
            if isinstance(metadata_value, str):
                print(f"Metadata CIFRADA: {len(metadata_value)} caracteres")
                print(f"Primeros 50 chars: {metadata_value[:50]}...")
            elif isinstance(metadata_value, dict):
                print(f"Metadata SIN CIFRAR: {list(metadata_value.keys())}")
                if 'bootstrap_features' in metadata_value:
                    print(f"  bootstrap_features: {len(metadata_value['bootstrap_features'])} elementos")
            else:
                print(f"Metadata tipo desconocido: {type(metadata_value)}")
            print("="*80)

            # UPSERT EN SUPABASE
            result = self.supabase.table('biometric_templates').upsert(template_data, on_conflict='template_id').execute()
            
            # DIAGNOSTICO: Verificar respuesta de Supabase
            print("=" * 80)
            print("DIAGNOSTICO _save_template - RESPUESTA SUPABASE")
            if result.data:
                print(f"Resultado count: {len(result.data)}")
                if len(result.data) > 0:
                    saved_template = result.data[0]
                    saved_metadata = saved_template.get('metadata', {})
                    print(f"Metadata type en respuesta: {type(saved_metadata)}")
                    print(f"Metadata keys en respuesta: {list(saved_metadata.keys()) if isinstance(saved_metadata, dict) else 'NO DICT'}")
                    print(f"bootstrap_features en respuesta: {'bootstrap_features' in saved_metadata if isinstance(saved_metadata, dict) else False}")
                    if isinstance(saved_metadata, dict) and 'bootstrap_features' in saved_metadata:
                        print(f"bootstrap_features dimension en respuesta: {len(saved_metadata['bootstrap_features'])}")
                    print(f"temporal_sequence en respuesta: {'temporal_sequence' in saved_metadata if isinstance(saved_metadata, dict) else False}")
                    if isinstance(saved_metadata, dict) and 'temporal_sequence' in saved_metadata:
                        temporal_seq = saved_metadata['temporal_sequence']
                        if temporal_seq:
                            print(f"temporal_sequence dimension en respuesta: {len(temporal_seq)} frames")
            else:
                print("WARNING: result.data es None o vacio")
            print("=" * 80)
        
            print(f"Template guardado en Supabase: {template.template_id}")

            # ============================================================================
            # DEBUG: VERIFICAR QUÉ SE GUARDÓ REALMENTE EN SUPABASE
            # ============================================================================
            print("="*80)
            print("DEBUG: VERIFICACIÓN INMEDIATA EN SUPABASE")
            print("="*80)
            try:
                verify = self.supabase.table('biometric_templates').select('*').eq('template_id', template.template_id).execute()
                
                if verify.data:
                    saved_template = verify.data[0]
                    saved_metadata = saved_template.get('metadata', {})
                    
                    print(f"Template recuperado de Supabase:")
                    
                    # Metadata puede ser string (cifrado) o dict
                    if isinstance(saved_metadata, str):
                        print(f"  Metadata CIFRADA guardada: {len(saved_metadata)} caracteres")
                    elif isinstance(saved_metadata, dict):
                        print(f"  Metadata keys guardados: {list(saved_metadata.keys())}")
                        if 'bootstrap_features' in saved_metadata:
                            bf_saved = saved_metadata['bootstrap_features']
                            print(f"    bootstrap_features: {len(bf_saved)} elementos")
                        if 'temporal_sequence' in saved_metadata:
                            ts_saved = saved_metadata['temporal_sequence']
                            print(f"    temporal_sequence: {len(ts_saved)} frames")
                    else:
                        print(f"  Metadata tipo: {type(saved_metadata)}")
                        
                else:
                    print(f"  ✗✗✗ NO SE PUDO RECUPERAR EL TEMPLATE DE SUPABASE")
                    
            except Exception as verify_error:
                print(f"  ERROR en verificación: {verify_error}")
                
            print("="*80)
            
        except Exception as e:
            print(f"ERROR guardando template en Supabase: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_template_bootstrap(self, template: BiometricTemplate):
        """Guarda template Bootstrap en Supabase (sin embeddings)."""
        try:
            print(f"Guardando Bootstrap en Supabase: {template.template_id}")
            
            # Igual que _save_template, pero los embeddings ya son None
            self._save_template(template)
            
            print(f"Bootstrap guardado en Supabase: {template.template_id}")
                
        except Exception as e:
            print(f"ERROR guardando Bootstrap en Supabase: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Error guardando Bootstrap: {e}")
    
    def _calculate_template_checksum(self, template: BiometricTemplate) -> str:
        """Calcula checksum de integridad del template."""
        try:
            import hashlib
            
            data_string = f"{template.user_id}{template.template_type.value}{template.created_at}"
            
            if template.anatomical_embedding is not None:
                data_string += str(np.sum(template.anatomical_embedding))
            
            if template.dynamic_embedding is not None:
                data_string += str(np.sum(template.dynamic_embedding))
            
            return hashlib.sha256(data_string.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"Error calculando checksum: {e}")
            return ""
    
    # ========================================================================
    # MÉTODOS DE ESTADÍSTICAS
    # ========================================================================
    
    def _update_stats(self):
        """Actualiza estadísticas de la base de datos."""
        try:
            # Las estadísticas se mantienen en memoria
            # No se guardan en Supabase (no hay tabla para esto)
            self.stats.last_updated = time.time()
                
        except Exception as e:
            logger.error(f"Error actualizando estadísticas: {e}")
    
    def get_database_stats(self) -> DatabaseStats:
        """Obtiene estadísticas actuales."""
        self._update_stats()
        return self.stats
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verifica integridad de la base de datos."""
        try:
            issues = []
            
            # Verificar en Supabase
            for user_id, user_profile in self.users.items():
                try:
                    response = self.supabase.table('users').select('user_id').eq('user_id', user_id).execute()
                    if not response.data:
                        issues.append(f"Usuario faltante en Supabase: {user_id}")
                except Exception as e:
                    issues.append(f"Error verificando usuario {user_id}: {e}")
            
            for template_id, template in self.templates.items():
                try:
                    response = self.supabase.table('biometric_templates').select('template_id').eq('template_id', template_id).execute()
                    if not response.data:
                        issues.append(f"Template faltante en Supabase: {template_id}")
                except Exception as e:
                    issues.append(f"Error verificando template {template_id}: {e}")
            
            return {
                'integrity_ok': len(issues) == 0,
                'issues': issues,
                'total_users': len(self.users),
                'total_templates': len(self.templates),
                'anatomical_index_size': len(self.anatomical_index.template_ids),
                'dynamic_index_size': len(self.dynamic_index.template_ids)
            }
            
        except Exception as e:
            logger.error(f"Error verificando integridad: {e}")
            return {'integrity_ok': False, 'error': str(e)}
    
    def export_database(self, export_path: str, include_embeddings: bool = True) -> bool:
        """Exporta la base de datos a un archivo JSON."""
        try:
            export_data = {
                'users': {},
                'templates': {},
                'stats': asdict(self.stats),
                'export_timestamp': time.time(),
                'version': '2.0-supabase'
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
            
            print(f"Base de datos exportada a: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la base de datos."""
        return {
            'storage_type': 'Supabase PostgreSQL',
            'total_users': len(self.users),
            'total_templates': len(self.templates),
            'anatomical_templates': len([t for t in self.templates.values() if t.anatomical_embedding is not None]),
            'dynamic_templates': len([t for t in self.templates.values() if t.dynamic_embedding is not None]),
            'encryption_enabled': self.encryption_enabled,
            'search_strategy': self.config['search_strategy'],
            'integrity_status': 'OK'
        }


# ============================================================================
# INSTANCIA GLOBAL
# ============================================================================

_biometric_db_instance = None

def get_biometric_database(db_path: Optional[str] = None) -> BiometricDatabase:
    """Obtiene instancia global de la base de datos biométrica con Supabase."""
    global _biometric_db_instance
    
    if _biometric_db_instance is None:
        _biometric_db_instance = BiometricDatabase(db_path)
    
    return _biometric_db_instance