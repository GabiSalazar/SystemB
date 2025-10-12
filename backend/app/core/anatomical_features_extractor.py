# =============================================================================
# MÓDULO 6: ANATOMICAL_FEATURES_EXTRACTOR
# Extractor de características anatómicas únicas para biometría
# =============================================================================

import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math

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


class FeatureCategory(Enum):
    """Categorías de características anatómicas."""
    FINGER_LENGTHS = "finger_lengths"
    PALM_DIMENSIONS = "palm_dimensions"
    JOINT_ANGLES = "joint_angles"
    FINGER_SPREADS = "finger_spreads"
    PALM_CURVATURE = "palm_curvature"
    HAND_PROPORTIONS = "hand_proportions"
    LANDMARK_DISTANCES = "landmark_distances"
    GEOMETRIC_RATIOS = "geometric_ratios"


@dataclass
class FingerMetrics:
    """Métricas detalladas de un dedo."""
    total_length: float
    proximal_length: float
    middle_length: float
    distal_length: float
    tip_to_base_ratio: float
    curvature_angle: float
    spread_angle: float


@dataclass
class PalmMetrics:
    """Métricas de la palma de la mano."""
    width: float
    height: float
    area: float
    aspect_ratio: float
    center_x: float
    center_y: float
    perimeter: float


@dataclass
class AnatomicalFeatureVector:
    """Vector completo de características anatómicas."""
    finger_features: np.ndarray
    palm_features: np.ndarray
    proportion_features: np.ndarray
    angle_features: np.ndarray
    distance_features: np.ndarray
    curvature_features: np.ndarray
    
    @property
    def complete_vector(self) -> np.ndarray:
        """Vector completo concatenado (180 dimensiones)."""
        return np.concatenate([
            self.finger_features,
            self.palm_features,
            self.proportion_features,
            self.angle_features,
            self.distance_features,
            self.curvature_features
        ])
    
    @property
    def dimension(self) -> int:
        """Dimensión total del vector."""
        return len(self.complete_vector)


class AnatomicalFeaturesExtractor:
    """
    Extractor de características anatómicas únicas para biometría de manos.
    Implementa micro-características detalladas para redes siamesas.
    """
    
    def __init__(self):
        """Inicializa el extractor de características."""
        # Cargar configuración
        self.feature_config = self._load_feature_config()
        
        # Definir estructura de landmarks MediaPipe (21 puntos)
        self.landmark_structure = self._define_landmark_structure()
        
        # Estadísticas
        self.extractions_performed = 0
        self.successful_extractions = 0
        
        logger.info("AnatomicalFeaturesExtractor inicializado")
    
    def _load_feature_config(self) -> Dict[str, Any]:
        """Carga configuración para extracción de características."""
        default_config = {
            'normalize_features': True,
            'use_world_landmarks': True,
            'feature_smoothing': True,
            'outlier_threshold': 3.0,
            'min_hand_size': 0.05,
            'angle_smoothing_factor': 0.1
        }
        
        return get_config('biometric.feature_extraction', default_config)
    
    def _define_landmark_structure(self) -> Dict[str, Dict[str, List[int]]]:
        """Define la estructura de landmarks MediaPipe para cada parte de la mano."""
        return {
            'wrist': {'base': [0]},
            'thumb': {
                'base': [1, 2], 'proximal': [2, 3], 'distal': [3, 4],
                'all': [1, 2, 3, 4]
            },
            'index': {
                'base': [5, 6], 'proximal': [6, 7], 'middle': [7, 8], 'distal': [8],
                'all': [5, 6, 7, 8]
            },
            'middle': {
                'base': [9, 10], 'proximal': [10, 11], 'middle': [11, 12], 'distal': [12],
                'all': [9, 10, 11, 12]
            },
            'ring': {
                'base': [13, 14], 'proximal': [14, 15], 'middle': [15, 16], 'distal': [16],
                'all': [13, 14, 15, 16]
            },
            'pinky': {
                'base': [17, 18], 'proximal': [18, 19], 'middle': [19, 20], 'distal': [20],
                'all': [17, 18, 19, 20]
            },
            'palm': {
                'boundary': [0, 1, 5, 9, 13, 17],
                'center_region': [0, 2, 5, 9, 13, 17]
            }
        }
        
    def extract_features(self, hand_landmarks, world_landmarks: Optional[Any] = None,
                    hand_side: str = "unknown") -> Optional[AnatomicalFeatureVector]:
        """Extrae características anatómicas completas de la mano - VERSION CORREGIDA."""
        try:
            self.extractions_performed += 1
            
            logger.debug(f"EXTRACT: Iniciando extraccion anatomica")
            logger.debug(f"EXTRACT: hand_landmarks tipo: {type(hand_landmarks)}")
            logger.debug(f"EXTRACT: world_landmarks tipo: {type(world_landmarks)}")
            
            if hand_landmarks is None:
                logger.error("EXTRACT: hand_landmarks es None")
                return None
            
            if not self._validate_landmarks(hand_landmarks):
                logger.error("EXTRACT: Validacion de landmarks fallo")
                return None
            
            logger.debug("EXTRACT: Validacion de landmarks exitosa")
            
            use_world = world_landmarks and self.feature_config.get('use_world_landmarks', False)
            if use_world and self._validate_landmarks(world_landmarks):
                primary_landmarks = world_landmarks
                logger.debug("EXTRACT: Usando world_landmarks como primarios")
            else:
                primary_landmarks = hand_landmarks
                logger.debug("EXTRACT: Usando hand_landmarks como primarios")
            
            feature_results = {}
            
            logger.debug("EXTRACT: Procesando caracteristicas de dedos...")
            try:
                finger_features = self._extract_finger_features(primary_landmarks, hand_landmarks)
                if finger_features is None:
                    logger.error("EXTRACT: _extract_finger_features retorno None")
                    return None
                feature_results['fingers'] = finger_features
                logger.debug("EXTRACT: Caracteristicas de dedos OK")
            except Exception as e:
                logger.error(f"EXTRACT: Error en dedos: {e}")
                return None
            
            logger.debug("EXTRACT: Procesando caracteristicas de palma...")
            try:
                palm_features = self._extract_palm_features(primary_landmarks, hand_landmarks)
                if palm_features is None:
                    logger.error("EXTRACT: _extract_palm_features retorno None")
                    return None
                feature_results['palm'] = palm_features
                logger.debug("EXTRACT: Caracteristicas de palma OK")
            except Exception as e:
                logger.error(f"EXTRACT: Error en palma: {e}")
                return None
            
            logger.debug("EXTRACT: Procesando proporciones...")
            try:
                proportion_features = self._extract_proportion_features(primary_landmarks)
                if proportion_features is None:
                    logger.error("EXTRACT: _extract_proportion_features retorno None")
                    return None
                feature_results['proportions'] = proportion_features
                logger.debug("EXTRACT: Proporciones OK")
            except Exception as e:
                logger.error(f"EXTRACT: Error en proporciones: {e}")
                return None
            
            logger.debug("EXTRACT: Procesando angulos...")
            try:
                angle_features = self._extract_angle_features(primary_landmarks)
                if angle_features is None:
                    logger.error("EXTRACT: _extract_angle_features retorno None")
                    return None
                feature_results['angles'] = angle_features
                logger.debug("EXTRACT: Angulos OK")
            except Exception as e:
                logger.error(f"EXTRACT: Error en angulos: {e}")
                return None
            
            logger.debug("EXTRACT: Procesando distancias...")
            try:
                distance_features = self._extract_distance_features(primary_landmarks)
                if distance_features is None:
                    logger.error("EXTRACT: _extract_distance_features retorno None")
                    return None
                feature_results['distances'] = distance_features
                logger.debug("EXTRACT: Distancias OK")
            except Exception as e:
                logger.error(f"EXTRACT: Error en distancias: {e}")
                return None
            
            logger.debug("EXTRACT: Procesando curvaturas...")
            try:
                curvature_features = self._extract_curvature_features(primary_landmarks)
                if curvature_features is None:
                    logger.error("EXTRACT: _extract_curvature_features retorno None")
                    return None
                feature_results['curvatures'] = curvature_features
                logger.debug("EXTRACT: Curvaturas OK")
            except Exception as e:
                logger.error(f"EXTRACT: Error en curvaturas: {e}")
                return None
            
            logger.debug("EXTRACT: Creando vector de caracteristicas...")
            try:
                feature_vector = AnatomicalFeatureVector(
                    finger_features=feature_results['fingers'],
                    palm_features=feature_results['palm'],
                    proportion_features=feature_results['proportions'],
                    angle_features=feature_results['angles'],
                    distance_features=feature_results['distances'],
                    curvature_features=feature_results['curvatures']
                )
                logger.debug("EXTRACT: Vector creado exitosamente")
            except Exception as e:
                logger.error(f"EXTRACT: Error creando vector: {e}")
                return None
            
            if self.feature_config.get('normalize_features', False):
                try:
                    feature_vector = self._normalize_features(feature_vector)
                    logger.debug("EXTRACT: Normalizacion aplicada")
                except Exception as e:
                    logger.error(f"EXTRACT: Error en normalizacion: {e}")
                    return None
            
            try:
                if self._validate_feature_quality(feature_vector):
                    self.successful_extractions += 1
                    logger.debug("EXTRACT: Extraccion COMPLETAMENTE exitosa")
                    return feature_vector
                else:
                    logger.error("EXTRACT: Vector no paso validacion de calidad")
                    return None
            except Exception as e:
                logger.error(f"EXTRACT: Error en validacion final: {e}")
                return None
                
        except Exception as e:
            logger.error(f"EXTRACT: Error GENERAL en extraccion: {e}", exc_info=True)
            return None
    
    def _validate_landmarks(self, landmarks) -> bool:
        """Valida que los landmarks sean válidos para extracción - CORRIGE WORLD LANDMARKS."""
        try:
            if not landmarks:
                logger.error("VALIDATE: landmarks es None o False")
                return False
                
            if not hasattr(landmarks, 'landmark'):
                logger.error(f"VALIDATE: landmarks no tiene atributo 'landmark', tipo: {type(landmarks)}")
                return False
            
            landmark_count = len(landmarks.landmark)
            if landmark_count != 21:
                logger.error(f"VALIDATE: Cantidad incorrecta de landmarks: {landmark_count}, esperados: 21")
                return False
            
            logger.debug(f"VALIDATE: Landmarks count OK: {landmark_count}")
            
            sample_landmarks = landmarks.landmark[:3]
            is_hand_landmarks = all(
                0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0 
                for lm in sample_landmarks
            )
            
            invalid_landmarks = []
            for i, landmark in enumerate(landmarks.landmark):
                try:
                    if not all(hasattr(landmark, attr) for attr in ['x', 'y', 'z']):
                        invalid_landmarks.append(f"landmark_{i}_missing_coords")
                        continue
                    
                    coords = [landmark.x, landmark.y, landmark.z]
                    
                    if not all(np.isfinite(coords)):
                        invalid_landmarks.append(f"landmark_{i}_infinite_coords")
                        continue
                    
                    if is_hand_landmarks:
                        if not all(0.0 <= coord <= 1.0 for coord in coords[:2]):
                            invalid_landmarks.append(f"landmark_{i}_out_of_range")
                            continue
                    else:
                        if any(abs(coord) > 10.0 for coord in coords):
                            invalid_landmarks.append(f"landmark_{i}_extreme_value")
                            continue
                            
                except Exception as e:
                    invalid_landmarks.append(f"landmark_{i}_error_{str(e)}")
            
            if invalid_landmarks:
                if is_hand_landmarks:
                    logger.error(f"VALIDATE: Hand landmarks inválidos: {invalid_landmarks[:5]}")
                    return False
                else:
                    if len(invalid_landmarks) > 5:
                        logger.error(f"VALIDATE: Demasiados world landmarks inválidos: {invalid_landmarks[:5]}")
                        return False
                    else:
                        logger.debug(f"VALIDATE: World landmarks con algunos valores extendidos (tolerado): {len(invalid_landmarks)}")
            
            logger.debug("VALIDATE: Todas las coordenadas son válidas")
            
            try:
                wrist = landmarks.landmark[0]
                middle_tip = landmarks.landmark[12]
                
                hand_size = np.sqrt((wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2)
                min_size = self.feature_config.get('min_hand_size', 0.1)
                
                if hand_size < min_size:
                    logger.error(f"VALIDATE: Mano muy pequena: {hand_size:.4f} < {min_size}")
                    return False
                
                logger.debug(f"VALIDATE: Tamano de mano OK: {hand_size:.4f}")
                
            except Exception as e:
                logger.error(f"VALIDATE: Error calculando tamano de mano: {e}")
                return False
            
            logger.debug("VALIDATE: Todos los checks pasaron exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"VALIDATE: Error general validando landmarks: {e}")
            return False
        
    def _extract_finger_features(self, landmarks, landmarks_2d) -> np.ndarray:
        """Extrae características detalladas de los dedos."""
        try:
            features = []
            finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            
            for finger_name in finger_names:
                finger_indices = self.landmark_structure[finger_name]['all']
                finger_metrics = self._calculate_finger_metrics(landmarks, finger_indices, finger_name)
                
                finger_features = [
                    finger_metrics.total_length,
                    finger_metrics.proximal_length,
                    finger_metrics.middle_length,
                    finger_metrics.distal_length,
                    finger_metrics.tip_to_base_ratio,
                    finger_metrics.curvature_angle,
                    finger_metrics.spread_angle,
                    self._calculate_finger_thickness(landmarks, finger_indices),
                    self._calculate_finger_straightness(landmarks, finger_indices),
                    self._calculate_finger_flexibility(landmarks, finger_indices)
                ]
                
                features.extend(finger_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características de dedos: {e}")
            return np.zeros(50, dtype=np.float32)
    
    def _calculate_finger_metrics(self, landmarks, finger_indices: List[int], 
                             finger_name: str) -> FingerMetrics:
        """Calcula métricas detalladas de un dedo específico - SOLO VALORES REALES CALCULADOS."""
        try:
            if landmarks is None or not hasattr(landmarks, 'landmark'):
                raise ValueError(f"No hay landmarks válidos para calcular métricas de {finger_name}")
            
            if len(landmarks.landmark) <= max(finger_indices):
                raise ValueError(f"Faltan landmarks para {finger_name}: necesarios hasta índice {max(finger_indices)}")
            
            points = []
            for i in finger_indices:
                if i >= len(landmarks.landmark):
                    raise ValueError(f"Índice {i} no existe en landmarks para {finger_name}")
                
                point = landmarks.landmark[i]
                if not (hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z')):
                    raise ValueError(f"Punto {i} no tiene coordenadas válidas para {finger_name}")
                
                if any(np.isnan([point.x, point.y, point.z])) or any(np.isinf([point.x, point.y, point.z])):
                    raise ValueError(f"Coordenadas inválidas en punto {i} para {finger_name}")
                
                points.append(point)
            
            segment_lengths = []
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i + 1]
                
                dx = p1.x - p2.x
                dy = p1.y - p2.y
                dz = p1.z - p2.z
                length = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if length <= 0 or np.isnan(length) or np.isinf(length):
                    raise ValueError(f"Longitud inválida calculada entre puntos {i} y {i+1} para {finger_name}: {length}")
                
                segment_lengths.append(length)
            
            if len(segment_lengths) == 0:
                raise ValueError(f"No se pudieron calcular segmentos para {finger_name}")
            
            if finger_name == 'thumb':
                if len(segment_lengths) < 3:
                    raise ValueError(f"Pulgar necesita 3 segmentos, solo se calcularon {len(segment_lengths)}")
                proximal = segment_lengths[0]
                middle = segment_lengths[1]
                distal = segment_lengths[2]
            else:
                if len(segment_lengths) < 3:
                    raise ValueError(f"Dedo {finger_name} necesita 3 segmentos, solo se calcularon {len(segment_lengths)}")
                proximal = segment_lengths[0]
                middle = segment_lengths[1]
                distal = segment_lengths[2]
            
            total_length = sum(segment_lengths)
            if total_length <= 0:
                raise ValueError(f"Longitud total inválida para {finger_name}: {total_length}")
            
            tip_to_base_ratio = distal / total_length
            
            try:
                curvature_angle = self._calculate_finger_curvature(points)
            except Exception as e:
                logger.error(f"No se pudo calcular curvatura para {finger_name}: {e}")
                curvature_angle = 0.0
                
            try:
                spread_angle = self._calculate_finger_spread(landmarks, finger_name)
            except Exception as e:
                logger.error(f"No se pudo calcular separación para {finger_name}: {e}")
                spread_angle = 0.0
            
            return FingerMetrics(
                total_length=total_length,
                proximal_length=proximal,
                middle_length=middle,
                distal_length=distal,
                tip_to_base_ratio=tip_to_base_ratio,
                curvature_angle=curvature_angle,
                spread_angle=spread_angle
            )
            
        except Exception as e:
            logger.error(f"IMPOSIBLE calcular métricas reales para dedo {finger_name}: {e}")
            return FingerMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _calculate_finger_curvature(self, finger_points: List) -> float:
        """Calcula el ángulo de curvatura de un dedo."""
        try:
            if len(finger_points) < 3:
                return 0.0
            
            p1 = finger_points[0]
            p_mid = finger_points[len(finger_points) // 2]
            p2 = finger_points[-1]
            
            v1 = np.array([p_mid.x - p1.x, p_mid.y - p1.y, p_mid.z - p1.z])
            v2 = np.array([p2.x - p_mid.x, p2.y - p_mid.y, p2.z - p_mid.z])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            return math.acos(cos_angle)
            
        except Exception as e:
            logger.error(f"Error calculando curvatura de dedo: {e}")
            return 0.0
    
    def _calculate_finger_spread(self, landmarks, finger_name: str) -> float:
        """Calcula el ángulo de separación entre dedos adyacentes."""
        try:
            finger_mapping = {
                'thumb': ('thumb', 'index'),
                'index': ('index', 'middle'),
                'middle': ('middle', 'ring'),
                'ring': ('ring', 'pinky'),
                'pinky': ('ring', 'pinky')
            }
            
            if finger_name not in finger_mapping:
                return 0.0
            
            f1_name, f2_name = finger_mapping[finger_name]
            
            f1_tip = self.landmark_structure[f1_name]['all'][-1]
            f2_tip = self.landmark_structure[f2_name]['all'][-1]
            wrist_idx = 0
            
            p1 = landmarks.landmark[f1_tip]
            p2 = landmarks.landmark[f2_tip]
            wrist = landmarks.landmark[wrist_idx]
            
            v1 = np.array([p1.x - wrist.x, p1.y - wrist.y, p1.z - wrist.z])
            v2 = np.array([p2.x - wrist.x, p2.y - wrist.y, p2.z - wrist.z])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            return math.acos(cos_angle)
            
        except Exception as e:
            logger.error(f"Error calculando separación del dedo {finger_name}: {e}")
            return 0.0
    
    def _calculate_finger_thickness(self, landmarks, finger_indices: List[int]) -> float:
        """Calcula un indicador de grosor del dedo."""
        try:
            if len(finger_indices) >= 2:
                p1 = landmarks.landmark[finger_indices[0]]
                p2 = landmarks.landmark[finger_indices[1]]
                return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
            return 0.0
        except:
            return 0.0
    
    def _calculate_finger_straightness(self, landmarks, finger_indices: List[int]) -> float:
        """Calcula qué tan recto está el dedo."""
        try:
            if len(finger_indices) < 3:
                return 1.0
            
            start = landmarks.landmark[finger_indices[0]]
            end = landmarks.landmark[finger_indices[-1]]
            direct_distance = np.sqrt((start.x - end.x)**2 + (start.y - end.y)**2 + (start.z - end.z)**2)
            
            segment_sum = 0
            for i in range(len(finger_indices) - 1):
                p1 = landmarks.landmark[finger_indices[i]]
                p2 = landmarks.landmark[finger_indices[i + 1]]
                segment_sum += np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
            
            return direct_distance / (segment_sum + 1e-8)
            
        except:
            return 1.0
    
    def _calculate_finger_flexibility(self, landmarks, finger_indices: List[int]) -> float:
        """Calcula un indicador de flexibilidad del dedo."""
        try:
            if len(finger_indices) < 4:
                return 0.0
            
            angles = []
            for i in range(len(finger_indices) - 2):
                p1 = landmarks.landmark[finger_indices[i]]
                p2 = landmarks.landmark[finger_indices[i + 1]]
                p3 = landmarks.landmark[finger_indices[i + 2]]
                
                v1 = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
                v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles.append(math.acos(cos_angle))
            
            return np.std(angles) if angles else 0.0
            
        except:
            return 0.0
    
    def _extract_palm_features(self, landmarks, landmarks_2d) -> np.ndarray:
        """Extrae características REALES de la palma - USANDO FUNCIONES ORIGINALES."""
        try:
            if landmarks is None or not hasattr(landmarks, 'landmark'):
                raise ValueError("No hay landmarks para calcular características de palma")
            
            if not hasattr(self, 'landmark_structure') or 'palm' not in self.landmark_structure:
                raise ValueError("Estructura de landmarks de palma no definida")
            
            palm_boundary = self.landmark_structure['palm']['boundary']
            
            max_index = max(palm_boundary)
            if len(landmarks.landmark) <= max_index:
                raise ValueError(f"Faltan landmarks de palma: necesario hasta índice {max_index}")
            
            palm_points = []
            for i in palm_boundary:
                point = landmarks.landmark[i]
                if not all(hasattr(point, attr) for attr in ['x', 'y', 'z']):
                    raise ValueError(f"Punto de palma {i} no tiene coordenadas válidas")
                
                if any(np.isnan([point.x, point.y, point.z])) or any(np.isinf([point.x, point.y, point.z])):
                    raise ValueError(f"Coordenadas inválidas en punto de palma {i}")
                
                palm_points.append(point)
            
            palm_metrics = self._calculate_palm_metrics(palm_points)
            
            features = [
                palm_metrics.width,
                palm_metrics.height,
                palm_metrics.area,
                palm_metrics.aspect_ratio,
                palm_metrics.perimeter,
                self._calculate_palm_roundness(palm_points),
                self._calculate_palm_symmetry(landmarks),
                self._calculate_palm_center_deviation(palm_points, palm_metrics),
                self._calculate_wrist_width(landmarks),
                self._calculate_palm_arch_height(landmarks),
                self._distance_normalized(landmarks, 0, 5),
                self._distance_normalized(landmarks, 0, 9),
                self._distance_normalized(landmarks, 0, 13),
                self._distance_normalized(landmarks, 0, 17),
                self._distance_normalized(landmarks, 5, 17),
                self._distance_normalized(landmarks, 1, 5),
                self._distance_normalized(landmarks, 5, 9),
                self._distance_normalized(landmarks, 9, 13),
                self._distance_normalized(landmarks, 13, 17),
                palm_metrics.center_y
            ]
            
            for i, feature in enumerate(features):
                if np.isnan(feature) or np.isinf(feature):
                    raise ValueError(f"Característica {i} tiene valor inválido: {feature}")
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"IMPOSIBLE extraer características reales de palma: {e}")
            return np.zeros(20, dtype=np.float32)
    
    def _calculate_palm_metrics(self, palm_points: List) -> PalmMetrics:
        """Calcula métricas detalladas de la palma."""
        try:
            x_coords = [p.x for p in palm_points]
            y_coords = [p.y for p in palm_points]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            area = 0.5 * abs(sum(x_coords[i] * y_coords[i+1] - x_coords[i+1] * y_coords[i] 
                               for i in range(-1, len(x_coords) - 1)))
            
            aspect_ratio = width / (height + 1e-8)
            
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            perimeter = sum(np.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)
                          for i in range(len(x_coords)))
            
            return PalmMetrics(
                width=width, height=height, area=area, aspect_ratio=aspect_ratio,
                center_x=center_x, center_y=center_y, perimeter=perimeter
            )
            
        except Exception as e:
            logger.error(f"Error calculando métricas de palma: {e}")
            return PalmMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_palm_roundness(self, palm_points: List) -> float:
        """Calcula qué tan redonda es la palma."""
        try:
            if len(palm_points) < 3:
                return 0.0
            
            x_coords = [p.x for p in palm_points]
            y_coords = [p.y for p in palm_points]
            
            area = 0.5 * abs(sum(x_coords[i] * y_coords[i+1] - x_coords[i+1] * y_coords[i] 
                               for i in range(-1, len(x_coords) - 1)))
            
            perimeter = sum(np.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)
                          for i in range(len(x_coords)))
            
            if perimeter > 0:
                return (4 * math.pi * area) / (perimeter ** 2)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculando redondez de palma: {e}")
            return 0.0
    
    def _calculate_palm_symmetry(self, landmarks) -> float:
        """Calcula simetría de la palma."""
        try:
            left_distances = [
                self._distance_normalized(landmarks, 0, 5),
                self._distance_normalized(landmarks, 0, 9),
            ]
            
            right_distances = [
                self._distance_normalized(landmarks, 0, 17),
                self._distance_normalized(landmarks, 0, 13),
            ]
            
            total_diff = sum(abs(l - r) for l, r in zip(left_distances, right_distances))
            avg_distance = np.mean(left_distances + right_distances)
            
            return 1.0 - (total_diff / (avg_distance + 1e-8))
            
        except Exception as e:
            logger.error(f"Error calculando simetría de palma: {e}")
            return 0.0
    
    def _calculate_palm_center_deviation(self, palm_points: List, metrics: PalmMetrics) -> float:
        """Calcula desviación del centro geométrico."""
        try:
            expected_center_x = (palm_points[0].x + palm_points[-1].x) / 2
            expected_center_y = (palm_points[0].y + palm_points[-1].y) / 2
            
            deviation = np.sqrt((metrics.center_x - expected_center_x)**2 + 
                              (metrics.center_y - expected_center_y)**2)
            
            return deviation
            
        except Exception as e:
            logger.error(f"Error calculando desviación del centro: {e}")
            return 0.0
    
    def _calculate_wrist_width(self, landmarks) -> float:
        """Calcula ancho de la muñeca."""
        try:
            thumb_base = landmarks.landmark[1]
            pinky_base = landmarks.landmark[17]
            
            return np.sqrt((thumb_base.x - pinky_base.x)**2 + 
                         (thumb_base.y - pinky_base.y)**2 + 
                         (thumb_base.z - pinky_base.z)**2)
        except:
            return 0.0
    
    def _calculate_palm_arch_height(self, landmarks) -> float:
        """Calcula altura del arco de la palma."""
        try:
            wrist = landmarks.landmark[0]
            middle_base = landmarks.landmark[9]
            
            return abs(wrist.z - middle_base.z)
        except:
            return 0.0
        
    def _extract_proportion_features(self, landmarks) -> np.ndarray:
        """Extrae proporciones generales (30 dimensiones)."""
        try:
            features = []
            
            finger_lengths = []
            for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                finger_indices = self.landmark_structure[finger_name]['all']
                length = self._calculate_total_finger_length(landmarks, finger_indices)
                finger_lengths.append(length)
            
            for i in range(len(finger_lengths) - 1):
                ratio = finger_lengths[i] / (finger_lengths[i + 1] + 1e-8)
                features.append(ratio)
            
            features.extend([
                finger_lengths[1] / (finger_lengths[2] + 1e-8),
                finger_lengths[2] / (finger_lengths[3] + 1e-8),
                finger_lengths[0] / (finger_lengths[1] + 1e-8),
                max(finger_lengths) / (min(finger_lengths) + 1e-8),
                np.std(finger_lengths) / (np.mean(finger_lengths) + 1e-8)
            ])
            
            hand_length = self._distance_normalized(landmarks, 0, 12)
            palm_width = self._distance_normalized(landmarks, 5, 17)
            
            features.extend([
                hand_length / (palm_width + 1e-8),
                finger_lengths[2] / (hand_length + 1e-8),
                palm_width / (hand_length + 1e-8),
            ])
            
            additional_features = [
                self._distance_normalized(landmarks, 4, 8) / (hand_length + 1e-8),
                self._distance_normalized(landmarks, 4, 20) / (hand_length + 1e-8),
                self._distance_normalized(landmarks, 8, 20) / (hand_length + 1e-8),
            ]
            
            features.extend(additional_features)
            
            while len(features) < 30:
                features.append(np.mean(features[-3:]) if len(features) >= 3 else 0.0)
            
            return np.array(features[:30], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características de proporción: {e}")
            return np.zeros(30, dtype=np.float32)
    
    def _extract_angle_features(self, landmarks) -> np.ndarray:
        """Extrae ángulos articulares (25 dimensiones)."""
        try:
            features = []
            
            for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                finger_indices = self.landmark_structure[finger_name]['all']
                finger_angles = self._calculate_finger_joint_angles(landmarks, finger_indices)
                features.extend(finger_angles[:3])
            
            for i in range(4):
                finger1 = ['thumb', 'index', 'middle', 'ring'][i]
                finger2 = ['index', 'middle', 'ring', 'pinky'][i]
                angle = self._calculate_inter_finger_angle(landmarks, finger1, finger2)
                features.append(angle)
            
            palm_angles = self._calculate_palm_angles(landmarks)
            features.extend(palm_angles)
            
            return np.array(features[:25], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características de ángulos: {e}")
            return np.zeros(25, dtype=np.float32)
    
    def _extract_distance_features(self, landmarks) -> np.ndarray:
        """Extrae distancias normalizadas (35 dimensiones)."""
        try:
            features = []
            
            key_distances = [
                (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
                (4, 8), (8, 12), (12, 16), (16, 20), (4, 20),
                (1, 5), (5, 9), (9, 13), (13, 17),
                (2, 6), (6, 10), (10, 14), (14, 18),
                (3, 7), (7, 11), (11, 15), (15, 19),
            ]
            
            for p1, p2 in key_distances:
                distance = self._distance_normalized(landmarks, p1, p2)
                features.append(distance)
            
            additional_distances = [
                self._distance_normalized(landmarks, 0, 1),
                self._distance_normalized(landmarks, 0, 5),
                self._distance_normalized(landmarks, 0, 9),
                self._distance_normalized(landmarks, 0, 13),
                self._distance_normalized(landmarks, 0, 17),
                self._distance_normalized(landmarks, 4, 12),
                self._distance_normalized(landmarks, 4, 16),
                self._distance_normalized(landmarks, 8, 16),
                self._distance_normalized(landmarks, 8, 20),
                self._distance_normalized(landmarks, 12, 20),
                abs(landmarks.landmark[4].z - landmarks.landmark[0].z),
                abs(landmarks.landmark[8].z - landmarks.landmark[0].z),
                abs(landmarks.landmark[12].z - landmarks.landmark[0].z),
            ]
            
            features.extend(additional_distances)
            
            return np.array(features[:35], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características de distancia: {e}")
            return np.zeros(35, dtype=np.float32)
    
    def _extract_curvature_features(self, landmarks) -> np.ndarray:
        """Extrae características de curvatura (20 dimensiones)."""
        try:
            features = []
            
            for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                finger_indices = self.landmark_structure[finger_name]['all']
                finger_points = [landmarks.landmark[i] for i in finger_indices]
                curvature = self._calculate_finger_curvature(finger_points)
                features.append(curvature)
            
            palm_boundary = self.landmark_structure['palm']['boundary']
            palm_curvatures = self._calculate_palm_curvatures(landmarks, palm_boundary)
            features.extend(palm_curvatures)
            
            global_curvatures = [
                self._calculate_overall_hand_curvature(landmarks),
                self._calculate_arch_curvature(landmarks),
                self._calculate_finger_spread_curvature(landmarks),
                np.std([features[i] for i in range(5)]),
                np.mean([features[i] for i in range(5)]),
                max([features[i] for i in range(5)]),
                min([features[i] for i in range(5)]),
                features[1] / (features[2] + 1e-8) if len(features) > 2 else 0,
                features[2] / (features[3] + 1e-8) if len(features) > 3 else 0,
                features[0] / (features[1] + 1e-8) if len(features) > 1 else 0,
            ]
            
            features.extend(global_curvatures)
            
            return np.array(features[:20], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo características de curvatura: {e}")
            return np.zeros(20, dtype=np.float32)
    
    def _distance_normalized(self, landmarks, idx1: int, idx2: int) -> float:
        """Calcula distancia normalizada entre dos landmarks."""
        try:
            p1 = landmarks.landmark[idx1]
            p2 = landmarks.landmark[idx2]
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        except:
            return 0.0
    
    def _calculate_total_finger_length(self, landmarks, finger_indices: List[int]) -> float:
        """Calcula longitud total de un dedo."""
        try:
            total_length = 0
            for i in range(len(finger_indices) - 1):
                p1 = landmarks.landmark[finger_indices[i]]
                p2 = landmarks.landmark[finger_indices[i + 1]]
                length = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                total_length += length
            return total_length
        except:
            return 0.0
    
    def _calculate_finger_joint_angles(self, landmarks, finger_indices: List[int]) -> List[float]:
        """Calcula ángulos REALES de articulaciones usando geometría de puntos."""
        try:
            if landmarks is None or not hasattr(landmarks, 'landmark'):
                raise ValueError("No hay landmarks para calcular ángulos de articulaciones")
            
            if len(finger_indices) < 3:
                raise ValueError(f"Se necesitan al menos 3 puntos para calcular ángulos, solo hay {len(finger_indices)}")
            
            max_index = max(finger_indices)
            if len(landmarks.landmark) <= max_index:
                raise ValueError(f"Faltan landmarks: necesario hasta índice {max_index}")
            
            angles = []
            
            for i in range(1, len(finger_indices) - 1):
                idx_prev = finger_indices[i - 1]
                idx_curr = finger_indices[i]
                idx_next = finger_indices[i + 1]
                
                p1 = landmarks.landmark[idx_prev]
                p2 = landmarks.landmark[idx_curr]
                p3 = landmarks.landmark[idx_next]
                
                points = [p1, p2, p3]
                for j, point in enumerate(points):
                    if not all(hasattr(point, attr) for attr in ['x', 'y', 'z']):
                        raise ValueError(f"Punto {j} no tiene coordenadas válidas en articulación {i}")
                    
                    if any(np.isnan([point.x, point.y, point.z])) or any(np.isinf([point.x, point.y, point.z])):
                        raise ValueError(f"Coordenadas inválidas en punto {j} de articulación {i}")
                
                v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
                v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 == 0 or norm2 == 0:
                    raise ValueError(f"Vector nulo en articulación {i}")
                
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = math.acos(cos_angle)
                
                if np.isnan(angle) or np.isinf(angle) or angle < 0 or angle > math.pi:
                    raise ValueError(f"Ángulo inválido calculado en articulación {i}: {angle}")
                
                angles.append(float(angle))
            
            while len(angles) < 3:
                angles.append(0.0)
            
            return angles[:3]
            
        except Exception as e:
            logger.error(f"IMPOSIBLE calcular ángulos reales de articulaciones: {e}")
            return [0.0, 0.0, 0.0]
    
    def _calculate_inter_finger_angle(self, landmarks, finger1: str, finger2: str) -> float:
        """Calcula ángulo entre dos dedos."""
        try:
            f1_tip = self.landmark_structure[finger1]['all'][-1]
            f2_tip = self.landmark_structure[finger2]['all'][-1]
            wrist = 0
            
            p1 = landmarks.landmark[f1_tip]
            p2 = landmarks.landmark[f2_tip]
            pw = landmarks.landmark[wrist]
            
            v1 = np.array([p1.x - pw.x, p1.y - pw.y, p1.z - pw.z])
            v2 = np.array([p2.x - pw.x, p2.y - pw.y, p2.z - pw.z])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            return math.acos(cos_angle)
            
        except:
            return 0.0
    
    def _calculate_palm_angles(self, landmarks) -> List[float]:
        """Calcula ángulos característicos de la palma."""
        try:
            angles = []
            
            palm_lines = [
                (0, 5, 9), (0, 9, 13), (0, 13, 17),
                (5, 0, 17), (1, 0, 5), (1, 0, 17),
            ]
            
            for p1_idx, vertex_idx, p2_idx in palm_lines:
                p1 = landmarks.landmark[p1_idx]
                vertex = landmarks.landmark[vertex_idx]
                p2 = landmarks.landmark[p2_idx]
                
                v1 = np.array([p1.x - vertex.x, p1.y - vertex.y, p1.z - vertex.z])
                v2 = np.array([p2.x - vertex.x, p2.y - vertex.y, p2.z - vertex.z])
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles.append(math.acos(cos_angle))
            
            return angles
            
        except Exception as e:
            logger.error(f"Error calculando ángulos de palma: {e}")
            return [0.0] * 6
    
    def _calculate_palm_curvatures(self, landmarks, boundary_indices: List[int]) -> List[float]:
        """Calcula curvaturas de diferentes regiones de la palma."""
        try:
            curvatures = []
            
            for i in range(len(boundary_indices) - 2):
                p1 = landmarks.landmark[boundary_indices[i]]
                p2 = landmarks.landmark[boundary_indices[i + 1]]
                p3 = landmarks.landmark[boundary_indices[i + 2]]
                
                v1 = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
                v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                curvatures.append(math.acos(cos_angle))
            
            while len(curvatures) < 5:
                curvatures.append(np.mean(curvatures) if curvatures else 0.0)
                
            return curvatures[:5]
            
        except Exception as e:
            logger.error(f"Error calculando curvaturas de palma: {e}")
            return [0.0] * 5
    
    def _calculate_overall_hand_curvature(self, landmarks) -> float:
        """Calcula curvatura general de la mano."""
        try:
            wrist = landmarks.landmark[0]
            tips = [landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
            
            z_coords = [tip.z for tip in tips]
            return np.std(z_coords)
            
        except:
            return 0.0
    
    def _calculate_arch_curvature(self, landmarks) -> float:
        """Calcula curvatura del arco de la mano."""
        try:
            center_z = landmarks.landmark[9].z
            wrist_z = landmarks.landmark[0].z
            side_z = (landmarks.landmark[5].z + landmarks.landmark[17].z) / 2
            
            return abs(center_z - (wrist_z + side_z) / 2)
            
        except:
            return 0.0
    
    def _calculate_finger_spread_curvature(self, landmarks) -> float:
        """Calcula curvatura basada en la separación de dedos."""
        try:
            spreads = []
            
            for finger in ['thumb', 'index', 'middle', 'ring']:
                angle = self._calculate_finger_spread(landmarks, finger)
                spreads.append(angle)
            
            return np.std(spreads) if spreads else 0.0
            
        except:
            return 0.0
    
    def _normalize_features(self, feature_vector: AnatomicalFeatureVector) -> AnatomicalFeatureVector:
        """Normaliza el vector de características."""
        try:
            def robust_normalize(arr):
                if len(arr) == 0:
                    return arr
                
                median = np.median(arr)
                mad = np.median(np.abs(arr - median))
                
                if mad == 0:
                    return arr - median
                
                return (arr - median) / mad
            
            normalized_finger = robust_normalize(feature_vector.finger_features)
            normalized_palm = robust_normalize(feature_vector.palm_features)
            normalized_proportion = robust_normalize(feature_vector.proportion_features)
            normalized_angle = robust_normalize(feature_vector.angle_features)
            normalized_distance = robust_normalize(feature_vector.distance_features)
            normalized_curvature = robust_normalize(feature_vector.curvature_features)
            
            return AnatomicalFeatureVector(
                finger_features=normalized_finger,
                palm_features=normalized_palm,
                proportion_features=normalized_proportion,
                angle_features=normalized_angle,
                distance_features=normalized_distance,
                curvature_features=normalized_curvature
            )
            
        except Exception as e:
            logger.error(f"Error normalizando características: {e}")
            return feature_vector
    
    def _validate_feature_quality(self, feature_vector: AnatomicalFeatureVector) -> bool:
        """Valida la calidad del vector de características extraído."""
        try:
            complete_vector = feature_vector.complete_vector
            
            if not np.all(np.isfinite(complete_vector)):
                return False
            
            if np.all(complete_vector == 0):
                return False
            
            if np.std(complete_vector) < 1e-6:
                return False
            
            z_scores = np.abs((complete_vector - np.mean(complete_vector)) / (np.std(complete_vector) + 1e-8))
            outlier_ratio = np.sum(z_scores > self.feature_config['outlier_threshold']) / len(complete_vector)
            
            if outlier_ratio > 0.1:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando calidad de características: {e}")
            return False
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de extracción."""
        success_rate = (self.successful_extractions / self.extractions_performed * 100) if self.extractions_performed > 0 else 0
        
        return {
            'extractions_performed': self.extractions_performed,
            'successful_extractions': self.successful_extractions,
            'success_rate_percent': round(success_rate, 2),
            'feature_dimension': 180,
            'feature_categories': len(FeatureCategory),
            'use_world_landmarks': self.feature_config['use_world_landmarks'],
            'normalize_features': self.feature_config['normalize_features']
        }
    
    def reset_stats(self):
        """Reinicia estadísticas de extracción."""
        self.extractions_performed = 0
        self.successful_extractions = 0
        logger.info("Estadísticas de extracción de características reiniciadas")


# ===== INSTANCIA GLOBAL =====
_extractor_instance = None

def get_anatomical_features_extractor() -> AnatomicalFeaturesExtractor:
    """Obtiene una instancia global del extractor de características anatómicas."""
    global _extractor_instance
    
    if _extractor_instance is None:
        _extractor_instance = AnatomicalFeaturesExtractor()
    
    return _extractor_instance