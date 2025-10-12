"""
Core Biometric System - Initialization
Expone todos los módulos del sistema biométrico
"""

# Módulo 1: Config Manager
from app.core.config_manager import (
    ConfigManager,
    get_config,
    get_logger,
    log_info,
    log_error
)

# Módulo 2: Camera Manager
from app.core.camera_manager import (
    CameraManager,
    get_camera_manager,
    release_camera,
    reset_camera_for_new_operation
)

# Módulo 3: MediaPipe Processor
from app.core.mediapipe_processor import (
    MediaPipeProcessor,
    ProcessingResult,
    HandDetectionResult,
    GestureRecognitionResult,
    get_mediapipe_processor,
    release_mediapipe
)

# Módulo 4: Quality Validator
from app.core.quality_validator import (
    QualityValidator,
    QualityAssessment,
    HandSizeMetrics,
    MovementAnalysis,
    VisibilityAnalysis,
    AreaValidation,
    DistanceStatus,
    ValidationStatus,
    get_quality_validator
)

# Módulo 5: Reference Area Manager
from app.core.reference_area_manager import (
    ReferenceAreaManager,
    AreaDimensions,
    AreaCoordinates,
    VisualFeedback,
    get_reference_area_manager
)

# Módulo 6: Anatomical Features Extractor
from app.core.anatomical_features_extractor import (
    AnatomicalFeaturesExtractor,
    AnatomicalFeatureVector,
    FeatureCategory,
    get_anatomical_features_extractor
)

# Módulo 7: Dynamic Features Extractor
from app.core.dynamic_features_extractor import (
    RealDynamicFeaturesExtractor,
    DynamicFeatureVector,
    TemporalFrame,
    TransitionPhase,
    MotionType,
    VelocityProfile,
    AccelerationProfile,
    TrajectoryProfile,
    get_real_dynamic_features_extractor
)

# Módulo 8: Sequence Manager
from app.core.sequence_manager import (
    SequenceManager,
    SequenceState,
    GestureValidation,
    SequenceEventType,
    get_sequence_manager
)

# Módulo 9: Siamese Anatomical Network
from app.core.siamese_anatomical_network import (
    RealSiameseAnatomicalNetwork,
    get_real_siamese_anatomical_network
)

# Módulo 10: Siamese Dynamic Network
from app.core.siamese_dynamic_network import (
    RealSiameseDynamicNetwork,
    get_real_siamese_dynamic_network
)

# Módulo 11: Feature Preprocessor
from app.core.feature_preprocessor import (
    RealFeaturePreprocessor,
    get_real_feature_preprocessor
)

# Módulo 12: Score Fusion System
from app.core.score_fusion_system import (
    RealScoreFusionSystem,
    get_real_score_fusion_system
)

# Módulo 13: Biometric Database
from app.core.biometric_database import (
    BiometricDatabase,
    UserProfile,
    BiometricTemplate,
    TemplateType,
    DatabaseStats,
    get_biometric_database
)

# Módulo 0: ROI Normalization
from app.core.roi_normalization import (
    ROINormalizationSystem,
    ROIExtractionResult,
    get_roi_normalization_system
)

# Módulo 0: Visual Feedback
from app.core.visual_feedback import (
    VisualFeedbackManager,
    FeedbackMessage,
    FeedbackLevel,
    get_visual_feedback_manager
)

# Módulo 14: Enrollment System
from app.core.enrollment_system import (
    RealEnrollmentSystem,
    RealEnrollmentWorkflow,
    RealEnrollmentSession,
    RealEnrollmentSample,
    RealEnrollmentConfig,
    RealQualityController,
    RealTemplateGenerator,
    EnrollmentPhase,
    EnrollmentStatus,
    SampleType,
    get_real_enrollment_system,
    get_enrollment_system,
    EnrollmentSystem
)

# ✅ Módulo 15: Authentication System
from app.core.authentication_system import (
    RealAuthenticationSystem,
    RealAuthenticationPipeline,
    RealAuthenticationAttempt,
    RealAuthenticationResult,
    RealAuthenticationConfig,
    RealSecurityAuditor,
    RealSessionManager,
    RealIndividualScores,
    AuthenticationMode,
    AuthenticationStatus,
    AuthenticationPhase,
    SecurityLevel,
    get_real_authentication_system,
    get_authentication_system,
    AuthenticationSystem
)

# Versión del sistema
__version__ = "2.1.0"

# Exports principales
__all__ = [
    # Config Manager
    'ConfigManager',
    'get_config',
    'get_logger',
    'log_info',
    'log_error',
    
    # Camera Manager
    'CameraManager',
    'get_camera_manager',
    'release_camera',
    'reset_camera_for_new_operation',
    
    # MediaPipe Processor
    'MediaPipeProcessor',
    'ProcessingResult',
    'HandDetectionResult',
    'GestureRecognitionResult',
    'get_mediapipe_processor',
    'release_mediapipe',
    
    # Quality Validator
    'QualityValidator',
    'QualityAssessment',
    'HandSizeMetrics',
    'MovementAnalysis',
    'VisibilityAnalysis',
    'AreaValidation',
    'DistanceStatus',
    'ValidationStatus',
    'get_quality_validator',
    
    # Reference Area Manager
    'ReferenceAreaManager',
    'AreaDimensions',
    'AreaCoordinates',
    'VisualFeedback',
    'get_reference_area_manager',
    
    # Anatomical Features
    'AnatomicalFeaturesExtractor',
    'AnatomicalFeatureVector',
    'FeatureCategory',
    'get_anatomical_features_extractor',
    
    # Dynamic Features
    'RealDynamicFeaturesExtractor',
    'DynamicFeatureVector',
    'TemporalFrame',
    'TransitionPhase',
    'MotionType',
    'VelocityProfile',
    'AccelerationProfile',
    'TrajectoryProfile',
    'get_real_dynamic_features_extractor',
    
    # Sequence Manager
    'SequenceManager',
    'SequenceState',
    'GestureValidation',
    'SequenceEventType',
    'get_sequence_manager',
    
    # Siamese Networks
    'RealSiameseAnatomicalNetwork',
    'get_real_siamese_anatomical_network',
    'RealSiameseDynamicNetwork',
    'get_real_siamese_dynamic_network',
    
    # Feature Preprocessor
    'RealFeaturePreprocessor',
    'get_real_feature_preprocessor',
    
    # Score Fusion
    'RealScoreFusionSystem',
    'get_real_score_fusion_system',
    
    # Biometric Database
    'BiometricDatabase',
    'UserProfile',
    'BiometricTemplate',
    'TemplateType',
    'DatabaseStats',
    'get_biometric_database',
    
    # ROI Normalization
    'ROINormalizationSystem',
    'ROIExtractionResult',
    'get_roi_normalization_system',
    
    # Visual Feedback
    'VisualFeedbackManager',
    'FeedbackMessage',
    'FeedbackLevel',
    'get_visual_feedback_manager',
    
    # Enrollment System
    'RealEnrollmentSystem',
    'RealEnrollmentWorkflow',
    'RealEnrollmentSession',
    'RealEnrollmentSample',
    'RealEnrollmentConfig',
    'RealQualityController',
    'RealTemplateGenerator',
    'EnrollmentPhase',
    'EnrollmentStatus',
    'SampleType',
    'get_real_enrollment_system',
    'get_enrollment_system',
    'EnrollmentSystem',
    
    # ✅ Authentication System
    'RealAuthenticationSystem',
    'RealAuthenticationPipeline',
    'RealAuthenticationAttempt',
    'RealAuthenticationResult',
    'RealAuthenticationConfig',
    'RealSecurityAuditor',
    'RealSessionManager',
    'RealIndividualScores',
    'AuthenticationMode',
    'AuthenticationStatus',
    'AuthenticationPhase',
    'SecurityLevel',
    'get_real_authentication_system',
    'get_authentication_system',
    'AuthenticationSystem',
    
    # Versión
    '__version__'
]

# Mensaje de confirmación
import logging
logger = logging.getLogger(__name__)
logger.info(f"✅ Core Biometric System v{__version__} inicializado")
logger.info(f"✅ 15 módulos principales + 2 auxiliares cargados")
logger.info(f"✅ Enrollment System (Módulo 14) integrado")
logger.info(f"✅ Authentication System (Módulo 15) integrado")