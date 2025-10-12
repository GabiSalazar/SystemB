from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # Información del proyecto
    PROJECT_NAME: str = "Biometric Gesture System"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000"]
    
    # Rutas
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    BIOMETRIC_DATA_DIR: Path = BASE_DIR / "biometric_data"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # MediaPipe
    MEDIAPIPE_MODEL_PATH: Optional[str] = None
    
    # Cámara
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    CAMERA_FPS: int = 30
    
    # Umbrales
    HAND_CONFIDENCE_THRESHOLD: float = 0.9
    GESTURE_CONFIDENCE_THRESHOLD: float = 0.6
    
    # Supabase
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    
    # Seguridad
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()