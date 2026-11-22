import os
import multiprocessing
from pathlib import Path

# Resolve base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Storage paths
    STORAGE_BASE_DIR: Path = Path(os.getenv("STORAGE_BASE_DIR", str(BASE_DIR / "storage"))).resolve()
    RAW_STORAGE_DIR: Path = STORAGE_BASE_DIR / "raw"
    INTERMEDIATE_STORAGE_DIR: Path = STORAGE_BASE_DIR / "intermediate"
    PROCESSED_STORAGE_DIR: Path = STORAGE_BASE_DIR / "processed"

    # Database
    DATABASE_PATH: Path = STORAGE_BASE_DIR / "autodata_hub.db"
    DATABASE_URL: str = f"sqlite:///{DATABASE_PATH.as_posix()}"

    # Resource & File Boundaries
    MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(1024 * 1024 * 1024))) # 1 GB
    MAX_ZIP_EXTRACT_SIZE_BYTES: int = int(os.getenv("MAX_ZIP_EXTRACT_SIZE_BYTES", str(2 * 1024 * 1024 * 1024))) # 2 GB
    MAX_ZIP_RATIO: float = 10.0 # 10:1 ratio ceiling to prevent zip bombs
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "5000"))

    # Hardware & Performance Constraints
    CPU_CORES: int = multiprocessing.cpu_count()
    # Workers = max(1, CPU_CORES - 1), capped at 3 for 8GB i5-6300U dev environment
    MAX_WORKER_PROCESSES: int = int(os.getenv("MAX_WORKER_PROCESSES", str(max(1, min(CPU_CORES - 1, 3)))))

    # Supported File Extensions
    TABULAR_EXTENSIONS: set = {".csv", ".tsv", ".xls", ".xlsx"}
    IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    ARCHIVE_EXTENSIONS: set = {".zip"}

settings = Settings()

# Ensure directories exist
for d in [settings.STORAGE_BASE_DIR, settings.RAW_STORAGE_DIR, settings.INTERMEDIATE_STORAGE_DIR, settings.PROCESSED_STORAGE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
