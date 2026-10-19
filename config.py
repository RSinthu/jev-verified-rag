"""Settings, logger and shared clients. Import from here instead of reading .env elsewhere."""

import logging
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typesafe_sdk import TypeSafeClient

BASE_DIR = Path(__file__).resolve().parent

# Paths
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}


class Settings(BaseSettings):
    """Read from environment variables or .env (names are case-insensitive, e.g. TAU=0.3)."""

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # API keys (required)
    typesafe_api_key: SecretStr
    openai_api_key: SecretStr

    # Models
    jev_model: str = "jev-latest"
    openai_model: str = "gpt-6.1-sol"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Uploads
    max_upload_mb: int = 20

    # Retrieval
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 20                       # candidates taken from each search and kept after fusion
    rrf_k: int = 60                       # reciprocal rank fusion: score = 1 / (rrf_k + rank)
    tau: float = Field(0.5, ge=0, le=1)   # Jev cutoff: keep chunks and pass gates at p >= tau

    log_level: str = "INFO"


settings = Settings()

# Logger
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide a log line for every API call
logger = logging.getLogger("jev_rag")

# Shared clients, created once when the app starts
embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
jev = TypeSafeClient(api_key=settings.typesafe_api_key.get_secret_value(), model=settings.jev_model)
llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
