from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    app_env: str = Field("local", alias="APP_ENV")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    api_title: str = Field("PokerRag", alias="API_TITLE")
    api_version: str = Field("0.0.1", alias="API_VERSION")
    api_docs: str = Field("/docs", alias="API_DOCS")
    cors_origins: str = Field("*", alias="CORS_ORIGINS")
    request_timeout_secs: int = Field(20, alias="REQUEST_TIMEOUT_SECS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
