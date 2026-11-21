"""All configuration in one place, read from .env or the environment."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    planner_model: str = "gpt-4.1-mini"
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2/studies"
    max_studies: int = 10000
    cache_dir: str = ".cache"


# Imported as `from app.settings import settings` everywhere else.
settings = Settings()
