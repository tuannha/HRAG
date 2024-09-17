import os
from distutils.util import strtobool

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgresql_url: str = os.environ["POSTGRESQL_URL"]
    neo4j_url: str = os.environ["NEO4J_URL"]
    neo4j_username: str = os.environ["NEO4J_USERNAME"]
    neo4j_password: str = os.environ["NEO4J_PASSWORD"]
    migration_dir: str = os.path.join(os.getcwd(), "migrations")
    debug: bool = strtobool(os.environ.get("DEBUG", "False"))
    client_location: str = os.environ.get("CLIENT_LOCATION", "https://localhost:3000")

    default_llm_provider: str = os.environ.get("DEFAULT_LLM_PROVIDER", "ollama")
    default_llm_endpoint: str = os.environ.get(
        "DEFAULT_LLM_ENDPOINT", "http://localhost:11434"
    )
    default_llm_api_key: str = os.environ.get("DEFAULT_LLM_API_KEY", "")
    default_llm_model: str = os.environ.get("DEFAULT_LLM_MODEL", "llama3.1:8b")
    default_embedding_model: str = os.environ.get(
        "DEFAULT_EMBEDDING_MODEL", "nomic-embed-text:latest"
    )
    enable_reranking: bool = strtobool(os.environ.get("ENABLE_RERANKING", "true"))
    create_tenant_if_not_exists: bool = strtobool(
        os.environ.get("CREATE_TENANT_IF_NOT_EXISTS", "true")
    )

    log_level: str = os.environ.get("LOG_LEVEL", "debug")


app_settings = Settings()
