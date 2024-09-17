import uuid
from datetime import datetime
from typing import Optional

from hrag.utils.enums import LanguageModelProvider
from pydantic import BaseModel


class TenantUpdateRequest(BaseModel):
    provider: Optional[LanguageModelProvider] = None
    provider_endpoint: Optional[str] = None
    provider_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    enable_summary_embedding: Optional[bool] = None


class TenantCreateRequest(TenantUpdateRequest):
    tenant: str
    provider: LanguageModelProvider = LanguageModelProvider.ollama
    enable_summary_embedding: Optional[bool] = True


class TenantResponse(TenantCreateRequest):
    id: uuid.UUID
    status: bool
    created_dt: datetime
    updated_dt: datetime
