import logging

import sqlalchemy as db
from fastapi.exceptions import HTTPException
from hrag.utils.enums import LanguageModelProvider, LLMFamily
from hrag.utils.llm_provider import get_llm_provider_enum, get_necessary_llms
from settings import app_settings

from .base import Base
from .base_model import BaseModel

log = logging.getLogger("gunicorn.error")

default_llm_provider = get_llm_provider_enum(app_settings.default_llm_provider)


class Tenant(BaseModel, Base):
    __tablename__ = "tenant"
    _primary_key_names = ["id"]

    tenant = db.Column(
        db.Unicode(128),
        nullable=False,
        unique=True,
        index=True,
    )
    provider = db.Column(
        db.Enum(LanguageModelProvider),
        nullable=False,
        default=LanguageModelProvider.ollama,
    )
    provider_endpoint = db.Column(
        db.Unicode(1024),
        nullable=True,
    )
    provider_api_key = db.Column(
        db.Unicode(1024),
        nullable=True,
    )
    llm_model = db.Column(
        db.Unicode(128),
        nullable=True,
    )
    embedding_model = db.Column(
        db.Unicode(128),
        nullable=True,
    )
    enable_summary_embedding = db.Column(
        db.Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    unique_columns = ["tenant"]
    prompt_family = None  # a non db field

    def __init__(
        self,
        tenant,
        provider,
        provider_endpoint=None,
        provider_api_key=None,
        llm_model=None,
        embedding_model=None,
        enable_summary_embedding=True,
        status=True,
    ):
        super().__init__(status=status)
        self.tenant = tenant
        self.provider = provider
        self.provider_endpoint = provider_endpoint
        self.provider_api_key = provider_api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.enable_summary_embedding = enable_summary_embedding

    @classmethod
    def get_tenant(
        cls,
        tenant: str,
        create_if_not_exist: bool = app_settings.create_tenant_if_not_exists,
        include_inactive: bool = False,
    ):
        from fastapi_sqlalchemy import db as db_session

        query = db_session.session.query(Tenant).filter(Tenant.tenant == tenant)
        if not include_inactive:
            query.filter(Tenant.status == True)

        tenant_obj = query.first()

        if not tenant_obj and create_if_not_exist:
            log.debug(f"Unknown tenant {tenant}, creating a default one ...")
            tenant_obj = cls.create_tenant(tenant=tenant)

        if not tenant_obj:
            log.debug("Unknown or inactive tenant")
            raise HTTPException(status_code=404, detail="No active tenant found.")

        if "llama" in tenant_obj.llm_model.lower():
            tenant_obj.prompt_family = LLMFamily.llama
        else:
            tenant_obj.prompt_family = LLMFamily.other
        log.debug(f"prompt family: {tenant_obj.prompt_family}")

        return tenant_obj

    @classmethod
    def create_tenant(
        cls,
        tenant: str,
        provider: LanguageModelProvider = default_llm_provider,
        provider_endpoint: str = app_settings.default_llm_endpoint,
        provider_api_key: str = app_settings.default_llm_api_key,
        llm_model: str = app_settings.default_llm_model,
        embedding_model: str = app_settings.default_embedding_model,
        enable_summary_embedding: bool = True,
    ):
        from fastapi_sqlalchemy import db as db_session

        tenant_obj = cls(
            tenant=tenant,
            provider=provider,
            provider_endpoint=provider_endpoint,
            provider_api_key=provider_api_key,
            llm_model=llm_model,
            embedding_model=embedding_model,
            enable_summary_embedding=enable_summary_embedding,
        )

        db_session.session.add(tenant_obj)
        db_session.session.flush()

        return tenant_obj

    @property
    def llms(self):
        return get_necessary_llms(self)

    @property
    def to_dict(self):
        return {
            "id": self.id,
            "tenant": self.tenant,
            "provider": self.provider,
            "provider_endpoint": self.provider_endpoint,
            "provider_api_key": self.provider_api_key,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "enable_summary_embedding": self.enable_summary_embedding,
            "status": self.status,
            "created_dt": self.created_dt,
            "updated_dt": self.updated_dt,
        }

    def deactivate(self):
        from fastapi_sqlalchemy import db as db_session

        self.status = False
        db_session.session.add(self)
        db_session.session.flush()

    def activate(self):
        from fastapi_sqlalchemy import db as db_session

        self.status = True
        db_session.session.add(self)
        db_session.session.flush()

    def update(
        self,
        provider: LanguageModelProvider = None,
        provider_endpoint: str = None,
        provider_api_key: str = None,
        llm_model: str = None,
        embedding_model: str = None,
        enable_summary_embedding: bool = None,
    ):
        from fastapi_sqlalchemy import db as db_session

        if provider is not None:
            self.provider = provider
        if provider_endpoint is not None:
            self.provider_endpoint = provider_endpoint
        if provider_api_key is not None:
            self.provider_api_key = provider_api_key
        if llm_model is not None:
            self.llm_model = llm_model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        if enable_summary_embedding is not None:
            self.enable_summary_embedding = enable_summary_embedding

        db_session.session.add(self)
        db_session.session.flush()
