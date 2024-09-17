from http import HTTPStatus

from fastapi.exceptions import HTTPException
from fastapi_pagination import LimitOffsetPage
from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_versioning.versioning import version
from hrag.models import Tenant
from hrag.schema import (
    TenantCreateRequest,
    TenantResponse,
    TenantUpdateRequest,
)
from sqlalchemy import select

from ..router import tenants


@tenants.post(
    "/",
    summary="Create a new tenant",
    status_code=HTTPStatus.CREATED,
    response_model=TenantResponse,
)
@version(1)
async def create_tenant(
    create_request: TenantCreateRequest,
):
    try:
        tenant_obj = Tenant.create_tenant(
            tenant=create_request.tenant,
            provider=create_request.provider,
            provider_endpoint=create_request.provider_endpoint,
            provider_api_key=create_request.provider_api_key,
            llm_model=create_request.llm_model,
            embedding_model=create_request.embedding_model,
            enable_summary_embedding=create_request.enable_summary_embedding,
        )

        return tenant_obj
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@tenants.get(
    "/",
    summary="Get tenant list",
    status_code=HTTPStatus.OK,
    response_model=LimitOffsetPage[TenantResponse],
)
@version(1)
async def get_tenant_list():
    from fastapi_sqlalchemy import db as db_session

    return paginate(
        db_session.session, select(Tenant).order_by(Tenant.created_dt.desc())
    )


@tenants.get(
    "/{tenant}/",
    summary="Get a tenant",
    status_code=HTTPStatus.OK,
    response_model=TenantResponse,
)
@version(1)
async def get_tenant(tenant: str):
    return Tenant.get_tenant(tenant, create_if_not_exist=False)


@tenants.patch(
    "/{tenant}/",
    summary="Update a tenant",
    status_code=HTTPStatus.OK,
    response_model=TenantResponse,
)
@version(1)
async def update_tenant(tenant: str, update_request: TenantUpdateRequest):
    tenant_obj = Tenant.get_tenant(tenant, create_if_not_exist=False)
    tenant_obj.update(
        provider=update_request.provider,
        provider_endpoint=update_request.provider_endpoint,
        provider_api_key=update_request.provider_api_key,
        llm_model=update_request.llm_model,
        embedding_model=update_request.embedding_model,
        enable_summary_embedding=update_request.enable_summary_embedding,
    )

    return tenant_obj


@tenants.post(
    "/{tenant}/deactivate/",
    summary="Deactivate a tenant",
    status_code=HTTPStatus.OK,
    response_model=TenantResponse,
)
@version(1)
async def deactivate_tenant(tenant: str):
    tenant = Tenant.get_tenant(tenant, create_if_not_exist=False)
    tenant.deactivate()

    return tenant


@tenants.post(
    "/{tenant}/activate/",
    summary="Activate a tenant",
    status_code=HTTPStatus.OK,
    response_model=TenantResponse,
)
@version(1)
async def activate_tenant(tenant: str):
    tenant = Tenant.get_tenant(tenant, create_if_not_exist=False, include_inactive=True)
    tenant.activate()

    return tenant
