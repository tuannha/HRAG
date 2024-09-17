from http import HTTPStatus

from fastapi import HTTPException, UploadFile
from fastapi_versioning import version
from hrag.routers.router import documents
from hrag.utils.document import (
    add_document_to_tenant,
    cleanup_documents_from_tenant,
)
from hrag.utils.exceptions import HybridRagException


@documents.post(
    "/",
    description="Add a document to tenant. Accepting txt, docx, pdf files.",
    status_code=HTTPStatus.ACCEPTED,
)
@version(1)
async def add_document(
    tenant: str,
    document: UploadFile,
):
    try:
        await add_document_to_tenant(tenant, document)
    except HybridRagException as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_GATEWAY, detail=str(e))


@documents.delete(
    "/",
    description="Remove existing known documents from tenant.",
    status_code=HTTPStatus.ACCEPTED,
)
@version(1)
async def cleanup_documents(tenant: str):
    try:
        await cleanup_documents_from_tenant(tenant)
    except HybridRagException as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_GATEWAY, detail=str(e))
