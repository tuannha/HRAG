import logging
import mimetypes
from tempfile import NamedTemporaryFile

from hrag.models import Tenant
from hrag.utils.embeddings import HybridRagEmbeddings, HybridRagEntityGraph
from hrag.utils.exceptions import HybridRagException
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader

logger = logging.getLogger("gunicorn.error")

EXTENSION_TO_LOADER_MAP = {
    "default": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}


async def cleanup_documents_from_tenant(tenant):
    tenant_obj = Tenant.get_tenant(tenant)
    if not tenant_obj:
        raise HybridRagException("Inactive tenant")

    embedding = HybridRagEmbeddings(tenant=tenant_obj)
    embedding.truncate()


async def add_document_to_tenant(tenant, document_file):
    tenant_obj = Tenant.get_tenant(tenant)
    if not tenant_obj:
        raise HybridRagException("Inactive tenant")

    file_name = document_file.filename
    logger.debug(f"processing document: {file_name}")
    mimetype = mimetypes.guess_type(file_name)[0]
    if mimetype:
        logger.debug(f"mime_type: {mimetype}")
        extension = mimetypes.guess_extension(mimetype)
    else:
        logger.debug("unknown mime_type, using file extension")
        extension = f".{file_name.split('.')[-1]}"

    logger.debug(f"file extension: {extension}")

    with NamedTemporaryFile(suffix=extension) as tmp:
        tmp.write(document_file.file.read())
        tmp.seek(0)

        docs = await to_embedding_documents(tmp.name, extension)

        if docs:
            # vector store
            embedding = HybridRagEmbeddings(tenant=tenant_obj)
            embedding.add_documents(docs)

            # graph relationship
            graph = HybridRagEntityGraph(tenant=tenant_obj)
            graph.add_documents(docs)


async def to_embedding_documents(file_name, extension):
    loader = EXTENSION_TO_LOADER_MAP.get(extension, EXTENSION_TO_LOADER_MAP["default"])

    return loader(file_name).load()
