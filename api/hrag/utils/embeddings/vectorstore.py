import logging

from hrag.models import Tenant
from langchain.chains.summarize import load_summarize_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

# from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from settings import app_settings

logger = logging.getLogger("gunicorn.error")


class HybridRagEmbeddings:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    def __init__(self, tenant: Tenant):
        super().__init__()

        self.retriever_name = "Tenant Retriever"
        self.tenant = tenant

        self.collection_name = self.tenant.tenant
        llms = self.tenant.llms

        self.embeddings = llms["embedding"]
        self.summary_llm = llms["chat"]

        self.db = PGVector(
            connection_string=app_settings.postgresql_url,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            use_jsonb=True,
        )

        # self.text_splitter = SemanticChunker(self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=256,
        )

    def add_documents(self, docs):
        splits = self.text_splitter.split_documents(docs)
        splits = [split for split in splits if split.page_content.strip()]
        self.db.add_documents(splits)

        if self.tenant.enable_summary_embedding:
            summary_chain = load_summarize_chain(
                llm=self.summary_llm, chain_type="map_reduce"
            )
            summary = summary_chain.invoke(splits)
            summary_doc = Document(
                page_content=summary["output_text"],
                metadata=splits[0].metadata,
            )
            self.db.add_documents([summary_doc])

    def get_retriever(self):
        base_retriever = self.db.as_retriever(
            search_type="similarity",
        )
        # reranking
        if app_settings.enable_reranking:
            compressor = FlashrankRerank()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
            )

            return compression_retriever

        return base_retriever

    def get_retriever_tool(self):
        retriever = self.get_retriever()
        return create_retriever_tool(
            retriever=retriever,
            name=self.retriever_name,
            description=f"""Extra documents provided by {self.tenant.tenant}. Use this tool for retrieving extra context when needed.""",  # NOQA
        )

    def truncate(self):
        self.db.delete_collection()
