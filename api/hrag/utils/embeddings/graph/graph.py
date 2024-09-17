import logging
import re

from hrag.models import Tenant
from hrag.utils.enums import PromptType
from hrag.utils.prompts import HybridRagPrompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.output_parsers import JsonOutputParser
from settings import app_settings

from .transformer import HybridRagGraphTransformer

logger = logging.getLogger("gunicorn.error")
logger.setLevel(app_settings.log_level.upper())


class HybridRagEntityGraph:
    def __init__(self, tenant: Tenant):
        super().__init__()

        self.tenant = tenant
        llms = self.tenant.llms

        self.llm = llms["chat"]
        self.embeddings = llms["embedding"]

        self.graph = Neo4jGraph(
            url=app_settings.neo4j_url,
            username=app_settings.neo4j_username,
            password=app_settings.neo4j_password,
        )
        self.graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )

        prompt = HybridRagPrompt.get_prompt(
            PromptType.EXTRACT_ENTITIES,
            tenant.prompt_family,
        )
        self.entity_chain = prompt | self.llm | JsonOutputParser()

    def add_documents(self, raw_documents):
        documents = self.text_splitter.split_documents(raw_documents)
        documents = [
            document for document in documents if document.page_content.strip()
        ]

        llm_transformer = HybridRagGraphTransformer(llm=self.llm, tenant=self.tenant)
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        self.graph.add_graph_documents(
            graph_documents, baseEntityLabel=True, include_source=True
        )

    def generate_full_text_query(self, input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    # Fulltext index query
    def retrieve_info(self, question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = self.entity_chain.invoke({"question": question})
        for entity in entities:
            entity = f"{self.tenant.tenant}::{'-'.join(entity.split())}"
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            logger.debug("query response: %s", response)
            for el in response:
                # convert the entities and relationships back to normal
                output = re.sub(
                    rf"{re.escape(self.tenant.tenant)}::(.+) - (.+) -> {re.escape(self.tenant.tenant)}::(.+)",
                    r"\1 ::start:: \2 ::end:: \3",
                    el["output"],
                )
                output = output.replace("-", " ")
                output = output.replace("::start::", "-")
                output = output.replace("::end::", "->")
                result += "\n" + output

        return result
