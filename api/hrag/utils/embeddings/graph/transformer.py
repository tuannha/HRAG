import logging
from typing import Any, Dict, Optional, cast

from hrag.models import Tenant
from langchain_core.runnables import RunnableConfig
from langchain_experimental.graph_transformers.llm import (
    Document,
    GraphDocument,
    LLMGraphTransformer,
    Node,
    Relationship,
    _convert_to_graph_document,
)
from settings import app_settings

logger = logging.getLogger("gunicorn.error")
logger.setLevel(app_settings.log_level.upper())


class HybridRagGraphTransformer(LLMGraphTransformer):
    """
    Override the original GraphTransformer process_response method to add the tenant prefix for each node
    """

    def __init__(self, tenant: Tenant, **kwargs):
        super().__init__(**kwargs)

        self.tenant = tenant

    def process_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints.
        """
        text = document.page_content
        raw_schema = self.chain.invoke({"input": text}, config=config)
        if self._function_call:
            raw_schema = cast(Dict[Any, Any], raw_schema)
            nodes, relationships = _convert_to_graph_document(raw_schema)
        else:
            nodes_set = set()
            relationships = []
            parsed_json = self.json_repair.loads(raw_schema.content)
            # handle invalid relationship
            for rel in parsed_json:
                if (
                    not rel.get("head")
                    or not rel.get("head_type")
                    or not rel.get("tail")
                    or not rel.get("tail_type")
                    or not rel.get("relation")
                    or not isinstance(rel.get("relation"), str)
                ):
                    logger.debug("%s is not a valid relationship", rel)
                    continue

                source_nodes = []
                target_nodes = []
                # Nodes need to be deduplicated using a set
                if isinstance(rel["head"], list):
                    for head in rel["head"]:
                        head_id = f"{self.tenant.tenant}::{'-'.join(head.split())}"
                        nodes_set.add((head_id, rel["head_type"]))
                        source_nodes.append(Node(id=head_id, type=rel["head_type"]))
                else:
                    head_id = f"{self.tenant.tenant}::{'-'.join(rel['head'].split())}"
                    nodes_set.add((head_id, rel["head_type"]))
                    source_nodes.append(Node(id=head_id, type=rel["head_type"]))

                if isinstance(rel["tail"], list):
                    for tail in rel["tail"]:
                        tail_id = f"{self.tenant.tenant}::{'-'.join(tail.split())}"
                        nodes_set.add((tail_id, rel["tail_type"]))
                        target_nodes.append(Node(id=tail_id, type=rel["tail_type"]))
                else:
                    tail_id = f"{self.tenant.tenant}::{'-'.join(rel['tail'].split())}"
                    nodes_set.add((tail_id, rel["tail_type"]))
                    target_nodes.append(Node(id=tail_id, type=rel["tail_type"]))

                for source_node in source_nodes:
                    for target_node in target_nodes:
                        relationships.append(
                            Relationship(
                                source=source_node,
                                target=target_node,
                                type=rel["relation"],
                            )
                        )
            # Create nodes list
            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]

        # Strict mode filtering
        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                    in [el.lower() for el in self.allowed_relationships]
                ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)
