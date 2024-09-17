import logging
from typing import List

from hrag.models import Tenant
from hrag.utils.embeddings import HybridRagEmbeddings, HybridRagEntityGraph
from hrag.utils.enums import PromptType
from hrag.utils.prompts import HybridRagPrompt
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
from settings import app_settings
from typing_extensions import TypedDict

logger = logging.getLogger("gunicorn.error")
logger.setLevel(app_settings.log_level.upper())


class GraphState(TypedDict):
    question: str
    reformed_question: str
    generation: str
    documents: List[str]
    relationships: str


class HybridRagGraph:
    def __init__(
        self,
        tenant: Tenant,
        chat_history: list = None,
        old_summary: str = None,
    ):
        self.chat_history = chat_history or []
        self.old_summary = old_summary or ""

        self.tenant = tenant
        self.llm = self.tenant.llms["chat"]
        self.json_llm = self.tenant.llms["json_chat"]

        # load old history
        self.message_history = ChatMessageHistory()
        for history in self.chat_history:
            if history["role"] == "ai":
                self.message_history.add_ai_message(history["content"])
            else:
                self.message_history.add_user_message(history["content"])

        if not old_summary:  # init memory buffer if needed
            self.memory = ConversationSummaryMemory.from_messages(
                llm=self.llm,
                chat_memory=self.message_history,
                memory_key="chat_history",
            )
        else:  # do not init memory buffer
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                chat_memory=self.message_history,
                buffer=old_summary,
                memory_key="chat_history",
            )

        self.graph = self.build_graph_workflow()

    # DEFINING LLM FUNCTIONS
    def tenant_retriever(self):
        return HybridRagEmbeddings(tenant=self.tenant).get_retriever()

    def graph_retriever(self):
        return HybridRagEntityGraph(tenant=self.tenant)

    def retriever_grader_chain(self):
        prompt = HybridRagPrompt.get_prompt(
            PromptType.RETRIEVER_GRADER, self.tenant.prompt_family
        )
        return prompt | self.json_llm | JsonOutputParser()

    def generate_rag_answer_chain(self):
        prompt = HybridRagPrompt.get_prompt(
            PromptType.GENERATE_RAG_ANSWER, self.tenant.prompt_family
        )
        return prompt | self.llm | StrOutputParser()

    def generate_regular_answer_chain(self):
        prompt = HybridRagPrompt.get_prompt(
            PromptType.GENERATE_REGULAR_ANSWER, self.tenant.prompt_family
        )
        return prompt | self.llm | StrOutputParser()

    def hallucination_grader_chain(self):
        prompt = HybridRagPrompt.get_prompt(
            PromptType.HALLUCINATION_GRADER, self.tenant.prompt_family
        )
        return prompt | self.json_llm | JsonOutputParser()

    def answer_grader_chain(self):
        prompt = HybridRagPrompt.get_prompt(
            PromptType.ANSWER_GRADER, self.tenant.prompt_family
        )
        return prompt | self.json_llm | JsonOutputParser()

    def reform_question_chain(self):
        prompt = HybridRagPrompt.get_prompt(
            PromptType.REFORM_QUESTION, self.tenant.prompt_family
        )
        return prompt | self.llm | StrOutputParser()

    # DEFINING LANG GRAPH NODES AND CONDITIONAL EDGES
    def reform_question(self, state):
        chain = self.reform_question_chain()
        question = state["question"]
        if len(self.message_history.messages) > 0:
            logger.debug("Reform question based on old chat history: %s", question)
            reformed_question = chain.invoke(
                {
                    "question": question,
                    "chat_history": get_buffer_string(
                        self.message_history.messages[-10:]
                    ),
                }
            )
            logger.debug("---Reformed question: %s", reformed_question)
        else:
            logger.debug("First initial message, do not reform the question")
            reformed_question = question
            logger.debug("---Reformed question: %s", reformed_question)

        return {
            "question": question,
            "reformed_question": reformed_question,
            "relationships": "",  # TODO move this to graph document
        }

    def retrieve_graph_documents(self, state):
        question = state["question"]
        reformed_question = state["reformed_question"]

        logger.debug("Retrieve graph documents: %s", question)
        logger.debug("---Reformed question: %s", reformed_question)
        relationships = self.graph_retriever().retrieve_info(reformed_question)
        logger.debug("---Retrieved graph documents: %s", relationships)

        return {
            "relationships": relationships,
            "question": question,
            "reformed_question": reformed_question,
        }

    def retrieve_tenant_documents(self, state):
        question = state["question"]
        relationships = state["relationships"]
        reformed_question = state["reformed_question"]

        logger.debug("Retrieve tenant documents: %s", question)
        logger.debug("---Reformed question: %s", reformed_question)
        documents = self.tenant_retriever().invoke(reformed_question)
        logger.debug("---Retrieved tenant documents: %s", documents)

        return {
            "documents": documents,
            "question": question,
            "reformed_question": reformed_question,
            "relationships": relationships,
        }

    def grade_documents(self, state):
        question = state["question"]
        documents = state["documents"]
        relationships = state["relationships"]
        reformed_question = state["reformed_question"]

        logger.debug("Grade documents: %s", question)
        logger.debug("---Reformed question: %s", reformed_question)

        grader = self.retriever_grader_chain()

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = grader.invoke(
                {"question": reformed_question, "document": d.page_content}
            )
            grade = score["score"]
            # Document relevant
            if grade.lower() == "yes":
                logger.debug("------grade: document relevant")
                filtered_docs.append(d)
            # Document not relevant
            else:
                logger.debug("------grade: document irrelevant")

        return {
            "documents": filtered_docs,
            "question": question,
            "reformed_question": reformed_question,
            "relationships": relationships,
        }

    def has_relevant_documents(self, state):
        question = state["question"]
        documents = state["documents"]

        logger.debug("Check if there is relevant document: %s", question)
        if len(documents) > 0:
            logger.debug("---has relevant documents")
            return "yes"

        logger.debug("---has no relevant documents")
        return "no"

    def generate_rag_answer(self, state):
        question = state["question"]
        documents = state["documents"]
        relationships = state["relationships"]
        reformed_question = state["reformed_question"]

        logger.debug("Generate RAG answer: %s", question)
        logger.debug("---relationships: %s", relationships)
        logger.debug("---document: %s", documents)
        logger.debug("---questions: %s", question)
        logger.debug("---reformed_questions: %s", reformed_question)

        # RAG generation
        generation = self.generate_rag_answer_chain().invoke(
            {
                "context": documents,
                "question": reformed_question,
                "relationships": relationships,
            }
        )
        return {
            "documents": documents,
            "relationships": relationships,
            "question": question,
            "reformed_question": reformed_question,
            "generation": generation,
        }

    def generate_regular_answer(self, state):
        question = state["question"]
        reformed_question = state["reformed_question"]
        relationships = state["relationships"]

        logger.debug("Generate regular answer: %s", question)
        logger.debug("---relationships: %s", relationships)
        logger.debug("---question: %s", question)
        logger.debug("---reformed_question: %s", reformed_question)
        logger.debug("---context: %s", self.memory.buffer)
        generation = self.generate_regular_answer_chain().invoke(
            {
                "summary": self.memory.buffer,
                "history": self.message_history.messages[-10:],
                "question": question,
            }
        )
        return {
            "question": question,
            "reformed_question": reformed_question,
            "generation": generation,
            "relationships": relationships,
        }

    def check_answer(self, state):
        question = state["question"]
        reformed_question = state["reformed_question"]
        documents = state["documents"]
        generation = state["generation"]
        relationships = state["relationships"]

        logger.debug("Check generated answer: %s", question)
        logger.debug("---reformed_question: %s", reformed_question)
        logger.debug("---generated answer: %s", generation)
        logger.debug("------Check hallucination for answer")
        logger.debug("---------Provided document: %s", documents)
        hallucination_score = self.hallucination_grader_chain().invoke(
            {
                "documents": documents,
                "relationships": relationships,
                "generation": generation,
                "chat_history": self.memory.buffer,
            }
        )
        logger.debug("---------hallucination score: %s", hallucination_score)

        # Check hallucination
        if hallucination_score.get("score").lower().strip() == "yes":
            logger.debug("---------grade: generation is grounded in documents")
            # Check question-answering
            logger.debug("------Check question-answering")
            answer_score = self.answer_grader_chain().invoke(
                {"question": question, "generation": generation}
            )
            logger.debug("---------question-answering score: %s", answer_score)
            if answer_score.get("score").lower().strip() == "yes":
                logger.debug("---------grade: generation addresses question")
                return "useful"
            else:
                logger.debug("---------grade: generation does not address question")
                return "not useful"
        else:
            logger.debug(
                "---------grade: generation is not grounded in documents, switching to regular answer"
            )
            return "not supported"

    def build_graph_workflow(self):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("reform_question", self.reform_question)
        workflow.add_node("retrieve_graph_documents", self.retrieve_graph_documents)
        workflow.add_node("retrieve_tenant_documents", self.retrieve_tenant_documents)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate_rag_answer", self.generate_rag_answer)
        workflow.add_node("generate_regular_answer", self.generate_regular_answer)

        # Define the edges
        workflow.set_entry_point("reform_question")
        workflow.add_edge("reform_question", "retrieve_graph_documents")
        workflow.add_edge("retrieve_graph_documents", "retrieve_tenant_documents")
        # workflow.add_edge("reform_question", "retrieve_tenant_documents")
        workflow.add_edge("retrieve_tenant_documents", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.has_relevant_documents,
            {
                "yes": "generate_rag_answer",
                "no": "generate_regular_answer",
            },
        )
        workflow.add_conditional_edges(
            "generate_rag_answer",
            self.check_answer,
            {
                "not supported": "generate_rag_answer",
                "useful": END,
                "not useful": "generate_regular_answer",
            },
        )
        workflow.add_edge("generate_regular_answer", END)
        return workflow.compile()

    def generate_response(self, message):
        inputs = {"question": message}
        response = self.graph.invoke(inputs)

        self.memory.save_context({"human": message}, {"ai": response["generation"]})
        logger.debug("New memory: %s", self.memory.buffer)

        return response["generation"]
