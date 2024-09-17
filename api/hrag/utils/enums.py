from enum import Enum


class LanguageModelProvider(Enum):
    ollama = "Ollama"
    open_ai = "OpenAI"


class PromptType(Enum):
    REFORM_QUESTION = "reform_question"
    SUMMARY_CONVERSATION = "summary_conversation"
    RETRIEVER_GRADER = "retriever_grader"
    GENERATE_RAG_ANSWER = "generate_rag_answer"
    GENERATE_REGULAR_ANSWER = "generate_regular_answer"
    HALLUCINATION_GRADER = "hallucination_grader"
    ANSWER_GRADER = "answer_grader"
    EXTRACT_ENTITIES = "extract_entities"


class LLMFamily(Enum):
    llama = "llama"
    other = "other"
