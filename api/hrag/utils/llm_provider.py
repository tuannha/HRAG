import logging

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from .enums import LanguageModelProvider

log = logging.getLogger("gunicorn.error")


def get_llm_provider_enum(provider_name: str):
    provider_map = {
        "ollama": LanguageModelProvider.ollama,
        "open_ai": LanguageModelProvider.open_ai,
    }

    return provider_map.get(
        provider_name.lower().strip(),
        LanguageModelProvider.ollama,
    )


def get_necessary_llms(tenant):
    if tenant.provider == LanguageModelProvider.open_ai:
        return __open_ai_llms(tenant)

    return __ollama_llms(tenant)


def __open_ai_llms(tenant):
    log.debug("Loading OpenAI provider")

    chat_llm = ChatOpenAI(
        model_name=tenant.llm_model,
        temperature=0,
        openai_api_key=tenant.provider_api_key,
    )
    json_chat_llm = ChatOpenAI(
        model_name=tenant.llm_model,
        temperature=0,
        openai_api_key=tenant.provider_api_key,
        format="json",
    )
    embedding_llm = OpenAIEmbeddings(
        model=tenant.embedding_model,
        openai_api_key=tenant.provider_api_key,
    )

    return {
        "chat": chat_llm,
        "json_chat": json_chat_llm,
        "embedding": embedding_llm,
    }


def __ollama_llms(tenant):
    log.debug("Loading Ollama provider")

    chat_llm = ChatOllama(
        model=tenant.llm_model,
        temperature=0,
        base_url=tenant.provider_endpoint,
    )
    json_chat_llm = ChatOllama(
        model=tenant.llm_model,
        temperature=0,
        base_url=tenant.provider_endpoint,
        format="json",
    )
    embedding_llm = OllamaEmbeddings(
        model=tenant.embedding_model,
        base_url=tenant.provider_endpoint,
    )

    return {
        "chat": chat_llm,
        "json_chat": json_chat_llm,
        "embedding": embedding_llm,
    }
