import os, sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from utils.config_handler import rag_conf

# DashScope (Alibaba Cloud Model Studio) OpenAI-compatible endpoint.
# The legacy dashscope.Generation API no longer serves newer models
# (qwen3.x series), so chat models go through the compatible-mode endpoint.
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=DASHSCOPE_BASE_URL,
            temperature=0.1,
            # Hard request timeout + minimal retries so a hung upstream call
            # fails fast (and the agent can degrade) instead of blocking forever.
            timeout=60,
            max_retries=1,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


class _LazyModel:
    """Deferred model construction.

    `ChatOpenAI` validates the api_key at *construction* time, so building it
    at import time breaks any environment without credentials (CI, tests).
    This proxy keeps the old `from model.factory import chat_model` import
    syntax working everywhere while the real model is only created on first
    attribute access / call — i.e. exactly when an agent actually needs it.
    """

    def __init__(self, factory: BaseModelFactory) -> None:
        self._factory = factory
        self._obj = None

    def _get(self):
        if self._obj is None:
            self._obj = self._factory.generator()
        return self._obj

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __call__(self, *args, **kwargs):
        return self._get()(*args, **kwargs)


chat_model = _LazyModel(ChatModelFactory())
embed_model = _LazyModel(EmbeddingsFactory())
