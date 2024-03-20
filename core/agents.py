"""agents.py"""

from typing import Any
from core.state import State
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.tasks import AI_Engineer_expert_task
from langchain_core.output_parsers.string import StrOutputParser


def index_retriever_agent(state: State, retriever: Any, k: int) -> State:
    """
    Retrieve documents based on a query.

    Args:
        state (State): The current state of the document retrieval process.
        retriever: The document retriever object capable of performing similarity searches.
        k (int): The number of documents to retrieve.

    Returns:
        State: Updated state with retrieved documents.
    """

    state["documents"] = retriever.similarity_search(
        state["query"], k=k
    )

    return state


def rerank_agent(state: State, reranker, k) -> State:
    """
    Rerank documents based on a given reranker.

    This function applies a reranking algorithm to the documents stored in the state,
    aiming to improve the relevance of the retrieved documents with respect to the query.

    Args:
        state (State): The current state of the document retrieval and processing pipeline.
        reranker: The reranking component capable of reordering the documents.
        k (int): The number of top documents to consider for reranking.

    Returns:
        State: Updated state with reranked documents.
    """
    reranked_documents = []

    for doc in reranker.rerank(
        query=state["query"], documents=state["documents"], top_n=k
    ):
        if doc["relevance_score"] > 0.7:
            reranked_documents.append(state["documents"][doc["index"]])

    state["documents"] = reranked_documents

    return state


def rag_website_agent(state: State, retriever, k) -> State:
    """
    Process documents from a Website, and then Retrieve documents based on a query.

    This agent assumes that documents have been previously retrieved and stored in the state.

    Args:
        state (State): The current state of the document retrieval process.
        retriever: The document retriever object capable of performing similarity searches.
        k (int): The number of documents to retrieve.

    Returns:
        State: Updated state with processed documents.
    """

    retrieved_document = state["documents"][0]

    loader = WebBaseLoader(retrieved_document.metadata["url"])
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )

    state["documents"] = retriever(
        documents=text_splitter.split_documents(documents)
    ).similarity_search(state["query"], k=k)

    return state


def expert_mistral_agent(state: State, llm: Any) -> State:
    """
    Process query and documents using a Large Language Model.

    This agent utilizes a LLM to generate output based on the provided query and context
    Args:
        state (State): The current state of the document processing.
        llm: The expert model for processing documents.

    Returns:
        State: Updated state with processed output.
    """

    chain = AI_Engineer_expert_task() | llm | StrOutputParser()

    context = "\n\n".join(doc.page_content for doc in state["documents"])

    state["generation"] = chain.invoke(
        {
            "query": state["query"],
            "context": context,
        }
    )

    return state
