"""state.py"""
from typing_extensions import TypedDict, List


class State(TypedDict):
    """
    Represents the state of our graph.

    This class defines the structure of the state used in the graph, including various attributes
    representing different aspects of the document retrieval and processing pipeline.

    Attributes:
        query (str): The query used for document retrieval or processing.
        retrieved_documents (List): List of retrieved documents.
        documents (List): List of processed documents.
        reranked_documents (List): List of documents after reranking, if applicable.
        generation (List[str]): List of generated content
    """
    query: str
    documents: List
    generation: str
