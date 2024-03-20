"""task.py"""
from textwrap import dedent
from langchain.prompts import PromptTemplate


def AI_Engineer_expert_task() -> PromptTemplate:
    """
    Generate a prompt template for the AI Engineer Expert Task.

    This function returns a PromptTemplate object containing a structured template
    for the AI Engineer Expert task. The template includes placeholders for the user's
    query and the context extracted from retrieved documents.

    Returns:
        PromptTemplate: Template for the AI Engineer Expert task.
    """
    return PromptTemplate(
        template=dedent("""
            # Task Description
            As an AI Engineer Expert at Mistral Company, specializing in developing cutting-edge AI technology for developers,
            your task is to analyze and offer comprehensive insights on user inquiries based on the contents of the retrieved documents.
            Ensure that your final responses are user-friendly and formatted as posts.
            If the question cannot be answered using the information provided answer with "I don't know".

            # User
            Question: {query}

            # Documents
            Here are the retrieved documents:
            "{context}"
            """),
        input_variables=["query", "context"],
    )
