"""deepeval"""

from deepeval.models import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.chat_completion.retry import retry_with_exponential_backoff


class GoogleGenerativeAIModel(DeepEvalBaseLLM):
    """
    Wrapper class for Google Generative AI model.

    This class extends DeepEvalBaseLLM and provides methods to load and utilize
    the Google Generative AI model for generation.

    Attributes:
        model_name (str): Name of the Google Generative AI model.

    Methods:
        __init__: Initializes the GoogleGenerativeAIModel instance.
        load_model: Loads the Google Generative AI model.
        generate: Generates chat response synchronously.
        a_generate: Generates chat response asynchronously.
        get_model_name: Returns the name of the loaded model.
    """

    def __init__(
        self,
        model: str = "gemini-pro",
        *args,
        **kwargs,
    ):
        """
        Initialize GoogleGenerativeAIModel instance.

        Args:
            model (Optional[str]): Name of the Google Generative AI model to use.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(model, *args, **kwargs)

    def load_model(self):
        """
        Load the Google Generative AI model.

        Returns:
            ChatGoogleGenerativeAI: Instance of the loaded model.
        """
        return ChatGoogleGenerativeAI(model=self.model_name)

    @retry_with_exponential_backoff
    def generate(self, prompt: str) -> str:
        """
        Generate chat response synchronously.

        Args:
            prompt (str): Prompt text to generate response.

        Returns:
            str: Generated chat response.
        """
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    @retry_with_exponential_backoff
    async def a_generate(self, prompt: str) -> str:
        """
        Generate chat response asynchronously.

        Args:
            prompt (str): Prompt text to generate response.

        Returns:
            str: Generated chat response.
        """
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        """
        Get the name of the loaded model.

        Returns:
            str: Name of the loaded model.
        """
        return self.model_name
