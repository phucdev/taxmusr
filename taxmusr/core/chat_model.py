import os
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model


class EnhancedChatModel:
    def __init__(self, **kwargs):
        load_dotenv(find_dotenv())
        self.model = init_chat_model(**kwargs)
        self.callback_handler = None

        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY")
            )
            self.callback_handler = CallbackHandler()

        # TODO: Improve token and cost tracking
