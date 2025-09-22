from taxmusr.core.chat_model import EnhancedChatModel
from taxmusr.core.schemas import WorkflowOutput
from taxmusr.domains.joint_assessment.prompts import EVALUATION_PROMPT


class Workflow:
    """Base Class for a GenAI workflow."""

    def __init__(
            self,
            model: str,
            temperature: float = 1.0,
            top_p: float = 1.0,
            max_tokens: int = 2048,
            cot: bool = False,
            num_examples: int = 0,  # zero-shot or few-shot
            few_shot_examples: list = None,
            **kwargs
    ):
        """
        Initializes the workflow with the given parameters.
        :param model: The {model_provider}:{model} to use for the workflow.
        :param temperature: The temperature to use for generation. Higher values mean more random completions.
        :param top_p: The top_p to use for nucleus sampling.
        :param max_tokens: The maximum number of tokens to generate in the completion.
        :param cot: Whether to use chain-of-thought prompting.
        :param num_examples: Number of few-shot examples to include. If 0, zero-shot is used.
        :param few_shot_examples: List of few-shot examples to use in the prompt.
        :param kwargs: Any additional parameters for specific workflows.
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.cot = cot
        self.num_examples = num_examples
        self.few_shot_examples = few_shot_examples if few_shot_examples is not None else []
        # Any additional parameters can be handled via kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.workflow = None
        self.callback_handler = None
        self.setup()

    def setup(self):
        """Set up the workflow, e.g., load models or resources."""
        pass

    def run(self, example) -> WorkflowOutput:
        """Run the workflow with the given input example.
        :param example: The input example to process.
        :return: A WorkflowOutput object containing the results.
        """
        pass


class BaselineWorkflow(Workflow):
    """A simple baseline workflow implementation."""
    def setup(self):
        # Set up any resources needed for the baseline workflow
        llm = EnhancedChatModel(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        generation_chain = EVALUATION_PROMPT | llm.model
        self.workflow = generation_chain
        self.callback_handler = llm.callback_handler

    def run(self, example) -> WorkflowOutput:
        chain_args = {
            "narrative": example["narrative"],
            "question": example["question"],
            "options": example["options"],
        }
        if self.cot:
            cot_text = "Explain your reasoning step by step before you answer."
            chain_args["cot"] = cot_text
        else:
            chain_args["cot"] = ""
        if self.few_shot_examples:
            example_blocks = []
            for ex in self.few_shot_examples[:self.num_examples]:
                ex_block = f"""STORY:\n{ex["narrative"]}\n\nQUESTION:\n{ex["question"]}\n\n"ANSWER: {ex["answer"]}"."""
                example_blocks.append(ex_block)
            chain_args["examples"] = "Here are examples:\n\n" + "\n".join(example_blocks)
        else:
            chain_args["examples"] = ""

        response = self.workflow.invoke(
            chain_args,
            config={"callbacks": [self.callback_handler] if self.callback_handler else []}
        )
        response_content = response.content.strip()
        reasoning = response_content[:response_content.find("ANSWER:")].strip()
        predicted_answer = response_content[response_content.find("ANSWER:"):].replace("ANSWER:", "").strip()
        token_usage = response.response_metadata.get("token_usage", {})
        return WorkflowOutput(
            predicted_answer=predicted_answer,
            reasoning=reasoning,
            token_usage=token_usage
        )
