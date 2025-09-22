from abc import ABC, abstractmethod
from taxmusr.core.schemas import ReasoningTree, GeneratedCase, StoryTemplate
from taxmusr.core.chat_model import EnhancedChatModel


class TaxDomain(ABC):
    """Abstract Base Class for a tax law domain."""

    @abstractmethod
    def construct_template(self) -> StoryTemplate:
        """Stage 1: Create gold facts and diversity facts."""
        pass

    @abstractmethod
    def complete_reasoning_tree(self, story_template: StoryTemplate, llm) -> ReasoningTree:
        """Stage 2: Expand facts into a full reasoning tree."""
        pass

    @abstractmethod
    def generate_story(self, reasoning_tree: ReasoningTree, llm: EnhancedChatModel) -> str:
        """Stage 3: Convert reasoning tree leaves into a narrative."""
        pass

    @abstractmethod
    def assemble_case(self, gold_facts, reasoning_tree, narrative) -> GeneratedCase:
        """Puts all the generated pieces together."""
        pass
