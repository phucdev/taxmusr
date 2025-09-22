from tqdm import tqdm
from typing import List

from taxmusr.core.schemas import GeneratedCase
from taxmusr.core.chat_model import EnhancedChatModel
from taxmusr.domains.base import TaxDomain

class CaseGenerator:
    """Orchestrates the case generation workflow."""
    def __init__(self, domain: TaxDomain, **llm_kwargs):
        """
        :param domain: The tax domain to generate cases for.
        :param llm_kwargs: Keyword arguments for the LLM, e.g., model, temperature, top_p, max_tokens.
        """
        self.domain = domain
        self.llm = EnhancedChatModel(**llm_kwargs)

    def generate(self, num_cases: int) -> List[GeneratedCase]:
        """Generates a list of tax cases.
        :param num_cases: The number of cases to generate.
        :return: A list of GeneratedCase objects.
        """
        cases = []
        for _ in tqdm(range(num_cases), desc="Generating cases"):
            # Stage 1: Tree Template Construction
            story_template = self.domain.construct_template()

            # Stage 2: Reasoning Tree Completion
            reasoning_tree = self.domain.complete_reasoning_tree(
                story_template, self.llm
            )

            # Stage 3: Story Generation
            narrative = self.domain.generate_story(reasoning_tree, self.llm)

            # Final assembly
            case = self.domain.assemble_case(story_template, reasoning_tree, narrative)
            cases.append(case)
        return cases