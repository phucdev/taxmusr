import random

from taxmusr.core.schemas import ReasoningTree, GeneratedCase, StoryTemplate, ReasoningNode
from taxmusr.domains.base import TaxDomain
from taxmusr.domains.home_office_deduction import prompts
from taxmusr.domains.home_office_deduction.rules import TAX_RULES
from taxmusr.domains import formatter


JOBS = ["Software Engineer", "Teacher", "Graphic Designer", "Photographer",
        "Interpreter", "Professor", "Secretary", "Writer", "Accountant", "Salesperson"]


class HomeOfficeDeductionDomain(TaxDomain):
    """Domain class for home office deduction tax cases."""
    def __init__(self, max_depth=1):
        self.name = "home_office_deduction"
        self.description = "Home office deduction tax cases."
        self.max_depth = max_depth  # maximum depth for reasoning tree expansion

    def construct_template(self) -> StoryTemplate:
        """Stage 1: Create gold facts and diversity facts."""
        answer = "flatrate" if random.random() > 0.3 else "pro-rata"
        gold_facts = []
        if answer == "pro-rata":
            gold_facts.append("The home office is eligible and the pro-rata costs can be deducted.")
        else:
            gold_facts.append("The home office is not eligible, but the taxpayer can use the home office flatrate.")
        # Diversity facts add context and make the story more interesting but are not directly relevant to the tax decision
        rooms_in_apartment = random.choice([2, 3])
        diversity_facts = []
        diversity_facts.append(f"The narrator works as a {random.choice(JOBS)}.")
        diversity_facts.append(f"The narrator lives in an apartment with {rooms_in_apartment} rooms.")
        question = "Can the narrator deduct the pro-rata costs for the home office or should they claim the flatrate?"

        return StoryTemplate(
            gold_facts=gold_facts,
            diversity_facts=diversity_facts,
            question=question,
            answer=answer
        )

    def complete_reasoning_tree(self, story_template, llm) -> ReasoningTree:
        """Stage 2: Expand facts into a full reasoning tree.
        In MuSR we usually start with a set of gold facts and diversity facts.
        But for simplicity we only start with one gold fact, which is also the expected conclusion.
        """
        root = ReasoningNode(statement=story_template.gold_facts[0], node_type="deduced_fact",
                             children=[
                                 ReasoningNode(statement=diversity_fact, node_type="story_fact")
                                 for diversity_fact in story_template.diversity_facts
                             ])
        generation_chain = prompts.FACT_EXPANSION_PROMPT | llm.model
        # recursive expansion:
        def expand_node(node: ReasoningNode, depth: int = 0):
            if depth <= self.max_depth:
                # Use the LLM to expand the current node
                # TODO: We could use structured outputs here to make parsing more robust
                # collect all story facts from the tree so far
                story_facts = formatter.extract_underlying_facts(ReasoningTree(root=root))
                response = generation_chain.invoke(
                    {
                        "fact": node.statement,
                        "story_facts": "\n".join(f"- {fact}" for fact in story_facts),
                        "rules": "You can use the following rule set:\n" + "\n".join(f"- {rule}" for rule in TAX_RULES)
                    },
                    config={"callbacks": [llm.callback_handler] if llm.callback_handler else []}
                )
                lines = [line.strip() for line in response.content.split("\n") if line.strip()]
                for line in lines:
                    # TODO: Add validators and checks to avoid duplicates and inconsistencies
                    if line.startswith("Story Fact:"):
                        story_fact = line[len("Story Fact:"):].strip().replace('"', '')
                        if story_fact:
                            story_node = ReasoningNode(statement=story_fact, node_type="story_fact")
                            node.children.append(story_node)
                            expand_node(story_node, depth + 1)
                    elif line.startswith("Rule:"):
                        rule_fact = line[len("Rule:"):].strip().replace('"', '')
                        if rule_fact:
                            rule_node = ReasoningNode(statement=rule_fact, node_type="rule_fact")
                            node.children.append(rule_node)

        expand_node(root)
        # TODO: Rerun validation, deduplication and consistency checks here
        return ReasoningTree(root=root)

    def generate_story(self, reasoning_tree: ReasoningTree, llm) -> str:
        """Stage 3: Convert reasoning tree leaves into a narrative."""
        # Extract all the story facts that must be included in the narrative
        story_facts = formatter.extract_underlying_facts(reasoning_tree)
        story_facts = list(set(story_facts))  # deduplicate

        # Use a LangChain prompt template to generate the narrative
        generation_chain = prompts.NARRATIVE_PROMPT | llm.model

        response = generation_chain.invoke(
            {
                "facts_list": "- " + "\n- ".join(story_facts)
            },
            config={"callbacks": [llm.callback_handler] if llm.callback_handler else []}
        )
        if len(response.content) > 0:
            return response.content
        else:
            raise ValueError("LLM returned empty narrative")

    def assemble_case(self, story_template, reasoning_tree, narrative) -> GeneratedCase:
        """Puts all the generated pieces together."""
        reasoning_trace = formatter.format_reasoning_trace(reasoning_tree)
        underlying_facts = formatter.extract_underlying_facts(reasoning_tree)
        rule_signals = formatter.extract_rule_signals(reasoning_tree)
        return GeneratedCase(
            domain=self.name,
            question=story_template.question,
            answer=story_template.answer,
            options=["pro-rata", "flatrate"],
            rule_signals=rule_signals,
            reasoning_trace=reasoning_trace,
            underlying_facts=underlying_facts,
            narrative=narrative,
            reasoning_tree=reasoning_tree,
        )