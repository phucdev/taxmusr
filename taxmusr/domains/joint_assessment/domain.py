import random

from taxmusr.core.schemas import ReasoningTree, GeneratedCase, StoryTemplate, ReasoningNode
from taxmusr.domains.base import TaxDomain
from taxmusr.domains.joint_assessment import prompts
from taxmusr.domains.joint_assessment.rules import TAX_RULES
from taxmusr.domains import formatter
from taxmusr.domains.joint_assessment.logic import sample_couple_input, compare_assessments


JOBS = ["Software Engineer", "Teacher", "Doctor", "Graphic Designer", "Chef", "Mechanic", "Nurse", "Photographer",
        "Electrician", "Plumber", "Carpenter", "Secretary", "Writer", "Accountant", "Salesperson"]


class JointAssessmentDomain(TaxDomain):
    """Domain class for joint assessment tax cases."""
    def __init__(self, max_depth=2):
        self.name = "joint_assessment"
        self.description = "Joint assessment tax cases involving married couples."
        self.max_depth = max_depth  # maximum depth for reasoning tree expansion

    def construct_template(self) -> StoryTemplate:
        """Stage 1: Create gold facts and diversity facts."""
        answer = "individual" if random.random() < 0.3 else "joint"
        gold_facts = []
        if answer == "joint":
            gold_facts.append("The couple is eligible for joint assessment and should opt for it to minimize their tax burden.")
        else:
            gold_facts.append(random.choice([
                "The couple is not eligible for joint assessment and must file individual assessments.",
                "The couple is eligible for joint assessment, but should opt for individual assessment to minimize their tax burden."])
            )
        # Diversity facts add context and make the story more interesting but are not directly relevant to the tax decision
        number_of_children = random.choices([0, 1, 2, 3], weights=[0.20, 0.24, 0.38, 0.18])[0]
        diversity_facts = []
        if number_of_children > 0:
            diversity_facts.append(f"The couple has {number_of_children} child{'ren' if number_of_children > 1 else ''}.")
        else:
            diversity_facts.append("The couple has no children.")
        question = "Should the couple opt for joint assessment or individual assessment to minimize their tax burden?"

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
        forbidden_words = [
            "joint assessment", "individual assessment"
        ]
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
                            if any(bad_word in story_fact.lower() for bad_word in forbidden_words):
                                node_type = "deduced_fact"
                            else:
                                node_type = "story_fact"
                            story_node = ReasoningNode(statement=story_fact, node_type=node_type)
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
            options=["joint", "individual"],
            rule_signals=rule_signals,
            reasoning_trace=reasoning_trace,
            underlying_facts=underlying_facts,
            narrative=narrative,
            reasoning_tree=reasoning_tree,
        )


class GroundedJointAssessmentDomain(JointAssessmentDomain):
    """A variant of the JointAssessmentDomain that samples facts and computes answer using realistic computation."""
    def __init__(self, max_depth=1):
        super().__init__(max_depth=max_depth)
        self.name = "grounded_joint_assessment"

    def construct_template(self) -> StoryTemplate:
        """Stage 1: Create gold facts and diversity facts."""
        couple_facts = sample_couple_input()
        assessment_result = compare_assessments(couple_facts)
        eligible_for_joint = couple_facts.married and couple_facts.a.fully_liable_for_tax and couple_facts.b.fully_liable_for_tax and couple_facts.live_together
        if not eligible_for_joint:
            # If not eligible for joint assessment, the answer is always individual
            answer = "individual"
        else:
            answer = assessment_result["recommendation"]

        rule_signals = [
            "Couples are eligible for joint assessment if married, both are fully liable for tax in Germany and have lived together for at least one day of the assessment year.",
        ]

        # Build the gold facts from the couple_facts: gold facts are relevant for the tax decision
        gold_facts = [
            f"Person A and Person B are {'married' if couple_facts.married else 'not married'}.",
            f"Person A has a taxable income of {couple_facts.income} euros." if couple_facts.a.income > 0 else "Person A has no taxable income.",
            f"Person B has a taxable income of {couple_facts.b.income} euros." if couple_facts.b.income > 0 else "Person B has no taxable income."
        ]
        if not couple_facts.a.fully_liable_for_tax:
            gold_facts.append("Person A is not fully liable for tax in Germany.")
        if not couple_facts.b.fully_liable_for_tax:
            gold_facts.append("Person B is not fully liable for tax in Germany.")
        if not couple_facts.live_together:
            gold_facts.append("The couple did not live together at any point during the year.")
        else:
            gold_facts.append("The couple lived together at least for one day during the year.")
        if couple_facts.a.wage_replacement > 0:
            gold_facts.append(
                f"Person A received {couple_facts.a.wage_replacement} euros in wage replacement benefits.")
        if couple_facts.b.wage_replacement > 0:
            gold_facts.append(
                f"Person B received {couple_facts.b.wage_replacement} euros in wage replacement benefits.")
        if couple_facts.a.medical_costs > 0:
            gold_facts.append(f"Person A paid {couple_facts.a.medical_costs} euros in medical costs out of pocket.")
        if couple_facts.b.medical_costs > 0:
            gold_facts.append(f"Person B paid {couple_facts.b.medical_costs} euros in medical costs out of pocket.")
        if couple_facts.a.pays_church_tax or couple_facts.b.pays_church_tax:
            gold_facts.append(f"The church tax rate is {couple_facts.church_tax_rate * 100} percent.")
            if couple_facts.a.pays_church_tax and couple_facts.b.pays_church_tax:
                gold_facts.append("Both Person A and Person B are members of a church that requires church tax.")
            elif couple_facts.a.pays_church_tax:
                gold_facts.append("Only Person A is a member of a church that requires church tax.")
            elif couple_facts.b.pays_church_tax:
                gold_facts.append("Only Person B is a member of a church that requires church tax.")
        # Diversity facts add context and make the story more interesting but are not directly relevant to the tax decision
        diversity_facts = [
            f"Person A is working as a {random.choice(JOBS)}" if couple_facts.a.income > 0 else "Person A is currently unemployed.",
            f"Person B is working as a {random.choice(JOBS)}" if couple_facts.b.income > 0 else "Person B is currently unemployed.",
        ]
        if couple_facts.children > 0:
            diversity_facts.append(
                f"The couple has {couple_facts.children} child{'ren' if couple_facts.children > 1 else ''}.")
        else:
            diversity_facts.append("The couple has no children.")
        question = "Should the couple opt for joint assessment or individual assessment to minimize their tax burden?"
        return StoryTemplate(
            gold_facts=gold_facts,
            diversity_facts=diversity_facts,
            question=question,
            answer=answer,
            rule_signals=rule_signals,
            meta_data={"couple_facts": couple_facts}
        )

    def complete_reasoning_tree(self, story_template, llm) -> ReasoningTree:
        """Stage 2: Expand facts into a full reasoning tree.
        In this we have a bunch of gold facts that we expand upon.
        """
        if story_template.answer == "individual":
            conclusion = "The couple should file individual assessments."
        else:
            conclusion = "The couple should opt for joint assessment to minimize their tax burden."
        root = ReasoningNode(statement=conclusion, node_type="deduced_fact",
                             children=[
                                 ReasoningNode(statement=fact, node_type="story_fact")
                                 for fact in story_template.gold_facts+story_template.diversity_facts
                             ])
        forbidden_words = [
            "joint assessment", "individual assessment"
        ]
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
                            if any(bad_word in story_fact.lower() for bad_word in forbidden_words):
                                node_type = "deduced_fact"
                            else:
                                node_type = "story_fact"
                            story_node = ReasoningNode(statement=story_fact, node_type=node_type)
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
