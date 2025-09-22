from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Optional


class ReasoningNode(BaseModel):
    """A single node in the reasoning tree."""
    statement: str
    node_type: Literal["deduced_fact", "story_fact", "rule_fact"]
    children: List['ReasoningNode'] = Field(default_factory=list)


class ReasoningTree(BaseModel):
    """The complete reasoning tree for a case."""
    root: ReasoningNode


class StoryTemplate(BaseModel):
    """A template for generating a story."""
    gold_facts: List[str]
    diversity_facts: List[str]
    question: str
    answer: str
    rule_signals: Optional[List[str]] = None
    meta_data: Dict[str, Any] = Field(default_factory=dict)


class GeneratedCase(BaseModel):
    """The final, complete output for a single generated case."""
    domain: str

    # The first-person narrative
    narrative: str

    # Explicit list of facts
    underlying_facts: List[str]

    # The tax rule(s)/ heuristic(s) that should be triggered
    rule_signals: List[str]

    # A human-readable reasoning trace
    reasoning_trace: str

    # The question, answer, and gold facts for evaluation
    question: str
    answer: str
    options: List[str] = Field(default_factory=list)

    # The raw tree
    reasoning_tree: ReasoningTree


class Person(BaseModel):
    income: float                          # taxable income
    pays_church_tax: bool = False          # church tax membership
    wage_replacement: float = 0.0          # e.g., Elterngeld, Krankengeld, ALG1, Kurzarbeit (Progressionsvorbehalt)
    medical_costs: float = 0.0             # only the part above a threshold based on income is deductible
    fully_liable_for_tax: bool = True      # false for people with no residence in Germany


class CoupleTaxInput(BaseModel):
    a: Person
    b: Person
    church_tax_rate: float = 0.09          # 9% typical; set 0.08 for 8%-states
    married: bool = True
    children: int = 0                      # number of children
    live_together: bool = True             # true if couple lives together at least for one day in the year


class WorkflowOutput(BaseModel):
    predicted_answer: str
    reasoning: str
    token_usage: Dict[str, Any] = Field(default_factory=dict)