from typing import List

from taxmusr.core.schemas import ReasoningTree


def format_reasoning_trace(tree: ReasoningTree) -> str:
    """Converts the reasoning tree into a human-readable string.
    """
    trace_lines = []
    def walk(node, depth=0):
        indent = "  " * depth
        trace_lines.append(f"{indent}- {node.statement} ({node.node_type})")
        for child in node.children:
            walk(child, depth + 1)
    walk(tree.root)
    return "\n".join(trace_lines)

def extract_underlying_facts(tree: ReasoningTree) -> List[str]:
    """Extracts all story_fact nodes from the tree."""
    story_facts = []
    def walk(node):
        if node.node_type == "story_fact":
            story_facts.append(node.statement)
        for child in node.children:
            walk(child)
    walk(tree.root)
    return story_facts

def extract_rule_signals(tree: ReasoningTree) -> List[str]:
    """Extracts all rule_fact nodes from the tree."""
    rules = []
    def walk(node):
        if node.node_type == "rule_fact":
            rules.append(node.statement)
        for child in node.children:
            walk(child)
    walk(tree.root)
    return rules
