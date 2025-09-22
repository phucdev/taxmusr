from langchain_core.prompts import ChatPromptTemplate

# Prompt to expand a fact into a story fact and a rule
FACT_EXPANSION_PROMPT = ChatPromptTemplate.from_template("""
The core fact is '{fact}'. Think of story facts that would imply this fact and 
give me a tax rule or commonsense rule that explains the entailment.
{rules}
Keep in mind that these are the story facts so far:\n{story_facts}
Make sure that your story facts are consistent with each other and with the core fact.
Only output one set of story facts and rule. Keep the same format as in the examples below.
New story facts should add value to the story and not be redundant with existing story facts.
If you cannot think of any more story facts that add value to the story, just return an empty list.

Here is an example:
Fact: "The home office is the center of professional activity."
Story Fact: "The home office is a separate room."
Story Fact: "The narrator does not have another office."
Rule: "A home office is the center of professional activity if the majority of professional activities are carried out in the home office."

Now you try:
Fact: "{fact}"
"""
)

# Prompt to write a narrative chapter from a set of facts
NARRATIVE_PROMPT = ChatPromptTemplate.from_template("""
Write a first-person mini story about a person's working situation at home in Germany given a list of facts.
Keep the story coherent and realistic and avoid tax jargon. Only output the story without any additional commentary.
The story must clearly imply the following facts without stating them like a list:\n\n{facts_list}


Critical constraints:
- Never mention terms like center of professional activity.
- Never explain how taxes work or are calculated.
""")

# Prompt to solve joint assessment question given a narrative
EVALUATION_PROMPT = ChatPromptTemplate.from_template("""
You are a tax expert in Germany. Given a story, answer the question at the end.

{examples}

STORY:
{narrative}

QUESTION:
{question}

Pick one of the following choices: {options}.
You must pick one option. 
{cot}
Finally, the last thing you generate should be "ANSWER: (your answer here)".
""")